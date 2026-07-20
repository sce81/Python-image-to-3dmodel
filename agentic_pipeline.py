"""Gated controller for the local raster-to-PBR asset pipeline.

The stage workers remain ``generate_props.py`` and ``blender_postprocess.py``.
This controller owns the durable state machine, artifact ledger, independent
stage execution, human approvals, rejection cleanup, and final promotion.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent
PYTHON = ROOT / ".venv" / "Scripts" / "python.exe"
if not PYTHON.exists():
    PYTHON = ROOT.parent / ".venv" / "Scripts" / "python.exe"
BLENDER = Path(r"C:\Program Files\Blender Foundation\Blender 5.1\blender.exe")
WORK_ROOT = ROOT / "WorkingFolder" / "agent_runs"
RAW_DIR = ROOT / "raw_meshes"
OUTPUT_DIR = ROOT / "Outputs"
VALIDATION_ROOT = ROOT / "WorkingFolder" / "validation_images"
REPORT_ROOT = ROOT / "WorkingFolder" / "validation_reports"
VALIDATOR = ROOT / "WorkingFolder" / "validate_asset_contract.py"
RENDERER = ROOT / "WorkingFolder" / "render_candidate_view.py"

STAGES = ("raster", "geometry", "geometry_review", "texture", "texture_review", "export")
APPROVAL_STAGES = {"raster", "geometry_review", "texture_review"}


class StageBlocked(RuntimeError):
    """Raised when the requested stage has not met its entry conditions."""


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def absolute(path: str | Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else (ROOT / candidate).resolve()


def state_path(asset_id: str) -> Path:
    return WORK_ROOT / asset_id / "state.json"


def load_state(asset_id: str) -> dict[str, Any]:
    path = state_path(asset_id)
    if not path.exists():
        raise FileNotFoundError(f"No pipeline state for '{asset_id}'. Run init first.")
    return json.loads(path.read_text(encoding="utf-8-sig"))


def save_state(state: dict[str, Any]) -> None:
    path = state_path(state["asset_id"])
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(state, indent=2), encoding="utf-8")
    temporary.replace(path)
    write_model_record(state)


def event(state: dict[str, Any], action: str, **details: Any) -> None:
    state.setdefault("ledger", []).append({"at": now(), "action": action, **details})


MODEL_RECORD_MARKER = "<!-- nexus-learning-json"


def model_record_path(state: dict[str, Any]) -> Path:
    raster = Path(state["inputs"]["approved_raster"]["path"])
    candidates = raster.parent if raster.parent.name == "raster_candidates" else raster.parent / "raster_candidates"
    candidates.mkdir(parents=True, exist_ok=True)
    safe_asset_id = "".join(char if char.isalnum() or char in "_-" else "_" for char in state["asset_id"])
    return candidates / f"{safe_asset_id}.md"


def read_model_learning(state: dict[str, Any]) -> dict[str, Any]:
    path = model_record_path(state)
    if not path.exists():
        return {"version": 1, "rejected_geometry_signatures": {}, "feedback": []}
    document = path.read_text(encoding="utf-8")
    if MODEL_RECORD_MARKER not in document:
        return {"version": 1, "rejected_geometry_signatures": {}, "feedback": []}
    try:
        payload = document.split(MODEL_RECORD_MARKER, 1)[1].split("-->", 1)[0].strip()
        loaded = json.loads(payload)
        return loaded if isinstance(loaded, dict) else {"version": 1, "rejected_geometry_signatures": {}, "feedback": []}
    except (json.JSONDecodeError, IndexError):
        return {"version": 1, "rejected_geometry_signatures": {}, "feedback": []}


def execution_score(state: dict[str, Any]) -> int:
    score = 0
    for item in state.get("ledger", []):
        action = item.get("action")
        if action == "approved":
            score += {"raster": 5, "geometry_review": 30, "texture_review": 60}.get(item.get("stage"), 0)
        elif action == "rejected":
            score -= 100
        elif action == "user_feedback":
            score += 50 if item.get("verdict") == "good" else -100
        elif action == "promoted":
            score += 20
    return score


def write_model_record(state: dict[str, Any]) -> None:
    """Persist the feedback ledger beside the approved conditioning raster."""
    path = model_record_path(state)
    prior = read_model_learning(state)
    rejected = dict(prior.get("rejected_geometry_signatures", {}))
    rejected.update(state.get("rejected_geometry_signatures", {}))
    feedback = list(prior.get("feedback", []))
    for item in state.get("ledger", []):
        if item.get("action") == "user_feedback" and item not in feedback:
            feedback.append(item)
    geometry = stage(state, "geometry")
    geometry_review = stage(state, "geometry_review")
    settings = state.get("settings", {}).get("geometry", {})
    score = execution_score(state)
    outcome = geometry_review.get("status", "pending")
    rejection = geometry_review.get("rejection_reason") or geometry.get("rejection_reason") or ""
    lines = [
        f"# {state['asset_id']} Model Record",
        "",
        "## Inputs",
        f"- Approved raster: `{state['inputs']['approved_raster']['path']}`",
        f"- Raster SHA-256: `{state['inputs']['approved_raster']['sha256']}`",
        f"- Asset contract: `{state['contract']['path']}`",
        "",
        "## Current Execution",
        f"- Mesh ID: `{geometry.get('mesh_id', 'not generated')}`",
        f"- Geometry status: `{geometry.get('status', 'pending')}`",
        f"- Review status: `{outcome}`",
        f"- Score: **{score}** (approval improves it; rejection or bad feedback reduces it)",
        f"- Settings: `{json.dumps(settings, sort_keys=True)}`",
    ]
    if rejection:
        lines.extend([f"- Rejection reason: {rejection}"])
    lines.extend(["", "## Feedback History"])
    if feedback:
        for item in feedback:
            lines.append(f"- {item.get('at', '')}: **{item.get('verdict', 'unknown')}** - {item.get('note', '')}")
    else:
        lines.append("- No explicit user feedback recorded yet.")
    lines.extend(["", "## Rejected Geometry Signatures"])
    if rejected:
        for signature, detail in rejected.items():
            lines.append(f"- `{signature}`: {detail.get('reason', 'rejected')}")
    else:
        lines.append("- None.")
    learning = {
        "version": 1,
        "asset_id": state["asset_id"],
        "overall_score": score,
        "rejected_geometry_signatures": rejected,
        "feedback": feedback,
        "last_updated": now(),
    }
    lines.extend(["", MODEL_RECORD_MARKER, json.dumps(learning, sort_keys=True), "-->", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def geometry_signature(state: dict[str, Any], settings: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    """Fingerprint the actual geometry-conditioning inputs, including false defaults.

    Preprocessing knobs (rembg model, alpha threshold, clay conditioning) are part
    of the fingerprint: two runs that mask or condition differently are different
    experiments and must not collide in the rejection ledger.
    """
    normalized = {
        "raster_sha256": state["inputs"]["approved_raster"]["sha256"],
        "geometry_source_sha256s": sorted(
            item["sha256"] for item in state["inputs"].get("geometry_sources", [])
        ),
        "quality": settings["quality"],
        "asset_target": settings["asset_target"],
        "seed": settings["seed"],
        "seed_candidates": int(settings.get("seed_candidates", 1)),
        "ss_steps": settings["ss_steps"],
        "slat_steps": settings["slat_steps"],
        "ss_guidance": settings["ss_guidance"],
        "slat_guidance": settings["slat_guidance"],
        "target_tris": settings["target_tris"],
        "multi_image_mode": settings.get("multi_image_mode", "multidiffusion"),
        "geometry_clay": bool(settings.get("geometry_clay", True)),
        "rembg_model": settings.get("rembg_model", "isnet-general-use"),
        "alpha_threshold": int(settings.get("alpha_threshold", 48)),
        "suppress_window_reflections": bool(settings.get("suppress_window_reflections", False)),
        "flatten_window_geometry": bool(settings.get("flatten_window_geometry", False)),
    }
    encoded = json.dumps(normalized, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest(), normalized


def require_unrejected_geometry_signature(state: dict[str, Any], settings: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    signature, normalized = geometry_signature(state, settings)
    rejected = dict(read_model_learning(state).get("rejected_geometry_signatures", {}))
    rejected.update(state.get("rejected_geometry_signatures", {}))
    if signature in rejected:
        reason = rejected[signature]["reason"]
        raise StageBlocked(
            "Geometry signature was previously rejected for this approved raster: "
            f"{reason}. Change the reconstruction method or generation inputs before retrying."
        )
    return signature, normalized

def stage(state: dict[str, Any], name: str) -> dict[str, Any]:
    return state["stages"][name]


def require_status(state: dict[str, Any], name: str, allowed: set[str]) -> None:
    current = stage(state, name)["status"]
    if current not in allowed:
        raise StageBlocked(f"Stage '{name}' is '{current}', expected one of {sorted(allowed)}.")


def run_worker(command: list[Path | str], env: dict[str, str], log_path: Path) -> None:
    merged = os.environ.copy()
    merged.update(env)
    result = subprocess.run([str(item) for item in command], cwd=ROOT, env=merged, capture_output=True, text=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(
        f"command={command}\nreturncode={result.returncode}\n\n[stdout]\n{result.stdout}\n\n[stderr]\n{result.stderr}",
        encoding="utf-8",
    )
    if result.returncode:
        raise RuntimeError(f"Worker failed. See {log_path}")


def init(args: argparse.Namespace) -> None:
    contract_path = absolute(args.contract)
    raster_path = absolute(args.raster) if args.raster else None
    if not contract_path.exists():
        raise FileNotFoundError(contract_path)
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    if raster_path is None:
        raster_path = absolute(contract["approved_raster"])
    if not raster_path.exists():
        raise FileNotFoundError(raster_path)
    if state_path(args.asset_id).exists():
        raise FileExistsError(f"State already exists for '{args.asset_id}'.")

    # Extra geometry-conditioning views (contract "geometry_sources" plus CLI
    # --geometry-source). When present, the geometry stage runs multi-image
    # conditioning: approved raster + these views constrain the unseen sides.
    geometry_sources = [
        absolute(item) for item in (contract.get("geometry_sources") or [])
    ] + [absolute(item) for item in args.geometry_source]
    for source in geometry_sources:
        if not source.exists():
            raise FileNotFoundError(source)

    state = {
        "version": 1,
        "asset_id": args.asset_id,
        "created_at": now(),
        "contract": {"path": str(contract_path), "sha256": file_hash(contract_path), "asset_id": contract.get("asset_id")},
        "inputs": {
            "approved_raster": {"path": str(raster_path), "sha256": file_hash(raster_path)},
            "geometry_sources": [
                {"path": str(source), "sha256": file_hash(source)} for source in geometry_sources
            ],
            "texture_sources": [str(absolute(item)) for item in args.texture_source],
        },
        "settings": {
            "geometry": {
                "quality": "very_high",
                "asset_target": "prop",
                "seed": 1234,
                # Best-of-N: sample N seeds, keep the lowest integrity penalty.
                "seed_candidates": 4,
                "ss_steps": 30,
                "slat_steps": 30,
                # TRELLIS-tuned defaults: high ss guidance = structure adheres to
                # the image (anti-cavity); low slat guidance = clean surfaces.
                "ss_guidance": 7.5,
                "slat_guidance": 3.5,
                "target_tris": 100000,
                "multi_image_mode": "multidiffusion",
                # Lift dark paint/glazing to mid-gray for geometry conditioning.
                "geometry_clay": True,
                "rembg_model": "isnet-general-use",
                "alpha_threshold": 48,
                "suppress_window_reflections": False,
                "flatten_window_geometry": False,
            }
        },
        "stages": {name: {"status": "pending", "artifacts": []} for name in STAGES},
        "ledger": [],
    }
    stage(state, "raster")["status"] = "approved" if args.raster_approved else "awaiting_review"
    event(state, "initialized", raster_approved=args.raster_approved)
    save_state(state)
    print(state_path(args.asset_id))


def run_geometry(state: dict[str, Any]) -> None:
    require_status(state, "raster", {"approved"})
    require_status(state, "geometry", {"pending", "rejected"})
    settings = state["settings"]["geometry"]
    signature, normalized_settings = require_unrejected_geometry_signature(state, settings)
    attempt = sum(1 for item in state["ledger"] if item["action"] == "geometry_started") + 1
    mesh_id = f"{state['asset_id']}__geometry_{attempt:03d}"
    raw = RAW_DIR / f"{mesh_id}.glb"
    log = state_path(state["asset_id"]).parent / "logs" / f"{mesh_id}_generate.log"
    env = {
        "NEXUS_ASSET_NAME": mesh_id,
        "NEXUS_QUALITY": settings["quality"],
        "NEXUS_ASSET_TARGET": settings["asset_target"],
        "NEXUS_SEED": str(settings["seed"]),
        "NEXUS_SEED_CANDIDATES": str(settings.get("seed_candidates", 1)),
        "NEXUS_SS_STEPS": str(settings["ss_steps"]),
        "NEXUS_SLAT_STEPS": str(settings["slat_steps"]),
        "NEXUS_SS_GUIDANCE": str(settings["ss_guidance"]),
        "NEXUS_SLAT_GUIDANCE": str(settings["slat_guidance"]),
        "NEXUS_GEOMETRY_ONLY": "1",
        "NEXUS_GEOMETRY_CLAY": "1" if settings.get("geometry_clay", True) else "0",
        "NEXUS_MULTI_MODE": settings.get("multi_image_mode", "multidiffusion"),
        "NEXUS_REMBG_MODEL": settings.get("rembg_model", "isnet-general-use"),
        "NEXUS_ALPHA_THRESHOLD": str(settings.get("alpha_threshold", 48)),
        "NEXUS_REMOVE_REFLECTIONS": "0",
        "NEXUS_SUPPRESS_WINDOW_REFLECTIONS": "1" if settings.get("suppress_window_reflections", False) else "0",
        "NEXUS_FLATTEN_WINDOW_GEOMETRY": "1" if settings.get("flatten_window_geometry", False) else "0",
    }
    geometry_sources = state["inputs"].get("geometry_sources", [])
    if geometry_sources:
        # Multi-image conditioning: stage the approved raster plus every extra
        # view into one folder and let generate_props run run_multi_image.
        staging = state_path(state["asset_id"]).parent / "geometry_sources" / mesh_id
        if staging.exists():
            shutil.rmtree(staging)
        staging.mkdir(parents=True, exist_ok=True)
        raster = Path(state["inputs"]["approved_raster"]["path"])
        shutil.copy2(raster, staging / f"00_{raster.name}")
        for index, item in enumerate(geometry_sources, start=1):
            source = Path(item["path"])
            shutil.copy2(source, staging / f"{index:02d}_{source.name}")
        env["NEXUS_SOURCE_DIR"] = str(staging)
    else:
        env["NEXUS_SINGLE"] = state["inputs"]["approved_raster"]["path"]
    stage(state, "geometry")["status"] = "running"
    stage(state, "geometry")["signature"] = signature
    event(state, "geometry_started", mesh_id=mesh_id, settings=normalized_settings, signature=signature)
    save_state(state)
    run_worker([PYTHON, ROOT / "generate_props.py"], env, log)
    if not raw.exists():
        raise RuntimeError(f"Geometry worker completed without {raw}")
    stage(state, "geometry").pop("rejection_reason", None)
    stage(state, "geometry").update({"status": "complete", "artifacts": [str(raw)], "mesh_id": mesh_id, "log": str(log)})
    event(state, "geometry_completed", raw=str(raw), sha256=file_hash(raw))
    save_state(state)


def run_geometry_review(state: dict[str, Any]) -> None:
    require_status(state, "geometry", {"complete"})
    require_status(state, "geometry_review", {"pending", "rejected"})
    mesh_id = stage(state, "geometry")["mesh_id"]
    candidate = OUTPUT_DIR / f"{mesh_id}_ue.glb"
    blend = OUTPUT_DIR / f"{mesh_id}_ue.blend"
    render = VALIDATION_ROOT / f"{mesh_id}_three_quarter.png"
    report = REPORT_ROOT / f"{mesh_id}.json"
    log = state_path(state["asset_id"]).parent / "logs" / f"{mesh_id}_review.log"
    settings = state["settings"]["geometry"]
    stage(state, "geometry_review")["status"] = "running"
    save_state(state)
    run_worker(
        [BLENDER, "--background", "--python", ROOT / "blender_postprocess.py"],
        {
            "NEXUS_MESH": mesh_id,
            "NEXUS_QUALITY": settings["quality"],
            "NEXUS_ASSET_TARGET": settings["asset_target"],
            "NEXUS_TRIS": str(settings["target_tris"]),
            "NEXUS_SYMMETRY": "none",
            "NEXUS_LENGTH_AXIS": "x",
            "NEXUS_SMOOTH_NORMALS": "1",
            "NEXUS_REMOVE_PRESENTATION_BASE": "0",
        },
        log,
    )
    if not candidate.exists() or not blend.exists():
        raise RuntimeError("Blender review worker did not produce candidate GLB and Blend files.")
    run_worker(
        [PYTHON, VALIDATOR, candidate, state["contract"]["path"], "--report", report, "--geometry-only"],
        {},
        log.with_name(f"{mesh_id}_geometry_audit.log"),
    )
    run_worker(
        [BLENDER, "--background", "--python", RENDERER],
        {"NEXUS_RENDER_ASSET": str(candidate), "NEXUS_RENDER_OUT": str(render)},
        log.with_name(f"{mesh_id}_render.log"),
    )
    stage(state, "geometry_review").update(
        {"status": "awaiting_review", "artifacts": [str(candidate), str(blend), str(render), str(report)], "log": str(log)}
    )
    event(state, "geometry_review_ready", candidate=str(candidate), render=str(render), report=str(report))
    save_state(state)


def register_texture(state: dict[str, Any], glb: Path, blend: Path | None, method: str) -> None:
    require_status(state, "geometry_review", {"approved"})
    if not glb.exists():
        raise FileNotFoundError(glb)
    artifacts = [str(glb)] + ([str(blend)] if blend else [])
    stage(state, "texture").update({"status": "complete", "artifacts": artifacts, "method": method})
    stage(state, "texture_review")["status"] = "awaiting_review"
    event(state, "texture_registered", glb=str(glb), blend=str(blend) if blend else None, method=method)
    save_state(state)


def approve(state: dict[str, Any], name: str, reviewer: str) -> None:
    if name not in APPROVAL_STAGES:
        raise ValueError(f"Approval is only valid for {sorted(APPROVAL_STAGES)}")
    require_status(state, name, {"awaiting_review"})
    stage(state, name).update({"status": "approved", "reviewer": reviewer, "approved_at": now()})
    event(state, "approved", stage=name, reviewer=reviewer)
    save_state(state)


def reject(state: dict[str, Any], name: str, reason: str) -> None:
    if name not in STAGES:
        raise ValueError(f"Unknown stage '{name}'")
    targets = []
    if name == "geometry_review":
        targets.extend(stage(state, "geometry").get("artifacts", []))
    targets.extend(stage(state, name).get("artifacts", []))
    for target in {Path(item) for item in targets}:
        if target.exists() and target.is_file() and ROOT in target.resolve().parents:
            target.unlink()
    stage(state, name).update({"status": "rejected", "rejection_reason": reason})
    if name == "geometry_review":
        geometry_stage = stage(state, "geometry")
        signature = geometry_stage.get("signature")
        if not signature:
            signature, _ = geometry_signature(state, state["settings"]["geometry"])
        state.setdefault("rejected_geometry_signatures", {})[signature] = {
            "at": now(),
            "reason": reason,
            "mesh_id": geometry_stage.get("mesh_id"),
        }
        geometry_stage["status"] = "rejected"
    event(state, "rejected", stage=name, reason=reason, deleted=[str(path) for path in targets])
    save_state(state)


def promote(state: dict[str, Any]) -> None:
    require_status(state, "texture_review", {"approved"})
    source_glb = Path(stage(state, "texture")["artifacts"][0])
    source_blend = next((Path(path) for path in stage(state, "texture")["artifacts"] if Path(path).suffix.lower() == ".blend"), None)
    if not source_glb.exists():
        raise FileNotFoundError(source_glb)
    final_glb = OUTPUT_DIR / f"{state['asset_id']}_ue.glb"
    final_blend = OUTPUT_DIR / f"{state['asset_id']}_ue.blend"
    shutil.copy2(source_glb, final_glb)
    if source_blend and source_blend.exists():
        shutil.copy2(source_blend, final_blend)
    stage(state, "export").update({"status": "promoted", "artifacts": [str(final_glb)] + ([str(final_blend)] if final_blend.exists() else [])})
    event(state, "promoted", glb=str(final_glb), blend=str(final_blend) if final_blend.exists() else None)
    save_state(state)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    init_parser = commands.add_parser("init", help="Create durable state for one asset.")
    init_parser.add_argument("--asset-id", required=True)
    init_parser.add_argument("--contract", required=True)
    init_parser.add_argument("--raster", default="")
    init_parser.add_argument("--geometry-source", action="append", default=[],
                             help="Extra conditioning view for multi-image geometry; repeatable.")
    init_parser.add_argument("--texture-source", action="append", default=[])
    init_parser.add_argument("--raster-approved", action="store_true")

    for command in ("status", "run", "approve", "reject", "register-texture", "promote", "feedback"):
        subparser = commands.add_parser(command)
        subparser.add_argument("--asset-id", required=True)
        if command == "run":
            subparser.add_argument("--stage", choices=("geometry", "geometry-review"), required=True)
        elif command == "approve":
            subparser.add_argument("--stage", choices=sorted(APPROVAL_STAGES), required=True)
            subparser.add_argument("--reviewer", required=True)
        elif command == "reject":
            subparser.add_argument("--stage", choices=STAGES, required=True)
            subparser.add_argument("--reason", required=True)
        elif command == "feedback":
            subparser.add_argument("--verdict", choices=("good", "bad"), required=True)
            subparser.add_argument("--note", required=True)
        elif command == "register-texture":
            subparser.add_argument("--glb", required=True)
            subparser.add_argument("--blend", default="")
            subparser.add_argument("--method", required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.command == "init":
        init(args)
        return 0
    state = load_state(args.asset_id)
    if args.command == "status":
        print(json.dumps(state, indent=2))
    elif args.command == "run":
        run_geometry(state) if args.stage == "geometry" else run_geometry_review(state)
    elif args.command == "approve":
        approve(state, args.stage, args.reviewer)
    elif args.command == "reject":
        reject(state, args.stage, args.reason)
    elif args.command == "feedback":
        event(state, "user_feedback", verdict=args.verdict, note=args.note, mesh_id=stage(state, "geometry").get("mesh_id"))
        if args.verdict == "bad":
            signature = stage(state, "geometry").get("signature")
            if signature:
                state.setdefault("rejected_geometry_signatures", {})[signature] = {
                    "at": now(), "reason": f"User feedback: {args.note}", "mesh_id": stage(state, "geometry").get("mesh_id"),
                }
        save_state(state)
    elif args.command == "register-texture":
        register_texture(state, absolute(args.glb), absolute(args.blend) if args.blend else None, args.method)
    elif args.command == "promote":
        promote(state)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileNotFoundError, StageBlocked, RuntimeError, ValueError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise SystemExit(2)
