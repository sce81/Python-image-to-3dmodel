"""
MeshAI reference learning loop for the Nexus asset pipeline.

This script coordinates candidate generation, Blender postprocess, and objective
scoring against Meshai-EQE-Model.glb. It does not train TRELLIS weights; it builds
a repeatable empirical loop over preprocessing/postprocess parameters and records
which settings move the generated asset closer to the reference.

Typical use:
    .\.venv\Scripts\python.exe learning_loop.py --max-candidates 3 --generate

Evaluate existing outputs only:
    .\.venv\Scripts\python.exe learning_loop.py --evaluate-existing
"""

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parent
PYTHON = ROOT / ".venv" / "Scripts" / "python.exe"
BLENDER = Path(r"C:\Program Files\Blender Foundation\Blender 5.1\blender.exe")
REFERENCE = ROOT / "Meshai-EQE-Model.glb"
SOURCE_DIR = ROOT / "Inputs" / "Cars" / "Mercedes" / "EQE"
RAW_DIR = ROOT / "raw_meshes"
OUTPUT_DIR = ROOT / "Outputs"
LOG_ROOT = ROOT / "Logs" / "LearningLoop"
EVALUATOR = ROOT / "evaluate_model_similarity.py"


CANDIDATES = [
    {
        "name": "MeshAI_A_prop_reference_calibrated",
        "quality": "very_high",
        "asset_target": "prop",
        "multi_mode": "multidiffusion",
        "seed": "42",
        "simplify": "0.12",
        "ss_steps": "50",
        "slat_steps": "50",
        "ss_guidance": "7.5",
        "slat_guidance": "7.5",
        "base_cut": "0.03",
        "tris": "140000",
    },
    {
        "name": "MeshAI_B_meshi_car_soft_cleanup",
        "quality": "very_high",
        "asset_target": "meshai_car",
        "multi_mode": "multidiffusion",
        "seed": "42",
        "simplify": "0.10",
        "ss_steps": "50",
        "slat_steps": "50",
        "ss_guidance": "6.5",
        "slat_guidance": "8.0",
        "body_bottom_keep": "0.78",
        "body_fill_start": "0.55",
        "body_dark_fill": "130",
        "base_cut": "0.035",
        "tris": "140000",
    },
    {
        "name": "MeshAI_C_meshi_car_more_structure",
        "quality": "very_high",
        "asset_target": "meshai_car",
        "multi_mode": "multidiffusion",
        "seed": "1234",
        "simplify": "0.08",
        "ss_steps": "60",
        "slat_steps": "60",
        "ss_guidance": "8.0",
        "slat_guidance": "8.5",
        "body_bottom_keep": "0.82",
        "body_fill_start": "0.60",
        "body_dark_fill": "115",
        "base_cut": "0.025",
        "tris": "160000",
    },
    {
        "name": "MeshAI_D_stochastic_preserve_features",
        "quality": "very_high",
        "asset_target": "prop",
        "multi_mode": "stochastic",
        "seed": "2026",
        "simplify": "0.10",
        "ss_steps": "50",
        "slat_steps": "50",
        "ss_guidance": "7.0",
        "slat_guidance": "8.0",
        "base_cut": "0.035",
        "tris": "140000",
    },
    {
        # TRELLIS-default guidance profile: ss 7.5 for structure adherence,
        # slat near the tuned default 3.0 (high slat CFG bred the artifact-heavy
        # candidates), plus best-of-4 seed selection.
        "name": "MeshAI_E_trellis_default_guidance",
        "quality": "very_high",
        "asset_target": "prop",
        "multi_mode": "multidiffusion",
        "seed": "42",
        "seed_candidates": "4",
        "simplify": "0.10",
        "ss_steps": "30",
        "slat_steps": "30",
        "ss_guidance": "7.5",
        "slat_guidance": "3.5",
        "base_cut": "0.035",
        "tris": "140000",
    },
]


def run(cmd, env=None, log_path=None, timeout=None):
    merged = os.environ.copy()
    if env:
        merged.update({k: str(v) for k, v in env.items() if v is not None})
    proc = subprocess.run(
        [str(part) for part in cmd],
        cwd=str(ROOT),
        env=merged,
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    if log_path:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text(
            f"command={cmd}\nreturncode={proc.returncode}\n\n[stdout]\n{proc.stdout}\n\n[stderr]\n{proc.stderr}",
            encoding="utf-8",
        )
    if proc.returncode != 0 or "Traceback (most recent call last)" in (proc.stderr or ""):
        raise RuntimeError(f"Command failed ({proc.returncode}): {cmd}\n{proc.stdout[-1200:]}\n{proc.stderr[-1200:]}")
    return proc


def generation_env(candidate):
    return {
        "NEXUS_SOURCE_DIR": str(SOURCE_DIR),
        "NEXUS_ASSET_NAME": candidate["name"],
        "NEXUS_QUALITY": candidate["quality"],
        "NEXUS_ASSET_TARGET": candidate["asset_target"],
        "NEXUS_MULTI_MODE": candidate["multi_mode"],
        "NEXUS_SEED": candidate.get("seed", "42"),
        "NEXUS_SEED_CANDIDATES": candidate.get("seed_candidates"),
        "NEXUS_SIMPLIFY": candidate.get("simplify"),
        "NEXUS_SS_STEPS": candidate.get("ss_steps"),
        "NEXUS_SLAT_STEPS": candidate.get("slat_steps"),
        "NEXUS_SS_GUIDANCE": candidate.get("ss_guidance"),
        "NEXUS_SLAT_GUIDANCE": candidate.get("slat_guidance"),
        "NEXUS_BODY_BOTTOM_KEEP": candidate.get("body_bottom_keep"),
        "NEXUS_BODY_FILL_START": candidate.get("body_fill_start"),
        "NEXUS_BODY_DARK_FILL": candidate.get("body_dark_fill"),
    }


def postprocess_env(candidate):
    return {
        "NEXUS_MESH": candidate["name"],
        "NEXUS_QUALITY": candidate["quality"],
        "NEXUS_ASSET_TARGET": candidate["asset_target"],
        "NEXUS_SYMMETRY": "none",
        "NEXUS_LENGTH_AXIS": "x",
        "NEXUS_SMOOTH_NORMALS": "1",
        "NEXUS_REFERENCE_MODEL": str(REFERENCE),
        "NEXUS_MATCH_REFERENCE_DIMS": "1",
        "NEXUS_BASE_CUT_HEIGHT": candidate.get("base_cut", "0.035"),
        "NEXUS_TRIS": candidate.get("tris", "140000"),
    }


def evaluate(candidate_name, run_dir):
    candidate_path = OUTPUT_DIR / f"{candidate_name}_ue.glb"
    if not candidate_path.exists():
        raise FileNotFoundError(candidate_path)
    out = run_dir / f"{candidate_name}_score.json"
    run(
        [
            BLENDER,
            "--background",
            "--python",
            EVALUATOR,
            "--",
            "--reference",
            REFERENCE,
            "--candidate",
            candidate_path,
            "--out",
            out,
        ],
        log_path=run_dir / f"{candidate_name}_evaluate.log",
        timeout=180,
    )
    return json.loads(out.read_text(encoding="utf-8"))


def process_candidate(candidate, run_dir, generate):
    name = candidate["name"]
    if generate:
        run(
            [PYTHON, ROOT / "generate_props.py"],
            env=generation_env(candidate),
            log_path=run_dir / f"{name}_generate.log",
        )
    raw = RAW_DIR / f"{name}.glb"
    if not raw.exists():
        raise FileNotFoundError(f"Raw mesh missing for {name}: {raw}")
    run(
        [BLENDER, "--background", "--python", ROOT / "blender_postprocess.py"],
        env=postprocess_env(candidate),
        log_path=run_dir / f"{name}_postprocess.log",
        timeout=240,
    )
    result = evaluate(name, run_dir)
    result["candidate_settings"] = candidate
    return result


def existing_candidates():
    found = []
    for path in sorted(OUTPUT_DIR.glob("*_ue.glb")):
        stem = path.stem[:-3] if path.stem.endswith("_ue") else path.stem
        found.append({
            "name": stem,
            "quality": "very_high",
            "asset_target": "unknown",
            "multi_mode": "unknown",
        })
    return found


def write_summary(results, run_dir):
    ranked = sorted(results, key=lambda r: r["comparison"]["accuracy_score"], reverse=True)
    summary = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "reference": str(REFERENCE),
        "source_dir": str(SOURCE_DIR),
        "best": ranked[0] if ranked else None,
        "ranked": [
            {
                "name": item["candidate_settings"]["name"],
                "score": item["comparison"]["accuracy_score"],
                "mean_projection_iou": item["comparison"]["mean_projection_iou"],
                "profile_error": item["comparison"]["profile_error"],
                "dimension_error": item["comparison"]["dimension_error"],
                "bottom_artifact_delta": item["comparison"]["bottom_artifact_delta"],
                "cavity_delta": item["comparison"].get("cavity_delta"),
                "output": item["candidate"]["path"],
            }
            for item in ranked
        ],
    }
    path = run_dir / "summary.json"
    path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary, path


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--generate", action="store_true", help="Run TRELLIS generation for candidates before postprocess/evaluation.")
    parser.add_argument("--evaluate-existing", action="store_true", help="Only evaluate existing Outputs/*_ue.glb files.")
    parser.add_argument("--max-candidates", type=int, default=2, help="Maximum configured candidates to run when --generate is set.")
    parser.add_argument("--candidate", default="", help="Run one configured candidate by exact name.")
    return parser.parse_args()


def main():
    args = parse_args()
    if not REFERENCE.exists():
        raise FileNotFoundError(REFERENCE)
    if not SOURCE_DIR.exists():
        raise FileNotFoundError(SOURCE_DIR)
    run_dir = LOG_ROOT / datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)

    if args.evaluate_existing:
        candidates = existing_candidates()
        results = []
        for candidate in candidates:
            try:
                result = evaluate(candidate["name"], run_dir)
                result["candidate_settings"] = candidate
                results.append(result)
                print(f"{candidate['name']}: {result['comparison']['accuracy_score']}")
            except Exception as exc:
                print(f"FAILED {candidate['name']}: {exc}", file=sys.stderr)
    else:
        if args.candidate:
            candidates = [candidate for candidate in CANDIDATES if candidate["name"] == args.candidate]
            if not candidates:
                raise ValueError(f"Unknown candidate: {args.candidate}")
        else:
            candidates = CANDIDATES[: max(1, args.max_candidates)]
        results = []
        for candidate in candidates:
            print(f"Running candidate {candidate['name']}")
            result = process_candidate(candidate, run_dir, args.generate)
            results.append(result)
            print(f"  score={result['comparison']['accuracy_score']}")

    if not results:
        raise RuntimeError("No candidates were evaluated")
    summary, path = write_summary(results, run_dir)
    print(json.dumps(summary["ranked"], indent=2))
    print(f"Summary: {path}")


if __name__ == "__main__":
    main()
