"""
Nexus Protocol â€” Asset Pipeline MCP Server
Runs on the Windows PC (RTX 4080 Super). Actioned from the M4 Pro laptop.

Exposes the TRELLIS -> Blender pipeline as MCP tools over the network so any MCP
client (Claude Desktop / Claude Code / your own agent) on the laptop can drive it.

Transport: streamable-HTTP (works across machines; stdio would be same-host only).

Run on Windows:
    pip install "mcp[cli]"
    python mcp_server.py
    # serves at http://0.0.0.0:8765/mcp

Then from the laptop, point your MCP client at:
    http://<WINDOWS-LAN-IP>:8765/mcp

SECURITY: this binds to your LAN. Keep it on a trusted private network. The token
check below is a minimal gate, not real auth â€” do NOT expose this to the internet
or port-forward it. For remote access use a VPN / Tailscale, not a public port.
"""

import os
import subprocess
import base64
import json
import uuid
from datetime import datetime
from pathlib import Path

from mcp.server.fastmcp import FastMCP

# --- Paths (Windows) ----------------------------------------------------
ROOT        = Path(r"C:\Users\simon\Documents\3D-Imaging-Pipeline")
INPUT_DIR   = ROOT / "Inputs"
RAW_DIR     = ROOT / "raw_meshes"
UE_DIR      = ROOT / "Outputs"
LOG_DIR     = ROOT / "Logs"
VENV_PY     = ROOT / ".venv" / "Scripts" / "python.exe"   # your pipeline venv
BLENDER     = r"C:\Program Files\Blender Foundation\Blender 5.1\blender.exe"

for d in (INPUT_DIR, RAW_DIR, UE_DIR, LOG_DIR):
    d.mkdir(parents=True, exist_ok=True)

# Minimal shared-secret gate. Set the same value as an env var on both machines.
EXPECTED_TOKEN = os.environ.get("NEXUS_MCP_TOKEN", "")

mcp = FastMCP("nexus-asset-pipeline", host="0.0.0.0", port=8765)


def _auth(token: str) -> None:
    if EXPECTED_TOKEN and token != EXPECTED_TOKEN:
        raise PermissionError("Bad or missing NEXUS_MCP_TOKEN.")


def _validate_quality(quality: str) -> str:
    quality = quality.lower()
    if quality not in {"high", "very_high"}:
        raise ValueError("quality must be 'high' or 'very_high'")
    return quality


def _validate_asset_target(asset_target: str) -> str:
    asset_target = asset_target.lower()
    if asset_target not in {"prop", "body_shell", "meshai_car"}:
        raise ValueError("asset_target must be 'prop', 'body_shell', or 'meshai_car'")
    return asset_target


def _log_subprocess(stage: str, mesh_id: str, proc: subprocess.CompletedProcess) -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    slug = "".join(c if c.isalnum() or c in "-_" else "_" for c in mesh_id).strip("_") or "mesh"
    log_path = LOG_DIR / f"{stamp}_{stage}_{slug}.log"
    log_path.write_text(
        f"returncode={proc.returncode}\n\n[stdout]\n{proc.stdout or ''}\n\n[stderr]\n{proc.stderr or ''}",
        encoding="utf-8",
    )
    return log_path


@mcp.tool()
def generate_prop(image_base64: str, name: str, quality: str = "high",
                  asset_target: str = "prop",
                  texture_size: int = 0,
                  token: str = "") -> str:
    """
    Generate a raw 3D mesh from a reference image using TRELLIS 2 on the 4080 Super.

    image_base64: PNG/JPG reference image, base64-encoded (sent from the laptop).
    name:         slug for the output file (no extension).
    quality:      'high' or 'very_high'.
    asset_target: 'prop' to preserve source fidelity; 'meshai_car' to match MeshAI-style car shells; 'body_shell' only for explicit shell conversion.
    texture_size: optional override; 0 uses the quality profile.
    Returns JSON: {mesh_id, raw_glb, status}.
    """
    _auth(token)
    quality = _validate_quality(quality)
    asset_target = _validate_asset_target(asset_target)
    slug = "".join(c for c in name if c.isalnum() or c in "-_") or uuid.uuid4().hex[:8]
    ref = INPUT_DIR / f"{slug}.png"
    ref.write_bytes(base64.b64decode(image_base64))

    # Reuse your generate script but target a single file via env override.
    env = dict(os.environ, NEXUS_SINGLE=str(ref), NEXUS_QUALITY=quality,
               NEXUS_ASSET_TARGET=asset_target)
    if texture_size > 0:
        env["NEXUS_TEXSIZE"] = str(texture_size)
    proc = subprocess.run(
        [str(VENV_PY), str(ROOT / "generate_props.py")],
        cwd=str(ROOT), env=env, capture_output=True, text=True,
    )
    log_path = _log_subprocess("generate", slug, proc)
    raw = RAW_DIR / f"{slug}.glb"
    screenshots = UE_DIR / "Screenshots" / slug
    status = "ok" if raw.exists() else "failed"
    return json.dumps({
        "mesh_id": slug,
        "raw_glb": str(raw) if raw.exists() else None,
        "screenshots": str(screenshots) if screenshots.exists() else None,
        "status": status,
        "quality": quality,
        "asset_target": asset_target,
        "log": str(log_path),
        "log_tail": (proc.stdout or proc.stderr)[-800:],
    })


@mcp.tool()
def postprocess(mesh_id: str, quality: str = "high", target_tris: int = 0,
                grid_cm: float = 75.0,
                retopo_mode: str = "decimate",
                asset_target: str = "prop",
                symmetry: str = "none",
                symmetry_source: str = "positive_x",
                length_axis: str = "x",
                smooth_normals: bool = True,
                reference_model: str = "",
                match_reference_dims: bool = False,
                token: str = "") -> str:
    """
    Retopo + grid-snap + UE5.7-ready GLB export via Blender background mode.
    Runs on the Windows box; operates on a mesh already generated by generate_prop.
    Returns JSON: {mesh_id, ue_glb, status}.
    """
    _auth(token)
    quality = _validate_quality(quality)
    asset_target = _validate_asset_target(asset_target)
    env = dict(os.environ, NEXUS_MESH=mesh_id,
               NEXUS_QUALITY=quality, NEXUS_GRID=str(grid_cm),
               NEXUS_RETOPO=retopo_mode, NEXUS_ASSET_TARGET=asset_target,
               NEXUS_SYMMETRY=symmetry, NEXUS_SYMMETRY_SOURCE=symmetry_source,
               NEXUS_LENGTH_AXIS=length_axis,
               NEXUS_SMOOTH_NORMALS="1" if smooth_normals else "0",
               NEXUS_REFERENCE_MODEL=reference_model,
               NEXUS_MATCH_REFERENCE_DIMS="1" if match_reference_dims else "0")
    if target_tris > 0:
        env["NEXUS_TRIS"] = str(target_tris)
    proc = subprocess.run(
        [BLENDER, "--background", "--python", str(ROOT / "blender_postprocess.py")],
        cwd=str(ROOT), env=env, capture_output=True, text=True,
    )
    log_path = _log_subprocess("blender", mesh_id, proc)
    ue = UE_DIR / f"{mesh_id}_ue.glb"
    return json.dumps({
        "mesh_id": mesh_id,
        "ue_glb": str(ue) if ue.exists() else None,
        "status": "ok" if ue.exists() else "failed",
        "quality": quality,
        "asset_target": asset_target,
        "symmetry": symmetry,
        "symmetry_source": symmetry_source,
        "length_axis": length_axis,
        "smooth_normals": smooth_normals,
        "reference_model": reference_model or None,
        "match_reference_dims": match_reference_dims,
        "log": str(log_path),
        "log_tail": (proc.stdout or proc.stderr)[-800:],
    })


@mcp.tool()
def get_mesh(mesh_id: str, stage: str = "ue", token: str = "") -> str:
    """
    Return a finished GLB back to the laptop as base64.
    stage: 'ue' (processed) or 'raw' (pre-retopo).
    """
    _auth(token)
    path = (UE_DIR / f"{mesh_id}_ue.glb") if stage == "ue" else (RAW_DIR / f"{mesh_id}.glb")
    if not path.exists():
        return json.dumps({"error": f"not found: {path.name}"})
    return json.dumps({
        "mesh_id": mesh_id,
        "stage": stage,
        "filename": path.name,
        "glb_base64": base64.b64encode(path.read_bytes()).decode(),
    })


@mcp.tool()
def list_meshes(token: str = "") -> str:
    """List all generated and processed meshes on the Windows box."""
    _auth(token)
    return json.dumps({
        "raw": [p.stem for p in RAW_DIR.glob("*.glb")],
        "ue_ready": [p.stem.replace("_ue", "") for p in UE_DIR.glob("*_ue.glb")],
    })


if __name__ == "__main__":
    # Streamable-HTTP so the laptop can reach it over the LAN.
    mcp.run(transport="streamable-http")

