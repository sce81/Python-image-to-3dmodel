# AGENTS.md - Nexus Asset Pipeline

Local AI 3D-asset pipeline for *Nexus Protocol* (UE5.7). Image -> mesh (TRELLIS 2) -> retopo/grid-snap (Blender) -> UE-ready GLB/Blend. Exposed as an MCP server so the M4 Pro laptop drives generation on the Windows RTX 4080 Super box over LAN.

## Runtime / versions (hard constraints)

- Python **3.11** only. Do NOT use 3.12 - some 3D deps break.
- Torch **2.13.0** / torchvision 0.28.0, **CUDA 13.2 build** (cu132 index), compiled with CUDA Toolkit **13.3**. Torchaudio is not required.
- Blender **5.1.x**, invoked as an external binary, not `pip install bpy`.
- MCP transport: **streamable-http** (cross-machine). Never switch to stdio for the networked server.
- Windows is the only supported deploy target for generation+texturing (PBR bake needs CUDA/nvdiffrast). Mac is geometry-only fallback - do not add Mac texturing code paths.
- TRELLIS checkout currently used: `C:\Users\simon\Documents\UnrealEngine-ProjectWork\TRELLIS`, commit `442aa1e1afb9014e80681d3bf604e8d728a86ee7` (`442aa1e`), latest local commit date 2025-11-05.

## Setup command (run in repo root)

```powershell
.\setup_cuda13.ps1
```

TRELLIS is cloned separately at the pinned commit listed above. The validated Windows stack uses xFormers 0.0.35 from cu130, spconv-cu124 2.3.8 with CUDA 12.4 runtime DLLs side-by-side, and locally rebuilt nvdiffrast/Gaussian rasterizer extensions against CUDA Toolkit 13.3. See DEPLOY_WINDOWS.md.

## Run / test commands

```powershell
# 1. Verify CUDA before anything else
python -c "import torch; print(torch.cuda.is_available())"   # must print True

# 2. Smoke-test the raw pipeline (do this BEFORE touching the MCP layer)
python generate_props.py            # batch: Inputs\*.png -> raw_meshes\*.glb
& "C:\Program Files\Blender Foundation\Blender 5.1\blender.exe" --background --python blender_postprocess.py

# 3. Single-item mode (how the MCP server calls the scripts)
$env:NEXUS_SINGLE="Inputs\test.png"; python generate_props.py
$env:NEXUS_MESH="test"; & "C:\...\blender.exe" --background --python blender_postprocess.py

# 3b. Multi-image mode: all images in a folder are source information for one asset
$env:NEXUS_SOURCE_DIR="C:\Users\simon\Documents\3D-Imaging-Pipeline\Inputs\Cars\Mercedes\EQE"
$env:NEXUS_ASSET_NAME="Mercedes_EQE"
$env:NEXUS_MULTI_MODE="stochastic"
$env:NEXUS_QUALITY="very_high"
$env:NEXUS_ASSET_TARGET="prop"
python generate_props.py

# 4. Launch the server
python mcp_server.py                 # serves http://0.0.0.0:8765/mcp
```

There is no unit-test suite. The smoke test in step 2 is the acceptance check: final `.glb` and `.blend` files in `Outputs\` mean the change is good. Always run it after editing either script when practical.

## Architecture

- `mcp_server.py` is a thin wrapper. It does NOT contain generation logic - it shells out to `generate_props.py` and `blender_postprocess.py` via env vars. Keep it thin.
- The scripts have multiple modes: **batch** (scan `Inputs/`), **single** (env vars `NEXUS_SINGLE` / `NEXUS_MESH`), and **multi-image asset** (`NEXUS_SOURCE_DIR` + `NEXUS_ASSET_NAME`). Preserve all modes when editing.
- Multi-image folder mode treats every image in the source folder as source information for one asset. Do not generate one mesh per image unless explicitly requested.
- `generate_props.py` writes validation screenshots before mesh generation to `Outputs\Screenshots\<ModelName>` and a `contact_sheet.png` for multi-image mode.
- `blender_postprocess.py` exports Blender-compatible `.glb` and `.blend` files to `Outputs\`.
- Logs go under `Logs\`.
- Env-var contract between server and scripts - do not rename these: `NEXUS_SINGLE`, `NEXUS_SOURCE_DIR`, `NEXUS_ASSET_NAME`, `NEXUS_TEXSIZE`, `NEXUS_MESH`, `NEXUS_TRIS`, `NEXUS_GRID`, `NEXUS_QUALITY`, `NEXUS_ASSET_TARGET`, `NEXUS_MULTI_MODE`, `NEXUS_LENGTH_AXIS`, `NEXUS_SMOOTH_NORMALS`, `NEXUS_REFERENCE_MODEL`, `NEXUS_MATCH_REFERENCE_DIMS`.

## MeshAI Vehicle Quality Target

- `Meshai-EQE-Model.glb` is the primary EQE quality reference.
- User visual review is authoritative. Automatic geometry scores are useful for triage but can prefer artifact-heavy outputs.
- Current human-preferred baseline: `Outputs\MeshAI_D_stochastic_preserve_features_ue.glb`.
- Higher-scoring `MeshAI_C_meshi_car_more_structure` has too many visible artifacts; do not treat it as the default.
- D-family postprocess variants (`MeshAI_D2_stochastic_meshy_tris`, `MeshAI_D4_stochastic_no_decimate`, `MeshAI_D5_stochastic_crisper_normals`) are useful visual-review candidates.
- Do not use the filtered/no-reflection branch as default. Excluding `06-mercedes-benz-eqe-saloon-5120x2880.jpeg` and enabling `NEXUS_REMOVE_REFLECTIONS=1` produced bad results.
- `NEXUS_REMOVE_REFLECTIONS` is experimental and must remain off unless explicitly testing it.

## MeshAI Source Image Standard

- Use `Inputs\Cars\MeshAI` as the reference for how source imagery should look before TRELLIS.
- These are normalized studio renders, not raw car photos.
- Preferred source image format:
  - square `1024 x 1024` RGB image
  - white or near-white studio background
  - vehicle fills about `92% - 96%` of image width
  - vehicle height about `48% - 52%` of image height
  - top margin about `25% - 28%`
  - bottom margin about `24% - 25%`
  - clean bottom band, with no floor reflection, pedestal, shadow band, or duplicate lower reflection
- Normalize raw photo references into this studio-render shape before generation. Do not just crop arbitrary lower bands; that degraded output quality.

## Domain constraint (critical - do not "fix" this)

AI generation is for **detail props / greebles only**. Base modular tiles (75x25x200cm walls, 75x75x200cm corners) are procedurally authored on-grid and must NOT be routed through this pipeline - generated meshes are organic and break grid snapping. `GRID_CM` defaults to 75; keep bbox sanity checks against 75cm multiples.

## Code style

- Targeted edits over rewrites. Match existing structure; don't restructure files.
- Keep the config constants at the top of each script as the single tuning surface.
- Prose in comments/docs: direct and plain. No hedging, no corporate filler.

## Boundaries - do NOT touch

- Never commit `NEXUS_MCP_TOKEN`, HF tokens, or any secret. They live in env vars only.
- Do not edit files under `raw_meshes/`, `Inputs/`, `Outputs/` unless the user explicitly asks; these are generated/input I/O.
- Do not add internet-facing exposure (port-forward, 0.0.0.0 on a public NIC). Remote access is Tailscale/VPN only; the token is a gate, not real auth.
- Do not weaken the `_auth` token check or bind the server outside the LAN.
- Do not add a `<form>`-style or auto-submit anything; this repo takes no web input.

## When you finish a task

Run the step-2 smoke test after editing either script when practical. For the current pipeline, report the final `Outputs\*_ue.glb` and `Outputs\*_ue.blend` filenames. If TRELLIS imports fail, the likely cause is the repo's module layout drifting from the import lines at the top of `generate_props.py` - flag it, don't guess.
