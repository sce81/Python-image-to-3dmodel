# Nexus Asset Pipeline

Nexus Asset Pipeline is a local Windows workflow that turns reference images into UE-ready 3D detail props. It runs TRELLIS 2 to create a textured raw mesh, then uses Blender to clean and export a compatible `.glb` and `.blend`.

The pipeline is for organic detail props and greebles, such as terminals, crates, signage, and vehicle assets. It must not be used for base modular tiles: those stay procedurally authored on the 75 cm grid.

## Pipeline stages

| Stage | Entry point | Result |
| --- | --- | --- |
| Generate | `generate_props.py` | Reference image(s) to `raw_meshes/<asset_id>.glb`. |
| Postprocess | `blender_postprocess.py` | Grid-checked, UE-ready `Outputs/<asset_id>_ue.glb` and `.blend`. |
| Remote control | `mcp_server.py` | Authenticated streamable-HTTP MCP access to the two stages over LAN/VPN. |

For vehicle-reference experiments, `evaluate_model_similarity.py` measures output against the MeshAI EQE model and `learning_loop.py` runs repeatable candidates.

## System requirements

- Windows generation host with an NVIDIA RTX GPU; the validated host is an RTX 4080 Super.
- Python 3.11 only.
- NVIDIA CUDA Toolkit 13.3.
- PyTorch 2.13.0 with the CUDA 13.2 build and torchvision 0.28.0.
- Blender 5.1.x installed externally at `C:\Program Files\Blender Foundation\Blender 5.1\blender.exe`.
- Separate TRELLIS checkout at the pinned commit documented in [AGENTS.md](AGENTS.md).
- 16 GB VRAM for the validated full-quality TRELLIS workflow.

## Install and deploy

1. Follow [Windows deployment instructions](Instructions/DEPLOY_WINDOWS.md) to install system dependencies, clone TRELLIS, create the environment, and configure the MCP server.
2. In the project root, run `./setup_cuda13.ps1` to build the validated Python environment from [requirements.txt](requirements.txt).
3. Verify CUDA before generation:

   ```powershell
   & .\.venv\Scripts\python.exe -c "import torch; print(torch.cuda.is_available())"
   ```

4. Run the local two-stage smoke test:

   ```powershell
   & .\.venv\Scripts\python.exe .\generate_props.py
   & "C:\Program Files\Blender Foundation\Blender 5.1\blender.exe" --background --python .\blender_postprocess.py
   ```

5. Launch the LAN/VPN MCP server when the smoke test succeeds:

   ```powershell
   & .\.venv\Scripts\python.exe .\mcp_server.py
   ```

The MCP server uses streamable HTTP on port 8765. Keep it on a trusted LAN or VPN; do not expose it publicly. Set `NEXUS_MCP_TOKEN` in the environment rather than committing credentials.

## Naming and layout

| Location | Purpose | Naming |
| --- | --- | --- |
| `Inputs/` | Reference images. | Preserve supplied source names; use one folder per multi-image asset. |
| `raw_meshes/` | TRELLIS stage output. | `<asset_id>.glb`. |
| `Outputs/` | Final exports. | `<asset_id>_ue.glb` and `<asset_id>_ue.blend`. |
| `Outputs/Screenshots/` | Input validation previews. | `<asset_id>/`; multi-image assets include `contact_sheet.png`. |
| `Logs/` | Stage logs. | `YYYYMMDD_HHMMSS_<stage>_<asset_id>.log`. |
| `Instructions/` | Human-facing reference and deployment documents. | `UPPER_SNAKE_CASE.md`. |

Use lowercase `snake_case` for new executable scripts and asset IDs. Keep final UE exports suffixed `_ue`; put experiment names before that suffix, for example `mercedes_eqe_d4_ue.glb`.

See [AGENTS.md](AGENTS.md) for runtime constraints and supported environment variables. See [Instructions/README.md](Instructions/README.md) for the documentation index.
