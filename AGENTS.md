# AGENTS.md - Nexus Asset Pipeline

Local Windows image-to-3D pipeline for Nexus Protocol. This file holds always-on constraints; load only the relevant stage document from [Instructions/Pipeline/00-index.md](Instructions/Pipeline/00-index.md) before acting.

## Runtime and architecture

- Use Python 3.11 only. Main TRELLIS uses Torch 2.13/cu132. TRELLIS.2 must run in `tools\ComfyUI\.venv-trellis2` with Torch 2.10/cu130.
- Use external Blender 5.1.x. Keep MCP transport streamable HTTP and retain the existing token gate.
- Keep `mcp_server.py` thin. Preserve batch, single, and multi-image modes and every documented `NEXUS_*` variable.
- TRELLIS 1 (`generate_props.py`) is the multi-view production route. TRELLIS.2 (`generate_props_t2.py`) is single-image only.
- T2 geometry-only review still uses official O-Voxel extraction. Never direct-export the latent T2 mesh.

## Stage documents

1. [Runtime and boundaries](Instructions/Pipeline/01-runtime-and-boundaries.md)
2. [Asset contract and source gate](Instructions/Pipeline/02-contract-and-source-gate.md)
3. [Conditioning renders](Instructions/Pipeline/03-conditioning-renders.md)
4. [TRELLIS 1 geometry](Instructions/Pipeline/04-trellis1-geometry.md)
5. [TRELLIS 2 geometry and extraction](Instructions/Pipeline/05-trellis2-geometry-and-extraction.md)
6. [TRELLIS 2 texturing](Instructions/Pipeline/06-trellis2-texturing.md)
7. [Mesh validation](Instructions/Pipeline/07-mesh-validation.md)
8. [Blender and release](Instructions/Pipeline/08-blender-and-release.md)

## I/O and safety

- Do not edit `Inputs/`, `raw_meshes/`, `Outputs/`, or secrets unless the user has explicitly requested it.
- Do not expose the service publicly, weaken `_auth`, or bind it beyond the LAN/VPN design.
- AI reconstruction is for detail props and greebles only; never route base modular grid tiles through it. Keep `GRID_CM` sanity checks.
- Preserve user changes in dirty worktrees. Use targeted edits and keep config tuning surfaces near the top of each script.

## Acceptance

- Gate source, extracted geometry, texture, and final exports in that order. User visual review is authoritative.
- Run the proportional smoke test after changing either generator or Blender script when practical. Report final `Outputs\*_ue.glb` and `Outputs\*_ue.blend` paths.
