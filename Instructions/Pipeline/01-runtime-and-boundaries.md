# Runtime and boundaries

- Use Python 3.11 only. Main TRELLIS uses Torch 2.13/cu132; T2 uses `tools\ComfyUI\.venv-trellis2` with Torch 2.10/cu130.
- Use Blender 5.1 externally. Keep MCP streamable HTTP and its token check unchanged.
- Do not edit generated `Inputs`, `raw_meshes`, `Outputs`, logs, or secrets unless the user has requested the operation.
- Generate detail props and greebles only. Never route modular grid tiles through AI reconstruction.
- Preserve all `NEXUS_*` environment-variable names and batch, single, and multi-image modes.
