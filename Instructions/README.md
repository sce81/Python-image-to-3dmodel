# Documentation Index

This folder holds human-facing reference material. It is not read by the runtime pipeline.

| File | Purpose |
| --- | --- |
| `DEPLOY_WINDOWS.md` | Reproducible Windows, CUDA, TRELLIS, Blender, and MCP server setup. |
| `EXTERNAL_MESHY_CAR_WORKFLOW.md` | Historical workflow for an external Meshy service. It is not part of the local TRELLIS-to-Blender pipeline. |

## Documentation naming

Use `UPPER_SNAKE_CASE.md` for durable project documents. Begin each document with a title that states its scope and whether it is runtime guidance, deployment guidance, or reference-only material.

Runtime entry points remain at the project root and use `snake_case`: `generate_props.py`, `blender_postprocess.py`, and `mcp_server.py`.
