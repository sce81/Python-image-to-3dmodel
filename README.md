# Python Image to 3D Model

Local Windows image-to-3D pipeline for detail props and greebles:

```text
approved raster -> TRELLIS / TRELLIS.2 -> validation -> Blender post-process -> GLB / Blend
```

## Core operation

Use Python 3.11. Install the Windows runtime with `./setup_cuda13.ps1`, then run `python generate_props.py` for the multi-view TRELLIS route. Use `generate_props_t2.py` only through the isolated TRELLIS.2 runtime documented in [Instructions/Pipeline/05-trellis2-geometry-and-extraction.md](Instructions/Pipeline/05-trellis2-geometry-and-extraction.md). Run Blender externally with `blender_postprocess.py`.

`mcp_server.py` exposes the same workers over streamable HTTP. Configuration and secrets remain local environment variables.

## Documentation

- [Runtime and boundaries](Instructions/Pipeline/01-runtime-and-boundaries.md)
- [Source gate and conditioning](Instructions/Pipeline/02-contract-and-source-gate.md)
- [TRELLIS geometry](Instructions/Pipeline/04-trellis1-geometry.md)
- [TRELLIS.2 geometry and extraction](Instructions/Pipeline/05-trellis2-geometry-and-extraction.md)
- [TRELLIS.2 texturing](Instructions/Pipeline/06-trellis2-texturing.md)
- [Validation and release](Instructions/Pipeline/07-mesh-validation.md)
- [AWS G7 AMI build](packer/README.md)
- [Codex pipeline skill](.codex/skills/nexus-asset-pipeline/SKILL.md)

Runtime inputs, generated outputs, logs, model weights, credentials, and machine-local configuration are ignored by Git.

## Agentic stage controller

`agentic_pipeline.py` keeps per-asset state and decision records under `WorkingFolder/agent_runs/`, while TRELLIS and Blender remain independent workers. Initialize an approved source with `init`, run a named stage, and use `approve` only after human review. Geometry and texture promotion remain separate gates.

```powershell
python .\agentic_pipeline.py init --asset-id <asset_id> --contract .\WorkingFolder\asset_contracts\<asset_id>.json --raster-approved
python .\agentic_pipeline.py run --asset-id <asset_id> --stage geometry
python .\agentic_pipeline.py approve --asset-id <asset_id> --stage geometry_review --reviewer <reviewer>
```
