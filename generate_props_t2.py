"""
Nexus Protocol - TRELLIS.2 worker (official Microsoft API, isolated venv).

Implements the same NEXUS_* env contract as generate_props.py but drives the
official checkout's ``trellis2.pipelines.Trellis2ImageTo3DPipeline``
(microsoft/TRELLIS.2-4B) and exports through ``o_voxel.postprocess.to_glb``,
per https://github.com/microsoft/TRELLIS.2. This is the real TRELLIS.2
validation path; generate_props.py remains the TRELLIS 1 production worker
(and the only multi-view route - TRELLIS.2 conditions on a single image).

Run with the dedicated venv (Torch 2.10+cu130 - the compiled o_voxel/flex_gemm
wheels are ABI-bound to it; never the main pipeline venv):

    & .\tools\ComfyUI\.venv-trellis2\Scripts\python.exe .\generate_props_t2.py

Hardware tuning (RTX 4080 Super 16 GB, 48 GB RAM):
- low_vram stage offloading is forced on (idle stages live in system RAM).
- pipeline_type defaults to '1024_cascade'; on OOM drop NEXUS_T2_MAX_TOKENS
  (e.g. 32768) or set NEXUS_T2_PIPELINE_TYPE=512. '1536_cascade' will OOM.
- Sparse attention runs on the SDPA shim (trellis2_sdpa_shim.py) - there is no
  flash-attn/xformers build for Torch 2.10+cu130 on Windows.
- Sampler params come from the model's pipeline.json (steps 12, guidance 7.5
  with rescale/interval - a different sampler family from TRELLIS 1). Do NOT
  blanket-override them; NEXUS_T2_* envs exist for deliberate experiments only.
"""

import json
import os
import sys
from pathlib import Path

# Environment must be set before generate_props/trellis2 imports read it.
os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("ATTN_BACKEND", "xformers")  # served by the SDPA shim
os.environ.setdefault("SPARSE_CONV_BACKEND", "flex_gemm")

ROOT = Path(__file__).resolve().parent
TRELLIS2_DIR = Path(os.environ.get(
    "NEXUS_TRELLIS2_DIR", str(Path(__file__).resolve().parent.parent / "TRELLIS.2")))
MODEL_DIR = Path(os.environ.get(
    "NEXUS_T2_MODEL", ROOT / "tools" / "ComfyUI" / "models" / "microsoft" / "TRELLIS.2-4B"))
DINOV3_DIR = Path(os.environ.get(
    "NEXUS_DINOV3_PATH", ROOT / "tools" / "ComfyUI" / "models" / "facebook" / "dinov3-vitl16-pretrain-lvd1689m"))
PIPELINE_TYPE = os.environ.get("NEXUS_T2_PIPELINE_TYPE", "1024_cascade")
MAX_NUM_TOKENS = int(os.environ.get("NEXUS_T2_MAX_TOKENS", "49152"))
DECIMATION_TARGET = int(os.environ.get("NEXUS_T2_DECIMATION", "1000000"))

if PIPELINE_TYPE not in {"512", "1024", "1024_cascade", "1536_cascade"}:
    raise ValueError("NEXUS_T2_PIPELINE_TYPE must be one of: 512, 1024, 1024_cascade, 1536_cascade")
if PIPELINE_TYPE == "1536_cascade":
    print("WARNING: 1536_cascade exceeds 16 GB VRAM; expect OOM on this host.")
if not TRELLIS2_DIR.exists():
    raise SystemExit(f"TRELLIS.2 checkout not found: {TRELLIS2_DIR} (set NEXUS_TRELLIS2_DIR)")
if not (MODEL_DIR / "pipeline.json").exists():
    raise SystemExit(f"TRELLIS.2-4B weights not found: {MODEL_DIR} (set NEXUS_T2_MODEL)")
sys.path.insert(0, str(TRELLIS2_DIR))

import torch  # noqa: E402

import trellis2_sdpa_shim  # noqa: E402
# Reuse the pipeline's model-agnostic pieces: masking + clay conditioning,
# best-of-N seed selection, integrity metrics, logging, and export helpers.
from generate_props import (  # noqa: E402
    GEOMETRY_ONLY,
    GLB_PARAMS,
    IMAGE_EXTENSIONS,
    MANIFEST,
    OUTPUT_DIR,
    SCREENSHOT_ROOT,
    audit_baked_texture,
    preprocess_subject,
    safe_slug,
    sample_best_candidate,
    setup_logging,
    to_geometry_only_glb,
)


def _stub_pipeline_rembg():
    """The adapter always supplies masked RGBA input; the pipeline's BiRefNet
    (gated briaai/RMBG-2.0) would eager-download at load for nothing."""
    import trellis2.pipelines.trellis2_image_to_3d as t2m

    class _AdapterMasksOnly:
        def __init__(self, **_):
            print("  pipeline rembg stubbed (adapter provides masked input)")

        def to(self, *_):
            pass

        def cuda(self):
            pass

        def cpu(self):
            pass

        def __call__(self, image):
            raise RuntimeError(
                "Pipeline rembg is stubbed. Run with NEXUS_FOREGROUND_ONLY=1 (default) "
                "so masking happens in preprocess_subject, or supply RGBA input."
            )

    t2m.rembg.BiRefNet = _AdapterMasksOnly


def _patch_dinov3_encoder_layout():
    """transformers 5.x nests the DINOv3 encoder at DINOv3ViTModel.model;
    TRELLIS.2 iterates DINOv3ViTModel.layer (the 4.x layout). Alias the layer
    list after load so the checkout stays at the pinned commit. Embeddings,
    rope_embeddings, and the per-layer forward signature are unchanged in 5.x."""
    import trellis2.modules.image_feature_extractor as ife

    original_init = ife.DinoV3FeatureExtractor.__init__

    def patched_init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        if not hasattr(self.model, "layer"):
            self.model.layer = self.model.model.layer
            print("  DINOv3: aliased encoder layers for transformers 5.x layout")

    ife.DinoV3FeatureExtractor.__init__ = patched_init


def _prepare_config() -> str:
    """Write pipeline_nexus.json: pipeline.json with DINOv3 pointed at the
    local snapshot (facebook's repo is gated - no anonymous runtime download)."""
    config = json.loads((MODEL_DIR / "pipeline.json").read_text(encoding="utf-8"))
    cond_args = config["args"]["image_cond_model"]["args"]
    if (DINOV3_DIR / "model.safetensors").exists():
        cond_args["model_name"] = str(DINOV3_DIR)
        print(f"DINOv3: {DINOV3_DIR}")
    else:
        print(f"WARNING: local DINOv3 snapshot not found at {DINOV3_DIR}")
        print("         facebook/dinov3-vitl16-pretrain-lvd1689m is gated - accept the")
        print("         license on huggingface.co and `hf auth login`, or set NEXUS_DINOV3_PATH.")
    target = MODEL_DIR / "pipeline_nexus.json"
    target.write_text(json.dumps(config, indent=1), encoding="utf-8")
    return target.name


def load_pipeline():
    if trellis2_sdpa_shim.install():
        print("sparse attention: SDPA shim (no xformers/flash-attn for Torch 2.10+cu130 on Windows)")
    from trellis2.pipelines import Trellis2ImageTo3DPipeline

    _stub_pipeline_rembg()
    _patch_dinov3_encoder_layout()
    config_file = _prepare_config()
    pipe = Trellis2ImageTo3DPipeline.from_pretrained(str(MODEL_DIR), config_file)
    pipe.low_vram = True  # 16 GB card: keep idle stages in system RAM
    pipe.cuda()
    return pipe


def _sampler_overrides(prefix: str) -> dict:
    """Optional env overrides; empty dict means the pipeline.json defaults."""
    params = {}
    steps = os.environ.get(f"NEXUS_T2_{prefix}_STEPS")
    guidance = os.environ.get(f"NEXUS_T2_{prefix}_GUIDANCE")
    if steps:
        params["steps"] = int(steps)
    if guidance:
        params["guidance_strength"] = float(guidance)
    return params


def export_mesh(mesh, out_path: Path):
    """Extract every review mesh through the official O-Voxel path.

    Geometry-only means defer texture acceptance; it must not bypass the
    extraction/remesh stage. Direct latent export preserved tens of thousands
    of open boundaries in controlled T2 comparisons, while the full official
    extraction was the successful single-image route.
    """
    import o_voxel

    mesh.simplify(16_777_216)  # nvdiffrast limit, per the official example
    glb = o_voxel.postprocess.to_glb(
        vertices=mesh.vertices,
        faces=mesh.faces,
        attr_volume=mesh.attrs,
        coords=mesh.coords,
        attr_layout=mesh.layout,
        voxel_size=mesh.voxel_size,
        aabb=[[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
        decimation_target=DECIMATION_TARGET,
        texture_size=GLB_PARAMS["texture_size"],
        remesh=True,
        remesh_band=1,
        remesh_project=0,
        verbose=True,
    )
    glb.export(str(out_path))


def generate_one(pipe, image_path: Path, out_path: Path) -> dict:
    screenshot_dir = SCREENSHOT_ROOT / out_path.stem
    image = preprocess_subject(pipe, image_path, screenshot_dir)

    ss_params = _sampler_overrides("SS")
    shape_params = _sampler_overrides("SHAPE")
    tex_params = _sampler_overrides("TEX")

    def run_sampler(seed: int):
        meshes = pipe.run(
            image,
            seed=seed,
            sparse_structure_sampler_params=ss_params,
            shape_slat_sampler_params=shape_params,
            tex_slat_sampler_params=tex_params,
            preprocess_image=False,
            pipeline_type=PIPELINE_TYPE,
            max_num_tokens=MAX_NUM_TOKENS,
        )
        return {"mesh": meshes}

    outputs, selection = sample_best_candidate(run_sampler)
    mesh = outputs["mesh"][0]
    export_mesh(mesh, out_path)

    texture_audit = {"status": "deferred_geometry_only"}
    if not GEOMETRY_ONLY:
        try:
            import trimesh

            texture_audit = audit_baked_texture(trimesh.load(str(out_path), force="scene"), image_path)
        except Exception as error:  # PBR multi-map GLB; audit is TRELLIS 1 calibrated
            texture_audit = {"status": "audit_unavailable", "detail": f"{type(error).__name__}: {error}"}

    return {
        "generator": "trellis2",
        "source": image_path.name,
        "output": out_path.name,
        "pipeline_type": PIPELINE_TYPE,
        "max_num_tokens": MAX_NUM_TOKENS,
        "texture_size": GLB_PARAMS["texture_size"],
        "decimation_target": DECIMATION_TARGET,
        "texture_audit": texture_audit,
        "texture_review_required": not GEOMETRY_ONLY,
        "screenshots": str(screenshot_dir),
        **selection,
    }


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    single = os.environ.get("NEXUS_SINGLE")
    source_dir = os.environ.get("NEXUS_SOURCE_DIR")
    log_label = os.environ.get("NEXUS_ASSET_NAME") or (
        Path(source_dir).name if source_dir else Path(single).stem if single else "batch"
    )
    setup_logging(f"t2_{log_label}")

    refs: list[Path]
    if source_dir:
        folder = Path(source_dir)
        found = sorted(p for ext in IMAGE_EXTENSIONS for p in folder.glob(ext) if p.is_file())
        if len(found) != 1:
            print(
                f"TRELLIS.2 conditions on a single image; {folder} has {len(found)}. "
                "Use generate_props.py (TRELLIS 1) for multi-view assets."
            )
            return
        refs = found
    elif single:
        ref = Path(single)
        if not ref.exists():
            print(f"NEXUS_SINGLE not found: {ref}")
            return
        refs = [ref]
    else:
        refs = sorted(p for ext in IMAGE_EXTENSIONS for p in (ROOT / "Inputs").glob(ext))
        if not refs:
            print(f"No reference images in {(ROOT / 'Inputs').resolve()}")
            return

    pipe = load_pipeline()
    manifest = []
    for index, ref in enumerate(refs, 1):
        mesh_id = safe_slug(os.environ.get("NEXUS_ASSET_NAME") or ref.stem) if len(refs) == 1 \
            else safe_slug(ref.stem)
        out = OUTPUT_DIR / f"{mesh_id}.glb"
        print(f"[{index}/{len(refs)}] {ref.name} -> {out.name} "
              f"(t2 {PIPELINE_TYPE}, max_tokens={MAX_NUM_TOKENS}, geometry_only={GEOMETRY_ONLY})")
        try:
            manifest.append(generate_one(pipe, ref, out))
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            print(
                f"  OOM on {ref.name}. Levers: NEXUS_T2_MAX_TOKENS=32768, "
                "NEXUS_T2_PIPELINE_TYPE=512, close other GPU apps."
            )
        except Exception as error:
            print(f"  FAILED {ref.name}: {type(error).__name__}: {error}")

    if manifest:
        MANIFEST.write_text(json.dumps(manifest, indent=2))
        print(f"\nDone. {len(manifest)} meshes -> {OUTPUT_DIR.resolve()}")
        print(f"Manifest: {MANIFEST}")


if __name__ == "__main__":
    main()
