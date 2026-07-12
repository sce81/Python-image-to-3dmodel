"""
Nexus Protocol Ã¢â‚¬â€ batch prop/detail generator (TRELLIS 2, local, RTX 4080 Super)

Stage 1 of the pipeline: image -> raw textured mesh (GLB).
Designed for DETAIL/PROP generation (greebles, terminals, crates, signage),
NOT for base modular tile shells Ã¢â‚¬â€ those stay procedural/hand-built on your grid.

Run headless in batches. Feeds Stage 2 (blender_postprocess.py) for retopo + grid snap.

Prereqs (one-time, in your venv):
    .\setup_cuda13.ps1
    # then TRELLIS per its repo install (spconv, flash-attn optional, etc.)

VRAM note: 16 GB (4080 Super) runs TRELLIS 2 at full quality. No quantization needed.
"""

import os
import json
import glob
import sys
import types
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import trimesh
from PIL import Image
import rembg

# --- TRELLIS import (adjust to your local install path/module name) ---
# from trellis.pipelines import TrellisImageTo3DPipeline
# from trellis.utils import postprocessing_utils

# Config -----------------------------------------------------------------
INPUT_DIR   = Path("./Inputs")          # drop reference images here (one prop per file)
OUTPUT_DIR  = Path("./raw_meshes")      # GLBs land here for Blender stage
MANIFEST    = OUTPUT_DIR / "manifest.json"
TRELLIS_DIR = Path(r"C:\Users\simon\Documents\UnrealEngine-ProjectWork\TRELLIS")
WARP_CACHE_DIR = Path("./.cache/warp")
PREPROCESS_DIR = Path("./.cache/preprocessed")
SCREENSHOT_ROOT = Path("./Outputs/Screenshots")
LOG_DIR = Path("./Logs")
IMAGE_EXTENSIONS = ("*.png", "*.jpg", "*.jpeg", "*.webp")

if TRELLIS_DIR.exists():
    sys.path.insert(0, str(TRELLIS_DIR))
os.environ.setdefault("WARP_CACHE_PATH", str(WARP_CACHE_DIR.resolve()))
os.environ.setdefault("ATTN_BACKEND", "sdpa")
os.environ.setdefault("SPARSE_ATTN_BACKEND", "xformers")

QUALITY = os.environ.get("NEXUS_QUALITY", "high").lower()
QUALITY_PRESETS = {
    "high": {
        "ss_steps": 25,
        "slat_steps": 25,
        "ss_guidance": 7.5,
        "slat_guidance": 7.5,
        "simplify": 0.6,
        "texture_size": 2048,
    },
    "very_high": {
        "ss_steps": 50,
        "slat_steps": 50,
        "ss_guidance": 7.5,
        "slat_guidance": 7.5,
        "simplify": 0.2,
        "texture_size": 4096,
    },
}
if QUALITY not in QUALITY_PRESETS:
    raise ValueError(f"NEXUS_QUALITY must be one of: {', '.join(QUALITY_PRESETS)}")
QUALITY_PROFILE = QUALITY_PRESETS[QUALITY]

# Higher = more surface richness and texture stability, slower.
GEN_PARAMS = {
    "seed": int(os.environ.get("NEXUS_SEED", "42")),
    "ss_guidance_strength": float(os.environ.get("NEXUS_SS_GUIDANCE", str(QUALITY_PROFILE["ss_guidance"]))),
    "ss_sampling_steps": int(os.environ.get("NEXUS_SS_STEPS", str(QUALITY_PROFILE["ss_steps"]))),
    "slat_guidance_strength": float(os.environ.get("NEXUS_SLAT_GUIDANCE", str(QUALITY_PROFILE["slat_guidance"]))),
    "slat_sampling_steps": int(os.environ.get("NEXUS_SLAT_STEPS", str(QUALITY_PROFILE["slat_steps"]))),
}

# GLB extraction: mesh simplification + texture bake resolution.
# simplify is the ratio of faces to remove. Keep it moderate for vehicles/hero props.
GLB_PARAMS = {
    "simplify": float(os.environ.get("NEXUS_SIMPLIFY", str(QUALITY_PROFILE["simplify"]))),
    "texture_size": int(os.environ.get("NEXUS_TEXSIZE", str(QUALITY_PROFILE["texture_size"]))),
}
FOREGROUND_ONLY = os.environ.get("NEXUS_FOREGROUND_ONLY", "1") != "0"
REMBG_MODEL = os.environ.get("NEXUS_REMBG_MODEL", "u2net")
REMOVE_REFLECTIONS = os.environ.get("NEXUS_REMOVE_REFLECTIONS", "0") == "1"
ASSET_TARGET = os.environ.get("NEXUS_ASSET_TARGET", "prop").lower()
if ASSET_TARGET not in {"prop", "body_shell", "meshai_car"}:
    raise ValueError("NEXUS_ASSET_TARGET must be 'prop', 'body_shell', or 'meshai_car'")
MULTI_IMAGE_MODE = os.environ.get("NEXUS_MULTI_MODE", "multidiffusion").lower()
if MULTI_IMAGE_MODE not in {"stochastic", "multidiffusion"}:
    raise ValueError("NEXUS_MULTI_MODE must be 'stochastic' or 'multidiffusion'")

if torch.cuda.is_available():
    torch.set_float32_matmul_precision("high")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True


class Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for stream in self.streams:
            stream.write(data)
            stream.flush()

    def flush(self):
        for stream in self.streams:
            stream.flush()


def safe_slug(value: str) -> str:
    slug = "".join(c if c.isalnum() or c in "-_" else "_" for c in value)
    return slug.strip("_") or "batch"


def setup_logging(label: str) -> Path:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = LOG_DIR / f"{stamp}_generate_{safe_slug(label)}.log"
    log_file = log_path.open("w", encoding="utf-8")
    print(f"Log: {log_path.resolve()}")
    if os.environ.get("NEXUS_LOG_TO_CONSOLE", "0") == "1":
        sys.stdout = Tee(sys.__stdout__, log_file)
        sys.stderr = Tee(sys.__stderr__, log_file)
    else:
        sys.stdout = log_file
        sys.stderr = log_file
    return log_path


def install_kaolin_testing_fallback():
    try:
        from kaolin.utils.testing import check_tensor as _check_tensor
        return
    except (ImportError, OSError):
        for name in tuple(sys.modules):
            if name == "kaolin" or name.startswith("kaolin."):
                del sys.modules[name]

    def check_tensor(tensor, shape=None, dtype=None, device=None, throw=True):
        if shape is not None:
            if len(shape) != tensor.ndim or any(
                dim is not None and tensor.shape[index] != dim
                for index, dim in enumerate(shape)
            ):
                if throw:
                    raise ValueError(f"tensor shape is {tensor.shape}, should be {shape}")
                return False
        if dtype is not None and dtype != tensor.dtype:
            if throw:
                raise TypeError(f"tensor dtype is {tensor.dtype}, should be {dtype}")
            return False
        if device is not None and device != tensor.device.type:
            if throw:
                raise TypeError(f"tensor device is {tensor.device.type}, should be {device}")
            return False
        return True

    kaolin = types.ModuleType("kaolin")
    kaolin.__path__ = []
    utils = types.ModuleType("kaolin.utils")
    utils.__path__ = []
    testing = types.ModuleType("kaolin.utils.testing")
    testing.check_tensor = check_tensor
    utils.testing = testing
    kaolin.utils = utils
    sys.modules["kaolin"] = kaolin
    sys.modules["kaolin.utils"] = utils
    sys.modules["kaolin.utils.testing"] = testing
    print("Kaolin native extension unavailable; using TRELLIS check_tensor fallback")

def load_pipeline():
    """Load TRELLIS 2 once and reuse across the batch (keeps it warm in VRAM)."""
    install_kaolin_testing_fallback()
    from trellis.pipelines import TrellisImageTo3DPipeline
    install_sparse_sdpa_fallback()
    pipe = TrellisImageTo3DPipeline.from_pretrained("microsoft/TRELLIS-image-large")
    pipe.cuda()
    return pipe


def _largest_component(mask: np.ndarray) -> np.ndarray:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    best = []

    for y in range(h):
        for x in range(w):
            if not mask[y, x] or seen[y, x]:
                continue
            stack = [(x, y)]
            seen[y, x] = True
            comp = []
            while stack:
                cx, cy = stack.pop()
                comp.append((cx, cy))
                for nx, ny in ((cx + 1, cy), (cx - 1, cy), (cx, cy + 1), (cx, cy - 1)):
                    if 0 <= nx < w and 0 <= ny < h and mask[ny, nx] and not seen[ny, nx]:
                        seen[ny, nx] = True
                        stack.append((nx, ny))
            if len(comp) > len(best):
                best = comp

    out = np.zeros_like(mask, dtype=bool)
    for x, y in best:
        out[y, x] = True
    return out


def save_validation_screenshot(image: Image.Image, image_path: Path, screenshot_dir: Path | None) -> None:
    if screenshot_dir is None:
        return
    screenshot_dir.mkdir(parents=True, exist_ok=True)
    out = screenshot_dir / f"{safe_slug(image_path.stem)}_source.png"
    image.save(out)
    print(f"  validation screenshot: {out}")


def save_contact_sheet(images: list[Image.Image], labels: list[str], screenshot_dir: Path) -> None:
    if not images:
        return
    thumb_size = 256
    cols = min(4, len(images))
    rows = (len(images) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * thumb_size, rows * thumb_size), (20, 20, 20))
    for i, image in enumerate(images):
        thumb = image.convert("RGB").resize((thumb_size, thumb_size), Image.Resampling.LANCZOS)
        x = (i % cols) * thumb_size
        y = (i // cols) * thumb_size
        sheet.paste(thumb, (x, y))
    screenshot_dir.mkdir(parents=True, exist_ok=True)
    out = screenshot_dir / "contact_sheet.png"
    sheet.save(out)
    print(f"  validation contact sheet: {out}")


def preprocess_subject(pipe, image_path: Path, screenshot_dir: Path | None = None) -> Image.Image:
    if not FOREGROUND_ONLY:
        image = Image.open(image_path).convert("RGB")
        save_validation_screenshot(image, image_path, screenshot_dir)
        return image

    original = Image.open(image_path).convert("RGB")
    max_size = max(original.size)
    scale = min(1, 1024 / max_size)
    work = original
    if scale < 1:
        work = original.resize(
            (int(original.width * scale), int(original.height * scale)),
            Image.Resampling.LANCZOS,
        )

    session = rembg.new_session(REMBG_MODEL)
    cutout = rembg.remove(
        work,
        session=session,
        alpha_matting=True,
        alpha_matting_foreground_threshold=240,
        alpha_matting_background_threshold=10,
        alpha_matting_erode_size=10,
    ).convert("RGBA")

    arr = np.array(cutout)
    mask = arr[:, :, 3] > int(os.environ.get("NEXUS_ALPHA_THRESHOLD", "48"))
    if not mask.any():
        raise ValueError(f"foreground mask is empty for {image_path.name}")

    mask = _largest_component(mask)
    arr[:, :, 3] = (mask.astype(np.uint8) * 255)
    if ASSET_TARGET in {"body_shell", "meshai_car"}:
        arr = preprocess_body_shell(arr, mask)
    elif REMOVE_REFLECTIONS:
        arr = remove_reflection_band(arr, mask)
    cutout = Image.fromarray(arr, "RGBA")

    processed = pipe.preprocess_image(cutout)
    PREPROCESS_DIR.mkdir(parents=True, exist_ok=True)
    preview = PREPROCESS_DIR / f"{image_path.stem}_subject.png"
    processed.save(preview)
    print(f"  subject preview: {preview}")
    save_validation_screenshot(processed, image_path, screenshot_dir)
    return processed


def remove_reflection_band(arr: np.ndarray, mask: np.ndarray) -> np.ndarray:
    ys, _ = np.where(mask)
    y1, y2 = int(ys.min()), int(ys.max())
    h = max(1, y2 - y1)
    bottom_keep = int(y1 + h * float(os.environ.get("NEXUS_REFLECTION_BOTTOM_KEEP", "0.78")))
    bottom_keep = min(y2, max(y1 + 1, bottom_keep))
    arr[bottom_keep + 1:, :, 3] = 0
    print(f"  reflection preprocessing: bottom_keep={bottom_keep}")
    return arr
def preprocess_body_shell(arr: np.ndarray, mask: np.ndarray) -> np.ndarray:
    ys, xs = np.where(mask)
    x1, x2 = int(xs.min()), int(xs.max())
    y1, y2 = int(ys.min()), int(ys.max())
    h = max(1, y2 - y1)
    w = max(1, x2 - x1)

    # Side-reference photos often include a glossy floor reflection. Keep the upper
    # body region and discard the reflection band before TRELLIS sees it.
    defaults = {
        "body_shell": {
            "bottom_keep": "0.62",
            "fill_start": "0.30",
            "dark_fill": "255",
            "sill_start": "0.32",
        },
        "meshai_car": {
            "bottom_keep": "0.72",
            "fill_start": "0.42",
            "dark_fill": "185",
            "sill_start": "0.44",
        },
    }[ASSET_TARGET]

    bottom_keep = int(y1 + h * float(os.environ.get("NEXUS_BODY_BOTTOM_KEEP", defaults["bottom_keep"])))
    bottom_keep = min(y2, max(y1 + 1, bottom_keep))
    arr[bottom_keep + 1:, :, 3] = 0

    work_mask = arr[:, :, 3] > 0
    band_y1 = int(y1 + h * 0.32)
    band_y2 = int(y1 + h * 0.58)
    body_band = work_mask[band_y1:band_y2, x1:x2 + 1]
    body_rgb = arr[band_y1:band_y2, x1:x2 + 1, :3]
    if body_band.any():
        pixels = body_rgb[body_band]
        brightness = pixels.mean(axis=1)
        pixels = pixels[brightness > 35]
        if len(pixels):
            body_color = np.median(pixels, axis=0).astype(np.uint8)
        else:
            body_color = np.array([105, 120, 125], dtype=np.uint8)
    else:
        body_color = np.array([105, 120, 125], dtype=np.uint8)

    lower_y1 = int(y1 + h * float(os.environ.get("NEXUS_BODY_FILL_START", defaults["fill_start"])))
    lower_y2 = bottom_keep
    lower = arr[lower_y1:lower_y2 + 1, x1:x2 + 1]
    lower_mask = lower[:, :, 3] > 0
    brightness = lower[:, :, :3].mean(axis=2)
    dark = lower_mask & (brightness < float(os.environ.get("NEXUS_BODY_DARK_FILL", defaults["dark_fill"])))
    lower_rgb = lower[:, :, :3]
    lower_rgb[dark] = body_color

    if ASSET_TARGET == "body_shell":
        # Fill the lower side silhouette so wheel openings become a continuous shell.
        fill_start = int(y1 + h * float(os.environ.get("NEXUS_BODY_SILL_START", defaults["sill_start"])))
        fill = arr[fill_start:bottom_keep + 1, x1:x2 + 1]
        row_span = max(1, int(w * 0.015))
        for row in range(fill.shape[0]):
            alpha = fill[row, :, 3] > 0
            if not alpha.any():
                continue
            left = max(0, int(np.argmax(alpha)) - row_span)
            right = min(fill.shape[1] - 1, int(len(alpha) - np.argmax(alpha[::-1]) - 1) + row_span)
            fill[row, left:right + 1, :3] = body_color
            fill[row, left:right + 1, 3] = 255

    print(f"  body shell preprocessing: target={ASSET_TARGET}, bottom_keep={bottom_keep}")
    return arr


def _sdpa(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    q = q.transpose(0, 1).unsqueeze(0)
    k = k.transpose(0, 1).unsqueeze(0)
    v = v.transpose(0, 1).unsqueeze(0)
    return F.scaled_dot_product_attention(q, k, v).squeeze(0).transpose(0, 1)


def _sparse_sdpa_attention(*args, **kwargs):
    if kwargs:
        names = {1: ["qkv"], 2: ["q", "kv"], 3: ["q", "k", "v"]}[len(args) + len(kwargs)]
        args = args + tuple(kwargs[name] for name in names[len(args):])

    if len(args) == 1:
        qkv = args[0]
        feats = qkv.feats
        outs = []
        for sl in qkv.layout:
            q, k, v = feats[sl].unbind(dim=1)
            outs.append(_sdpa(q, k, v))
        return qkv.replace(torch.cat(outs, dim=0))

    if len(args) == 2:
        q, kv = args
        if hasattr(q, "feats"):
            outs = []
            for i, sl in enumerate(q.layout):
                q_i = q.feats[sl]
                if hasattr(kv, "feats"):
                    k_i, v_i = kv.feats[kv.layout[i]].unbind(dim=1)
                else:
                    k_i, v_i = kv[i].unbind(dim=1)
                outs.append(_sdpa(q_i, k_i, v_i))
            return q.replace(torch.cat(outs, dim=0))

        outs = []
        for i, sl in enumerate(kv.layout):
            k_i, v_i = kv.feats[sl].unbind(dim=1)
            outs.append(_sdpa(q[i], k_i, v_i))
        return torch.stack(outs, dim=0)

    if len(args) == 3:
        q, k, v = args
        if hasattr(q, "feats"):
            outs = []
            for i, sl in enumerate(q.layout):
                q_i = q.feats[sl]
                if hasattr(k, "feats"):
                    k_i = k.feats[k.layout[i]]
                    v_i = v.feats[v.layout[i]]
                else:
                    k_i = k[i]
                    v_i = v[i]
                outs.append(_sdpa(q_i, k_i, v_i))
            return q.replace(torch.cat(outs, dim=0))

        outs = []
        for i, sl in enumerate(k.layout):
            outs.append(_sdpa(q[i], k.feats[sl], v.feats[v.layout[i]]))
        return torch.stack(outs, dim=0)

    raise ValueError(f"Invalid sparse attention argument count: {len(args)}")


def install_sparse_sdpa_fallback():
    from trellis.modules.sparse.attention import full_attn, modules
    full_attn.sparse_scaled_dot_product_attention = _sparse_sdpa_attention
    modules.sparse_scaled_dot_product_attention = _sparse_sdpa_attention


def generate_one(pipe, image_path: Path, out_path: Path) -> dict:
    screenshot_dir = SCREENSHOT_ROOT / out_path.stem
    image = preprocess_subject(pipe, image_path, screenshot_dir)

    outputs = pipe.run(
        image,
        seed=GEN_PARAMS["seed"],
        sparse_structure_sampler_params={
            "steps": GEN_PARAMS["ss_sampling_steps"],
            "cfg_strength": GEN_PARAMS["ss_guidance_strength"],
        },
        slat_sampler_params={
            "steps": GEN_PARAMS["slat_sampling_steps"],
            "cfg_strength": GEN_PARAMS["slat_guidance_strength"],
        },
        preprocess_image=False,
    )

    glb = to_pipeline_glb(outputs)
    glb.export(str(out_path))

    return {
        "source": image_path.name,
        "output": out_path.name,
        "texture_size": GLB_PARAMS["texture_size"],
        "simplify": GLB_PARAMS["simplify"],
        "screenshots": str(screenshot_dir),
    }


def generate_from_refs(pipe, refs: list[Path], out_path: Path) -> dict:
    screenshot_dir = SCREENSHOT_ROOT / out_path.stem
    images = [preprocess_subject(pipe, ref, screenshot_dir) for ref in refs]
    save_contact_sheet(images, [ref.name for ref in refs], screenshot_dir)

    outputs = pipe.run_multi_image(
        images,
        seed=GEN_PARAMS["seed"],
        sparse_structure_sampler_params={
            "steps": GEN_PARAMS["ss_sampling_steps"],
            "cfg_strength": GEN_PARAMS["ss_guidance_strength"],
        },
        slat_sampler_params={
            "steps": GEN_PARAMS["slat_sampling_steps"],
            "cfg_strength": GEN_PARAMS["slat_guidance_strength"],
        },
        preprocess_image=False,
        mode=MULTI_IMAGE_MODE,
    )

    glb = to_pipeline_glb(outputs)
    glb.export(str(out_path))

    return {
        "sources": [ref.name for ref in refs],
        "output": out_path.name,
        "texture_size": GLB_PARAMS["texture_size"],
        "simplify": GLB_PARAMS["simplify"],
        "multi_image_mode": MULTI_IMAGE_MODE,
        "screenshots": str(screenshot_dir),
    }


def collect_image_refs(folder: Path) -> list[Path]:
    refs = sorted(
        p for ext in IMAGE_EXTENSIONS
        for p in folder.glob(ext)
    )
    return [p for p in refs if p.is_file()]


def to_pipeline_glb(outputs) -> trimesh.Trimesh:
    try:
        from trellis.utils import postprocessing_utils
        return postprocessing_utils.to_glb(
            outputs["gaussian"][0],
            outputs["mesh"][0],
            simplify=GLB_PARAMS["simplify"],
            texture_size=GLB_PARAMS["texture_size"],
        )
    except ModuleNotFoundError as e:
        if e.name != "nvdiffrast":
            raise
        print("nvdiffrast not available; exporting geometry-only GLB.")
        return to_geometry_only_glb(outputs["mesh"][0])


def to_geometry_only_glb(mesh) -> trimesh.Trimesh:
    vertices = mesh.vertices.detach().cpu().numpy()
    faces = mesh.faces.detach().cpu().numpy()
    vertices = vertices @ np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]])
    material = trimesh.visual.material.PBRMaterial(
        roughnessFactor=1.0,
        baseColorFactor=np.array([180, 180, 180, 255], dtype=np.uint8),
    )
    return trimesh.Trimesh(
        vertices=vertices,
        faces=faces,
        visual=trimesh.visual.TextureVisuals(material=material),
        process=False,
    )


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # --- Single-item mode (used by the MCP server) --------------------
    # If NEXUS_SINGLE points at one image, process just that file.
    single = os.environ.get("NEXUS_SINGLE")
    source_dir = os.environ.get("NEXUS_SOURCE_DIR")
    log_label = os.environ.get("NEXUS_ASSET_NAME") or (
        Path(source_dir).name if source_dir else Path(single).stem if single else "batch"
    )
    setup_logging(log_label)
    tex_override = os.environ.get("NEXUS_TEXSIZE")
    if tex_override:
        GLB_PARAMS["texture_size"] = int(tex_override)
    simplify_override = os.environ.get("NEXUS_SIMPLIFY")
    if simplify_override:
        GLB_PARAMS["simplify"] = float(simplify_override)

    if source_dir:
        folder = Path(source_dir)
        if not folder.exists() or not folder.is_dir():
            print(f"NEXUS_SOURCE_DIR not found or not a folder: {folder}")
            return
        refs = collect_image_refs(folder)
        if not refs:
            print(f"No reference images in {folder.resolve()}")
            return
        mesh_id = safe_slug(os.environ.get("NEXUS_ASSET_NAME") or folder.name)
        out = OUTPUT_DIR / f"{mesh_id}.glb"
        pipe = load_pipeline()
        print(
            f"[multi] {len(refs)} refs in {folder} -> {out.name} "
            f"(tex={GLB_PARAMS['texture_size']}, mode={MULTI_IMAGE_MODE}, target={ASSET_TARGET})"
        )
        try:
            entry = generate_from_refs(pipe, refs, out)
            MANIFEST.write_text(json.dumps([entry], indent=2))
            print(f"Done. -> {out}")
            print(f"Manifest: {MANIFEST}")
        except torch.cuda.OutOfMemoryError:
            print(f"OOM on {folder}; clear cache and retry with NEXUS_QUALITY=high or lower NEXUS_TEXSIZE.")
            torch.cuda.empty_cache()
        except BaseException as e:
            print(f"FAILED {folder}: {type(e).__name__}: {e}")
        return

    if single:
        ref = Path(single)
        if not ref.exists():
            print(f"NEXUS_SINGLE not found: {ref}")
            return
        pipe = load_pipeline()
        out = OUTPUT_DIR / f"{ref.stem}.glb"
        print(f"[single] {ref.name} -> {out.name} (tex={GLB_PARAMS['texture_size']})")
        try:
            generate_one(pipe, ref, out)
            print(f"Done. -> {out}")
        except BaseException as e:
            print(f"FAILED {ref.name}: {type(e).__name__}: {e}")
        return

    # --- Batch mode (folder scan) -------------------------------------
    refs = sorted(
        p for ext in IMAGE_EXTENSIONS
        for p in INPUT_DIR.glob(ext)
    )
    if not refs:
        print(f"No reference images in {INPUT_DIR.resolve()}")
        return

    pipe = load_pipeline()
    manifest = []

    for i, ref in enumerate(refs, 1):
        out = OUTPUT_DIR / f"{ref.stem}.glb"
        print(f"[{i}/{len(refs)}] {ref.name} -> {out.name}")
        try:
            entry = generate_one(pipe, ref, out)
            manifest.append(entry)
        except torch.cuda.OutOfMemoryError:
            # 4080 Super shouldn't hit this on props; if it does, drop texture_size.
            print(f"  OOM on {ref.name}; clearing cache and skipping.")
            torch.cuda.empty_cache()
        except Exception as e:
            print(f"  FAILED {ref.name}: {e}")

    MANIFEST.write_text(json.dumps(manifest, indent=2))
    print(f"\nDone. {len(manifest)} meshes -> {OUTPUT_DIR.resolve()}")
    print(f"Manifest: {MANIFEST}")


if __name__ == "__main__":
    main()
