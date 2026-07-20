"""
Nexus Protocol ÃƒÂ¢Ã¢â€šÂ¬Ã¢â‚¬Â batch prop/detail generator (TRELLIS, local, RTX 4080 Super)

Stage 1 of the pipeline: image -> raw textured mesh (GLB).
Designed for DETAIL/PROP generation (greebles, terminals, crates, signage),
NOT for base modular tile shells ÃƒÂ¢Ã¢â€šÂ¬Ã¢â‚¬Â those stay procedural/hand-built on your grid.

Run headless in batches. Feeds Stage 2 (blender_postprocess.py) for retopo + grid snap.

Prereqs (one-time, in your venv):
    .\setup_cuda13.ps1
    # then TRELLIS per its repo install (spconv, flash-attn optional, etc.)

VRAM note: 16 GB (4080 Super) runs TRELLIS image-large at full quality. No quantization needed.
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
TRELLIS_DIR = Path(os.environ.get("NEXUS_TRELLIS_DIR", str(Path(__file__).resolve().parent.parent / "TRELLIS")))
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
# Guidance follows the TRELLIS-tuned defaults: ss_guidance 7.5 keeps the sparse
# structure faithful to the image (low values hallucinate cavities/missing parts);
# slat_guidance stays near the model default 3.0 - high slat CFG (7.5+) is the
# artifact knob and produced the rejected artifact-heavy candidates.
QUALITY_PRESETS = {
    "high": {
        "ss_steps": 25,
        "slat_steps": 25,
        "ss_guidance": 7.5,
        "slat_guidance": 3.5,
        "simplify": 0.6,
        "texture_size": 2048,
    },
    "very_high": {
        "ss_steps": 50,
        "slat_steps": 50,
        "ss_guidance": 7.5,
        "slat_guidance": 3.5,
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
TEXTURE_SIZES = {1024, 2048, 4096, 8192}
GLB_PARAMS = {
    "simplify": float(os.environ.get("NEXUS_SIMPLIFY", str(QUALITY_PROFILE["simplify"]))),
    "texture_size": int(os.environ.get("NEXUS_TEXSIZE", str(QUALITY_PROFILE["texture_size"]))),
}
if GLB_PARAMS["texture_size"] not in TEXTURE_SIZES:
    raise ValueError(f"NEXUS_TEXSIZE must be one of: {sorted(TEXTURE_SIZES)}")
TEXTURE_VIEW_RESOLUTION = int(os.environ.get("NEXUS_TEXTURE_VIEW_RES", "1024"))
TEXTURE_VIEWS = int(os.environ.get("NEXUS_TEXTURE_VIEWS", "100"))
TEXTURE_TV_WEIGHT = float(os.environ.get("NEXUS_TEXTURE_TV_WEIGHT", "0.0001"))
GEOMETRY_ONLY = os.environ.get("NEXUS_GEOMETRY_ONLY", "0") == "1"
# Clay conditioning: TRELLIS composites the RGBA cutout onto black, so near-black
# paint/glazing is indistinguishable from empty space and gets carved into
# through-cavities. For geometry-only runs, lift the foreground into a mid-gray
# clay range by default; the texture stage never sees this image.
GEOMETRY_CLAY = os.environ.get("NEXUS_GEOMETRY_CLAY", "1" if GEOMETRY_ONLY else "0") == "1"
CLAY_VALUE_FLOOR = int(os.environ.get("NEXUS_CLAY_VALUE_FLOOR", "110"))
if not 0 <= CLAY_VALUE_FLOOR <= 200:
    raise ValueError("NEXUS_CLAY_VALUE_FLOOR must be between 0 and 200")
# Sample N seeds per asset and keep the mesh with the lowest integrity penalty
# (see mesh_integrity_metrics). Structure sampling is cheap next to the bake.
SEED_CANDIDATES = max(1, int(os.environ.get("NEXUS_SEED_CANDIDATES", "1")))
if not 1024 <= TEXTURE_VIEW_RESOLUTION <= 2048:
    raise ValueError("NEXUS_TEXTURE_VIEW_RES must be between 1024 and 2048")
if TEXTURE_VIEWS < 100:
    raise ValueError("NEXUS_TEXTURE_VIEWS must be at least 100")
if not 0.0 <= TEXTURE_TV_WEIGHT <= 0.01:
    raise ValueError("NEXUS_TEXTURE_TV_WEIGHT must be between 0.0 and 0.01")
FOREGROUND_ONLY = os.environ.get("NEXUS_FOREGROUND_ONLY", "1") != "0"
# isnet-general-use segments hard-surface edges far better than u2net.
REMBG_MODEL = os.environ.get("NEXUS_REMBG_MODEL", "isnet-general-use")
REMBG_ERODE = int(os.environ.get("NEXUS_REMBG_ERODE", "5"))
# Approved rasters that already carry a real alpha channel skip rembg entirely;
# re-masking a hand-approved cutout can only degrade the silhouette.
SKIP_REMBG_WITH_ALPHA = os.environ.get("NEXUS_SKIP_REMBG_WITH_ALPHA", "1") != "0"
REMOVE_REFLECTIONS = os.environ.get("NEXUS_REMOVE_REFLECTIONS", "0") == "1"
SUPPRESS_WINDOW_REFLECTIONS = os.environ.get("NEXUS_SUPPRESS_WINDOW_REFLECTIONS", "0") == "1"
FLATTEN_WINDOW_GEOMETRY = os.environ.get("NEXUS_FLATTEN_WINDOW_GEOMETRY", "0") == "1"
ASSET_TARGET = os.environ.get("NEXUS_ASSET_TARGET", "prop").lower()
if ASSET_TARGET not in {"prop", "body_shell", "meshai_car"}:
    raise ValueError("NEXUS_ASSET_TARGET must be 'prop', 'body_shell', or 'meshai_car'")
MULTI_IMAGE_MODE = os.environ.get("NEXUS_MULTI_MODE", "multidiffusion").lower()
if MULTI_IMAGE_MODE not in {"stochastic", "multidiffusion"}:
    raise ValueError("NEXUS_MULTI_MODE must be 'stochastic' or 'multidiffusion'")
# Installed checkout is original TRELLIS (1); image-large is the matching weights.
# If TRELLIS 2 is ever installed, point this at its weights via NEXUS_TRELLIS_MODEL.
TRELLIS_MODEL_ID = os.environ.get("NEXUS_TRELLIS_MODEL", "microsoft/TRELLIS-image-large")
# The serialized SDPA monkeypatch replaces TRELLIS sparse attention on every run.
# Keep default 1 (current validated behavior); set 0 to A/B against the native
# xformers sparse path and confirm numerical equivalence.
SPARSE_SDPA_FALLBACK = os.environ.get("NEXUS_SPARSE_SDPA_FALLBACK", "1") != "0"

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
    """Load the TRELLIS pipeline once and reuse across the batch (keeps it warm in VRAM)."""
    install_kaolin_testing_fallback()
    from trellis.pipelines import TrellisImageTo3DPipeline
    if SPARSE_SDPA_FALLBACK:
        install_sparse_sdpa_fallback()
    pipe = TrellisImageTo3DPipeline.from_pretrained(TRELLIS_MODEL_ID)
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
        # Still normalize through the TRELLIS preprocessor (crop/centre/resize);
        # skipping it conditioned the model on raw full-frame imagery.
        image = Image.open(image_path).convert("RGB")
        processed = pipe.preprocess_image(image)
        save_validation_screenshot(processed, image_path, screenshot_dir)
        return processed

    original = Image.open(image_path)
    alpha_extrema = original.getchannel("A").getextrema() if original.mode == "RGBA" else None
    has_real_alpha = alpha_extrema is not None and alpha_extrema[0] < 250
    max_size = max(original.size)
    scale = min(1, 1024 / max_size)
    work = original
    if scale < 1:
        work = original.resize(
            (int(original.width * scale), int(original.height * scale)),
            Image.Resampling.LANCZOS,
        )

    if has_real_alpha and SKIP_REMBG_WITH_ALPHA:
        cutout = work.convert("RGBA")
        print("  foreground mask: source alpha channel (rembg skipped)")
    else:
        session = rembg.new_session(REMBG_MODEL)
        cutout = rembg.remove(
            work.convert("RGB"),
            session=session,
            alpha_matting=True,
            alpha_matting_foreground_threshold=240,
            alpha_matting_background_threshold=10,
            alpha_matting_erode_size=REMBG_ERODE,
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
    if SUPPRESS_WINDOW_REFLECTIONS:
        arr = suppress_window_reflections(arr)
    if FLATTEN_WINDOW_GEOMETRY:
        arr = flatten_window_geometry(arr)
    if GEOMETRY_CLAY:
        arr = clay_conditioning(arr)
    cutout = Image.fromarray(arr, "RGBA")

    processed = pipe.preprocess_image(cutout)
    PREPROCESS_DIR.mkdir(parents=True, exist_ok=True)
    preview = PREPROCESS_DIR / f"{image_path.stem}_subject.png"
    processed.save(preview)
    print(f"  subject preview: {preview}")
    save_validation_screenshot(processed, image_path, screenshot_dir)
    return processed


def clay_conditioning(arr: np.ndarray) -> np.ndarray:
    """Lift foreground value into a mid-gray clay range for geometry conditioning.

    TRELLIS multiplies RGB by alpha (composite onto black), so near-black pixels
    carry no shape signal and read as empty voxels. A linear value lift maps
    0 -> CLAY_VALUE_FLOOR and 255 -> 255, turning dark paint and glazing opaque
    while preserving shading gradients. Desaturated: geometry does not need hue.
    """
    mask = arr[:, :, 3] > 0
    if not mask.any():
        return arr
    luminance = arr[:, :, :3].astype(np.float32).mean(axis=2)
    lifted = CLAY_VALUE_FLOOR + luminance * ((255.0 - CLAY_VALUE_FLOOR) / 255.0)
    gray = np.clip(lifted, 0, 255).astype(np.uint8)
    for channel in range(3):
        arr[:, :, channel][mask] = gray[mask]
    print(f"  clay conditioning: value floor {CLAY_VALUE_FLOOR}, {int(mask.sum())} px lifted")
    return arr


def _projection_hole_ratio(occupied: np.ndarray) -> float:
    """Fraction of see-through cells enclosed by the silhouette of a 2D occupancy grid.

    One dilation pass closes sampling pinholes so only real openings register.
    """
    dilated = occupied.copy()
    dilated[1:, :] |= occupied[:-1, :]
    dilated[:-1, :] |= occupied[1:, :]
    dilated[:, 1:] |= occupied[:, :-1]
    dilated[:, :-1] |= occupied[:, 1:]
    occupied = dilated
    reachable = np.zeros_like(occupied, dtype=bool)
    frontier = np.zeros_like(occupied, dtype=bool)
    frontier[0, :] = ~occupied[0, :]
    frontier[-1, :] |= ~occupied[-1, :]
    frontier[:, 0] |= ~occupied[:, 0]
    frontier[:, -1] |= ~occupied[:, -1]
    while frontier.any():
        reachable |= frontier
        grow = np.zeros_like(frontier)
        grow[1:, :] |= frontier[:-1, :]
        grow[:-1, :] |= frontier[1:, :]
        grow[:, 1:] |= frontier[:, :-1]
        grow[:, :-1] |= frontier[:, 1:]
        frontier = grow & ~occupied & ~reachable
    interior_empty = (~occupied) & (~reachable)
    denominator = int(occupied.sum()) + int(interior_empty.sum())
    return float(interior_empty.sum()) / denominator if denominator else 0.0


INTEGRITY_GRID = 128


def mesh_integrity_metrics(mesh) -> dict:
    """Score structural hallucination on the raw TRELLIS mesh.

    Projection hole ratio: enclosed empty regions in the axis-aligned silhouettes
    (a window through-cavity is a see-through hole in at least one projection).
    Boundary edge ratio: open surface borders. Relative ranker, not an absolute gate.
    """
    vertices = mesh.vertices.detach().cpu().numpy()
    faces = mesh.faces.detach().cpu().numpy()
    edges = np.sort(faces[:, [0, 1, 1, 2, 2, 0]].reshape(-1, 2), axis=1)
    _, counts = np.unique(edges, axis=0, return_counts=True)
    boundary_ratio = float((counts == 1).sum()) / max(1, len(counts))
    centroids = vertices[faces].mean(axis=1)
    # Vertices + edge midpoints + centroids: dense coverage even on large faces.
    edge_midpoints = (vertices[faces[:, [0, 1, 2]]] + vertices[faces[:, [1, 2, 0]]]).reshape(-1, 3) / 2.0
    points = np.vstack([vertices, centroids, edge_midpoints])
    mins = points.min(axis=0)
    spans = np.maximum(points.max(axis=0) - mins, 1e-9)
    norm = (points - mins) / spans
    holes = {}
    for name, (a, b) in {"xy": (0, 1), "xz": (0, 2), "yz": (1, 2)}.items():
        grid = np.zeros((INTEGRITY_GRID, INTEGRITY_GRID), dtype=bool)
        ix = np.clip((norm[:, a] * (INTEGRITY_GRID - 1)).astype(int), 0, INTEGRITY_GRID - 1)
        iy = np.clip((norm[:, b] * (INTEGRITY_GRID - 1)).astype(int), 0, INTEGRITY_GRID - 1)
        grid[iy, ix] = True
        holes[name] = round(_projection_hole_ratio(grid), 5)
    worst = max(holes.values())
    penalty = worst * 100.0 + boundary_ratio * 10.0
    return {
        "projection_holes": holes,
        "worst_projection_hole_ratio": worst,
        "boundary_edge_ratio": round(boundary_ratio, 5),
        "integrity_penalty": round(penalty, 5),
    }


def sample_best_candidate(run_sampler) -> tuple[dict, dict]:
    """Run SEED_CANDIDATES seeds through the sampler and keep the cleanest mesh."""
    best = None
    reports = []
    for offset in range(SEED_CANDIDATES):
        seed = GEN_PARAMS["seed"] + offset
        outputs = run_sampler(seed)
        metrics = mesh_integrity_metrics(outputs["mesh"][0])
        report = {"seed": seed, **metrics}
        reports.append(report)
        print(
            f"  candidate seed={seed}: penalty={metrics['integrity_penalty']} "
            f"holes={metrics['projection_holes']} boundary={metrics['boundary_edge_ratio']}"
        )
        if best is None or metrics["integrity_penalty"] < best[1]["integrity_penalty"]:
            best = (outputs, report)
        else:
            del outputs
        torch.cuda.empty_cache()
    print(f"  selected seed {best[1]['seed']} of {SEED_CANDIDATES} candidate(s)")
    selection = {"selected_seed": best[1]["seed"], "seed_candidates": reports}
    return best[0], selection


def remove_reflection_band(arr: np.ndarray, mask: np.ndarray) -> np.ndarray:
    ys, _ = np.where(mask)
    y1, y2 = int(ys.min()), int(ys.max())
    h = max(1, y2 - y1)
    bottom_keep = int(y1 + h * float(os.environ.get("NEXUS_REFLECTION_BOTTOM_KEEP", "0.78")))
    bottom_keep = min(y2, max(y1 + 1, bottom_keep))
    arr[bottom_keep + 1:, :, 3] = 0
    print(f"  reflection preprocessing: bottom_keep={bottom_keep}")
    return arr


def flatten_window_geometry(arr: np.ndarray) -> np.ndarray:
    """Use uniform dark glazing for geometry conditioning; retain detail for the texture stage."""
    alpha = arr[:, :, 3] > 0
    ys, xs = np.where(alpha)
    if not len(xs):
        return arr
    x1, x2 = int(xs.min()), int(xs.max())
    y1, y2 = int(ys.min()), int(ys.max())
    h, w = max(1, y2 - y1), max(1, x2 - x1)
    left, right = x1 + int(w * 0.06), x1 + int(w * 0.72)
    top, bottom = y1 + int(h * 0.10), y1 + int(h * 0.52)
    region = arr[top:bottom + 1, left:right + 1]
    active = region[:, :, 3] > 0
    luminance = region[:, :, :3].mean(axis=2)
    dark = active & (luminance < 95)
    colour = np.median(region[:, :, :3][dark], axis=0).astype(np.uint8) if dark.any() else np.array([22, 26, 30], dtype=np.uint8)
    region[:, :, :3][active] = colour
    print(f"  flattened window geometry band: {int(active.sum())} pixels")
    return arr
def suppress_window_reflections(arr: np.ndarray) -> np.ndarray:
    """Suppress bright mirror-like details inside the side-window geometry band."""
    alpha = arr[:, :, 3] > 0
    ys, xs = np.where(alpha)
    if not len(xs):
        return arr
    x1, x2 = int(xs.min()), int(xs.max())
    y1, y2 = int(ys.min()), int(ys.max())
    h, w = max(1, y2 - y1), max(1, x2 - x1)
    left, right = x1 + int(w * 0.06), x1 + int(w * 0.72)
    top, bottom = y1 + int(h * 0.10), y1 + int(h * 0.52)
    region = arr[top:bottom + 1, left:right + 1]
    region_alpha = region[:, :, 3] > 0
    luminance = region[:, :, :3].mean(axis=2)
    dark = region_alpha & (luminance < 95)
    if not dark.any():
        return arr
    window_colour = np.median(region[:, :, :3][dark], axis=0).astype(np.uint8)
    reflection = region_alpha & (luminance > float(os.environ.get("NEXUS_WINDOW_REFLECTION_THRESHOLD", "165")))
    region[:, :, :3][reflection] = window_colour
    print(f"  window reflection suppression: {int(reflection.sum())} pixels")
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

    def run_sampler(seed: int):
        return pipe.run(
            image,
            seed=seed,
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

    outputs, selection = sample_best_candidate(run_sampler)

    glb = to_geometry_only_glb(outputs["mesh"][0]) if GEOMETRY_ONLY else to_pipeline_glb(outputs)
    glb.export(str(out_path))
    texture_audit = {"status": "deferred_geometry_only"}
    if not GEOMETRY_ONLY:
        try:
            texture_audit = audit_baked_texture(glb, image_path)
        except Exception:
            out_path.unlink(missing_ok=True)
            raise

    return {
        "source": image_path.name,
        "output": out_path.name,
        "texture_size": GLB_PARAMS["texture_size"],
        "simplify": GLB_PARAMS["simplify"],
        "texture_audit": texture_audit,
        "texture_review_required": not GEOMETRY_ONLY,
        "screenshots": str(screenshot_dir),
        **selection,
    }


def generate_from_refs(pipe, refs: list[Path], out_path: Path) -> dict:
    screenshot_dir = SCREENSHOT_ROOT / out_path.stem
    images = [preprocess_subject(pipe, ref, screenshot_dir) for ref in refs]
    save_contact_sheet(images, [ref.name for ref in refs], screenshot_dir)

    def run_sampler(seed: int):
        return pipe.run_multi_image(
            images,
            seed=seed,
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

    outputs, selection = sample_best_candidate(run_sampler)

    glb = to_geometry_only_glb(outputs["mesh"][0]) if GEOMETRY_ONLY else to_pipeline_glb(outputs)
    glb.export(str(out_path))
    texture_audit = {"status": "deferred_geometry_only"}
    if not GEOMETRY_ONLY:
        try:
            texture_audit = audit_baked_texture(glb, refs[0])
        except Exception:
            out_path.unlink(missing_ok=True)
            raise

    return {
        "sources": [ref.name for ref in refs],
        "output": out_path.name,
        "texture_size": GLB_PARAMS["texture_size"],
        "simplify": GLB_PARAMS["simplify"],
        "texture_audit": texture_audit,
        "texture_review_required": not GEOMETRY_ONLY,
        "multi_image_mode": MULTI_IMAGE_MODE,
        "screenshots": str(screenshot_dir),
        **selection,
    }


def collect_image_refs(folder: Path) -> list[Path]:
    refs = sorted(
        p for ext in IMAGE_EXTENSIONS
        for p in folder.glob(ext)
    )
    return [p for p in refs if p.is_file()]


def _detail_energy(image: Image.Image) -> float:
    """Use a stable high-frequency metric to flag clearly blurred texture bakes."""
    gray = np.asarray(image.convert("L"), dtype=np.float32)
    if min(gray.shape) > 2048:
        scale = 2048 / min(gray.shape)
        gray = np.asarray(image.convert("L").resize((round(image.width * scale), round(image.height * scale)), Image.Resampling.LANCZOS), dtype=np.float32)
    laplacian = -4.0 * gray + np.roll(gray, 1, axis=0) + np.roll(gray, -1, axis=0) + np.roll(gray, 1, axis=1) + np.roll(gray, -1, axis=1)
    return float(np.var(laplacian[1:-1, 1:-1]))


def _texture_images(asset) -> list[Image.Image]:
    geometries = asset.geometry.values() if isinstance(asset, trimesh.Scene) else [asset]
    images = []
    for geometry in geometries:
        material = getattr(getattr(geometry, "visual", None), "material", None)
        image = getattr(material, "image", None) or getattr(material, "baseColorTexture", None)
        if isinstance(image, Image.Image):
            images.append(image)
    return images


def audit_baked_texture(asset, source_path: Path) -> dict:
    """Reject missing, undersized, or insufficiently sharper candidate bakes."""
    images = _texture_images(asset)
    if not images:
        raise RuntimeError("Texture audit failed: the generated asset contains no base-color image.")
    max_side = max(max(image.size) for image in images)
    if max_side < GLB_PARAMS["texture_size"]:
        raise RuntimeError(f"Texture audit failed: generated atlas is {max_side}px, below requested {GLB_PARAMS['texture_size']}px.")
    source_energy = _detail_energy(Image.open(source_path).convert("RGB"))
    bake_energy = max(_detail_energy(image) for image in images)
    detail_ratio = bake_energy / max(source_energy, 1e-6)
    minimum_ratio = float(os.environ.get("NEXUS_TEXTURE_MIN_DETAIL_RATIO", "0.015"))
    if detail_ratio < minimum_ratio:
        raise RuntimeError(f"Texture audit failed: bake detail energy is below the configured minimum ({detail_ratio:.4f} < {minimum_ratio:.4f}).")
    baseline_path = os.environ.get("NEXUS_TEXTURE_BASELINE")
    baseline_multiplier = None
    if baseline_path:
        baseline = trimesh.load(baseline_path, force="scene")
        baseline_images = _texture_images(baseline)
        if not baseline_images:
            raise RuntimeError("Texture audit failed: configured baseline has no base-color image.")
        baseline_energy = max(_detail_energy(image) for image in baseline_images)
        baseline_multiplier = bake_energy / max(baseline_energy, 1e-6)
        required_multiplier = float(os.environ.get("NEXUS_TEXTURE_TARGET_MULTIPLIER", "3.0"))
        if baseline_multiplier < required_multiplier:
            raise RuntimeError(f"Texture audit failed: candidate detail is below the required baseline multiplier ({baseline_multiplier:.4f} < {required_multiplier:.4f}).")
    return {"atlas_count": len(images), "atlas_max_side": max_side, "source_detail_energy": round(source_energy, 4), "bake_detail_energy": round(bake_energy, 4), "detail_ratio": round(detail_ratio, 4), "baseline_detail_multiplier": round(baseline_multiplier, 4) if baseline_multiplier else None, "status": "visual_artifact_review_required"}

def to_pipeline_glb(outputs) -> trimesh.Trimesh:
    try:
        from trellis.utils import postprocessing_utils
        original_render = postprocessing_utils.render_multiview
        original_bake = postprocessing_utils.bake_texture

        def detail_render(app_rep, resolution=1024, nviews=100):
            return original_render(app_rep, resolution=TEXTURE_VIEW_RESOLUTION, nviews=TEXTURE_VIEWS)

        def detail_bake(*args, **kwargs):
            kwargs["lambda_tv"] = TEXTURE_TV_WEIGHT
            return original_bake(*args, **kwargs)

        postprocessing_utils.render_multiview = detail_render
        postprocessing_utils.bake_texture = detail_bake
        try:
            return postprocessing_utils.to_glb(
                outputs["gaussian"][0],
                outputs["mesh"][0],
                simplify=GLB_PARAMS["simplify"],
                texture_size=GLB_PARAMS["texture_size"],
            )
        finally:
            postprocessing_utils.render_multiview = original_render
            postprocessing_utils.bake_texture = original_bake
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
        mesh_id = safe_slug(os.environ.get("NEXUS_ASSET_NAME") or ref.stem)
        out = OUTPUT_DIR / f"{mesh_id}.glb"
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
