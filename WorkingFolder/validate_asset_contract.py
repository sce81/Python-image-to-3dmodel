"""Generic post-generation contract gate for any image-to-3D asset."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
import trimesh


def merged_mesh(path: Path) -> trimesh.Trimesh:
    loaded = trimesh.load(path, force="scene")
    meshes = [node for node in loaded.geometry.values() if isinstance(node, trimesh.Trimesh)]
    if not meshes:
        raise RuntimeError("No mesh geometry found")
    return trimesh.util.concatenate(meshes)


def texture_max_side(mesh: trimesh.Trimesh) -> int:
    material = getattr(getattr(mesh, "visual", None), "material", None)
    image = getattr(material, "image", None) or getattr(material, "baseColorTexture", None)
    return max(image.size) if image is not None else 0


def _flood_reachable(occupied: np.ndarray) -> np.ndarray:
    """Cells reachable from the grid border without crossing occupied cells."""
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
    return reachable


def projection_hole_ratios(mesh: trimesh.Trimesh, grid: int = 160, samples: int = 300000) -> dict[str, float]:
    """See-through cells enclosed by the silhouette in each axis-aligned projection.

    A window through-cavity is an empty region inside the filled silhouette of at
    least one projection, regardless of the export's up-axis convention. Uniform
    surface sampling keeps coverage dense even where decimation left large flat
    triangles; centroids alone leak flood fill through the gaps.
    """
    sampled = trimesh.sample.sample_surface(mesh, samples)[0]
    points = np.vstack([sampled, mesh.vertices, mesh.triangles_center])
    mins = points.min(axis=0)
    spans = np.maximum(points.max(axis=0) - mins, 1e-9)
    norm = (points - mins) / spans
    ratios = {}
    for name, (a, b) in {"xy": (0, 1), "xz": (0, 2), "yz": (1, 2)}.items():
        occupied = np.zeros((grid, grid), dtype=bool)
        ix = np.clip((norm[:, a] * (grid - 1)).astype(int), 0, grid - 1)
        iy = np.clip((norm[:, b] * (grid - 1)).astype(int), 0, grid - 1)
        occupied[iy, ix] = True
        # One dilation pass closes random-sampling pinholes; real openings survive.
        dilated = occupied.copy()
        dilated[1:, :] |= occupied[:-1, :]
        dilated[:-1, :] |= occupied[1:, :]
        dilated[:, 1:] |= occupied[:, :-1]
        dilated[:, :-1] |= occupied[:, 1:]
        interior_empty = (~dilated) & (~_flood_reachable(dilated))
        denominator = int(dilated.sum()) + int(interior_empty.sum())
        ratios[name] = round(float(interior_empty.sum()) / denominator, 6) if denominator else 0.0
    return ratios


def lower_profile(mesh: trimesh.Trimesh, bins: int = 64) -> tuple[float, list[float]]:
    vertices = mesh.vertices
    x = vertices[:, 0]
    z = vertices[:, 2]
    span = max(float(x.max() - x.min()), 1e-8)
    height = max(float(z.max() - z.min()), 1e-8)
    values = []
    for index in range(bins):
        left = x.min() + span * index / bins
        right = x.min() + span * (index + 1) / bins
        sample = z[(x >= left) & (x <= right)]
        values.append(float(np.quantile(sample, 0.03)) if len(sample) >= 8 else np.nan)
    profile = np.asarray(values, dtype=float)
    valid = np.isfinite(profile)
    if valid.sum() < 8:
        return float("inf"), values
    profile[~valid] = np.interp(np.flatnonzero(~valid), np.flatnonzero(valid), profile[valid])
    smooth = np.convolve(np.pad(profile, 2, mode="edge"), np.ones(5) / 5, mode="valid")
    dip = float(np.max(smooth - profile) / height)
    return dip, profile.round(6).tolist()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("asset", type=Path)
    parser.add_argument("contract", type=Path)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--geometry-only", action="store_true", help="Skip base-color atlas checks for an untextured geometry candidate.")
    args = parser.parse_args()
    contract = json.loads(args.contract.read_text(encoding="utf-8"))
    mesh = merged_mesh(args.asset)
    dip, profile = lower_profile(mesh)
    holes = projection_hole_ratios(mesh)
    worst_hole = max(holes.values())
    atlas = texture_max_side(mesh)
    failures = []
    if len(mesh.faces) < 20000:
        failures.append("mesh density below minimum")
    required_atlas = int(contract.get("required_atlas_px", 0))
    if not args.geometry_only and required_atlas and atlas < required_atlas:
        failures.append(f"base-color atlas {atlas}px below required {required_atlas}px")
    hole_limit = contract.get("max_projection_hole_ratio")
    if hole_limit is not None and worst_hole > float(hole_limit):
        failures.append(
            f"through-cavity: projection hole ratio {worst_hole:.4f} exceeds {float(hole_limit):.4f}"
        )
    report_note = None
    if contract.get("require_continuous_lower_body"):
        limit = float(contract.get("lower_profile_max_dip_ratio", 0.05))
        if dip > limit:
            failures.append(
                f"lower-body discontinuity: profile dip {dip:.4f} exceeds {limit:.4f}"
            )
    report = {"asset": str(args.asset), "contract": contract["asset_id"], "faces": int(len(mesh.faces)), "vertices": int(len(mesh.vertices)), "atlas_max_side": atlas, "lower_profile_dip_ratio": round(dip, 6), "lower_profile": profile, "projection_hole_ratios": holes, "worst_projection_hole_ratio": round(worst_hole, 6), "profile_anomaly": report_note, "failures": failures, "status": "pass" if not failures else "reject"}
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0 if not failures else 2

if __name__ == "__main__":
    raise SystemExit(main())
