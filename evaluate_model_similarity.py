"""
Evaluate generated vehicle GLBs against a reference model.

Run through Blender:
    blender --background --python evaluate_model_similarity.py -- --reference Meshai-EQE-Model.glb --candidate Outputs\\Mercedes_EQE_ue.glb

The score is intentionally geometry-first. It compares normalized side/top/front
occupancy, lengthwise width/height profiles, dimensions, low-base artifacts,
mesh density, and texture resolution. Higher accuracy_score is better.
"""

import argparse
import json
import math
from pathlib import Path

import bpy


VIEW_AXES = {
    "side_xz": (0, 2),
    "top_xy": (0, 1),
    "front_yz": (1, 2),
}


def clean_scene():
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)
    for block in (bpy.data.meshes, bpy.data.materials, bpy.data.images):
        for item in list(block):
            block.remove(item)


def import_model(path: Path):
    before = set(bpy.data.objects)
    suffix = path.suffix.lower()
    if suffix == ".fbx":
        bpy.ops.import_scene.fbx(filepath=str(path))
    elif suffix in {".glb", ".gltf"}:
        bpy.ops.import_scene.gltf(filepath=str(path))
    else:
        raise ValueError(f"Unsupported model format: {path}")
    imported = [obj for obj in bpy.data.objects if obj not in before]
    meshes = [obj for obj in imported if obj.type == "MESH"]
    if not meshes:
        raise ValueError(f"No mesh objects found in {path}")
    bpy.context.view_layer.update()
    return meshes


def mesh_points(meshes):
    points = []
    faces = 0
    verts = 0
    for obj in meshes:
        data = obj.data
        verts += len(data.vertices)
        faces += len(data.polygons)
        world = obj.matrix_world
        local = [world @ vertex.co for vertex in data.vertices]
        for co in local:
            points.append((co.x, co.y, co.z))
        for poly in data.polygons:
            if not poly.vertices:
                continue
            cx = cy = cz = 0.0
            for idx in poly.vertices:
                co = local[idx]
                cx += co.x
                cy += co.y
                cz += co.z
            n = len(poly.vertices)
            points.append((cx / n, cy / n, cz / n))
    if not points:
        raise ValueError("Mesh has no points")
    return points, verts, faces


def bounds(points):
    mins = [min(p[i] for p in points) for i in range(3)]
    maxs = [max(p[i] for p in points) for i in range(3)]
    dims = [maxs[i] - mins[i] for i in range(3)]
    return mins, maxs, dims


def normalize_points(points):
    mins, maxs, dims = bounds(points)
    out = []
    for p in points:
        out.append(tuple((p[i] - mins[i]) / dims[i] if dims[i] > 1e-9 else 0.5 for i in range(3)))
    return out, dims


def occupancy(points, axes, grid=192):
    cells = set()
    ax0, ax1 = axes
    for p in points:
        x = max(0, min(grid - 1, int(p[ax0] * (grid - 1))))
        y = max(0, min(grid - 1, int(p[ax1] * (grid - 1))))
        cells.add((x, y))
    return cells


def iou(a, b):
    if not a and not b:
        return 1.0
    union = len(a | b)
    return len(a & b) / union if union else 0.0


def profile(points, axis=0, bins=64):
    values = []
    for idx in range(bins):
        lo = idx / bins
        hi = (idx + 1) / bins
        band = [p for p in points if lo <= p[axis] < hi or (idx == bins - 1 and p[axis] <= hi)]
        if not band:
            values.append({"width": 0.0, "height": 0.0, "center_z": 0.0})
            continue
        ys = [p[1] for p in band]
        zs = [p[2] for p in band]
        values.append({
            "width": max(ys) - min(ys),
            "height": max(zs) - min(zs),
            "center_z": sum(zs) / len(zs),
        })
    return values


def profile_error(ref, cand):
    total = 0.0
    count = 0
    for a, b in zip(ref, cand):
        for key in ("width", "height", "center_z"):
            total += abs(a[key] - b[key])
            count += 1
    return total / max(1, count)


def texture_stats():
    sizes = []
    for image in bpy.data.images:
        if image.size[0] > 0 and image.size[1] > 0:
            sizes.append([int(image.size[0]), int(image.size[1])])
    max_texel = max([max(size) for size in sizes], default=0)
    return sizes, max_texel


def bottom_artifact(points):
    band = [p for p in points if p[2] <= 0.055]
    if not band:
        return 0.0
    xs = [p[0] for p in band]
    ys = [p[1] for p in band]
    footprint = (max(xs) - min(xs)) * (max(ys) - min(ys))
    density = len(band) / max(1, len(points))
    return footprint * density


def dimension_error(ref_dims, cand_dims):
    return sum(abs(cand_dims[i] - ref_dims[i]) / max(ref_dims[i], 1e-9) for i in range(3)) / 3


def analyze(path: Path):
    clean_scene()
    meshes = import_model(path)
    points, verts, faces = mesh_points(meshes)
    tex_sizes, max_texel = texture_stats()
    norm_points, dims = normalize_points(points)
    profiles = profile(norm_points)
    occ = {name: occupancy(norm_points, axes) for name, axes in VIEW_AXES.items()}
    return {
        "path": str(path),
        "mesh_count": len(meshes),
        "verts": verts,
        "faces": faces,
        "dims": dims,
        "points": len(points),
        "texture_sizes": tex_sizes,
        "max_texture_size": max_texel,
        "profiles": profiles,
        "occupancy": occ,
        "bottom_artifact": bottom_artifact(norm_points),
    }


def compare(reference, candidate):
    occ_iou = {name: iou(reference["occupancy"][name], candidate["occupancy"][name]) for name in VIEW_AXES}
    mean_iou = sum(occ_iou.values()) / len(occ_iou)
    prof_err = profile_error(reference["profiles"], candidate["profiles"])
    dim_err = dimension_error(reference["dims"], candidate["dims"])
    face_ratio = candidate["faces"] / max(reference["faces"], 1)
    face_penalty = abs(math.log(max(face_ratio, 1e-6))) / 2.0
    texture_penalty = 0.0 if candidate["max_texture_size"] >= 4096 else (4096 - candidate["max_texture_size"]) / 4096
    base_delta = max(0.0, candidate["bottom_artifact"] - reference["bottom_artifact"])

    penalty = (
        (1.0 - mean_iou) * 45.0
        + prof_err * 35.0
        + dim_err * 20.0
        + face_penalty * 5.0
        + texture_penalty * 5.0
        + base_delta * 35.0
    )
    score = max(0.0, min(100.0, 100.0 - penalty))
    return {
        "accuracy_score": round(score, 3),
        "projection_iou": {k: round(v, 4) for k, v in occ_iou.items()},
        "mean_projection_iou": round(mean_iou, 4),
        "profile_error": round(prof_err, 5),
        "dimension_error": round(dim_err, 5),
        "face_ratio_vs_reference": round(face_ratio, 4),
        "texture_penalty": round(texture_penalty, 5),
        "bottom_artifact_delta": round(base_delta, 5),
    }


def public_model_summary(model):
    return {
        "path": model["path"],
        "mesh_count": model["mesh_count"],
        "verts": model["verts"],
        "faces": model["faces"],
        "dims": [round(v, 5) for v in model["dims"]],
        "texture_sizes": model["texture_sizes"],
        "max_texture_size": model["max_texture_size"],
        "bottom_artifact": round(model["bottom_artifact"], 5),
    }



def main():
    argv = []
    if "--" in __import__("sys").argv:
        argv = __import__("sys").argv[__import__("sys").argv.index("--") + 1:]
    else:
        argv = __import__("sys").argv[1:]
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--out", default="")
    parsed = parser.parse_args(argv)

    reference = analyze(Path(parsed.reference))
    candidate = analyze(Path(parsed.candidate))
    result = {
        "reference": public_model_summary(reference),
        "candidate": public_model_summary(candidate),
        "comparison": compare(reference, candidate),
    }
    text = json.dumps(result, indent=2)
    print(text)
    if parsed.out:
        out = Path(parsed.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
