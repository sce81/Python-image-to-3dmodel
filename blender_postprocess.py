"""
Nexus Protocol — Stage 2: Blender background post-processor

Takes raw TRELLIS GLBs (dense, organic, arbitrary scale) and makes them
engine-ready for UE5.7: retopo to a sane polycount, snap to your cm grid,
recentre origin, orient +Z up, re-export clean GLB.

Run headless — no Blender UI needed:
    blender --background --python blender_postprocess.py

Requires Blender 5.1.x (you're already on it).

DESIGN NOTE: this is tuned for PROPS/DETAILS, not base tile shells.
Base modular tiles (75x25x200 walls, 75x75x200 corners) should be authored
procedurally on-grid; use this pipeline for the set-dressing that hangs off them.
"""

import bpy
import os
import glob
import math
import sys
import bmesh
from datetime import datetime
from pathlib import Path

# --- Config (edit to taste) ---------------------------------------------
IN_DIR       = Path("./raw_meshes")
OUT_DIR      = Path("./Outputs")
LOG_DIR      = Path("./Logs")
QUALITY      = os.environ.get("NEXUS_QUALITY", "high").lower()
TRIS_BY_QUALITY = {
    "high": 50000,
    "very_high": 100000,
}
if QUALITY not in TRIS_BY_QUALITY:
    raise ValueError(f"NEXUS_QUALITY must be one of: {', '.join(TRIS_BY_QUALITY)}")
GRID_CM      = float(os.environ.get("NEXUS_GRID", "75.0"))   # master grid unit
TARGET_TRIS  = int(os.environ.get("NEXUS_TRIS", str(TRIS_BY_QUALITY[QUALITY])))  # per-prop poly budget
RETOPO_MODE  = os.environ.get("NEXUS_RETOPO", "decimate").lower()
ASSET_TARGET = os.environ.get("NEXUS_ASSET_TARGET", "prop").lower()
SYMMETRY = os.environ.get("NEXUS_SYMMETRY", "none").lower()
SYMMETRY_SOURCE = os.environ.get("NEXUS_SYMMETRY_SOURCE", "positive_x").lower()
LENGTH_AXIS = os.environ.get("NEXUS_LENGTH_AXIS", "x").lower()
SMOOTH_NORMALS = os.environ.get("NEXUS_SMOOTH_NORMALS", "1") != "0"
REFERENCE_MODEL = os.environ.get("NEXUS_REFERENCE_MODEL", "")
MATCH_REFERENCE_DIMS = os.environ.get("NEXUS_MATCH_REFERENCE_DIMS", "0") == "1"
REMOVE_PRESENTATION_BASE = os.environ.get("NEXUS_REMOVE_PRESENTATION_BASE", "0") == "1"
REMOVE_MICRO_ISLANDS = os.environ.get("NEXUS_REMOVE_MICRO_ISLANDS", "1") != "0"
MICRO_ISLAND_MAX_FACES = int(os.environ.get("NEXUS_MICRO_ISLAND_MAX_FACES", "32"))
MICRO_ISLAND_MAX_EXTENT_RATIO = float(os.environ.get("NEXUS_MICRO_ISLAND_MAX_EXTENT_RATIO", "0.01"))
UP_AXIS_FIX  = True          # rotate so +Z is up (TRELLIS/GLB often Y-up)
SINGLE_MESH  = os.environ.get("NEXUS_MESH")  # if set, process only this mesh_id


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
    log_path = LOG_DIR / f"{stamp}_blender_{safe_slug(label)}.log"
    log_file = log_path.open("w", encoding="utf-8")
    print(f"Log: {log_path.resolve()}")
    if os.environ.get("NEXUS_LOG_TO_CONSOLE", "0") == "1":
        sys.stdout = Tee(sys.__stdout__, log_file)
        sys.stderr = Tee(sys.__stderr__, log_file)
    else:
        sys.stdout = log_file
        sys.stderr = log_file
    return log_path


def clean_scene():
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)
    for block in (bpy.data.meshes, bpy.data.materials, bpy.data.images):
        for item in list(block):
            block.remove(item)


def import_glb(path: Path):
    bpy.ops.import_scene.gltf(filepath=str(path))
    objs = [o for o in bpy.context.selected_objects if o.type == "MESH"]
    return objs


def join_meshes(objs):
    if len(objs) <= 1:
        return objs[0] if objs else None
    bpy.ops.object.select_all(action="DESELECT")
    for o in objs:
        o.select_set(True)
    bpy.context.view_layer.objects.active = objs[0]
    bpy.ops.object.join()
    return bpy.context.view_layer.objects.active


def remove_micro_islands(obj):
    """Remove only extraction debris, never meaningful disconnected asset parts."""
    if not REMOVE_MICRO_ISLANDS:
        print("  micro-island cleanup disabled")
        return

    bm = bmesh.new()
    bm.from_mesh(obj.data)
    remaining = set(bm.verts)
    components = []
    while remaining:
        seed = remaining.pop()
        stack = [seed]
        verts = {seed}
        while stack:
            vertex = stack.pop()
            for edge in vertex.link_edges:
                other = edge.other_vert(vertex)
                if other not in verts:
                    verts.add(other)
                    remaining.discard(other)
                    stack.append(other)
        faces = {face for vertex in verts for face in vertex.link_faces}
        minimum = [min(vertex.co[axis] for vertex in verts) for axis in range(3)]
        maximum = [max(vertex.co[axis] for vertex in verts) for axis in range(3)]
        extent = max(maximum[axis] - minimum[axis] for axis in range(3))
        components.append((verts, faces, extent))

    primary_extent = max((component[2] for component in components), default=0.0)
    threshold = primary_extent * MICRO_ISLAND_MAX_EXTENT_RATIO
    debris = [
        component for component in components
        if len(component[1]) <= MICRO_ISLAND_MAX_FACES and component[2] <= threshold
    ]
    if not debris:
        bm.free()
        print("  micro-island cleanup: none found")
        return

    for verts, _faces, _extent in debris:
        bmesh.ops.delete(bm, geom=list(verts), context="VERTS")
    bm.to_mesh(obj.data)
    bm.free()
    obj.data.update()
    print(
        f"  removed {len(debris)} micro-island(s) "
        f"(<= {MICRO_ISLAND_MAX_FACES} faces and <= "
        f"{MICRO_ISLAND_MAX_EXTENT_RATIO:.1%} primary extent)"
    )

def _apply_triangulate(obj):
    mod = obj.modifiers.new("triangulate_for_export", "TRIANGULATE")
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.modifier_apply(modifier=mod.name)


def retopo(obj, target_tris):
    """Reduce only when needed, then triangulate for engine import."""
    bpy.context.view_layer.objects.active = obj
    source_faces = len(obj.data.polygons)

    if RETOPO_MODE == "none":
        print(f"  retopo skipped ({source_faces} faces)")
        _apply_triangulate(obj)
        return

    if RETOPO_MODE == "quadriflow":
        # Quadriflow gives cleaner quads, but it can wash out generated hard-surface detail.
        try:
            bpy.ops.object.quadriflow_remesh(
                target_faces=max(200, target_tris // 2),
                use_preserve_sharp=True,
                use_preserve_boundary=True,
            )
        except Exception as e:
            print(f"  Quadriflow failed ({e}); falling back to Decimate modifier.")
            mod = obj.modifiers.new("dec", "DECIMATE")
            mod.ratio = min(1.0, target_tris / max(1, source_faces))
            bpy.ops.object.modifier_apply(modifier=mod.name)
        _apply_triangulate(obj)
        return

    if source_faces > target_tris:
        mod = obj.modifiers.new("dec", "DECIMATE")
        mod.ratio = min(1.0, target_tris / max(1, source_faces))
        bpy.ops.object.modifier_apply(modifier=mod.name)
        print(f"  decimated {source_faces} -> {len(obj.data.polygons)} faces")
    else:
        print(f"  decimate skipped ({source_faces} faces <= {target_tris} target)")

    _apply_triangulate(obj)


def orient_long_axis(obj):
    if LENGTH_AXIS != "x":
        return
    bpy.context.view_layer.update()
    dims = obj.dimensions
    if dims.y > dims.x:
        for vertex in obj.data.vertices:
            x = vertex.co.x
            vertex.co.x = vertex.co.y
            vertex.co.y = -x
        obj.data.update()
        bpy.context.view_layer.update()
        print("  remapped mesh coordinates so long axis is X")


def object_bounds(objs):
    mins = [float("inf"), float("inf"), float("inf")]
    maxs = [float("-inf"), float("-inf"), float("-inf")]
    for obj in objs:
        for vertex in obj.data.vertices:
            co = obj.matrix_world @ vertex.co
            mins[0] = min(mins[0], co.x)
            mins[1] = min(mins[1], co.y)
            mins[2] = min(mins[2], co.z)
            maxs[0] = max(maxs[0], co.x)
            maxs[1] = max(maxs[1], co.y)
            maxs[2] = max(maxs[2], co.z)
    return mins, maxs, [maxs[i] - mins[i] for i in range(3)]


def reference_dimensions(path: str):
    ref = Path(path)
    if not ref.exists():
        print(f"  reference model not found: {ref}")
        return None

    before = set(bpy.data.objects)
    if ref.suffix.lower() == ".fbx":
        bpy.ops.import_scene.fbx(filepath=str(ref))
    else:
        bpy.ops.import_scene.gltf(filepath=str(ref))
    imported = [obj for obj in bpy.data.objects if obj not in before]
    meshes = [obj for obj in imported if obj.type == "MESH"]
    if not meshes:
        for obj in imported:
            bpy.data.objects.remove(obj, do_unlink=True)
        return None
    _, _, dims = object_bounds(meshes)
    for obj in imported:
        bpy.data.objects.remove(obj, do_unlink=True)
    return dims


def match_reference_dimensions(obj):
    if not MATCH_REFERENCE_DIMS or not REFERENCE_MODEL:
        return
    target = reference_dimensions(REFERENCE_MODEL)
    if not target:
        return

    _, _, dims = object_bounds([obj])
    scales = [target[i] / dims[i] if dims[i] > 1e-6 else 1.0 for i in range(3)]
    center = [
        sum((obj.matrix_world @ v.co)[i] for v in obj.data.vertices) / max(1, len(obj.data.vertices))
        for i in range(3)
    ]
    for vertex in obj.data.vertices:
        vertex.co.x = center[0] + (vertex.co.x - center[0]) * scales[0]
        vertex.co.y = center[1] + (vertex.co.y - center[1]) * scales[1]
        vertex.co.z = center[2] + (vertex.co.z - center[2]) * scales[2]
    obj.data.update()
    print(f"  matched reference dims: {target[0]:.3f} x {target[1]:.3f} x {target[2]:.3f}")


def smooth_normals(obj):
    if not SMOOTH_NORMALS:
        return
    for poly in obj.data.polygons:
        poly.use_smooth = True
    mod = obj.modifiers.new("weighted_normals", "WEIGHTED_NORMAL")
    mod.keep_sharp = True
    mod.weight = 50
    bpy.ops.object.select_all(action="DESELECT")
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.modifier_apply(modifier=mod.name)
    print("  applied smooth shading + weighted normals")


def remove_generated_base(obj):
    # Low faces are often the vehicle underbody. Never remove them merely by height.
    # Presentation-base removal is opt-in and must be paired with visual validation.
    if ASSET_TARGET not in {"body_shell", "meshai_car"} or not REMOVE_PRESENTATION_BASE:
        print("  presentation base removal disabled; preserving lower-body geometry")
        return

    mesh = obj.data
    bm = bmesh.new()
    bm.from_mesh(mesh)
    bm.faces.ensure_lookup_table()
    bm.verts.ensure_lookup_table()
    if not bm.verts:
        bm.free()
        return

    min_z = min(v.co.z for v in bm.verts)
    max_z = max(v.co.z for v in bm.verts)
    height = max(1e-6, max_z - min_z)
    cutoff = min_z + height * float(os.environ.get("NEXUS_BASE_CUT_HEIGHT", "0.045"))

    remove_faces = []
    for face in bm.faces:
        if all(v.co.z <= cutoff for v in face.verts):
            remove_faces.append(face)

    if remove_faces:
        bmesh.ops.delete(bm, geom=remove_faces, context="FACES")
        print(f"  removed generated base faces: {len(remove_faces)}")
    else:
        print("  no generated base faces removed")

    bm.to_mesh(mesh)
    bm.free()
    mesh.update()



def symmetrize_width(obj):
    if SYMMETRY != "mirror_x":
        return
    if SYMMETRY_SOURCE not in {"positive_x", "negative_x"}:
        raise ValueError("NEXUS_SYMMETRY_SOURCE must be 'positive_x' or 'negative_x'")

    mesh = obj.data
    if not mesh.vertices:
        return

    min_x = min(v.co.x for v in mesh.vertices)
    max_x = max(v.co.x for v in mesh.vertices)
    center_x = (min_x + max_x) / 2

    for v in mesh.vertices:
        v.co.x -= center_x
    obj.location.x += center_x
    mesh.update()

    bm = bmesh.new()
    bm.from_mesh(mesh)
    geom = list(bm.verts) + list(bm.edges) + list(bm.faces)
    plane_no = (1, 0, 0)
    if SYMMETRY_SOURCE == "positive_x":
        result = bmesh.ops.bisect_plane(
            bm,
            geom=geom,
            dist=0.0001,
            plane_co=(0, 0, 0),
            plane_no=plane_no,
            clear_inner=True,
            clear_outer=False,
        )
    else:
        result = bmesh.ops.bisect_plane(
            bm,
            geom=geom,
            dist=0.0001,
            plane_co=(0, 0, 0),
            plane_no=plane_no,
            clear_inner=False,
            clear_outer=True,
        )
    bmesh.ops.remove_doubles(bm, verts=bm.verts, dist=0.0005)
    bm.to_mesh(mesh)
    bm.free()
    mesh.update()

    mod = obj.modifiers.new("mirror_width", "MIRROR")
    mod.use_axis[0] = True
    mod.use_clip = True
    mod.use_mirror_merge = True
    mod.merge_threshold = 0.001
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.modifier_apply(modifier=mod.name)
    print(f"  symmetrized width from {SYMMETRY_SOURCE} across X center {center_x:.4f}")


def fix_orientation(obj):
    if UP_AXIS_FIX:
        bpy.ops.object.select_all(action="DESELECT")
        obj.select_set(True)
        bpy.context.view_layer.objects.active = obj
        obj.rotation_euler[0] = math.radians(90)
        bpy.ops.object.transform_apply(rotation=True)


def recentre_and_snap(obj, grid_cm):
    # Origin to geometry base, then place on world origin.
    bpy.ops.object.origin_set(type="ORIGIN_GEOMETRY", center="BOUNDS")
    obj.location = (0, 0, 0)
    bpy.context.view_layer.update()

    # Report bounding box in cm so you can sanity-check against grid multiples.
    dims = obj.dimensions  # in Blender units; set scene to metric/cm on import
    print(f"  bbox (units): {dims.x:.3f} x {dims.y:.3f} x {dims.z:.3f}")


def export_glb(obj, out_path: Path):
    bpy.ops.object.select_all(action="DESELECT")
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj
    bpy.ops.export_scene.gltf(
        filepath=str(out_path),
        use_selection=True,
        export_format="GLB",
        export_apply=True,
    )


def save_blend(obj, out_path: Path):
    bpy.ops.object.select_all(action="DESELECT")
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj
    bpy.ops.wm.save_as_mainfile(filepath=str(out_path))


def process_one(path: Path):
    clean_scene()
    objs = import_glb(path)
    if not objs:
        print(f"  no mesh in {path.name}")
        return
    obj = join_meshes(objs)
    fix_orientation(obj)
    orient_long_axis(obj)
    symmetrize_width(obj)
    remove_micro_islands(obj)
    remove_generated_base(obj)
    match_reference_dimensions(obj)
    retopo(obj, TARGET_TRIS)
    smooth_normals(obj)
    recentre_and_snap(obj, GRID_CM)
    glb_out = OUT_DIR / f"{path.stem}_ue.glb"
    blend_out = OUT_DIR / f"{path.stem}_ue.blend"
    export_glb(obj, glb_out)
    save_blend(obj, blend_out)
    print(f"  -> {glb_out.name}")
    print(f"  -> {blend_out.name}")


def main():
    setup_logging(SINGLE_MESH or "batch")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    # Set scene units to centimetres so dimensions read in cm against your grid.
    bpy.context.scene.unit_settings.system = "METRIC"
    bpy.context.scene.unit_settings.length_unit = "CENTIMETERS"

    if SINGLE_MESH:
        one = IN_DIR / f"{SINGLE_MESH}.glb"
        glbs = [one] if one.exists() else []
        if not glbs:
            print(f"NEXUS_MESH set but not found: {one}")
    else:
        glbs = sorted(IN_DIR.glob("*.glb"))
    print(f"Found {len(glbs)} raw meshes.")
    for p in glbs:
        print(f"Processing {p.name}")
        process_one(p)
    print(f"\nDone. UE-ready GLBs and Blender files in {OUT_DIR.resolve()}")


if __name__ == "__main__":
    main()



