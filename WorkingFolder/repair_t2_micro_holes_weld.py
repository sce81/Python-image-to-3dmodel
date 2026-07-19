"""Weld coincident vertices and fill only small, closed boundary loops."""

import json
import os
from pathlib import Path

import bmesh
import bpy


source = Path(os.environ["NEXUS_REPAIR_INPUT"])
target = Path(os.environ["NEXUS_REPAIR_OUTPUT"])
weld_distance = float(os.environ.get("NEXUS_MICRO_WELD_DISTANCE", "0.00001"))
max_extent_ratio = float(os.environ.get("NEXUS_MICRO_HOLE_MAX_EXTENT_RATIO", "0.02"))
max_perimeter_ratio = float(os.environ.get("NEXUS_MICRO_HOLE_MAX_PERIMETER_RATIO", "0.06"))
max_vertices = int(os.environ.get("NEXUS_MICRO_HOLE_MAX_VERTICES", "256"))


def boundary_loops(bm):
    pending = {edge for edge in bm.edges if len(edge.link_faces) == 1}
    result = []
    while pending:
        seed = pending.pop()
        edges, stack = {seed}, [seed]
        while stack:
            edge = stack.pop()
            for vert in edge.verts:
                for linked in vert.link_edges:
                    if linked in pending and len(linked.link_faces) == 1:
                        pending.remove(linked)
                        edges.add(linked)
                        stack.append(linked)
        verts = {vert for edge in edges for vert in edge.verts}
        degrees = {vert: 0 for vert in verts}
        for edge in edges:
            for vert in edge.verts:
                degrees[vert] += 1
        coords = [vert.co for vert in verts]
        result.append({
            "edges": edges,
            "closed": all(value == 2 for value in degrees.values()),
            "vertices": len(verts),
            "extent": max(max(co[axis] for co in coords) - min(co[axis] for co in coords) for axis in range(3)),
            "perimeter": sum(edge.calc_length() for edge in edges),
        })
    return result


bpy.ops.wm.read_factory_settings(use_empty=True)
bpy.ops.import_scene.gltf(filepath=str(source))
objects = [obj for obj in bpy.context.scene.objects if obj.type == "MESH"]
bpy.ops.object.select_all(action="DESELECT")
for obj in objects:
    obj.select_set(True)
bpy.context.view_layer.objects.active = objects[0]
if len(objects) > 1:
    bpy.ops.object.join()
obj = bpy.context.view_layer.objects.active
bm = bmesh.new()
bm.from_mesh(obj.data)
verts_before = len(bm.verts)
bmesh.ops.remove_doubles(bm, verts=list(bm.verts), dist=weld_distance)
verts_after_weld = len(bm.verts)
extent = max(max(vert.co[axis] for vert in bm.verts) - min(vert.co[axis] for vert in bm.verts) for axis in range(3))
before = boundary_loops(bm)
eligible = [
    loop for loop in before
    if loop["closed"] and loop["vertices"] <= max_vertices
    and loop["extent"] <= extent * max_extent_ratio
    and loop["perimeter"] <= extent * max_perimeter_ratio
]
for loop in eligible:
    bmesh.ops.triangle_fill(bm, edges=list(loop["edges"]), use_beauty=True)
after = boundary_loops(bm)
bm.to_mesh(obj.data)
bm.free()
obj.data.update()
target.parent.mkdir(parents=True, exist_ok=True)
bpy.ops.object.select_all(action="DESELECT")
obj.select_set(True)
bpy.context.view_layer.objects.active = obj
bpy.ops.export_scene.gltf(filepath=str(target), export_format="GLB", use_selection=True)
report = {
    "source": str(source), "target": str(target), "weld_distance": weld_distance,
    "vertices_before": verts_before, "vertices_after_weld": verts_after_weld,
    "boundary_components_before_fill": len(before), "closed_loops_filled": len(eligible),
    "boundary_components_after_fill": len(after),
    "thresholds": {"max_vertices": max_vertices, "max_extent_ratio": max_extent_ratio, "max_perimeter_ratio": max_perimeter_ratio},
}
target.with_suffix(".micro_hole_repair.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
print(json.dumps(report, indent=2))
