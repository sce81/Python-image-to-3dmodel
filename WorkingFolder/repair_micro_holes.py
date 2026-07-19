"""Conservatively fill verified microscopic boundary loops in a GLB.

Run with Blender and set NEXUS_REPAIR_INPUT and NEXUS_REPAIR_OUTPUT.
The pass never fills an open chain, a loop larger than the configured bounds,
or any feature that needs more than 128 boundary vertices.
"""

import json
import os
from pathlib import Path

import bmesh
import bpy


INPUT = Path(os.environ["NEXUS_REPAIR_INPUT"])
OUTPUT = Path(os.environ["NEXUS_REPAIR_OUTPUT"])
REPORT = OUTPUT.with_suffix(".micro_hole_repair.json")
MAX_VERTICES = int(os.environ.get("NEXUS_MICRO_HOLE_MAX_VERTICES", "128"))
MAX_EXTENT_RATIO = float(os.environ.get("NEXUS_MICRO_HOLE_MAX_EXTENT_RATIO", "0.003"))
MAX_PERIMETER_RATIO = float(os.environ.get("NEXUS_MICRO_HOLE_MAX_PERIMETER_RATIO", "0.012"))


def clean_scene():
    bpy.ops.wm.read_factory_settings(use_empty=True)


def joined_mesh():
    bpy.ops.import_scene.gltf(filepath=str(INPUT))
    meshes = [obj for obj in bpy.context.scene.objects if obj.type == "MESH"]
    if not meshes:
        raise RuntimeError(f"No mesh found in {INPUT}")
    bpy.ops.object.select_all(action="DESELECT")
    for obj in meshes:
        obj.select_set(True)
    bpy.context.view_layer.objects.active = meshes[0]
    if len(meshes) > 1:
        bpy.ops.object.join()
    return bpy.context.view_layer.objects.active


def boundary_components(bm):
    edges = {edge for edge in bm.edges if len(edge.link_faces) == 1}
    loops = []
    while edges:
        seed = edges.pop()
        stack = [seed]
        component = {seed}
        while stack:
            edge = stack.pop()
            for vertex in edge.verts:
                for linked in vertex.link_edges:
                    if linked in edges and len(linked.link_faces) == 1:
                        edges.remove(linked)
                        component.add(linked)
                        stack.append(linked)
        verts = {vertex for edge in component for vertex in edge.verts}
        degree = {vertex: 0 for vertex in verts}
        for edge in component:
            for vertex in edge.verts:
                degree[vertex] += 1
        closed = bool(component) and all(value == 2 for value in degree.values())
        coords = [vertex.co for vertex in verts]
        extent = max(
            max(co[axis] for co in coords) - min(co[axis] for co in coords)
            for axis in range(3)
        )
        perimeter = sum(edge.calc_length() for edge in component)
        loops.append({
            "edges": component,
            "verts": verts,
            "closed": closed,
            "vertex_count": len(verts),
            "extent": float(extent),
            "perimeter": float(perimeter),
        })
    return loops


def main():
    clean_scene()
    obj = joined_mesh()
    bm = bmesh.new()
    bm.from_mesh(obj.data)
    primary_extent = max(
        max(vertex.co[axis] for vertex in bm.verts) - min(vertex.co[axis] for vertex in bm.verts)
        for axis in range(3)
    )
    max_extent = primary_extent * MAX_EXTENT_RATIO
    max_perimeter = primary_extent * MAX_PERIMETER_RATIO
    before = boundary_components(bm)
    eligible = [
        loop for loop in before
        if loop["closed"]
        and loop["vertex_count"] <= MAX_VERTICES
        and loop["extent"] <= max_extent
        and loop["perimeter"] <= max_perimeter
    ]
    for loop in eligible:
        bmesh.ops.triangle_fill(bm, edges=list(loop["edges"]), use_beauty=True)
    bm.to_mesh(obj.data)
    bm.free()
    obj.data.update()
    bpy.context.view_layer.update()
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    bpy.ops.object.select_all(action="DESELECT")
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj
    bpy.ops.export_scene.gltf(filepath=str(OUTPUT), export_format="GLB", use_selection=True)

    audit = {
        "input": str(INPUT),
        "output": str(OUTPUT),
        "primary_extent": primary_extent,
        "thresholds": {
            "max_vertices": MAX_VERTICES,
            "max_extent": max_extent,
            "max_perimeter": max_perimeter,
        },
        "boundary_loops_before": len(before),
        "loops_filled": len(eligible),
        "loops_rejected": len(before) - len(eligible),
        "filled_loop_max_extent": max((loop["extent"] for loop in eligible), default=0.0),
        "filled_loop_max_perimeter": max((loop["perimeter"] for loop in eligible), default=0.0),
    }
    REPORT.write_text(json.dumps(audit, indent=2), encoding="utf-8")
    print(json.dumps(audit, indent=2))


main()
