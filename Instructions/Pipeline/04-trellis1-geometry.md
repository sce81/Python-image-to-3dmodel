# TRELLIS 1 geometry

Use `generate_props.py` for multi-view reconstruction. Stage all selected views into one source folder; do not create one mesh per image.

Use clay conditioning for dark geometry, default tuned guidance, and best-of-N selection. Record candidate integrity metrics. Keep `NEXUS_REMOVE_REFLECTIONS` off except for an explicit experiment.

Export a raw review mesh, then send it to mesh validation before Blender or PBR promotion.
