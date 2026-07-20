# TRELLIS 2 geometry and extraction

Use `generate_props_t2.py` only for one approved conditioning raster and the isolated T2 environment.

- Default to `1024_cascade`, 49,152 tokens, model sampler defaults, and four candidate seeds. Compare direct `1024` only as a controlled A/B.
- Do not raise guidance, port TRELLIS 1 guidance, or use `1536_cascade` on the 16 GB card.
- Generate from the paired clay geometry raster. `NEXUS_GEOMETRY_ONLY=1` means defer texture acceptance only; it must still export through `o_voxel.postprocess.to_glb`.
- Always extract review GLBs with `remesh=True`, `remesh_band=1`, `remesh_project=0`, a one-million triangle target, and a 4096 atlas. Do not direct-export the latent mesh or inherit `remesh_project=0.9`.
- The Honda comparison proved that the direct latent branch produced 33,274 boundary components while the full O-Voxel extraction produced 1,623 on like-for-like T2 output.
