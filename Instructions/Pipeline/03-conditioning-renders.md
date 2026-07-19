# Conditioning renders

Create paired renders from accepted source evidence.

- Geometry render: a square studio raster with a near-white background, continuous silhouette, no floor/reflection/pedestal, and opaque readable glazing, intakes, wheel interiors, and lower body. Use this only for geometry.
- Texture render: preserve exactly the geometry render camera, scale, silhouette, and background. Restore material evidence: paint, glass, lamps, trim, tyres, wheels, and grille detail. Use this only in the texture stage.
- Inspect the saved raster before inference. The geometry and texture renders must stay camera-aligned.
