# Asset contract and source gate

Create a contract before inference: identity, required forms, exclusions, geometry limits, texture target, acceptance views, and fallback route.

Score every geometry conditioning raster out of 100: identity/forms 30, silhouette/matte 25, geometry observability 25, framing/resolution 10, integrity 10. Reject below 85, below 20 geometry observability, or any failed/unknown category.

Reject ambiguous silhouettes, black regions that read as voids, reflections, floor bands, text, blur, severe compression, missing forms, or unverified framing. Do not solve a rejected source with seed retries.
