# Car Generation — Meshy AI Instructions
## Camden/London UE5 Asset Collection

---

## Overview

All cars in this collection follow a consistent pipeline:
clean body shell, no wheels, no wheel arches, no branding,
UK specification lighting, right-hand drive layout.
Outputs are GLB (source) + FBX (Unreal Engine ready).

---

## Reference Images Required

Provide **3 reference photos** of the target vehicle:
- **Front** — shows headlight shape, grille, bumper profile
- **Rear** — shows tail lights, boot/trunk line, rear bumper
- **Top/Side** — shows roofline, shoulder line, overall silhouette

Preferred: clean studio or press photos, single car,
neutral background. Avoid lifestyle/in-situ shots.

---

## Step 1 — Concept Image (image_to_image)

Generate a clean body shell concept from the 3 references.

**Prompt guidance:**
- Reference all 3 uploaded images
- Remove wheels, wheel arches, wheel openings
- Remove all branding, badges, logos
- Smooth body where arches were — no cutouts or cavities
- UK specification: right-hand drive, UK headlight clusters,
  UK rear light configuration, UK number plate recess
- Studio lighting, white background, isometric 3/4 view
- Keep prompt under 600 characters

**Common issues:**
- Wheel arches visible → re-run image_to_image with stronger
  "completely smooth body, no arch openings" language
- Branding visible → add "remove all badges and logos" explicitly
- Wrong drive side → specify "right-hand drive, RHD" clearly

---

## Step 2 — 3D Generation (image_to_3d)

Convert the approved concept image to 3D.

**Settings:**
- model_type: standard
- symmetry_mode: auto
- should_texture: true
- hd_texture: true
- enable_pbr: true
- remove_lighting: true

**Texture prompt example:**
"[Make] [Model] body shell, smooth painted finish,
UK spec lighting, no wheels, no arches, no badging,
PBR metallic paint"

**Note:** Dimensions cannot be set precisely at generation.
Scale the model to exact real-world dimensions in UE5
using the vehicle's spec sheet measurements (mm).

---

## Step 3 — FBX Export (convert_model_formats)

Convert GLB → FBX for Unreal Engine import.

- Input: GLB artifact from Step 2
- Target format: fbx
- No other settings required

---

## Step 4 — Unreal Engine Import

1. Import FBX into UE5 Content Browser
2. Scale to real-world dimensions using spec sheet (mm → cm ÷ 10)
3. Apply custom vehicle paint material (PBR metallic)
4. Attach wheel/tyre meshes as separate static mesh components
5. Attach wheel arch trim as separate components if required

---

## Vehicles Completed

| Vehicle | GLB | FBX |
|---|:---:|:---:|
| Ferrari Lusso | Yes | Yes |
| VW Polo Mk6 | Yes | Yes |
| Mercedes EQS | Yes | Yes |

---

## Real-World Dimensions Reference

| Vehicle        | Length (mm) | Width (mm) | Height (mm) |
|----------------|-------------|------------|-------------|
| Ferrari Lusso  | 5026        | 1999       | 1544        |
| VW Polo Mk6    | 4053        | 1751       | 1461        |
| Mercedes EQS   | 5223        | 1926       | 1512        |

Scale in UE5: divide mm by 10 to get UE5 units (cm).

---

## Prompt Length Warning

All image_to_image and image_to_3d prompts must be
**under 600 characters**. Longer prompts will fail.
Keep descriptions concise — prioritise geometry and
lighting instructions over decorative detail.

---

## Notes

- Never chain retexture immediately after image_to_3d —
  the generated model already has textures applied.
- Retexture is only used when changing to a different
  visual style on an existing model.
- Wheel arches should always be removed at the 2D concept
  stage, not attempted post-3D.
- All models are body shells only — wheels, interiors,
  and glass are separate UE5 components.
