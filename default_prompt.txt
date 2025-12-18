Goal: Extract only the product from the asset and composite it onto the provided background.
Inputs:
- Background reference: use as the final backdrop; keep its structure/lighting; remove any furniture there if the product replaces it.
- Product asset: object to extract; preserve its scale, camera perspective, and lens feel.
- Object reference photos (optional): material guide for transparency, gloss, luminosity, and reflectivity.
- Glass opacity hint (optional): match the provided transparency level; preserve natural refractions/reflections.
Rules:
- Use only the provided background and product; do not add props, text, people, or logos.
- Remove any placeholder/old furniture or duplicates from the background.
- Keep cutout edges clean; no halos, matte spill, or invented edges.
- Lighting/shadows: match the background’s lighting direction/intensity; keep any soft shadows from the asset but do not create new or heavier shadows where none exist.
- Materials: match materials from object refs; glass must stay clear with the background visible through it; no haze/tint; keep specular highlights; metals should reflect naturally.
- Preserve product scale/perspective consistent with the asset.
Output:
- High-res, realistic composite; clean edges; no watermarks.
