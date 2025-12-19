You are performing a high-fidelity product image compositing task.

GOAL:
Create a new, high-resolution product image by placing the product from Image 1 onto the background shown in Image 2, while preserving exact product proportions, geometry, and material surface behavior.

INPUT IMAGES:
- Image 1 (primary product asset image): The source product. This image defines the exact product geometry, proportions, orientation, position and physical structure. Do NOT alter, redesign, or reinterpret the product.
- Image 2 (background image): The target environment. This background must remain intact and unmodified (do not warp or alter the background itself) aside from product perspective alignment and exposure integration.
- Images 3+ (material reference images, zero or more): Reference images to define the way material surfaces visually react to the environment (e.g., transparency).

REQUIREMENTS:
- The product must retain the exact proportions, silhouette, and structure from Image 1.
- The the output image should match the dimensions of Image 2.
- Image 2 fills and is constrained to the dimensions of output image.
- The entire product must be contained within the edges of Image 2.
- The output background hue and saturation must match the background reference image(Image 2) exactly.
- The product must be centered, vertically aligned, and naturally placed within the background from Image 2
- Light appears to originate softly from within the scene rather than reflecting external sources
- The product must NOT cast shadows onto the background in the output image (no cast or contact shadows on the background or self).
- Product surfaces have no specular response to environment
- Reference transparent components in material reference images(3+) as examples for how transparent components of product should look
- The product surfaces do not reflect background walls, ceiling, floor, objects or lighting not produced by the product.
- Do NOT use material reference Images 3+ for product geometry.
- No visible cutout edges, halos, warping, or scale inconsistencies.
- No redesign, no stylization, no artistic interpretation.
- Do NOT add, remove, or reposition components of the product.


OUTPUT:
- A single final image at the same aspect ratio as Image 1. If upscaling is applied, limit it to a maximum of 2× per dimension (approximately 4× total pixels).
- As a last step before outputting the new image, enhance the perceived detail existing background textures by increasing local contrast, clarity, and fine tonal separation, without changing resolution. Do not upscale, resize, or add new detail. Emphasize existing material grain, fabric weave, or surface texture only where it already exists. Avoid oversharpening, halos, noise amplification, or artificial texture patterns. Maintain original lighting direction, color balance, highlights, and product self-shadows, adjusting only subtle micro-contrast to improve texture legibility.
- Photorealistic, clean, catalog-ready result suitable for professional e-commerce use.

FAIL CONDITIONS (DO NOT DO THESE):
- Do not invent new product features or components.
- Do not modify the product’s geometry or proportions.
- Do not alter background colors or composition.
- Do not crop, tilt, or partially obscure the product.
- Product must not extend beyond image edges.

PRIORITY ORDER:
1. Product geometry accuracy (Image 1)
2. Background integration and realism (Image 2)
3. Material surface fidelity (Images 3+)
4. Overall photorealism and clarity
