You are performing a high-fidelity product image compositing task.

GOAL:
Create a new, high-resolution product image by placing the product from Image 1 onto the background shown in Image 2, while preserving exact product proportions, geometry, and material behavior.

INPUT IMAGES:
- Image 1 (primary product asset image): The source product. This image defines the exact product geometry, scale, proportions, orientation, position and physical structure. Do NOT alter, redesign, or reinterpret the product.
- Image 2 (background image): The target environment. This background must remain intact and unmodified aside from perspective, and exposure integration.
- Images 3+ (material reference images, zero or more): Reference images that define the correct material properties of the product (e.g., transparency, reflectivity, refraction, edge behavior).

REQUIREMENTS:
- The product must retain the exact proportions, silhouette, and structure from Image 1.
- The entire product must be visible and contaied within the edges of the output image
- The output background must match the background reference image exactly.
- The product must be centered, vertically aligned, and naturally placed within the background from Image 2.
- The product must NOT cast shadows onto the background in the output image.
- The product material transparency must closely match the material reference images.
- Do NOT use material reference images for product geometry.
- No visible cutout edges, halos, warping, or scale inconsistencies.
- No redesign, no stylization, no artistic interpretation.
- Do NOT add, remove, or reposition components of the product.

OUTPUT:
- A single high-resolution image.
- Same aspect ratio and proportions as Image 1.
- Photorealistic, clean, catalog-ready result suitable for professional e-commerce use.

FAIL CONDITIONS (DO NOT DO THESE):
- Do not invent new product features or components.
- Do not modify the product’s geometry or proportions.
- Do not exaggerate or reduce material transparency beyond reference behavior.
- Do not alter background colors, textures, or composition.
- Do not crop, tilt, or partially obscure the product.

PRIORITY ORDER:
1. Product geometry accuracy (Image 1)
2. Background integration and lighting realism (Image 2)
3. Material fidelity (Images 3+)
4. Overall photorealism and clarity
