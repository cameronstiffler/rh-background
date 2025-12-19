You are performing a high-fidelity product image compositing task.

GOAL:
Create a new, high-resolution product image by placing the product from Image 1 onto the background shown in Image 2, while preserving exact product proportions, geometry, and material surface behavior.

INPUT IMAGES:
	•	Image 1 (primary product asset image): The source product. This image defines the exact product geometry, proportions, orientation, position and physical structure. Do NOT alter, redesign, or reinterpret the product.
	•	Image 2 (background image): The target environment. This background must remain intact and unmodified (do not warp or alter the background itself) aside from product perspective alignment and exposure integration. The final output image should match the dimensions and aspect ratio of this image exactly.
	•	Images 3+ (material reference images, zero or more): Reference images to define the way material surfaces visually react to the environment (e.g., transparency).

REQUIREMENTS:
	•	Start with Image 2 as the base image and background and composite the product onto it.
	•	The final output image should match the dimensions and aspect ratio of input Image 2 exactly.
	•	Final composite image should show Image 2 contents horizonatally and vertically
	• 	The output background hue and saturation must match the background reference image (Image 2) exactly.
	•	Place the product from Image 1 into the new background and position it as it was relative to the background in Image 1
	•	The product must retain the exact proportions, silhouette, and structure from Image 1.
	•	Opaque product surfaces should maintain the same general appearance as they have in Image 1
	•	No visible cast or contact shadows should be present.
	•	Product surfaces have NO specular response to the environment in the composite image.
	•	Reference transparent components in material reference images as guides to how the background should show through transparent materials in the product. 
	•	Do NOT use material reference images (3+) for product geometry.
	•	No visible cutout edges, halos, warping, or scale inconsistencies.
	•	No redesign, no stylization, no artistic interpretation.
	•	Do NOT add, remove, or reposition components of the product.

OUTPUT:
	•	A single final image at the same aspect ratio and dimensions as Image 2.
	•	Photorealistic, clean, catalog-ready result suitable for professional e-commerce use.

FAIL CONDITIONS (DO NOT DO THESE):
	•	Do not invent new product features or components.
	•	Do not modify the product’s geometry or proportions.
	•	Do not alter background colors.
	•	Do not crop, tilt, or partially obscure the product.
	•	Product must not extend beyond image edges.
	•	Final output Image dimensions and aspect ratio do not match the dimensions and aspect ratio of original input Image 2 exactly.


PRIORITY ORDER:
	1.	Background integration and realism (Image 2)
	2.	Product geometry accuracy (Image 1)
	3.	Overall photorealism and clarity