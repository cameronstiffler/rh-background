"""
Gemini 3 background replacement helper.

Reads product shots from test_assets/<product>/, pairs them with reference
backgrounds in ref_background/<product>/, and sends them to the Gemini API
to generate composited images saved into gen_background/<product>/.
"""

from __future__ import annotations

import argparse
import base64
import io
import os
import sys
from pathlib import Path
from typing import Iterator, Optional, Tuple

import google.generativeai as genai
from dotenv import load_dotenv
from PIL import Image


ASSETS_ROOT = Path("test_assets")
BACKGROUND_ROOT = Path("ref_background")
OUTPUT_ROOT = Path("output")

ASSET_PREFERRED_EXTENSIONS = {".tif", ".tiff"}
ASSET_FALLBACK_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
BACKGROUND_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".webp"}

DEFAULT_SAFETY_SETTINGS = [
    {"category": cat, "threshold": "BLOCK_NONE"}
    for cat in [
        "HARM_CATEGORY_HATE_SPEECH",
        "HARM_CATEGORY_DANGEROUS_CONTENT",
        "HARM_CATEGORY_SEXUALLY_EXPLICIT",
        "HARM_CATEGORY_HARASSMENT",
    ]
]

# Prompt keeps instructions focused on compositing the product object onto the
# provided background reference.
DEFAULT_PROMPT = """You are a senior photo retoucher. Extract the product from the provided asset (object only) and place it naturally onto the provided background reference.
Rules:
- Keep the product scale and camera perspective consistent with the asset photo.
- Use only the provided background; do not invent extra props or text.
- Preserve realistic lighting, reflections, and shadows; keep clean edges and soft natural shadows.
- Maintain glass transparency: do not fill clear glass; keep background visible through glass while preserving reflections."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate background replacements with the Gemini 3 API."
    )
    parser.add_argument(
        "--product",
        help="Only process a single product folder (matches folder name under test_assets/).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Max number of images to process (across all products).",
    )
    parser.add_argument(
        "--prompt-file",
        type=Path,
        help="Optional text file to override the default prompt.",
    )
    parser.add_argument(
        "--max-side",
        type=int,
        default=int(os.getenv("MAX_SIDE", "4096")),
        help="Max long-edge pixels sent to Gemini (0 to disable; default 4096).",
    )
    parser.add_argument(
        "--model",
        default=os.getenv("GEMINI_MODEL", "models/gemini-3-pro-image-preview"),
        help="Gemini model name/path.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.25,
        help="Generation temperature for Gemini.",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=420,
        help="Request timeout in seconds per Gemini call.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=OUTPUT_ROOT,
        help="Output directory for generated images.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="List the files that would be processed without calling Gemini.",
    )
    return parser.parse_args()


def load_prompt(prompt_file: Optional[Path]) -> str:
    if prompt_file:
        return prompt_file.read_text().strip()
    return DEFAULT_PROMPT


def ensure_model_path(model: str) -> str:
    return model if model.startswith("models/") else f"models/{model}"


def resize_if_needed(img: Image.Image, max_side: int) -> Image.Image:
    """Downscale image to max_side on the longest edge to keep payload reasonable."""
    if max_side <= 0:
        return img
    width, height = img.size
    longest = max(width, height)
    if longest <= max_side:
        return img
    scale = max_side / float(longest)
    new_size = (max(1, int(width * scale)), max(1, int(height * scale)))
    return img.resize(new_size, resample=Image.LANCZOS)


def first_image(folder: Path, extensions: set[str]) -> Optional[Path]:
    if not folder.exists():
        return None
    for path in sorted(folder.iterdir()):
        if path.is_file() and path.suffix.lower() in extensions:
            return path
    return None


def select_asset_image(product_dir: Path) -> Optional[Path]:
    preferred = first_image(product_dir, ASSET_PREFERRED_EXTENSIONS)
    if preferred:
        return preferred
    return first_image(product_dir, ASSET_FALLBACK_EXTENSIONS)


def discover_work(
    product_filter: Optional[str],
) -> Iterator[Tuple[str, Path, Path, Optional[Path]]]:
    for product_dir in sorted(ASSETS_ROOT.iterdir()):
        if not product_dir.is_dir():
            continue
        if product_filter and product_dir.name != product_filter:
            continue

        target_path = select_asset_image(product_dir)
        if not target_path:
            print(f"[skip] No image asset found for {product_dir.name}", file=sys.stderr)
            continue
        if target_path.suffix.lower() not in ASSET_PREFERRED_EXTENSIONS:
            print(
                f"[info] Using non-TIF asset for {product_dir.name}: {target_path.name}",
                file=sys.stderr,
            )

        ref_dir = BACKGROUND_ROOT / product_dir.name
        background_path = first_image(ref_dir, BACKGROUND_EXTENSIONS)
        if not background_path:
            print(f"[skip] No background reference for {product_dir.name}", file=sys.stderr)
            continue

        # Shadows disabled for now; no shadow ref expected.
        shadow_path = None

        yield product_dir.name, target_path, background_path, shadow_path


def to_png_bytes(image_path: Path, max_side: int) -> bytes:
    with Image.open(image_path) as im:
        convert_mode = "RGBA" if im.mode in ("RGBA", "LA") else "RGB"
        converted = im.convert(convert_mode)
        converted = resize_if_needed(converted, max_side)
        buffer = io.BytesIO()
        converted.save(buffer, format="PNG")
        return buffer.getvalue()


def image_part_from_bytes(data: bytes) -> dict:
    encoded = base64.b64encode(data).decode("ascii")
    return {"inline_data": {"mime_type": "image/png", "data": encoded}}


def normalize_inline_data(data) -> bytes:
    if isinstance(data, bytes):
        return data
    if isinstance(data, str):
        return base64.b64decode(data)
    raise TypeError(f"Unsupported inline data type: {type(data)}")


def extract_image_from_response(response) -> bytes:
    for candidate in getattr(response, "candidates", []):
        content = getattr(candidate, "content", None)
        if not content:
            continue
        for part in getattr(content, "parts", []):
            inline = getattr(part, "inline_data", None)
            if inline and getattr(inline, "data", None):
                return normalize_inline_data(inline.data)
    raise RuntimeError("No image returned by Gemini response.")


def summarize_response_for_debug(response) -> str:
    summaries = []
    for idx, candidate in enumerate(getattr(response, "candidates", [])):
        reason = getattr(candidate, "finish_reason", None) or "-"
        part_types = []
        content = getattr(candidate, "content", None)
        for part in getattr(content, "parts", []) if content else []:
            if getattr(part, "inline_data", None):
                part_types.append("inline_data")
            elif getattr(part, "text", None):
                part_types.append("text")
            else:
                part_types.append(type(part).__name__)
        summaries.append(f"cand{idx}: reason={reason}, parts={','.join(part_types) or 'none'}")
    return "; ".join(summaries) if summaries else "no candidates"


def call_gemini(
    api_key: str,
    model_name: str,
    product_name: str,
    background_png: bytes,
    asset_png: bytes,
    shadow_png: Optional[bytes],
    prompt: str,
    temperature: float,
    timeout: int,
) -> bytes:
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(ensure_model_path(model_name))
    last_exc: Exception | None = None
    for attempt in range(3):
        try:
            parts = [
                {"mime_type": "image/png", "data": background_png},
                {"text": "Background reference (use as backdrop)."},
                {"mime_type": "image/png", "data": asset_png},
                {"text": "Product object (extract and place onto the background)."},
            ]
            response = model.generate_content(
                parts + [prompt],
                generation_config={"temperature": temperature},
                safety_settings=DEFAULT_SAFETY_SETTINGS,
                request_options={"timeout": timeout},
            )
            try:
                return extract_image_from_response(response)
            except RuntimeError as exc:
                debug_summary = summarize_response_for_debug(response)
                prompt_feedback = getattr(response, "prompt_feedback", None)
                raise RuntimeError(
                    f"{exc} (response summary: {debug_summary}, prompt_feedback={prompt_feedback})"
                ) from exc
        except Exception as exc:  # pylint: disable=broad-except
            last_exc = exc
            if attempt < 2:
                # Brief backoff before retrying transient errors.
                import time

                time.sleep(2 * (attempt + 1))
                continue
            raise
    # Should not reach here, but keeps mypy happy.
    raise RuntimeError(f"Gemini request failed for {product_name}: {last_exc}")


def main() -> None:
    args = parse_args()
    load_dotenv()

    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("Missing GEMINI_API_KEY in environment or .env file.", file=sys.stderr)
        sys.exit(1)

    prompt = load_prompt(args.prompt_file)
    work_items = list(discover_work(args.product))

    if args.limit:
        work_items = work_items[: args.limit]

    if not work_items:
        print("No matching product shots found to process.", file=sys.stderr)
        return

    if args.dry_run:
        for product_name, asset_path, background_path, shadow_path in work_items:
            print(
                f"[dry-run] {product_name}: target={asset_path} background={background_path} shadow_ref={shadow_path or 'none'}"
            )
        return

    for product_name, asset_path, background_path, shadow_path in work_items:
        print(
            f"[gemini] {product_name}: {asset_path.name} with {background_path.name} (shadow: disabled)"
        )
        out_dir = args.output / product_name
        out_dir.mkdir(parents=True, exist_ok=True)

        asset_png = to_png_bytes(asset_path, args.max_side)
        asset_png_path = out_dir / "asset.png"
        asset_png_path.write_bytes(asset_png)

        background_png = to_png_bytes(background_path, args.max_side)
        shadow_png = None  # Shadows disabled currently

        try:
            output_bytes = call_gemini(
                api_key=api_key,
                model_name=args.model,
                product_name=product_name,
                background_png=background_png,
                asset_png=asset_png,
                shadow_png=shadow_png,
                prompt=prompt,
                temperature=args.temperature,
                timeout=args.timeout,
            )
        except Exception as exc:  # pylint: disable=broad-except
            print(f"  ✖ Failed: {exc}", file=sys.stderr)
            continue

        out_name = f"{product_name}.png"
        out_path = out_dir / out_name
        out_path.write_bytes(output_bytes)
        print(f"  ✔ Saved -> {out_path}")


if __name__ == "__main__":
    main()
