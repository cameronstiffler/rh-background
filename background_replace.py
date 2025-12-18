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
from typing import Iterator, Optional

import google.generativeai as genai
from dotenv import load_dotenv
from PIL import Image


ASSETS_ROOT = Path("test_assets")
BACKGROUND_ROOT = Path("ref_background")
OUTPUT_ROOT = Path("output")
OBJECT_REFERENCE_ROOT = Path("ref_objects")
ASSET_DUMP_ROOT = Path("png")
DEFAULT_PROMPT_PATH = Path("default_prompt.md")

ASSET_PREFERRED_EXTENSIONS = {".tif", ".tiff"}
ASSET_FALLBACK_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
BACKGROUND_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".webp"}
OBJECT_REFERENCE_EXTENSIONS = BACKGROUND_EXTENSIONS
RESOLUTION_MAP = {"1K": 1024, "2K": 2048, "4K": 4096}
DEFAULT_MAX_OBJECT_REFS = 4
DEFAULT_PROMPT_FALLBACK = """Goal: Extract only the product from the asset and composite it onto the provided background.
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
- High-res, realistic composite; clean edges; no watermarks."""

DEFAULT_SAFETY_SETTINGS = [
    {"category": cat, "threshold": "BLOCK_NONE"}
    for cat in [
        "HARM_CATEGORY_HATE_SPEECH",
        "HARM_CATEGORY_DANGEROUS_CONTENT",
        "HARM_CATEGORY_SEXUALLY_EXPLICIT",
        "HARM_CATEGORY_HARASSMENT",
    ]
]

def parse_args() -> argparse.Namespace:
    # Ensure .env values are available for defaults.
    load_dotenv()

    env_resolution = os.getenv("RESOLUTION")
    resolution_default = (
        env_resolution.strip().upper() if env_resolution else "4K"
    )
    if resolution_default not in RESOLUTION_MAP:
        resolution_default = "4K"

    env_temp_raw = os.getenv("GEMINI_TEMPERATURE", os.getenv("TEMPERATURE"))
    try:
        env_temp_val = float(env_temp_raw) if env_temp_raw is not None else 0.25
    except ValueError:
        env_temp_val = 0.25

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
        help="Max long-edge pixels sent to Gemini (0 to disable; default 4096). Overridden by --resolution.",
    )
    parser.add_argument(
        "--model",
        default=os.getenv("GEMINI_MODEL", "models/gemini-3-pro-image-preview"),
        help="Gemini model name/path.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=env_temp_val,
        help="Generation temperature for Gemini (defaults to GEMINI_TEMPERATURE/TEMPERATURE env).",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=420,
        help="Request timeout in seconds per Gemini call.",
    )
    env_seed_raw = os.getenv("GEMINI_SEED", os.getenv("SEED"))
    try:
        env_seed_val = int(env_seed_raw) if env_seed_raw is not None else None
    except ValueError:
        env_seed_val = None
    parser.add_argument(
        "--seed",
        type=int,
        default=env_seed_val,
        help="Optional deterministic seed for Gemini. Defaults to GEMINI_SEED/SEED env vars when set.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=OUTPUT_ROOT,
        help="Output directory for generated images.",
    )
    parser.add_argument(
        "--resolution",
        choices=sorted(RESOLUTION_MAP.keys()),
        default=resolution_default,
        help="Target long-edge resolution for both request inputs and final output (overrides --max-side).",
    )
    parser.add_argument(
        "--glass-opacity",
        "--glass_opacity",
        dest="glass_opacity",
        type=glass_opacity_value,
        default=None,
        help="Optional glass opacity hint (0.0 fully transparent to 1.0 fully opaque); appends guidance to the prompt.",
    )
    parser.add_argument(
        "--results",
        type=positive_int,
        default=1,
        help="Number of results to generate per product (runs the request multiple times and saves with unique names).",
    )
    parser.add_argument(
        "--max-object-refs",
        type=int,
        default=DEFAULT_MAX_OBJECT_REFS,
        help="Max number of object reference images from ref_objects/ to include (0 disables).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="List the files that would be processed without calling Gemini.",
    )
    return parser.parse_args()


def load_prompt(prompt_file: Optional[Path]) -> str:
    path = prompt_file or DEFAULT_PROMPT_PATH
    try:
        return path.read_text(encoding="utf-8").strip()
    except FileNotFoundError:
        if prompt_file:
            raise
        print(
            f"[warn] Default prompt file missing at {path}; using built-in fallback.",
            file=sys.stderr,
        )
        return DEFAULT_PROMPT_FALLBACK


def glass_opacity_value(raw: str) -> float:
    try:
        value = float(raw)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("glass opacity must be a number between 0.0 and 1.0") from exc
    if not 0.0 <= value <= 1.0:
        raise argparse.ArgumentTypeError("glass opacity must be between 0.0 and 1.0")
    return value


def positive_int(raw: str) -> int:
    try:
        value = int(raw)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("value must be an integer") from exc
    if value < 1:
        raise argparse.ArgumentTypeError("value must be >= 1")
    return value


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


def collect_images(folder: Path, extensions: set[str]) -> list[Path]:
    if not folder.exists():
        return []
    return [
        path
        for path in sorted(folder.iterdir())
        if path.is_file() and path.suffix.lower() in extensions
    ]


def select_asset_image(product_dir: Path) -> Optional[Path]:
    preferred = first_image(product_dir, ASSET_PREFERRED_EXTENSIONS)
    if preferred:
        return preferred
    return first_image(product_dir, ASSET_FALLBACK_EXTENSIONS)


def find_object_reference_images(product_name: str) -> list[Path]:
    """Find extra product reference photos from ref_objects/<product>/ to guide materials."""
    if not OBJECT_REFERENCE_ROOT.exists():
        return []
    product_specific = OBJECT_REFERENCE_ROOT / product_name
    return collect_images(product_specific, OBJECT_REFERENCE_EXTENSIONS)


def log_image_submission(label: str, path: Path, meta: dict) -> None:
    src_w, src_h = meta["original_size"]
    final_w, final_h = meta["final_size"]
    print(
        f"    {label}: {path} -> {final_w}x{final_h}px PNG "
        f"(source {src_w}x{src_h} {meta['original_format']}/{meta['original_mode']})"
    )


def discover_work(
    product_filter: Optional[str],
) -> Iterator[tuple[str, Path, Path, list[Path], Optional[Path]]]:
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

        object_refs = find_object_reference_images(product_dir.name)

        # Shadows disabled for now; no shadow ref expected.
        shadow_path = None

        yield product_dir.name, target_path, background_path, object_refs, shadow_path


def to_png_bytes_with_meta(image_path: Path, max_side: int) -> tuple[bytes, dict]:
    with Image.open(image_path) as im:
        original_size = im.size
        original_mode = im.mode
        original_format = im.format or image_path.suffix.upper().lstrip(".")
        convert_mode = "RGBA" if im.mode in ("RGBA", "LA") else "RGB"
        converted = im.convert(convert_mode)
        converted = resize_if_needed(converted, max_side)
        final_size = converted.size
        buffer = io.BytesIO()
        converted.save(buffer, format="PNG")
        return buffer.getvalue(), {
            "original_size": original_size,
            "final_size": final_size,
            "original_mode": original_mode,
            "final_mode": converted.mode,
            "original_format": original_format,
            "mime": "image/png",
        }


def image_part_from_bytes(data: bytes) -> dict:
    encoded = base64.b64encode(data).decode("ascii")
    return {"inline_data": {"mime_type": "image/png", "data": encoded}}


def resize_output_if_needed(image_bytes: bytes, max_side: int) -> bytes:
    if max_side <= 0:
        return image_bytes
    with Image.open(io.BytesIO(image_bytes)) as im:
        width, height = im.size
        longest = max(width, height)
        if longest == max_side:
            return image_bytes
        scale = max_side / float(longest)
        new_size = (max(1, int(width * scale)), max(1, int(height * scale)))
        resized = im.resize(new_size, resample=Image.LANCZOS)
        buffer = io.BytesIO()
        resized.save(buffer, format=im.format or "PNG")
        return buffer.getvalue()


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
    object_refs: list[bytes],
    shadow_png: Optional[bytes],
    prompt: str,
    temperature: float,
    timeout: int,
    seed: Optional[int],
) -> bytes:
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(ensure_model_path(model_name))
    last_exc: Exception | None = None
    if seed is not None:
        print(
            "[info] Seed provided but currently ignored (Gemini generation_config does not accept a seed).",
            file=sys.stderr,
        )
    for attempt in range(3):
        try:
            gen_config = {"temperature": temperature}
            parts = [
                {"mime_type": "image/png", "data": background_png},
                {"text": "Background reference (use as backdrop)."},
                {"mime_type": "image/png", "data": asset_png},
                {"text": "Product object (extract and place onto the background)."},
            ]
            if object_refs:
                parts.append(
                    {
                        "text": (
                            "Additional photos of the same product; use them to match "
                            "material transparency, gloss, luminosity, and reflectivity."
                        )
                    }
                )
                for ref_bytes in object_refs:
                    parts.append({"mime_type": "image/png", "data": ref_bytes})
            response = model.generate_content(
                parts + [prompt],
                generation_config=gen_config,
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

    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("Missing GEMINI_API_KEY in environment or .env file.", file=sys.stderr)
        sys.exit(1)

    work_items = list(discover_work(args.product))

    if args.limit:
        work_items = work_items[: args.limit]

    if not work_items:
        print("No matching product shots found to process.", file=sys.stderr)
        return

    prompt = load_prompt(args.prompt_file)
    if args.glass_opacity is not None:
        prompt += (
            "\nGlass opacity reference: match the real object's glass opacity at "
            f"{args.glass_opacity:.2f} (0 = fully transparent, 1 = fully opaque). "
            "Preserve realistic transparency, refractions, reflections, and visible background consistent with this opacity."
        )
    print("[prompt being sent to Gemini]")
    print(prompt)
    print("[/prompt]")

    if args.dry_run:
        for (
            product_name,
            asset_path,
            background_path,
            object_refs,
            shadow_path,
        ) in work_items:
            print(
                "[dry-run] "
                f"{product_name}: target={asset_path} background={background_path} "
                f"object_refs={len(object_refs)} shadow_ref={shadow_path or 'none'} "
                f"results={args.results}"
            )
        return

    input_max_side = args.max_side
    output_max_side = args.max_side
    if args.resolution:
        target_side = RESOLUTION_MAP[args.resolution]
        input_max_side = target_side
        output_max_side = target_side

    max_object_refs = max(0, args.max_object_refs)
    results_count = args.results

    for (
        product_name,
        asset_path,
        background_path,
        object_refs,
        shadow_path,
    ) in work_items:
        opacity_note = (
            f", glass_opacity={args.glass_opacity:.2f}"
            if args.glass_opacity is not None
            else ""
        )
        print(
            f"[gemini] {product_name}: {asset_path.name} with {background_path.name} "
            f"({len(object_refs)} object refs, shadow: disabled{opacity_note}, results={results_count})"
        )
        out_dir = args.output / product_name
        out_dir.mkdir(parents=True, exist_ok=True)
        asset_dump_dir = ASSET_DUMP_ROOT / product_name
        asset_dump_dir.mkdir(parents=True, exist_ok=True)

        run_targets = []
        for run_idx in range(results_count):
            out_name = (
                f"{product_name}.png"
                if results_count == 1
                else f"{product_name}_{run_idx + 1}.png"
            )
            out_path = out_dir / out_name
            if out_path.exists():
                print(f"  ↷ Skip result {run_idx + 1}/{results_count}: {out_path} already exists")
                continue
            run_targets.append((run_idx + 1, out_path))

        if not run_targets:
            print("  ↷ Skip: all requested results already exist")
            continue

        asset_png, asset_meta = to_png_bytes_with_meta(asset_path, input_max_side)
        asset_png_path = asset_dump_dir / "asset.png"
        asset_png_path.write_bytes(asset_png)
        print(f"  cached converted asset -> {asset_png_path}")

        background_png, background_meta = to_png_bytes_with_meta(
            background_path, input_max_side
        )
        shadow_png = None  # Shadows disabled currently
        object_ref_paths = object_refs[:max_object_refs] if max_object_refs else []
        object_ref_payloads = []
        for ref_path in object_ref_paths:
            ref_bytes, ref_meta = to_png_bytes_with_meta(ref_path, input_max_side)
            object_ref_payloads.append((ref_path, ref_bytes, ref_meta))

        print("  submitting to Gemini with:")
        log_image_submission("background", background_path, background_meta)
        log_image_submission("asset", asset_path, asset_meta)
        if object_ref_payloads:
            for idx, (ref_path, _, ref_meta) in enumerate(object_ref_payloads, start=1):
                log_image_submission(f"object_ref_{idx}", ref_path, ref_meta)
        else:
            print("    object refs: none")

        for run_idx, out_path in run_targets:
            print(f"  → Request {run_idx}/{results_count}")
            try:
                output_bytes = call_gemini(
                    api_key=api_key,
                    model_name=args.model,
                    product_name=product_name,
                    background_png=background_png,
                    asset_png=asset_png,
                    object_refs=[ref_bytes for _, ref_bytes, _ in object_ref_payloads],
                    shadow_png=shadow_png,
                    prompt=prompt,
                    temperature=args.temperature,
                    timeout=args.timeout,
                    seed=args.seed,
                )
            except Exception as exc:  # pylint: disable=broad-except
                print(f"    ✖ Failed: {exc}", file=sys.stderr)
                continue

            output_bytes = resize_output_if_needed(output_bytes, output_max_side)
            out_path.write_bytes(output_bytes)
            print(f"    ✔ Saved -> {out_path}")


if __name__ == "__main__":
    main()
