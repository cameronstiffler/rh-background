"""
Gemini 3 background replacement helper.
Reads product shots from test_assets/<product>/, pairs them with reference
backgrounds in ref_background/<product>/, and sends them to the Gemini API
(or Vertex AI) to generate composited images saved into output/<product>/.

Unified Google Gen AI SDK (`from google import genai`) supports both the public
Gemini API and Vertex AI via a single client interface. Backend is selected by
`GOOGLE_GENAI_USE_VERTEXAI`:
  - `GOOGLE_GENAI_USE_VERTEXAI=true` (default if unset): Vertex AI with
    `GOOGLE_CLOUD_PROJECT` and `GOOGLE_CLOUD_LOCATION="global"` — required for
    Gemini 3 Pro Image Preview.
  - `GOOGLE_GENAI_USE_VERTEXAI=false`: Public Gemini API using `GOOGLE_API_KEY`
    (or `GEMINI_API_KEY`).

MODEL_ID default: "gemini-3-pro-image-preview".

Variable override priority ordering (CLI arguments override all others):
  1. CLI arguments override
  2. .env environment variables (e.g. GEMINI_MODEL) override
  3. Hardcoded defaults ("gemini-3-pro-image-preview")

Notes:
  - Gemini 3 Pro Image (Preview) is available on Vertex AI at the global endpoint.
  - For public Gemini API, use publicly available image-capable model IDs.
"""
from __future__ import annotations
import argparse
import base64
import io
import os
import re
import sys
from pathlib import Path
from typing import Iterator, Optional

from dotenv import load_dotenv
from PIL import Image

# --- Google Gen AI (unified) ---
from google import genai
from google.genai import types

# Constants / paths
ASSETS_ROOT = Path("test_assets")
BACKGROUND_ROOT = Path("ref_background")
OUTPUT_ROOT = Path("output")
OBJECT_REFERENCE_ROOT = Path("ref_objects")
ASSET_DUMP_ROOT = Path("png")
PROMPTS_ROOT = Path("prompts")
DEFAULT_PROMPT_PATH = PROMPTS_ROOT / "default_prompt_PID0.md"

ASSET_PREFERRED_EXTENSIONS = {".tif", ".tiff"}
ASSET_FALLBACK_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
BACKGROUND_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".webp"}
OBJECT_REFERENCE_EXTENSIONS = BACKGROUND_EXTENSIONS

RESOLUTION_MAP = {"1K": 1024, "2K": 2048, "4K": 4096}
DEFAULT_MAX_OBJECT_REFS = 4

MODEL_ID = "gemini-3-pro-image-preview"

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

# --- CLI ---
def parse_args() -> argparse.Namespace:
    # Ensure .env values are loaded for defaults.
    load_dotenv()
    env_resolution = os.getenv("RESOLUTION")
    resolution_default = (env_resolution.strip().upper() if env_resolution else "4K")
    if resolution_default not in RESOLUTION_MAP:
        resolution_default = "4K"

    env_temp_raw = os.getenv("GEMINI_TEMPERATURE", os.getenv("TEMPERATURE"))
    try:
        env_temp_val = float(env_temp_raw) if env_temp_raw is not None else 0.25
    except ValueError:
        env_temp_val = 0.25

    parser = argparse.ArgumentParser(
        description="Generate background replacements with the Gemini 3 API or Vertex AI."
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
        "--pid",
        type=positive_int,
        help="Select prompt by PID number (looks for prompts/*PID#.md). Cannot be used with --prompt-file.",
    )
    parser.add_argument(
        "--max-side",
        type=int,
        default=int(os.getenv("MAX_SIDE", "4096")),
        help="Max long-edge pixels sent to the model (0 disables; default 4096). Overridden by --resolution.",
    )
    parser.add_argument(
        "--model",
        default=os.getenv("GEMINI_MODEL", MODEL_ID),
        help="Model name/id. Vertex: e.g., 'gemini-3-pro-image-preview'. Public: e.g., 'models/gemini-2.5-flash-image'.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=env_temp_val,
        help="Generation temperature (defaults from GEMINI_TEMPERATURE/TEMPERATURE).",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=420,
        help="Request timeout in seconds per call.",
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
        help="Optional deterministic seed (may be ignored by backend).",
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
        help="Target long-edge resolution for inputs and final output (overrides --max-side).",
    )
    parser.add_argument(
        "--glass-opacity", "--glass_opacity",
        dest="glass_opacity",
        type=glass_opacity_value,
        default=None,
        help="Optional glass opacity hint (0.0 transparent to 1.0 opaque); appends guidance to the prompt.",
    )
    parser.add_argument(
        "--results",
        type=positive_int,
        default=1,
        help="Number of results to generate per product.",
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
        help="List the files that would be processed without calling the model.",
    )
    return parser.parse_args()

# --- Prompt helpers ---
def load_prompt(prompt_file: Optional[Path]) -> str:
    path = prompt_file or DEFAULT_PROMPT_PATH
    try:
        return path.read_text(encoding="utf-8").strip()
    except FileNotFoundError:
        if prompt_file:
            raise
        print(f"[warn] Default prompt file missing at {path}; using built-in fallback.", file=sys.stderr)
        return DEFAULT_PROMPT_FALLBACK


def prompt_id_from_path(path: Path) -> str:
    match = re.search(r"(pid\d+)", path.stem, flags=re.IGNORECASE)
    if match:
        return match.group(1).upper()
    return "PIDX"


def find_prompt_by_pid(pid: int) -> Path:
    if not PROMPTS_ROOT.exists():
        raise FileNotFoundError(f"Prompts folder missing at {PROMPTS_ROOT}")
    pid_token = f"pid{pid}".lower()
    for candidate in sorted(PROMPTS_ROOT.glob("*.md")):
        if pid_token in candidate.stem.lower():
            return candidate
    raise FileNotFoundError(f"No prompt file found for PID{pid} in {PROMPTS_ROOT}")

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

# --- Image helpers ---


def format_bytes(num_bytes: int) -> str:
    value = float(num_bytes)
    for unit in ("B", "KB", "MB", "GB"):
        if value < 1024 or unit == "GB":
            return f"{value:.0f} {unit}" if unit == "B" else f"{value:.2f} {unit}"
        value /= 1024
    return f"{value:.2f} GB"


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
    payload_size = format_bytes(meta.get("payload_bytes", 0))
    print(
        f" {label}: {path} -> {final_w}x{final_h}px PNG "
        f"(upload ~{payload_size}; source {src_w}x{src_h} {meta['original_format']}/{meta['original_mode']})"
    )

# --- Work discovery ---

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
        shadow_path = None  # Shadows disabled
        yield product_dir.name, target_path, background_path, object_refs, shadow_path

# --- PNG conversion ---

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
        data = buffer.getvalue()
        return data, {
            "original_size": original_size,
            "final_size": final_size,
            "original_mode": original_mode,
            "final_mode": converted.mode,
            "original_format": original_format,
            "mime": "image/png",
            "payload_bytes": len(data),
        }

def image_part_from_bytes(data: bytes) -> dict:
    encoded = base64.b64encode(data).decode("ascii")
    return {"inline_data": {"mime_type": "image/png", "data": encoded}}

# --- Output resize ---

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

# --- Response parsing ---

def normalize_inline_data(data) -> bytes:
    if isinstance(data, bytes):
        return data
    if isinstance(data, str):
        return base64.b64decode(data)
    raise TypeError(f"Unsupported inline data type: {type(data)}")


def extract_image_from_response(response) -> bytes:
    # Prefer the newer SDK shape (response.parts), then fall back.
    parts = getattr(response, "parts", None)
    if parts:
        for part in parts:
            inline = getattr(part, "inline_data", None)
            if inline and getattr(inline, "data", None):
                return normalize_inline_data(inline.data)
    # Fallback to candidates->content->parts
    for candidate in getattr(response, "candidates", []):
        content = getattr(candidate, "content", None)
        if not content:
            continue
        for part in getattr(content, "parts", []) or []:
            inline = getattr(part, "inline_data", None)
            if inline and getattr(inline, "data", None):
                return normalize_inline_data(inline.data)
    raise RuntimeError("No image returned by model response.")


def summarize_response_for_debug(response) -> str:
    summaries = []
    parts = getattr(response, "parts", None)
    if parts:
        ptypes = []
        for part in parts:
            if getattr(part, "inline_data", None):
                ptypes.append("inline_data")
            elif getattr(part, "text", None):
                ptypes.append("text")
            else:
                ptypes.append(type(part).__name__)
        summaries.append(f"response.parts={','.join(ptypes) or 'none'}")
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

# --- Backend clients ---

def _ensure_credentials() -> str:
    """Ensure GOOGLE_APPLICATION_CREDENTIALS is set (Vertex AI only)."""
    creds = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
    if not creds:
        # default to local vertex.json next to the script
        creds = os.path.abspath("vertex.json")
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = creds
    if not os.path.isfile(creds):
        raise FileNotFoundError(
            f"GOOGLE_APPLICATION_CREDENTIALS points to '{creds}', which does not exist."
        )
    return creds


def _create_client() -> genai.Client:
    """Create a Google Gen AI client for Vertex AI or public Gemini API.
    Controlled by GOOGLE_GENAI_USE_VERTEXAI (default true).
    """
    use_vertex = use_vertex_backend()
    if use_vertex:
        project_id = os.environ.get("GOOGLE_CLOUD_PROJECT")
        location = os.environ.get("GOOGLE_CLOUD_LOCATION", "global")
        if not project_id:
            raise RuntimeError("Missing GOOGLE_CLOUD_PROJECT for Vertex AI usage.")
        _ensure_credentials()
        os.environ.setdefault("GOOGLE_GENAI_USE_VERTEXAI", "true")
        return genai.Client(vertexai=True, project=project_id, location=location)
    else:
        api_key = (os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY"))
        if not api_key:
            raise RuntimeError("Missing GOOGLE_API_KEY or GEMINI_API_KEY for Gemini API usage.")
        return genai.Client(api_key=api_key)

# --- Model id normalization ---

def normalize_model_id(model: str, use_vertex: bool) -> str:
    if use_vertex:
        # Vertex expects bare id
        return model[7:] if model.startswith("models/") else model
    else:
        # Public API expects models/... style
        return model if model.startswith("models/") else f"models/{model}"


def use_vertex_backend() -> bool:
    return os.environ.get("GOOGLE_GENAI_USE_VERTEXAI", "true").lower() != "false"

# --- Model call ---

def call_gemini(
    api_key: str,            # kept for compatibility; ignored when client is provided
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
    client: Optional[genai.Client] = None,
    aspect_ratio: str = "16:9",
    image_size: str = "2K",
) -> bytes:
    if client is None:
        raise RuntimeError("Gen AI client not initialized.")

    use_vertex = use_vertex_backend()
    model_id = normalize_model_id(model_name, use_vertex)

    # NOTE: Seed is not currently supported in GenerateContentConfig; we print a note only.
    if seed is not None:
        print("[info] Seed provided but may be ignored by backend.", file=sys.stderr)

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
                image_part_from_bytes(background_png),
                {"text": "Background reference (use as backdrop)."},
                image_part_from_bytes(asset_png),
                {"text": "Product object (extract and place onto the background)."},
            ]
            if object_refs:
                parts.append({
                    "text": (
                        "Additional photos of the same product; use them to match "
                        "material transparency, gloss, luminosity, and reflectivity."
                    )
                })
                for ref_bytes in object_refs:
                    parts.append(image_part_from_bytes(ref_bytes))
            parts.append(prompt)

            response = client.models.generate_content(
                model=model_id,
                contents=parts,
                config=types.GenerateContentConfig(
                    temperature=temperature,
                    response_modalities=["TEXT", "IMAGE"],
                    image_config={"aspect_ratio": aspect_ratio, "image_size": image_size},
                ),
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
                import time
                time.sleep(2 * (attempt + 1))
                continue
            raise

    raise RuntimeError(f"Gen AI request failed for {product_name}: {last_exc}")

# --- Main ---

def main() -> None:
    args = parse_args()

    # Create client for either Vertex AI or public Gemini API based on env
    try:
        client = _create_client()
    except Exception as e:
        print(str(e), file=sys.stderr)
        sys.exit(1)

    use_vertex = use_vertex_backend()
    model_tag = "MOD-VERTEX" if use_vertex else "MOD-G3PIP"

    work_items = list(discover_work(args.product))
    if args.limit:
        work_items = work_items[: args.limit]
    if not work_items:
        print("No matching product shots found to process.", file=sys.stderr)
        return

    if args.pid is not None and args.prompt_file:
        print("Cannot use --pid together with --prompt-file. Pick one.", file=sys.stderr)
        sys.exit(1)

    if args.pid is not None:
        try:
            prompt_path = find_prompt_by_pid(args.pid)
        except FileNotFoundError as exc:
            print(str(exc), file=sys.stderr)
            sys.exit(1)
    else:
        prompt_path = args.prompt_file or DEFAULT_PROMPT_PATH

    prompt = load_prompt(prompt_path)
    prompt_id = prompt_id_from_path(prompt_path)
    if args.glass_opacity is not None:
        prompt += (
            "\nGlass opacity reference: match the real object's glass opacity at "
            f"{args.glass_opacity:.2f} (0 = fully transparent, 1 = fully opaque). "
            "Preserve realistic transparency, refractions, reflections, and visible background consistent with this opacity."
        )
    print("[prompt being sent]")
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
        object_ref_paths,
        shadow_path,
    ) in work_items:
        opacity_note = (
            f", glass_opacity={args.glass_opacity:.2f}"
            if args.glass_opacity is not None
            else ""
        )
        print(
            f"[genai] {product_name}: {asset_path.name} with {background_path.name} "
            f"(prompt {prompt_id}, model {model_tag}; {len(object_ref_paths)} object refs, shadow: disabled{opacity_note}, results={results_count})"
        )
        out_dir = args.output / product_name
        out_dir.mkdir(parents=True, exist_ok=True)
        asset_dump_dir = ASSET_DUMP_ROOT / product_name
        asset_dump_dir.mkdir(parents=True, exist_ok=True)

        run_targets = []
        base_out_name = f"{product_name}_{prompt_id}_{model_tag}"
        for run_idx in range(results_count):
            out_name = f"{base_out_name}_R{run_idx + 1}.png"
            out_path = out_dir / out_name
            if out_path.exists():
                print(f" ↷ Skip result {run_idx + 1}/{results_count}: {out_path} already exists")
                continue
            run_targets.append((run_idx + 1, out_path))
        if not run_targets:
            print(" ↷ Skip: all requested results already exist")
            continue

        asset_png, asset_meta = to_png_bytes_with_meta(asset_path, input_max_side)
        asset_png_path = asset_dump_dir / "asset.png"
        asset_png_path.write_bytes(asset_png)
        print(f" cached converted asset -> {asset_png_path}")

        background_png, background_meta = to_png_bytes_with_meta(background_path, input_max_side)
        shadow_png = None  # Shadows disabled currently

        # Prepare object refs
        object_ref_payloads = []
        for ref_path in object_ref_paths[:max_object_refs] if max_object_refs else []:
            ref_bytes, ref_meta = to_png_bytes_with_meta(ref_path, input_max_side)
            object_ref_payloads.append((ref_path, ref_bytes, ref_meta))

        print(" submitting to Gen AI with:")
        log_image_submission("background", background_path, background_meta)
        log_image_submission("asset", asset_path, asset_meta)
        if object_ref_payloads:
            for idx, (ref_path, _, ref_meta) in enumerate(object_ref_payloads, start=1):
                log_image_submission(f"object_ref_{idx}", ref_path, ref_meta)
        else:
            print(" object refs: none")

        # Choose image_size from resolution
        image_size = args.resolution if args.resolution in RESOLUTION_MAP else "2K"
        aspect_ratio = "16:9"  # default; adjust if needed

        for run_idx, out_path in run_targets:
            print(f" → Request {run_idx}/{results_count}")
            try:
                output_bytes = call_gemini(
                    api_key=os.getenv("GEMINI_API_KEY", ""),  # ignored when client is provided
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
                    client=client,
                    aspect_ratio=aspect_ratio,
                    image_size=image_size,
                )
            except Exception as exc:  # pylint: disable=broad-except
                print(f" ✖ Failed: {exc}", file=sys.stderr)
                continue

            output_bytes = resize_output_if_needed(output_bytes, output_max_side)
            out_path.write_bytes(output_bytes)
            print(f" ✔ Saved -> {out_path}")

if __name__ == "__main__":
    main()
