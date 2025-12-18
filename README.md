# Gemini 3 Background Replacement

Simple helper to send product shots and reference backgrounds to the Gemini 3 API and save the composited results.

## Setup
- Python 3.11+ recommended.
- Copy `.env.example` to `.env` and fill in `GEMINI_API_KEY`; adjust `GEMINI_MODEL` or other values if desired.
- Install deps:
  ```bash
  python -m venv .venv
  source .venv/bin/activate
  pip install -r requirements.txt
  ```
- Drop your local assets into `test_assets/<product>/`, matching backgrounds into `ref_background/<product>/`, and any extra product-only photos into `ref_objects/<product>/` (one subfolder per product, same name as in `test_assets/`). These folders are `.gitignore`d to keep large image files out of git.
- The default instructions live in `prompts/default_prompt_PID0.md`; edit that file to change the baseline prompt, or provide your own via `--prompt-file`.

## Running
For each product folder:
- Source asset: the first `.tif/.tiff` in `test_assets/<product>/` (converted to `asset.png` stored in `png/<product>/` for reference); if no TIFF is present, the first other image (`.png/.jpg/.jpeg/.webp/.bmp`) is used.
- Background: the first image in `ref_background/<product>/` (any common image type).
- Shadow reference: currently ignored (shadows disabled).
- Reference images: ref_objects/<product>/<anyname>.(`.png/.jpg/.jpeg/.webp/.bmp`)
- Output: saved to `output/<product>/<product>.png`; the converted asset copy lives in `png/<product>/asset.png`.

Example run for the chandelier assets:
```bash
python background_replace.py --product "Glass Globe Chandelier"
```
```bash
python python3 background_replace.py --product "Glass Globe Chandelier" --prompt-file alternate_prompt_2.md --results 3
```

Useful flags:
- `--dry-run` shows which files would be sent without calling Gemini.
- `--limit 2` processes only the first two assets.
- `--pid 2` selects the prompt file whose name contains `PID2` under `prompts/` (e.g., `prompts/alternate_prompt_PID2.md`). `--prompt-file` and `--pid` are mutually exclusive. Output filenames include the `PID#` suffix to record which prompt was used.
- `--prompt-file prompts/custom_prompt_PID#.md` swaps in your own instructions (default comes from `prompts/default_prompt_PID0.md`). Output filenames include the `PID#` suffix to record which prompt was used.
- `--model models/gemini-3-pro-image-preview` overrides the model if you change it later (defaults to `GEMINI_MODEL` in `.env`).
- `--temperature 0.25` controls randomness; defaults to `GEMINI_TEMPERATURE` / `TEMPERATURE` in `.env` when set.
- `--seed 123` accepted for forward-compatibility (or set `GEMINI_SEED` / `SEED` in `.env`), but currently ignored because Gemini generation_config does not support seeds.
- `--resolution 4K` (or `2K`/`1K`) sets the long edge for both upload and output; overrides `--max-side`. Defaults to `4K` (set `RESOLUTION` in `.env` to change).
- `--max-side 4096` controls downscaling before upload (0 disables; default 4096). Overridden by `--resolution`.
- `--max-object-refs 4` limits how many photos from `ref_objects/<product>/` are sent with each request (0 disables). Place these under `ref_objects/<product>/` (matching the product folder name in `test_assets/`); they help Gemini match transparency, reflectivity, and fine material details of the product.
- `--glass-opacity 0.35` optionally appends a glass-opacity hint (0.0 fully transparent to 1.0 fully opaque) so Gemini renders glass with the specified transparency.
- `--results 3` requests multiple outputs per product; files are saved as `<product>.png` when requesting 1 result, otherwise `<product>_1.png`, `<product>_2.png`, etc.

Each generated file is saved as `<product>.png` in the output folder. If a product folder has no `.tif` asset or matching reference background, the script will skip it and print a note.

Notes:
- The target and background are converted to PNG and downscaled if needed before sending to Gemini; the model response is expected to contain inline image data, which is written to disk.
- If the model refuses to return image data, try an image-capable model such as `models/gemini-1.5-flash-002` via `--model` or `GEMINI_MODEL`.
