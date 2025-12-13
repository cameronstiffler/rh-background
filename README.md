# Gemini 3 Background Replacement

Simple helper to send product shots and reference backgrounds to the Gemini 3 API and save the composited results.

## Setup
- Python 3.11+ recommended.
- Add your API credentials to `.env` (already present): `GEMINI_API_KEY` and optional `GEMINI_MODEL` (defaults to `models/gemini-3-pro-image-preview`).
- Install deps:
  ```bash
  python -m venv .venv
  source .venv/bin/activate
  pip install -r requirements.txt
  ```
- Drop your local assets into `test_assets/<product>/` and matching backgrounds into `ref_background/<product>/`. These folders are `.gitignore`d to keep large image files out of git.

## Running
For each product folder:
- Source asset: the first `.tif/.tiff` in `test_assets/<product>/` (converted to `asset.png`); if no TIFF is present, the first other image (`.png/.jpg/.jpeg/.webp/.bmp`) is used.
- Background: the first image in `ref_background/<product>/` (any common image type).
- Shadow reference: currently ignored (shadows disabled).
- Output: saved to `output/<product>/<product>.png` (and the converted `asset.png` for reference).

Example run for the chandelier assets:
```bash
python background_replace.py --product "Glass Globe Chandelier"
```

Useful flags:
- `--dry-run` shows which files would be sent without calling Gemini.
- `--limit 2` processes only the first two assets.
- `--prompt-file custom_prompt.txt` swaps in your own instructions.
- `--model models/gemini-3-pro-image-preview` overrides the model if you change it later (defaults to `GEMINI_MODEL` in `.env`).
- `--max-side 4096` controls downscaling before upload (0 disables; default 4096).

Each generated file is saved as `<product>.png` in the output folder. If a product folder has no `.tif` asset or matching reference background, the script will skip it and print a note.

Notes:
- The target and background are converted to PNG and downscaled if needed before sending to Gemini; the model response is expected to contain inline image data, which is written to disk.
- If the model refuses to return image data, try an image-capable model such as `models/gemini-1.5-flash-002` via `--model` or `GEMINI_MODEL`.
