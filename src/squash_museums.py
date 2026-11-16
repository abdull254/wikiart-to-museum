"""
Make 256 x 256 copies of the museum images so they match WikiArt (Artificio/WikiArt_Full),
whose images are all squashed to 256 x 256.

For each museum it writes two versions next to the originals (originals are not changed):
  images_256_squash/   whole painting resized to 256 x 256, shape distorted (matches WikiArt)
  images_256_crop/     shape kept, center square cut out, then resized to 256 x 256
                       (for the center-crop vs full-frame ablation)

Usage (inside your venv):
  python squash_museums.py                 # both museums, both versions
  python squash_museums.py --museum rijks  # one museum only
  python squash_museums.py --mode squash   # one version only

Re-running is safe: images already made are skipped.
"""

import argparse
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image, ImageOps
from tqdm import tqdm

BASE = Path(__file__).resolve().parent / "data"
MUSEUMS = {"rijks": BASE / "rijks" / "images", "met": BASE / "met" / "images"}
SIZE = 256
QUALITY = 90                  # same JPEG quality the harvest scripts used
RESAMPLE = Image.LANCZOS      # high-quality downscaling filter


def make(src, dst, mode):
    if dst.exists() and dst.stat().st_size > 0:
        return "skipped"
    try:
        img = Image.open(src).convert("RGB")
        if mode == "squash":
            out = img.resize((SIZE, SIZE), RESAMPLE)                   # ignore aspect ratio
        else:
            out = ImageOps.fit(img, (SIZE, SIZE), RESAMPLE, centering=(0.5, 0.5))  # crop center
        out.save(dst, "JPEG", quality=QUALITY)
        return "ok"
    except Exception as e:
        return f"error: {type(e).__name__}"


def run(museum, mode, workers):
    src_dir = MUSEUMS[museum]
    if not src_dir.exists():
        print(f"{museum}: no images folder at {src_dir}, skipping")
        return {}
    dst_dir = src_dir.parent / f"images_{SIZE}_{mode}"
    dst_dir.mkdir(exist_ok=True)
    files = sorted(src_dir.glob("*.jpg"))
    results = {}
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futures = [ex.submit(make, f, dst_dir / f.name, mode) for f in files]
        for fut in tqdm(as_completed(futures), total=len(futures), desc=f"{museum} {mode}"):
            r = fut.result()
            results[r] = results.get(r, 0) + 1
    print(f"{museum} {mode}: {results} -> {dst_dir}")
    return results


def log(entry):
    path = BASE / "preprocessing_log.json"
    entry["at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    history = json.loads(path.read_text()) if path.exists() else []
    history.append(entry)
    path.write_text(json.dumps(history, indent=2))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--museum", choices=["rijks", "met", "both"], default="both")
    ap.add_argument("--mode", choices=["squash", "crop", "both"], default="both")
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    museums = ["rijks", "met"] if a.museum == "both" else [a.museum]
    modes = ["squash", "crop"] if a.mode == "both" else [a.mode]
    for m in museums:
        for mode in modes:
            res = run(m, mode, a.workers)
            if res:
                log({"step": "resize", "museum": m, "mode": mode, "size": SIZE,
                     "filter": "LANCZOS", "jpeg_quality": QUALITY, "results": res})
