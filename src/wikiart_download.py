"""
Download a WikiArt release that has artist names, for an artist-disjoint split.

Default source: Hugging Face "Artificio/WikiArt_Full" (about 103,000 images, ~3.3 GB),
which stores artist, title, date, style and genre as text.
(huggan/wikiart was tested and rejected: 58% of its images are "Unknown Artist".)

Steps
  probe     - read a random sample only. Reports: share of images with a named artist,
              number of artists, styles, and image sizes. Run this FIRST.
  download  - stream the whole dataset, save every image (long side at most --size),
              and write data/wikiart/metadata.csv. All styles are kept; choose later.

Usage (inside your venv):
  python wikiart_download.py probe --n 3000
  python wikiart_download.py download --size 512
  python wikiart_download.py probe --repo huggan/wikiart     # any other repo

The download step can be stopped and re-run: images already saved are skipped.
License: WikiArt images are for non-commercial research use only. Do not redistribute them;
share the metadata CSV (filenames, artists, styles) instead.
"""

import argparse
import csv
import json
import statistics
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from datasets import load_dataset
from tqdm import tqdm

DEFAULT_REPO = "Artificio/WikiArt_Full"
TEXT_FIELDS = ["filename", "title", "artist", "date", "style", "genre"]   # kept if present

OUT = Path(__file__).resolve().parent / "data" / "wikiart"
IMAGES = OUT / "images"
META_FILE = OUT / "metadata.csv"
LOG_FILE = OUT / "download_log.json"


def open_stream(repo):
    ds = load_dataset(repo, split="train", streaming=True)
    feats = ds.features or {}
    # Drop large feature-vector columns we don't need (embeddings etc.)
    keep = {"image", *TEXT_FIELDS}
    drop = [k for k in feats if k not in keep]
    if drop:
        ds = ds.remove_columns(drop)
    # Class-label columns (integers) need their names; plain text columns don't
    names = {k: feats[k].names for k in TEXT_FIELDS
             if k in feats and hasattr(feats[k], "names")}
    fields = [k for k in TEXT_FIELDS if k in feats] or TEXT_FIELDS
    return ds, names, fields


def value(ex, names, key):
    v = ex.get(key, "")
    if key in names and isinstance(v, int):
        return names[key][v]
    return "" if v is None else str(v)


def is_unknown(artist):
    a = artist.strip().lower().replace("_", " ").replace("-", " ")
    return not a or a.startswith("unknown") or a in {"anonymous", "none", "nan"}


def step_probe(repo, n):
    ds, names, fields = open_stream(repo)
    print(f"Repo: {repo}\nFields: {', '.join(fields)}")
    ds = ds.shuffle(seed=0, buffer_size=2000)
    styles, artists = Counter(), Counter()
    long_sides, unknown = [], 0
    for ex in tqdm(ds.take(n), total=n, desc="probe"):
        a = value(ex, names, "artist")
        artists[a] += 1
        unknown += is_unknown(a)
        styles[value(ex, names, "style")] += 1
        img = ex.get("image")
        if img is not None:
            long_sides.append(max(img.size))
    print(f"\nChecked {n} randomly mixed images (a rough estimate for the whole dataset).")
    print(f"Images WITH a named artist:     {n - unknown} ({(n - unknown) / n:.0%})")
    print(f"Images with unknown/no artist:  {unknown} ({unknown / n:.0%})")
    print(f"Different named artists seen:   {sum(1 for a in artists if not is_unknown(a))}")
    print("Most common artists:")
    for a, c in artists.most_common(8):
        print(f"  {c:>5}  {a or '(empty)'}")
    if long_sides:
        print(f"\nImage size (long side, px): median {int(statistics.median(long_sides))}, "
              f"min {min(long_sides)}, max {max(long_sides)}; "
              f"{sum(s >= 512 for s in long_sides) / len(long_sides):.0%} are at least 512 px")
    print("\nStyles in this sample:")
    for s, c in styles.most_common():
        print(f"  {c:>5}  {s}")
    log({"step": "probe", "repo": repo, "n": n,
         "named_artist_share": round((n - unknown) / n, 3),
         "median_long_side": int(statistics.median(long_sides)) if long_sides else None})


def step_download(repo, size):
    IMAGES.mkdir(parents=True, exist_ok=True)
    ds, names, fields = open_stream(repo)
    cols = ["wikiart_idx", *fields, "width", "height", "orig_width", "orig_height"]
    new_file = not META_FILE.exists()
    done = set()
    if not new_file:
        with META_FILE.open(encoding="utf-8") as fh:
            done = {r["wikiart_idx"] for r in csv.DictReader(fh)}
    saved = 0
    with META_FILE.open("a", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        if new_file:
            w.writeheader()
        for idx, ex in enumerate(tqdm(ds, desc="wikiart")):
            if str(idx) in done:
                continue
            img = ex["image"].convert("RGB")
            ow, oh = img.size
            img.thumbnail((size, size))    # long side at most `size`; never enlarges
            img.save(IMAGES / f"{idx}.jpg", "JPEG", quality=90)
            row = {"wikiart_idx": idx, "width": img.width, "height": img.height,
                   "orig_width": ow, "orig_height": oh}
            row.update({k: value(ex, names, k) for k in fields})
            w.writerow(row)
            saved += 1
            if saved % 500 == 0:
                fh.flush()
    log({"step": "download", "repo": repo, "size_long_side": size, "saved_this_run": saved})
    print(f"Saved {saved} new images to {IMAGES}")


def log(entry):
    OUT.mkdir(parents=True, exist_ok=True)
    entry["at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    history = json.loads(LOG_FILE.read_text()) if LOG_FILE.exists() else []
    history.append(entry)
    LOG_FILE.write_text(json.dumps(history, indent=2))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("step", choices=["probe", "download"])
    ap.add_argument("--repo", default=DEFAULT_REPO, help="Hugging Face dataset id")
    ap.add_argument("--n", type=int, default=3000, help="images to read in probe mode")
    ap.add_argument("--size", type=int, default=512, help="long side of saved images in pixels")
    a = ap.parse_args()
    if a.step == "probe":
        step_probe(a.repo, a.n)
    else:
        step_download(a.repo, a.size)