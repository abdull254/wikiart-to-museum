"""
Pack exactly what training needs into one zip for Google Colab.

Includes: train.py, data/final/ (split, museum test set, style list), the WikiArt images
used by the split, and the three museum image versions (256 squash, 256 crop, 512 original).
Leaves out everything else (unused WikiArt styles, caches, raw metadata).

Usage:
  python pack_for_colab.py
Then upload colab_data.zip to the top of your Google Drive ("My Drive").
"""

import csv
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
OUT = ROOT / "colab_data.zip"


def main():
    files = [ROOT / "train.py"]
    files += [p for p in (DATA / "final").glob("*.csv")]
    with (DATA / "final" / "wikiart_split.csv").open(encoding="utf-8") as fh:
        files += [DATA / r["image"] for r in csv.DictReader(fh)]
    with (DATA / "final" / "museum_test.csv").open(encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            for col in ("image", "image_crop", "image_512"):
                files.append(DATA / r[col])

    missing = [f for f in files if not f.exists()]
    present = [f for f in files if f.exists()]
    print(f"Packing {len(present)} files ({len(missing)} missing)...")
    with zipfile.ZipFile(OUT, "w", compression=zipfile.ZIP_STORED) as z:   # JPEGs don't shrink
        for i, f in enumerate(present, 1):
            z.write(f, f.relative_to(ROOT))
            if i % 5000 == 0:
                print(f"  {i}/{len(present)}")
    print(f"Done: {OUT}  ({OUT.stat().st_size / 1e9:.2f} GB)")
    if missing:
        print("Missing (first 5):", *[str(m.relative_to(ROOT)) for m in missing[:5]], sep="\n  ")
    print("Next: upload colab_data.zip to 'My Drive' in Google Drive.")


if __name__ == "__main__":
    main()
