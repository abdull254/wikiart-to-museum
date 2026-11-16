"""
Keep only the Rijksmuseum's paintings collection (object numbers starting with "SK-").

The Search API's type=painting also returns some drawings (RP), Asian art (AK)
and decorative objects (BK, NG, ...). This removes them without deleting any files.

Writes:
  data/rijks/metadata_paintings.csv   paintings kept (use this from now on)
  data/rijks/excluded_not_sk.csv      everything removed, with the reason

Usage:
  python filter_rijks.py
"""

import csv
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

BASE = Path(__file__).resolve().parent / "data" / "rijks"
META = BASE / "metadata.csv"
KEPT = BASE / "metadata_paintings.csv"
DROPPED = BASE / "excluded_not_sk.csv"
LOG = BASE / "harvest_log.json"
IMAGES = BASE / "images_256_squash"

PREFIX_NAMES = {
    "RP": "drawing or print (print room)",
    "AK": "Asian art",
    "BK": "decorative object",
    "NG": "historical object",
    "KOG": "Royal Antiquarian Society object",
}


def main():
    with META.open(encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        fields = reader.fieldnames
        rows = list(reader)

    kept, dropped = [], []
    for r in rows:
        prefix = r["object_number"].split("-")[0]
        if prefix == "SK":
            kept.append(r)
        else:
            dropped.append({**r, "reason_excluded":
                            f"not in paintings collection ({prefix}: "
                            f"{PREFIX_NAMES.get(prefix, 'other collection')})"})

    with KEPT.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(kept)
    with DROPPED.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields + ["reason_excluded"])
        w.writeheader()
        w.writerows(dropped)

    with_image = sum(1 for r in kept if (IMAGES / f"{r['rijks_id']}.jpg").exists())
    by_prefix = Counter(r["object_number"].split("-")[0] for r in dropped)

    print(f"Paintings kept (SK): {len(kept)} of {len(rows)}  -> {KEPT.name}")
    print(f"  with a 256 px image on disk: {with_image}")
    print(f"Removed: {len(dropped)}  -> {DROPPED.name}")
    for p, c in by_prefix.most_common():
        print(f"  {c:>5}  {p}  ({PREFIX_NAMES.get(p, 'other collection')})")

    history = json.loads(LOG.read_text()) if LOG.exists() else []
    history.append({"step": "filter_sk", "kept": len(kept), "removed": len(dropped),
                    "removed_by_prefix": dict(by_prefix), "kept_with_image": with_image,
                    "at": datetime.now(timezone.utc).isoformat(timespec="seconds")})
    LOG.write_text(json.dumps(history, indent=2))


if __name__ == "__main__":
    main()
