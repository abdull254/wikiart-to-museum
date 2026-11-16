"""
Harvest Met Museum paintings (metadata + images) for the WikiArt museum-photo project.
Same layout as the Rijksmuseum harvest, so both museums can be compared directly.

Steps
  ids     - list object IDs in the painting departments          -> data/met/ids.txt
  meta    - fetch each object (cached as JSON), keep public-domain paintings with an image
                                                                  -> data/met/metadata.csv
  images  - download each image and resize it to the same size as the Rijksmuseum images
                                                                  -> data/met/images/<object_id>.jpg

Usage (inside your venv):
  pip install requests tqdm pillow
  python met_harvest.py ids
  python met_harvest.py meta --limit 50     # quick test first
  python met_harvest.py meta
  python met_harvest.py images --size 512
Every step can be stopped and re-run: finished work is skipped.

API: https://metmuseum.github.io/  (no key; keep under 80 requests/second)
"""

import argparse
import csv
import io
import json
import re
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import requests
from PIL import Image
from tqdm import tqdm

API = "https://collectionapi.metmuseum.org/public/collection/v1"
# Departments that hold most of the Met's European paintings
DEPARTMENT_NAMES = ["European Paintings", "Robert Lehman Collection"]

OUT = Path(__file__).resolve().parent / "data" / "met"
CACHE = OUT / "cache"
IMAGES = OUT / "images"
IDS_FILE = OUT / "ids.txt"
META_FILE = OUT / "metadata.csv"
ALL_FILE = OUT / "all_objects.csv"     # every object fetched, before filtering
LOG_FILE = OUT / "harvest_log.json"

FIELDS = [
    "met_id", "accession_number", "title", "artist", "artist_prefix", "artist_role",
    "artist_wikidata", "artist_ulan", "artist_begin", "artist_end", "artist_nationality",
    "object_wikidata", "year_begin", "year_end", "date_text", "classification",
    "object_name", "medium", "culture", "department", "is_public_domain",
    "image_url", "image_url_small", "reason_excluded",
]

session = requests.Session()
session.headers.update({"User-Agent": "MSU-Mankato CIS631 research harvest (wikiart-museum-style)"})


# ---------- HTTP ----------

def get_json(url, tries=5):
    for attempt in range(tries):
        try:
            r = session.get(url, timeout=30)
            if r.status_code == 200:
                return r.json()
            if r.status_code == 404:
                return None
            if r.status_code in (403, 429, 500, 502, 503, 504):
                time.sleep(2 ** attempt)
                continue
            r.raise_for_status()
        except (requests.ConnectionError, requests.Timeout, ValueError):
            time.sleep(2 ** attempt)
    raise RuntimeError(f"failed after {tries} tries: {url}")


def qid(url):
    m = re.search(r"(Q\d+)", url or "")
    return m.group(1) if m else ""


def ulan(url):
    m = re.search(r"ulan/(\d+)", url or "")
    return m.group(1) if m else ""


# ---------- steps ----------

def step_ids():
    OUT.mkdir(parents=True, exist_ok=True)
    deps = get_json(f"{API}/departments")["departments"]
    chosen = [d for d in deps if any(n.lower() in d["displayName"].lower() for n in ["european paintings", "lehman"])]
    if not chosen:
        raise SystemExit("Department names not found. Available: "
                         + ", ".join(d["displayName"] for d in deps))
    ids_param = "|".join(str(d["departmentId"]) for d in chosen)
    data = get_json(f"{API}/objects?departmentIds={ids_param}")
    ids = data.get("objectIDs") or []
    IDS_FILE.write_text("\n".join(map(str, ids)) + "\n")
    log({"step": "ids", "departments": {d["displayName"]: d["departmentId"] for d in chosen},
         "ids_saved": len(ids)})
    print(f"Saved {len(ids)} object IDs from {', '.join(d['displayName'] for d in chosen)}")


def fetch(obj_id):
    path = CACHE / f"{obj_id}.json"
    if path.exists():
        return json.loads(path.read_text())
    data = get_json(f"{API}/objects/{obj_id}")
    if data is not None:
        path.write_text(json.dumps(data))
    return data


def to_row(o):
    return {
        "met_id": o.get("objectID", ""),
        "accession_number": o.get("accessionNumber", ""),
        "title": o.get("title", ""),
        "artist": o.get("artistDisplayName", ""),
        "artist_prefix": o.get("artistPrefix", ""),      # e.g. "Workshop of", "Attributed to"
        "artist_role": o.get("artistRole", ""),
        "artist_wikidata": qid(o.get("artistWikidata_URL")),
        "artist_ulan": ulan(o.get("artistULAN_URL")),
        "artist_begin": o.get("artistBeginDate", ""),
        "artist_end": o.get("artistEndDate", ""),
        "artist_nationality": o.get("artistNationality", ""),
        "object_wikidata": qid(o.get("objectWikidata_URL")),
        "year_begin": o.get("objectBeginDate", ""),
        "year_end": o.get("objectEndDate", ""),
        "date_text": o.get("objectDate", ""),
        "classification": o.get("classification", ""),
        "object_name": o.get("objectName", ""),
        "medium": o.get("medium", ""),
        "culture": o.get("culture", ""),
        "department": o.get("department", ""),
        "is_public_domain": o.get("isPublicDomain", ""),
        "image_url": o.get("primaryImage", ""),
        "image_url_small": o.get("primaryImageSmall", ""),
        "reason_excluded": "",
    }


def exclusion_reason(row):
    """Keep public-domain paintings that have an image. Everything else is recorded, not lost."""
    if row["classification"] != "Paintings":
        return f"not a painting ({row['classification'] or 'no classification'})"
    if row["is_public_domain"] is not True:
        return "not public domain"
    if not row["image_url"]:
        return "no image"
    return ""


def harvest_one(obj_id):
    try:
        o = fetch(obj_id)
        if o is None:
            return {**{f: "" for f in FIELDS}, "met_id": obj_id, "reason_excluded": "not found"}
        row = to_row(o)
        row["reason_excluded"] = exclusion_reason(row)
        return row
    except Exception as e:
        return {**{f: "" for f in FIELDS}, "met_id": obj_id,
                "reason_excluded": f"error: {type(e).__name__}: {e}"[:200]}


def step_meta(limit=None, workers=8):
    CACHE.mkdir(parents=True, exist_ok=True)
    ids = [l.strip() for l in IDS_FILE.read_text().splitlines() if l.strip()]
    if limit:
        ids = ids[:limit]
    rows = []
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = [ex.submit(harvest_one, i) for i in ids]
        for f in tqdm(as_completed(futures), total=len(futures), desc="metadata"):
            rows.append(f.result())
    rows.sort(key=lambda r: int(r["met_id"]))
    write_csv(ALL_FILE, rows)
    kept = [r for r in rows if not r["reason_excluded"]]
    write_csv(META_FILE, kept)

    reasons = Counter(r["reason_excluded"] or "kept" for r in rows)
    no_wd = sum(1 for r in kept if not r["artist_wikidata"])
    log({"step": "meta", "fetched": len(rows), "kept": len(kept),
         "reasons": dict(reasons), "kept_without_artist_wikidata": no_wd})
    print(f"Fetched {len(rows)} objects, kept {len(kept)} paintings -> {META_FILE}")
    for reason, c in reasons.most_common():
        print(f"  {c:>5}  {reason}")
    print(f"Kept paintings without an artist Wikidata ID: {no_wd}")


def download_one(row, size):
    path = IMAGES / f"{row['met_id']}.jpg"
    if path.exists() and path.stat().st_size > 0:
        return "skipped"
    for attempt in range(5):
        try:
            r = session.get(row["image_url"], timeout=90)
            if r.status_code == 200 and r.content:
                img = Image.open(io.BytesIO(r.content)).convert("RGB")
                img.thumbnail((size, size), Image.LANCZOS)   # long side = size, keeps aspect ratio
                img.save(path, "JPEG", quality=90)
                return "ok"
            if r.status_code in (403, 429, 500, 502, 503, 504):
                time.sleep(2 ** attempt)
                continue
            return f"http {r.status_code}"
        except (requests.ConnectionError, requests.Timeout):
            time.sleep(2 ** attempt)
        except Exception as e:
            return f"bad image: {type(e).__name__}"
    return "failed"


def step_images(size=512, workers=4):
    IMAGES.mkdir(parents=True, exist_ok=True)
    with META_FILE.open(encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    results = Counter()
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = [ex.submit(download_one, r, size) for r in rows]
        for f in tqdm(as_completed(futures), total=len(futures), desc="images"):
            results[f.result()] += 1
    log({"step": "images", "size_long_side": size, "resize": "Pillow LANCZOS, JPEG q90",
         "results": dict(results)})
    print("Images:", dict(results))


# ---------- utils ----------

def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)


def log(entry):
    entry["at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    history = json.loads(LOG_FILE.read_text()) if LOG_FILE.exists() else []
    history.append(entry)
    LOG_FILE.write_text(json.dumps(history, indent=2))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("step", choices=["ids", "meta", "images", "all"])
    ap.add_argument("--limit", type=int, help="only process the first N objects (for testing)")
    ap.add_argument("--size", type=int, default=512, help="long side of saved images in pixels")
    ap.add_argument("--workers", type=int, default=8, help="parallel requests")
    a = ap.parse_args()
    if a.step in ("ids", "all"):
        step_ids()
    if a.step in ("meta", "all"):
        step_meta(a.limit, a.workers)
    if a.step in ("images", "all"):
        step_images(a.size, min(a.workers, 4))
