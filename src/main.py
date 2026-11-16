"""
Harvest Rijksmuseum paintings (metadata + images) for the WikiArt museum-photo project.

Steps
  ids     - page through the Search API (type=painting, imageAvailable=true) -> data/rijks/ids.txt
  meta    - resolve each painting -> visual item -> digital object (cached as JSON),
            and write data/rijks/metadata.csv
  images  - download each image from IIIF at a fixed size -> data/rijks/images/<rijks_id>.jpg
  rename  - ONE-TIME: rename images from the old <object_number>.jpg names to <rijks_id>.jpg

Every step can be stopped and re-run: finished work is skipped.

Fixing the first harvest (run in this order, once):
  python main.py rename
  python main.py meta          # re-parses from the cache, so it's fast
  python main.py images
  python check.py
"""

import argparse
import csv
import json
import re
import shutil
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import requests
from tqdm import tqdm

SEARCH_URL = "https://data.rijksmuseum.nl/search/collection?type=painting&imageAvailable=true"
RESOLVE = "https://data.rijksmuseum.nl/{num}?_profile=la-framed"   # la-framed returns JSON-LD
ENGLISH = "300388277"          # Getty AAT: English
OBJECT_NUMBER = "300312355"    # Getty AAT: accession number / object number

OUT = Path(__file__).resolve().parent / "data" / "rijks"
CACHE = OUT / "cache"
IMAGES = OUT / "images"
IDS_FILE = OUT / "ids.txt"
META_FILE = OUT / "metadata.csv"
LOG_FILE = OUT / "harvest_log.json"
RENAMED_FLAG = OUT / ".renamed_to_rijks_id"

FIELDS = [
    "rijks_id", "object_number", "title", "artist", "artist_rijks_id",
    "artist_wikidata", "artist_ulan", "artist_rkd", "n_artists",
    "influence_artist", "influence_wikidata", "attribution",
    "date_begin", "date_end", "year_begin", "year_end",
    "type", "iiif_url", "image_license", "error",
]

session = requests.Session()
session.headers.update({
    "Accept": "application/ld+json, application/json",
    "User-Agent": "MSU-Mankato CIS631 research harvest (wikiart-to-museum)",
})


# ---------- HTTP ----------

def get_json(url, tries=5):
    for attempt in range(tries):
        try:
            r = session.get(url, timeout=30)
            if r.status_code == 200:
                return r.json()
            if r.status_code == 404:
                return None
            if r.status_code in (429, 500, 502, 503, 504):
                time.sleep(2 ** attempt)
                continue
            r.raise_for_status()
        except (requests.ConnectionError, requests.Timeout, ValueError):
            time.sleep(2 ** attempt)
    raise RuntimeError(f"failed after {tries} tries: {url}")


def num_of(uri):
    """'https://id.rijksmuseum.nl/200100988' -> '200100988'"""
    return str(uri).rstrip("/").rsplit("/", 1)[-1]


def resolve(uri):
    """Fetch a Linked Art record, using a local JSON cache."""
    n = num_of(uri)
    path = CACHE / f"{n}.json"
    if path.exists():
        return json.loads(path.read_text())
    data = get_json(RESOLVE.format(num=n))
    if data is not None:
        path.write_text(json.dumps(data))
    return data


# ---------- small Linked Art helpers ----------

def as_list(x):
    if x is None:
        return []
    return x if isinstance(x, list) else [x]


def dicts(x):
    return [d for d in as_list(x) if isinstance(d, dict)]


def text_of(node):
    """Best human-readable label for a node (handles _label, content, notation, names)."""
    if isinstance(node, str):
        return node
    if isinstance(node, list):
        return next((t for t in (text_of(n) for n in node) if t), "")
    if not isinstance(node, dict):
        return ""
    for key in ("_label", "content"):
        if isinstance(node.get(key), str) and node[key].strip():
            return node[key]
    for n in as_list(node.get("notation")):
        if isinstance(n, dict) and n.get("@value"):
            return n["@value"]
        if isinstance(n, str):
            return n
    for ib in dicts(node.get("identified_by")):
        if ib.get("type") == "Name" and ib.get("content"):
            return ib["content"]
    return ""


def all_texts(node):
    """Every label on a node, lower-cased (to match Dutch or English wording)."""
    out = []
    for key in ("_label", "content"):
        if isinstance(node.get(key), str):
            out.append(node[key].lower())
    for n in as_list(node.get("notation")):
        if isinstance(n, dict) and n.get("@value"):
            out.append(str(n["@value"]).lower())
        elif isinstance(n, str):
            out.append(n.lower())
    return out


def is_english(node):
    return any(ENGLISH in (lang.get("id", "") if isinstance(lang, dict) else str(lang))
               for lang in as_list(node.get("language")))


def equivalent_ids(node):
    """Equivalents come either as {"id": ...} objects or as plain URL strings."""
    out = []
    for e in as_list(node.get("equivalent")):
        if isinstance(e, dict):
            out.append(e.get("id", ""))
        elif isinstance(e, str):
            out.append(e)
    return out


def pick(ids, pattern):
    for i in ids:
        m = re.search(pattern, i)
        if m:
            return m.group(1)
    return ""


# ---------- parse one painting ----------

def object_number_of(obj):
    """The identifier classified as 'object number', not just the first identifier."""
    idents = [i for i in dicts(obj.get("identified_by"))
              if i.get("type") == "Identifier" and i.get("content")]
    for i in idents:
        for c in dicts(i.get("classified_as")):
            texts = all_texts(c)
            if OBJECT_NUMBER in c.get("id", "") or any(
                    "object number" in t or "objectnummer" in t for t in texts):
                return i["content"]
    for i in idents:   # fallback: looks like SK-A-1272
        if re.match(r"^[A-Z]{1,5}-", i["content"]):
            return i["content"]
    return idents[0]["content"] if idents else ""


ACTOR_TYPES = {"Person", "Group", "Actor"}
PRODUCER_NOTE = "300435416"   # Getty AAT: the readable "painter: ..." statement


def notes_of(node):
    """English 'painter: ...' style notes on a production record."""
    out = []
    for r in dicts(node.get("referred_to_by")):
        content = r.get("content", "")
        if not content or (r.get("language") and not is_english(r)):
            continue
        if any(PRODUCER_NOTE in c.get("id", "") for c in dicts(r.get("classified_as"))):
            out.append(content)
    return out


def is_rejected(node):
    """True if the museum marks this attribution as rejected."""
    for r in dicts(node.get("referred_to_by")):
        c = r.get("content", "").lower()
        if "rejected" in c or "verworpen" in c:
            return True
    return False


def collect_production(prods):
    """Return (makers, influences, notes) from the production record.

    makers     - the people who actually painted it (incl. 'attributed to')
    influences - artists it is 'after', 'workshop of', 'circle of', 'follower of', ...
    notes      - the museum's own readable lines, e.g. 'painter: after Peter Paul Rubens'
    Rejected attributions are skipped.
    """
    makers, influences, notes = [], [], []

    def add_actor(a, bucket):
        # An unnamed workshop/circle is a Group "formed by" the influence of a named artist
        if a.get("type") == "Group" and a.get("formed_by"):
            for f in dicts(a["formed_by"]):
                influences.extend(dicts(f.get("influenced_by")))
            return
        bucket.append(a)

    def visit(p):
        if not isinstance(p, dict) or is_rejected(p):
            return
        notes.extend(notes_of(p))
        for a in dicts(p.get("carried_out_by")):
            add_actor(a, makers)
        for a in dicts(p.get("influenced_by")):
            add_actor(a, influences)
        for aa in dicts(p.get("assigned_by")):
            if is_rejected(aa):
                continue
            prop = aa.get("assigned_property", "")
            for x in dicts(aa.get("assigned")):
                if x.get("type") in ACTOR_TYPES:
                    add_actor(x, influences if prop == "influenced_by" else makers)
                elif x.get("type") in ("Production", "Activity"):
                    visit(x)
        for part in dicts(p.get("part")):
            visit(part)

    for p in prods:
        visit(p)
    return dedupe(makers), dedupe(influences), list(dict.fromkeys(notes))


def dedupe(actors):
    seen, out = set(), []
    for a in actors:
        key = a.get("id") or text_of(a)
        if key and key not in seen:
            seen.add(key)
            out.append(a)
    return out


def wikidata_of(a):
    return pick(equivalent_ids(a), r"wikidata\.org/entity/(Q\d+)")


def best(actors):
    """First actor with a Wikidata ID, otherwise the first actor."""
    return next((a for a in actors if wikidata_of(a)), actors[0]) if actors else None


def find_timespan(prods):
    stack = list(prods)
    while stack:
        n = stack.pop(0)
        if not isinstance(n, dict):
            continue
        ts = n.get("timespan")
        for t in dicts(ts):
            if t.get("begin_of_the_begin") or t.get("end_of_the_end"):
                return t
        stack += dicts(n.get("part"))
    return {}


def year(s):
    m = re.match(r"^(-?\d{1,4})", s or "")
    return m.group(1) if m else ""


def parse_object(obj):
    row = {f: "" for f in FIELDS}
    row["rijks_id"] = num_of(obj.get("id", ""))

    names = [i for i in dicts(obj.get("identified_by")) if i.get("type") == "Name"]
    english = [n for n in names if is_english(n)]
    row["title"] = (english or names or [{}])[0].get("content", "")
    row["object_number"] = object_number_of(obj)
    row["type"] = "; ".join(dict.fromkeys(
        filter(None, (text_of(c) for c in dicts(obj.get("classified_as"))))))

    prods = dicts(obj.get("produced_by"))
    makers, influences, notes = collect_production(prods)
    # keep the "painter: ..." lines when there are any; they repeat the short versions
    row["attribution"] = "; ".join([n for n in notes if ":" in n] or notes)
    row["n_artists"] = str(len(makers))
    a = best(makers)
    if a:
        eq = equivalent_ids(a)
        row["artist"] = "; ".join(filter(None, (text_of(x) for x in makers)))
        row["artist_rijks_id"] = num_of(a["id"]) if a.get("id") else ""
        row["artist_wikidata"] = wikidata_of(a)
        row["artist_ulan"] = pick(eq, r"ulan/(\d+)")
        row["artist_rkd"] = pick(eq, r"rkd\.nl/.*?artists/(\d+)")
    i = best(influences)
    if i:
        row["influence_artist"] = "; ".join(filter(None, (text_of(x) for x in influences)))
        row["influence_wikidata"] = wikidata_of(i)

    ts = find_timespan(prods)
    row["date_begin"] = ts.get("begin_of_the_begin", "")
    row["date_end"] = ts.get("end_of_the_end", "")
    row["year_begin"] = year(row["date_begin"])
    row["year_end"] = year(row["date_end"])
    return row


def find_image(obj):
    """painting -> shows (VisualItem) -> digitally_shown_by (DigitalObject) -> access_point."""
    for vi_ref in dicts(obj.get("shows")):
        vi = resolve(vi_ref["id"])
        if not vi:
            continue
        lic = "; ".join(text_of(c) or c.get("id", "")
                        for r in dicts(vi.get("subject_to")) for c in dicts(r.get("classified_as")))
        for do_ref in dicts(vi.get("digitally_shown_by")):
            do = resolve(do_ref["id"])
            if not do:
                continue
            for ap in dicts(do.get("access_point")):
                url = ap.get("id", "")
                if "iiif" in url:
                    return url, lic
    return "", ""


# ---------- steps ----------

def step_ids():
    OUT.mkdir(parents=True, exist_ok=True)
    ids, url, pages, total = [], SEARCH_URL, 0, None
    with tqdm(desc="search pages", unit="page") as bar:
        while url:
            page = get_json(url)
            total = total or page.get("partOf", {}).get("totalItems")
            ids += [it["id"] for it in page.get("orderedItems", [])]
            url = (page.get("next") or {}).get("id")
            pages += 1
            bar.update(1)
            bar.set_postfix(ids=len(ids), total=total)
    ids = list(dict.fromkeys(ids))
    IDS_FILE.write_text("\n".join(ids) + "\n")
    log({"step": "ids", "search_url": SEARCH_URL, "total_reported": total,
         "ids_saved": len(ids), "pages": pages})
    print(f"Saved {len(ids)} ids (API reported {total}) to {IDS_FILE}")


def harvest_one(uri):
    try:
        obj = resolve(uri)
        if obj is None:
            return {**{f: "" for f in FIELDS}, "rijks_id": num_of(uri), "error": "not found"}
        row = parse_object(obj)
        row["iiif_url"], row["image_license"] = find_image(obj)
        if not row["iiif_url"]:
            row["error"] = "no image url"
        return row
    except Exception as e:  # keep going; the error is recorded in the CSV
        return {**{f: "" for f in FIELDS}, "rijks_id": num_of(uri),
                "error": f"{type(e).__name__}: {e}"[:200]}


def step_meta(limit=None, workers=4):
    CACHE.mkdir(parents=True, exist_ok=True)
    ids = [l.strip() for l in IDS_FILE.read_text().splitlines() if l.strip()]
    if limit:
        ids = ids[:limit]
    rows = []
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = [ex.submit(harvest_one, u) for u in ids]
        for f in tqdm(as_completed(futures), total=len(futures), desc="metadata"):
            rows.append(f.result())
    rows.sort(key=lambda r: r["rijks_id"])
    with META_FILE.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)
    errors = sum(1 for r in rows if r["error"])
    no_wd = sum(1 for r in rows if not r["artist_wikidata"] and not r["error"])
    no_artist = sum(1 for r in rows if not r["artist"] and not r["error"])
    log({"step": "meta", "rows": len(rows), "errors": errors,
         "missing_artist_wikidata": no_wd, "no_artist": no_artist})
    print(f"Wrote {len(rows)} rows to {META_FILE}: {errors} errors, "
          f"{no_artist} with no artist, {no_wd} without an artist Wikidata ID")


def iiif_sized(url, size):
    """Ask IIIF for the image scaled so its long side is at most `size` px."""
    return re.sub(r"/full/[^/]+/0/default\.jpg$", f"/full/!{size},{size}/0/default.jpg", url)


def download_one(row, size):
    path = IMAGES / f"{row['rijks_id']}.jpg"      # rijks_id is always unique
    if path.exists() and path.stat().st_size > 0:
        return "skipped"
    url = iiif_sized(row["iiif_url"], size)
    for attempt in range(5):
        try:
            r = session.get(url, timeout=60, headers={"Accept": "image/jpeg"})
            if r.status_code == 200 and r.content:
                path.write_bytes(r.content)
                return "ok"
            if r.status_code in (429, 500, 502, 503, 504):
                time.sleep(2 ** attempt)
                continue
            return f"http {r.status_code}"
        except (requests.ConnectionError, requests.Timeout):
            time.sleep(2 ** attempt)
    return "failed"


def step_images(size=512, workers=4):
    IMAGES.mkdir(parents=True, exist_ok=True)
    with META_FILE.open(encoding="utf-8") as fh:
        rows = [r for r in csv.DictReader(fh) if r["iiif_url"]]
    results = Counter()
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = [ex.submit(download_one, r, size) for r in rows]
        for f in tqdm(as_completed(futures), total=len(futures), desc="images"):
            results[f.result()] += 1
    log({"step": "images", "size_long_side": size, "results": dict(results)})
    print("Images:", dict(results))


def step_rename():
    """One-time: images were saved as <object_number>.jpg, which collided for 76 paintings.
    Rename the safe ones to <rijks_id>.jpg and move the collided ones aside."""
    if RENAMED_FLAG.exists():
        print("Already renamed; nothing to do.")
        return
    with META_FILE.open(encoding="utf-8") as fh:
        rows = [r for r in csv.DictReader(fh) if r["iiif_url"]]
    old_name = lambda r: (r["object_number"] or r["rijks_id"]).replace("/", "_")
    counts = Counter(old_name(r) for r in rows)
    renamed = 0
    for r in rows:
        old = IMAGES / f"{old_name(r)}.jpg"
        new = IMAGES / f"{r['rijks_id']}.jpg"
        if counts[old_name(r)] == 1 and old.exists() and not new.exists():
            old.rename(new)
            renamed += 1
    trash = IMAGES / "_to_delete"
    moved = 0
    for name, c in counts.items():
        f = IMAGES / f"{name}.jpg"
        if c > 1 and f.exists():
            trash.mkdir(exist_ok=True)
            shutil.move(str(f), str(trash / f.name))
            moved += 1
    RENAMED_FLAG.write_text(datetime.now(timezone.utc).isoformat())
    log({"step": "rename", "renamed": renamed, "collided_files_moved": moved})
    print(f"Renamed {renamed} images to <rijks_id>.jpg. Moved {moved} collided files to {trash} "
          f"(you can delete that folder). Next: python main.py meta, then python main.py images")


def log(entry):
    entry["at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    history = json.loads(LOG_FILE.read_text()) if LOG_FILE.exists() else []
    history.append(entry)
    LOG_FILE.write_text(json.dumps(history, indent=2))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("step", choices=["ids", "meta", "images", "rename", "all"])
    ap.add_argument("--limit", type=int, help="only process the first N paintings (for testing)")
    ap.add_argument("--size", type=int, default=512, help="long side of downloaded images in pixels")
    ap.add_argument("--workers", type=int, default=4, help="parallel requests (keep this small)")
    a = ap.parse_args()
    if a.step == "rename":
        step_rename()
    if a.step in ("ids", "all"):
        step_ids()
    if a.step in ("meta", "all"):
        step_meta(a.limit, a.workers)
    if a.step in ("images", "all"):
        step_images(a.size, a.workers)