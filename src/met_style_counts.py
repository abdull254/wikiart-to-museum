#!/usr/bin/env python3
"""Count Met Open Access images per style under label map v0 (metadata only).

Downloads object metadata from the Met Collection API (no images), derives a
style label for each public-domain painting (and, optionally, Japanese
woodblock prints) using the label-map v0 rules, and writes per-class counts.

Rules, applied in order (see the plan doc, "Style label map v0"):
  1. Artist anchor: artist has >= 10 WikiArt images and >= 70% of them fall in
     one shared class -> that class.
  2. Date and region fallback, only for pre-1780 classes and Ukiyo-e.
  3. "Copy after" / "Follower of" / "Imitator of" / "Style of" / "After" made
     more than 50 years after the artist's death -> excluded.
  4. Anything else (post-1780 without an anchor, no WikiArt class such as
     Neoclassicism) -> excluded, with the reason recorded.

Usage:
  pip install requests
  python met_style_counts.py                      # depts 11, 1, 21
  python met_style_counts.py --ukiyoe             # also look for Japanese prints
  python met_style_counts.py --max-objects 300    # quick smoke test

Outputs go to ./met_out/. Object JSON is cached in ./met_out/cache/, so a
rerun (or a crash part-way) does not refetch anything.
"""

import argparse
import csv
import json
import re
import sys
import threading
import time
import unicodedata
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests

API = "https://collectionapi.metmuseum.org/public/collection/v1"
SEARCH_V11 = "https://collectionapi.metmuseum.org/public/collection/v1.1/search"
WIKIART_CSVS = [
    "https://raw.githubusercontent.com/cs-chan/ArtGAN/master/WikiArt%20Dataset/Style/style_train.csv",
    "https://raw.githubusercontent.com/cs-chan/ArtGAN/master/WikiArt%20Dataset/Style/style_val.csv",
]

# Departments: 11 European Paintings, 1 The American Wing, 21 Modern and Contemporary Art
DEFAULT_DEPARTMENTS = [11, 1, 21]
ASIAN_ART = 6

MIN_TEST_IMAGES = 30
TEST_FRACTION = 0.70
ANCHOR_MIN_IMAGES = 10
ANCHOR_MIN_SHARE = 0.70

# Label map v0: WikiArt style -> shared class. Styles not listed are dropped.
LABEL_MAP_V0 = {
    "Early_Renaissance": "Early Renaissance",
    "High_Renaissance": "High Renaissance and Mannerism",
    "Mannerism_Late_Renaissance": "High Renaissance and Mannerism",
    "Northern_Renaissance": "Northern Renaissance",
    "Baroque": "Baroque",
    "Rococo": "Rococo",
    "Romanticism": "Romanticism",
    "Realism": "Realism",
    "Impressionism": "Impressionism",
    "Post_Impressionism": "Post-Impressionism",
    "Pointillism": "Post-Impressionism",
    "Symbolism": "Symbolism",
    "Ukiyo_e": "Ukiyo-e",
}
SHARED_CLASSES = sorted(set(LABEL_MAP_V0.values()))

ITALIAN = {"italian", "florentine", "venetian", "sienese", "neapolitan", "bolognese",
           "roman", "lombard", "ferrarese", "umbrian", "milanese", "genoese",
           "veronese", "paduan", "emilian", "marchigian", "piedmontese"}
NORTHERN = {"netherlandish", "flemish", "dutch", "german", "french", "austrian",
            "swiss", "bohemian", "south netherlandish", "north netherlandish"}
EUROPEAN = ITALIAN | NORTHERN | {"spanish", "british", "english", "scottish", "irish",
                                 "portuguese", "danish", "swedish", "norwegian",
                                 "polish", "russian", "hungarian", "czech", "belgian"}
COPY_PREFIXES = ("copy after", "follower of", "imitator of", "style of", "after")


class RateLimiter:
    def __init__(self, per_second):
        self.interval = 1.0 / per_second
        self.lock = threading.Lock()
        self.next_time = time.monotonic()

    def wait(self):
        with self.lock:
            now = time.monotonic()
            if self.next_time > now:
                time.sleep(self.next_time - now)
            self.next_time = max(now, self.next_time) + self.interval


def slugify(name):
    name = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")


def artist_slugs(display_name):
    """Candidate WikiArt slugs for a Met artistDisplayName.

    'Rembrandt (Rembrandt van Rijn)' -> ['rembrandt', 'rembrandt-van-rijn']
    """
    name = display_name.split("|")[0].strip()
    if not name:
        return []
    candidates = [re.sub(r"\(.*?\)", "", name).strip()]
    candidates += re.findall(r"\((.*?)\)", name)
    slugs = []
    for c in candidates:
        s = slugify(c)
        if s and s not in slugs:
            slugs.append(s)
        stripped = re.sub(r"-the-(elder|younger)$", "", s)
        if stripped and stripped not in slugs:
            slugs.append(stripped)
    return slugs


def load_wikiart(cache_dir):
    """Return {artist_slug: Counter(shared class or 'DROPPED')} and {artist_slug: set(title slugs)}."""
    styles = defaultdict(Counter)
    titles = defaultdict(set)
    for url in WIKIART_CSVS:
        path = cache_dir / Path(url).name
        if not path.exists():
            r = requests.get(url, timeout=120)
            r.raise_for_status()
            path.write_bytes(r.content)
        with path.open() as f:
            for row in csv.reader(f):
                style, filename = row[0].split("/", 1)
                artist, _, title = filename.rsplit(".", 1)[0].partition("_")
                styles[artist][LABEL_MAP_V0.get(style, "DROPPED")] += 1
                titles[artist].add(re.sub(r"-?\(?\d+\)?$", "", title))
    return styles, titles


def get_json(session, limiter, url, params=None, retries=5):
    for attempt in range(retries):
        limiter.wait()
        try:
            r = session.get(url, params=params, timeout=30)
            if r.status_code == 404:
                return None
            if r.status_code in (403, 429, 500, 502, 503):
                time.sleep(2 ** attempt)
                continue
            r.raise_for_status()
            return r.json()
        except requests.RequestException:
            time.sleep(2 ** attempt)
    raise RuntimeError(f"giving up on {url} {params or ''}")


def department_object_ids(session, limiter, dept):
    data = get_json(session, limiter, f"{API}/objects", {"departmentIds": dept})
    return (data or {}).get("objectIDs") or []


def ukiyoe_candidate_ids(session, limiter):
    """Best effort: search Asian Art for woodblock prints with images."""
    params = {"departmentId": ASIAN_ART, "hasImages": "true", "q": "woodblock print"}
    ids, offset, limit = [], 0, 500
    while True:
        data = get_json(session, limiter, SEARCH_V11, {**params, "offset": offset, "limit": limit})
        batch = (data or {}).get("objectIDs") or []
        new = [i for i in batch if i not in set(ids)]
        if not new:  # empty page, or the API ignored offset and repeated a page
            break
        ids += new
        if len(ids) >= (data or {}).get("total", float("inf")):
            break
        offset += len(batch)
    if not ids:  # v1.1 unavailable or shaped differently: try the old endpoint
        data = get_json(session, limiter, f"{API}/search", params)
        ids = (data or {}).get("objectIDs") or []
    return ids


def fetch_object(session, limiter, cache_dir, object_id):
    path = cache_dir / f"{object_id}.json"
    if path.exists():
        return json.loads(path.read_text())
    obj = get_json(session, limiter, f"{API}/objects/{object_id}") or {}
    path.write_text(json.dumps(obj))
    return obj


def year(value):
    m = re.search(r"-?\d{3,4}", str(value or ""))
    return int(m.group()) if m else None


def region(obj):
    nat = (obj.get("artistNationality") or "").split("|")[0].strip().lower()
    nat = re.sub(r",.*$", "", nat)
    if nat in ITALIAN:
        return "italian"
    if nat in NORTHERN:
        return "northern"
    if nat in EUROPEAN or obj.get("department") == "European Paintings":
        return "european"
    if "japan" in (obj.get("culture") or "").lower():
        return "japanese"
    return None


def date_fallback(obj, mid, reg):
    is_print = obj.get("classification", "").lower().startswith("print")
    if reg == "japanese" and is_print and 1680 <= mid <= 1900:
        return "Ukiyo-e"
    if is_print:
        return None
    if reg == "italian" and 1400 <= mid < 1490:
        return "Early Renaissance"
    if reg == "italian" and 1490 <= mid < 1600:
        return "High Renaissance and Mannerism"
    if reg == "northern" and 1400 <= mid < 1580:
        return "Northern Renaissance"
    if reg in ("italian", "northern", "european") and 1600 <= mid < 1720:
        return "Baroque"
    if reg in ("italian", "northern", "european") and 1720 <= mid < 1780:
        return "Rococo"
    return None


def label(obj, wikiart_styles, wikiart_titles):
    """Return (label or None, rule or exclusion reason, matched slug, title match)."""
    begin, end = obj.get("objectBeginDate"), obj.get("objectEndDate")
    mid = (begin + end) / 2 if isinstance(begin, int) and isinstance(end, int) else None

    prefix = (obj.get("artistPrefix") or "").strip().lower()
    death = year(obj.get("artistEndDate"))
    if prefix.startswith(COPY_PREFIXES) and death and begin and begin > death + 50:
        return None, "excluded: late copy or follower", "", False

    slug = next((s for s in artist_slugs(obj.get("artistDisplayName") or "")
                 if s in wikiart_styles), "")
    title_match = bool(slug) and slugify(obj.get("title") or "") in wikiart_titles.get(slug, ())

    if slug:
        counts = wikiart_styles[slug]
        total = sum(counts.values())
        top, n = counts.most_common(1)[0]
        if total >= ANCHOR_MIN_IMAGES and top != "DROPPED" and n / total >= ANCHOR_MIN_SHARE:
            return top, "artist anchor", slug, title_match

    if mid is None:
        return None, "excluded: no date", slug, title_match
    reg = region(obj)
    fallback = date_fallback(obj, mid, reg)
    if fallback:
        return fallback, "date and region", slug, title_match
    if mid >= 1780:
        reason = "excluded: post-1780, artist mixed-style" if slug else "excluded: post-1780, artist not in WikiArt"
    elif reg is None:
        reason = "excluded: unknown region"
    else:
        reason = "excluded: outside fallback windows"
    return None, reason, slug, title_match


def eligible(obj, include_prints):
    if not obj.get("isPublicDomain") or not obj.get("primaryImage"):
        return False
    cls = (obj.get("classification") or "").lower()
    if cls == "paintings":
        return True
    return include_prints and cls.startswith("print") and "japan" in (obj.get("culture") or "").lower()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--departments", type=int, nargs="+", default=DEFAULT_DEPARTMENTS)
    ap.add_argument("--ukiyoe", action="store_true", help="also search Asian Art for Japanese woodblock prints")
    ap.add_argument("--max-objects", type=int, default=None, help="limit fetches, for a smoke test")
    ap.add_argument("--rate", type=float, default=40.0, help="requests per second (Met asks for <= 80)")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--out", type=Path, default=Path("met_out"))
    args = ap.parse_args()

    cache_dir = args.out / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    session = requests.Session()
    session.headers["User-Agent"] = "CIS631-wikiart-museum-study/0.1 (metadata count, research)"
    limiter = RateLimiter(args.rate)

    print("Loading WikiArt artist styles...", file=sys.stderr)
    wikiart_styles, wikiart_titles = load_wikiart(args.out)
    print(f"  {len(wikiart_styles)} WikiArt artists", file=sys.stderr)

    ids = []
    for dept in args.departments:
        dept_ids = department_object_ids(session, limiter, dept)
        print(f"Department {dept}: {len(dept_ids)} objects", file=sys.stderr)
        ids += dept_ids
    if args.ukiyoe:
        uk = ukiyoe_candidate_ids(session, limiter)
        print(f"Ukiyo-e search: {len(uk)} candidates", file=sys.stderr)
        ids += uk
    ids = list(dict.fromkeys(ids))[: args.max_objects]

    objects = []
    with ThreadPoolExecutor(args.workers) as pool:
        futures = [pool.submit(fetch_object, session, limiter, cache_dir, i) for i in ids]
        for n, fut in enumerate(as_completed(futures), 1):
            objects.append(fut.result())
            if n % 500 == 0 or n == len(futures):
                print(f"  fetched {n}/{len(futures)}", file=sys.stderr)

    rows, reasons = [], Counter()
    for obj in objects:
        if not eligible(obj, args.ukiyoe):
            continue
        lab, rule, slug, title_match = label(obj, wikiart_styles, wikiart_titles)
        reasons[rule] += 1
        rows.append({
            "objectID": obj["objectID"],
            "department": obj.get("department", ""),
            "classification": obj.get("classification", ""),
            "title": obj.get("title", ""),
            "artistPrefix": obj.get("artistPrefix", ""),
            "artistDisplayName": obj.get("artistDisplayName", ""),
            "artistNationality": obj.get("artistNationality", ""),
            "objectBeginDate": obj.get("objectBeginDate", ""),
            "objectEndDate": obj.get("objectEndDate", ""),
            "label_v0": lab or "",
            "rule": rule,
            "wikiart_artist": slug,
            "possible_wikiart_duplicate": title_match,
            "primaryImage": obj.get("primaryImage", ""),
            "objectURL": obj.get("objectURL", ""),
        })

    with (args.out / "met_objects.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else ["objectID"])
        writer.writeheader()
        writer.writerows(sorted(rows, key=lambda r: r["objectID"]))

    per_class = Counter(r["label_v0"] for r in rows if r["label_v0"])
    artists = defaultdict(set)
    dupes = Counter()
    for r in rows:
        if r["label_v0"]:
            artists[r["label_v0"]].add(r["artistDisplayName"])
            dupes[r["label_v0"]] += r["possible_wikiart_duplicate"]

    summary = []
    for cls in SHARED_CLASSES:
        n = per_class[cls]
        est_test = int(n * TEST_FRACTION)
        summary.append({
            "class": cls,
            "n_labeled": n,
            "n_artists": len(artists[cls]),
            "est_test_70pct": est_test,
            "est_finetune_30pct": n - est_test,
            "possible_wikiart_duplicates": dupes[cls],
            "evaluable": est_test >= MIN_TEST_IMAGES,
        })
    with (args.out / "met_class_counts.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary[0].keys()))
        writer.writeheader()
        writer.writerows(summary)
    with (args.out / "met_rule_counts.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["rule_or_reason", "n"])
        writer.writerows(reasons.most_common())

    labeled = sum(per_class.values())
    print(f"\nEligible public-domain objects with images: {len(rows)}")
    print(f"Labeled under v0: {labeled}  (excluded: {len(rows) - labeled})\n")
    print(f"{'class':32} {'n':>6} {'artists':>8} {'est test':>9} {'dupes?':>7}  evaluable")
    for s in summary:
        print(f"{s['class']:32} {s['n_labeled']:>6} {s['n_artists']:>8} {s['est_test_70pct']:>9} "
              f"{s['possible_wikiart_duplicates']:>7}  {'yes' if s['evaluable'] else 'no'}")
    if labeled:
        top = max(summary, key=lambda s: s["n_labeled"])
        k = sum(s["evaluable"] for s in summary)
        print(f"\nMajority floor: {top['class']} = {top['n_labeled'] / labeled:.1%} of labeled objects")
        if k:
            print(f"Chance floor over {k} evaluable classes: {1 / k:.1%}")
    print("\nRules and exclusion reasons:")
    for rule, n in reasons.most_common():
        print(f"  {rule:45} {n}")
    mixed = Counter(r["wikiart_artist"] for r in rows if r["rule"] == "excluded: post-1780, artist mixed-style")
    if mixed:
        print("\nMixed-style artists (candidates for hand labeling by period):")
        for slug, n in mixed.most_common(15):
            print(f"  {slug:35} {n:>4} objects  WikiArt: {dict(wikiart_styles[slug].most_common(3))}")
    print(f"\nWrote {args.out}/met_objects.csv, met_class_counts.csv, met_rule_counts.csv")


if __name__ == "__main__":
    main()
