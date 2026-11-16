"""
Check the Rijksmuseum harvest made by main.py.

Run it from anywhere:
    python check_rijks.py

It prints a short report and saves three CSV files in data/rijks/checks/:
    duplicate_object_numbers.csv   paintings that share an object number (and so one image file)
    missing_wikidata.csv           paintings whose artist has no Wikidata ID
    errors.csv                     paintings that failed during the metadata step
"""

import csv
from collections import Counter
from pathlib import Path

# Paths are relative to this script, so it works from any folder
BASE = Path(__file__).resolve().parent / "data" / "rijks"
META = BASE / "metadata.csv"
IMAGES = BASE / "images"
CHECKS = BASE / "checks"


def load_rows():
    if not META.exists():
        raise SystemExit(f"Can't find {META}. Run 'python main.py meta' first.")
    with META.open(encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def save(name, rows, fields):
    CHECKS.mkdir(parents=True, exist_ok=True)
    path = CHECKS / name
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    return path


def heading(text):
    print(f"\n{text}\n{'-' * len(text)}")


def main():
    rows = load_rows()
    ok = [r for r in rows if not r["error"]]
    with_url = [r for r in rows if r["iiif_url"]]
    on_disk = {p.stem for p in IMAGES.glob("*.jpg")} if IMAGES.exists() else set()
    if "influence_wikidata" not in rows[0]:
        print("Note: metadata.csv is from the old parser. Run 'python main.py meta' again.")

    heading("Overview")
    print(f"Paintings in metadata.csv:   {len(rows):>5}")
    print(f"  without errors:            {len(ok):>5}")
    print(f"  with an image URL:         {len(with_url):>5}")
    print(f"Image files on disk:         {len(on_disk):>5}")

    # 1. Duplicate object numbers -> paintings that share one image file
    heading("1. Paintings sharing an object number")
    file_name = lambda r: r["rijks_id"]          # images are saved as <rijks_id>.jpg
    obj_num = lambda r: r["object_number"] or "(empty)"
    counts = Counter(obj_num(r) for r in with_url)
    dup_rows = sorted((r for r in with_url if counts[obj_num(r)] > 1), key=obj_num)
    dup_names = {obj_num(r) for r in dup_rows}
    odd = [r for r in with_url if r["object_number"] and not r["object_number"][:1].isalpha()]
    print(f"Object numbers used more than once: {len(dup_names)}")
    print(f"Paintings involved:                 {len(dup_rows)}")
    print(f"Object numbers that don't look like SK-A-123: {len(odd)}")
    counts = Counter(obj_num(r) for r in with_url)
    print(f"Paintings with an empty object number: "
          f"{sum(1 for r in rows if not r['object_number'])}")
    if dup_rows:
        for name in sorted(dup_names)[:5]:
            print(f"  e.g. {name}: {counts[name]} paintings")
        p = save("duplicate_object_numbers.csv", dup_rows,
                 ["object_number", "rijks_id", "title", "artist", "year_begin", "iiif_url"])
        print(f"Saved list to {p}")

    # 2. Images that should exist but don't
    heading("2. Missing image files")
    missing_img = [r for r in with_url if file_name(r) not in on_disk]
    print(f"Paintings with an image URL but no file: {len(missing_img)}")
    for r in missing_img[:5]:
        print(f"  {r['object_number'] or r['rijks_id']}  {r['title'][:50]}")

    # 3. Missing artist Wikidata IDs
    heading("3. Artists without a Wikidata ID")
    no_wd = [r for r in ok if not r["artist_wikidata"]]
    has_rkd = sum(1 for r in no_wd if r["artist_rkd"])
    has_ulan = sum(1 for r in no_wd if r["artist_ulan"])
    either = sum(1 for r in no_wd if r["artist_rkd"] or r["artist_ulan"])
    no_artist = sum(1 for r in no_wd if not r["artist"])
    print(f"Paintings without an artist Wikidata ID: {len(no_wd)}")
    print(f"  have an RKD ID:                {has_rkd:>5}")
    print(f"  have a ULAN ID:                {has_ulan:>5}")
    print(f"  have RKD or ULAN (rescuable):  {either:>5}")
    print(f"  no artist listed at all:       {no_artist:>5}")
    print(f"  date-only fallback needed:     {len(no_wd) - either:>5}")
    print("Most common artist names among them:")
    for name, c in Counter(r["artist"] or "(none)" for r in no_wd).most_common(10):
        print(f"  {c:>5}  {name}")
    infl = [r for r in no_wd if r.get("influence_wikidata")]
    print(f"  no painter, but 'after/workshop/circle of' an artist with Wikidata: {len(infl)}")
    print("Most common attribution notes among them:")
    for note, c in Counter(r["attribution"] or "(none)" for r in no_wd).most_common(10):
        print(f"  {c:>5}  {note[:90]}")
    p = save("missing_wikidata.csv", no_wd,
             ["rijks_id", "object_number", "title", "artist", "influence_artist",
              "influence_wikidata", "attribution", "artist_rkd", "artist_ulan",
              "year_begin", "year_end"])

    heading("3b. Attribution types (all paintings)")
    kinds = Counter()
    for r in ok:
        a = r.get("attribution", "").lower()
        for k in ("anonymous", "attributed to", "workshop of", "circle of", "follower of",
                  "after", "manner of", "copy after", "school of"):
            if k in a:
                kinds[k] += 1
                break
        else:
            kinds["named painter" if r["artist"] else "(no note)"] += 1
    for k, c in kinds.most_common():
        print(f"  {c:>5}  {k}")

    heading("3c. Object-number prefixes (SK = paintings collection)")
    for p_, c in Counter(r["object_number"].split("-")[0] for r in ok).most_common(10):
        print(f"  {c:>5}  {p_}")
    print(f"Saved list to {p}")

    # 4. Errors from the metadata step
    heading("4. Metadata errors")
    errs = [r for r in rows if r["error"]]
    for e, c in Counter(r["error"] for r in errs).most_common():
        print(f"  {c:>5}  {e}")
    if errs:
        p = save("errors.csv", errs, ["rijks_id", "object_number", "error"])
        print(f"Saved list to {p}")

    # 5. Date coverage (needed for the date-based style fallback)
    heading("5. Dates")
    no_year = sum(1 for r in ok if not r["year_begin"])
    print(f"Paintings without a start year: {no_year}")
    years = [int(r["year_begin"]) for r in ok if r["year_begin"].lstrip("-").isdigit()]
    if years:
        print(f"Year range: {min(years)} to {max(years)}")
        by_century = Counter((y // 100 + 1) for y in years)
        for c in sorted(by_century):
            print(f"  {c:>2}th century: {by_century[c]:>5}")


if __name__ == "__main__":
    main()