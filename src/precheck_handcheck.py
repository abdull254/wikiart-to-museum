"""
Speed up the hand check: compare each derived museum label with WikiArt's labels for
the same artist, and sort hand_check.csv so the rows that need real attention come first.

Adds three columns to data/final/hand_check.csv (your filled-in answers are kept):
  wikiart_styles   the styles WikiArt gives this artist, with counts
  check_first      "1 disagree", "2 unknown artist" or "3 agrees"  (sort order)
  suggested        what the label probably should be, to help you decide

You still fill in 'correct' yourself. For "3 agrees" rows a quick glance is enough.

Usage:
  python precheck_handcheck.py
"""

import csv
from collections import Counter
from pathlib import Path

from prepare_dataset import MERGE, ArtistIndex

BASE = Path(__file__).resolve().parent / "data"
HANDCHECK = BASE / "final" / "hand_check.csv"
WIKI = BASE / "wikiart" / "metadata.csv"


def main():
    rows = list(csv.DictReader(HANDCHECK.open(encoding="utf-8")))
    wiki = list(csv.DictReader(WIKI.open(encoding="utf-8")))

    by_artist = {}
    for w in wiki:
        s = MERGE.get(w["style"], w["style"])
        by_artist.setdefault(w["artist"], Counter())[s] += 1
    index = ArtistIndex(list(by_artist))

    for r in rows:
        styles = Counter()
        for a in index.matches(r["artist"]):
            styles.update(by_artist[a])
        total = sum(styles.values())
        main_styles = [s for s, c in styles.most_common() if c / max(total, 1) >= 0.2]
        r["wikiart_styles"] = "; ".join(f"{s} {c}" for s, c in styles.most_common(4))
        if not total:
            r["check_first"], r["suggested"] = "2 unknown artist", ""
        elif r["derived_style"] in main_styles:
            r["check_first"], r["suggested"] = "3 agrees", r["derived_style"]
        else:
            r["check_first"], r["suggested"] = "1 disagree", main_styles[0] if main_styles else ""

    rows.sort(key=lambda r: (r["check_first"], r["museum"]))
    fields = list(rows[0].keys())
    for col in ("correct", "correct_style", "note"):      # keep these at the end, easy to fill
        fields.remove(col)
        fields.append(col)
    with HANDCHECK.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)

    c = Counter(r["check_first"] for r in rows)
    print(f"Sorted {len(rows)} rows in {HANDCHECK.name}:")
    print(f"  {c['1 disagree']:>3}  disagree with WikiArt   -> look carefully (the 'suggested' column helps)")
    print(f"  {c['2 unknown artist']:>3}  artist not in WikiArt  -> check year + country against the style table")
    print(f"  {c['3 agrees']:>3}  agree with WikiArt      -> a quick glance; usually 'yes'")


if __name__ == "__main__":
    main()
