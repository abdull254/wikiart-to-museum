"""
Derive a style label for every museum painting (Met + Rijksmuseum).
The method is written out as numbered rules in labeling_rules.md; each row records
which rule produced its label (rule_id), so every label can be traced.

Inputs
  data/rijks/metadata_paintings.csv   (from filter_rijks.py, SK paintings only)
  data/met/metadata.csv               (from met_harvest.py)
Outputs (data/labels/)
  museum_labels.csv        one row per painting: style, rule_id, source, flags
  style_counts.csv         paintings per style per museum
  unmapped_movements.csv   Wikidata movements no rule maps yet (review these)
  wikidata_cache.json      Wikidata answers, so re-runs are fast and repeatable

Usage
  python label_museums.py
"""

import csv
import json
import re
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import requests

BASE = Path(__file__).resolve().parent / "data"
RIJKS = BASE / "rijks" / "metadata_paintings.csv"
MET = BASE / "met" / "metadata.csv"
OUT = BASE / "labels"
CACHE = OUT / "wikidata_cache.json"
SPARQL = "https://query.wikidata.org/sparql"
HEADERS = {"User-Agent": "MSU-Mankato CIS631 research (wikiart-museum-style)",
           "Accept": "application/sparql-results+json"}

# ---------------------------------------------------------------------------
# Rule 4: Wikidata movement label -> project style (WikiArt style names).
# Checked top to bottom; the first keyword found in the label wins.
# ---------------------------------------------------------------------------
MOVEMENT_RULES = [
    (["post-impressionism", "post impressionism", "neo-impressionism", "pointillism",
      "divisionism", "les nabis", "nabis", "cloisonnism", "synthetism"], "Post-Impressionism"),
    (["impressionism"], "Impressionism"),
    (["rococo"], "Rococo"),
    (["neoclassic", "neo-classic"], "Neoclassicism"),
    (["romanticism", "romantic", "biedermeier", "nazarene", "düsseldorf school"], "Romanticism"),
    (["academic art", "academicism", "academism", "pompier"], "Academicism"),
    (["realism", "barbizon", "hague school"], "Realism"),
    (["symbolism", "pre-raphaelite", "aestheticism"], "Symbolism"),
    (["caravagg", "tenebrism"], "Baroque"),
    (["baroque", "dutch golden age", "golden age of dutch", "flemish baroque"], "Baroque"),
    (["mannerism", "fontainebleau"], "Mannerism (Late Renaissance)"),
    (["high renaissance"], "High Renaissance"),
    (["early netherlandish", "northern renaissance", "german renaissance", "danube school",
      "flemish primitives"], "Northern Renaissance"),
    (["early renaissance", "quattrocento"], "Early Renaissance"),
    (["international gothic"], "International Gothic"),
    (["proto-renaissance", "proto renaissance", "trecento", "gothic"], "Proto Renaissance"),
    (["renaissance"], "RENAISSANCE_BY_DATE"),     # generic -> Rule 5 decides which one
    # Rule 4b: regional schools and broad labels that name a place, not a style.
    # The painting's date decides (Rule 7), with Italian schools counted as Italian.
    (["sienese school", "venetian school", "veronese school", "florentine school",
      "bolognese school", "ferrarese school", "lombard school", "neapolitan school",
      "genoese school", "umbrian school", "roman school", "paduan school", "milanese school",
      "school of ferrara", "school of venice", "school of siena"], "ITALIAN_BY_DATE"),
    (["classicism", "fijnschilder", "leiden school", "delft school", "haarlem school",
      "antwerp school", "utrecht school"], "BY_DATE"),
]

# Rule 6: rough active period of each style, used to choose between candidates
PERIODS = {
    "Proto Renaissance": (1250, 1400), "International Gothic": (1350, 1450),
    "Early Renaissance": (1400, 1495), "Northern Renaissance": (1420, 1580),
    "High Renaissance": (1490, 1530), "Mannerism (Late Renaissance)": (1520, 1610),
    "Baroque": (1590, 1730), "Rococo": (1715, 1790), "Neoclassicism": (1760, 1850),
    "Romanticism": (1780, 1880), "Academicism": (1800, 1910), "Realism": (1830, 1910),
    "Symbolism": (1860, 1920), "Impressionism": (1860, 1920), "Post-Impressionism": (1880, 1920),
}

NORTHERN = {"dutch", "netherlandish", "flemish", "german", "belgian", "north netherlandish",
            "south netherlandish", "austrian", "swiss", "english", "british", "french"}
ITALIAN = {"italian", "italy", "venetian", "venice", "florentine", "florence", "tuscany",
           "sienese", "siena", "bolognese", "genoese", "genoa", "neapolitan", "naples",
           "lombard", "milanese", "milan", "papal states", "sicily", "mantua", "ferrara"}

# Rule 3: attributions where the named artist did NOT paint it in their own time
COPY_WORDS = ["after", "copy", "manner of", "style of", "imitator", "pastiche"]
# Rule 2: attributions close enough to the master to inherit the master's movement
CLOSE_WORDS = ["workshop", "studio", "circle", "follower", "school of", "attributed", "pupil"]


# ---------------------------------------------------------------------------
# Wikidata
# ---------------------------------------------------------------------------

def load_cache():
    return json.loads(CACHE.read_text()) if CACHE.exists() else {}


def fetch_wikidata(qids, cache):
    todo = sorted(q for q in qids if q and q not in cache)
    for i in range(0, len(todo), 150):
        batch = todo[i:i + 150]
        values = " ".join(f"wd:{q}" for q in batch)
        query = f"""
        SELECT ?item ?mov ?movLabel ?parentLabel ?birth ?death ?countryLabel WHERE {{
          VALUES ?item {{ {values} }}
          OPTIONAL {{ ?item wdt:P135 ?mov .
                     OPTIONAL {{ ?mov wdt:P279|wdt:P361 ?parent . }} }}
          OPTIONAL {{ ?item wdt:P569 ?birth . }}
          OPTIONAL {{ ?item wdt:P570 ?death . }}
          OPTIONAL {{ ?item wdt:P27 ?country . }}
          SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en". }}
        }}"""
        for attempt in range(5):
            r = requests.get(SPARQL, params={"query": query}, headers=HEADERS, timeout=120)
            if r.status_code == 200:
                break
            time.sleep(5 * (attempt + 1))
        r.raise_for_status()
        for q in batch:
            cache.setdefault(q, {"movements": {}, "birth": "", "death": "", "countries": []})
        for b in r.json()["results"]["bindings"]:
            q = b["item"]["value"].rsplit("/", 1)[-1]
            e = cache[q]
            if "mov" in b:
                m = b["mov"]["value"].rsplit("/", 1)[-1]
                mv = e["movements"].setdefault(m, {"label": b.get("movLabel", {}).get("value", m),
                                                   "parents": []})
                p = b.get("parentLabel", {}).get("value")
                if p and p not in mv["parents"]:
                    mv["parents"].append(p)
            if "birth" in b and not e["birth"]:
                e["birth"] = b["birth"]["value"][:5].lstrip("+")[:4]
            if "death" in b and not e["death"]:
                e["death"] = b["death"]["value"][:5].lstrip("+")[:4]
            c = b.get("countryLabel", {}).get("value")
            if c and c not in e["countries"]:
                e["countries"].append(c)
        CACHE.write_text(json.dumps(cache, indent=1))
        print(f"  Wikidata: {min(i + 150, len(todo))}/{len(todo)} items")
        time.sleep(1)
    return cache


# ---------------------------------------------------------------------------
# Rules
# ---------------------------------------------------------------------------

def map_label(label):
    l = label.lower()
    for keys, style in MOVEMENT_RULES:
        if any(k in l for k in keys):
            return style
    return None


def map_movement(mv):
    """Rule 4: map a movement by its own label, else by its parent movements' labels."""
    s = map_label(mv["label"])
    if s:
        return s
    for p in mv["parents"]:
        s = map_label(p)
        if s:
            return s
    return None


def region_of(nationality, countries):
    text = " ".join([nationality or ""] + (countries or [])).lower()
    if any(w in text for w in ITALIAN):
        return "italian"
    if any(w in text for w in NORTHERN) or any(
            c in text for c in ["netherlands", "belgium", "germany", "holy roman", "dutch republic"]):
        return "northern"
    return ""


def renaissance_by_date(year, region):
    """Rule 5: split 'Renaissance' by date and region."""
    if year is None:
        return None
    if region == "northern":
        return "Northern Renaissance" if year < 1580 else "Mannerism (Late Renaissance)"
    if region == "italian":
        if year < 1495:
            return "Early Renaissance"
        if year < 1527:
            return "High Renaissance"
        return "Mannerism (Late Renaissance)"
    return None


def date_rule(year, region):
    """Rule 7: date-only fallback, used only where one style dominates a period."""
    if year is None:
        return None, "no date"
    if 1400 <= year < 1600:
        s = renaissance_by_date(year, region)
        return (s, "") if s else (None, "1400-1600 needs a region")
    if 1600 <= year < 1720:
        return "Baroque", ""
    if 1720 <= year < 1780:
        return "Rococo", ""
    return None, f"{year}: several styles overlap, no date rule"


def fits(style, year, margin=30):
    """Rule 6: a movement more than `margin` years outside its period is not used."""
    lo, hi = PERIODS.get(style, (-9999, 9999))
    return year is None or lo - margin <= year <= hi + margin


def pick_by_period(styles, year):
    """Rule 6: several candidate styles -> keep the one(s) whose period contains the year."""
    styles = list(dict.fromkeys(styles))
    if len(styles) <= 1 or year is None:
        return (styles[0] if styles else None), len(styles) > 1
    fits = [s for s in styles if PERIODS.get(s, (0, 9999))[0] <= year <= PERIODS.get(s, (0, 9999))[1]]
    if len(fits) == 1:
        return fits[0], False
    pool = fits or styles
    # still several: closest period midpoint to the year
    best = min(pool, key=lambda s: abs(sum(PERIODS.get(s, (year, year))) / 2 - year))
    return best, True


def year_of(r):
    ys = [int(v) for v in (r.get("year_begin"), r.get("year_end"))
          if v and str(v).lstrip("-").isdigit()]
    return round(sum(ys) / len(ys)) if ys else None


def styles_from(qid, cache, year, region, unmapped):
    e = cache.get(qid) or {}
    out = []
    for m, mv in e.get("movements", {}).items():
        s = map_movement(mv)
        if s == "RENAISSANCE_BY_DATE":
            s = renaissance_by_date(year, region)
        elif s == "ITALIAN_BY_DATE":
            s = date_rule(year, "italian")[0]
        elif s == "BY_DATE":
            s = date_rule(year, region)[0]
        if s:
            out.append(s)
        else:
            unmapped[mv["label"]] += 1
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def read(path):
    with path.open(encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def clean_name(n):
    n = re.sub(r"\(.*?\)", "", n or "")
    n = re.sub(r"^(attributed to|workshop of|circle of|follower of|after|manner of|"
               r"copy after|school of|studio of)\s+", "", n.strip(), flags=re.I)
    return re.sub(r"\s+", " ", n.split(",")[0]).strip().lower()


def name_from_attribution(text):
    """'painter: George Hendrik Breitner (mentioned on object), Netherlands' -> name"""
    m = re.search(r"(?:painter|schilder)\s*:\s*([^;]+)", text or "", flags=re.I)
    return clean_name(m.group(1)) if m else ""


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rijks, met = read(RIJKS), read(MET)

    # Common record shape for both museums
    recs = []
    for r in met:
        recs.append({"museum": "met", "id": r["met_id"], "object_number": r["accession_number"],
                     "title": r["title"], "artist": r["artist"], "qualifier": r["artist_prefix"],
                     "artist_qid": r["artist_wikidata"], "object_qid": r["object_wikidata"],
                     "influence_qid": "", "nationality": r["artist_nationality"],
                     "attribution": r["artist_prefix"], "year": year_of(r)})
    for r in rijks:
        recs.append({"museum": "rijks", "id": r["rijks_id"], "object_number": r["object_number"],
                     "title": r["title"], "artist": r["artist"], "qualifier": "",
                     "artist_qid": r["artist_wikidata"], "object_qid": "",
                     "influence_qid": r.get("influence_wikidata", ""), "nationality": "",
                     "attribution": r.get("attribution", ""), "year": year_of(r)})

    # Rule 1b: name rescue table (artist name -> Wikidata ID) from rows that have both
    name_to_qid = {}
    for x in recs:
        if x["artist_qid"] and x["artist"]:
            name_to_qid.setdefault(clean_name(x["artist"].split(";")[0]), x["artist_qid"])
    for x in recs:
        if not x["artist_qid"]:
            for cand in (clean_name(x["artist"].split(";")[0]), name_from_attribution(x["attribution"])):
                if cand and cand in name_to_qid:
                    x["artist_qid"], x["name_rescued"] = name_to_qid[cand], True
                    break

    qids = {x[k] for x in recs for k in ("artist_qid", "object_qid", "influence_qid") if x[k]}
    print(f"Looking up {len(qids)} Wikidata items (cached ones are skipped)...")
    cache = fetch_wikidata(qids, load_cache())

    unmapped = Counter()
    rows = []
    for x in recs:
        year = x["year"]
        e = cache.get(x["artist_qid"], {})
        region = region_of(x["nationality"], e.get("countries", []))
        attr = (x["attribution"] or "").lower()
        is_copy = any(w in attr for w in COPY_WORDS)
        is_close = any(w in attr for w in CLOSE_WORDS)
        style, rule, source, flags = None, "", "", []

        # Rule 1: the artwork's own movement
        if x["object_qid"]:
            cands = styles_from(x["object_qid"], cache, year, region, unmapped)
            if cands:
                style, amb = pick_by_period(cands, year)
                rule, source = "R1", "artwork movement (Wikidata)"
                if amb:
                    flags.append("several movements")

        if style and not fits(style, year):  # painter's/artwork movement doesn't fit this painting's date
            flags.append(f"{rule} movement {style} does not fit {year}; dropped")
            style, rule, source = None, "", ""

        # Rule 2: the painter's movement (skipped for copies, Rule 3)
        if not style and x["artist_qid"] and not is_copy:
            cands = styles_from(x["artist_qid"], cache, year, region, unmapped)
            if cands:
                style, amb = pick_by_period(cands, year)
                rule = "R2b" if x.get("name_rescued") else ("R2c" if is_close else "R2a")
                source = {"R2a": "painter's movement", "R2b": "painter's movement (name match)",
                          "R2c": "attributed/workshop: master's movement"}[rule]
                if amb:
                    flags.append("several movements")

        if style and not fits(style, year):  # painter's/artwork movement doesn't fit this painting's date
            flags.append(f"{rule} movement {style} does not fit {year}; dropped")
            style, rule, source = None, "", ""

        # Rule 2c: Rijksmuseum workshop/circle rows whose master is stored as an influence
        if not style and x["influence_qid"] and is_close and not is_copy:
            cands = styles_from(x["influence_qid"], cache, year, region, unmapped)
            if cands:
                style, amb = pick_by_period(cands, year)
                rule, source = "R2c", "attributed/workshop: master's movement"

        if style and not fits(style, year):  # painter's/artwork movement doesn't fit this painting's date
            flags.append(f"{rule} movement {style} does not fit {year}; dropped")
            style, rule, source = None, "", ""

        # Rule 7: date fallback
        if not style:
            style, why = date_rule(year, region)
            if style:
                rule, source = "R7", "date rule" + (" (copy)" if is_copy else "")
            else:
                rule, source = "R8", "unlabeled"
                flags.append(why)

        # Rule 6 check: label far outside its style's period -> flag for review
        if style and year and style in PERIODS:
            lo, hi = PERIODS[style]
            if year < lo - 30 or year > hi + 30:
                flags.append(f"date {year} outside {style} {lo}-{hi}")

        rows.append({"museum": x["museum"], "id": x["id"], "object_number": x["object_number"],
                     "title": x["title"], "artist": x["artist"], "attribution": x["attribution"],
                     "year": year if year is not None else "", "artist_wikidata": x["artist_qid"],
                     "region": region, "style": style or "", "rule_id": rule, "source": source,
                     "flags": "; ".join(flags)})

    fields = list(rows[0].keys())
    with (OUT / "museum_labels.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)

    # Counts per style per museum
    counts = defaultdict(Counter)
    for r in rows:
        counts[r["style"] or "(unlabeled)"][r["museum"]] += 1
    with (OUT / "style_counts.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["style", "met", "rijks", "total"])
        for s, c in sorted(counts.items(), key=lambda kv: -sum(kv[1].values())):
            w.writerow([s, c["met"], c["rijks"], c["met"] + c["rijks"]])

    with (OUT / "unmapped_movements.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["wikidata_movement", "paintings_affected"])
        w.writerows(unmapped.most_common())

    # Report
    print(f"\nLabeled {sum(1 for r in rows if r['style'])} of {len(rows)} paintings")
    print("\nBy rule:")
    for k, c in sorted(Counter(r["rule_id"] for r in rows).items()):
        print(f"  {k:<4} {c:>5}")
    print(f"\n{'Style':<32}{'Met':>6}{'Rijks':>7}")
    for s, c in sorted(counts.items(), key=lambda kv: -sum(kv[1].values())):
        print(f"  {s:<30}{c['met']:>6}{c['rijks']:>7}")
    print(f"\nFlagged for review: {sum(1 for r in rows if r['flags'] and r['style'])}")
    if unmapped:
        print(f"Unmapped Wikidata movements: {len(unmapped)} (see unmapped_movements.csv). Top 10:")
        for m, c in unmapped.most_common(10):
            print(f"  {c:>5}  {m}")

    log = OUT / "labeling_log.json"
    hist = json.loads(log.read_text()) if log.exists() else []
    hist.append({"at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                 "paintings": len(rows), "labeled": sum(1 for r in rows if r["style"]),
                 "by_rule": dict(Counter(r["rule_id"] for r in rows))})
    log.write_text(json.dumps(hist, indent=2))


if __name__ == "__main__":
    main()
