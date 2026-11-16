"""
Finish Phase 1: turn the three downloaded datasets into the final, clean experiment data.

Run after label_museums.py. Steps run in this order (or all at once with "all"):

  wikiart    map WikiArt styles to the project's style names; drop non-paintings
             (sketches, illustrations, posters...) and black-and-white images
  hash       perceptual hash of every candidate image (cached, so re-runs are fast)
  dedup      remove near-duplicates inside WikiArt
  overlap    remove museum paintings that are also in WikiArt (image hash + artist/title)
  styles     count paintings per style in all three datasets; keep styles above the minimum
  split      artist-disjoint 70/15/15 WikiArt split (StratifiedGroupKFold)
  seen       tag each museum painting as seen-artist or unseen-artist
  handcheck  draw 50 random labeled paintings per museum to check by hand
  report     score the hand check after you fill in the 'correct' column

Usage (inside your venv):
  pip install imagehash scikit-learn numpy
  python prepare_dataset.py all
  python prepare_dataset.py styles --min-museum 30 --min-wikiart 200   # re-pick the style list
  python prepare_dataset.py report                                     # after the hand check

Outputs go to data/final/. Every removed row keeps its reason, and every step adds its
counts to data/final/pipeline_log.json, so the paper can report exact numbers.
"""

import argparse
import csv
import json
import random
import re
import unicodedata
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import imagehash
import numpy as np
from PIL import Image
from tqdm import tqdm

BASE = Path(__file__).resolve().parent / "data"
FINAL = BASE / "final"
WIKI_META = BASE / "wikiart" / "metadata.csv"
WIKI_IMG = BASE / "wikiart" / "images"
MUSEUM_LABELS = BASE / "labels" / "museum_labels.csv"
MUSEUM_IMG = {"met": BASE / "met" / "images_256_squash", "rijks": BASE / "rijks" / "images_256_squash"}

WIKI_TABLE = FINAL / "wikiart_table.csv"
MUSEUM_TABLE = FINAL / "museum_table.csv"
HASHES = FINAL / "hashes.json"
STYLE_TABLE = FINAL / "final_styles.csv"
SPLIT = FINAL / "wikiart_split.csv"
MUSEUM_TEST = FINAL / "museum_test.csv"
HANDCHECK = FINAL / "hand_check.csv"
LOG = FINAL / "pipeline_log.json"

SEED = 42

# Same style names as label_museums.py (WikiArt naming)
PROJECT_STYLES = [
    "Proto Renaissance", "International Gothic", "Early Renaissance", "Northern Renaissance",
    "High Renaissance", "Mannerism (Late Renaissance)", "Baroque", "Rococo", "Neoclassicism",
    "Romanticism", "Academicism", "Realism", "Symbolism", "Impressionism", "Post-Impressionism",
]
# WikiArt sub-styles merged the same way label_museums.py merges Wikidata movements
MERGE = {"Tenebrism": "Baroque", "Pointillism": "Post-Impressionism",
         "Divisionism": "Post-Impressionism", "Cloisonnism": "Post-Impressionism",
         "Synthetism": "Post-Impressionism", "Biedermeier": "Romanticism"}
# WikiArt genres that are not paintings
NON_PAINTING_GENRES = ["sketch", "study", "illustration", "design", "poster", "caricature",
                       "sculpture", "photo", "installation", "graffiti", "tapestry", "mosaic",
                       "calligraphy", "advertisement", "architecture", "furniture", "stabile",
                       "mobile", "digital", "video", "performance"]
MONOCHROME_BELOW = 10.0   # colorfulness (Hasler & Suesstrunk 2003); below this = black-and-white

HASH_SAME_WIKIART = 4     # Hamming distance for near-duplicates inside WikiArt
HASH_OVERLAP = 8          # museum vs WikiArt: same painting by image alone
HASH_OVERLAP_SAME_ARTIST = 14   # looser, but only when the artist also matches
TITLE_OVERLAP = 0.6       # title word overlap (Jaccard) when the artist matches...
TITLE_MIN_WORDS = 2       # ...the museum title has at least this many meaningful words...
TITLE_MAX_HASH = 22       # ...and the two images are at least loosely similar
                          # (unrelated images average ~32; generic titles like
                          # "Madonna and Child" repeat across different paintings)
REVIEW_PAIRS = 60         # removed pairs shown on the visual review page


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def read(path):
    with path.open(encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def write(path, rows, fields=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = fields or list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def log(step, **info):
    hist = json.loads(LOG.read_text()) if LOG.exists() else []
    hist.append({"step": step, "at": datetime.now(timezone.utc).isoformat(timespec="seconds"), **info})
    LOG.write_text(json.dumps(hist, indent=2))


def exclude(row, reason):
    if not row.get("reason_excluded"):
        row["reason_excluded"] = reason


def kept(rows):
    return [r for r in rows if not r.get("reason_excluded")]


def strip_accents(s):
    return "".join(c for c in unicodedata.normalize("NFKD", s) if not unicodedata.combining(c))


PARTICLES = {"van", "de", "der", "den", "von", "di", "da", "del", "della", "la", "le", "el", "the",
             "i", "ii", "iii", "iv", "elder", "younger", "y", "du", "des", "ter", "ten", "and"}


def name_variants(name):
    """'Rembrandt (Rembrandt van Rijn)' -> [{'rembrandt'}, {'rembrandt', 'rijn'}]"""
    name = strip_accents(name or "").lower()
    parts = [re.sub(r"\(.*?\)", " ", name)] + re.findall(r"\((.*?)\)", name)
    out = []
    for p in parts:
        p = re.sub(r"^(attributed to|workshop of|circle of|follower of|copy after|after|"
                   r"style of|manner of|school of|studio of)\s+", "", p.strip())
        toks = {t for t in re.findall(r"[a-z]+", p) if t not in PARTICLES}
        if toks:
            out.append(frozenset(toks))
    return out


def same_artist(a_vars, b_vars):
    """Every word of the shorter name appears in the longer one (single words need 5+ letters)."""
    for a in a_vars:
        for b in b_vars:
            short, long_ = (a, b) if len(a) <= len(b) else (b, a)
            if short <= long_ and (len(short) >= 2 or len(next(iter(short))) >= 5):
                return True
    return False


class ArtistIndex:
    """Fast lookup: which WikiArt artist names match a museum artist name?"""

    def __init__(self, names):
        self.vars = {n: name_variants(n) for n in set(names) if n}
        self.by_token = defaultdict(set)
        for n, vs in self.vars.items():
            for v in vs:
                for t in v:
                    self.by_token[t].add(n)
        self.cache = {}

    def matches(self, name):
        if name in self.cache:
            return self.cache[name]
        mv = name_variants(name)
        cands = set().union(*(self.by_token.get(t, set()) for v in mv for t in v)) if mv else set()
        res = {n for n in cands if same_artist(mv, self.vars[n])}
        self.cache[name] = res
        return res


STOP = {"the", "a", "an", "of", "and", "in", "with", "on", "at", "to", "de", "la", "le", "van",
        "portrait", "painting", "study"}


def title_words(t):
    return {w for w in re.findall(r"[a-z]+", strip_accents(t or "").lower()) if w not in STOP and len(w) > 2}


def jaccard(a, b):
    return len(a & b) / len(a | b) if a and b else 0.0


def colorfulness(path):
    img = Image.open(path).convert("RGB").resize((64, 64))
    a = np.asarray(img, dtype=np.float32)
    rg = a[..., 0] - a[..., 1]
    yb = 0.5 * (a[..., 0] + a[..., 1]) - a[..., 2]
    return float(np.sqrt(rg.std() ** 2 + yb.std() ** 2) + 0.3 * np.sqrt(rg.mean() ** 2 + yb.mean() ** 2))


M1, M2, M4 = np.uint64(0x5555555555555555), np.uint64(0x3333333333333333), np.uint64(0x0F0F0F0F0F0F0F0F)
H01 = np.uint64(0x0101010101010101)


def popcount(x):
    x = x - ((x >> np.uint64(1)) & M1)
    x = (x & M2) + ((x >> np.uint64(2)) & M2)
    x = (x + (x >> np.uint64(4))) & M4
    return (x * H01) >> np.uint64(56)


def wiki_path(r):
    return WIKI_IMG / f"{r['wikiart_idx']}.jpg"


def museum_path(r):
    return MUSEUM_IMG[r["museum"]] / f"{r['id']}.jpg"


# ---------------------------------------------------------------------------
# steps
# ---------------------------------------------------------------------------

def step_wikiart():
    rows = read(WIKI_META)
    for r in rows:
        r["reason_excluded"] = ""
        s = (r.get("style") or "").strip()
        r["project_style"] = s if s in PROJECT_STYLES else MERGE.get(s, "")
        if not r["project_style"]:
            exclude(r, f"style not in project list ({s or 'none'})")
        g = (r.get("genre") or "").lower()
        if any(k in g for k in NON_PAINTING_GENRES):
            exclude(r, f"not a painting (genre: {r.get('genre')})")
        r["artist_key"] = " ".join(sorted(name_variants(r.get("artist", ""))[0])) \
            if name_variants(r.get("artist", "")) else (r.get("artist") or "").lower()
    todo = kept(rows)
    for r in tqdm(todo, desc="color check"):
        try:
            c = colorfulness(wiki_path(r))
            r["colorfulness"] = round(c, 1)
            if c < MONOCHROME_BELOW:
                exclude(r, "black-and-white image")
        except Exception as e:
            exclude(r, f"image unreadable ({type(e).__name__})")
    write(WIKI_TABLE, rows, list(rows[0].keys()) + ["colorfulness"])
    reasons = Counter(r["reason_excluded"].split(" (")[0] if r["reason_excluded"] else "kept" for r in rows)
    genres = Counter(r["reason_excluded"] for r in rows if r["reason_excluded"].startswith("not a painting"))
    print(f"WikiArt: {len(kept(rows))} of {len(rows)} kept")
    for k, c in reasons.most_common():
        print(f"  {c:>6}  {k}")
    print("  non-painting genres removed:", dict(genres.most_common(8)))
    log("wikiart", total=len(rows), kept=len(kept(rows)), reasons=dict(reasons))

    # Museum table starts from the labeled paintings
    m = read(MUSEUM_LABELS)
    for r in m:
        r["reason_excluded"] = "" if r["style"] else "no style label"
    # same black-and-white check as WikiArt: some museum images are old b/w photos
    for r in tqdm(kept(m), desc="museum color check"):
        try:
            c = colorfulness(museum_path(r))
            r["colorfulness"] = round(c, 1)
            if c < MONOCHROME_BELOW:
                exclude(r, "black-and-white image")
        except Exception as e:
            exclude(r, f"image unreadable ({type(e).__name__})")
    write(MUSEUM_TABLE, m, list(m[0].keys()) + ["colorfulness"])
    bw = Counter(r["museum"] for r in m if r["reason_excluded"] == "black-and-white image")
    print(f"Museums: {len(kept(m))} labeled paintings carried forward "
          f"(black-and-white removed: {dict(bw)})")
    log("museum_color", kept=len(kept(m)), black_and_white=dict(bw))


def step_hash():
    hashes = json.loads(HASHES.read_text()) if HASHES.exists() else {}
    jobs = [(f"wikiart:{r['wikiart_idx']}", wiki_path(r)) for r in kept(read(WIKI_TABLE))]
    jobs += [(f"{r['museum']}:{r['id']}", museum_path(r)) for r in kept(read(MUSEUM_TABLE))]
    todo = [(k, p) for k, p in jobs if k not in hashes]
    for i, (k, p) in enumerate(tqdm(todo, desc="phash")):
        try:
            hashes[k] = str(imagehash.phash(Image.open(p)))
        except Exception:
            hashes[k] = ""
        if i % 5000 == 4999:
            HASHES.write_text(json.dumps(hashes))
    HASHES.write_text(json.dumps(hashes))
    print(f"Hashes ready: {sum(1 for k, _ in jobs if hashes.get(k))} of {len(jobs)}")


def as_int(h):
    return np.uint64(int(h, 16))


def step_dedup():
    rows = read(WIKI_TABLE)
    hashes = json.loads(HASHES.read_text())
    live = [r for r in kept(rows) if hashes.get(f"wikiart:{r['wikiart_idx']}")]
    H = np.array([as_int(hashes[f"wikiart:{r['wikiart_idx']}"]) for r in live], dtype=np.uint64)

    # Pigeonhole: two 64-bit hashes within distance 4 share at least one of 5 pieces exactly
    widths, shifts, s = [13, 13, 13, 13, 12], [], 0
    for w in widths:
        shifts.append((s, w))
        s += w
    parent = list(range(len(live)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    pairs = set()
    for (off, w) in tqdm(shifts, desc="near-duplicates"):
        keys = (H >> np.uint64(off)) & np.uint64((1 << w) - 1)
        order = np.argsort(keys, kind="stable")
        ks = keys[order]
        starts = np.flatnonzero(np.r_[True, ks[1:] != ks[:-1]])
        ends = np.r_[starts[1:], len(ks)]
        for a, b in zip(starts, ends):
            if b - a < 2:
                continue
            idx = order[a:b]
            sub = H[idx]
            for c in range(0, len(idx), 2000):          # chunked, so big buckets fit in memory
                d = popcount(sub[c:c + 2000, None] ^ sub[None, :])
                ii, jj = np.nonzero(d <= HASH_SAME_WIKIART)
                for x, y in zip(ii + c, jj):
                    if x < y:
                        parent[find(idx[x])] = find(idx[y])
                        pairs.add((min(idx[x], idx[y]), max(idx[x], idx[y])))

    groups = defaultdict(list)
    for i in range(len(live)):
        groups[find(i)].append(i)
    removed_dup, removed_conflict = 0, 0
    for members in groups.values():
        if len(members) < 2:
            continue
        styles = {live[i]["project_style"] for i in members}
        members.sort(key=lambda i: int(live[i]["wikiart_idx"]))
        if len(styles) > 1:          # same image, different style labels: keep none
            for i in members:
                exclude(live[i], "near-duplicate with conflicting style labels")
                removed_conflict += 1
        else:
            keep = live[members[0]]["wikiart_idx"]
            for i in members[1:]:
                exclude(live[i], f"near-duplicate of wikiart {keep}")
                removed_dup += 1
    write(WIKI_TABLE, rows)
    print(f"Near-duplicate pairs found: {len(pairs)}")
    print(f"Removed {removed_dup} near-duplicates and {removed_conflict} with conflicting labels "
          f"(Hamming distance <= {HASH_SAME_WIKIART})")
    log("dedup", threshold=HASH_SAME_WIKIART, removed_duplicates=removed_dup,
        removed_conflicting=removed_conflict)


def step_overlap():
    wiki = kept(read(WIKI_TABLE))
    museum = read(MUSEUM_TABLE)
    hashes = json.loads(HASHES.read_text())
    wiki = [r for r in wiki if hashes.get(f"wikiart:{r['wikiart_idx']}")]
    W = np.array([as_int(hashes[f"wikiart:{r['wikiart_idx']}"]) for r in wiki], dtype=np.uint64)
    w_artist = [r.get("artist", "") for r in wiki]
    w_titles = [title_words(r.get("title", "")) for r in wiki]
    by_artist = defaultdict(list)
    for i, a in enumerate(w_artist):
        by_artist[a].append(i)
    index = ArtistIndex(w_artist)

    live = [r for r in kept(museum) if hashes.get(f"{r['museum']}:{r['id']}")]
    M = np.array([as_int(hashes[f"{r['museum']}:{r['id']}"]) for r in live], dtype=np.uint64)
    stats = Counter()
    review = []
    B = 128
    for start in tqdm(range(0, len(live), B), desc="museum vs WikiArt"):
        block = M[start:start + B]
        d = popcount(block[:, None] ^ W[None, :])
        best = d.argmin(axis=1)
        for k, r in enumerate(live[start:start + B]):
            dmin = int(d[k, best[k]])
            r["overlap_hash_dist"] = dmin
            same = [i for a in index.matches(r["artist"]) for i in by_artist[a]]
            r["artist_in_wikiart"] = "yes" if same else "no"
            reason = ""
            if dmin <= HASH_OVERLAP:
                reason, m = "image match", wiki[best[k]]["wikiart_idx"]
            elif same:
                tw = title_words(r["title"])
                hd = min((int(d[k, i]), i) for i in same)
                title_hits = [(jaccard(tw, w_titles[i]), i) for i in same
                              if int(d[k, i]) <= TITLE_MAX_HASH] if len(tw) >= TITLE_MIN_WORDS else []
                tj = max(title_hits) if title_hits else (0.0, None)
                if tj[0] >= TITLE_OVERLAP:
                    reason, m = "artist + title + loosely similar image", wiki[tj[1]]["wikiart_idx"]
                elif hd[0] <= HASH_OVERLAP_SAME_ARTIST:
                    reason, m = "artist + similar image", wiki[hd[1]]["wikiart_idx"]
            if reason:
                exclude(r, f"also in WikiArt ({reason}, wikiart {m})")
                stats[(r["museum"], reason)] += 1
                review.append((r, m, reason, dmin))
    write(MUSEUM_TABLE, museum, list(museum[0].keys()) + ["overlap_hash_dist", "artist_in_wikiart"])
    print("Removed from museum sets because the painting is also in WikiArt:")
    for mus in ("met", "rijks"):
        tot = sum(c for (m_, _), c in stats.items() if m_ == mus)
        detail = ", ".join(f"{why}: {c}" for (m_, why), c in stats.items() if m_ == mus)
        print(f"  {mus:<6} {tot:>5}   ({detail})")
    write_review_page(review)
    log("overlap", thresholds={"image": HASH_OVERLAP, "same_artist_image": HASH_OVERLAP_SAME_ARTIST,
                               "title_jaccard": TITLE_OVERLAP, "title_min_words": TITLE_MIN_WORDS,
                               "title_max_hash": TITLE_MAX_HASH},
        removed={f"{m}:{w}": c for (m, w), c in stats.items()})


def write_review_page(review):
    """A local web page showing removed museum/WikiArt pairs side by side, to spot-check."""
    rng = random.Random(SEED)
    by_reason = defaultdict(list)
    for item in review:
        by_reason[item[2]].append(item)
    pick = []
    per = max(1, REVIEW_PAIRS // max(1, len(by_reason)))
    for items in by_reason.values():
        pick += rng.sample(items, min(per, len(items)))
    cards = []
    for r, m, reason, dmin in pick:
        museum_img = f"../{r['museum']}/images_256_squash/{r['id']}.jpg"
        wiki_img = f"../wikiart/images/{m}.jpg"
        cards.append(f"""<div class="pair"><div class="imgs"><img src="{museum_img}" alt="">
<img src="{wiki_img}" alt=""></div><p><b>{reason}</b> (hash distance {dmin})<br>
{r['museum']} {r['object_number']}: {r['title'][:70]}<br>{r['artist'][:60]}</p>
<p class="ask">Same painting? &nbsp;yes / no</p></div>""")
    html = f"""<!doctype html><meta charset="utf-8"><title>Overlap review</title>
<style>body{{font-family:Arial,sans-serif;margin:24px;background:#f6f6f4;color:#222}}
.grid{{display:grid;grid-template-columns:repeat(auto-fill,minmax(330px,1fr));gap:16px}}
.pair{{background:#fff;border:1px solid #ddd;padding:10px}}.imgs{{display:flex;gap:6px}}
img{{width:155px;height:155px;object-fit:contain;background:#eee}}p{{font-size:13px;margin:6px 0}}
.ask{{color:#666}}</style>
<h1>Removed museum paintings: museum (left) vs WikiArt match (right)</h1>
<p>{len(pick)} random pairs out of {len(review)} removed. Count how many are truly the same painting.</p>
<div class="grid">{''.join(cards)}</div>"""
    path = FINAL / "overlap_review.html"
    path.write_text(html, encoding="utf-8")
    print(f"Visual check: open {path} in your browser")


def step_styles(min_museum, min_wikiart):
    wiki, museum = read(WIKI_TABLE), read(MUSEUM_TABLE)
    for r in wiki:
        if r["reason_excluded"].startswith("style below minimum"):
            r["reason_excluded"] = ""
    for r in museum:
        if r["reason_excluded"].startswith("style below minimum"):
            r["reason_excluded"] = ""
    cw = Counter(r["project_style"] for r in kept(wiki))
    cm = Counter(r["style"] for r in kept(museum) if r["museum"] == "met")
    cr = Counter(r["style"] for r in kept(museum) if r["museum"] == "rijks")
    table, final = [], []
    for s in PROJECT_STYLES:
        ok = cw[s] >= min_wikiart and cm[s] >= min_museum and cr[s] >= min_museum
        if ok:
            final.append(s)
        table.append({"style": s, "wikiart": cw[s], "met": cm[s], "rijks": cr[s],
                      "kept": "yes" if ok else "no",
                      "why_dropped": "" if ok else ", ".join(
                          f"{n} {c} < {mn}" for n, c, mn in [("WikiArt", cw[s], min_wikiart),
                                                             ("Met", cm[s], min_museum),
                                                             ("Rijks", cr[s], min_museum)] if c < mn)})
    for r in kept(wiki):
        if r["project_style"] not in final:
            exclude(r, "style below minimum in at least one dataset")
    for r in kept(museum):
        if r["style"] not in final:
            exclude(r, "style below minimum in at least one dataset")
    write(WIKI_TABLE, wiki)
    write(MUSEUM_TABLE, museum)
    write(STYLE_TABLE, table)
    print(f"Minimum: {min_wikiart} per style in WikiArt, {min_museum} per style in each museum\n")
    print(f"  {'Style':<30}{'WikiArt':>8}{'Met':>6}{'Rijks':>7}  Kept")
    for t in table:
        print(f"  {t['style']:<30}{t['wikiart']:>8}{t['met']:>6}{t['rijks']:>7}  {t['kept']}"
              + (f"   ({t['why_dropped']})" if t["why_dropped"] else ""))
    print(f"\nFinal style list ({len(final)}): {', '.join(final)}")
    log("styles", min_museum=min_museum, min_wikiart=min_wikiart, final_styles=final,
        counts={t["style"]: [t["wikiart"], t["met"], t["rijks"]] for t in table})


def step_split():
    from sklearn.model_selection import StratifiedGroupKFold
    wiki = kept(read(WIKI_TABLE))
    y = [r["project_style"] for r in wiki]
    groups = [r["artist_key"] for r in wiki]
    sgkf = StratifiedGroupKFold(n_splits=20, shuffle=True, random_state=SEED)
    fold = np.zeros(len(wiki), dtype=int)
    for f, (_, test_idx) in enumerate(sgkf.split(np.zeros(len(y)), y, groups)):
        fold[test_idx] = f
    out = []
    for r, f in zip(wiki, fold):
        split = "test" if f < 3 else ("val" if f < 6 else "train")   # 3/20, 3/20, 14/20
        out.append({"wikiart_idx": r["wikiart_idx"], "image": f"wikiart/images/{r['wikiart_idx']}.jpg",
                    "style": r["project_style"], "artist": r["artist"], "artist_key": r["artist_key"],
                    "title": r.get("title", ""), "split": split})
    write(SPLIT, out)
    by = defaultdict(set)
    for r in out:
        by[r["split"]].add(r["artist_key"])
    leak = (by["train"] & by["val"]) | (by["train"] & by["test"]) | (by["val"] & by["test"])
    sizes = Counter(r["split"] for r in out)
    print(f"Split (seed {SEED}): " + ", ".join(f"{s} {sizes[s]} ({sizes[s] / len(out):.0%})"
                                               for s in ("train", "val", "test")))
    print(f"Artists: train {len(by['train'])}, val {len(by['val'])}, test {len(by['test'])}; "
          f"artists in more than one split: {len(leak)}")
    ct = defaultdict(Counter)
    for r in out:
        ct[r["style"]][r["split"]] += 1
    print(f"\n  {'Style':<30}{'train':>7}{'val':>6}{'test':>6}")
    for s, c in sorted(ct.items()):
        print(f"  {s:<30}{c['train']:>7}{c['val']:>6}{c['test']:>6}")
    log("split", seed=SEED, sizes=dict(sizes), artists={k: len(v) for k, v in by.items()},
        leaked_artists=len(leak))


def step_seen():
    split = read(SPLIT)
    train_artists = ArtistIndex([r["artist"] for r in split if r["split"] == "train"])
    museum = read(MUSEUM_TABLE)
    out = []
    for r in kept(museum):
        seen = "seen" if train_artists.matches(r["artist"]) else "unseen"
        out.append({**r, "image": f"{r['museum']}/images_256_squash/{r['id']}.jpg",
                    "image_crop": f"{r['museum']}/images_256_crop/{r['id']}.jpg",
                    "image_512": f"{r['museum']}/images/{r['id']}.jpg", "artist_group": seen})
    write(MUSEUM_TEST, out)
    c = Counter((r["museum"], r["artist_group"]) for r in out)
    for m in ("met", "rijks"):
        print(f"{m:<6} test paintings: {c[(m, 'seen')] + c[(m, 'unseen')]:>5}   "
              f"seen artist {c[(m, 'seen')]:>5}   unseen artist {c[(m, 'unseen')]:>5}")
    log("seen", counts={f"{m}:{g}": n for (m, g), n in c.items()})


def step_handcheck(n):
    rows = read(MUSEUM_TEST)
    rng = random.Random(SEED)
    out = []
    # keep paintings you already checked (if still in the test set), then top up to n per museum
    old = {(r["museum"], r["id"]): r for r in read(HANDCHECK) if r.get("correct", "").strip()} \
        if HANDCHECK.exists() else {}
    style_now = {(r["museum"], r["id"]): r["style"] for r in rows}
    for m in ("met", "rijks"):
        # an answer only still counts if the painting kept the same label
        keep = [r for k, r in old.items() if k[0] == m and style_now.get(k) == r["derived_style"]][:n]
        out += keep
        pool = [r for r in rows if r["museum"] == m and (m, r["id"]) not in old]
        for r in rng.sample(pool, max(0, min(n - len(keep), len(pool)))):
            url = (f"https://www.metmuseum.org/art/collection/search/{r['id']}" if m == "met"
                   else f"https://www.rijksmuseum.nl/en/collection/{r['object_number']}")
            out.append({"museum": m, "id": r["id"], "object_number": r["object_number"],
                        "title": r["title"], "artist": r["artist"], "year": r["year"],
                        "derived_style": r["style"], "rule_id": r["rule_id"], "museum_page": url,
                        "image": r["image"], "correct": "", "correct_style": "", "note": ""})
    fields = list(dict.fromkeys(k for r in out for k in r))
    write(HANDCHECK, out, fields)
    print(f"Kept {sum(bool(r['correct']) for r in out)} answers you already gave")
    print(f"Wrote {len(out)} paintings to {HANDCHECK}")
    print("Fill in 'correct' (yes/no) for each row, and 'correct_style' + 'note' for the misses.")
    print("Then run: python prepare_dataset.py report")


def step_report():
    rows = [r for r in read(HANDCHECK) if r["correct"].strip().lower()[:1] in ("y", "n")]  # skip "unsure"
    if not rows:
        raise SystemExit("No rows filled in yet. Fill the 'correct' column in hand_check.csv first.")
    yes = lambda rs: sum(r["correct"].strip().lower().startswith("y") for r in rs)
    print(f"Checked {len(rows)} paintings: {yes(rows)} correct ({yes(rows) / len(rows):.0%})")
    for m in ("met", "rijks"):
        rs = [r for r in rows if r["museum"] == m]
        if rs:
            print(f"  {m:<6} {yes(rs)}/{len(rs)} ({yes(rs) / len(rs):.0%})")
    by_rule = defaultdict(list)
    for r in rows:
        by_rule[r["rule_id"]].append(r)
    print("By rule:")
    for k, rs in sorted(by_rule.items()):
        print(f"  {k:<4} {yes(rs)}/{len(rs)}")
    misses = [r for r in rows if not r["correct"].strip().lower().startswith("y")]
    if misses:
        print("Misses:")
        for r in misses:
            print(f"  {r['museum']} {r['object_number']}: {r['derived_style']} -> "
                  f"{r['correct_style'] or '?'}  ({r['rule_id']}) {r['note']}")
    log("handcheck_report", checked=len(rows), correct=yes(rows))


# ---------------------------------------------------------------------------

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("step", choices=["wikiart", "hash", "dedup", "overlap", "styles", "split", "seen",
                                     "handcheck", "report", "all"])
    ap.add_argument("--min-museum", type=int, default=30, help="minimum paintings per style per museum")
    ap.add_argument("--min-wikiart", type=int, default=200, help="minimum WikiArt images per style")
    ap.add_argument("--sample", type=int, default=50, help="hand-check paintings per museum")
    a = ap.parse_args()
    FINAL.mkdir(parents=True, exist_ok=True)
    steps = ["wikiart", "hash", "dedup", "overlap", "styles", "split", "seen", "handcheck"] \
        if a.step == "all" else [a.step]
    for s in steps:
        print(f"\n=== {s} ===")
        if s == "styles":
            step_styles(a.min_museum, a.min_wikiart)
        elif s == "handcheck":
            step_handcheck(a.sample)
        else:
            globals()[f"step_{s}"]()
