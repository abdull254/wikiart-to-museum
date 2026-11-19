"""Final analyses for the paper, computed from the committed prediction files (no GPU, no images).

Usage (from the repository root):
    python src/final_analysis.py

Reads
    results/runs/resnet50_seed{1,2,3}/preds_*.csv   ResNet-50 baseline, 3 seeds
    results/probes/preds_{clip,dinov2}_*.csv         CLIP and DINOv2 linear probes
    data/final/wikiart_split.csv, data/final/museum_test.csv   painters, label rules

Writes results/final/
    table_of_record.csv       every model x test set x artist group: bal acc, macro-F1, top-1, top-3, ECE,
                              mean +- SD over seeds and a painter-level bootstrap 95% CI
    gaps_equivalence.csv      WikiArt test minus museum (unseen artists) with 95% and 90% CIs and a TOST
                              equivalence test against a +-5 point margin
    paired_models.csv         paired model comparisons on the same paintings (painter-level paired bootstrap,
                              McNemar's exact test, effect sizes)
    label_source.csv          RQ1 split by how the museum label was made (Wikidata movement vs date rule)
    class_balanced.csv        museum scores re-weighted to the WikiArt test style mix (label shift vs image shift)
    confusions.csv            most-confused style pairs
    per_style_recall.csv      recall per style, model and test set
    merged_styles.csv         6 styles next to the post-hoc 4-group and 3-era scores
    confident_errors.csv      the 60 most confident wrong museum predictions (for the failure tally)
    rijks_mannerism.csv       every Rijksmuseum Mannerism test painting with its rule and predictions
    confusion_matrices.png, label_source.png
    final_report.txt          everything above as text
"""
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
OUT = RES / "final"
OUT.mkdir(parents=True, exist_ok=True)
B = 2000          # bootstrap samples
MARGIN = 5.0      # equivalence margin in balanced-accuracy points
RNG_SEED = 0

split = pd.read_csv(ROOT / "data/final/wikiart_split.csv")
museum = pd.read_csv(ROOT / "data/final/museum_test.csv")
museum["uid"] = museum["museum"] + ":" + museum["id"].astype(str)
CLASSES = sorted(split["style"].unique())
SHORT = {c: c.replace(" (Late Renaissance)", "") for c in CLASSES}
C = len(CLASSES)
IDX = {c: i for i, c in enumerate(CLASSES)}

# painter of every test painting; anonymous paintings each count as their own cluster
ARTIST = dict(zip(split["wikiart_idx"].astype(str), "w:" + split["artist_key"].astype(str)))
for uid, a in zip(museum["uid"], museum["artist"]):
    anon = (not isinstance(a, str)) or a.strip().lower() in ("", "unknown", "anonymous", "anoniem", "onbekend")
    ARTIST[uid] = f"anon:{uid}" if anon else "m:" + a
IMG2UID = dict(zip(split["image"], split["wikiart_idx"].astype(str)))
IMG2UID.update(dict(zip(museum["image"], museum["uid"])))
META = museum.set_index("uid")

SETS = {"wikiart_test": "WikiArt test", "met": "Met", "rijks": "Rijksmuseum"}
PCOLS = [f"p_{c}" for c in CLASSES]


def load_resnet(k):
    frames = []
    for s in (1, 2, 3):
        f = pd.read_csv(RES / f"runs/resnet50_seed{s}/preds_{k}.csv", dtype={"id": str})
        if k != "wikiart_test":
            f["id"] = k + ":" + f["id"].str.replace(f"{k}:", "", regex=False)
        frames.append(f.sort_values("id").reset_index(drop=True))
    assert all((f["id"].values == frames[0]["id"].values).all() for f in frames)
    return frames[0]["id"].values, frames[0]["true"].map(IDX).values, np.stack([f[PCOLS].values for f in frames])


def load_probe(name, k):
    kk = "test" if k == "wikiart_test" else k
    f = pd.read_csv(RES / f"probes/preds_{name}_{kk}.csv")
    f["id"] = f["image"].map(IMG2UID)
    f = f.sort_values("id").reset_index(drop=True)
    return f["id"].values, f["style"].map(IDX).values, f[PCOLS].values[None]


MODELS = {"ResNet-50": load_resnet, "CLIP probe": lambda k: load_probe("clip", k),
          "DINOv2 probe": lambda k: load_probe("dinov2", k)}
DATA = {}   # (model, set) -> dict(ids, y, P [S, n, C])
for m, fn in MODELS.items():
    for k in SETS:
        ids, y, P = fn(k)
        DATA[(m, k)] = dict(ids=ids, y=y, P=P)
# all models must score the same paintings
for k in SETS:
    ref = DATA[("ResNet-50", k)]
    for m in MODELS:
        assert (DATA[(m, k)]["ids"] == ref["ids"]).all(), (m, k)
        assert (DATA[(m, k)]["y"] == ref["y"]).all(), (m, k)


def group_mask(k, ids, group):
    if k == "wikiart_test" or group == "all":
        return np.ones(len(ids), bool)
    return META.loc[ids, "artist_group"].values == group


# ---------------------------------------------------------------- metrics
def ece(y, P, bins=15):
    conf, pred = P.max(1), P.argmax(1)
    edges = np.linspace(0, 1, bins + 1)
    e = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            e += m.mean() * abs((pred[m] == y[m]).mean() - conf[m].mean())
    return e


def metrics(y, P):
    p = P.argmax(1)
    present = [c for c in range(C) if (y == c).any()]
    rec = [(p[y == c] == c).mean() for c in present]
    f1 = [2 * ((p == c) & (y == c)).sum() / max(1, (p == c).sum() + (y == c).sum()) for c in present]
    top3 = (np.argsort(-P, 1)[:, :3] == y[:, None]).any(1).mean()
    return dict(bal_acc=100 * np.mean(rec), macro_f1=100 * np.mean(f1), top1=100 * (p == y).mean(),
                top3=100 * top3, ece=ece(y, P))


class PainterBoot:
    """Painter-level (cluster) bootstrap of balanced accuracy / macro-F1 from per-painter count tables."""

    def __init__(self, y, preds, artists, seed=RNG_SEED, weights=None):
        self.ids, a = np.unique(artists, return_inverse=True)
        self.A = len(self.ids)
        self.N = np.zeros((self.A, C)); np.add.at(self.N, (a, y), 1)
        self.K, self.Q = [], []
        for p in preds:                              # one prediction vector per seed
            K = np.zeros((self.A, C)); Q = np.zeros((self.A, C))
            np.add.at(K, (a, y), (p == y)); np.add.at(Q, (a, p), 1)
            self.K.append(K); self.Q.append(Q)
        rng = np.random.default_rng(seed)
        self.W = rng.multinomial(self.A, np.ones(self.A) / self.A, size=B) if weights is None else weights

    def bal_acc(self):
        n = self.W @ self.N
        vals = [np.nanmean(np.where(n > 0, (self.W @ K) / np.where(n > 0, n, 1), np.nan), 1) for K in self.K]
        return 100 * np.mean(vals, 0)

    def macro_f1(self):
        n = self.W @ self.N
        vals = []
        for K, Q in zip(self.K, self.Q):
            tp, q = self.W @ K, self.W @ Q
            f = np.where(n > 0, 2 * tp / np.maximum(n + q, 1), np.nan)
            vals.append(np.nanmean(f, 1))
        return 100 * np.mean(vals, 0)


def ci(x, lo=2.5, hi=97.5):
    return np.percentile(x, lo), np.percentile(x, hi)


def subset(m, k, group):
    d = DATA[(m, k)]
    keep = group_mask(k, d["ids"], group)
    return d["ids"][keep], d["y"][keep], d["P"][:, keep]


report = []
def say(*a):
    s = " ".join(str(x) for x in a)
    print(s); report.append(s)


# ---------------------------------------------------------------- 1. table of record
rows, BOOT = [], {}
for m in MODELS:
    for k in SETS:
        for g in (("all",) if k == "wikiart_test" else ("all", "seen", "unseen")):
            ids, y, P = subset(m, k, g)
            per = pd.DataFrame([metrics(y, P[s]) for s in range(len(P))])
            arts = np.array([ARTIST[i] for i in ids])
            pb = PainterBoot(y, P.argmax(2), arts)
            bb = pb.bal_acc(); BOOT[(m, k, g)] = bb
            lo, hi = ci(bb)
            flo, fhi = ci(pb.macro_f1())
            rows.append(dict(model=m, set=SETS[k], group=g, n=len(y), painters=pb.A, seeds=len(P),
                             bal_acc=per.bal_acc.mean(), bal_acc_sd=per.bal_acc.std(ddof=1) if len(P) > 1 else 0.0,
                             bal_ci_low=lo, bal_ci_high=hi, macro_f1=per.macro_f1.mean(), f1_ci_low=flo, f1_ci_high=fhi,
                             top1=per.top1.mean(), top3=per.top3.mean(), ece=per.ece.mean()))
tor = pd.DataFrame(rows)
tor.round(3).to_csv(OUT / "table_of_record.csv", index=False)
say("TABLE OF RECORD (balanced accuracy = mean of per-style recall; top-1 = plain accuracy; top-3 chance = 50%)")
say("CIs: painter-level bootstrap, 2,000 resamples; ResNet-50 = mean of 3 seeds, probes = 1 deterministic run\n")
say(tor.round(1).to_string(index=False))

# ---------------------------------------------------------------- 2. gaps + equivalence
say("\nMUSEUM GAP = WikiArt test minus museum unseen artists (balanced accuracy, points)")
say(f"Painter-level bootstrap (independent painters in the two sets). Equivalence margin +-{MARGIN:.0f} points (TOST):")
say("the gap is 'equivalent to zero' if the 90% CI lies inside the margin.\n")
grows = []
for m in MODELS:
    for k in ("met", "rijks"):
        a, b = BOOT[(m, "wikiart_test", "all")], BOOT[(m, k, "unseen")]
        obs = tor[(tor.model == m) & (tor.set == "WikiArt test")].bal_acc.iloc[0] - \
              tor[(tor.model == m) & (tor.set == SETS[k]) & (tor.group == "unseen")].bal_acc.iloc[0]
        d = a - b
        p = min(1.0, 2 * min((d <= 0).mean(), (d >= 0).mean()))
        p_lo = (d <= -MARGIN).mean()            # H0: gap <= -margin
        p_hi = (d >= MARGIN).mean()             # H0: gap >= +margin
        l95, h95 = ci(d); l90, h90 = ci(d, 5, 95)
        grows.append(dict(model=m, museum=SETS[k], gap=obs, ci95_low=l95, ci95_high=h95, p_diff=p,
                          ci90_low=l90, ci90_high=h90, tost_p=max(p_lo, p_hi),
                          equivalent=bool(l90 > -MARGIN and h90 < MARGIN)))
gaps = pd.DataFrame(grows)
gaps.round(3).to_csv(OUT / "gaps_equivalence.csv", index=False)
say(gaps.round(3).to_string(index=False))

# seen-artist boost (painter-level)
say("\nSEEN-ARTIST BOOST = museum seen minus unseen artists (balanced accuracy, painter-level bootstrap)")
for m in MODELS:
    for k in ("met", "rijks"):
        d = BOOT[(m, k, "seen")] - BOOT[(m, k, "unseen")]
        obs = tor[(tor.model == m) & (tor.set == SETS[k]) & (tor.group == "seen")].bal_acc.iloc[0] - \
              tor[(tor.model == m) & (tor.set == SETS[k]) & (tor.group == "unseen")].bal_acc.iloc[0]
        lo, hi = ci(d)
        say(f"  {m:<13} {SETS[k]:<12} {obs:+5.1f}  [{lo:+5.1f}, {hi:+5.1f}]")

# ---------------------------------------------------------------- 3. paired model comparisons
say("\nPAIRED MODEL COMPARISONS on the same paintings (unseen artists for museums)")
say("diff = model A minus model B, balanced accuracy; painter-level paired bootstrap (same resample for both).")
say("McNemar exact test on top-1 correctness (ResNet-50: prediction from the mean of the 3 seeds' probabilities).")
say("Effect sizes: diff in points, Cohen's h on top-1 accuracy, discordant odds ratio b/c.\n")
prow = []
pairs = [("CLIP probe", "ResNet-50"), ("DINOv2 probe", "ResNet-50"), ("CLIP probe", "DINOv2 probe")]
for k in SETS:
    g = "all" if k == "wikiart_test" else "unseen"
    for A_, B_ in pairs:
        ids, y, PA = subset(A_, k, g); _, _, PB = subset(B_, k, g)
        arts = np.array([ARTIST[i] for i in ids])
        pa = PainterBoot(y, PA.argmax(2), arts)
        pbb = PainterBoot(y, PB.argmax(2), arts, weights=pa.W)
        d = pa.bal_acc() - pbb.bal_acc()
        obs = np.mean([metrics(y, P)["bal_acc"] for P in PA]) - np.mean([metrics(y, P)["bal_acc"] for P in PB])
        ca = PA.mean(0).argmax(1) == y; cb = PB.mean(0).argmax(1) == y
        b_, c_ = int((ca & ~cb).sum()), int((~ca & cb).sum())
        p_mc = stats.binomtest(b_, b_ + c_, 0.5).pvalue if b_ + c_ else 1.0
        h = 2 * np.arcsin(np.sqrt(ca.mean())) - 2 * np.arcsin(np.sqrt(cb.mean()))
        lo, hi = ci(d)
        prow.append(dict(set=SETS[k], group=g, model_a=A_, model_b=B_, diff=obs, ci_low=lo, ci_high=hi,
                         boot_p=min(1.0, 2 * min((d <= 0).mean(), (d >= 0).mean())),
                         only_a_right=b_, only_b_right=c_, mcnemar_p=p_mc, odds_ratio=b_ / max(c_, 1), cohens_h=h))
paired = pd.DataFrame(prow)
paired.round(4).to_csv(OUT / "paired_models.csv", index=False)
say(paired.round(3).to_string(index=False))

# ---------------------------------------------------------------- 4. label source
say("\nRQ1 BY LABEL SOURCE (museum, unseen artists). 'movement' = Wikidata artwork or painter movement (rules R1, R2a-c);")
say("'date rule' = R7 (no movement found). Balanced accuracy over the styles present in BOTH groups of that museum.\n")
lrows, lrec = [], []
SRC = {"R1": "movement", "R2a": "movement", "R2b": "movement", "R2c": "movement", "R7": "date rule"}
for k in ("met", "rijks"):
    ids0, y0, _ = subset("ResNet-50", k, "unseen")
    src0 = META.loc[ids0, "rule_id"].map(SRC).values
    shared = [c for c in range(C) if all(((y0 == c) & (src0 == s)).sum() >= 5 for s in ("movement", "date rule"))]
    for m in MODELS:
        ids, y, P = subset(m, k, "unseen")
        src = META.loc[ids, "rule_id"].map(SRC).values
        bs = {}
        for s in ("movement", "date rule"):
            keep = (src == s) & np.isin(y, shared)
            arts = np.array([ARTIST[i] for i in ids[keep]])
            pb = PainterBoot(y[keep], P[:, keep].argmax(2), arts, seed=1)
            bs[s] = pb.bal_acc()
            lo, hi = ci(bs[s])
            pr = P[:, keep].argmax(2)
            val = np.mean([np.mean([(p[y[keep] == c] == c).mean() for c in shared]) for p in pr]) * 100
            lrows.append(dict(museum=SETS[k], model=m, source=s, n=int(keep.sum()), styles="+".join(SHORT[CLASSES[c]] for c in shared),
                              bal_acc=val, ci_low=lo, ci_high=hi))
            for c in shared:
                mk = y[keep] == c
                lrec.append(dict(museum=SETS[k], model=m, source=s, style=SHORT[CLASSES[c]], n=int(mk.sum()),
                                 recall=100 * np.mean([(p[mk] == c).mean() for p in pr])))
        d = bs["movement"] - bs["date rule"]
        lo, hi = ci(d)
        lrows.append(dict(museum=SETS[k], model=m, source="movement minus date rule", n=np.nan, styles="",
                          bal_acc=lrows[-2]["bal_acc"] - lrows[-1]["bal_acc"], ci_low=lo, ci_high=hi))
lab = pd.DataFrame(lrows)
lab.round(2).to_csv(OUT / "label_source.csv", index=False)
labrec = pd.DataFrame(lrec)
labrec.round(2).to_csv(OUT / "label_source_recall.csv", index=False)
say(lab.round(1).to_string(index=False))
say("\nRecall per style by label source:")
say(labrec.pivot_table(index=["museum", "model", "style"], columns="source", values=["recall", "n"]).round(1).to_string())

# ---------------------------------------------------------------- 5. class-balanced rescoring
say("\nCLASS-BALANCED RESCORING: museum paintings re-weighted so the style mix equals WikiArt test's.")
say("Top-1 accuracy and macro-F1 depend on the style mix; balanced accuracy does not (it is the uniform-mix accuracy).\n")
prior = np.bincount(DATA[("ResNet-50", "wikiart_test")]["y"], minlength=C) / len(DATA[("ResNet-50", "wikiart_test")]["y"])
crow = []
for m in MODELS:
    w_ids, wy, wP = subset(m, "wikiart_test", "all")
    wt = np.mean([metrics(wy, P)["top1"] for P in wP]); wf = np.mean([metrics(wy, P)["macro_f1"] for P in wP])
    for k in ("met", "rijks"):
        ids, y, P = subset(m, k, "unseen")
        own = np.bincount(y, minlength=C) / len(y)
        w = prior[y] / own[y]
        t_raw, t_rw, f_raw, f_rw = [], [], [], []
        for Ps in P:
            p = Ps.argmax(1)
            t_raw.append(100 * (p == y).mean()); t_rw.append(100 * np.average(p == y, weights=w))
            f1s, f1w = [], []
            for c in range(C):
                if not (y == c).any():
                    continue
                tp = ((p == c) & (y == c)); fp = ((p == c) & (y != c)); fn = ((p != c) & (y == c))
                f1s.append(2 * tp.sum() / max(1, 2 * tp.sum() + fp.sum() + fn.sum()))
                f1w.append(2 * (w * tp).sum() / max(1e-9, 2 * (w * tp).sum() + (w * fp).sum() + (w * fn).sum()))
            f_raw.append(100 * np.mean(f1s)); f_rw.append(100 * np.mean(f1w))
        crow.append(dict(model=m, museum=SETS[k], wikiart_top1=wt, top1_raw=np.mean(t_raw), top1_wikiart_mix=np.mean(t_rw),
                         wikiart_macro_f1=wf, macro_f1_raw=np.mean(f_raw), macro_f1_wikiart_mix=np.mean(f_rw),
                         largest_style_share=100 * own.max(), largest_style=SHORT[CLASSES[own.argmax()]]))
cb = pd.DataFrame(crow)
cb.round(2).to_csv(OUT / "class_balanced.csv", index=False)
say("WikiArt test style mix: " + ", ".join(f"{SHORT[c]} {100*p:.0f}%" for c, p in zip(CLASSES, prior)))
say(cb.round(1).to_string(index=False))

# ---------------------------------------------------------------- 6. confusions, per-style recall, merged styles
say("\nPER-STYLE RECALL (%), museums = unseen artists")
rrows = []
CM = {}
for m in MODELS:
    for k in SETS:
        g = "all" if k == "wikiart_test" else "unseen"
        ids, y, P = subset(m, k, g)
        cm = np.zeros((C, C))
        for Ps in P:
            np.add.at(cm, (y, Ps.argmax(1)), 1)
        cm /= len(P)
        CM[(m, k)] = cm
        for c in range(C):
            rrows.append(dict(model=m, set=SETS[k], style=SHORT[CLASSES[c]], n=int((y == c).sum()),
                              recall=100 * cm[c, c] / max(1, cm[c].sum())))
rec = pd.DataFrame(rrows)
rec.round(2).to_csv(OUT / "per_style_recall.csv", index=False)
say(rec.pivot_table(index=["model", "style"], columns="set", values="recall").round(1).to_string())

say("\nMOST-CONFUSED STYLE PAIRS (share of the true style's paintings predicted as the other style)")
crow = []
for m in MODELS:
    for k in SETS:
        cm = CM[(m, k)]
        rn = cm / np.maximum(cm.sum(1, keepdims=True), 1)
        for i in range(C):
            for j in range(C):
                if i != j:
                    crow.append(dict(model=m, set=SETS[k], true=SHORT[CLASSES[i]], predicted=SHORT[CLASSES[j]],
                                     share=100 * rn[i, j], count=cm[i, j]))
conf = pd.DataFrame(crow).sort_values(["model", "set", "share"], ascending=[True, True, False])
conf.round(2).to_csv(OUT / "confusions.csv", index=False)
for (m, s), g in conf.groupby(["model", "set"], sort=False):
    say(f"  {m:<13} {s:<12} " + "; ".join(f"{r.true}->{r.predicted} {r.share:.0f}%" for r in g.head(4).itertuples()))

GROUPS = {
    "4 groups": {"Northern Renaissance": "Renaissance", "Mannerism (Late Renaissance)": "Renaissance",
                 "Baroque": "Baroque/Rococo", "Rococo": "Baroque/Rococo", "Romanticism": "Romanticism", "Realism": "Realism"},
    "3 eras": {"Northern Renaissance": "Renaissance", "Mannerism (Late Renaissance)": "Renaissance",
               "Baroque": "Baroque/Rococo", "Rococo": "Baroque/Rococo", "Romanticism": "19th century", "Realism": "19th century"}}
say("\nMERGED STYLES (post hoc; probabilities summed within a group). Balanced accuracy; chance 16.7 / 25 / 33.3")
mrows = []
for m in MODELS:
    for k in SETS:
        g = "all" if k == "wikiart_test" else "unseen"
        ids, y, P = subset(m, k, g)
        row = dict(model=m, set=SETS[k], six_styles=np.mean([metrics(y, Ps)["bal_acc"] for Ps in P]))
        for gname, gm in GROUPS.items():
            names = sorted(set(gm.values()))
            M = np.zeros((C, len(names)))
            for c in CLASSES:
                M[IDX[c], names.index(gm[c])] = 1
            yg = M[y].argmax(1)
            row[gname] = np.mean([metrics(yg, Ps @ M)["bal_acc"] for Ps in P])
        mrows.append(row)
merged = pd.DataFrame(mrows)
merged.round(2).to_csv(OUT / "merged_styles.csv", index=False)
say(merged.round(1).to_string(index=False))

# ---------------------------------------------------------------- 7. confident errors and Rijks Mannerism
erows = []
for k in ("met", "rijks"):
    ids, y, P = subset("ResNet-50", k, "unseen")
    Pm = P.mean(0)
    p = Pm.argmax(1); conf_ = Pm.max(1)
    _, _, PC = subset("CLIP probe", k, "unseen")
    for i in np.where(p != y)[0]:
        r = META.loc[ids[i]]
        erows.append(dict(uid=ids[i], museum=k, title=r["title"], artist=r["artist"], year=r["year"], rule_id=r["rule_id"],
                          true=SHORT[CLASSES[y[i]]], resnet_pred=SHORT[CLASSES[p[i]]], resnet_conf=conf_[i],
                          clip_pred=SHORT[CLASSES[PC[0, i].argmax()]], image=r["image_512"]))
err = pd.DataFrame(erows).sort_values("resnet_conf", ascending=False)
top = pd.concat([err[err.museum == "met"].head(30), err[err.museum == "rijks"].head(30)])
top.to_csv(OUT / "confident_errors.csv", index=False)
say(f"\nCONFIDENT ERRORS: {len(err)} wrong museum predictions (unseen artists, ResNet-50 seed mean); "
    f"{(err.resnet_conf >= 0.8).sum()} with confidence >= 0.8. The 30 most confident per museum are in confident_errors.csv.")

ids, y, P = subset("ResNet-50", "rijks", "all")
_, _, PC = subset("CLIP probe", "rijks", "all")
mi = IDX["Mannerism (Late Renaissance)"]
keep = np.where(y == mi)[0]
mrows = []
for i in keep:
    r = META.loc[ids[i]]
    mrows.append(dict(uid=ids[i], object_number=r["object_number"], title=r["title"], artist=r["artist"], year=r["year"],
                      rule_id=r["rule_id"], source=r["source"], artist_group=r["artist_group"],
                      resnet_pred=SHORT[CLASSES[P.mean(0)[i].argmax()]], clip_pred=SHORT[CLASSES[PC[0, i].argmax()]],
                      portrait="portrait" in str(r["title"]).lower() or "portret" in str(r["title"]).lower()))
rm = pd.DataFrame(mrows)
rm.to_csv(OUT / "rijks_mannerism.csv", index=False)
say(f"\nRIJKSMUSEUM MANNERISM: {len(rm)} test paintings; rules: {rm.rule_id.value_counts().to_dict()}; "
    f"portraits (by title): {int(rm.portrait.sum())}")
for col in ("resnet_pred", "clip_pred"):
    say(f"  {col}: {rm[col].value_counts().to_dict()}")
    say(f"    recall on portraits {100*(rm[rm.portrait][col]=='Mannerism').mean():.0f}%, "
        f"on other paintings {100*(rm[~rm.portrait][col]=='Mannerism').mean():.0f}%")
say("  by rule (ResNet-50 recall): " + ", ".join(f"{k} {100*(g.resnet_pred=='Mannerism').mean():.0f}% (n={len(g)})"
                                                 for k, g in rm.groupby("rule_id")))
early = rm[(rm.year < 1580) & rm.rule_id.str.startswith("R2")]
say(f"  Painter-movement labels on works dated before 1580 (the date rule R5/R7 would call a northern work of these years "
    f"Northern Renaissance): {len(early)} of {len(rm)}; ResNet-50 recall on them "
    f"{100*(early.resnet_pred=='Mannerism').mean():.0f}%, NR predictions {100*(early.resnet_pred=='Northern Renaissance').mean():.0f}%")
r7 = rm[rm.rule_id == "R7"]
say(f"  Date-rule (R7) Mannerism labels: {len(r7)}, of which portraits/group portraits by title: "
    f"{int(r7.title.str.contains('Portrait|Portret|Wardens|Company|Burgomaster|Hortensia|Fourmenois|Gualtero', case=False).sum())}")
# AI-assisted visual pre-check of the 72 images (contact sheets, row order of this table); the author's own
# hand check goes in your_check. M = looks like a Mannerist history/allegory painting, G = grisaille (near monochrome),
# P = portrait or group portrait, I = Italian religious work that looks High Renaissance, O = other (seascape, genre,
# Northern-Renaissance-looking panel).
PRE = {**{i: "M" for i in (1, 4, 5, 6, 7, 8, 9, 10, 11, 21, 24, 30, 34, 35, 37, 38, 39, 46, 47, 56, 59, 60, 61, 62, 66, 70)},
       **{i: "G" for i in (12, 13, 14, 15, 68, 69)},
       **{i: "I" for i in (28, 29, 32, 33, 40, 51)}, **{i: "O" for i in (22, 25, 31, 45, 67)}}
rm["ai_precheck"] = [PRE.get(i, "P") for i in range(len(rm))]
say("  AI-assisted visual pre-check (M Mannerist-looking history, G grisaille, P portrait, I Italian High-Renaissance-looking, O other):")
for code, g in rm.groupby("ai_precheck"):
    say(f"    {code}: n={len(g):>2}  ResNet-50 recall {100*(g.resnet_pred=='Mannerism').mean():3.0f}%  "
        f"CLIP {100*(g.clip_pred=='Mannerism').mean():3.0f}%  ResNet predictions {g.resnet_pred.value_counts().to_dict()}")
rm["your_check"] = ""   # for the hand check: yes / no / unsure
rm.to_csv(OUT / "rijks_mannerism.csv", index=False)
met_m = META[(META.museum == "met") & (META["style"] == "Mannerism (Late Renaissance)")]
say(f"  Met Mannerism for comparison: n={len(met_m)}, rules {met_m.rule_id.value_counts().to_dict()}, "
    f"region {met_m.region.value_counts().to_dict()}")
rij_m = META[(META.museum == "rijks") & (META["style"] == "Mannerism (Late Renaissance)")]
say(f"  Rijks Mannerism region {rij_m.region.value_counts().to_dict()}")
wm = split[(split["style"] == "Mannerism (Late Renaissance)") & (split.split == "train")]
say(f"  WikiArt train Mannerism: {len(wm)} paintings, top painters: " + ", ".join(wm.artist.value_counts().head(8).index))

# ---------------------------------------------------------------- 7b. WikiArt test painters (DINOv2 Met > WikiArt)
wt = split[split.split == "test"]
say(f"\nWIKIART TEST PAINTERS: {wt.artist.nunique()} painters in total; per style: "
    + ", ".join(f"{SHORT[c]} {wt[wt['style'] == c].artist.nunique()}" for c in CLASSES))
say("Recall per painter on WikiArt test for the styles where DINOv2 does better at the Met (painters with >= 20 paintings):")
for c in ("Northern Renaissance", "Mannerism (Late Renaissance)"):
    for m in MODELS:
        ids, y, P = subset(m, "wikiart_test", "all")
        arts = split.set_index(split.wikiart_idx.astype(str)).loc[ids, "artist"].values
        keep = y == IDX[c]
        parts = []
        for a in pd.Series(arts[keep]).value_counts().index:
            mk = keep & (arts == a)
            if mk.sum() >= 20:
                parts.append(f"{a} {100*np.mean([(Ps[mk].argmax(1) == IDX[c]).mean() for Ps in P]):.0f}% (n={mk.sum()})")
        say(f"  {SHORT[c]:<21} {m:<13} " + "; ".join(parts))

# ---------------------------------------------------------------- 8. figures
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 3, figsize=(13, 8.2))
    lab_ = [SHORT[c].replace("Northern Renaissance", "N. Renaiss.") for c in CLASSES]
    for r_, m in enumerate(("ResNet-50", "CLIP probe")):
        for c_, k in enumerate(SETS):
            cm = CM[(m, k)]; rn = 100 * cm / np.maximum(cm.sum(1, keepdims=True), 1)
            ax = axes[r_, c_]
            ax.imshow(rn, cmap="Blues", vmin=0, vmax=100)
            for i in range(C):
                for j in range(C):
                    ax.text(j, i, f"{rn[i, j]:.0f}", ha="center", va="center", fontsize=8,
                            color="white" if rn[i, j] > 55 else "black")
            ax.set_xticks(range(C)); ax.set_xticklabels(lab_, rotation=40, ha="right", fontsize=8)
            ax.set_yticks(range(C)); ax.set_yticklabels(lab_ if c_ == 0 else [], fontsize=8)
            ttl = SETS[k] + ("" if k == "wikiart_test" else " (unseen)")
            ax.set_title(f"{m}: {ttl}", fontsize=10)
            if c_ == 0: ax.set_ylabel("true style")
            if r_ == 1: ax.set_xlabel("predicted style")
    fig.suptitle("Row-normalized confusion matrices (% of each true style)", fontsize=11)
    fig.tight_layout(); fig.savefig(OUT / "confusion_matrices.png", dpi=200); plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    sub = lab[lab.source.isin(["movement", "date rule"])]
    xs = np.arange(len(MODELS)); w = 0.2
    colors = {"movement": "#4C72B0", "date rule": "#DD8452"}
    for j, (k, s) in enumerate([(SETS[k], s) for k in ("met", "rijks") for s in ("movement", "date rule")]):
        v = sub[(sub.museum == k) & (sub.source == s)].set_index("model").loc[list(MODELS)]
        ax.bar(xs + (j - 1.5) * w, v.bal_acc, w, yerr=[v.bal_acc - v.ci_low, v.ci_high - v.bal_acc], capsize=2,
               color=colors[s], alpha=1.0 if k == "Met" else 0.55, label=f"{k}, {s}")
    ax.set_xticks(xs); ax.set_xticklabels(list(MODELS)); ax.set_ylabel("balanced accuracy (%)")
    ax.set_ylim(0, 100); ax.legend(fontsize=7, ncol=2, loc="upper left"); ax.set_title("Unseen artists, by label source", fontsize=10)
    fig.tight_layout(); fig.savefig(OUT / "label_source.png", dpi=200); plt.close(fig)
except ImportError:
    say("matplotlib not installed: figures skipped")

(OUT / "final_report.txt").write_text("\n".join(report) + "\n")
print("\nSaved to", OUT)
