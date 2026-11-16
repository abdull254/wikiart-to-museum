"""
Statistics across seeds: bootstrap confidence intervals, the museum-gap test,
per-style recall, and (optionally) a paired comparison of two model groups.

Reads the preds_<set>.csv files that train.py saves in each run folder.

Usage:
  python stats.py --runs runs/resnet50_seed1 runs/resnet50_seed2 runs/resnet50_seed3
  # compare two groups on the same test images (e.g. baseline vs grayscale training):
  python stats.py --runs runs/resnet50_seed{1,2,3} --compare runs/resnet50_gray0.2_seed{1,2,3}

Method
  * Every number is the mean over seeds.
  * 95% bootstrap intervals: resample the test paintings (2,000 times), recompute each
    seed's score on the same resample, and average over seeds; the interval is the
    2.5th to 97.5th percentile.
  * Museum gap: WikiArt test vs a museum's unseen-artist paintings are different
    images, so each set is resampled on its own; the gap's interval and a two-sided
    bootstrap p-value come from the resampled differences.
  * Paired comparison (--compare): same images, so they are resampled together;
    McNemar's test (exact, on correct/incorrect) is also given per seed.
Writes stats_summary.csv, stats_per_class.csv and stats_report.txt next to the first run.
"""

import argparse
from math import comb
from pathlib import Path

import numpy as np
import pandas as pd

B = 2000
SEED = 0
SETS = ["wikiart_test", "met", "rijks"]


def balanced_acc(y, p, k):
    recalls = [(p[y == c] == c).mean() for c in range(k) if (y == c).any()]
    return float(np.mean(recalls))


def macro_f1(y, p, k):
    f = []
    for c in range(k):
        tp = np.sum((p == c) & (y == c))
        fp = np.sum((p == c) & (y != c))
        fn = np.sum((p != c) & (y == c))
        if tp + fn == 0:
            continue
        f.append(0.0 if tp == 0 else 2 * tp / (2 * tp + fp + fn))
    return float(np.mean(f))


METRICS = {"balanced_accuracy": balanced_acc, "macro_f1": macro_f1}


def load(runs, set_name, group=None):
    """Return y (n,), P (seeds, n) predictions, classes; rows aligned by id across seeds."""
    frames = []
    for r in runs:
        f = Path(r) / f"preds_{set_name}.csv"
        if not f.exists():
            return None
        frames.append(pd.read_csv(f).sort_values("id").reset_index(drop=True))
    base = frames[0]
    if group:
        keep = base["artist_group"] == group
        frames = [d[keep.values].reset_index(drop=True) for d in frames]
        base = frames[0]
    classes = sorted({c[2:] for c in base.columns if c.startswith("p_")})
    idx = {c: i for i, c in enumerate(classes)}
    y = base["true"].map(idx).to_numpy()
    P = np.stack([d["pred"].map(idx).to_numpy() for d in frames])
    return y, P, classes


def seed_mean(metric, y, P, k, sel=None):
    if sel is not None:
        y, P = y[sel], P[:, sel]
    return float(np.mean([metric(y, p, k) for p in P]))


def boot(metric, y, P, k, rng):
    n = len(y)
    return np.array([seed_mean(metric, y, P, k, rng.integers(0, n, n)) for _ in range(B)])


def fmt(x):
    return f"{100 * x:5.1f}"


def mcnemar_exact(a_right, b_right):
    b = int(np.sum(a_right & ~b_right))
    c = int(np.sum(~a_right & b_right))
    n = b + c
    if n == 0:
        return b, c, 1.0
    k = min(b, c)
    p = 2 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n
    return b, c, min(1.0, p)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--compare", nargs="+", help="a second group of runs, same seeds and order")
    a = ap.parse_args()
    rng = np.random.default_rng(SEED)
    out = Path(a.runs[0]).parent
    lines, rows, per_class = [], [], []

    def say(s=""):
        print(s)
        lines.append(s)

    say(f"Runs: {', '.join(Path(r).name for r in a.runs)}  ({len(a.runs)} seeds, {B} bootstrap samples)\n")
    say(f"{'set':<22}{'n':>6}   {'balanced accuracy':<26}{'macro-F1':<26}")
    boots = {}
    for s in SETS:
        for group in ([None] if s == "wikiart_test" else [None, "seen", "unseen"]):
            d = load(a.runs, s, group)
            if d is None:
                continue
            y, P, classes = d
            k = len(classes)
            name = s + (f" ({group})" if group else "")
            row = {"set": name, "n": len(y)}
            txt = f"{name:<22}{len(y):>6}   "
            for mname, m in METRICS.items():
                est = seed_mean(m, y, P, k)
                bs = boot(m, y, P, k, rng)
                lo, hi = np.percentile(bs, [2.5, 97.5])
                seeds = [m(y, p, k) for p in P]
                row.update({mname: est, f"{mname}_lo": lo, f"{mname}_hi": hi,
                            f"{mname}_seed_sd": float(np.std(seeds, ddof=1)) if len(seeds) > 1 else 0.0})
                txt += f"{fmt(est)} [{fmt(lo)}, {fmt(hi)}]       "
                boots[(name, mname)] = bs
            say(txt)
            rows.append(row)
            if group is None:
                for c_i, c in enumerate(classes):
                    m = y == c_i
                    if m.any():
                        per_class.append({"set": s, "style": c, "n": int(m.sum()),
                                          "recall": float(np.mean([(p[m] == c_i).mean() for p in P]))})

    say("\nMuseum gap: WikiArt test minus museum unseen-artist paintings (balanced accuracy)")
    for mus in ("met", "rijks"):
        kw, km = ("wikiart_test", "balanced_accuracy"), (f"{mus} (unseen)", "balanced_accuracy")
        if kw in boots and km in boots:
            point = [r for r in rows if r["set"] == "wikiart_test"][0]["balanced_accuracy"] - \
                    [r for r in rows if r["set"] == f"{mus} (unseen)"][0]["balanced_accuracy"]
            diff = boots[kw] - boots[km]
            lo, hi = np.percentile(diff, [2.5, 97.5])
            p = min(1.0, 2 * min(np.mean(diff <= 0), np.mean(diff >= 0)))
            say(f"  {mus:<6} gap {fmt(point)} points  95% CI [{fmt(lo)}, {fmt(hi)}]  p = {p:.3f}")
            rows.append({"set": f"gap: wikiart_test - {mus} (unseen)", "balanced_accuracy": point,
                         "balanced_accuracy_lo": lo, "balanced_accuracy_hi": hi, "p_value": p})

    say("\nSeen-artist boost (seen minus unseen, balanced accuracy)")
    for mus in ("met", "rijks"):
        ks, ku = (f"{mus} (seen)", "balanced_accuracy"), (f"{mus} (unseen)", "balanced_accuracy")
        if ks in boots and ku in boots:
            diff = boots[ks] - boots[ku]
            point = np.mean(diff)
            lo, hi = np.percentile(diff, [2.5, 97.5])
            say(f"  {mus:<6} boost {fmt(point)} points  95% CI [{fmt(lo)}, {fmt(hi)}]")

    say("\nRecall per style (mean over seeds)")
    pc = pd.DataFrame(per_class)
    if not pc.empty:
        table = pc.pivot(index="style", columns="set", values="recall")[[s for s in SETS if s in pc["set"].unique()]]
        ns = pc.pivot(index="style", columns="set", values="n")
        for style, r in table.iterrows():
            say(f"  {style:<30}" + "".join(f"{s}: {fmt(r[s])} (n={int(ns.loc[style, s])})   " for s in table.columns))

    if a.compare:
        say(f"\nPaired comparison: {Path(a.compare[0]).name}... vs {Path(a.runs[0]).name}...")
        for s in SETS:
            dA, dB = load(a.runs, s), load(a.compare, s)
            if dA is None or dB is None:
                continue
            (y, PA, classes), (_, PB, _) = dA, dB
            k, n = len(classes), len(y)
            point = seed_mean(balanced_acc, y, PB, k) - seed_mean(balanced_acc, y, PA, k)
            diffs = []
            for _ in range(B):
                i = rng.integers(0, n, n)
                diffs.append(seed_mean(balanced_acc, y, PB, k, i) - seed_mean(balanced_acc, y, PA, k, i))
            lo, hi = np.percentile(diffs, [2.5, 97.5])
            mc = [mcnemar_exact(PA[j] == y, PB[j] == y)[2] for j in range(min(len(PA), len(PB)))]
            say(f"  {s:<13} change {fmt(point)} points  95% CI [{fmt(lo)}, {fmt(hi)}]  "
                f"McNemar p per seed: {', '.join(f'{p:.3f}' for p in mc)}")

    pd.DataFrame(rows).to_csv(out / "stats_summary.csv", index=False)
    pc.to_csv(out / "stats_per_class.csv", index=False)
    (out / "stats_report.txt").write_text("\n".join(lines) + "\n")
    say(f"\nSaved stats_summary.csv, stats_per_class.csv and stats_report.txt in {out}")


if __name__ == "__main__":
    main()
