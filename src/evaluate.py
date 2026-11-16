"""
Test-time ablations: score an already-trained model on changed versions of the test images.
No retraining. Answers "why is there a gap?" by changing one thing about the images at a time.

Variants (each applied to the Met and Rijksmuseum sets; the ones marked * also to WikiArt
test, as a control, so you can tell a domain effect from the change itself):
  base         256 x 256 squashed images (the normal setting)
  crop         center square, shape kept (crop vs full frame). No WikiArt control: WikiArt
               is only available already squashed.
  colornorm    one color transform per museum: that museum's pooled channel mean/std mapped to
               WikiArt train's. Removes the museum's overall color shift, keeps each painting's
               own colors. (On WikiArt test this would be close to a no-op, so no control row.)
  colorwhiten* old per-image version: every image forced to the WikiArt average mean/std.
               This also erases each painting's own brightness and contrast.
  down128*     downsampled to 128 px and back up to 256 (resolution). Resolution is otherwise
  down64*      matched by design: WikiArt and museum images are both 256 x 256.
  jpegmatch    museum images re-saved at WikiArt's estimated JPEG quality (matched compression)
  jpeg50*      re-saved at JPEG quality 50 (compression stress test, not matched)
  gray*        converted to grayscale (how much the model relies on color)

The base WikiArt test row must match metrics.json (within 0.01), otherwise best.pt is not the
model that produced metrics.json and the script stops.

Usage:
  python evaluate.py --run runs/resnet50_seed1
  python evaluate.py --run runs/resnet50_seed1 --variants base crop gray

Writes to the run folder: ablations.csv (one row per variant x test set, including
seen/unseen artists), ablations.json, and preds_<set>_<variant>.csv for later statistics.
"""

import argparse
import io
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from train import DATA, MUSEUM, SPLIT, build_model, make_transforms, pick_device, predict, save_preds, score

ALL = ["base", "crop", "colornorm", "colorwhiten", "down128", "down64", "jpegmatch", "jpeg50", "gray"]
COLUMN = {"crop": "image_crop"}      # other variants use "image"
ALSO_WIKIART = {"colorwhiten", "down128", "down64", "jpeg50", "gray"}

# IJG standard luminance quantization table, used to estimate a JPEG file's quality setting
IJG_LUMA = [16, 11, 10, 16, 24, 40, 51, 61, 12, 12, 14, 19, 26, 58, 60, 55, 14, 13, 16, 24, 40, 57, 69, 56,
            14, 17, 22, 29, 51, 87, 80, 62, 18, 22, 37, 56, 68, 109, 103, 77, 24, 35, 55, 64, 81, 104, 113, 92,
            49, 64, 78, 87, 103, 121, 120, 101, 72, 92, 95, 98, 112, 100, 103, 99]


def reference_color(paths, n=1000, seed=0):
    """Average per-channel mean and std of WikiArt training images (the color target)."""
    rng = np.random.default_rng(seed)
    pick = rng.choice(len(paths), size=min(n, len(paths)), replace=False)
    means, stds = [], []
    for i in pick:
        a = np.asarray(Image.open(paths[i]).convert("RGB").resize((64, 64)), dtype=np.float32)
        means.append(a.reshape(-1, 3).mean(0))
        stds.append(a.reshape(-1, 3).std(0) + 1e-6)
    return np.mean(means, 0), np.mean(stds, 0)


def dataset_color(paths, n=1000, seed=0):
    """Pooled per-channel mean and std over a sample of a whole dataset."""
    rng = np.random.default_rng(seed)
    px = [np.asarray(Image.open(paths[i]).convert("RGB").resize((64, 64)), np.float32).reshape(-1, 3)
          for i in rng.choice(len(paths), size=min(n, len(paths)), replace=False)]
    px = np.concatenate(px)
    return px.mean(0), px.std(0) + 1e-6


def jpeg_quality(paths, n=300, seed=0):
    """Median estimated JPEG quality of a sample of files (from their luminance quantization tables)."""
    tables = {}
    for q in range(1, 101):
        scale = 5000 / q if q < 50 else 200 - 2 * q
        tables[q] = sum(min(max((t * scale + 50) // 100, 1), 255) for t in IJG_LUMA)
    rng = np.random.default_rng(seed)
    qs = []
    for i in rng.choice(len(paths), size=min(n, len(paths)), replace=False):
        qt = getattr(Image.open(paths[i]), "quantization", None)
        if qt and 0 in qt:
            total = sum(qt[0])
            qs.append(min(tables, key=lambda q: abs(tables[q] - total)))
    return int(np.median(qs)) if qs else None


def resave(img, quality):
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=quality)
    return Image.open(io.BytesIO(buf.getvalue())).convert("RGB")


def make_change(variant, ref, src=None, quality=None):
    if variant == "colornorm":
        ref_mean, ref_std = ref
        src_mean, src_std = src

        def f(img):
            a = np.asarray(img, dtype=np.float32)
            a = (a - src_mean) / src_std * ref_std + ref_mean
            return Image.fromarray(np.clip(a, 0, 255).astype(np.uint8))
        return f
    if variant == "colorwhiten":
        ref_mean, ref_std = ref

        def f(img):
            a = np.asarray(img, dtype=np.float32)
            m, s = a.reshape(-1, 3).mean(0), a.reshape(-1, 3).std(0) + 1e-6
            a = (a - m) / s * ref_std + ref_mean
            return Image.fromarray(np.clip(a, 0, 255).astype(np.uint8))
        return f
    if variant == "jpeg50":
        return lambda img: resave(img, 50)
    if variant == "jpegmatch":
        return lambda img: resave(img, quality)
    if variant in ("down128", "down64"):
        r = int(variant[4:])
        return lambda img: img.resize((r, r), Image.BICUBIC).resize(img.size, Image.BICUBIC)
    if variant == "gray":
        return lambda img: img.convert("L").convert("RGB")
    return None


class Changed(Dataset):
    def __init__(self, df, col, classes, tf, change):
        self.paths = [DATA / p for p in df[col]]
        self.labels = [classes.index(s) for s in df["style"]]
        self.ids = df["_id"].tolist()
        self.tf, self.change = tf, change

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        img = Image.open(self.paths[i]).convert("RGB")
        if self.change:
            img = self.change(img)
        return self.tf(img), self.labels[i], i


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True, help="a run folder from train.py, e.g. runs/resnet50_seed1")
    ap.add_argument("--variants", nargs="+", default=ALL, choices=ALL)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--device", default="auto", choices=["auto", "mps", "cuda", "cpu"])
    args = ap.parse_args()

    run = Path(args.run)
    config = json.loads((run / "config.json").read_text())
    classes = config["classes"]
    device = pick_device(args.device)
    model = build_model(len(classes), pretrained=False)
    model.load_state_dict(torch.load(run / "best.pt", map_location=device))
    model.to(device)
    _, tf_test = make_transforms(argparse.Namespace(grayscale_aug=0.0))

    split = pd.read_csv(SPLIT)
    split = split[split["style"].isin(classes)].copy()
    split["_id"] = split["wikiart_idx"].astype(str)
    wiki_test = split[split["split"] == "test"]
    train_paths = [DATA / p for p in split.loc[split["split"] == "train", "image"]]

    m = pd.read_csv(MUSEUM)
    m = m[m["style"].isin(classes)].copy()
    m["_id"] = m["museum"] + ":" + m["id"].astype(str)

    expected = json.loads((run / "metrics.json").read_text())["wikiart_test"]["balanced_accuracy"]
    museum_paths = {k: [DATA / p for p in m.loc[m["museum"] == k, "image"] if (DATA / p).exists()]
                    for k in ("met", "rijks")}
    ref = src = None
    if "colorwhiten" in args.variants:
        ref = reference_color(train_paths)
    if "colornorm" in args.variants:
        cref = dataset_color(train_paths)
        src = {k: dataset_color(v) for k, v in museum_paths.items()}
    quality = None
    if "jpegmatch" in args.variants:
        quality = jpeg_quality(train_paths)
        print(f"Estimated JPEG quality: WikiArt train {quality}, "
              + ", ".join(f"{k} {jpeg_quality(v)}" for k, v in museum_paths.items()))
        print("  (jpegmatch only matters if a museum's quality is higher than WikiArt's)\n")
        if quality is None:
            print("  Could not read WikiArt JPEG tables: skipping jpegmatch\n")
            args.variants = [v for v in args.variants if v != "jpegmatch"]
    rows, results = [], {}
    print(f"Model: {run.name}   device: {device}\n")
    print(f"  {'variant':<12}{'set':<14}{'n':>6}{'bal acc':>9}{'macro-F1':>10}{'ECE':>7}   seen / unseen")
    for v in args.variants:
        col = COLUMN.get(v, "image")
        sets = {"met": m[m["museum"] == "met"], "rijks": m[m["museum"] == "rijks"]}
        if v in ALSO_WIKIART or v == "base":
            sets = {"wikiart_test": wiki_test, **sets}
        for name, df in sets.items():
            if v == "colornorm":
                change = make_change(v, cref, src[name])
            else:
                change = make_change(v, ref, quality=quality)
            c = "image" if name == "wikiart_test" else col
            df = df[df[c].map(lambda p: (DATA / p).exists())]
            ds = Changed(df, c, classes, tf_test, change)
            dl = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)
            y, probs, idx = predict(model, dl, device)
            s = score(y, probs.argmax(1), probs, classes)
            row = {"variant": v, "set": name, "n": s["n"], "balanced_accuracy": s["balanced_accuracy"],
                   "macro_f1": s["macro_f1"], "ece": s["ece"], "accuracy": s["accuracy"]}
            extra = None
            if name != "wikiart_test":
                groups = df["artist_group"].values[idx]
                for g in ("seen", "unseen"):
                    mask = groups == g
                    if mask.any():
                        row[f"{g}_balanced_accuracy"] = balanced(y[mask], probs[mask].argmax(1))
                        row[f"{g}_n"] = int(mask.sum())
                extra = df[["_id", "artist_group", "rule_id"]].rename(columns={"_id": "id"})
            rows.append(row)
            results[f"{v}:{name}"] = s
            save_preds(run / f"preds_{name}_{v}.csv", ds, y, probs, idx, classes, extra)
            su = (f"{row.get('seen_balanced_accuracy', float('nan')):.3f} / "
                  f"{row.get('unseen_balanced_accuracy', float('nan')):.3f}" if name != "wikiart_test" else "")
            print(f"  {v:<12}{name:<14}{s['n']:>6}{s['balanced_accuracy']:>9.3f}{s['macro_f1']:>10.3f}"
                  f"{s['ece']:>7.3f}   {su}", flush=True)
            if v == "base" and name == "wikiart_test" and abs(s["balanced_accuracy"] - expected) > 0.01:
                raise SystemExit(f"best.pt gives {s['balanced_accuracy']:.3f}, metrics.json says "
                                 f"{expected:.3f}: wrong checkpoint, ablations would be invalid")
    pd.DataFrame(rows).to_csv(run / "ablations.csv", index=False)
    (run / "ablations.json").write_text(json.dumps(results, indent=2))
    print(f"\nSaved {run / 'ablations.csv'}")


def balanced(y, p):
    from sklearn.metrics import balanced_accuracy_score
    return float(balanced_accuracy_score(y, p))


if __name__ == "__main__":
    main()
