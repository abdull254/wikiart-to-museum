"""
Train and evaluate a style classifier on the artist-disjoint WikiArt split.

Model: ResNet-50 pretrained on ImageNet, all layers fine-tuned (the main baseline).
Tuning uses the WikiArt validation set only. The museum sets are scored only when you
pass --eval-museums, so you can't tune on them by accident.

Usage (inside your venv):
  pip install torch torchvision scikit-learn pandas
  python train.py --quick                      # test run: small subset, 1 epoch, ~minutes
  python train.py --seed 1                     # full run
  python train.py --seed 1 --eval-museums      # full run + Met and Rijksmuseum scores
  python train.py --seed 1 --grayscale-aug 0.2 # later: grayscale-augmentation test
  python train.py --seed 1 --color-jitter      # colour-jitter test (training images only)

Each run writes to runs/<name>/:
  config.json     every setting, the device, library versions and the split file's hash
  history.json    loss and validation scores per epoch
  best.pt         weights from the epoch with the best validation balanced accuracy
  metrics.json    final scores per evaluation set
  preds_<set>.csv one row per image: true style, predicted style, confidence, all class
                  probabilities (needed later for bootstrap intervals and McNemar's test)
"""

import argparse
import hashlib
import json
import platform
import random
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, f1_score
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
SPLIT = DATA / "final" / "wikiart_split.csv"
MUSEUM = DATA / "final" / "museum_test.csv"
STYLES = DATA / "final" / "final_styles.csv"
RUNS = ROOT / "runs"
MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]   # ImageNet statistics


# ---------------------------------------------------------------------------
# setup
# ---------------------------------------------------------------------------

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def pick_device(choice="auto"):
    if choice != "auto":
        return torch.device(choice)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")      # Apple Silicon GPU
    return torch.device("cpu")


def file_hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]


def load_classes():
    s = pd.read_csv(STYLES)
    return sorted(s.loc[s["kept"] == "yes", "style"].tolist())


# ---------------------------------------------------------------------------
# data
# ---------------------------------------------------------------------------

class Images(Dataset):
    def __init__(self, df, image_col, classes, tf):
        self.paths = [DATA / p for p in df[image_col]]
        self.labels = [classes.index(s) for s in df["style"]]
        self.ids = df["_id"].tolist()
        self.tf = tf

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        img = Image.open(self.paths[i]).convert("RGB")
        return self.tf(img), self.labels[i], i


def make_transforms(args):
    train = [transforms.RandomResizedCrop(224, scale=(0.7, 1.0), ratio=(0.9, 1.1)),
             transforms.RandomHorizontalFlip()]
    if args.grayscale_aug > 0:
        train.append(transforms.RandomGrayscale(p=args.grayscale_aug))
    if args.color_jitter:   # WikiArt training images only; val, test and museum images are unchanged
        train.append(transforms.ColorJitter(brightness=0.2, saturation=0.2))
    train += [transforms.ToTensor(), transforms.Normalize(MEAN, STD)]
    # Images are already 256 x 256 squares: resize the whole image, no cropping
    test = [transforms.Resize((224, 224)), transforms.ToTensor(), transforms.Normalize(MEAN, STD)]
    return transforms.Compose(train), transforms.Compose(test)


def loader(ds, args, shuffle):
    return DataLoader(ds, batch_size=args.batch_size, shuffle=shuffle, num_workers=args.workers,
                      pin_memory=torch.cuda.is_available(), persistent_workers=args.workers > 0)


# ---------------------------------------------------------------------------
# model
# ---------------------------------------------------------------------------

def build_model(n_classes, pretrained):
    weights = models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
    m = models.resnet50(weights=weights)
    m.fc = nn.Linear(m.fc.in_features, n_classes)
    return m


# ---------------------------------------------------------------------------
# metrics
# ---------------------------------------------------------------------------

def ece(conf, correct, bins=15):
    """Expected calibration error: gap between confidence and accuracy, averaged over bins."""
    edges = np.linspace(0, 1, bins + 1)
    e = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            e += m.mean() * abs(correct[m].mean() - conf[m].mean())
    return float(e)


def score(y, p, probs, classes):
    conf = probs.max(1)
    out = {
        "n": int(len(y)),
        "accuracy": float((y == p).mean()),
        "balanced_accuracy": float(balanced_accuracy_score(y, p)),
        "macro_f1": float(f1_score(y, p, average="macro", labels=range(len(classes)), zero_division=0)),
        "ece": ece(conf, (y == p).astype(float)),
        "per_class_recall": {},
        "confusion_matrix": confusion_matrix(y, p, labels=range(len(classes))).tolist(),
    }
    for k, c in enumerate(classes):
        m = y == k
        out["per_class_recall"][c] = float((p[m] == k).mean()) if m.any() else None
    return out


@torch.no_grad()
def predict(model, dl, device):
    model.eval()
    ys, ps, idx = [], [], []
    for x, y, i in dl:
        logits = model(x.to(device, non_blocking=True))
        ps.append(torch.softmax(logits.float(), 1).cpu().numpy())
        ys.append(y.numpy())
        idx.append(i.numpy())
    return np.concatenate(ys), np.concatenate(ps), np.concatenate(idx)


def save_preds(path, ds, y, probs, idx, classes, extra=None):
    df = pd.DataFrame({"id": [ds.ids[i] for i in idx],
                       "true": [classes[k] for k in y],
                       "pred": [classes[k] for k in probs.argmax(1)],
                       "confidence": probs.max(1).round(4)})
    for k, c in enumerate(classes):
        df[f"p_{c}"] = probs[:, k].round(4)
    if extra is not None:
        df = df.merge(extra, on="id", how="left")
    df.to_csv(path, index=False)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--label-smoothing", type=float, default=0.1)
    ap.add_argument("--no-class-weights", action="store_true",
                    help="turn off inverse-frequency class weights in the loss")
    ap.add_argument("--grayscale-aug", type=float, default=0.0, help="probability of grayscale training images")
    ap.add_argument("--color-jitter", action="store_true",
                    help="colour jitter (brightness 0.2, saturation 0.2) on training images")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--quick", action="store_true", help="test run: 2,000 train / 500 val images, 1 epoch")
    ap.add_argument("--eval-museums", action="store_true", help="also score the Met and Rijksmuseum sets")
    ap.add_argument("--museum-image", default="image", choices=["image", "image_crop", "image_512"],
                    help="museum image version: squashed 256 (default), center crop, or 512 originals")
    ap.add_argument("--no-pretrained", action="store_true", help="random weights (for code tests only)")
    ap.add_argument("--name", default=None)
    ap.add_argument("--device", default="auto", choices=["auto", "mps", "cuda", "cpu"],
                    help="force a device (auto picks cuda, then mps, then cpu)")
    args = ap.parse_args()

    set_seed(args.seed)
    device = pick_device(args.device)
    classes = load_classes()
    name = args.name or (f"{'quick_' if args.quick else ''}resnet50"
                         f"{'_gray' + str(args.grayscale_aug) if args.grayscale_aug else ''}"
                         f"{'_cj' if args.color_jitter else ''}_seed{args.seed}")
    out = RUNS / name
    out.mkdir(parents=True, exist_ok=True)

    split = pd.read_csv(SPLIT)
    split = split[split["style"].isin(classes)].copy()
    split["_id"] = split["wikiart_idx"].astype(str)
    train_df, val_df, test_df = (split[split["split"] == s] for s in ("train", "val", "test"))
    if args.quick:
        train_df = train_df.sample(min(2000, len(train_df)), random_state=args.seed)
        val_df = val_df.sample(min(500, len(val_df)), random_state=args.seed)
        test_df = test_df.sample(min(500, len(test_df)), random_state=args.seed)
        args.epochs = 1

    tf_train, tf_test = make_transforms(args)
    dl_train = loader(Images(train_df, "image", classes, tf_train), args, shuffle=True)
    ds_val = Images(val_df, "image", classes, tf_test)
    dl_val = loader(ds_val, args, shuffle=False)

    counts = train_df["style"].value_counts()
    print(f"Device: {device}   classes: {len(classes)}   train {len(train_df)}  val {len(val_df)}  "
          f"test {len(test_df)}")
    for c in classes:
        print(f"  {c:<30} {counts.get(c, 0):>6} train images")

    model = build_model(len(classes), pretrained=not args.no_pretrained).to(device)
    if args.no_class_weights:
        weight = None
    else:   # rarer styles count more, so the model doesn't ignore Mannerism
        w = np.array([len(train_df) / (len(classes) * counts.get(c, 1)) for c in classes], dtype=np.float32)
        weight = torch.tensor(w, device=device)
    loss_fn = nn.CrossEntropyLoss(weight=weight, label_smoothing=args.label_smoothing)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    steps = args.epochs * len(dl_train)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=steps, pct_start=0.1)
    use_amp = device.type == "cuda"
    scaler = (torch.amp.GradScaler("cuda", enabled=use_amp) if hasattr(torch.amp, "GradScaler")
              else torch.cuda.amp.GradScaler(enabled=use_amp))

    config = {**vars(args), "name": name, "device": str(device), "classes": classes,
              "train_size": len(train_df), "val_size": len(val_df), "test_size": len(test_df),
              "split_file_sha256": file_hash(SPLIT), "torch": torch.__version__,
              "python": platform.python_version(),
              "started": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    (out / "config.json").write_text(json.dumps(config, indent=2))

    history, best, t0 = [], -1.0, time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        total, n, te = 0.0, 0, time.time()
        if epoch == 1:
            print("  loading the first batch of images...", flush=True)
        for b, (x, y, _) in enumerate(dl_train, 1):
            if epoch == 1 and b <= 3:
                print(f"  batch {b} loaded after {time.time() - te:.1f} s, training on it...", flush=True)
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
                loss = loss_fn(model(x), y)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            total += loss.item() * len(y)
            n += len(y)
            if epoch == 1 and b <= 3:
                print(f"  batch {b} done after {time.time() - te:.1f} s (loss {loss.item():.3f})", flush=True)
            if b == 10 and epoch == 1:
                per_batch = (time.time() - te) / 10
                full = per_batch * (len(split[split['split'] == 'train']) / args.batch_size)
                print(f"  speed: {per_batch:.2f} s per batch -> about {full / 60:.0f} min per full epoch, "
                      f"{full * 12 / 3600:.1f} h for a 12-epoch run", flush=True)
            if b % 50 == 0:
                print(f"  epoch {epoch} batch {b}/{len(dl_train)}  loss {total / n:.3f}")
        y, probs, _ = predict(model, dl_val, device)
        s = score(y, probs.argmax(1), probs, classes)
        history.append({"epoch": epoch, "train_loss": total / n, "val_accuracy": s["accuracy"],
                        "val_balanced_accuracy": s["balanced_accuracy"], "val_macro_f1": s["macro_f1"],
                        "minutes": round((time.time() - te) / 60, 1)})
        print(f"epoch {epoch}: train loss {total / n:.3f} | val acc {s['accuracy']:.3f} "
              f"bal acc {s['balanced_accuracy']:.3f} macro-F1 {s['macro_f1']:.3f} "
              f"({(time.time() - te) / 60:.1f} min)")
        if s["balanced_accuracy"] > best:
            best = s["balanced_accuracy"]
            torch.save(model.state_dict(), out / "best.pt")
        (out / "history.json").write_text(json.dumps(history, indent=2))

    # Final evaluation with the best epoch's weights
    model.load_state_dict(torch.load(out / "best.pt", map_location=device))
    sets = {"wikiart_val": (val_df, "image"), "wikiart_test": (test_df, "image")}
    if args.eval_museums:
        m = pd.read_csv(MUSEUM)
        m = m[m["style"].isin(classes)].copy()
        m["_id"] = m["museum"] + ":" + m["id"].astype(str)
        for mus in ("met", "rijks"):
            sets[mus] = (m[m["museum"] == mus], args.museum_image)
    metrics = {}
    for set_name, (df, col) in list(sets.items()):
        exists = df[col].map(lambda p: (DATA / p).exists())
        if not exists.all():
            print(f"  {set_name}: skipping {int((~exists).sum())} painting(s) with no image file")
            df = df[exists]
            sets[set_name] = (df, col)
    for set_name, (df, col) in sets.items():
        ds = Images(df, col, classes, tf_test)
        y, probs, idx = predict(model, loader(ds, args, shuffle=False), device)
        metrics[set_name] = score(y, probs.argmax(1), probs, classes)
        extra = None
        if set_name in ("met", "rijks"):
            extra = df[["_id", "artist_group", "rule_id"]].rename(columns={"_id": "id"})
            for g in ("seen", "unseen"):
                mask = df["artist_group"].values[idx] == g
                if mask.any():
                    metrics[set_name][f"{g}_artists"] = score(y[mask], probs[mask].argmax(1), probs[mask], classes)
        save_preds(out / f"preds_{set_name}.csv", ds, y, probs, idx, classes, extra)
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2))

    print(f"\nDone in {(time.time() - t0) / 60:.1f} min. Results in {out}")
    print(f"  {'set':<14}{'n':>6}{'acc':>8}{'bal acc':>9}{'macro-F1':>10}{'ECE':>7}")
    for k, v in metrics.items():
        print(f"  {k:<14}{v['n']:>6}{v['accuracy']:>8.3f}{v['balanced_accuracy']:>9.3f}"
              f"{v['macro_f1']:>10.3f}{v['ece']:>7.3f}")


if __name__ == "__main__":
    main()