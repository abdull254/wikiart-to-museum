"""Flag museum images that are black-and-white photos rather than colour images.

    python find_bw_images.py data/final/museum_test.csv --root data --out data/final/bw_images.csv

An image counts as black-and-white when its mean colour saturation is below --thresh
(0-255 scale; 12 by default). Check the flagged list by eye and adjust the threshold
if needed. Report results with and without these images.
"""
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image

ap = argparse.ArgumentParser()
ap.add_argument("csv")
ap.add_argument("--root", default="data")
ap.add_argument("--col", default="image")
ap.add_argument("--thresh", type=float, default=12)
ap.add_argument("--out", default="bw_images.csv")
args = ap.parse_args()

df = pd.read_csv(args.csv)
sat = []
for rel in df[args.col]:
    p = Path(args.root) / rel
    if not p.exists():
        sat.append(np.nan)
        continue
    hsv = np.asarray(Image.open(p).convert("RGB").resize((128, 128)).convert("HSV"))
    sat.append(float(hsv[..., 1].mean()))
df["saturation"] = sat
df["bw"] = df.saturation < args.thresh

print(f"missing images: {df.saturation.isna().sum()}")
by = [c for c in ["museum", "style", "derived_style"] if c in df.columns][:2]
print(df.groupby(by[:1] or [lambda _: "all"]).bw.agg(["sum", "count"]))
df[df.bw].sort_values("saturation").to_csv(args.out, index=False)
print(f"{df.bw.sum()} black-and-white images written to {args.out}")
