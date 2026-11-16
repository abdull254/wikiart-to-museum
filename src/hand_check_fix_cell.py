# Fix a wrong click in the hand check. Paste into a NEW Colab cell and run it.
# Step 1: run as is -> it lists every painting you answered "no" (numbered).
# Step 2: put the numbers you want to change into FIX, e.g. FIX = {3: "yes"}, and run again.
#         Allowed answers: "yes", "no", "unsure".

FIX = {}   # e.g. {3: "yes", 7: "unsure"}

import pandas as pd
from pathlib import Path
HAND = Path("/content/data/final/hand_check.csv")
HUMAN = Path("/content/drive/MyDrive/hand_check_human.csv")

hc = pd.read_csv(HAND, dtype={"id": str})
hu = pd.read_csv(HUMAN, dtype={"id": str})
hu["n"] = range(1, len(hu) + 1)  # click order

for n, ans in FIX.items():
    hu.loc[hu.n == n, "correct"] = ans
    if ans != "no":
        hu.loc[hu.n == n, "correct_style"] = ""
hu.drop(columns="n").to_csv(HUMAN, index=False)

hc = hc.drop(columns=[c for c in ["correct", "correct_style"] if c in hc.columns]) \
       .merge(hu[["museum", "id", "correct", "correct_style", "n"]], on=["museum", "id"], how="left")
hc.drop(columns="n").to_csv(HAND, index=False)

nos = hc[hc.correct == "no"].sort_values("n")
print(f"{len(nos)} paintings answered 'no':\n")
for _, r in nos.iterrows():
    print(f"  #{int(r.n):<4} {str(r.title)[:50]:<50} {str(r.artist)[:25]:<25} {r.year}  label: {r.derived_style}  ->  you said: {r.correct_style}")
print(f"\nTotals: {hc.correct.value_counts().to_dict()}")

if FIX:
    (Path("/content/drive/MyDrive") / "hand_check.csv").write_bytes(HAND.read_bytes())
    from google.colab import files
    files.download(str(HAND))
    print("Saved. Downloaded the fixed hand_check.csv (copy also in My Drive).")
