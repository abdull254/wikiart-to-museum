# Hand-check tool: paste this whole file into one Colab (or Jupyter) cell and run it.
#
# Shows one painting at a time with the style the rules gave it. Click
#   Yes     the style is right
#   No      the style is wrong (pick the right one in the dropdown first, or "none of the six")
#   Unsure  you can't tell (open the museum page link to read the curator's text)
# Every click is saved at once, so you can stop and rerun the cell; it resumes where you left off.
#
# The AI draft ("correct", "suggested", "note") is hidden on purpose so it doesn't steer you.
# When all rows are done, the cell writes hand_check.csv with your answers in `correct`
# and the AI's answers kept as `ai_correct` / `ai_suggestion`. Then run:
#     python prepare_dataset.py report

import pandas as pd, ipywidgets as w
from pathlib import Path
from IPython.display import display, clear_output, HTML

def locate_hand_check():
    for base in [Path("/content/data"), Path("data"), Path(".")]:  # Colab, or your repo folder
        hits = list(base.rglob("hand_check.csv")) if base.exists() else []
        if hits:
            return hits[0]
    try:  # not found: upload it from your computer (data/final/hand_check.csv in your repo)
        from google.colab import files
        print("hand_check.csv not found. Pick it from your computer:")
        name = next(iter(files.upload()))
        dest = Path("/content/data/final"); dest.mkdir(parents=True, exist_ok=True)
        (dest / "hand_check.csv").write_bytes(Path(name).read_bytes())
        return dest / "hand_check.csv"
    except ImportError:
        raise SystemExit("Set HAND to the path of hand_check.csv")

HAND = locate_hand_check()
print("Using", HAND)
# your answers, saved after every click (to Google Drive in Colab, so a disconnect loses nothing)
DRIVE = Path("/content/drive/MyDrive")
OUT = (DRIVE if DRIVE.exists() else HAND.parent) / "hand_check_human.csv"
print("Answers saved to", OUT)
# index every jpg under the data folder by file name, so image paths work wherever the zip put them
DATA = next((d for d in [Path("/content/data"), Path("data")] if d.exists()), HAND.parent)
IMAGES = {}
for p in DATA.rglob("*.jpg"):
    IMAGES.setdefault(p.name, []).append(p)
STYLES = ["Northern Renaissance", "Mannerism (Late Renaissance)", "Baroque",
          "Rococo", "Romanticism", "Realism", "none of the six"]

rows = pd.read_csv(HAND)
done = pd.read_csv(OUT) if OUT.exists() else pd.DataFrame(columns=["museum", "id", "correct", "correct_style"])
done_keys = set(zip(done.museum, done.id.astype(str)))
todo = [r for _, r in rows.iterrows() if (r.museum, str(r.id)) not in done_keys]

def find_image(rel):
    rel = Path(rel)
    cands = IMAGES.get(rel.name, [])
    best = [p for p in cands if rel.parts[0] in p.parts]  # met vs rijks
    loose = [p for p in cands if not {"met", "rijks"} & set(p.parts)]  # flat folder, no museum name
    p = (best or loose or [None])[0]
    return p.read_bytes() if p else None

img, info, status = w.Image(width=420), w.HTML(), w.HTML()
fix = w.Dropdown(options=STYLES, description="If no:")
b_yes, b_no, b_unsure = (w.Button(description=d, button_style=s)
                         for d, s in [("Yes", "success"), ("No", "danger"), ("Unsure", "warning")])
i = 0

def finish():
    global done
    h = done.assign(id=done.id.astype(str))
    out = rows.assign(id=rows.id.astype(str)).rename(
        columns={"correct": "ai_correct", "note": "ai_suggestion"}).drop(columns=["correct_style"])
    out = out.merge(h, on=["museum", "id"], how="left")
    out.to_csv(HAND, index=False)
    n = out.correct.isin(["yes", "no"]).sum()
    yes = (out.correct == "yes").sum()
    agree = (out.correct == out.ai_correct).mean()
    clear_output()
    print(f"Done. {yes}/{n} correct ({out.correct.eq('unsure').sum()} unsure).")
    print(f"You agreed with the AI draft on {agree:.0%} of rows.")
    print(f"Wrote {HAND}. Next: python prepare_dataset.py report")
    if DRIVE.exists():  # Colab: put a copy on Drive and download it to your computer
        (DRIVE / "hand_check.csv").write_bytes(HAND.read_bytes())
        print("Copied to My Drive/hand_check.csv. Put it in data/final/ on your computer.")
        try:
            from google.colab import files
            files.download(str(HAND))
        except Exception:
            pass

def show():
    if i >= len(todo):
        return finish()
    r = todo[i]
    data = find_image(r.image)
    img.value = data or b""
    missing = "" if data else f"<b style='color:red'>image not found: {r.image}</b><br>"
    info.value = (f"{missing}<h3>{r.derived_style}</h3>"
                  f"{r.title}<br><i>{r.artist}</i>, {r.year} &nbsp;·&nbsp; {r.museum} &nbsp;·&nbsp; rule {r.rule_id}<br>"
                  f"<a href='{r.museum_page}' target='_blank'>museum page</a>")
    status.value = f"{len(rows) - len(todo) + i + 1} of {len(rows)}"
    fix.value = "none of the six"

def answer(label):
    global i, done
    r = todo[i]
    style = fix.value if label == "no" else ""
    done = pd.concat([done, pd.DataFrame([{"museum": r.museum, "id": r.id,
                                           "correct": label, "correct_style": style}])])
    done.to_csv(OUT, index=False)
    i += 1
    show()

b_yes.on_click(lambda _: answer("yes"))
b_no.on_click(lambda _: answer("no"))
b_unsure.on_click(lambda _: answer("unsure"))
display(w.VBox([status, w.HBox([img, info]), fix, w.HBox([b_yes, b_no, b_unsure])]))
show()
