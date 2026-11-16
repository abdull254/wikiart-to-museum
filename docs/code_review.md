# Code review: WikiArt-to-museum pipeline (2026-10-26)

Overall the pipeline is solid: every step is cached, re-runnable, logs its counts, and records a reason for every excluded row. The fixes below are ordered by how much they affect results.

## 1. Out-of-period labels are flagged but never removed (affects the test set)

`label_museums.py` writes a flag like `date 1890 outside Baroque 1590-1730`, but nothing downstream acts on it. 16 flagged paintings are in `museum_test.csv` (Kneller 1692 as Realism, Alma-Tadema 1890 as Baroque, Jan Victors 1648 as Realism, ...). Three more conflicts are not flagged because they sit inside the 30-year margin.

Cause: `pick_by_period` returns a painter's only movement even when it doesn't fit the painting's date.

Fix in `label_museums.py`, so an ill-fitting painter movement falls through to the date rule (R7) or to "unlabeled":

```python
def fits(style, year, margin=30):
    lo, hi = PERIODS.get(style, (-9999, 9999))
    return year is None or lo - margin <= year <= hi + margin

# Rule 1 and Rule 2 (and 2c): after pick_by_period(...)
if style and not fits(style, year):
    flags.append(f"{rule} movement {style} does not fit {year}; dropped")
    style, rule, source = None, "", ""
```

Then rerun `label_museums.py` and `prepare_dataset.py all`, and retrain. Add one line to `labeling_rules.md` under R6: "A movement more than 30 years outside its period is not used."

## 2. Hand check is not a human check yet

`hand_check.csv` notes all start with "AI-draft". Fill `correct` yourself (ideally a second person too), then run `python prepare_dataset.py report`. Rename `note` to `ai_suggestion` so the two are not confused.

## 3. Top-1 / top-3 accuracy (instructor asked for both)

`train.py` already saves every class probability, so top-3 can be computed from existing `preds_*.csv` files without retraining. Add to `score()`:

```python
top3 = np.argsort(-probs, axis=1)[:, :3]
out["top3_accuracy"] = float((top3 == y[:, None]).any(axis=1).mean())
```

`accuracy` in `score()` is already top-1.

## 4. Logs append every run

`pipeline_log.json`, `harvest_log.json`, `labeling_log.json` keep earlier runs, so readers see two sets of overlap counts. Either start the log fresh on `prepare_dataset.py all`, or add a `run_id` and report only the last run.

## 5. Small things

- `check.py` docstring says `python check_rijks.py`; rename the file to `check_rijks.py`.
- `PERIODS` gives High Renaissance 1490-1530 but R5 uses 1495-1526. Pick one.
- `COPY_WORDS` uses substring matching, so "after" also matches inside other words. Use `re.search(r"\b(after|copy|...)\b", attr)`.
- `train.py` speed message hard-codes 12 epochs; use `args.epochs`.
- `wikiart_colab.ipynb` here is the original; the fixed version is now `notebooks/wikiart_colab.ipynb`, which won't copy unfinished runs to Drive.
- `RIjk.json` is one raw API page; leave it out of the repo or move it to `docs/examples/`.

## 6. `data_notes.md` placeholders to fill

- Download and harvest dates: from `harvest_log.json` and `download_log.json`; pipeline ran 2026-10-20.
- Overlap review: 46 of 60 removed pairs were the same painting or another version.
- Minimum per style: 200 in WikiArt, 30 in each museum.
- In-gallery set decision: left out (Nov 16), see data_notes.md section 3.

## 7. Still missing from the repo

- `evaluate.py` and `stats.py` (used by the notebook, not uploaded)
- `README.md`, `requirements.txt` (requests, tqdm, pillow, imagehash, numpy, pandas, scikit-learn, torch, torchvision), `.gitignore` (`data/*/images*/`, `data/*/cache/`, `*.pt`, `*.zip`, `runs/`)
- Small run outputs: each run's `config.json`, `history.json`, `metrics.json`, plus `stats_report.txt`

## 8. `wikiart_download.py` (added after first review)

- **Pin the dataset version.** `wikiart_idx` is the position in the stream, so if `Artificio/WikiArt_Full` is updated, every index (and your split file) silently points at different images. Pass `revision="<commit hash>"` to `load_dataset` and write the hash into `download_log.json`. Find it on the dataset's Hugging Face page under "Files and versions", or with `huggingface_hub.HfApi().dataset_info(repo).sha`.
- `--size 512` has no effect: the source images are already 256 x 256 and `thumbnail` never enlarges. Say "256 x 256 as distributed" in `data_notes.md`.
- Every WikiArt image is re-saved once at JPEG quality 90, while museum images are re-saved twice (download resize, then squash). Small, but mention it next to the JPEG ablation.
- Resume logic is safe: the image is written before its row, and rows already in `metadata.csv` are skipped.

## 9. `stats.py` and label outputs

`stats.py` is correct: predictions are aligned by id across seeds, the WikiArt-vs-museum gap resamples each set on its own, and the paired comparison resamples both groups together with an exact McNemar test. Additions the instructor asked for:

- **Top-3 accuracy and ECE.** Both can be computed from the `p_<style>` columns already in `preds_*.csv`. Add them to `METRICS` (top-3 needs the probabilities, so `load()` should also return them).
- **Effect sizes.** The "change in points" in `--compare` is an effect size; label it as such in the report, and add Cohen's h for the accuracy difference if you want a standard one.
- **State what the CI covers.** The bootstrap intervals resample test paintings with seeds averaged, so they cover test-set uncertainty, not training-seed variation; the seed SD is reported separately. One sentence in the methods.
- Small: `macro_f1` here skips classes absent from the truth, while `train.py` uses sklearn with all labels. They agree on the current test sets (all six styles present) but can differ on small subsets.

Label outputs:

- `unmapped_movements.csv` lists "Sienese school" (39) as unmapped, but the rules do map it; those paintings are from before 1400, so the date rule returns nothing. Write those as "outside date rule" instead of "unmapped", so the file doesn't suggest a missing rule.
- Orientalism (129) and Expressionism (29) staying unmapped is right: one is a subject, the other is outside the six styles. Say so in `labeling_rules.md`.
- `style_counts.csv` is before overlap removal (Met Baroque 491 vs 398 in the test set). Label it as pre-overlap in `data_notes.md`.
- `labeling_log.json` holds two runs; the second (5,344 labeled) is the one used.
- Commit `wikidata_cache.json`. Wikidata is edited continuously, so the cache is the only record of the movement answers your labels came from.

`wikiart_museum-...zip` is an earlier prototype (6-7 October: four styles, minimum 100, empty hand-check results), not the run results. Leave it out of the repo or put it in `archive/` with a note that it is superseded. The `My Drive/wikiart_runs/` files are still missing.

## 10. `evaluate.py`

The structure is good: one change at a time, WikiArt test as a control for the color, JPEG and grayscale changes, seen/unseen split, and saved predictions. Four points change how the ablation results should be read or run:

**a. Color normalization erases each painting's own colors.** `colornorm` rescales every image to the average WikiArt mean and std. A dark Rembrandt and a bright Rococo scene end up with the same brightness and contrast, which removes style information. That is why it lowers WikiArt test too (−4.7). The ablation should remove the *museum's* color shift and keep each painting's own colors. Do that with one transform per dataset, not per image:

```python
def dataset_color(paths, n=1000, seed=0):
    # pooled per-channel mean/std over a sample of a whole dataset
    rng = np.random.default_rng(seed)
    px = [np.asarray(Image.open(paths[i]).convert("RGB").resize((64, 64)), np.float32).reshape(-1, 3)
          for i in rng.choice(len(paths), min(n, len(paths)), replace=False)]
    px = np.concatenate(px)
    return px.mean(0), px.std(0) + 1e-6

# per museum: src = dataset_color(museum_paths), ref = dataset_color(wikiart_train_paths)
def f(img):
    a = np.asarray(img, np.float32)
    a = (a - src_mean) / src_std * ref_std + ref_mean
    return Image.fromarray(np.clip(a, 0, 255).astype(np.uint8))
```

Keep the current per-image version as a second variant (`colorwhiten`) if you want to report both.

**b. `res512` does not change resolution.** The test transform resizes everything to 224 x 224, so the 512 px originals reach the model at the same size as the 256 px ones. The 0.0 result means "resampling path doesn't matter," not "resolution doesn't matter." Since WikiArt and museum images are already both 256 x 256, resolution is matched by construction; say that. For a real resolution test, degrade instead: downsample to 128 or 64 px and back up to 256, applied to WikiArt test as well.

**c. `jpeg50` is a stress test, not matched compression.** The approved design asked for compression matched to WikiArt. Estimate WikiArt's JPEG quality from the files' quantization tables (`Image.open(p).quantization`), then re-save museum images at that quality. Keep `jpeg50` as an extra robustness row.

**d. Add a sanity check.** This would have caught the epoch-1 seed 3 checkpoint:

```python
expected = json.loads((run / "metrics.json").read_text())["wikiart_test"]["balanced_accuracy"]
# after the base wikiart_test row:
if abs(s["balanced_accuracy"] - expected) > 0.01:
    raise SystemExit(f"best.pt gives {s['balanced_accuracy']:.3f}, metrics.json says {expected:.3f}: wrong checkpoint")
```

Small:
- `crop` has no WikiArt control, because WikiArt is only available already squashed. Say so.
- Ablation predictions are saved as `preds_<set>_<variant>.csv`, but `stats.py` only reads `preds_<set>.csv`. Add a `--variant` option to `stats.py` so ablation rows also get seed means and bootstrap intervals.
- Top-3 accuracy: same two lines as in `train.py`'s `score()`.
- CORAL (the domain-adaptation baseline the instructor requires) is not here yet. It needs training, so it belongs in `train.py`, with an option to add the CORAL loss on unlabeled museum images.
