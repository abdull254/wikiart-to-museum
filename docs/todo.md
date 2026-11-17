# To-do list: WikiArt to museum project (2026-10-26)

Ordered by priority. Details for each code item are in `code_review.md` (section number in brackets).

## A. Fix the data before retraining

- [x] Drop labels that don't fit the painting's date: add the `fits()` guard in `label_museums.py`, then rerun `label_museums.py` and `prepare_dataset.py all` (1)
- [x] (2026-10-26: 92/92 decided correct, 8 unsure) Do the hand check yourself: fill `correct` in `hand_check.csv` (second person if possible), run `prepare_dataset.py report`, rename `note` to `ai_suggestion` (2)
- [ ] Pin the WikiArt dataset version (`revision=` hash in `wikiart_download.py`, saved in the log) (8)

## B. Retrain and re-evaluate

- [x] Use the fixed Colab notebook (only copies finished runs, never overwrites)
- [x] (2026-10-27) Retrain baseline seeds 1, 2, 3 on the corrected labels; check each `best.pt` is about 94 MB
- [ ] Add top-3 accuracy to `train.py` `score()` (3)
- [x] Add the checkpoint sanity check to `evaluate.py` (10d)

## C. Fix the ablations (slide 11 needs redoing)

- [x] (script fixed 2026-10-27) Color normalization: one transform per museum, not per image (10a)
- [x] (script fixed) Resolution: say it's matched by design; for a real test, downsample to 128/64 px (10b)
- [x] (script fixed) Compression: match WikiArt's JPEG quality; keep `jpeg50` as a stress test (10c)
- [ ] Run ablations on all 3 seeds; add `--variant` to `stats.py` for intervals (10)
- [ ] Grayscale-augmentation training: seeds 2 and 3

## D. Experiments not started (instructor requires)

- [x] (run 2026-11-17, results/rq2/) RQ2 few-shot curve: 5, 10, 25, 50 per style, Met → Rijks and Rijks → Met
- [ ] RQ2 follow-up: artist-level bootstrap; per-style trade-off (Baroque down, small styles up)
- [ ] CORAL domain adaptation in `train.py`, compared with each simple fix on its own
- [x] (done 2026-10-28, see rq1_results.md) DINOv2 and CLIP linear probes (note CLIP may have seen these paintings)
- [x] Decide on an in-gallery test set: left out (Nov 16); stated as a threat to external validity and future work

## E. Statistics

- [ ] Top-3 accuracy and ECE in `stats.py` (9)
- [ ] One sentence: bootstrap intervals cover test-set uncertainty; seed SD reported separately (9)
- [ ] Paired comparisons with effect sizes (baseline vs grayscale, vs CORAL)
- [ ] Lead RQ1 with unseen-artist results

## F. Writing

- [ ] Related work: Cetinić 2018, Zhao 2021 and 2022, Liang 2025, Castellano and Vessio 2021, plus 2024–2026 work; documented search for the gap claim; credit Mashfika and Yisihaq
- [ ] Failure analysis: at least 50 confident errors tagged by cause; most-confused pairs (historical overlap or label problem)
- [ ] Threats to validity, ethics and licensing, reproducibility sections
- [ ] Same-painting test with the 728 removed pairs (optional, strong)

## G. Repo cleanup

- [x] Add `README.md`, `requirements.txt`, `.gitignore` (done 2026-11-16)
- [ ] Add `evaluate.py`, `stats.py`, run results (`config.json`, `history.json`, `metrics.json`, `preds_*.csv`, `stats_report.txt`)
- [ ] Commit `wikidata_cache.json`; leave out images, caches, `best.pt`, zips, the old prototype zip
- [ ] Fill placeholders in `data_notes.md` (dates, 46/60, minimums 200 and 30); note `style_counts.csv` is pre-overlap
- [ ] Logs: keep only the final run, or mark earlier runs superseded (4)
- [ ] Small fixes: rename `check.py` to `check_rijks.py`; align High Renaissance dates; word-boundary match for copy words; "outside date rule" instead of "unmapped" for Sienese school (5, 9)
