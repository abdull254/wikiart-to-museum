# Can an Art-Style Classifier Trained on WikiArt Recognize Museum Images?

**Author:** Abdullahi Jimale
**Course:** CIS 631, Minnesota State University, Mankato, Fall 2026
**Instructor:** Dr. Suboh Alkhushayni

## Final submission

| Item | Where |
|---|---|
| 1. Final paper (PDF) | [paper/WikiArt_to_Museum_Final_Paper.pdf](paper/WikiArt_to_Museum_Final_Paper.pdf) |
| 2. LaTeX source | [paper/main.tex](paper/main.tex), [paper/refs.bib](paper/refs.bib), figures in [paper/figures/](paper/figures/) (IEEE conference format; build with `pdflatex main && bibtex main && pdflatex main && pdflatex main`) |
| 3. Final slides | [slides/WikiArt_to_Museum_Final.pptx](slides/WikiArt_to_Museum_Final.pptx) (speaker notes on every slide) |
| 4. Video recording | Submitted on D2L |
| 5. All code | [src/](src/) (scripts) and [notebooks/](notebooks/) (Colab notebooks, most with outputs) |
| 6. README | This file |
| 7. Datasets (links) | [Datasets](#datasets-links) below, plus the ID, label and split files in [data/](data/) |

## Overview

Art-style classifiers are usually trained on WikiArt. Museums may want to use them on their own collections, but images from different sources differ in digitization, color and framing. This project measures how much a WikiArt-trained model loses on museum digitizations from the Metropolitan Museum of Art (the Met) and the Rijksmuseum, and whether a few museum images help at the other museum.

The museum images are professional collection digitizations, not photographs taken in the gallery.

## Research questions

1. **RQ1, transfer.** How much does performance change when a model trained only on WikiArt is tested on the Met and the Rijksmuseum?
2. **RQ2, few-shot.** Does fine-tuning on 5, 10, 25 or 50 images per style from one museum improve results at the other museum? Tested both ways (Met → Rijks and Rijks → Met).

## Datasets (links)

The images are not in this repository (see [Ethics and licensing](#ethics-and-licensing)). They can be rebuilt from these public sources with the scripts in `src/`:

| Dataset | Link | What we used | IDs / labels in this repo |
|---|---|---|---|
| WikiArt (`Artificio/WikiArt_Full`) | https://huggingface.co/datasets/Artificio/WikiArt_Full | 26,144 paintings in the six styles (artist named for every image), 256 x 256 | [data/final/wikiart_split.csv](data/final/wikiart_split.csv) (index, artist, style, split), [data/wikiart/download_log.json](data/wikiart/download_log.json) |
| The Met Collection API (Open Access) | https://metmuseum.github.io/ (API: https://collectionapi.metmuseum.org/public/collection/v1) | European Paintings and Robert Lehman Collection, public-domain paintings | [data/met/ids.txt](data/met/ids.txt), [data/met/harvest_log.json](data/met/harvest_log.json) |
| Rijksmuseum Data Services (Search API + IIIF) | https://data.rijksmuseum.nl/ | Paintings collection (`SK-` object numbers) | [data/rijks/ids.txt](data/rijks/ids.txt), [data/rijks/harvest_log.json](data/rijks/harvest_log.json) |
| Wikidata (movement labels, P135) | https://www.wikidata.org/ | Artwork and painter movements for the museum labels | [data/labels/museum_labels.csv](data/labels/museum_labels.csv) (label, rule and source for all 6,909 museum paintings) |

Both museums were harvested on 2026-10-20. Final museum test sets: [data/final/museum_test.csv](data/final/museum_test.csv) (Met 937, Rijksmuseum 3,092). Removed overlaps: [data/final/museum_wikiart_overlap.csv](data/final/museum_wikiart_overlap.csv); WikiArt near-duplicates: [data/final/wikiart_near_duplicates.csv](data/final/wikiart_near_duplicates.csv).

## Data

| Dataset | Source | Role |
|---|---|---|
| WikiArt | Hugging Face `Artificio/WikiArt_Full` (artist named for every image) | Train, validation, in-domain test |
| The Met | Met Collection API, European Paintings and Robert Lehman Collection, public domain | Test |
| Rijksmuseum | Rijksmuseum Search API, paintings (`SK-` numbers), IIIF images | Test |

- **Paintings only.** Non-paintings, black-and-white images (including 103 B&W museum images removed on Oct 26) and near-duplicates (perceptual hash) are removed.
- **Artist-disjoint split.** WikiArt train / val / test = 18,501 / 3,786 / 3,857, with no artist in more than one split.
- **No overlap.** Museum paintings that also appear in WikiArt were removed (443 Met, 278 Rijks in the final Oct 26 run; list in `data/final/museum_wikiart_overlap.csv`).
- **Six shared styles:** Northern Renaissance, Mannerism, Baroque, Rococo, Romanticism, Realism.
- **Museum labels** come from Wikidata movements (P135), the painter's movement and a date rule. Since Oct 26 a movement that does not fit the painting's date is dropped. A blind human hand check found 92 of 92 decided labels correct (8 unsure).
- **Test sets:** Met 937 paintings, Rijksmuseum 3,092.

## Results

**Table of record (final, Nov 19):** [results/final/table_of_record.csv](results/final/table_of_record.csv), recomputed by `python src/final_analysis.py` from the committed prediction files. It uses a **painter-level bootstrap** (painters, not paintings, are resampled), so its intervals are wider than the older painting-level intervals in the sections below; the point estimates are the same. All final analyses are in [results/final/final_report.txt](results/final/final_report.txt).

### Final analyses (painter-level statistics, Nov 19)

Unseen artists, top-1 balanced accuracy (%), painter-level 95% CI:

| Model | WikiArt test | Met | Rijksmuseum | Gap Met | Gap Rijks |
|---|---:|---:|---:|---:|---:|
| ResNet-50 (3 seeds) | 61.1 [55.2, 66.0] | 57.5 [52.9, 62.1] | 58.2 [55.2, 63.6] | 3.5 [-4.4, 10.0] | 2.8 [-5.7, 8.7] |
| CLIP probe | 70.9 [65.2, 75.9] | 68.2 [62.9, 75.6] | 66.4 [62.7, 71.5] | 2.7 [-6.9, 10.1] | 4.5 [-3.5, 10.6] |
| DINOv2 probe | 58.7 [51.5, 66.8] | 61.1 [54.6, 66.4] | 57.3 [53.0, 61.0] | -2.4 [-11.1, 7.6] | 1.4 [-6.8, 10.6] |

- **Equivalence (TOST, +-5 points):** not shown for any model; the 90% intervals reach 6 to 10 points. A gap above about 10 points is ruled out for every model. ([gaps_equivalence.csv](results/final/gaps_equivalence.csv))
- **Seen-artist boost at the Met:** ResNet-50 +12.4 [1.2, 22.0], CLIP +12.1 [0.2, 21.9], DINOv2 +12.0 [2.0, 24.6].
- **Paired model tests** ([paired_models.csv](results/final/paired_models.csv)): CLIP beats ResNet-50 by +10.6 [4.7, 18.9] at the Met and +8.2 [3.5, 12.3] at the Rijksmuseum (McNemar p = 0.055 and < 0.001); DINOv2 is not better than ResNet-50.
- **Label shift:** re-weighting each museum to WikiArt's style mix raises Rijksmuseum macro-F1 from 46.0 to 56.2 (ResNet-50) and 55.5 to 66.2 (CLIP), so its low macro-F1 comes from the Baroque-heavy mix ([class_balanced.csv](results/final/class_balanced.csv)).
- **Label source:** Wikidata-movement vs date-rule labels differ by -3.0 to +7.1 points, all CIs include zero ([label_source.csv](results/final/label_source.csv)).
- **DINOv2 Met > WikiArt:** not significant; WikiArt test has only 8 Northern Renaissance painters and DINOv2 gets Holbein (107 of 221 paintings) right only 56%.
- **Rijksmuseum Mannerism:** 63 of 72 are northern (Dutch) works, while WikiArt's Mannerism is Italian. ResNet-50 recall: Italian-looking works 83%, Dutch history paintings 38%, portraits 52%, grisailles 0% ([rijks_mannerism.csv](results/final/rijks_mannerism.csv)).
- **Failure tally** (60 most confident errors, [error_tally.csv](results/final/error_tally.csv)): 55% style boundary, 15% label problems (later copies, date rule), 13% subject from another period or non-European work, 12% digitization or format (frame, oval, triptych, grisaille), 5% unclear. No glare, crop or low-resolution cases.
- **ECE** for every model and test set is in the table of record (ResNet-50 0.039 WikiArt, 0.069 Met, 0.056 Rijks; CLIP 0.099 / 0.155 / 0.125).
- **Merged styles** (post hoc, next to the six-style results): [merged_styles.csv](results/final/merged_styles.csv); final ensemble 3 eras 92.1 / 87.0 / 90.9.
- **CORAL and grayscale-augmentation training:** [notebooks/final_extras.ipynb](notebooks/final_extras.ipynb); results go to `results/final_extras/`.

### Stronger models (one run, Nov 18)

Unseen artists, balanced accuracy (%), chosen on WikiArt val only ([results/improve/](results/improve/)):

| Model | WikiArt test | Met | Rijksmuseum | Met 3 eras | Rijks 3 eras |
|---|---:|---:|---:|---:|---:|
| SigLIP so400m probe | 80.6 | 70.9 | 70.9 | 90.9 | 89.1 |
| CLIP + DINOv2 + SigLIP feature ensemble | 79.4 | 72.6 | 69.3 | 91.2 | 89.2 |
| Final ensemble | 78.3 | 70.3 | 71.0 | 87.0 | 90.9 |

The older sections below keep the numbers as first reported, with painting-level intervals.

All numbers are **balanced accuracy (%)**. Chance is 16.7%. Museum columns use **unseen artists** (painters not in WikiArt training), the fair test for RQ1.

### RQ1: ResNet-50 baseline (3 seeds, fixed labels, Oct 27; painting-level CIs)

| Test set | Balanced acc. [95% CI] | Macro-F1 |
|---|---:|---:|
| WikiArt test | 61.1 [59.5, 62.6] | 59.7 |
| Met, unseen artists | 57.5 [53.8, 61.4] | 55.7 |
| Rijksmuseum, unseen artists | 58.2 [55.0, 61.5] | 46.0 |

- The museum gap is small and **not significant**: 3.5 points at the Met (p = 0.10) and 2.9 at the Rijksmuseum (p = 0.12).
- Paintings by artists seen in training score **about 12 points higher** at the Met (+12.2 [5.1, 19.4]), so the model partly recognizes painters.
- Rijksmuseum macro-F1 is low because the set is 69% Baroque; Romanticism and Mannerism are the weak styles.

### Ablations (3 seeds, test-time changes)

No digitization fix closes the gap. Matching color (`colornorm`) or JPEG quality (`jpegmatch`) changes results by less than 0.3 points. Center crop helps the Met by +2.5 (noisy). Grayscale costs 4 to 9 points everywhere, so color carries style information. Museum images are no more fragile than WikiArt. Full table: [docs/rq1_results.md](docs/rq1_results.md).

### Colour-jitter training (3 seeds, Nov 18)

ResNet-50 retrained with `ColorJitter(brightness=0.2, saturation=0.2)` on WikiArt training images only; everything else as the baseline. Mean of 3 seeds, difference vs baseline with a painter-level paired bootstrap 95% CI.

| Test set | Baseline | Colour jitter | Diff [95% CI] | Top-1 diff |
|---|---:|---:|---:|---:|
| WikiArt test | 61.1 | 60.6 | -0.5 [-4.2, 1.1] | -1.9 |
| Met, unseen artists | 57.5 | 60.3 | +2.7 [0.0, 5.0], p = 0.048 | -1.9 |
| Rijksmuseum, unseen artists | 58.2 | 59.4 | +1.2 [-0.6, 3.7], p = 0.16 | -3.1 |

- A small, borderline balanced-accuracy gain at the Met only; none of the paired t-tests over seeds is significant (p >= 0.15), and top-1 accuracy and Rijks macro-F1 (-1.3) go down.
- The gain is a trade between styles, not better recognition overall: Mannerism recall rises by 14 to 18 points while Baroque (-5 to -10) and Northern Renaissance (-6 to -9) fall. Full report: [results/colorjitter/colorjitter_report.txt](results/colorjitter/colorjitter_report.txt).

### Foundation models (frozen features + linear probe, Oct 28)

| Model | WikiArt | Met unseen | Rijks unseen |
|---|---:|---:|---:|
| ResNet-50 fine-tuned | 61.1 | 57.5 | 58.2 |
| CLIP ViT-L/14 probe | 70.9 | 68.2 | 66.4 |
| DINOv2 ViT-L/14 probe | 58.7 | 61.1 | 57.3 |
| CLIP zero-shot | 44.6 | 39.1 | 47.2 |

CLIP is best by about 10 points, but it may have seen these famous paintings during pretraining. The museum gap stays small for every model. Top-3 accuracy for the CLIP probe is about 97% on all three sets; no style reaches 97% top-1.

### RQ2: few-shot

Run on Nov 17, 2026 ([notebooks/rq2_fewshot.ipynb](notebooks/rq2_fewshot.ipynb), with outputs). Fixed test sets: painters absent from WikiArt train and from the source museum (Met → Rijks n = 2,279; Rijks → Met n = 424). Balanced accuracy (%), mean of 3 seeds (probes at k = 0: one deterministic run); gain = k = 50 minus k = 0, paired bootstrap.

| Model | Met → Rijks, k = 0 | k = 50 | gain [95% CI] | Rijks → Met, k = 0 | k = 50 | gain [95% CI] |
|---|---:|---:|---:|---:|---:|---:|
| ResNet-50 fine-tune | 57.0 | 59.5 | +2.5 [−0.1, 5.2] | 57.7 | 58.5 | +0.8 [−2.1, 3.7] |
| CLIP probe | 65.0 | 66.4 | +1.4 [−1.6, 4.3] | 67.0 | 66.4 | −0.5 [−4.2, 3.2] |
| DINOv2 probe | 58.1 | 58.3 | +0.2 [−2.8, 3.0] | 61.4 | 62.8 | +1.4 [−2.0, 4.6] |

A few labeled paintings from one museum barely help on the other: none of the 24 gains (3 models × 2 directions × k = 5, 10, 25, 50) is significant; the largest is ResNet-50, Met → Rijks, k = 50 (p = 0.06). The WikiArt-only control (same 200 steps) gives −1.6 and −0.4, and WikiArt test stays at 59–61 (no forgetting). The shots mainly move errors between styles (ResNet-50, Met → Rijks, k = 50: Romanticism recall 39 → 59, Baroque 59 → 52; ECE 0.07 → 0.15). Full table: [results/rq2/rq2_report.txt](results/rq2/rq2_report.txt); curve: [results/rq2/rq2_curve.png](results/rq2/rq2_curve.png).

## Repository layout

```
src/          Python scripts
notebooks/    Colab notebooks
docs/         results write-up, related work, code review, to-do list
results/      result files behind every table, labels
data/         split, labels, object IDs, overlap and near-duplicate lists, logs (no images)
slides/       latest presentation deck
reports/      progress reports
```

| File | What it does |
|---|---|
| `src/wikiart_download.py` | Download `Artificio/WikiArt_Full` (probe, then download) |
| `src/main.py` | Rijksmuseum harvest: IDs, metadata, IIIF images |
| `src/check.py` | Check the Rijksmuseum harvest (duplicates, missing Wikidata, errors) |
| `src/filter_rijks.py` | Keep only the Rijksmuseum paintings collection (`SK-`) |
| `src/met_harvest.py` | Met harvest: IDs, metadata, images |
| `src/squash_museums.py` | 256 px squash and center-crop versions of museum images |
| `src/pack_for_colab.py` | Pack `train.py`, `data/final/` and the images into `colab_data.zip` |
| `src/precheck_handcheck.py` | Sort the hand check by agreement with WikiArt |
| `src/check_gpu.py` | Quick PyTorch test on the Mac GPU (MPS) |
| `src/label_museums.py` | Museum style labels from Wikidata and the date rule (with the date-fit fix) |
| `src/prepare_dataset.py` | Builds the final splits, dedup, overlap removal, hand-check report |
| `src/evaluate.py` | Test-time ablations: crop, colornorm, colorwhiten, down128/64, jpegmatch, jpeg50, gray; checkpoint sanity check |
| `src/find_bw_images.py` | Finds black-and-white museum images |
| `src/met_style_counts.py` | Counts Met paintings per style |
| `src/hand_check_tool.py`, `src/hand_check_fix_cell.py` | Blind human hand check of labels |
| `src/color_baseline.py` | Color-only baseline |
| `notebooks/wikiart_colab.ipynb` | Train and evaluate ResNet-50 in Colab (only copies finished runs to Drive) |
| `notebooks/fix_labels_and_handcheck.ipynb` | Applies the label fix and B&W filter, then the hand check |
| `notebooks/hand_check_colab.ipynb` | Hand check in Colab |
| `notebooks/met_style_counts.ipynb` | Met style counts in Colab |
| `notebooks/foundation_probes.ipynb` | CLIP and DINOv2 probes, CLIP zero-shot |
| `notebooks/boost_accuracy.ipynb` | Accuracy boost: SigLIP, feature ensemble, CLIP fine-tune + WiSE-FT, TTA |
| `notebooks/improve_accuracy.ipynb` | One-hour T4 run of the accuracy levers (SigLIP, feature ensemble, CLIP fine-tune, ensemble) with 6-style, 4-group and 3-era scores and painter-level bootstrap |
| `docs/improving_accuracy.md` | Which levers can raise accuracy, the evidence, and the Nov 18 results (unseen-artist museums: ResNet 57-58 to 70-73 balanced accuracy with SigLIP / ensembles; files in `results/improve/`) |
| `notebooks/colorjitter.ipynb` | Colour-jitter test (with outputs): retrains the ResNet-50 baseline (3 seeds) with `ColorJitter(brightness=0.2, saturation=0.2)` on WikiArt training images only and compares it with the baseline (paired t-test, painter-level bootstrap, McNemar); report in `results/colorjitter/` |
| `notebooks/rq2_fewshot.ipynb` | RQ2 few-shot curve, both directions |
| `notebooks/final_extras.ipynb` | CORAL on frozen CLIP / DINOv2 / SigLIP features (CPU, minutes) and grayscale-augmentation training, 3 seeds (T4, about 1 h) |
| `src/final_analysis.py` | Final statistics from the committed predictions, no GPU: table of record, painter-level CIs, equivalence and paired tests, label source, class-balanced rescoring, confusions, merged styles, confident errors, Rijksmuseum Mannerism, figures |
| `results/final/` | Output of `src/final_analysis.py` (CSVs, `final_report.txt`, confusion-matrix and label-source figures) and the confident-error tally (`error_tally.csv`) |
| `paper/` | Final paper: LaTeX source, bibliography, figures and PDF |
| `docs/rq1_results.md` | Full RQ1, ablation and probe results |
| `docs/related_work_2024_2026.md` | Related work, 2024 to 2026 |
| `docs/labeling_rules.md` | Numbered rules for museum style labels |
| `docs/data_notes.md` | Sources, versions, style mapping, cleaning steps and thresholds |
| `docs/settings.md` | Split, seeds, model, probe and RQ2 settings |
| `docs/code_review.md`, `docs/todo.md` | Code review and the to-do list |
| `src/train.py` | ResNet-50 training and evaluation; writes config, history, metrics and predictions per run |
| `src/stats.py` | Bootstrap intervals, museum-gap test, seen-artist boost, paired comparison |
| `results/style_date_conflicts.csv` | Labels that conflicted with the painting's date |
| `results/stats_summary.csv`, `stats_per_class.csv`, `stats_report.txt` | 3-seed ResNet-50 statistics (RQ1 table) |
| `results/runs/resnet50_seed{1,2,3}/` | Config, training history, metrics, ablations and per-painting predictions |
| `results/probes/` | CLIP and DINOv2 probe metrics, report and predictions |
| `results/rq2/` | RQ2 few-shot report (all models, directions, k), test-set sizes and floors, curve |
| `results/labels/` | Museum test labels with rules, hand-check record |
| `data/final/wikiart_split.csv` | Artist-disjoint WikiArt split (train / val / test), SHA-256 starts `d5bb835c8cd9526a`, as in every run config |
| `data/final/museum_table.csv` | All 6,909 museum paintings with label, rule and the reason each excluded one was dropped |
| `data/final/museum_test.csv` | Final museum test sets (Met 937, Rijks 3,092) with seen/unseen artist group |
| `data/final/museum_wikiart_overlap.csv` | 721 museum paintings removed because they are also in WikiArt (match type, hash distance, WikiArt index) |
| `data/final/wikiart_near_duplicates.csv` | 425 WikiArt near-duplicates removed (337 duplicates, 88 with conflicting labels) |
| `data/final/overlap_review.html` | 60-pair overlap review page (needs the local images) |
| `data/final/final_styles.csv` | Style counts per dataset and why styles were dropped |
| `data/final/pipeline_log.json` | Counts and settings for every `prepare_dataset.py` step |
| `data/labels/museum_labels.csv` | Label, rule and source for every museum painting, including unlabeled ones |
| `data/labels/` (other files) | Style counts, Wikidata movements with no mapping, labeling log |
| `data/met/ids.txt`, `data/rijks/ids.txt` | Object IDs used for the harvest |
| `data/*/harvest_log.json`, `data/wikiart/download_log.json`, `data/preprocessing_log.json` | Export dates and step counts |
| `data/rijks/excluded_not_sk.csv`, `data/rijks/checks/` | Rijksmuseum exclusions and harvest checks |

### Which numbers are current

Use the numbers in [docs/rq1_results.md](docs/rq1_results.md) (Oct 27-28). Presentation 2 showed unseen-artist Met 56.3 and Rijks 57.7. Those came from the old labels, before the date-fit fix and the B&W filter (Oct 26). After retraining all three seeds on the fixed labels, the numbers are Met 57.5 and Rijks 58.2.

## Instructor checklist (repository contents)

| Requested | Status | Where |
|---|---|---|
| All code and notebooks, including RQ2 | Done | `src/` (harvest, labeling, dataset, training, evaluation, statistics), `notebooks/` (RQ2 in `notebooks/rq2_fewshot.ipynb`). See the pipeline below |
| `labeling_rules.md` | Done | [docs/labeling_rules.md](docs/labeling_rules.md); rule counts in [results/labels/README.md](results/labels/README.md) |
| Label and rule for every museum painting | Done | [data/labels/museum_labels.csv](data/labels/museum_labels.csv) (all 6,909), test sets in [results/labels/museum_test_labels.csv](results/labels/museum_test_labels.csv) (Met 937, Rijks 3,092) |
| Met and Rijksmuseum object IDs with export dates | Done | `data/met/ids.txt`, `data/rijks/ids.txt`; all harvested on 2026-10-20 (logs in `data/*/harvest_log.json`, dates in [docs/data_notes.md](docs/data_notes.md)) |
| Removed WikiArt-museum overlaps; near-duplicate list | Done | [data/final/museum_wikiart_overlap.csv](data/final/museum_wikiart_overlap.csv) (721), [data/final/wikiart_near_duplicates.csv](data/final/wikiart_near_duplicates.csv) (425); method in [docs/data_notes.md](docs/data_notes.md) section 5 |
| Artist-disjoint split, seeds, model and probe settings | Done | [data/final/wikiart_split.csv](data/final/wikiart_split.csv) (hash `d5bb835c8cd9526a`, matches `results/runs/*/config.json`), [docs/settings.md](docs/settings.md) |
| Result files behind every table, incl. `rq1_results.md` | Done for RQ1, ablations, probes and RQ2 | `results/stats_*`, `results/runs/resnet50_seed{1,2,3}/` (metrics, ablations, predictions), `results/probes/`, `results/rq2/` |
| README with library versions and steps to reproduce | Done | Below. Training versions from the run configs; full Colab `pip freeze` in [requirements_colab.txt](requirements_colab.txt) |

## Reproducing the main tables

**Versions used for the training runs** (authoritative, recorded in every `results/runs/*/config.json`): Python 3.13.16, PyTorch 2.11.0+cu130 (Colab, CUDA GPU).

**Current Colab environment** ([requirements_colab.txt](requirements_colab.txt), full `pip freeze` taken Nov 16 on a CPU runtime). This is a newer runtime than the training runs (it has torch 2.14.1+cpu), so use it for the other library versions, not for PyTorch. Key libraries in it:

| Library | Version |
|---|---|
| numpy | 2.4.6 |
| pandas | 2.2.3 |
| scikit-learn | 1.6.1 |
| scipy | 1.16.3 |
| pillow | 11.3.0 |
| timm | 1.0.31 |
| transformers | 5.22.1 |
| datasets | 4.8.5 |
| huggingface_hub | 1.33.0 |
| matplotlib | 3.10.0 |
| tqdm | 4.67.3 |
| requests | 2.32.4 |

`imagehash` (used by `src/prepare_dataset.py` on the computer) is not in the Colab freeze; install it with `pip install imagehash`. `requirements.txt` lists the libraries the scripts need.

1. **Data (computer).** The scripts look for `data/` next to themselves, so run them from one folder (for example copy `src/*.py` to the project folder). In order:
   - `python wikiart_download.py probe`, then `python wikiart_download.py download`
   - `python main.py ids`, `python main.py meta`, `python main.py images`, then `python check.py` and `python filter_rijks.py`
   - `python met_harvest.py ids`, `python met_harvest.py meta`, `python met_harvest.py images`
   - `python squash_museums.py`
   - `python label_museums.py`
   - `python prepare_dataset.py all`
   - `python pack_for_colab.py`, then upload `colab_data.zip` to Google Drive
2. **ResNet-50 table (Colab).** Run `notebooks/wikiart_colab.ipynb` with `SEED = 1`, `2` and `3`.
3. **Ablation table (Colab).** For each seed, run `python src/evaluate.py --run runs/resnet50_seed<N>`.
4. **Statistics (Colab).** Run `python src/stats.py --runs runs/resnet50_seed1 runs/resnet50_seed2 runs/resnet50_seed3`. Output matches `results/stats_report.txt`.
5. **Foundation-model table (Colab).** Run `notebooks/foundation_probes.ipynb`.
6. **RQ2 curve (Colab).** Run `notebooks/rq2_fewshot.ipynb` after step 5 (it reuses the probe features). Output matches `results/rq2/rq2_report.txt`.
7. **Final statistics (any computer, no GPU).** `pip install numpy pandas scipy matplotlib`, then `python src/final_analysis.py` from the repository root. Output matches `results/final/`.
8. **CORAL and grayscale training (Colab).** Run `notebooks/final_extras.ipynb` (needs the feature files from step 5 and `colab_data.zip`).
9. **Paper.** In `paper/`: `pdflatex main && bibtex main && pdflatex main && pdflatex main`.

## Ethics and licensing

- This repository releases **IDs, source URLs, labels and label maps, not images**. WikiArt images are for non-commercial research only and may not be redistributed. Anyone can rebuild the image sets from the IDs with the scripts.
- Met Open Access and Rijksmuseum images are public domain / CC0. Wikidata is CC0.
- No personal data is collected.
- CLIP and DINOv2 were pretrained on web-scale data that likely includes WikiArt and museum images. The ImageNet-initialized ResNet-50 does not have this problem, so it is the cleanest test of transfer.

## Why Google Drive is still used

The project started on Google Drive because the training runs in Google Colab, and Colab reads its data from Drive. The code and results moved to GitHub on Nov 16, at the instructor's request. Drive still holds the files that cannot or should not go on GitHub:

- **Images.** `colab_data.zip` (about 784 MB) holds the WikiArt and museum images at 256 px. WikiArt images are for non-commercial research only and may not be shared publicly, so they are kept out of this public repo (see `.gitignore`).
- **Trained models.** Each ResNet-50 checkpoint (`best.pt`, about 94 MB per seed) is too large for a normal Git repository.
- **Feature files.** The CLIP and DINOv2 feature files (`features_*.npz`) are large and only needed to rerun the probes.
- **Colab runs.** Every notebook (training, probes, accuracy boost, RQ2 few-shot) mounts Drive to load `colab_data.zip` and saves new runs to `MyDrive/wikiart_runs/`.

This repository keeps everything needed to check and rebuild the work: all code and notebooks, the split, labels, object IDs, overlap lists, logs and the result files behind every table. Anyone can rebuild the image sets from the IDs with the scripts in `src/`.

## What is not in this repo

- Images, datasets, caches and model checkpoints (`.pt`, `.zip`).
- `data/final/wikiart_table.csv` (16 MB, full WikiArt table) and `data/final/hashes.json` (perceptual hashes); both are rebuilt by `src/prepare_dataset.py`.

## Status

Done: RQ1 (3 seeds, painter-level statistics, equivalence and paired tests, label source, class-balanced rescoring, ECE), ablations on 3 seeds, colour-jitter training, foundation-model probes, stronger models, RQ2 in both directions, failure analysis and the Rijksmuseum Mannerism check, final paper and slides. CORAL and grayscale-augmentation training run in `notebooks/final_extras.ipynb`.

## Limitations

- Museum images are professional digitizations, not gallery photos. No in-gallery set is used (decided Nov 16), so results do not cover visitor photos; that is a threat to external validity and future work (see [docs/data_notes.md](docs/data_notes.md) section 3).
- Labels come from Wikidata and date rules.
- Romanticism and Mannerism have few museum examples.
- Only two Western museums.
- CLIP may have seen these paintings during pretraining.

## Credits
Project idea and its two-question framing: classmates Mashfika Jahan and Yisihaq Yemiu, used with their agreement.

Data: WikiArt via `Artificio/WikiArt_Full`; The Metropolitan Museum of Art Open Access (CC0); Rijksmuseum (public domain / CC0); Wikidata (CC0).
