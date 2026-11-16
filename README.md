# Can an Art-Style Classifier Trained on WikiArt Recognize Museum Images?

**Author:** Abdullahi Jimale
**Course:** CIS 631, Minnesota State University, Mankato, Fall 2026
**Instructor:** Dr. Suboh Alkhushayni

The research idea was first proposed by classmates **Mashfika Jahan** and **Yisihaq Yemiu** and is used here with their agreement.

> Work was kept in Google Drive until November 2026 and moved to this repository at the instructor's request. Images, datasets and model checkpoints stay on Drive (see [What is not in this repo](#what-is-not-in-this-repo)).

## Overview

Art-style classifiers are usually trained on WikiArt. Museums may want to use them on their own collections, but images from different sources differ in digitization, color and framing. This project measures how much a WikiArt-trained model loses on museum digitizations from the Metropolitan Museum of Art (the Met) and the Rijksmuseum, and whether a few museum images help at the other museum.

The museum images are professional collection digitizations, not photographs taken in the gallery.

## Research questions

1. **RQ1, transfer.** How much does performance change when a model trained only on WikiArt is tested on the Met and the Rijksmuseum?
2. **RQ2, few-shot.** Does fine-tuning on 5, 10, 25 or 50 images per style from one museum improve results at the other museum? Tested both ways (Met → Rijks and Rijks → Met).

## Data

| Dataset | Source | Role |
|---|---|---|
| WikiArt | Hugging Face `Artificio/WikiArt_Full` (artist named for every image) | Train, validation, in-domain test |
| The Met | Met Collection API, European Paintings and Robert Lehman Collection, public domain | Test |
| Rijksmuseum | Rijksmuseum Search API, paintings (`SK-` numbers), IIIF images | Test |

- **Paintings only.** Non-paintings, black-and-white images (including 103 B&W museum images removed on Oct 26) and near-duplicates (perceptual hash) are removed.
- **Artist-disjoint split.** WikiArt train / val / test = 18,501 / 3,786 / 3,857, with no artist in more than one split.
- **No overlap.** Museum paintings that also appear in WikiArt were removed (446 Met, 282 Rijks).
- **Six shared styles:** Northern Renaissance, Mannerism, Baroque, Rococo, Romanticism, Realism.
- **Museum labels** come from Wikidata movements (P135), the painter's movement and a date rule. Since Oct 26 a movement that does not fit the painting's date is dropped. A blind human hand check found 92 of 92 decided labels correct (8 unsure).
- **Test sets:** Met 937 paintings, Rijksmuseum 3,092.

## Results so far

All numbers are **balanced accuracy (%)**. Chance is 16.7%. Museum columns use **unseen artists** (painters not in WikiArt training), the fair test for RQ1.

### RQ1: ResNet-50 baseline (3 seeds, fixed labels, Oct 27)

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

### Foundation models (frozen features + linear probe, Oct 28)

| Model | WikiArt | Met unseen | Rijks unseen |
|---|---:|---:|---:|
| ResNet-50 fine-tuned | 61.1 | 57.5 | 58.2 |
| CLIP ViT-L/14 probe | 70.9 | 68.2 | 66.4 |
| DINOv2 ViT-L/14 probe | 58.7 | 61.1 | 57.3 |
| CLIP zero-shot | 44.6 | 39.1 | 47.2 |

CLIP is best by about 10 points, but it may have seen these famous paintings during pretraining. The museum gap stays small for every model. Top-3 accuracy for the CLIP probe is about 97% on all three sets; no style reaches 97% top-1.

### RQ2: few-shot

Notebook ready ([notebooks/rq2_fewshot.ipynb](notebooks/rq2_fewshot.ipynb)), **not run yet**, so no results.

## Repository layout

```
src/          Python scripts
notebooks/    Colab notebooks
docs/         results write-up, related work, code review, to-do list
results/      small result tables (CSV)
slides/       latest presentation deck
reports/      progress reports
```

| File | What it does |
|---|---|
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
| `notebooks/rq2_fewshot.ipynb` | RQ2 few-shot curve, both directions |
| `docs/rq1_results.md` | Full RQ1, ablation and probe results |
| `docs/related_work_2024_2026.md` | Related work, 2024 to 2026 |
| `docs/settings.md` | Split, seeds, model, probe and RQ2 settings |
| `docs/code_review.md`, `docs/todo.md` | Code review and the to-do list |
| `results/style_date_conflicts.csv` | Labels that conflicted with the painting's date |

### Which numbers are current

Use the numbers in [docs/rq1_results.md](docs/rq1_results.md) (Oct 27-28). Presentation 2 showed unseen-artist Met 56.3 and Rijks 57.7. Those came from the old labels, before the date-fit fix and the B&W filter (Oct 26). After retraining all three seeds on the fixed labels, the numbers are Met 57.5 and Rijks 58.2.

## Instructor checklist (repository contents)

| Requested | Status | Where |
|---|---|---|
| All code and notebooks, including RQ2 | Partly | `src/`, `notebooks/` (RQ2 in `notebooks/rq2_fewshot.ipynb`). Missing: download and harvest scripts, `train.py`, `stats.py`, `squash_museums.py`, `pack_for_colab.py`, `check.py`, `filter_rijks.py` |
| `labeling_rules.md` | Missing | On Drive / local copy |
| Label and rule for every museum painting | Missing | `data/labels/museum_labels.csv` (on Drive) |
| Met and Rijksmuseum object IDs with export dates | Missing | `data/met/`, `data/rijks/` metadata CSVs (on Drive) |
| Removed WikiArt-museum overlaps; near-duplicate list | Missing | `data/final/` (on Drive) |
| Artist-disjoint split, seeds, model and probe settings | Settings done; split file missing | [docs/settings.md](docs/settings.md); split CSVs in `data/final/` (on Drive) |
| Result files behind every table, incl. `rq1_results.md` | Partly | `docs/rq1_results.md`. Missing: `wikiart_runs/stats_*.csv`, `stats_report.txt`, ablation CSVs, `probes/` results, per-run `metrics.json` |
| README with library versions and steps to reproduce | Steps done; exact versions missing | Below. Exact versions need `pip freeze` from the Colab runtime |

## Reproducing the main tables

Library versions: `requirements.txt` lists the libraries. The exact versions used in Colab are still to be saved: in a Colab cell, run `!pip freeze > /content/drive/MyDrive/wikiart_runs/colab_versions.txt` and add that file to `results/`.

1. **Data (computer).** Build the datasets and `colab_data.zip` with the pipeline scripts (download, harvest, `src/label_museums.py`, `src/prepare_dataset.py all`). Upload the zip to Google Drive.
2. **ResNet-50 table (Colab).** Run `notebooks/wikiart_colab.ipynb` with `SEED = 1`, `2` and `3`.
3. **Ablation table (Colab).** For each seed, run `python src/evaluate.py --run runs/resnet50_seed<N>`.
4. **Statistics (Colab).** Run `python stats.py --runs runs/resnet50_seed1 runs/resnet50_seed2 runs/resnet50_seed3` (script still to be added).
5. **Foundation-model table (Colab).** Run `notebooks/foundation_probes.ipynb`.
6. **RQ2 curve (Colab).** Run `notebooks/rq2_fewshot.ipynb`.

## Ethics and licensing

- This repository releases **IDs, source URLs, labels and label maps, not images**. WikiArt images are for non-commercial research only and may not be redistributed. Anyone can rebuild the image sets from the IDs with the scripts.
- Met Open Access and Rijksmuseum images are public domain / CC0. Wikidata is CC0.
- No personal data is collected.
- CLIP and DINOv2 were pretrained on web-scale data that likely includes WikiArt and museum images. The ImageNet-initialized ResNet-50 does not have this problem, so it is the cleanest test of transfer.

## What is not in this repo

- Images, datasets, caches and model checkpoints (`.pt`, `.zip`).
- Files still on Google Drive or the author's computer: see the "Missing" rows in the instructor checklist above.

## Next steps

1. Run the RQ2 few-shot notebook.
2. Run the accuracy-boost notebook.
3. Add CORAL domain adaptation.
4. Failure analysis, especially Rijksmuseum Mannerism (low for every model).
5. Write threats to validity, ethics and reproducibility.

## Limitations

- Museum images are professional digitizations, not gallery photos.
- Labels come from Wikidata and date rules.
- Romanticism and Mannerism have few museum examples.
- Only two Western museums.
- CLIP may have seen these paintings during pretraining.

## Credits

Research framing: Mashfika Jahan and Yisihaq Yemiu. Data: WikiArt via `Artificio/WikiArt_Full`; The Metropolitan Museum of Art Open Access (CC0); Rijksmuseum (public domain / CC0); Wikidata (CC0).
