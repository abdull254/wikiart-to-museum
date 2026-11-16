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
- **No overlap.** Museum paintings that also appear in WikiArt were removed (443 Met, 278 Rijks in the final Oct 26 run; list in `data/final/museum_wikiart_overlap.csv`).
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
| `notebooks/rq2_fewshot.ipynb` | RQ2 few-shot curve, both directions |
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
| Result files behind every table, incl. `rq1_results.md` | Done for RQ1, ablations and probes | `results/stats_*`, `results/runs/resnet50_seed{1,2,3}/` (metrics, ablations, predictions), `results/probes/` |
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
6. **RQ2 curve (Colab).** Run `notebooks/rq2_fewshot.ipynb`.

## Ethics and licensing

- This repository releases **IDs, source URLs, labels and label maps, not images**. WikiArt images are for non-commercial research only and may not be redistributed. Anyone can rebuild the image sets from the IDs with the scripts.
- Met Open Access and Rijksmuseum images are public domain / CC0. Wikidata is CC0.
- No personal data is collected.
- CLIP and DINOv2 were pretrained on web-scale data that likely includes WikiArt and museum images. The ImageNet-initialized ResNet-50 does not have this problem, so it is the cleanest test of transfer.

## What is not in this repo

- Images, datasets, caches and model checkpoints (`.pt`, `.zip`).
- `data/final/wikiart_table.csv` (16 MB, full WikiArt table) and `data/final/hashes.json` (perceptual hashes); both are rebuilt by `src/prepare_dataset.py`.

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
