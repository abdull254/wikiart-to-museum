# Splits, seeds and model settings

Everything below is fixed before museum data is scored. Museum sets are never used for tuning.

## Data split

| Item | Setting | Where |
|---|---|---|
| WikiArt split | Artist-disjoint, `StratifiedGroupKFold` (20 folds, grouped by artist, stratified by style) | `src/prepare_dataset.py` |
| Split seed | 42 | `src/prepare_dataset.py` (`SEED = 42`) |
| Split sizes | train 18,501 / val 3,786 / test 3,857 | `data/final/` (on Drive, see README) |
| Styles | Northern Renaissance, Mannerism (Late Renaissance), Baroque, Rococo, Romanticism, Realism | |
| Style minimums | at least 200 images in WikiArt and 30 in each museum | |
| Image size | 256 × 256 (WikiArt is only available at 256 px; museum images resized to match) | |
| Museum test sets | Met 937, Rijksmuseum 3,092 (after date-fit label fix and B&W filter, Oct 26) | |

## ResNet-50 baseline

| Item | Setting |
|---|---|
| Init | ImageNet weights, all layers fine-tuned |
| Epochs | 12 |
| Optimizer | AdamW, learning rate 1e-4 |
| Loss | Cross-entropy, label smoothing 0.1, inverse-frequency class weights |
| Model selection | Best epoch by WikiArt validation balanced accuracy |
| Training seeds | 1, 2, 3 |
| Notebook | `notebooks/wikiart_colab.ipynb` |

## Foundation-model probes (`notebooks/foundation_probes.ipynb`)

| Item | Setting |
|---|---|
| CLIP | ViT-L/14 (OpenAI weights), frozen image features |
| DINOv2 | ViT-L/14 (`facebook/dinov2-large`), frozen, CLS + mean patch token |
| Probe | Logistic regression on WikiArt train features |
| C (chosen on WikiArt val) | CLIP 0.1, DINOv2 0.003 |
| CLIP zero-shot | Average of 3 prompts per style: "a painting in the style of {}.", "a {} painting.", "an artwork from the {} period." |
| Runs | One each (deterministic) |

## Accuracy boost (`notebooks/boost_accuracy.ipynb`)

SigLIP (`google/siglip-so400m-patch14-384`) probe, feature ensemble, partial CLIP fine-tune (8 blocks, 6 epochs, batch 32) with WiSE-FT weight averaging, test-time augmentation. Not yet run.

## RQ2 few-shot (`notebooks/rq2_fewshot.ipynb`)

- Shots per style: k = 0, 5, 10, 25, 50, nested (the 5 are inside the 10, and so on), the same picks for every model.
- Directions: Met → Rijksmuseum and Rijksmuseum → Met.
- Test set per direction: target-museum painters who are not in WikiArt train and not anywhere in the source museum.
- ResNet-50: starts from the seed 1-3 checkpoints, 200 fixed steps, each batch half museum shots and half WikiArt.
- Control `k0ft`: the same 200 steps on WikiArt only.
- Forgetting check: rescore WikiArt test at k0ft and k = 50.
- Probes: refit the CLIP and DINOv2 logistic regression on WikiArt train plus the shots (C = 0.1 and 0.003; shots and WikiArt get equal total weight; warm start from the WikiArt-only fit).
- ResNet-50 settings: AdamW, lr 2e-5, cosine schedule, batch 32 (16 shots + 16 class-balanced WikiArt), BatchNorm statistics frozen, label smoothing 0.1.
- Run on Nov 17, 2026 (T4 GPU, about 50 min). Results: `results/rq2/`.

## Statistics

- 2,000 bootstrap samples, 95% percentile intervals.
- Final analyses (`src/final_analysis.py`, Nov 19): painter-level (cluster) bootstrap, 2,000 resamples, seed 0; ResNet-50 seeds averaged inside each resample.
- Museum gap: WikiArt test minus museum unseen artists; TOST equivalence with a +-5-point margin (90% CI inside the margin).
- Model comparisons: painter-level paired bootstrap, McNemar's exact test (ResNet-50 from the mean of the 3 seeds' probabilities), Cohen's h and discordant odds ratio.
- Class-balanced rescoring: museum paintings re-weighted to WikiArt test's style proportions.
- ECE: 15 equal-width confidence bins.

## CORAL (`notebooks/final_extras.ipynb`)

- Frozen CLIP, DINOv2 and SigLIP features, L2-normalized and standardized on WikiArt train.
- WikiArt train features whitened with their covariance (+ identity) and re-colored with the target museum's covariance (+ identity), re-centered on the museum mean; target = all that museum's test images, no labels.
- Logistic regression with the same C as without adaptation (chosen on WikiArt val).

## Grayscale-augmentation training (`notebooks/final_extras.ipynb`)

- `src/train.py --grayscale-aug 0.2`, otherwise identical to the baseline; seeds 1-3; scored with `src/evaluate.py --variants base gray`.
