# Improving museum accuracy: what can move it

Written Nov 17, 2026, after instructor feedback that the current accuracy is too low. Numbers below are for museum paintings by **painters not in WikiArt train** (unseen artists).

## Where we are (measured)

| Model | Met bal acc | Rijks bal acc | Met top-1 | Rijks top-1 | Top-3 (Met / Rijks) |
|---|---|---|---|---|---|
| ResNet-50, mean of 3 seeds | 57.5 | 58.2 | | | |
| ResNet-50, seed 1 | 56.0 | 58.0 | 64.2 | 63.3 | 86.9 / 84.9 |
| CLIP ViT-L/14 linear probe | 68.2 | 66.4 | 71.5 | 73.7 | 96.3 / 96.5 |

The museum scores are close to the WikiArt test score (CLIP probe 70.9, ResNet 61-62). So the problem is mostly that **six-way style on new painters is hard for any model**, not that museum photos break the model. To raise museum accuracy we have to raise accuracy in general.

## Levers

1. **More epochs: unlikely to help.** In all three ResNet-50 runs (`results/runs/resnet50_seed*/history.json`), WikiArt val balanced accuracy is about 54-56% from epoch 4 to epoch 12 while the training loss keeps falling (0.89 to 0.65). Extra epochs fit WikiArt more, not new painters.
2. **More WikiArt data: not available for these styles.** All 26,144 WikiArt paintings in the six styles are already in train/val/test (18,501 / 3,786 / 3,857).
3. **Stronger model: the biggest measured lever so far.** Switching from ResNet-50 to frozen CLIP features already adds about 10 points. Next steps, tested in [`notebooks/improve_accuracy.ipynb`](../notebooks/improve_accuracy.ipynb): SigLIP so400m, a CLIP + DINOv2 + SigLIP feature ensemble, fine-tuning CLIP ViT-L/14 (last 8 blocks, WiSE-FT), mirror-image averaging, and a final ensemble. Results below (Nov 18 run).
4. **Broader style groups: large gain, but it changes the question.** The most common errors are between neighbouring styles (Mannerism vs Baroque / Northern Renaissance, Rococo vs Baroque, Romanticism vs Realism/Rococo). Merging the CLIP probe's six-style probabilities afterwards (`collapse` in the notebook) gives:

| Grouping (chance) | WikiArt test | Met | Rijks |
|---|---|---|---|
| 6 styles (17%) | 70.9 | 68.2 | 66.4 |
| 4 groups: Renaissance (Northern + Mannerism), Baroque + Rococo, Romanticism, Realism (25%) | 76.8 | 74.3 | 82.6 |
| 3 eras: Renaissance, Baroque/Rococo, 19th century (33%) | 84.7 | 86.2 | 87.3 |

   Balanced accuracy, CLIP probe, same predictions as `results/probes/`, merged after prediction (not retrained). The grouping follows the style names (Mannerism is "Late Renaissance", Rococo is late Baroque). Tradeoff: higher numbers against a higher chance floor, and the study then asks about periods rather than fine styles. Best reported as an extra table next to the six-style result, not instead of it.

## Honest expectation

Published WikiArt style classifiers on many styles reach roughly 60-80% accuracy. For six neighbouring styles on new painters, a realistic target with the stronger models is the 70s for balanced accuracy, about 75-90 for 4 groups / 3 eras, and above 95 for top-3. Upper-90s six-way accuracy would need leaking painters between train and test, which the artist-disjoint split is designed to prevent.

## Results of the Nov 18 run (seed 1, Colab T4, 62 min)

Files: [results/improve/](../results/improve/) (summary table, bootstrap report, full run log, executed notebook). Unseen-artist museum paintings; balanced accuracy unless noted; chance 17% for 6 styles.

| Model | Met bal | Rijks bal | Met top-1 | Rijks top-1 | Met / Rijks top-3 | Met / Rijks 3 eras |
|---|---|---|---|---|---|---|
| ResNet-50 (RQ1, mean of 3 seeds) | 57.5 | 58.2 | | | | |
| CLIP probe, class-balanced | 68.3 | 67.1 | 72.1 | 73.2 | 96.7 / 97.0 | 86.0 / 89.0 |
| SigLIP probe | 70.9 | 70.9 | 80.2 | 82.8 | 98.8 / 98.9 | 90.9 / 89.1 |
| CLIP + DINOv2 + SigLIP probe | **72.6** | 69.3 | 80.0 | 81.0 | 98.6 / 99.0 | 91.2 / 89.2 |
| CLIP ViT-L/14 fine-tuned | 68.3 | 67.7 | 68.8 | 68.6 | 95.6 / 97.1 | 84.5 / 90.5 |
| Final ensemble (best on WikiArt val) | 70.3 | **71.0** | 75.4 | 76.0 | 97.5 / 98.5 | 87.0 / 90.9 |

What it shows:

1. **The stronger frozen models raise museum accuracy by about 12-14 points over ResNet-50** (57-58 to 70-73 balanced accuracy, 80-83 top-1, top-3 about 99).
2. **Over the CLIP probe, the gains are small and not significant on the museums.** Final ensemble vs class-balanced CLIP probe, painter-level bootstrap: Met +2.0 [-2.7, +7.1] p = .31, Rijks +3.8 [-2.0, +8.1] p = .22. On WikiArt test the same gain is significant (+5.6 [+3.2, +7.7]).
3. **Fine-tuning CLIP did not help.** WikiArt val peaked after epoch 1 (68.0) and then fell while the loss kept dropping, and museum scores stayed at the CLIP-probe level. It is the same pattern as more ResNet epochs: more WikiArt fitting, no better on new painters.
4. **The WikiArt-to-museum gap grows with the stronger models.** WikiArt test is 78-81 for SigLIP and the ensembles, against 70-73 on the museums (about 7-10 points; ResNet: about 3). The better a model fits WikiArt, the more of that gain stays in WikiArt.
5. **Broader groups:** 3 eras reach 87.0 (Met) and 90.9 (Rijks) with the final ensemble. The 4-group merge is not better than the CLIP probe (Met 70.4 vs 73.4). A probe trained directly on the 4 groups reached 77.7 on WikiArt val; its museum numbers did not print in this run.
6. **Remaining errors** (final ensemble): Met Romanticism recall 35 (52% called Rococo), Met Realism 61 (30% called Romanticism), Rijks Mannerism 59 (27% called Baroque). Northern Renaissance (75-89) and Mannerism on the Met (88) are now strong.

Caveats: one seed (the probes are deterministic, the fine-tune is not); the SigLIP feature file was already on Drive and was reused, not extracted in this run; SigLIP and CLIP were pre-trained on web images that may include these famous paintings (contamination threat, as for the earlier probes).
