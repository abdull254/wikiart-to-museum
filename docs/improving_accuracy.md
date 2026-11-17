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
3. **Stronger model: the biggest measured lever so far.** Switching from ResNet-50 to frozen CLIP features already adds about 10 points. Next steps, tested in [`notebooks/improve_accuracy.ipynb`](../notebooks/improve_accuracy.ipynb): SigLIP so400m, a CLIP + DINOv2 + SigLIP feature ensemble, fine-tuning CLIP ViT-L/14 (last 8 blocks, WiSE-FT), mirror-image averaging, and a final ensemble. Results pending.
4. **Broader style groups: large gain, but it changes the question.** The most common errors are between neighbouring styles (Mannerism vs Baroque / Northern Renaissance, Rococo vs Baroque, Romanticism vs Realism/Rococo). Merging the CLIP probe's six-style probabilities afterwards (`collapse` in the notebook) gives:

| Grouping (chance) | WikiArt test | Met | Rijks |
|---|---|---|---|
| 6 styles (17%) | 70.9 | 68.2 | 66.4 |
| 4 groups: Renaissance (Northern + Mannerism), Baroque + Rococo, Romanticism, Realism (25%) | 76.8 | 74.3 | 82.6 |
| 3 eras: Renaissance, Baroque/Rococo, 19th century (33%) | 84.7 | 86.2 | 87.3 |

   Balanced accuracy, CLIP probe, same predictions as `results/probes/`, merged after prediction (not retrained). The grouping follows the style names (Mannerism is "Late Renaissance", Rococo is late Baroque). Tradeoff: higher numbers against a higher chance floor, and the study then asks about periods rather than fine styles. Best reported as an extra table next to the six-style result, not instead of it.

## Honest expectation

Published WikiArt style classifiers on many styles reach roughly 60-80% accuracy. For six neighbouring styles on new painters, a realistic target with the stronger models is the 70s for balanced accuracy, about 75-90 for 4 groups / 3 eras, and above 95 for top-3. Upper-90s six-way accuracy would need leaking painters between train and test, which the artist-disjoint split is designed to prevent.
