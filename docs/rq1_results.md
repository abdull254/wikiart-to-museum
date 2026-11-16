# RQ1 results: fixed labels, 3 seeds (2026-10-27)

ResNet-50 baseline, seeds 1-3, 2000 bootstrap samples. Balanced accuracy [95% CI], macro-F1 [95% CI].

| set | n | bal acc | macro-F1 |
|---|---|---|---|
| WikiArt test (all unseen artists) | 3857 | 61.1 [59.5, 62.6] | 59.7 [58.1, 61.2] |
| Met | 937 | 61.0 [58.0, 64.2] | 59.2 [56.2, 61.9] |
| Met, seen artists | 206 | 69.9 [63.9, 75.5] | 69.4 [62.9, 74.5] |
| Met, unseen artists | 731 | 57.5 [53.8, 61.4] | 55.7 [52.2, 59.0] |
| Rijks | 3092 | 58.0 [54.9, 61.3] | 46.3 [44.5, 48.0] |
| Rijks, seen artists | 98 | 59.9 [46.3, 75.7] | 55.6 [41.5, 68.0] |
| Rijks, unseen artists | 2994 | 58.2 [55.0, 61.5] | 46.0 [44.2, 47.7] |

Per-seed bal acc (WikiArt test / Met / Rijks): s1 61.8/60.4/57.9, s2 61.8/62.1/57.9, s3 59.6/60.6/58.3.

## Main findings

1. **Museum gap is small and not significant.** WikiArt test minus museum unseen artists: Met 3.5 pts [-0.7, 7.6], p = 0.10; Rijks 2.9 pts [-0.7, 6.4], p = 0.12. Word it as "no significant drop; the data rule out a drop larger than about 7-8 points", not "no gap".
2. **Seen artists inflate accuracy at the Met.** +12.2 pts [5.1, 19.4]. The model partly recognizes painters, not only styles, which is why RQ1 must lead with unseen-artist results. Rijks has too few seen-artist paintings (98) to say (CI -12.5 to 17.6).
3. **Rijks macro-F1 (46) is far below its bal acc (58).** Rijks is 69% Baroque (2124/3092), so the errors on Baroque paintings land in the small classes and pull their precision down. Report both metrics and show the confusion matrix.
4. **Per style:** Romanticism is weakest everywhere (WikiArt 46.9, Met 38.4, Rijks 42.1). Realism drops at the Met (75.7 to 53.5). Northern Renaissance transfers best (Met 88.7). Rijks Mannerism is low (39.8, n = 72). Museum classes with fewer than about 60 paintings (Met Romanticism 53, Rijks Romanticism 38) have wide uncertainty.

## Compared with the deck (old labels)
Unseen-artist bal acc was WikiArt 61.1, Met 56.3, Rijks 57.7. Now 61.1 / 57.5 / 58.2. The label fix and B&W filter changed the conclusion very little.

Files on Drive: wikiart_runs/stats_summary.csv, stats_per_class.csv, stats_report.txt.

## Ablations (test-time changes, 3 seeds, new evaluate.py, 2026-10-27)

Balanced accuracy, mean over seeds 1-3, with change from base (mean ± SD over seeds). Museum columns are unseen-artist paintings only. Sanity check passed for all seeds. JPEG quality estimated at 90 for WikiArt, Met and Rijks.

| variant | WikiArt test | Met unseen | Rijks unseen |
|---|---|---|---|
| base | 61.1 | 57.5 | 58.2 |
| crop | n/a | 60.0 (+2.5 ± 2.5) | 58.4 (+0.2 ± 0.7) |
| colornorm (per museum) | n/a | 57.4 (-0.1 ± 0.9) | 58.5 (+0.3 ± 0.8) |
| colorwhiten (per image) | 56.5 (-4.6 ± 1.0) | 54.4 (-3.1 ± 1.1) | 55.4 (-2.8 ± 1.6) |
| down128 | 55.5 (-5.6 ± 2.8) | 51.1 (-6.4 ± 1.8) | 53.6 (-4.6 ± 2.1) |
| down64 | 36.4 (-24.7 ± 2.3) | 34.3 (-23.2 ± 4.2) | 36.6 (-21.6 ± 3.3) |
| jpegmatch (q 90) | n/a | 57.5 (0.0 ± 0.3) | 58.3 (+0.1 ± 0.2) |
| jpeg50 | 53.7 (-7.4 ± 0.7) | 53.9 (-3.7 ± 2.9) | 55.9 (-2.3 ± 1.4) |
| gray | 52.2 (-8.9 ± 2.0) | 51.3 (-6.2 ± 4.5) | 54.3 (-3.9 ± 0.6) |

Findings:
1. **No digitization fix closes the gap, and there is little gap to close.** Matching color (colornorm) and compression (jpegmatch) change nothing (within ±0.3). Compression and resolution are matched by construction (all JPEG q 90, all 256 px).
2. **Crop is the only change that helps**, at the Met: +2.5 on unseen artists, which would shrink the Met gap from 3.5 to about 1 point. It is noisy (SD 2.5, 3 seeds) and does nothing at Rijks. It suggests the squashing of non-square museum frames costs a little; it has no WikiArt control because WikiArt is only available pre-squashed.
3. **Museum images are not more fragile than WikiArt.** Every degradation (gray, jpeg50, down128, down64, colorwhiten) hurts the museums about as much as WikiArt or less. Gray and jpeg50 hurt WikiArt most.
4. **The model relies on color**: grayscale costs 4-9 points. Per-image color whitening costs 3-5, so each painting's own brightness/contrast carries style information.
5. **Resolution matters steeply below 128 px** (64 px roughly loses 22-25 points everywhere), but since all sets are 256 px this is not a source of the museum gap.

Still to do: bootstrap intervals for the ablation rows (stats.py `--variant`), paired test for crop.

## Foundation models (frozen features + logistic regression, 2026-10-28)

CLIP ViT-L/14 (openai) and DINOv2 ViT-L/14, frozen; logistic-regression probe on WikiArt train, C chosen on WikiArt val (CLIP C = 0.1, DINOv2 C = 0.003). One run each (deterministic), 2000-sample bootstrap. Notebook: notebooks/foundation_probes.ipynb; outputs on Drive wikiart_runs/probes/.

Balanced accuracy [95% CI]; museum columns are unseen-artist paintings.

| model | WikiArt test | Met unseen | Rijks unseen | Met seen | gap Met | gap Rijks |
|---|---|---|---|---|---|---|
| ResNet-50 fine-tuned (3 seeds) | 61.1 | 57.5 | 58.2 | 69.9 | 3.5 (p 0.10) | 2.9 (p 0.12) |
| CLIP probe | 70.9 [69.1, 72.8] | 68.2 [63.7, 72.8] | 66.4 [62.8, 69.8] | 80.2 | 2.7 (p 0.26) | 4.5 [0.5, 8.6] (p 0.03) |
| DINOv2 probe | 58.7 [56.8, 60.7] | 61.1 [56.4, 65.4] | 57.3 [53.6, 61.2] | 73.0 | -2.4 (p 0.35) | 1.4 (p 0.53) |
| CLIP zero-shot | 44.6 [43.1, 46.1] | 39.1 [35.5, 42.9] | 47.2 [43.9, 50.4] | 46.7 | 5.6 (p 0.01) | -2.6 (p 0.18) |

All sets (incl. seen): CLIP probe macro-F1 71.1 / 67.9 / 55.8, top-3 97 / 96 / 97, ECE 0.10 / 0.15 / 0.12. DINOv2 probe macro-F1 60.4 / 60.6 / 45.9, ECE 0.09 / 0.06 / 0.05.

Findings:
1. **CLIP is best by about 10 points everywhere**, with only a linear layer on frozen features. Caveat: CLIP's web training data likely contains these famous paintings and their style descriptions, so part of this may be prior exposure (contamination), not better style understanding.
2. **The museum gap stays small for every model** (-2.4 to +5.6 points). Only CLIP probe at Rijks (4.5, p = 0.03) and CLIP zero-shot at the Met (5.6, p = 0.01) reach significance, and there were 6 gap tests, so treat those as weak. This supports the RQ1 conclusion: little transfer loss from WikiArt to museum scans.
3. **Seen-artist boost appears in all trained models** (Met: ResNet +12, CLIP +12, DINOv2 +12), so it is not a fine-tuning artifact; any model trained on those artists' WikiArt paintings benefits.
4. **DINOv2 (self-supervised, no text) is no better than the fine-tuned ResNet** (58.7 vs 61.1 on WikiArt), but it shows no museum drop at all.
5. **CLIP zero-shot is far above chance (16.7) but far below the probes**: knowing the style names is not enough. Its plain accuracy at Rijks is only 19.6%, so it rarely predicts Baroque.
6. **Rijks Mannerism is low for every model** (ResNet 40, CLIP 35, DINOv2 36% recall) while Met Mannerism is fine (CLIP 78). That points to a Rijks label issue (for example Dutch Mannerists or date-rule labels) — check in the failure analysis.
7. CLIP probe is less calibrated (ECE 0.10 to 0.15) than ResNet (0.03 to 0.06).
