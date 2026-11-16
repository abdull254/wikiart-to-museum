# Museum labels

## `museum_test_labels.csv`

This file lists every museum test painting with its style label and the rule that produced it. It has one row per painting: Met 937, Rijksmuseum 3,092.

| Column | Meaning |
|---|---|
| `museum` | `met` or `rijks` |
| `object_id` | Met object ID or Rijksmuseum object number |
| `style` | The style label used for scoring |
| `rule_id` | The rule that assigned the label (see below) |
| `artist_group` | `seen` if the painter is in WikiArt train, otherwise `unseen` |

The rows are taken from the `id`, `true`, `rule_id` and `artist_group` columns of `results/runs/resnet50_seed*/preds_{met,rijks}.csv`. All three seeds contain the same rows.

## Rules (from `src/label_museums.py`)

| Rule | Source of the label | Met | Rijks |
|---|---|---:|---:|
| R1 | The artwork's own movement in Wikidata | 61 | 0 |
| R2a | The painter's movement in Wikidata | 431 | 675 |
| R2b | The painter's movement, with the painter found by name match | 1 | 302 |
| R2c | Attributed or workshop painting: the master's movement | 40 | 7 |
| R7 | Date rule, used only where one style dominates a period | 404 | 2,108 |

Two rules check the labels above:

- **Rule 3:** copies "after" an artist do not inherit that artist's movement.
- **Rule 6:** a movement more than 30 years outside its style's period is dropped. This is the date-fit fix from Oct 26.

About half of all labels come from the date rule (R7). At the Rijksmuseum it is about two thirds.

## `hand_check.csv`

This is the record of the label review: 100 paintings, 50 from each museum.

The review had two stages:

1. **AI draft.** An AI assistant drafted a verdict for each painting. It is kept in the `ai_correct` and `ai_suggestion` columns.
2. **Human check.** The author then checked every painting blind to that draft, using `notebooks/hand_check_colab.ipynb`. The author's answers are in `correct` and `correct_style`.

The reported result (92 of 92 decided labels correct, 8 unsure) uses the author's answers only.
