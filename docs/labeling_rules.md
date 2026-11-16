# Style labels for museum paintings: method

Applies to the Met (2,214 paintings) and the Rijksmuseum (paintings collection, `SK-` object numbers only).
Script: `label_museums.py`. Every painting's row in `data/labels/museum_labels.csv` records the rule that labeled it (`rule_id`).

Style names follow the WikiArt style names (for example `Mannerism (Late Renaissance)`), so museum and WikiArt labels can be compared directly.

## Rules

Rules are applied in order. The first rule that gives a label is used.

**R1. The artwork's own movement.**
If the painting has its own Wikidata entry (Met only) and that entry lists a movement (Wikidata property P135), use it.

**R2. The painter's movement.**
If R1 gives nothing, look up the painter on Wikidata and use the painter's movement (P135).
- **R2a.** The painter is linked to Wikidata by the museum.
- **R2b.** The painter has no link, but the same name appears elsewhere in either museum's data with a Wikidata link. The name is taken from the artist field or from the museum's text (for example "painter: George Hendrik Breitner (mentioned on object)").
- **R2c.** The painting is "attributed to", "workshop of", "studio of", "circle of", "follower of", "pupil of" or "school of" a master. These were made in the master's time and setting, so they take the master's movement.

**R3. Copies do not inherit the artist's movement.**
If the attribution says "after", "copy after", "manner of", "style of", "imitator" or "pastiche", R2 is skipped. Copies can be made long after the original (for example a 1747 copy after Rubens, who died in 1640). These go to the date rule (R7).

**R4. From Wikidata movement to project style.**
Each Wikidata movement is mapped to a project style by keywords in its English name, checked in a fixed order (the list is `MOVEMENT_RULES` in the script). For example: "Dutch Golden Age painting" and "Caravaggism" become Baroque; "Hague School" and "Barbizon school" become Realism; "Early Netherlandish painting" becomes Northern Renaissance. If a movement's own name does not match, the names of its parent movements are tried. Movements that still do not match are listed in `unmapped_movements.csv` for review and are not used.

**R4b. Regional schools are decided by date.**
Some Wikidata "movements" name a place rather than a style, for example "Sienese school", "Venetian school", "Florentine School", "Classicism" or "fijnschilder" (the Leiden fine painters). For these, the painting's date decides, using the date rule (R7). Italian schools are always treated as Italian for the Renaissance split.

**R5. A general "Renaissance" label is split by date and region.**
- Italian painters: before 1495 Early Renaissance, 1495–1526 High Renaissance, from 1527 Mannerism (Late Renaissance).
- Northern painters (Netherlands, Flanders, Germany, France, Britain and nearby): before 1580 Northern Renaissance, from 1580 Mannerism (Late Renaissance).
- Region comes from the Met's nationality field or the painter's Wikidata country of citizenship. With no region, no label is given.

**R6. When a painter has several movements, the painting's date decides.**
Each style has an approximate active period (the table `PERIODS` in the script). The movement whose period contains the painting's date is used. If several still fit, the one whose period midpoint is closest to the date is used, and the row is flagged "several movements". The painting's date is the middle of its begin and end years.

**R7. Date rule, when no movement is found.**
Used only for periods where one style clearly dominates:
- 1400–1599: Renaissance, split by region as in R5 (no region means no label)
- 1600–1719: Baroque
- 1720–1779: Rococo

**R8. No label.**
From 1780 onward, several styles overlap (Neoclassicism, Romanticism, Academicism, Realism, Impressionism), so the date alone cannot decide. Paintings from these years with no movement from R1–R2 are left unlabeled and excluded. Paintings with no date and no movement are also excluded.

## Checks

- **Out-of-period flag.** If a painting's date is more than 30 years outside its style's period, the row is flagged for review. This catches a painter's movement that does not fit a particular work.
- **Hand check.** A random sample of labels from each museum is checked by hand against the museum's own description and standard art-historical sources. The agreement rate is reported.
- **Reporting.** `style_counts.csv` gives paintings per style per museum. `labeling_log.json` records when labels were made and how many each rule produced.

## Limitations

- Wikidata movements are crowd-edited and uneven. Well-known painters have detailed movements; minor painters often have none and fall to the date rule or are excluded.
- A painter's movement describes their career, not every single work. R6 and the out-of-period flag reduce, but do not remove, this error.
- The date rule assumes one dominant style per period. It is used only where that is a fair simplification (1400–1779).
- The excluded paintings are not a random sample: they lean toward anonymous and 19th-century works. The count per museum is reported.
