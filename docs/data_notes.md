# Data notes: sources, versions and decisions

Fill in the dates in [brackets] from the log files (`harvest_log.json`, `download_log.json`, `pipeline_log.json`).

## 1. Sources and versions

| Dataset | Source | Version / export date | Licence |
|---|---|---|---|
| WikiArt | Hugging Face `Artificio/WikiArt_Full`, ~103,250 images, 256 x 256 | Downloaded [date] | Non-commercial research use only; images not redistributed |
| Met | Met Collection API (`collectionapi.metmuseum.org`), departments European Paintings and The Robert Lehman Collection | Harvested [date] | Open Access, public-domain images (CC0) |
| Rijksmuseum | Rijksmuseum Search API (`data.rijksmuseum.nl`, `type=painting&imageAvailable=true`), Linked Art `la-framed` records, IIIF images | Harvested [date] | Public domain / CC0 |

**Why this WikiArt release.** The project needs an artist name for every image, for an artist-disjoint split. `huggan/wikiart`, the common 27-style release, was tested first: in a random sample of 3,000 images, 58% were "Unknown Artist", because its artist field covers only 129 artists. `Artificio/WikiArt_Full` named an artist for 100% of a 3,000-image sample (1,033 different artists). Its trade-off is that every image is 256 x 256 with the shape squashed, so the museum images are processed the same way (see section 4).

## 2. How the project styles map to the standard 27-style WikiArt set

Most published WikiArt results use the 27-style set (as in `huggan/wikiart`). Our style names follow `Artificio/WikiArt_Full`, which uses finer style tags. The mapping:

| Project style | Standard 27-style name | Notes |
|---|---|---|
| Early Renaissance | Early_Renaissance | same |
| High Renaissance | High_Renaissance | same |
| Northern Renaissance | Northern_Renaissance | same |
| Mannerism (Late Renaissance) | Mannerism_Late_Renaissance | same |
| Baroque | Baroque | our Baroque also includes WikiArt "Tenebrism" |
| Rococo | Rococo | same |
| Romanticism | Romanticism | ours also includes "Biedermeier" |
| Realism | Realism | same |
| Impressionism | Impressionism | same |
| Post-Impressionism | Post_Impressionism | ours also includes "Divisionism", "Cloisonnism", "Synthetism" and "Pointillism" (Pointillism is a separate class in the 27-style set) |
| Symbolism | Symbolism | same |
| Neoclassicism | (none) | not in the 27-style set |
| Academicism | (none) | not in the 27-style set |
| Proto Renaissance, International Gothic | (none) | not in the 27-style set |

Only the styles that pass the minimum count in all three datasets are used (`data/final/final_styles.csv`). For a comparison with published numbers, the same models can be scored on the 27-style release, restricted to the styles above that exist there.

## 3. Which "museum photo" domain this study tests

Both museum test sets are **professional digitizations**, not photographs taken in the gallery. The Met and the Rijksmuseum publish images made by their own photography studios: the painting is lit evenly, photographed straight on, color-calibrated, and usually cropped to the painted surface without the frame. Visitor-style conditions such as glare, visible frames, angled views, uneven lighting and phone-camera processing are therefore mostly absent. Any accuracy drop measured on these sets reflects differences between digitization pipelines (color handling, resolution, compression, cropping, collection mix), not the effect of photographing a painting in the room. Results are reported as transfer to museum digitizations, and claims about in-gallery photos are not made from these sets.

**Decision on an in-gallery set:** [choose one]
- **Add a small set** (for example 100–200 photos of paintings in the project's styles, taken at a local museum where photography is permitted, or taken from a dataset licensed for research). This allows a separate "photographed in the room" result. Check the museum's photography policy before using the photos for research.
- **Leave it out** and state the limitation in threats to validity.

## 4. Image processing

- WikiArt images are distributed as 256 x 256 with the shape squashed. Museum images were downloaded at 512 px on the long side and then resized the same way to 256 x 256 (`images_256_squash`, LANCZOS filter, JPEG quality 90), so image size and shape do not differ between the datasets.
- Two extra museum versions are kept for the ablations: `images_256_crop` (shape kept, center square) for crop versus full frame, and the original 512 px images for resolution.

## 5. Cleaning steps and counts

All counts are written to `data/final/pipeline_log.json` by `prepare_dataset.py`.

1. **Paintings only.** Met: objects classified as Paintings and public domain. Rijksmuseum: object numbers starting with `SK-` (the paintings collection). WikiArt: genres such as sketch and study, illustration, design and poster removed.
2. **Black-and-white images removed from WikiArt** (colorfulness below 10, Hasler & Süsstrunk 2003), so the model cannot use "black-and-white" as a style cue.
3. **Near-duplicates inside WikiArt removed** (perceptual hash, Hamming distance 4 or less). When duplicates carried different style labels, all copies were removed.
4. **Museum paintings that are also in WikiArt removed** from the museum sets, when any of these hold: perceptual hash distance 8 or less; same artist and hash distance 14 or less; same artist, title word overlap (Jaccard) 0.6 or more on a title of at least two meaningful words, and hash distance 22 or less. The last condition stops generic titles such as "Madonna and Child" from matching different paintings by the same artist. A random sample of removed pairs was checked by eye (`overlap_review.html`): [N] of [N] were the same painting.
5. **Minimum per style:** [N] images per style in WikiArt and [N] per style in each museum. Styles below it in any dataset are dropped.
6. **Artist-disjoint WikiArt split:** `StratifiedGroupKFold` (20 folds, seed 42) grouped by artist: 3 folds test, 3 folds validation, 14 folds train (70/15/15). No artist appears in more than one split.
7. **Seen versus unseen artists:** a museum painting is "seen" if its artist appears in the WikiArt training split.

## 6. What is shared

Image IDs, source URLs, label maps, splits and preprocessing settings are shared with the code. WikiArt images are not redistributed.
