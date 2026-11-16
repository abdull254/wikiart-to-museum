# Related work, 2024–2026 (search done 2026-10-28)

Checked by reading each paper's page; venue/year given only where the paper or arXiv ID shows it. Verify full bibliographic details before citing.

## Closest to this project

1. **Ashton, R. "Style or Signature? Artist-Disjoint Evaluation of Style Classification in Frozen Vision Embeddings." arXiv 2608.14435 (2026).** https://arxiv.org/abs/2608.14435
   - Same WikiArt release (Artificio/WikiArt_Full), frozen CLIP ViT-B/32 and ViT-L/14, DINOv2, ResNet-50, 5-NN. Accuracy falls from 86.9% (random split) to 76.6% (leave-one-artist-out). Artist identity is strongly encoded.
   - Small: 320 images, 4 modern movements. No museum images, no domain shift.
   - **How we use it:** independent support for our artist-disjoint design and our +12 seen-artist boost. Our gap claim must be stated carefully: they already show the artist effect on WikiArt. What we add is museum transfer (Met, Rijks), historical styles 1400–1850, a much larger test set, ablations and both fine-tuned and frozen models.

2. **Kipke, Hansen, Eriksen, Baunvig, Nielbo. "A Benchmark Dilemma: Between Representativeness and Domain-Specificity in Visual Arts Domains." Journal of Open Humanities Data 12 (2026).** https://openhumanitiesdata.metajnl.com/articles/10.5334/johd.623
   - Benchmarks CLIP, SigLIP, DINOv2 on WikiArt style (best CLIP-bigG ~0.77 F1; DINOv2-base ~0.63) and compares WikiArt vs Wikidata images for the same artists (small differences). Grayscale and lighting changes barely hurt; sketch transforms hurt a lot.
   - **How we use it:** matches our ranking (CLIP > DINOv2) and our finding that source changes matter little. They criticize WikiArt labels as opinionated, which supports our label-checking work.

3. **Strafforello, Soydaner, Willems, Maerten, De Winter. "Have Large Vision-Language Models Mastered Art History?" arXiv 2409.03521 (2024).** https://arxiv.org/abs/2409.03521
   - Zero-shot style, artist and period prediction with CLIP, LLaVA, OpenFlamingo and GPT-4o on WikiArt, JenAesthetics and ArTest. CLIP zero-shot style on WikiArt is 24.6%, GPT-4o 51.6%. They suggest famous works being online may help models (contamination).
   - **How we use it:** context for our CLIP zero-shot result (39–47% balanced accuracy on 6 classes). It also supports the contamination caveat.

4. **Asperti, Dessì, Tonetti, Wu. "Does CLIP perceive art the same way we do?" arXiv 2505.05229 (2025).** https://arxiv.org/abs/2505.05229
   - CLIP style recognition on National Gallery of Art paintings is "moderate" (recall@1 0.46). CLIP favors content over brushwork and colour. Romanticism is among the most confused styles.
   - **How we use it:** a museum-image CLIP result, and it agrees that Romanticism is the hardest style (ours: 38–62% recall).

## Museum photos and domain shift (for the "museum-photo domain" item)

5. **Bartkowiak et al. "SynGallery: A Synthetic Gallery of Real Paintings for Instance-Level Artwork Recognition." arXiv 2607.18907 (2026).** https://arxiv.org/abs/2607.18907
   - Renders catalog images of paintings in synthetic 3D galleries (angles, lighting, frames, glass) to match visitor photos. Uses the Met benchmark's real visitor photos. Viewpoint variation drives most of the gain.
   - **How we use it:** names the gap we do not test (catalog scans vs gallery photos). It is a possible way to build a gallery-style test set.

6. Older but needed as anchors: **Ypsilantis et al. "The Met Dataset: Instance-level Recognition for Artworks" (NeurIPS 2021 Datasets track)**, with Met visitor photos as queries; and **Koniusz et al. "Museum Exhibit Identification Challenge for Domain Adaptation and Beyond" (arXiv 1802.01093, 2018)**, on studio-to-gallery domain shift.

## Recent supervised style classifiers (background only)

7. **Menis-Mastromichalakis, Sofou, Stamou. "Deep Ensemble Art Style Recognition." arXiv 2405.11675 (2024).** Stacking ensemble, 68.6% on 21 WikiArt classes, random 80/10/10 split with no artist control. It is an example of the leakage our artist-disjoint split avoids.
8. Other 2025 style classifiers (ArtFusionNet, Scientific Reports 2025; self-supervised dual-teacher distillation, Applied Soft Computing 2025) report WikiArt accuracy without museum testing or artist-disjoint splits. Mention as a group, if at all.

## The gap claim (one sentence for the report)
Recent work tests style classifiers on WikiArt only (often with artists shared across splits), or tests frozen models artist-disjointly on small WikiArt subsets. We found no work that trains on WikiArt and tests style on museum digitizations from two separate collections with artist-disjoint, overlap-removed test sets. Search used: arXiv and web search for "art style classification" + WikiArt / museum / domain shift / CLIP / DINOv2 / artist-disjoint, 2024–2026.

Still to add from the to-do list: Cetinić 2018, Zhao 2021 and 2022, Liang 2025, Castellano and Vessio 2021. Credit Mashfika Jahan and Yisihaq Yemiu for the original idea.
