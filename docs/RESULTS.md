# Results

The current numbers (run of 2026-10-13/14: ViT-B/16 encoders, 10-epoch budget, three seeds per trained row, every
region of Slide 1 expert-reviewed). The sources are `results/region/test/<run>/` and the summary tables under
`outputs/analysis/`; the notebooks named in each section draw them. Mean ± std over seeds where a row has three.

## Level 1 — cell segmentation (notebooks 02, 07)

**Slide 2** (SLIDE-0236, 6 reviewed regions, 6,152 confirmed cells)

| Method | seeds | F1@0.5 | AP@0.5 | AP@0.5:0.9 | PQ | Precision | Recall | Count ratio |
|---|---|---|---|---|---|---|---|---|
| Cellpose (DINOv3 B/16), fine-tuned | 3 | 0.901 ± 0.005 | 0.819 ± 0.008 | 0.551 ± 0.011 | 0.726 ± 0.007 | 0.923 | 0.879 | 0.952 |
| KODIAK B/16, fine-tuned | 3 | 0.877 ± 0.008 | 0.781 ± 0.012 | 0.504 ± 0.009 | 0.697 ± 0.007 | 0.929 | 0.830 | 0.894 |
| U-Net, flow objective | 3 | 0.875 ± 0.003 | 0.778 ± 0.005 | 0.480 ± 0.005 | 0.685 ± 0.004 | 0.915 | 0.838 | 0.915 |
| DINOv3 B/16, fine-tuned | 3 | 0.863 ± 0.010 | 0.759 ± 0.015 | 0.451 ± 0.017 | 0.665 ± 0.011 | 0.924 | 0.809 | 0.875 |
| DINOv3 B/16, frozen | 3 | 0.840 ± 0.002 | 0.724 ± 0.003 | 0.424 ± 0.009 | 0.644 ± 0.005 | 0.919 | 0.773 | 0.841 |
| KODIAK B/16, frozen | 3 | 0.814 ± 0.008 | 0.686 ± 0.011 | 0.392 ± 0.007 | 0.618 ± 0.006 | 0.910 | 0.736 | 0.809 |
| SAM 3, exemplar prompts | 1 | 0.842 | 0.727 | 0.403 | 0.633 | 0.857 | 0.828 | 0.966 |
| SAM 3, text prompt | 1 | 0.805 | 0.674 | 0.308 | 0.568 | 0.768 | 0.846 | 1.102 |
| Cellpose-SAM, zero-shot | 1 | 0.781 | 0.641 | 0.292 | 0.550 | 0.701 | 0.883 | 1.261 |
| Cellpose (DINOv3), zero-shot | 1 | 0.781 | 0.641 | 0.276 | 0.541 | 0.717 | 0.858 | 1.197 |
| InstanSeg, zero-shot | 1 | 0.766 | 0.620 | 0.313 | 0.556 | 0.856 | 0.693 | 0.810 |
| SAM 2.1, prototype prompts | 1 | 0.270 | 0.156 | 0.049 | 0.168 | 0.301 | 0.245 | 0.813 |

**Slide 3** (SLIDE-0233, 4 immune-rich fields, 2,329 confirmed cells)

| Method | F1@0.5 | AP@0.5:0.9 | PQ | Precision | Recall |
|---|---|---|---|---|---|
| Cellpose (DINOv3 B/16), fine-tuned | 0.797 ± 0.003 | 0.445 ± 0.008 | 0.640 ± 0.005 | 0.859 | 0.743 |
| U-Net | 0.786 ± 0.002 | 0.397 ± 0.008 | 0.612 ± 0.005 | 0.861 | 0.723 |
| KODIAK B/16, fine-tuned | 0.784 ± 0.007 | 0.422 ± 0.009 | 0.623 ± 0.007 | 0.872 | 0.712 |
| DINOv3 B/16, fine-tuned | 0.765 ± 0.010 | 0.372 ± 0.009 | 0.589 ± 0.008 | 0.863 | 0.688 |
| DINOv3 B/16, frozen | 0.754 ± 0.002 | 0.353 ± 0.009 | 0.575 ± 0.004 | 0.868 | 0.667 |
| KODIAK B/16, frozen | 0.726 ± 0.009 | 0.328 ± 0.005 | 0.551 ± 0.006 | 0.850 | 0.633 |
| SAM 3, exemplar prompts | 0.734 | 0.324 | 0.550 | 0.810 | 0.672 |
| SAM 3, text prompt | 0.719 | 0.264 | 0.509 | 0.729 | 0.709 |
| Cellpose (DINOv3), zero-shot | 0.662 | 0.230 | 0.465 | 0.604 | 0.732 |
| InstanSeg, zero-shot | 0.657 | 0.266 | 0.486 | 0.767 | 0.574 |
| Cellpose-SAM, zero-shot | 0.655 | 0.234 | 0.465 | 0.586 | 0.741 |
| SAM 2.1, prototype prompts | 0.233 | 0.045 | 0.148 | 0.291 | 0.194 |

Readings: the fine-tuned Cellpose leads on every metric and its lead widens at strict IoU, so its boundaries are tighter,
not only its detections. KODIAK beats its DINOv3 initialization by about 0.014 F1 and 0.03 PQ on both slides — the
pretraining effect — and matches the U-Net on F1 while ahead on the strict-IoU metrics. Every trained method under-counts
(count ratio 0.81–0.95) at precision above 0.91: the errors are missed cells. The HALO reference itself, with the
flagged cells counted as its errors, has precision 0.803 on Slide 2. Only 2.6 % of confirmed Slide-2 cells are found by
no method (a recall ceiling near 0.97).

**Error taxonomy, Slide 2** (per cent of confirmed cells; notebook 02): correct / split / merged / missed —
Cellpose fine-tuned 84.7 / 0.7 / 3.4 / 11.1; KODIAK B/16 79.5 / 0.5 / 4.7 / 15.1; U-Net 79.4 / 0.6 / 5.7 / 14.1;
DINOv3 B/16 76.2 / 0.3 / 5.9 / 17.3; SAM 3 exemplar 77.3 / 1.6 / 3.9 / 16.7; SAM 2.1 13.2 / 1.0 / 47.8 / 32.5.

## Level 2 — phenotype assignment (notebook 03)

Hierarchical phenotypes under the calibrated gates, on the cells each method matched (Slide 2, mean over seeds):

| Method | lineage agreement % | phenotype agreement % | uPAR⁺ state kept % |
|---|---|---|---|
| Cellpose (DINOv3 B/16), fine-tuned | 96.9 | 95.9 | 94.4 |
| KODIAK B/16, fine-tuned | 96.7 | 95.5 | 93.6 |
| U-Net | 96.3 | 95.0 | 93.8 |
| DINOv3 B/16, fine-tuned | 96.0 | 94.7 | 94.0 |
| DINOv3 B/16, frozen | 95.9 | 94.6 | 92.8 |
| KODIAK B/16, frozen | 95.8 | 94.3 | 92.7 |
| SAM 3, exemplar prompts | 95.8 | 94.5 | 89.5 |
| InstanSeg | 95.7 | 94.2 | 86.0 |
| Cellpose-SAM, zero-shot | 95.4 | 93.8 | 87.9 |
| Cellpose (DINOv3), zero-shot | 95.2 | 93.6 | 86.1 |
| SAM 3, text prompt | 94.9 | 93.4 | 84.4 |
| SAM 2.1 | 88.5 | 84.1 | 72.8 |

Phenotype assignment is robust to the segmentation differences: every trained row agrees with the expert's lineage on
96–97 % of matched cells; the differences live in the cells that are missed, not in the cells that are found.

## Level 3 — neighborhood structure (notebook 04)

Delaunay cell graphs (edges ≤ 15 µm), Slide 2:

| Method | edge recall, matched cells | edge recall, all expert edges | edge precision | contact recall | composition L1 |
|---|---|---|---|---|---|
| Cellpose (DINOv3 B/16), fine-tuned | 0.975 | 0.837 | 0.896 | 0.824 | 0.125 |
| U-Net | 0.972 | 0.808 | 0.877 | 0.807 | 0.142 |
| KODIAK B/16, fine-tuned | 0.976 | 0.780 | 0.862 | 0.752 | 0.149 |
| DINOv3 B/16, fine-tuned | 0.973 | 0.765 | 0.852 | 0.741 | 0.161 |
| DINOv3 B/16, frozen | 0.975 | 0.713 | 0.826 | 0.687 | 0.176 |
| KODIAK B/16, frozen | 0.975 | 0.669 | 0.803 | 0.627 | 0.188 |
| Cellpose-SAM, zero-shot | 0.969 | 0.933 | 0.948 | 0.919 | 0.124 |

Among matched cells the graph is recovered by everyone (edge recall 0.97); over all expert edges it tracks recall, so the
missed cells are what breaks neighborhoods. The over-segmenting zero-shot rows keep the most edges at the cost of precision.

## Level 4 — niches and tumor–immune organization (notebooks 05, 10)

Niche agreement on Slide 2 (notebook 05): 90.5 % U-Net, 89.9 % KODIAK B/16, 90.5 % DINOv3 B/16, 91.6 % Cellpose
fine-tuned, 87–88 % for the frozen rows, 91.5–91.7 % for the zero-shot Cellpose rows; immune-niche recall 89–93 %.

Tumor–CD8⁺ contacts on Slide 2 (notebook 10; the expert reference has 40 contacts and 61 CD8 cells; mean over regions and seeds):

| Method | contact F1 | PQ |
|---|---|---|
| Cellpose (DINOv3 B/16), fine-tuned | 0.62 | 0.73 |
| SAM 3, text prompt | 0.61 | 0.57 |
| U-Net | 0.58 | 0.69 |
| Cellpose (DINOv3), zero-shot | 0.58 | 0.55 |
| Cellpose-SAM, zero-shot | 0.57 | 0.56 |
| DINOv3 B/16, fine-tuned | 0.53 | 0.67 |
| InstanSeg | 0.51 | 0.56 |
| KODIAK B/16, fine-tuned | 0.50 | 0.70 |
| SAM 3, exemplar prompts | 0.49 | 0.64 |
| DINOv3 B/16, frozen | 0.47 | 0.65 |
| KODIAK B/16, frozen | 0.45 | 0.62 |
| SAM 2.1 | 0.00 | 0.17 |

Spearman correlation with contact F1 over the 120 run × region points: PQ 0.18, Dice 0.19, F1 0.22; the error rate at
tumor–immune interfaces −0.78 (near-circular, it counts the endpoint's own unmatched cells) and its fair version, errors
at the other tumor–immune interfaces, −0.39. On the one CD8-infiltrated region (R01, 18 of the slide's 40 contacts),
U-Net and KODIAK B/16 have the same PQ (0.707 vs 0.708) and keep 5 and 9 of 14 contacts (contact F1 0.40 vs 0.58).
Coarser statistics — the fraction of CD8 cells touching tumor, distant from it, or with a Treg within 20 µm — are
preserved by almost every method (ρ = 0.79–0.90 with the expert).

## Caveats of this run

* The KODIAK ViT-B/16 pretraining of record ran 66 of its 100 planned epochs (last checkpoint: epoch 65); every KODIAK
  row started from that encoder.
* Under the 10-epoch budget, U-Net, DINOv3 B/16 and the frozen DINOv3 keep the last epoch for two of three seeds, so
  their validation loss was still falling (notebook 06).
* Slide 3's field 1 is unreviewed; its three reviewed fields carry the Slide 3 numbers.
