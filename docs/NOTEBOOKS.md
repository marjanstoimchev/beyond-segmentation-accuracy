# Notebooks

One notebook per analysis, executed in place and committed with their outputs, so the figures and the numbers of the paper
are traceable. Conventions: figures carry labels only (region and marker chips, method names); the reading goes into the
**Figure N** caption under each figure; tables pool the seeds (mean ± std) and figures draw seed 0; every notebook derives
its method list from the runs that exist under `outputs/region/test/`, in the paper's order; the test slide is
`TEST_SLIDE = "SLIDE-0236"` and the second held-out slide is reported apart.

Run from `notebooks/` (`cd notebooks && jupyter nbconvert --to notebook --execute --inplace <name>.ipynb`;
`analyses/10` from `notebooks/analyses`).

## The data

| notebook | what it shows | reads |
|---|---|---|
| `00_dataset` | Slide 1: tissue gate, regions by split, the labeled grid and owned cores, the tables (cells per patch, cell size, flags per region), one patch folder marker by marker, the stored normalization, what the networks see | the dataset, one region |
| `00b_dataset_slide0236` | the same for Slide 2 | |
| `00c_annotated_cells` | the annotation tool's window on Slide 1 against the HALO reference (inspection only) | `SLIDE-0235/phenotyping/original_delivery/window_R1` |
| `00d_phenotyping_data` | inventory of every phenotype label on the three slides (`mxdata.phenotyping.summary`), the labeled areas over tissue, label statistics, Slide 2's four tool windows, Slide 3's fields | `<slide>/phenotyping/` |

## The encoder

| notebook | what it shows | reads |
|---|---|---|
| `08_kodiak_pretraining_curves` | loss parts per step and epoch, prototype utilization, the schedules as applied, the codebook at each checkpoint | `runs/kodiak_mxif_vitb` |
| `01_niches_kodiak_vs_dinov3` | what KODIAK's codes, tokens and CLS encode on Slide 1's validation regions, side by side with its DINOv3 initialization (k-means codebook) | the checkpoint, DINOv3 weights |
| `01b_niches_matched` | the matched comparison on the test slide: leave-one-region-out probes (niche, lineage; full width and PCA 64) against a marker baseline, the same k-means on both encoders (K = 4, 16, 128, three seeds) | the checkpoint, DINOv3 weights; writes `outputs/analysis/cells_expert_SLIDE-0236.csv` |

## The four levels (the chain `02 → 03 → 04 → 05`)

Each writes the `outputs/analysis/` tables the next one reads.

| notebook | section | what it shows | writes |
|---|---|---|---|
| `02_segmentation_benchmark` | §5.1 | the benchmark table (Slide 2, then Slide 3 from the stored per-region results), F1 against the IoU threshold, precision–recall, the error taxonomy, qualitative crops | `benchmark_region.csv` (+`_sd`, `_runs`, `_per_region`), `benchmark_region_SLIDE-0233.csv`, `taxonomy_<run>.csv`, `taxonomy_summary.csv` |
| `03_phenotype_fidelity` | §5.2 | marker quantification under predicted boundaries, lineage and phenotype agreement, the uPAR⁺ state, agreement by error class | `cells_<run>.csv`, `cells_expert*.csv`, `phenotype_summary.csv` (+`_sd`) |
| `04_neighborhood_structure` | §5.3 | cell-graph edges recovered, neighborhood composition, Keren's mixing score, contact recall | `neighborhood_summary.csv` (+`_sd`), `neighborhood_per_region.csv` |
| `05_tissue_niches` | §5.4–5.5 | niches from neighborhood composition, their agreement and abundance per method, the propagation of error across the four levels | `niche_summary.csv` |
| `analyses/10_tumor_immune_spatial` | §5.5 | the tumor–immune endpoints: tumor–CD8 contact fidelity, CD8 infiltration zones, Treg proximity, PQ against fidelity, interface error rates, the ROI figure (its region chosen by a stated rule) | `tumor_immune_fidelity.csv`, `tumor_immune_expert_regions.csv`, the paper's spatial panels |

## Inference and training diagnostics

| notebook | what it shows | reads |
|---|---|---|
| `07_test_inference` | the inference report: the table with mean ± std over seeds per slide, IoU curves, per-region scores, error anatomy (missed and invented cells, matched IoU, AJI, the Mesmer-style taxonomy, recall by cell size and by nearest-nucleus distance, epithelium against stroma), consensus across methods, test–retest over seeds, whole-region error maps and verdict crops | `results/`, `outputs/region/test/` |
| `06_finetuning_curves` | loss and validation metrics per epoch for every row, seeds pooled and as small multiples, the kept epoch per run | `runs/<row>_s<seed>/history.csv` |
| `09_cell_phenotyping` | cell phenotyping against the biologist's tags on Slide 1's window: gating policies, what the frozen representations know before fine-tuning, every fine-tuned segmenter run on the tagged window (kept, mistyped, lost cells; marker fidelity), the gate's effect on the test slide's composition | `runs/<row>_s0/weights.pt`, the phenotype store |

## Versioned outputs

Only the summary tables of `outputs/analysis/` are versioned (benchmark, phenotype, neighborhood, niche and tumor–immune
summaries, gate calibration). Per-cell tables (`cells_<run>.csv`, `taxonomy_<run>.csv`), stored predictions
(`outputs/region/`) and rendered figures (`outputs/figures/`) are regenerated by the chain above.
