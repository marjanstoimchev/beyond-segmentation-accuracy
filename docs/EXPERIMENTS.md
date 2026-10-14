# Experiments

Everything in the paper comes from `configs/`, `scripts/finetune_all.sh` and the notebooks. This page states the
protocol and gives every row as a command.

## The protocol: leave-slide-out

`split:` in `configs/dataset.yaml` pins it:

| slide | role | regions | cells (confirmed / flagged) |
|---|---|---|---|
| SLIDE-0235, "Slide 1" | **development**: KODIAK pretraining on its tissue, supervised training on regions 1–13 and 16, checkpoint selection on regions 14 and 15, gate calibration, every analysis setting | 16 fields of view, all expert-reviewed | 16,985 / 4,668 |
| SLIDE-0236, "Slide 2" | **test**: held out at every stage, scored once per finished run | 6 fields, all reviewed | 6,152 / 1,510 |
| SLIDE-0233, "Slide 3" | **test**: a second held-out slide, immune-rich fields | 4 fields, 3 reviewed | 2,514 / 937 |

The test slides' labeled *and* unlabeled patches carry `split = test`, so no pretraining or training can read them.
Every number for a trained method is the mean over three seeds (`<row>_s0`, `_s1`, `_s2`); the seed changes initialization
and augmentation draws, nothing else. The two test slides are reported apart, never pooled (`mxseg eval` writes a pooled
`metrics.json`; the per-slide tables are re-aggregated from `per_region.csv` in notebooks 02 and 07).

## What the networks see (`input:`)

A pretrained three-channel encoder is used as is; the `input:` block of every config assigns one marker (or blank) to each
channel, and the dataset's stored statistics (`<slide>/stats.json`, 1st–99.7th percentile per marker) scale it to [0, 1].
The resolved recipe travels with every model's weights, so `mxseg eval` renders exactly the training input.

* **Segmenters** — `red: LAP2, green: null, blue: DAPI`: the nuclear envelope and the nucleus, the two channels the
  reference boundaries were drawn from, nothing else.
* **KODIAK pretraining** — red and blue the same; green drawn once per sample from a weighted set of the other markers
  or left blank (`data.input.green` in `configs/kodiak_pretrain*.yaml`), shared by every crop of the sample.
* **Prompted and zero-shot rows** take the stored DAPI channel (InstanSeg: DAPI + LAP2), normalized their own way.

## KODIAK pretraining on multiplex tissue

```bash
python -m kodiak.pretrain -c configs/kodiak_pretrain_vitb.yaml                       # ViT-B/16, the paper's encoder -> runs/kodiak_mxif_vitb/
python -m kodiak.pretrain -c configs/kodiak_pretrain.yaml                            # ViT-S/16
python -m kodiak.pretrain -c configs/kodiak_pretrain_vitb.yaml --batch-size 8 --accelerator mps   # on a Mac
```

`src/kodiak/` is the method of [KODIAK](https://github.com/marjanstoimchev/KODIAK) (DINOv3 teacher–student, masked
prototype prediction with Sinkhorn-balanced targets, multi-crop prototype-CLS alignment, KoLeo, EMA teacher, the DINOv3
schedules), started from the DINOv3 ViT-B/16 weights, with compute-only changes (the student decoder and prototype head
run on the masked tokens only, pinned by tests) and one option for the masked pass (`model.mask_padded_keys`). The model
and losses are untouched; what is new is the data layer: `kodiak/data.py` (the stored patch stacks, the per-sample green
draw) and `kodiak/transforms.py` (`photometric: mxif` — per-channel gain and gamma, blur, flips and 90° rotations, in place
of the natural-image color jitter, grayscale and solarization, which would swap marker roles or erase the DAPI anchor).
It reads the 5,028 Slide-1 patches (550 labeled + 4,478 unlabeled), images only.

## The rows

`scripts/finetune_all.sh` trains and scores the six trained rows (three seeds each) and, with `--zero-shot`, the six rows
that need no training. The paper's budget is `--epochs 10`.

| row | config | what it is |
|---|---|---|
| `unet_flows` | `train_unet.yaml` | three-channel U-Net with the Cellpose flow objective — the supervised baseline |
| `kodiakb_flows` | `train_kodiak.yaml` + ViT-B + `runs/kodiak_mxif_vitb` | KODIAK ViT-B/16 encoder + UNETR head, fully fine-tuned |
| `dinov3b_flows` | `train_kodiak.yaml` + ViT-B + DINOv3 weights | the same model from the DINOv3 initialization: the pretraining ablation |
| `kodiakb_frozen` | … `kodiak.train_blocks=0` | encoder locked, only the head trains: the dense analog of a linear probe |
| `dinov3b_frozen` | … `kodiak.train_blocks=0` | the same probe on DINOv3 |
| `cellpose_ft` | `train_cellpose.yaml` | Cellpose's DINOv3 ViT-B/16 model (`cpdino-vitb`, 86M), fully fine-tuned under the shared protocol |
| `cellpose` | `baselines/cellpose.yaml` | Cellpose-SAM v2, zero-shot |
| `cpdino` | `baselines/cpdino.yaml` | Cellpose DINOv3 ViT-B/16, zero-shot: the untrained partner of `cellpose_ft` |
| `instanseg` | `baselines/instanseg.yaml` | InstanSeg (fluorescence, channel-invariant), zero-shot |
| `sam3` | `baselines/sam3.yaml` | SAM 3, eight exemplar boxes cached as one prompt |
| `sam3_text` | `baselines/sam3_text.yaml` | SAM 3, the text prompt alone |
| `sam2` | `baselines/sam2.yaml` | SAM 2.1, prompted by a prototype of eight training nuclei (PerSAM-style) |

The ViT-S/16 rows (`kodiak_flows`, `dinov3_flows`, `kodiak_frozen`, `dinov3_frozen`) remain available by name; they
need the ViT-S pretraining (`configs/kodiak_pretrain.yaml` → `runs/kodiak_mxif`) first.

```bash
scripts/finetune_all.sh --epochs 10 --zero-shot                 # the paper
scripts/finetune_all.sh unet_flows kodiakb_flows --seeds "0"    # a subset, one seed
scripts/finetune_all.sh --eval-only                             # re-score every run that has weights
scripts/finetune_all.sh --smoke                                 # the whole path in minutes: unet_flows, 1 epoch, 1 region
```

The commands it runs, for one row and seed:

```bash
mxseg train -c configs/train_kodiak.yaml --name kodiakb_flows_s0 --set seed=0 --epochs 10 \
            --set kodiak.vit_embed_dim=768 --set kodiak.vit_heads=12 --set kodiak.checkpoint=runs/kodiak_mxif_vitb/checkpoints/last.ckpt
mxseg eval  -b configs/baselines/kodiakb_flows.yaml -p weights=runs/kodiakb_flows_s0/weights.pt --name kodiakb_flows_s0 \
            --split test --unit region --panels
```

## One protocol for every trained model (`src/mxseg/train/`)

The same patches and seeded D4 augmentation draws (a fixed manifest, `data_manifest.json`, records the exact tiles and
ordering), the same input, the same optimizer (`optim:` — AdamW, peak lr 2e-3 at the new head, layer-wise decay 0.75 down a
pretrained encoder, weight decay 0.05 on matrix parameters, 2 warm-up epochs then cosine to 1 %), the same budget
(10 epochs in the paper; the YAMLs default to 20), effective batch 16, the flagged-cell ignore mask in the loss, and the
weights of the epoch with the minimum validation loss on regions 14 and 15 of Slide 1.

## Evaluation (`src/mxseg/evaluate.py`)

Whole annotated regions (`--unit region`): predictions from 256² tiles on the training grid, stitched by ownership (an
object is taken whole from the tile whose core contains its centroid), matched to the reference by Hungarian assignment on
instance IoU. Reported: F1 and AP at IoU 0.5, AP averaged over IoU 0.5–0.9, Panoptic Quality, precision, recall, count
ratio, plus an error taxonomy per reference cell (correct, missed, merged, split, poor boundary; gained for predictions
touching nothing). Flagged cells are never ground truth, and a prediction on one is neither a hit nor a false positive.
`--unit patch` scores the stored patches (owned cells) instead.

## Downstream levels (notebooks)

The same pipeline for the expert reference and every run: marker means under the predicted masks; hierarchical
phenotypes under fixed gates in slide-normalized units (`configs/panel.yaml`, calibrated once on the biologist's labels of
Slide 1 by `scripts/calibrate_gates.py`, never on a test slide); Delaunay cell graphs (contacts ≤ 15 µm) and neighborhood
composition; the tumor compartment from PanCK⁺ cells, CD8 infiltration zones, tumor–CD8 contacts and Treg proximity.
[NOTEBOOKS.md](NOTEBOOKS.md) has the chain; [RESULTS.md](RESULTS.md) the numbers.

## Paper figures

`scripts/figure_slides.py` (the slides and their regions), `figure_errors.py`, `figure_qualitative.py`,
`figure_spatial.py` (from the table notebook 10 writes), `figure_tsne.py` and `figure_regions_0235.py` draw the paper's
panels at print size from `outputs/` and `results/`.
