# Quick start

### Installation

Python 3.10+ and PyTorch 2.1+. Apple silicon (MPS), CUDA and CPU all work; the paper's runs were made on a Mac (MPS).

```bash
git clone https://github.com/marjanstoimchev/beyond-segmentation-accuracy.git
cd beyond-segmentation-accuracy
python -m venv .venv && source .venv/bin/activate          # or a conda env
pip install torch torchvision                              # the build for your platform, https://pytorch.org
pip install -e ".[train,kodiak,notebooks,dev]"             # or: make install
```

Extras (`pyproject.toml`): `torch` (datasets and loaders), `baselines` (Cellpose, InstanSeg), `train` (Lightning
fine-tuning), `kodiak` (pretraining), `sam2` / `sam3` (the prompted rows), `notebooks`, `dev`. The core `mxdata` package
installs standalone and needs no deep-learning stack to ingest, validate and extract.

Check that everything works, without any data (a synthetic slide is built in a temporary folder):

```bash
make test            # 132 tests, about a minute
make lint            # ruff
```

### The data

The pipeline reads one folder per slide under `MXDATA_ROOT` (default `~/Desktop/MedAI/Data/mxdata`):

```
$MXDATA_ROOT/SLIDE-0235/
  SLIDE-0235_<round>_R000_<fluor>_<marker>-…_FINAL[_AFR]_F.ome.tif      one pyramidal OME-TIFF per marker
  HALO_layers_0235/<prefix>SLIDE-0235_Region_<k>.geojson                  nucleus contours, one layer per annotated field of view
  QC_for_slide-0235/*Region_<k>*.csv                                      the expert's review: Flag = "Wrong" or blank
  phenotyping/                                                            the biologist's phenotype labels (optional)
```

[DATA.md](DATA.md) describes every file and the rules the pipeline applies. The slides of the paper are not distributed;
any CellDIVE + HALO export in this layout runs unchanged.

```bash
export MXDATA_ROOT=/path/to/slides
mxdata report  -c configs/dataset.yaml      # geometry, regions, review coverage, overlapping fields, validation — per slide
mxdata sample  -c configs/dataset.yaml      # = make data: the dataset of 256² patches -> $MXDATA_ROOT/dataset/ (resumable)
mxdata dataset                              # summarize what was built
```

### Pretrain, train, evaluate

```bash
# 1. KODIAK pretraining on the development slide (every marker, labeled + unlabeled tissue; labels never read)
python -m kodiak.pretrain -c configs/kodiak_pretrain_vitb.yaml                 # -> runs/kodiak_mxif_vitb/checkpoints/last.ckpt
python -m kodiak.pretrain -c configs/kodiak_pretrain_vitb.yaml --fast-dev-run  # pipeline check

# 2. every trained row x three seeds, each scored on the held-out slides right after training, then the zero-shot rows
scripts/finetune_all.sh --epochs 10 --zero-shot          # --dry-run prints the commands; --smoke runs the whole path in minutes
```

Where things land, per run `<row>_s<seed>`:

| | |
|---|---|
| `runs/<run>/` | `weights.pt` (epoch of minimum validation loss), `checkpoints/`, `history.csv`, `resolved_config.yaml`, `data_manifest.json` |
| `results/region/test/<run>/` | `metrics.json` (pooled over every held-out region), `per_region.csv` (one row per region: the per-slide tables are re-aggregated from it), `run.json` |
| `outputs/region/test/<run>/` | one instance map per held-out region (`<SLIDE>_R<k>.npz`), what the notebooks read |
| `outputs/figures/region/test/<region>/` | PNG panels: image, reference, review, one verdict map per method |
| `runs/finetune_all.log` | everything, with timestamps |

One run by hand, and the comparison table:

```bash
mxseg train -c configs/train_unet.yaml --name unet_flows_s0 --set seed=0 --epochs 10
mxseg eval  -b configs/baselines/unet_flows.yaml -p weights=runs/unet_flows_s0/weights.pt --name unet_flows_s0 --split test --unit region --panels
mxseg eval  -b configs/baselines/instanseg.yaml --split test --unit region            # a zero-shot row needs no training
mxseg compare --split test --unit region --seeds                                       # mean ± std over the seeds, one row per method
```

### Analyze

```bash
python scripts/calibrate_gates.py                 # phenotype gates (configs/panel.yaml) from the biologist's labels on Slide 1
cd notebooks
jupyter nbconvert --to notebook --execute --inplace 02_segmentation_benchmark.ipynb    # then 03, 04, 05 in that order
jupyter nbconvert --to notebook --execute --inplace 07_test_inference.ipynb
cd analyses && jupyter nbconvert --to notebook --execute --inplace 10_tumor_immune_spatial.ipynb
```

[NOTEBOOKS.md](NOTEBOOKS.md) lists every notebook, what it reads and what it writes under `outputs/analysis/`.
