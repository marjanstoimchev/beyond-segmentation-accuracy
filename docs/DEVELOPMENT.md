# Development

```bash
pip install -e ".[train,kodiak,notebooks,dev]"
make lint        # ruff on src and tests
make test        # pytest on a synthetic slide (no real data, no GPU), about a minute
```

The suite builds a tiny CellDIVE-like slide (pyramidal OME-TIFFs, HALO GeoJSON layers, a review table) in a temporary
folder and runs every stage on it: ingest, validation, planning, sampling, loaders, the segmenters' inputs, the
metrics, a few training steps of every model family, the KODIAK pretraining step, and the CLIs. Tests marked `realdata`
check facts of the paper's slides and are skipped when `MXDATA_ROOT` holds no slides. The same suite runs in GitHub
Actions on every push and pull request (`.github/workflows/ci.yml`, Python 3.10 and 3.12, CPU PyTorch).

### Project structure

```
beyond-segmentation-accuracy/
├── configs/
│   ├── dataset.yaml                   the dataset: slides, markers stored per patch, patch geometry, normalization, the leave-slide-out split
│   ├── panel.yaml                     the marker panel: roles, display colors, the phenotype hierarchy and the calibrated gates
│   ├── kodiak_pretrain[_vitb].yaml    KODIAK pretraining: the method's configuration + the MxIF input and augmentation
│   ├── train_{unet,kodiak,cellpose}.yaml   the trained rows under the shared protocol
│   └── baselines/*.yaml               one file per evaluated row (name, parameters, input)
├── src/
│   ├── mxdata/                        slides, annotations, QC, the dataset pipeline, sources and loaders, splits, the phenotype store   (`mxdata` CLI)
│   ├── mxseg/                         inputs, baselines, metrics, evaluation, Lightning training, downstream analysis           (`mxseg` CLI)
│   ├── kodiak/                        KODIAK (model, losses, learner, masking) + its data layer for this dataset
│   └── dinov3/                        vendored DINOv3 layers (Meta, see its LICENSE.md)
├── scripts/                           finetune_all.sh, calibrate_gates.py, build_phenotyping.py, figure_*.py, prepare_slide0233.py
├── notebooks/                         the analyses, executed and committed (docs/NOTEBOOKS.md)
├── tests/                             pytest on synthetic slides
├── results/region/test/<run>/         the metrics of every run of the paper (versioned)
├── outputs/analysis/                  the summary tables the notebooks write (versioned; predictions and figures are not)
├── docs/ · media/ · Makefile · pyproject.toml · CITATION.cff · LICENSE
```

Not versioned: `runs/` (weights and checkpoints), `weights/` (third-party pretrained weights), `outputs/region/`
(stored predictions) and `outputs/figures/`, the slides themselves.

### Practices the code follows

Stages with contracts and a pure, countable plan (`mxdata sample --dry-run`); image geometry checked on ingest, never
silently resampled; every coordinate in level-0 pixels; strict configs (unknown keys rejected) with the resolved config
saved next to every output; write-then-record resumability; QC that annotates rather than deletes; per-marker
normalization from tissue windows, stored once and used everywhere; lazily opened image handles and per-worker seeding;
errors that name themselves; a test suite on synthetic slides plus a dependency-manifest test.

### Conventions

* Region keys are `<slide>_R<id:02d>`; patch folders are `R<id>_x<x0>_y<y0>` (labeled) and `U_x<x0>_y<y0>` (unlabeled).
* A run is `<row>_s<seed>`; a zero-shot row is its bare name. `mxseg eval --name` sets the folder under `results/` and `outputs/`.
* `mask_true.png` and `cells.csv` carry the review: models learn from confirmed cells, flagged cells go to the ignore mask.
* Phenotype names are those of `configs/panel.yaml`; gates are in slide-normalized units and convert to raw counts per slide.
