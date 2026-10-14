# Pipeline

```
ingest ──► validate ──► plan ──► stats ──► sample ──► serve
 files      refuse /     region     per-marker  tissue gate,     PatchSource -> RegionDataset -> DataLoader
 geometry   flag         units,     (lo, hi)    labeled +        (LiveSource for whole FOVs, inspection only)
 checked                 grids                  unlabeled patches
```

| stage | function | hands on |
|---|---|---|
| ingest | `pipeline.ingest(slide)` | `SlidePaths`, `MxSlide` (geometry read + checked), `IngestReport` |
| validate | `pipeline.validate(slide, regions, cfg)` | `ValidationReport` — errors stop a run; warnings/info are data |
| plan | `plan_units`, `tile_grid`, `plan_labeled`, `plan_unlabeled` | pure lists of units / windows with owned cores; `sample --dry-run` counts them |
| stats | `pipeline.compute_norm_stats` | `NormStats` per slide, seeded, saved as `stats.json` with provenance |
| sample | `pipeline.sample_patches(cfg, out_dir)` | the dataset directory (below) |
| serve | `data.build_datasets(cfg[, dataset_dir])` | `DataBundle(train, val)` → `build_loaders` |

`prepare()` chains ingest → validate → plan for one slide; every stage takes `cfg.slides` (a list).

## Configuration — two lifetimes

* **Experiment** (`configs/*.yaml`): slides, markers, level, mode, QC policy, patches, normalization, split,
  augment, loader. Typed dataclasses; unknown keys and bad values fail loudly.
* **Environment**: `MXDATA_ROOT` / `--root` (slide folders), `-o` (dataset directory), `loader.num_workers`.
* The resolved config (defaults included) is written into every slide sub-folder of the dataset.

## Sample: idempotent and resumable

* Labeled units = regions; unlabeled units = bands of `rows_per_unit` whole-slide grid rows (one read each).
  Unit keys derive from identity (`<slide>_R01`, `<slide>_U003`), patch keys from slide/region/x0/y0.
* Per unit: read once → cut every window → gate (labeled: nucleus fraction ≥ `min_fg` — the labels say what is there;
  unlabeled: tissue ≥ `min_tissue` on the coarse mask and no FOV intersection) → write kept files → write the unit's
  tables → **then** record the unit in `.done.json`. A failed unit is recorded with its reason and the run continues.
* At the end of every run the top-level tables are rebuilt from all unit tables and the **split** is assigned
  (`assign_splits`: overlap groups within a slide; `val_slides` / `val_regions` / seeded `val_frac`; `test_regions` /
  `test_slides` pin the held-out test set, which is never drawn at random), so adding
  a slide re-splits deterministically under the same config.
* One patch geometry and channel set per dataset directory (`dataset.json["geometry"]`); a different config is refused.

## Dataset directory

```
<dataset>/
  dataset.json      format, geometry, kinds, markers, slides, counts, conventions
  patches.csv       one row per window (kept and rejected): kind, unit, slide, region, group, split, box, owned core,
                    counts per QC flag, fg / tissue fraction, fov_overlap, kept, reject_reason, image, mask
  instances.csv     one row per (labeled patch, nucleus): local id -> nucleus id, flag, area, centroid, touches_border, owned
  slides.csv        one row per slide: geometry, pixel size, tissue gate, counts, lo/hi stats, source signature
  <SLIDE>/<R01_x8073_y2896 | U_x10368_y9728>/<marker>.tif             one folder per patch, one image per marker
  <SLIDE>/R01_x8073_y2896/mask.png mask_true.png cells.csv             labeled only: complete mask, confirmed mask (WRONG removed), cell table
  <SLIDE>/tables/<unit>_patches.csv, _instances.csv   stats.json  tissue.json  reports  .done.json
```

## Serve

* `PatchSource(root, split, kind, slides)` reads the tables once and the files on demand (picklable; works in
  workers). `kind="labeled"` for supervised, `None` for self-supervised.
* `RegionDataset` normalizes per slide, applies the QC policy (`with_labels=True`) or serves `image` + `meta`
  only (`with_labels=False`), caches per process, augments with a per-worker RNG.
* `make_dataloader`: `worker_init_fn` reseeds numpy / python / the dataset RNG per worker, persistent workers,
  `drop_last` + seeded shuffle for training, `meta` collated as a list.

## Tests

`pytest` builds two tiny synthetic slides in the real folder convention (two markers, three regions with an overlap,
QC on two, a tiny fragment, unannotated tissue outside the FOVs) and runs unit, contract and integration tests on
them — no real data, no GPU. `tests/test_dependency_manifest.py` checks every import against `pyproject.toml` and
that the core imports without torch. `tests/test_realdata.py` runs only when `MXDATA_ROOT/SLIDE-0235` exists.

## A note on `PYTHONPATH`

`mxdata`, `mxseg` and `kodiak` are installed editable, so they already resolve to `src/` — **do not set
`PYTHONPATH=src`**. Doing so puts the vendored `src/dinov3` (Meta code, kept verbatim, no `hub` submodule) ahead of the
pip-installed `dinov3` on the path, and Cellpose's DINOv3 backbone then fails with
`NameError: name 'dinov3_vitb16' is not defined`. Encoder outputs are unaffected either way (verified bit-identical);
only the Cellpose ViT-B variants (`cellpose_ft`, `cpdino`) break.
