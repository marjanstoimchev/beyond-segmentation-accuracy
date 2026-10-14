# Data model

## The slides of the benchmark

| slide | role in the protocol | fields of view | nuclei | reviewed | flagged Wrong |
|---|---|---|---|---|---|
| SLIDE-0235 ("Slide 1") | development: pretraining, training, validation (R14 + R15), gates | 16 | 21,690 | all 16 | 4,668 |
| SLIDE-0236 ("Slide 2") | test | 6 | 7,675 | all 6 | 1,510 |
| SLIDE-0233 ("Slide 3") | test | 4 immune-rich fields (one HALO export split by field) | 3,451 | 3 of 4 | 937 |

Nine markers on every slide: DAPI, LAP2, PanCK, CD45, CD3, CD4, CD8, FOXP3, uPAR (`configs/panel.yaml`).

**Access.** The slides, annotations and review tables are private clinical research data and are not distributed with the
code. Contact **Nevenka Dimitrova**, corresponding author of the paper, for a data-use agreement. The layout below is what the
pipeline expects of any CellDIVE + HALO export, and `make test` exercises every stage on a synthetic slide in that layout.

## On disk (one folder per slide under `MXDATA_ROOT`)

```
SLIDE-0235/
  SLIDE-0235_1.0.4_R000_DAPI__FINAL_F.ome.tif                    ┐ one pyramidal OME-TIFF per marker
  SLIDE-0235_9.0.4_R000_FITC_PanCK-C11-AF488_FINAL_AFR_F.ome.tif ┘ <slide>_<round>_R000_<fluor>_<marker>-<clone>-<dye>_FINAL[_AFR]_F.ome.tif
  HALO_layers_0235/CellDIVE_SLIDE-0235_Region_<k>.geojson         one layer per annotated field of view (FOV)
  QC_for_slide-0235/*Region_<k>*.csv                              the expert's review table, one per region (Flag = "Wrong" or blank)
  phenotyping/                                                    the biologist's phenotype labels, unified (see below)
  SS For  SLIDE-0235/{*.html, images/*.png}                       HALO screenshots — reference only
```

`SlidePaths.discover(slide)` resolves all of it and reports files it cannot classify.

## Images

* Every marker is a **tiled (512²), LZW, 6-level pyramidal OME-TIFF**, `uint16`, same geometry for all
  markers (20437 × 13048 px on SLIDE-0235; 0.325 µm/px → 6.6 × 4.2 mm). `MxSlide` reads the level shapes
  and derives downsample factors from them; it does not assume powers of two.
* The physical pixel size comes from the OME-XML `PhysicalSizeX`. CellDIVE writes the unit attribute as
  `PysicalSizeXUnit` (typo in the scanner software); the reader tolerates that and assumes microns.
* The **reference resolution grid** is the level-0 shape most channels share; a channel on another grid is
  excluded and recorded in the ingest report (never silently resampled). Unreadable channels are excluded
  with a warning.
* Lossless only. Fluorescence intensity is a measurement.

## Annotations

* Each GeoJSON layer is a `FeatureCollection`; every feature is a **closed `LineString`** with **integer
  level-0 pixel coordinates** (`x` = column, `y` = row). One feature = one nucleus contour. There is no
  transform between contour space and image space — verified visually (contours sit on nucleus edges).
* A region's **box** is the tight bounding box of its contours (that *is* the FOV). On SLIDE-0235 all FOVs
  are 1489 × 932 px (484 × 303 µm) except R16 (1047 × 932).
* Contours are rasterized with OpenCV in **descending area order** (a small nucleus painted on top of a
  large one is never hidden). `label = nucleus_id + 1`, where `nucleus_id` is the feature index.

## QC table

One row per nucleus with an integer bbox (`XMin/XMax/YMin/YMax`) equal to the contour bbox up to a +0/+1 px
rounding, `Flag` (`"Wrong"` or blank = correct) and a run summary in `Total Cells Flagged`. Rows are joined
to contours by nearest bbox (L∞ ≤ 1 px, one-to-one); a row that matches nothing is a validation **error**
(table and layer disagree). Contours without a row are flagged, not dropped — on SLIDE-0235 they are all
degenerate ≤ 7 px fragments, which `min_area_px` flags `TINY` anyway.

### `NucleusFlag` — QC as data

| flag | meaning |
|---|---|
| `UNKNOWN` | the region has no QC table |
| `OK` | QC row present, `Flag` blank |
| `WRONG` | QC row present, `Flag == "Wrong"` (merged / under-segmented; ~2× the area of OK nuclei) |
| `NO_QC_ROW` | region has a QC table, this contour has no row |
| `TINY` | area < `min_area_px` — an artefact, never painted into the label map |

Flags are stored per nucleus (`nuclei.csv`, `flags` array in every region file) and the **QC policy is applied
when a sample is loaded**, so changing the policy never requires re-extraction:

| `qc_policy` | labels | `ignore` |
|---|---|---|
| `keep_all` | all non-TINY nuclei | empty |
| `ignore_wrong` (default) | all non-TINY nuclei | pixels of WRONG nuclei |
| `drop_wrong` | non-TINY minus WRONG | pixels of WRONG nuclei |

## The dataset (derived; `MXDATA_ROOT/dataset`)

`mxdata sample -c configs/dataset.yaml` writes one dataset: every marker of every window, labeled and unlabeled. KODIAK
pretrains on all of it; the segmenters read only the DAPI + LAP2 files of the labeled patches (the reference boundaries were
drawn from those two — `PatchSource(markers=...)` opens nothing else); the analysis reads the other markers of the same
patches. A labeled patch folder holds `mask.png` (every HALO cell, local ids), `mask_true.png` (the same with `WRONG` ids
set to 0) and `cells.csv` (its `instances.csv` rows). Models train on the confirmed cells only — `WRONG` pixels are ignored
at load time — the complete mask is for inspection.

| file | one row per | key columns |
|---|---|---|
| `patches.csv` | grid window (kept and rejected) | `key, kind (labeled\|unlabeled), unit, slide, region_id (-1 for unlabeled), region_key, group, split, x0..y1, core_*, n_instances, n_owned, n_border, n_ok, n_wrong, n_no_qc_row, n_unknown, fg_fraction, tissue_fraction, fov_overlap, kept, reject_reason (tissue<…, fg<…, fov_overlap, subsampled), image, mask` |
| `instances.csv` | (patch, nucleus) | `patch_key, local_id, nucleus_id, label, flag, area_px, cx, cy, touches_border, owned` |
| `slides.csv` | slide | geometry, `pixel_size_um`, tissue method/threshold/fraction, `lo`/`hi` normalization stats, `source_signature`, `sampled_at` |
| `dataset.json` | dataset | format, geometry (size, strides, markers, level), patch config, slides, split rules, counts, conventions |

Images are `uint16` TIFF (zlib, `(C, H, W)` or `(H, W)`), masks `uint16` PNG with **local** instance ids
(0 = background). All coordinates are level-0 pixels; the patch box and its owned core are in `patches.csv`.
The QC policy is applied at load time from `instances.csv` flags.

`kind = labeled`: the window lies inside an annotated FOV — its mask is complete. `kind = unlabeled`: the window
lies on tissue (coarse mask fraction ≥ `min_tissue`) and intersects **no** FOV — image only, `mask`
empty, no `instances.csv` rows, `split` follows its slide (`test_slides` → test, `val_slides` → val, else train). Windows that touch a
FOV without being inside one are recorded as rejected (`fov_overlap`): a partial mask is never written.

## SLIDE-0235 facts (from `mxdata report`)

* 6 markers: DAPI, FOXP3, CD4, CD45, PanCK, uPAR.
* 16 HALO regions, 21 690 nuclei (816–1723 per FOV); QC tables for **all 16 regions** since the delivery of
  2026-10-13 (before it only 11–16 were reviewed): 21 681 QC rows, 4 668 = 21.5 % flagged Wrong, 4 contours without
  a QC row. The earlier region-13 table (302 flagged, 4 more than the current one) is kept in `QC_for_slide-0235/superseded/`.
* FOVs **6-7-8-9** (chain) and **11-12** overlap by 21–39 % and carry independently traced duplicates
  → `split_regions` keeps `{6,7,8,9,11,12}` on one side of any split.
* DAPI p99.5 varies ~2× between FOVs → normalization is per marker from a seeded sample of the FOVs
  (`normalization.mode: slide`), so tiles stay comparable; `region` mode is available for EDA.

## SLIDE-0236 facts (from `mxdata report`)

* 18 626 × 14 977 px at level 0, nine markers (same panel and file naming as SLIDE-0235); HALO layers named
  `R<k>_HALO_SLIDE-0236_Region_<k>.geojson`.
* 6 regions, none overlapping, all reviewed: 7 675 nuclei, 1 510 flagged Wrong (19.7 %).
* Dataset: 259 labeled patches + 4 103 unlabeled, all `split = test` (leave-slide-out: SLIDE-0236 is the test slide);
  7 662 owned cells (6 152 confirmed + 1 510 Wrong) — the rest lie in the one rejected window.
* SLIDE-0235 under the same protocol: train R1–R13, R16 (19 259 cells, 4 633 reviewed); val R14 + R15 (2 398 cells, all reviewed;
  both stand alone — R12 would drag its overlap chain R6–R9, R11 along).

## Phenotype labels (`<slide>/phenotyping/`)

The biologist's phenotype labels — bounding boxes drawn in HALO around cells of one type per session, and tags on the
annotation tool's own cell polygons — unified into one layout per slide by `scripts/build_phenotyping.py` and read by
`mxdata.phenotyping` (`load_areas`, `load_labels`, `load_tagged_nuclei`, `summary`):

```
<slide>/phenotyping/
  areas.csv            one row per annotated area: a HALO region, a tool window, a field; its box, frame, nuclei and flagged counts
  labels.csv           one row per human label (box or polygon): phenotype, bounds, the HALO nucleus it lands on and its region_id
  tagged_nuclei.csv    one row per HALO nucleus with at least one tag — the phenotype reference
  geojson/             <area>_<phenotype>.geojson, the labels as polygons
  original_delivery/   everything as delivered, unchanged
```

A label locates a cell; it never defines a boundary. Slide 1's labels (regions 11 and 12, one tool window in region 1)
calibrate the gates of `configs/panel.yaml` (`scripts/calibrate_gates.py`); Slide 2's four tool windows and Slide 3's two
fields are held-out references, used for scoring only.

