"""Patches: the plan (pure), the per-unit samplers, and the patch files.

Two kinds of patch come out of a slide:

* ``labeled``   — a grid *with overlap* inside each annotated FOV (overlap > a nucleus, so every border
                  nucleus is whole in some patch; each patch owns its stride core), kept when it holds enough
                  nucleus pixels (``min_fg``: the labels say what is there) → image + instance mask;
* ``unlabeled`` — a non-overlapping grid over the whole slide, kept when the coarse tissue mask says it is tissue
                  (``min_tissue``) and it is **disjoint from every FOV** (a window touching a FOV would carry a
                  partial mask) → image only.

Every patch is one folder named by what it is and where it sits — ``<slide>/R01_x8073_y2896/`` (region window) or
``<slide>/U_x10368_y9728/`` (whole-slide window) — holding one lossless ``uint16`` zlib TIFF per marker (``DAPI.tif``,
``LAP2.tif``, ...) and, for labeled patches, ``mask.png`` (``uint16`` instance ids local to the patch). A patch can be
browsed by eye; readers assemble the ``(C, H, W)`` stack in the dataset's marker order. A labeled folder also holds
``mask_true.png`` — the same mask with the cells the expert flagged *wrong* removed (what training and scoring use; a blank
QC row is a confirmed cell) — and ``cells.csv``, one row per cell of the patch with its HALO id, flag, ownership and
centroid, so the complete (inspection) and the confirmed (training) views never lose track of each other. Keys derive
from identity (slide, region / unlabeled, x0, y0), so re-runs are idempotent.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence, Tuple

import cv2
import numpy as np
import tifffile

from ..annotations.raster import NucleusFlag, apply_qc_policy
from ..annotations.regions import RegionAnnotation
from ..config import DataConfig
from ..errors import DatasetError
from ..io.geometry import Box
from ..io.slide import MxSlide
from .plan import RegionUnit, Tile, tile_grid
from .tissue import TissueGate

KINDS = ("labeled", "unlabeled")
PATCH_COLUMNS = ["key", "kind", "unit", "slide", "region_id", "region_key", "group", "split", "row", "col",
                 "x0", "y0", "x1", "y1", "core_x0", "core_y0", "core_x1", "core_y1", "height", "width",
                 "level", "downsample", "markers", "pixel_size_um", "has_qc",
                 "n_instances", "n_owned", "n_border", "n_ok", "n_wrong", "n_no_qc_row", "n_unknown",
                 "fg_fraction", "tissue_fraction", "fov_overlap", "kept", "reject_reason", "image", "mask"]
INSTANCE_COLUMNS = ["patch_key", "local_id", "slide", "region_id", "nucleus_id", "label", "flag",
                    "area_px", "cx", "cy", "touches_border", "owned"]


# ---------------------------------------------------------------------------
# plan
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class PatchSpec:
    key: str
    slide: str
    region_id: int          # -1 for unlabeled
    tile: Tile
    tissue: float = 1.0     # tissue-mask fraction (unlabeled: decided at plan time)
    fov_overlap: float = 0.0
    reject: str = ""        # unlabeled: "" = keep, else the reason


def plan_labeled(units: Sequence[RegionUnit], slide_name: str, size: int, stride: int, downsample: float = 1.0) -> List[PatchSpec]:
    """Every grid window of every region unit."""
    return [PatchSpec(f"{slide_name}_R{u.region_id:02d}_x{t.box.x0}_y{t.box.y0}", slide_name, u.region_id, t)
            for u in units for t in tile_grid(u.box, size, stride, downsample)]


@dataclass(frozen=True)
class UnlabeledUnit:
    """One resumable band of the whole-slide grid (read once), its windows already gated."""

    key: str                    # "<slide>_U<band:03d>"
    band: Box
    windows: Tuple[PatchSpec, ...]


def plan_unlabeled(slide: MxSlide, fov_boxes: Sequence[Box], gate: TissueGate, cfg: DataConfig) -> List[UnlabeledUnit]:
    """Whole-slide grid, gated on the coarse tissue mask, disjoint from every FOV, optionally capped (seeded)."""
    pc = cfg.patches
    tiles = tile_grid(slide.full_box, pc.size, pc.effective_unlabeled_stride, slide.downsamples[cfg.level])
    specs: List[PatchSpec] = []
    for t in tiles:
        ov = max((i.area / t.box.area for f in fov_boxes if (i := t.box.intersect(f))), default=0.0)
        tf = gate.fraction(t.box)
        reject = "fov_overlap" if ov > 0 else (f"tissue<{pc.min_tissue}" if tf < pc.min_tissue else "")
        specs.append(PatchSpec(f"{slide.name}_U_x{t.box.x0}_y{t.box.y0}", slide.name, -1, t, tf, ov, reject))
    keep = [i for i, s in enumerate(specs) if not s.reject]
    if pc.max_unlabeled_per_slide is not None and len(keep) > pc.max_unlabeled_per_slide:
        chosen = set(np.random.default_rng(pc.seed).choice(keep, size=pc.max_unlabeled_per_slide, replace=False).tolist())
        specs = [s if (s.reject or i in chosen) else PatchSpec(s.key, s.slide, -1, s.tile, s.tissue, s.fov_overlap, "subsampled")
                 for i, s in enumerate(specs)]
    units = []
    for b, r0 in enumerate(range(0, max((t.row for t in tiles), default=-1) + 1, pc.rows_per_unit)):
        band = [s for s in specs if r0 <= s.tile.row < r0 + pc.rows_per_unit]
        if band:
            units.append(UnlabeledUnit(f"{slide.name}_U{b:03d}", Box.union_all(s.tile.box for s in band), tuple(band)))
    return units


# ---------------------------------------------------------------------------
# files
# ---------------------------------------------------------------------------
def patch_dir(slide_dir: Path, key: str) -> Path:
    """``<slide_dir>/<key without the slide prefix>/`` — e.g. ``SLIDE-0235/R01_x8073_y2896``."""
    return slide_dir / key.split("_", 1)[1]


def write_image(folder: Path, image: np.ndarray, markers: Sequence[str]) -> None:
    """One ``<marker>.tif`` per channel of ``image`` (C, H, W) into the patch folder."""
    folder.mkdir(parents=True, exist_ok=True)
    for c, m in enumerate(markers):
        tmp = folder / f"{m}.tif.tmp"
        tifffile.imwrite(tmp, image[c], compression="zlib", photometric="minisblack", metadata={"axes": "YX", "marker": m})
        os.replace(tmp, folder / f"{m}.tif")


def write_cells(path: Path, rows: List[dict]) -> None:
    """The patch's cell table (``cells.csv``): local id -> HALO nucleus id, QC flag, area, centroid, border, ownership."""
    import pandas as pd
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    pd.DataFrame(rows, columns=[c for c in INSTANCE_COLUMNS if c not in ("patch_key", "slide", "region_id")]).to_csv(tmp, index=False)
    os.replace(tmp, path)


def write_mask(path: Path, labels: np.ndarray) -> None:
    if labels.max() > 65535:
        raise DatasetError(f"{path.name}: more than 65535 instances in one patch")
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.stem + ".tmp.png")
    if not cv2.imwrite(str(tmp), labels.astype(np.uint16)):
        raise DatasetError(f"could not write mask {path}")
    os.replace(tmp, path)


def read_image(folder: "str | Path", markers: Sequence[str]) -> np.ndarray:
    """The ``(C, H, W)`` uint16 stack of a patch folder, channels in ``markers`` order (one ``<marker>.tif`` each)."""
    p = Path(folder)
    planes = []
    for m in markers:
        f = p / f"{m}.tif"
        if not f.exists():
            raise DatasetError(f"patch image missing: {f}", hint="re-run `mxdata sample` (resume fills the gaps)")
        planes.append(tifffile.imread(f))
    return np.stack(planes)


def read_patch(folder: "str | Path", mask_path: "str | Path", markers: Sequence[str]) -> Tuple[np.ndarray, np.ndarray]:
    """``(image (C, H, W) uint16, labels (H, W) int32)``."""
    labels = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)
    if labels is None:
        raise DatasetError(f"cannot read mask {mask_path}")
    return read_image(folder, markers), labels.astype(np.int32)


# ---------------------------------------------------------------------------
# samplers (one region unit / one band -> table rows + files)
# ---------------------------------------------------------------------------
def _row(spec: PatchSpec, kind: str, unit_key: str, cfg: DataConfig, slide: MxSlide, kept: bool, **extra) -> dict:
    b, c = spec.tile.box, spec.tile.core
    row = {k: 0 for k in PATCH_COLUMNS}
    row.update(key=spec.key, kind=kind, unit=unit_key, slide=spec.slide, region_id=spec.region_id, region_key="", group="", split="",
               row=spec.tile.row, col=spec.tile.col, x0=b.x0, y0=b.y0, x1=b.x1, y1=b.y1,
               core_x0=c.x0, core_y0=c.y0, core_x1=c.x1, core_y1=c.y1, height=b.height, width=b.width,
               level=cfg.level, downsample=slide.downsamples[cfg.level], markers=",".join(cfg.markers),
               pixel_size_um=slide.pixel_size_um, has_qc=False, fg_fraction=np.nan, tissue_fraction=round(spec.tissue, 4),
               fov_overlap=round(spec.fov_overlap, 4), kept=kept, reject_reason=spec.reject,
               image=f"{spec.slide}/{spec.key.split('_', 1)[1]}" if kept else "", mask="")
    row.update(extra)
    return row


def _relabel_local(crop: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Global labels inside a crop -> ``1..K``; returns ``(local, global_label_per_local_id)``."""
    ids = np.unique(crop)
    ids = ids[ids > 0]
    lut = np.zeros(int(crop.max()) + 1 if crop.size else 1, dtype=np.int32)
    lut[ids] = np.arange(1, len(ids) + 1)
    return lut[crop], ids


def sample_labeled_unit(unit: RegionUnit, reg: RegionAnnotation, specs: Sequence[PatchSpec], cfg: DataConfig, slide: MxSlide,
                        gate: TissueGate, slide_dir: Path) -> Tuple[List[dict], List[dict]]:
    """Read the FOV once, cut every window, gate it, write the kept ones (image + mask); return table rows."""
    ds = slide.downsamples[cfg.level]
    stack = slide.read_region(unit.box, cfg.markers, cfg.level)
    full, _ = apply_qc_policy(reg.label_map(unit.box, ds), reg.flags, "keep_all")     # drops TINY only
    h, w = min(stack.shape[1], full.shape[0]), min(stack.shape[2], full.shape[1])
    stack, full = stack[:, :h, :w], full[:h, :w]
    ux0, uy0 = unit.box.at_downsample(ds)[:2]
    flags, nuc = reg.flags, reg.nuclei.set_index("label")
    prows, irows = [], []
    for s in specs:
        px0, py0, px1, py1 = s.tile.box.at_downsample(ds)
        img, lab = stack[:, py0 - uy0:py1 - uy0, px0 - ux0:px1 - ux0], full[py0 - uy0:py1 - uy0, px0 - ux0:px1 - ux0]
        local, glob = _relabel_local(lab)
        fg = float((lab > 0).mean())
        reject = f"fg<{cfg.patches.min_fg}" if fg < cfg.patches.min_fg else ""     # the labels decide; tissue fraction is recorded as data
        s = PatchSpec(s.key, s.slide, s.region_id, s.tile, gate.fraction(s.tile.box), 1.0, reject)
        fl = flags[glob]
        edge = np.unique(np.concatenate([local[0], local[-1], local[:, 0], local[:, -1]]))
        border = np.isin(np.arange(1, len(glob) + 1), edge)
        owned = np.array([s.tile.core.contains(nuc.at[g, "cx"], nuc.at[g, "cy"]) for g in glob], dtype=bool)
        kept = not reject
        if kept:
            folder = patch_dir(slide_dir, s.key)
            write_image(folder, img, cfg.markers)
            write_mask(folder / "mask.png", local)                                              # complete: every HALO cell (inspection)
            wrong_local = np.array([li for li, g in enumerate(glob, 1) if fl[li - 1] == NucleusFlag.WRONG], dtype=np.int32)
            write_mask(folder / "mask_true.png", np.where(np.isin(local, wrong_local), 0, local))   # confirmed cells only (training)
            rows = [{"patch_key": s.key, "local_id": li, "slide": s.slide, "region_id": s.region_id, "nucleus_id": int(g) - 1,
                     "label": int(g), "flag": NucleusFlag(int(flags[g])).name, "area_px": float(nuc.at[g, "area_px"]),
                     "cx": float(nuc.at[g, "cx"]), "cy": float(nuc.at[g, "cy"]), "touches_border": bool(border[li - 1]),
                     "owned": bool(owned[li - 1])} for li, g in enumerate(glob, 1)]
            write_cells(folder / "cells.csv", rows)
            irows += rows
        prows.append(_row(s, "labeled", unit.key, cfg, slide, kept, region_key=reg.key, has_qc=reg.has_qc, fg_fraction=round(fg, 4),
                          n_instances=len(glob), n_owned=int(owned.sum()), n_border=int(border.sum()),
                          n_ok=int((fl == NucleusFlag.OK).sum()), n_wrong=int((fl == NucleusFlag.WRONG).sum()),
                          n_no_qc_row=int((fl == NucleusFlag.NO_QC_ROW).sum()), n_unknown=int((fl == NucleusFlag.UNKNOWN).sum()),
                          mask=f"{s.slide}/{s.key.split('_', 1)[1]}/mask.png" if kept else ""))
    return prows, irows


def sample_unlabeled_unit(unit: UnlabeledUnit, cfg: DataConfig, slide: MxSlide, slide_dir: Path) -> List[dict]:
    """Read the band once, write every kept window as an image; return rows for kept and rejected windows."""
    ds = slide.downsamples[cfg.level]
    stack = slide.read_region(unit.band, cfg.markers, cfg.level) if any(not s.reject for s in unit.windows) else None
    bx0, by0 = unit.band.at_downsample(ds)[:2]
    rows = []
    for s in unit.windows:
        if not s.reject:
            px0, py0, px1, py1 = s.tile.box.at_downsample(ds)
            write_image(patch_dir(slide_dir, s.key), stack[:, py0 - by0:py1 - by0, px0 - bx0:px1 - bx0], cfg.markers)
        rows.append(_row(s, "unlabeled", unit.key, cfg, slide, not s.reject))
    return rows
