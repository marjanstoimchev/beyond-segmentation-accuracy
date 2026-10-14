"""One annotated field of view (FOV): nucleus contours, QC flags as data, and its bounding box."""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence

import numpy as np
import pandas as pd

from ..errors import AnnotationError
from ..io.geometry import Box
from ..io.paths import SlidePaths
from .geojson import GeoJsonLayer, load_halo_geojson, polygon_area, polygon_bbox
from .qc import QC_WRONG, load_qc_csv, match_qc_to_polygons
from .raster import NucleusFlag, rasterize_polygons

logger = logging.getLogger(__name__)

NUCLEUS_COLUMNS = ["nucleus_id", "label", "x0", "y0", "x1", "y1", "cx", "cy", "area_px", "n_points",
                   "qc_row", "qc_flag", "flag", "qc_area_um2", "qc_perimeter_um"]


def _nuclei_frame(rings: Sequence[np.ndarray], qc: Optional[pd.DataFrame], min_area_px: float):
    """One row per contour; QC joined by bbox; a NucleusFlag per contour (nothing dropped here)."""
    rows = []
    for i, r in enumerate(rings):
        b = polygon_bbox(r)
        rows.append({"nucleus_id": i, "label": i + 1, "x0": b.x0, "y0": b.y0, "x1": b.x1, "y1": b.y1,
                     "cx": float(r[:, 0].mean()), "cy": float(r[:, 1].mean()), "area_px": polygon_area(r), "n_points": len(r)})
    df = pd.DataFrame(rows, columns=NUCLEUS_COLUMNS[:10])
    df["qc_row"], df["qc_flag"], df["qc_area_um2"], df["qc_perimeter_um"] = -1, "", np.nan, np.nan
    flag = np.full(len(df), int(NucleusFlag.UNKNOWN), dtype=np.int8)
    match = None
    if qc is not None and len(df):
        rows_of, match = match_qc_to_polygons(rings, qc)
        m = rows_of >= 0
        df.loc[m, "qc_row"] = rows_of[m]
        df.loc[m, "qc_flag"] = qc["Flag"].fillna("").astype(str).str.strip().to_numpy()[rows_of[m]]
        for col, prefix in (("qc_area_um2", "Region Area"), ("qc_perimeter_um", "Region Perimeter")):
            src = next((c for c in qc.columns if c.startswith(prefix)), None)
            if src:
                df.loc[m, col] = pd.to_numeric(qc[src], errors="coerce").to_numpy()[rows_of[m]]
        flag[m] = NucleusFlag.OK
        flag[m & (df.qc_flag == QC_WRONG).to_numpy()] = NucleusFlag.WRONG
        flag[~m] = NucleusFlag.NO_QC_ROW
    if min_area_px > 0 and len(df):
        flag[(df.area_px < min_area_px).to_numpy()] = NucleusFlag.TINY
    df["flag"] = [NucleusFlag(int(f)).name for f in flag]
    return df[NUCLEUS_COLUMNS], match


@dataclass
class RegionAnnotation:
    slide: str
    region_id: int
    layer: GeoJsonLayer
    nuclei: pd.DataFrame                      # one row per contour (NUCLEUS_COLUMNS)
    qc_path: Optional[Path] = None
    qc: Optional[pd.DataFrame] = None
    qc_match: Optional[dict] = None
    min_area_px: float = 0.0

    @property
    def key(self) -> str:
        """Identity across slides: ``<slide>_R<id:02d>``."""
        return f"{self.slide}_R{self.region_id:02d}"

    @property
    def rings(self) -> List[np.ndarray]:
        return self.layer.rings

    @property
    def box(self) -> Box:
        """Tight box around all contours: the annotated FOV."""
        if not len(self.nuclei):
            raise AnnotationError(f"{self.key}: no contours, no box")
        return Box(self.nuclei.x0.min(), self.nuclei.y0.min(), self.nuclei.x1.max(), self.nuclei.y1.max())

    def __len__(self) -> int:
        return len(self.rings)

    @property
    def has_qc(self) -> bool:
        return self.qc is not None

    @property
    def flags(self) -> np.ndarray:
        """``int8`` look-up table ``flags[label] = NucleusFlag`` (index 0 = background)."""
        lut = np.zeros(len(self.nuclei) + 1, dtype=np.int8)
        lut[1:] = [int(NucleusFlag[n]) for n in self.nuclei.flag]
        return lut

    def count(self, flag: NucleusFlag) -> int:
        return int((self.nuclei.flag == flag.name).sum())

    @property
    def n_wrong(self) -> int:
        return self.count(NucleusFlag.WRONG)

    def label_map(self, box: Optional[Box] = None, downsample: float = 1.0) -> np.ndarray:
        """Every contour painted (``label = nucleus_id + 1``); apply a QC policy afterwards with :func:`apply_qc_policy`."""
        return rasterize_polygons(self.rings, box or self.box, downsample, ids=self.nuclei.label.to_numpy())

    def row(self, pixel_size_um: Optional[float] = None) -> dict:
        b = self.box
        d = {"region": self.region_id, "n_nuclei": len(self), "x0": b.x0, "y0": b.y0, "x1": b.x1, "y1": b.y1,
             "width_px": b.width, "height_px": b.height, "has_qc": self.has_qc, "n_qc_rows": len(self.qc) if self.has_qc else 0,
             "n_ok": self.count(NucleusFlag.OK), "n_wrong": self.n_wrong,
             "pct_wrong": round(100.0 * self.n_wrong / len(self), 2) if self.has_qc else np.nan,
             "n_no_qc_row": self.count(NucleusFlag.NO_QC_ROW), "n_tiny": self.count(NucleusFlag.TINY),
             "median_area_px": float(self.nuclei.area_px.median()), "geojson_issues": len(self.layer.issues)}
        if pixel_size_um:
            d.update(width_um=round(b.width * pixel_size_um, 1), height_um=round(b.height * pixel_size_um, 1),
                     median_area_um2=round(d["median_area_px"] * pixel_size_um ** 2, 2))
        return d


def load_region(paths: SlidePaths, region_id: int, min_area_px: float = 0.0) -> RegionAnnotation:
    layer = load_halo_geojson(paths.halo_files[region_id])
    qc_path = paths.qc_files.get(region_id)
    qc = load_qc_csv(qc_path) if qc_path else None
    nuclei, match = _nuclei_frame(layer.rings, qc, min_area_px)
    return RegionAnnotation(paths.slide, region_id, layer, nuclei, qc_path, qc, match, min_area_px)


def load_regions(paths: SlidePaths, region_ids: Optional[Sequence[int]] = None, min_area_px: float = 0.0) -> List[RegionAnnotation]:
    """Every (or the selected) HALO region layer of a slide, QC joined where a table exists."""
    ids = list(region_ids) if region_ids is not None else paths.region_ids
    if missing := [r for r in ids if r not in paths.halo_files]:
        raise AnnotationError(f"no HALO layer for region(s) {missing} of {paths.slide}", hint=f"available regions: {paths.region_ids}")
    regs = [load_region(paths, r, min_area_px) for r in ids]
    logger.info("%s: loaded %d regions, %d nuclei", paths.slide, len(regs), sum(len(r) for r in regs))
    return regs


def regions_table(regions: Sequence[RegionAnnotation], pixel_size_um: Optional[float] = None) -> pd.DataFrame:
    rows = [r.row(pixel_size_um) for r in regions]
    return pd.DataFrame(rows).set_index("region") if rows else pd.DataFrame(columns=["region"]).set_index("region")


def nuclei_table(regions: Sequence[RegionAnnotation]) -> pd.DataFrame:
    """Every nucleus of every region in one table (``region`` column first)."""
    return pd.concat([r.nuclei.assign(region=r.region_id) for r in regions], ignore_index=True)[["region"] + NUCLEUS_COLUMNS]


def overlap_table(regions: Sequence[RegionAnnotation], centroid_tol_px: float = 3.0) -> pd.DataFrame:
    """Pairs of regions whose FOV boxes overlap, with how many nuclei sit in the shared area and how many
    coincide (centroid within ``centroid_tol_px``) — i.e. duplicated annotations."""
    from scipy.spatial import cKDTree

    cols = ["region_a", "region_b", "overlap_w", "overlap_h", "overlap_frac_a", "overlap_frac_b", "nuclei_a", "nuclei_b", "coincident"]
    rows = []
    for i, a in enumerate(regions):
        for b in regions[i + 1:]:
            inter = a.box.intersect(b.box)
            if inter is None:
                continue
            ina = a.nuclei[[inter.contains(x, y) for x, y in zip(a.nuclei.cx, a.nuclei.cy)]]
            inb = b.nuclei[[inter.contains(x, y) for x, y in zip(b.nuclei.cx, b.nuclei.cy)]]
            n_dup = int((cKDTree(inb[["cx", "cy"]].to_numpy()).query(ina[["cx", "cy"]].to_numpy())[0] <= centroid_tol_px).sum()) if len(ina) and len(inb) else 0
            rows.append([a.region_id, b.region_id, inter.width, inter.height, round(inter.area / a.box.area, 4),
                         round(inter.area / b.box.area, 4), len(ina), len(inb), n_dup])
    return pd.DataFrame(rows, columns=cols)
