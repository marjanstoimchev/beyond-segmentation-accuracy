"""Tissue gate: a coarse mask from one nuclear-channel thumbnail, then a per-window coverage look-up.

Otsu (in log intensity: fluorescence spans decades) picks the threshold from the data; when its two classes
are not separated — a near-uniform image, where Otsu happily splits noise in half — or one class is (almost)
empty, a percentile cut is used instead. A binary closing fills pinholes. Which path was taken is recorded.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from ..io.geometry import Box
from ..io.slide import MxSlide


def otsu_threshold(values: np.ndarray, bins: int = 256) -> float:
    """Otsu's threshold on a 1-D sample (the cut lies after the bin that maximises between-class variance)."""
    v = np.asarray(values, dtype=np.float64).ravel()
    if v.size == 0:
        return 0.0
    hist, edges = np.histogram(v, bins=bins)
    centers = (edges[:-1] + edges[1:]) / 2.0
    w = hist.astype(np.float64)
    w0, s0 = np.cumsum(w), np.cumsum(w * centers)
    w1, s1 = w.sum() - w0, s0[-1] - s0
    valid = (w0 > 0) & (w1 > 0)
    between = np.where(valid, w0 * w1 * (s0 / np.maximum(w0, 1e-12) - s1 / np.maximum(w1, 1e-12)) ** 2, -1.0)
    return float(edges[int(np.argmax(between)) + 1]) if between.max() > 0 else float(np.median(v))


def tissue_mask(thumb: np.ndarray, fallback_percentile: float = 60.0, closing: int = 5,
                min_fraction: float = 0.005, max_fraction: float = 0.995, min_separation: float = 0.25) -> tuple:
    """``(mask, info)`` for a nuclear-channel thumbnail; ``min_separation`` is in log1p units (0.25 ≈ 28 %)."""
    x = np.log1p(np.asarray(thumb, dtype=np.float64))
    thr, method = otsu_threshold(x), "otsu"
    mask = x > thr
    frac = float(mask.mean())
    sep = float(x[mask].mean() - x[~mask].mean()) if 0 < frac < 1 else 0.0
    if frac < min_fraction or frac > max_fraction or sep < min_separation:
        thr, method = float(np.percentile(x, fallback_percentile)), f"percentile_{fallback_percentile:g}"
        mask = x > thr
    if closing > 1:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (closing, closing))
        mask = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_CLOSE, k).astype(bool)
    return mask, {"method": method, "threshold_log1p": thr, "threshold_intensity": float(np.expm1(thr)),
                  "class_separation_log1p": sep, "tissue_fraction": float(mask.mean()), "shape": list(mask.shape)}


@dataclass
class TissueGate:
    """A slide's coarse tissue mask at its downsample; ``fraction(box)`` answers per level-0 window."""

    mask: np.ndarray
    downsample: float
    info: dict = field(default_factory=dict)

    @classmethod
    def from_slide(cls, slide: MxSlide, marker: str = "DAPI", max_dim: int = 2048) -> "TissueGate":
        thumb, ds = slide.thumbnail(max_dim=max_dim, markers=[marker])
        mask, info = tissue_mask(thumb[0])
        return cls(mask, ds, {**info, "marker": marker, "downsample": ds})

    def fraction(self, box: Box) -> float:
        x0, y0, x1, y1 = box.at_downsample(self.downsample)
        h, w = self.mask.shape
        x0, y0 = min(max(0, x0), w - 1), min(max(0, y0), h - 1)
        return float(self.mask[y0:max(min(h, y1), y0 + 1), x0:max(min(w, x1), x0 + 1)].mean())
