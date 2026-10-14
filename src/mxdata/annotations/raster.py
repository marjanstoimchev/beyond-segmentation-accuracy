"""Contours -> instance label map, and the QC policy that turns per-nucleus flags into ``(labels, ignore)``."""
from __future__ import annotations

from enum import IntEnum
from typing import Optional, Sequence, Tuple

import cv2
import numpy as np

from ..errors import AnnotationError
from ..io.geometry import Box
from .geojson import polygon_area

QC_POLICIES = ("keep_all", "ignore_wrong", "drop_wrong")


class NucleusFlag(IntEnum):
    """Per-nucleus status, stored as an ``int8`` look-up table indexed by label id (0 = background)."""

    UNKNOWN = 0      # region has no QC table
    OK = 1           # QC row present, Flag blank
    WRONG = 2        # QC row present, Flag == "Wrong"
    NO_QC_ROW = 3    # region has a QC table but this contour has no row
    TINY = 4         # area below min_area_px: an artefact, never painted


def rasterize_polygons(rings: Sequence[np.ndarray], box: Box, downsample: float = 1.0,
                       ids: Optional[Sequence[int]] = None) -> np.ndarray:
    """Paint contours into an ``int32`` label map covering ``box`` (level-0 px), largest first so a small
    nucleus on top of a big one is never hidden. ``ids[i]`` is the label of ring ``i`` (default ``i + 1``)."""
    x0, y0, x1, y1 = box.at_downsample(downsample)
    lab = np.zeros((y1 - y0, x1 - x0), dtype=np.int32)
    ids = np.arange(1, len(rings) + 1) if ids is None else np.asarray(ids)
    if len(ids) != len(rings):
        raise AnnotationError("ids must have one entry per ring")
    for i in np.argsort([-polygon_area(r) for r in rings], kind="stable"):
        if len(rings[i]) >= 3:
            pts = np.round(rings[i] / downsample - [x0, y0]).astype(np.int32).reshape(-1, 1, 2)
            cv2.fillPoly(lab, [pts], int(ids[i]))
    return lab


def apply_qc_policy(labels: np.ndarray, flags: np.ndarray, policy: str) -> Tuple[np.ndarray, np.ndarray]:
    """``(labels, ignore)`` under a QC policy. ``flags[label_id]`` is a :class:`NucleusFlag`. TINY nuclei are
    always removed; then ``keep_all`` ignores nothing, ``ignore_wrong`` puts WRONG pixels in ``ignore``,
    ``drop_wrong`` additionally removes WRONG nuclei from the labels."""
    if policy not in QC_POLICIES:
        raise AnnotationError(f"qc_policy must be one of {QC_POLICIES}, got {policy!r}")
    flags = np.asarray(flags, dtype=np.int8)
    if labels.size and int(labels.max()) >= len(flags):
        raise AnnotationError(f"flags has {len(flags)} entries but labels go up to {int(labels.max())}")
    pix = flags[labels]
    labels = np.where(pix == NucleusFlag.TINY, 0, labels)
    ignore = (pix == NucleusFlag.WRONG) if policy != "keep_all" else np.zeros(labels.shape, dtype=bool)
    if policy == "drop_wrong":
        labels = np.where(ignore, 0, labels)
    return labels, ignore
