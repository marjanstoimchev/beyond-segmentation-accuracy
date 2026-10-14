"""D4 augmentation shared by every model, and Cellpose's flow targets with the matching vector transform.

A horizontal flip negates dX, a vertical flip negates dY, a 90° counter-clockwise rotation maps (dY, dX) -> (-dX, dY).
"""
from __future__ import annotations

import logging
from typing import Tuple

import numpy as np


def d4_arrays(*arrays: np.ndarray, k: int, fh: bool, fv: bool) -> Tuple[np.ndarray, ...]:
    """Apply ``rot90`` ``k`` times (counter-clockwise) then horizontal / vertical flips to arrays whose last two axes are (H, W)."""
    out = []
    for a in arrays:
        a = np.rot90(a, k % 4, axes=(-2, -1)) if k % 4 else a
        a = np.flip(a, -1) if fh else a
        a = np.flip(a, -2) if fv else a
        out.append(np.ascontiguousarray(a))
    return tuple(out)


def d4_flows(target: np.ndarray, k: int, fh: bool, fv: bool) -> np.ndarray:
    """The same D4 transform for a Cellpose target ``(3, H, W)`` = [cellprob, dY, dX], with the vectors transformed too."""
    tgt = target
    for _ in range(k % 4):
        tgt = np.rot90(tgt, 1, axes=(-2, -1))
        tgt = np.stack([tgt[0], -tgt[2], tgt[1]])                     # (dY, dX) -> (-dX, dY)
    if fh:
        tgt = np.flip(tgt, -1)
        tgt = np.stack([tgt[0], tgt[1], -tgt[2]])
    if fv:
        tgt = np.flip(tgt, -2)
        tgt = np.stack([tgt[0], -tgt[1], tgt[2]])
    return np.ascontiguousarray(tgt)


def labels_to_targets(labels: np.ndarray, device=None) -> np.ndarray:
    """``(3, H, W)`` float32 = [cellprob (0/1), dY, dX] from an instance label map (``cellpose.dynamics.labels_to_flows``)."""
    from cellpose import dynamics
    logging.getLogger("cellpose.dynamics").setLevel(logging.WARNING)
    if not labels.any():
        return np.zeros((3, *labels.shape), np.float32)
    flows = dynamics.labels_to_flows([labels.astype(np.int32).copy()], device=device)[0]   # (4, H, W): labels, cellprob, dY, dX
    return flows[1:].astype(np.float32)


def cellpose_normalize(image: np.ndarray) -> np.ndarray:
    """Cellpose's per-image, per-channel percentile (1, 99) normalization — what ``CellposeModel.eval`` applies, so training
    and inference see identical inputs. ``(C, H, W)`` in, float32 ``(C, H, W)`` out."""
    from cellpose.transforms import normalize_img
    return normalize_img(image.astype(np.float32), normalize=True, axis=0).astype(np.float32)
