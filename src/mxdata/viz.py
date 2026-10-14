"""Display helpers for notebooks (arrays in, arrays out)."""
from __future__ import annotations

from typing import Optional

import cv2
import numpy as np


def to_uint8(img: np.ndarray, pmin: float = 1.0, pmax: float = 99.5) -> np.ndarray:
    """Percentile-stretch a 2-D array to uint8 for display."""
    img = np.asarray(img).astype(np.float32)
    lo, hi = np.percentile(img, [pmin, pmax])
    return (np.clip((img - lo) / max(hi - lo, 1e-6), 0, 1) * 255).astype(np.uint8)


def gray_to_rgb(u8: np.ndarray) -> np.ndarray:
    return np.repeat(u8[..., None], 3, axis=-1)


def colorize_labels(labels: np.ndarray, seed: int = 0) -> np.ndarray:
    """Random colour per instance id (background black)."""
    lut = np.random.default_rng(seed).integers(40, 255, size=(int(labels.max()) + 1, 3), dtype=np.uint8)
    lut[0] = 0
    return lut[labels]


def label_contours(labels: np.ndarray, thickness: int = 1) -> np.ndarray:
    """Nucleus pixels with a 4-neighbour of a different label (complete on every side)."""
    b = np.zeros(labels.shape, dtype=bool)
    b[:, 1:] |= labels[:, 1:] != labels[:, :-1]
    b[:, :-1] |= labels[:, 1:] != labels[:, :-1]
    b[1:, :] |= labels[1:, :] != labels[:-1, :]
    b[:-1, :] |= labels[1:, :] != labels[:-1, :]
    b &= labels > 0
    if thickness > 1:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * thickness - 1, 2 * thickness - 1))
        b = cv2.dilate(b.astype(np.uint8), k).astype(bool) & (labels > 0)
    return b


def overlay_labels(rgb: np.ndarray, labels: np.ndarray, color=(0, 255, 0), ignore: Optional[np.ndarray] = None,
                   ignore_color=(255, 60, 60), thickness: int = 1) -> np.ndarray:
    """Instance outlines (and the ignore mask, tinted) over an RGB uint8 image — for checking alignment against pixels."""
    out = rgb.copy()
    edges = label_contours(labels, thickness)
    out[edges] = color
    if ignore is not None and ignore.any():
        out[ignore] = (out[ignore] * 0.5 + np.array(ignore_color) * 0.5).astype(np.uint8)
        out[edges & ignore] = ignore_color
    return out


def overlay_masks(rgb: np.ndarray, labels: np.ndarray, ignore: Optional[np.ndarray] = None, alpha: float = 0.45,
                  ignore_color=(255, 60, 60), seed: int = 0, colors: Optional[np.ndarray] = None) -> np.ndarray:
    """The **filled** instance masks blended over an RGB uint8 image — one colour per nucleus (or the per-id ``colors``
    LUT), ``ignore`` pixels tinted ``ignore_color``. What you see is exactly the label map."""
    out = rgb.astype(np.float32)
    fill = (colors[labels] if colors is not None else colorize_labels(labels, seed)).astype(np.float32)
    m = labels > 0
    if ignore is not None and ignore.any():
        fill[ignore] = np.asarray(ignore_color, np.float32)
        m = m | ignore
    out[m] = out[m] * (1 - alpha) + fill[m] * alpha
    return np.clip(out, 0, 255).astype(np.uint8)
