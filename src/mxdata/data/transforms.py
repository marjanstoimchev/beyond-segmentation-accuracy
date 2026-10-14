"""Normalization and joint image/mask augmentation on dict samples (numpy in, numpy out).

A sample is ``{"image": (C,H,W) float32, "labels": (H,W) int32, "ignore": (H,W) bool, "meta": dict}`` (labels/ignore
absent in self-supervised mode). Random transforms take an explicit ``rng`` so every DataLoader worker draws from
its own seeded generator.
"""
from __future__ import annotations

from typing import Dict, Optional, Sequence

import numpy as np

from ..errors import ValidationError

Sample = Dict[str, object]
SPATIAL_KEYS = ("image", "labels", "ignore")


def normalize_fixed(image: np.ndarray, lo: Sequence[float], hi: Sequence[float]) -> np.ndarray:
    """``(x - lo) / (hi - lo)`` per channel with fixed bounds, clipped to [0, 1], float32."""
    lo, hi = np.asarray(lo, np.float32).reshape(-1, 1, 1), np.asarray(hi, np.float32).reshape(-1, 1, 1)
    return np.clip((image.astype(np.float32) - lo) / np.maximum(hi - lo, 1e-6), 0.0, 1.0)


def normalize_percentile(image: np.ndarray, pmin: float = 1.0, pmax: float = 99.5) -> np.ndarray:
    """Per-channel percentile stretch computed on *this* array."""
    flat = image.reshape(image.shape[0], -1)
    return normalize_fixed(image, np.percentile(flat, pmin, axis=1), np.percentile(flat, pmax, axis=1))


class Normalizer:
    """``slide``: fixed per-marker bounds, one pair **per slide** (from ``{slide: NormStats}``); ``region``: percentiles of
    the array being normalized; ``none``: divide by the dtype maximum. Called as ``normalizer(image, slide_name)``."""

    def __init__(self, mode: str = "slide", per_slide: Optional[dict] = None, pmin: float = 1.0, pmax: float = 99.5):
        if mode not in ("slide", "region", "none"):
            raise ValidationError(f"unknown normalization mode {mode!r}")
        if mode == "slide" and not per_slide:
            raise ValidationError("slide-level normalization needs per-slide (lo, hi) bounds — compute NormStats first")
        self.mode, self.pmin, self.pmax = mode, pmin, pmax
        self.per_slide = {k: (np.asarray(a, np.float32), np.asarray(b, np.float32)) for k, (a, b) in (per_slide or {}).items()}

    @classmethod
    def from_stats(cls, stats: dict, markers: Sequence[str], pmin: float = 1.0, pmax: float = 99.5) -> "Normalizer":
        return cls("slide", {k: v.arrays(markers) for k, v in stats.items()}, pmin, pmax)

    def bounds(self, slide: Optional[str] = None):
        if slide in self.per_slide:
            return self.per_slide[slide]
        if len(self.per_slide) == 1:
            return next(iter(self.per_slide.values()))
        raise ValidationError(f"no normalization bounds for slide {slide!r} (have {sorted(self.per_slide)})")

    def __call__(self, image: np.ndarray, slide: Optional[str] = None) -> np.ndarray:
        if self.mode == "slide":
            lo, hi = self.bounds(slide)
            if len(lo) != image.shape[0]:
                raise ValidationError(f"normalizer has {len(lo)} channels, image has {image.shape[0]}")
            return normalize_fixed(image, lo, hi)
        if self.mode == "region":
            return normalize_percentile(image, self.pmin, self.pmax)
        return image.astype(np.float32) / (float(np.iinfo(image.dtype).max) if np.issubdtype(image.dtype, np.integer) else 1.0)

    def __repr__(self):
        if self.mode != "slide":
            return f"Normalizer({self.mode})"
        return "Normalizer(slide, " + ", ".join(f"{k}: lo={np.round(a, 1).tolist()} hi={np.round(b, 1).tolist()}" for k, (a, b) in self.per_slide.items()) + ")"


def flip_rot90(sample: Sample, rng: np.random.Generator, flip: bool = True, rot90: bool = True) -> Sample:
    """A random D4 transform applied jointly to image and masks."""
    k = int(rng.integers(4)) if rot90 else 0
    fh, fv = (bool(rng.random() < 0.5) if flip else False for _ in range(2))
    out = dict(sample)
    for key in SPATIAL_KEYS:
        if key in out:
            a = np.rot90(np.asarray(out[key]), k, axes=(-2, -1)) if k else np.asarray(out[key])
            a = np.flip(a, axis=-1) if fh else a
            out[key] = np.ascontiguousarray(np.flip(a, axis=-2) if fv else a)
    return out


class Augment:
    """Config-driven augmentation ``(sample, rng) -> sample``."""

    def __init__(self, flip: bool = True, rot90: bool = True):
        self.flip, self.rot90 = flip, rot90

    def __call__(self, sample: Sample, rng: np.random.Generator) -> Sample:
        return flip_rot90(sample, rng, self.flip, self.rot90) if (self.flip or self.rot90) else sample

    def __repr__(self):
        return f"Augment(flip={self.flip}, rot90={self.rot90})"
