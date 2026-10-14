"""Marker images, pure numpy. ``render_channels`` is THE network input: three channels (red, green, blue), each one marker of
the panel on its slide-level [0, 1] scale — or blank — so a pretrained three-channel encoder is used as is. ``compose_rgb``
is the additive display composite of any number of markers with the panel's colors: figures only, never a network input."""
from __future__ import annotations

from typing import Dict, Optional, Sequence, Tuple

import numpy as np

Channels = Tuple[Optional[str], Optional[str], Optional[str]]


def hex_to_rgb(hexcol: str) -> np.ndarray:
    h = hexcol.lstrip("#")
    return np.array([int(h[i:i + 2], 16) / 255 for i in (0, 2, 4)], np.float32)


def marker_index(markers: Sequence[str], name: str) -> int:
    """Position of ``name`` in ``markers`` (case-insensitive)."""
    low = [m.lower() for m in markers]
    if name.lower() not in low:
        raise KeyError(f"marker {name!r} is not among {list(markers)}")
    return low.index(name.lower())


def select_markers(stack: np.ndarray, markers: Sequence[str], wanted: Sequence[str]) -> np.ndarray:
    """``(C, H, W)`` stack with channels ``markers`` -> the ``wanted`` markers, in that order."""
    return stack[[marker_index(markers, m) for m in wanted]]


def scaled(x: np.ndarray, lo: float, hi: float) -> np.ndarray:
    """Intensities -> [0, 1] on a slide-level (lo, hi) window, float32."""
    return np.clip((x.astype(np.float32) - lo) / max(hi - lo, 1e-6), 0.0, 1.0)


def render_channels(stack: np.ndarray, markers: Sequence[str], norms: Dict[str, Tuple[float, float]], channels: Channels) -> np.ndarray:
    """``stack`` (C, H, W) native intensities whose channels are ``markers`` -> ``(3, H, W)`` float32 in [0, 1]: channel k holds
    marker ``channels[k]`` scaled by its slide-level ``norms[marker] = (lo, hi)``, or zeros where ``channels[k]`` is None."""
    if len(channels) != 3:
        raise ValueError(f"three channels (red, green, blue) expected, got {channels}")
    out = np.zeros((3, *stack.shape[1:]), np.float32)
    for k, m in enumerate(channels):
        if m is not None:
            out[k] = scaled(stack[marker_index(markers, m)], *norms_of(norms, m))
    return out


def norms_of(norms: Dict[str, Tuple[float, float]], name: str) -> Tuple[float, float]:
    """``norms[name]``, matching the marker name case-insensitively."""
    for m, lohi in norms.items():
        if m.lower() == name.lower():
            return lohi
    raise KeyError(f"no statistics for marker {name!r}; have {list(norms)}")


def compose_rgb(stack: np.ndarray, markers: Sequence[str], colors: Dict[str, str],
                norms: Optional[Dict[str, Tuple[float, float]]] = None) -> np.ndarray:
    """Additive display composite: each marker normalized to [0, 1] (slide-level ``norms``, else per-patch 1--99 percentiles),
    tinted with its panel color, summed and clipped. Returns (3, H, W) float32 RGB in [0, 1]."""
    rgb = np.zeros((3, *stack.shape[1:]), np.float32)
    for c, name in enumerate(markers):
        tint = hex_to_rgb(colors[name])
        x = stack[c].astype(np.float32)
        lo, hi = (norms or {}).get(name) or np.percentile(x, (1.0, 99.0))
        rgb += tint[:, None, None] * scaled(x, lo, hi)[None]
    return np.clip(rgb, 0.0, 1.0)
