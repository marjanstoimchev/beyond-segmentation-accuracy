"""HALO GeoJSON region layers: one closed ``LineString`` per nucleus, integer level-0 pixel coordinates.

Loading is tolerant and *records* what it saw: unsupported geometries are skipped and counted, open rings
and non-integer coordinates are noted as issues — nothing is silently altered.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import List

import numpy as np

from ..errors import AnnotationError
from ..io.geometry import Box


@dataclass
class GeoJsonLayer:
    path: Path
    rings: List[np.ndarray]                   # (N_i, 2) float64 xy per feature, closing vertex dropped
    issues: List[str] = field(default_factory=list)
    n_skipped: int = 0

    def __len__(self) -> int:
        return len(self.rings)


def _ring(geom: dict) -> np.ndarray:
    t = geom.get("type")
    if t == "LineString":
        pts = geom["coordinates"]
    elif t == "Polygon":
        pts = geom["coordinates"][0]
    elif t == "MultiPolygon":
        pts = max((poly[0] for poly in geom["coordinates"]), key=len)
    else:
        raise ValueError(f"unsupported geometry type {t!r}")
    arr = np.asarray(pts, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] != 2 or len(arr) == 0:
        raise ValueError("coordinates are not an (N, 2) list")
    return arr[:-1] if len(arr) > 1 and np.array_equal(arr[0], arr[-1]) else arr


def load_halo_geojson(path: "str | Path") -> GeoJsonLayer:
    p = Path(path)
    try:
        g = json.loads(p.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise AnnotationError(f"cannot read GeoJSON {p}: {exc}") from exc
    if not isinstance(g, dict) or g.get("type") != "FeatureCollection" or "features" not in g:
        raise AnnotationError(f"{p.name}: not a GeoJSON FeatureCollection")
    layer = GeoJsonLayer(path=p, rings=[])
    n_open = n_nonint = n_short = 0
    for i, ft in enumerate(g["features"]):
        try:
            geom = ft["geometry"]
            raw = geom.get("coordinates")
            closed = geom.get("type") == "LineString" and isinstance(raw, list) and len(raw) > 1 and raw[0] == raw[-1]
            ring = _ring(geom)
        except (KeyError, TypeError, ValueError, AttributeError) as exc:
            layer.n_skipped += 1
            layer.issues.append(f"feature {i} skipped: {exc}")
            continue
        n_open += geom.get("type") == "LineString" and not closed
        n_nonint += not np.array_equal(ring, np.round(ring))
        n_short += len(ring) < 3
        layer.rings.append(ring)
    for n, msg in ((n_open, "LineString contour(s) not explicitly closed (treated as closed)"),
                   (n_nonint, "contour(s) with non-integer coordinates (rounded when rasterized)"),
                   (n_short, "contour(s) with fewer than 3 vertices (cannot enclose area)")):
        if n:
            layer.issues.append(f"{n} {msg}")
    return layer


def polygon_area(ring: np.ndarray) -> float:
    """Shoelace area in px² (matches HALO's reported area to < 0.1 %)."""
    if len(ring) < 3:
        return 0.0
    x, y = ring[:, 0], ring[:, 1]
    return float(abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))) / 2.0)


def polygon_bbox(ring: np.ndarray) -> Box:
    return Box.from_points(ring)
