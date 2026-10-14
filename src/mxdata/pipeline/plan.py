"""Plan: decide the work before doing any of it — pure functions, no IO, countable for a dry run.

A *unit* of work is a region (annotated FOV). :func:`tile_grid` lays windows over a box and gives each
one an **owned core** (its stride cell; the last row/column own to the edge): an object belongs to the
window whose core contains its centroid, so nothing is counted twice and no IoU matching is needed.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence

from ..annotations.regions import RegionAnnotation
from ..config import DataConfig
from ..io.geometry import Box
from ..io.slide import MxSlide


@dataclass(frozen=True)
class RegionUnit:
    key: str            # "<slide>_R<id:02d>"; output names derive from it
    region_id: int
    box: Box
    n_nuclei: int
    has_qc: bool

    def to_dict(self) -> dict:
        return {"key": self.key, "region_id": self.region_id, "box": self.box.to_dict(), "n_nuclei": self.n_nuclei, "has_qc": self.has_qc}


@dataclass(frozen=True)
class Tile:
    row: int
    col: int
    box: Box            # the window (read this)
    core: Box           # the owned core


def plan_units(cfg: DataConfig, slide: MxSlide, regions: Sequence[RegionAnnotation]) -> List[RegionUnit]:
    """The explicit, ordered list of region units a run will process."""
    return [RegionUnit(f"{slide.name}_R{r.region_id:02d}", r.region_id, r.box, len(r), r.has_qc)
            for r in sorted(regions, key=lambda r: r.region_id) if cfg.regions is None or r.region_id in cfg.regions]


def _positions(length: int, size: int, stride: int) -> List[int]:
    if length <= size:
        return [0]
    pos = list(range(0, length - size + 1, stride))
    return pos if pos[-1] == length - size else pos + [length - size]


def tile_grid(box: Box, size: int, stride: Optional[int] = None, downsample: float = 1.0) -> List[Tile]:
    """Windows of ``size`` px (at the read level) covering ``box`` (level-0) on a ``stride`` grid.

    Edge windows are shifted inward so every window lies inside the box (a box smaller than one window
    yields a single window covering it). Cores partition the box exactly.
    """
    stride = stride or size
    if size <= 0 or not 0 < stride <= size:
        raise ValueError("need 0 < stride <= size")
    s0, st0 = max(1, round(size * downsample)), max(1, round(stride * downsample))
    ys, xs = _positions(box.height, s0, st0), _positions(box.width, s0, st0)
    y_edges, x_edges = [box.y0 + y for y in ys] + [box.y1], [box.x0 + x for x in xs] + [box.x1]
    return [Tile(r, c, Box(box.x0 + x, box.y0 + y, min(box.x0 + x + s0, box.x1), min(box.y0 + y + s0, box.y1)),
                 Box(x_edges[c], y_edges[r], x_edges[c + 1], y_edges[r + 1]))
            for r, y in enumerate(ys) for c, x in enumerate(xs)]
