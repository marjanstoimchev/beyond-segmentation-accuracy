"""``Box``: a rectangle in **slide level-0 pixels**, half-open ``[x0, x1) x [y0, y1)``.

Every coordinate in mxdata is level-0; reading at a coarser pyramid level scales the box, so a region
means the same thing at any resolution.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional, Tuple

import numpy as np


@dataclass(frozen=True)
class Box:
    x0: int
    y0: int
    x1: int
    y1: int

    def __post_init__(self):
        for n in ("x0", "y0", "x1", "y1"):
            object.__setattr__(self, n, int(getattr(self, n)))
        if self.x1 <= self.x0 or self.y1 <= self.y0:
            raise ValueError(f"empty box: ({self.x0}, {self.y0}, {self.x1}, {self.y1})")

    @property
    def width(self) -> int:
        return self.x1 - self.x0

    @property
    def height(self) -> int:
        return self.y1 - self.y0

    @property
    def shape(self) -> Tuple[int, int]:
        return (self.height, self.width)

    @property
    def area(self) -> int:
        return self.width * self.height

    @property
    def xyxy(self) -> Tuple[int, int, int, int]:
        return (self.x0, self.y0, self.x1, self.y1)

    def pad(self, n: int, clamp_to: Optional[Tuple[int, int]] = None) -> "Box":
        """Grow by ``n`` px on every side, optionally clamped to a ``(width, height)`` canvas."""
        b = Box(self.x0 - n, self.y0 - n, self.x1 + n, self.y1 + n)
        return b.clamp(*clamp_to) if clamp_to else b

    def clamp(self, width: int, height: int) -> "Box":
        return Box(max(0, self.x0), max(0, self.y0), min(width, self.x1), min(height, self.y1))

    def intersect(self, other: "Box") -> Optional["Box"]:
        x0, y0, x1, y1 = max(self.x0, other.x0), max(self.y0, other.y0), min(self.x1, other.x1), min(self.y1, other.y1)
        return Box(x0, y0, x1, y1) if (x1 > x0 and y1 > y0) else None

    def overlaps(self, other: "Box") -> bool:
        return self.intersect(other) is not None

    def contains(self, x: float, y: float) -> bool:
        return self.x0 <= x < self.x1 and self.y0 <= y < self.y1

    def inside(self, width: int, height: int) -> bool:
        return self.x0 >= 0 and self.y0 >= 0 and self.x1 <= width and self.y1 <= height

    def at_downsample(self, ds: float) -> Tuple[int, int, int, int]:
        """The box at a coarser level: floor the origin, ceil the far corner (never loses a pixel)."""
        return (int(np.floor(self.x0 / ds)), int(np.floor(self.y0 / ds)), int(np.ceil(self.x1 / ds)), int(np.ceil(self.y1 / ds)))

    def to_dict(self) -> dict:
        return {"x0": self.x0, "y0": self.y0, "x1": self.x1, "y1": self.y1}

    @classmethod
    def from_points(cls, xy: np.ndarray) -> "Box":
        """Tight half-open box around pixel coordinates ``(N, 2)``."""
        xy = np.asarray(xy, dtype=np.float64)
        if xy.ndim != 2 or len(xy) == 0:
            raise ValueError("from_points needs an (N, 2) array with N >= 1")
        return cls(np.floor(xy[:, 0].min()), np.floor(xy[:, 1].min()), np.floor(xy[:, 0].max()) + 1, np.floor(xy[:, 1].max()) + 1)

    @classmethod
    def union_all(cls, boxes: Iterable["Box"]) -> "Box":
        bs = list(boxes)
        if not bs:
            raise ValueError("union_all of no boxes")
        return cls(min(b.x0 for b in bs), min(b.y0 for b in bs), max(b.x1 for b in bs), max(b.y1 for b in bs))
