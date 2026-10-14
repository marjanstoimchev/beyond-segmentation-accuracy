"""Lazy, pyramid-aware reader for a per-marker folder of pyramidal OME-TIFFs.

Every marker of a CellDIVE slide is its own tiled, pyramidal OME-TIFF. ``MxSlide`` keeps one lazy zarr
view per (marker, level) and reads only the tiles a window touches. On open it checks — not assumes —
the geometry: unreadable channels are excluded with a warning, the reference resolution grid is the
level-0 shape most channels share (a channel on another grid is excluded, never silently resampled),
pyramid downsamples are read from the level shapes, and an implausible pixel size is dropped.

Reads take a level-0 :class:`Box`, scale it to the level, clamp it to the image, and refuse an empty
result. The object is picklable (paths + metadata only; handles open lazily per process) so it works
inside DataLoader workers.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import tifffile
import zarr

from ..errors import SlideError
from .channels import ChannelInfo
from .geometry import Box

logger = logging.getLogger(__name__)
PLAUSIBLE_UM_PER_PX = (0.05, 5.0)


@dataclass
class ChannelGeometry:
    level_shapes: List[Tuple[int, int]]      # (H, W) per level, finest first
    dtype: str
    pixel_size_um: Optional[float]
    tile: Optional[Tuple[int, int]]
    compression: str
    warnings: List[str] = field(default_factory=list)


def _ome_pixel_size(xml: Optional[str]) -> Optional[float]:
    """PhysicalSizeX in microns. CellDIVE writes the unit attribute as ``PysicalSizeXUnit`` (sic); when no
    unit is found microns are assumed."""
    m = re.search(r'PhysicalSizeX="([0-9.eE+-]+)"', xml or "")
    if not m:
        return None
    size = float(m.group(1))
    unit = re.search(r'P[h]?ysicalSizeXUnit="([^"]+)"', xml)
    scale = {"nm": 1e-3, "mm": 1e3}.get(unit.group(1) if unit else "", 1.0)
    return size * scale


def read_geometry(path: Path) -> ChannelGeometry:
    """Shapes / pyramid / pixel size of one channel file, without decoding pixels."""
    warnings: List[str] = []
    with tifffile.TiffFile(path) as tf:
        s = tf.series[0]
        if s.ndim != 2:
            raise SlideError(f"{path.name}: expected a 2-D (Y, X) image series, got axes {s.axes!r}")
        shapes = [tuple(int(v) for v in lvl.shape[-2:]) for lvl in s.levels]
        if any(not (h1 < h0 and w1 < w0) for (h0, w0), (h1, w1) in zip(shapes, shapes[1:])):
            raise SlideError(f"{path.name}: pyramid levels are not strictly decreasing: {shapes}")
        pix = _ome_pixel_size(tf.ome_metadata)
        if pix is not None and not (PLAUSIBLE_UM_PER_PX[0] <= pix <= PLAUSIBLE_UM_PER_PX[1]):
            warnings.append(f"implausible pixel size {pix} in {path.name}; ignored")
            pix = None
        page = tf.pages[0]
        if not page.is_tiled:
            warnings.append(f"{path.name} is stripped, not tiled: region reads decode full-width strips")
        return ChannelGeometry(shapes, str(np.dtype(s.dtype)), pix, (page.tilelength, page.tilewidth) if page.is_tiled else None,
                               page.compression.name, warnings)


class MxSlide:
    """A multiplex slide = a stack of single-marker pyramidal images over the same tissue."""

    def __init__(self, channels: Sequence[ChannelInfo], name: Optional[str] = None):
        if not channels:
            raise SlideError("no channels given")
        self.name = name or channels[0].slide
        self.excluded: List[dict] = []
        self.warnings: List[str] = []
        geoms: Dict[str, ChannelGeometry] = {}
        for c in channels:
            try:
                geoms[c.marker] = read_geometry(c.path)
            except Exception as exc:                       # unreadable: skip with a warning, never crash the slide
                self.excluded.append({"marker": c.marker, "file": c.path.name, "reason": f"unreadable: {exc}"})
                logger.warning("channel %s excluded: %s", c.marker, exc)
        if not geoms:
            raise SlideError(f"{self.name}: no readable channel", hint=str(self.excluded))
        shapes = [g.level_shapes[0] for g in geoms.values()]
        ref_shape = max(set(shapes), key=shapes.count)     # reference grid chosen numerically
        self.channels: List[ChannelInfo] = []
        for c in channels:
            g = geoms.get(c.marker)
            if g is None:
                continue
            if g.level_shapes[0] != ref_shape:
                self.excluded.append({"marker": c.marker, "file": c.path.name, "reason": f"resolution grid {g.level_shapes[0]} != reference {ref_shape}"})
                logger.warning("channel %s excluded: grid %s != reference %s", c.marker, g.level_shapes[0], ref_shape)
                continue
            self.channels.append(c)
            self.warnings.extend(g.warnings)
        ref = geoms[self.channels[0].marker]
        self.level_shapes = list(ref.level_shapes)
        self.dtype = np.dtype(ref.dtype)
        self.tile, self.compression = ref.tile, ref.compression
        sizes = [geoms[c.marker].pixel_size_um for c in self.channels if geoms[c.marker].pixel_size_um]
        self.pixel_size_um: Optional[float] = sizes[0] if sizes else None
        if self.pixel_size_um is None:
            self.warnings.append("no plausible physical pixel size in any channel; micron measures unavailable")
        self._files: Dict[str, tifffile.TiffFile] = {}
        self._arrays: Dict[Tuple[str, int], object] = {}

    @classmethod
    def from_paths(cls, paths, markers: Optional[Sequence[str]] = None) -> "MxSlide":
        return cls(paths.channels if markers is None else [paths.channel(m) for m in markers], name=paths.slide)

    def __getstate__(self):                                 # handles never cross a process boundary
        return {**self.__dict__, "_files": {}, "_arrays": {}}

    # ---- metadata --------------------------------------------------------------
    @property
    def markers(self) -> List[str]:
        return [c.marker for c in self.channels]

    @property
    def height(self) -> int:
        return self.level_shapes[0][0]

    @property
    def width(self) -> int:
        return self.level_shapes[0][1]

    @property
    def n_levels(self) -> int:
        return len(self.level_shapes)

    @property
    def downsamples(self) -> List[float]:
        """Read from the level shapes (vendors are not strict about powers of two)."""
        return [self.width / float(w) for _, w in self.level_shapes]

    @property
    def full_box(self) -> Box:
        return Box(0, 0, self.width, self.height)

    def has_marker(self, marker: str) -> bool:
        return any(c.marker.lower() == marker.lower() for c in self.channels)

    def level_for_max_dim(self, max_dim: int) -> int:
        """Finest level whose longest side fits in ``max_dim`` (coarsest level if none does)."""
        return next((i for i, (h, w) in enumerate(self.level_shapes) if max(h, w) <= max_dim), self.n_levels - 1)

    def info(self) -> dict:
        return {"name": self.name, "markers": self.markers, "width": self.width, "height": self.height, "n_levels": self.n_levels,
                "level_shapes": [list(s) for s in self.level_shapes], "downsamples": [round(d, 4) for d in self.downsamples],
                "dtype": str(self.dtype), "pixel_size_um": self.pixel_size_um, "tile": list(self.tile) if self.tile else None,
                "compression": self.compression, "excluded_channels": self.excluded, "warnings": self.warnings}

    def summary(self) -> str:
        um = f"{self.pixel_size_um:.4f} um/px" if self.pixel_size_um else "unknown px size"
        s = (f"{self.name}: {self.width} x {self.height} px, {um}, {self.dtype}, {self.n_levels} pyramid levels "
             f"(downsamples {[round(d, 2) for d in self.downsamples]}), tile {self.tile}, {self.compression}\n  markers: {', '.join(self.markers)}")
        return s + (f"\n  excluded channels: {self.excluded}" if self.excluded else "")

    # ---- lazy IO -----------------------------------------------------------------
    def _array(self, marker: str, level: int):
        key = (marker, level)
        if key not in self._arrays:
            ci = next((c for c in self.channels if c.marker.lower() == marker.lower()), None)
            if ci is None:
                raise SlideError(f"marker {marker!r} is not a channel of {self.name}", hint=f"available: {self.markers}")
            if marker not in self._files:
                self._files[marker] = tifffile.TiffFile(ci.path)
            self._arrays[key] = zarr.open(self._files[marker].series[0].aszarr(level=level), mode="r")
        return self._arrays[key]

    def read_region(self, box: Box, markers: Optional[Sequence[str]] = None, level: int = 0) -> np.ndarray:
        """Read ``box`` (level-0 px) at pyramid ``level`` for ``markers`` -> ``(C, H, W)`` native dtype."""
        markers = list(markers) if markers else self.markers
        if not (0 <= level < self.n_levels):
            raise SlideError(f"level {level} out of range for {self.name} ({self.n_levels} levels)")
        lh, lw = self.level_shapes[level]
        x0, y0, x1, y1 = box.at_downsample(self.downsamples[level])
        x0, y0, x1, y1 = max(0, x0), max(0, y0), min(lw, x1), min(lh, y1)
        if x1 <= x0 or y1 <= y0:
            raise SlideError(f"box {box.xyxy} lies outside {self.name} ({self.width}x{self.height}) at level {level}")
        return np.stack([np.asarray(self._array(m, level)[y0:y1, x0:x1]) for m in markers])

    def read_channel(self, box: Box, marker: str = "DAPI", level: int = 0) -> np.ndarray:
        return self.read_region(box, [marker], level)[0]

    def thumbnail(self, max_dim: int = 1024, markers: Optional[Sequence[str]] = None) -> Tuple[np.ndarray, float]:
        """Finest level whose longest side fits ``max_dim`` -> ``((C, h, w), downsample)``."""
        lvl = self.level_for_max_dim(max_dim)
        return self.read_region(self.full_box, markers, level=lvl), self.downsamples[lvl]

    def close(self) -> None:
        for f in self._files.values():
            f.close()
        self._files.clear()
        self._arrays.clear()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def __repr__(self):
        return f"MxSlide({self.name!r}, {self.width}x{self.height}, levels={self.n_levels}, markers={self.markers})"
