"""Ingest: discover the slide's files, open the channels lazily, read + check metadata."""
from __future__ import annotations

import logging
from dataclasses import asdict, dataclass, field
from typing import List, Optional, Tuple

from ..io.paths import SlidePaths
from ..io.slide import MxSlide

logger = logging.getLogger(__name__)


@dataclass
class IngestReport:
    slide: str
    markers: List[str]
    excluded_channels: List[dict]
    width: int
    height: int
    n_levels: int
    downsamples: List[float]
    dtype: str
    pixel_size_um: Optional[float]
    tile: Optional[List[int]]
    compression: str
    halo_regions: List[int]
    qc_regions: List[int]
    n_screenshots: int
    unparsed_tiffs: List[str]
    source_signature: str
    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)

    def summary(self) -> str:
        um = f"{self.pixel_size_um:.4f} um/px" if self.pixel_size_um else "pixel size unknown"
        s = [f"{self.slide}: {self.width} x {self.height} px, {um}, {self.dtype}, {self.n_levels} levels "
             f"(downsamples {[round(d, 2) for d in self.downsamples]}), tile {self.tile}, {self.compression}",
             f"  markers ({len(self.markers)}): {', '.join(self.markers)}",
             f"  HALO regions ({len(self.halo_regions)}): {self.halo_regions}",
             f"  QC regions   ({len(self.qc_regions)}): {self.qc_regions}"]
        if self.excluded_channels:
            s.append(f"  EXCLUDED channels: {self.excluded_channels}")
        s.extend(f"  warning: {w}" for w in self.warnings)
        return "\n".join(s)


def ingest(slide: "str | SlidePaths" = "SLIDE-0235", root=None) -> Tuple[SlidePaths, MxSlide, IngestReport]:
    paths = slide if isinstance(slide, SlidePaths) else SlidePaths.discover(slide, root)
    mx = MxSlide.from_paths(paths)
    info = mx.info()
    report = IngestReport(paths.slide, mx.markers, mx.excluded, mx.width, mx.height, mx.n_levels, info["downsamples"], str(mx.dtype),
                          mx.pixel_size_um, info["tile"], mx.compression, paths.region_ids, sorted(paths.qc_files),
                          len(paths.ss_images), [p.name for p in paths.unparsed_tiffs], paths.signature(), list(mx.warnings))
    logger.info("ingested %s: %dx%d px, %d levels, %d markers", paths.slide, mx.width, mx.height, mx.n_levels, len(mx.markers))
    return paths, mx, report
