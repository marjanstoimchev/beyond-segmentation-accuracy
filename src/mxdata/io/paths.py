"""Discover the files that make up one slide.

Expected layout, one folder per slide under ``MXDATA_ROOT``::

    <root>/SLIDE-0235/
        SLIDE-0235_<round>_R000_<fluor>_<marker>_FINAL[_AFR]_F.ome.tif   one per marker
        HALO_layers_0235/<prefix>SLIDE-0235_Region_<k>.geojson           nucleus contours per FOV (CellDIVE_… or R<k>_HALO_… exports)
        QC_for_slide-0235/*Region_<k>*.csv                               biologist QC (Flag column)
        SS For  SLIDE-0235/{*.html, images/*.png}                        HALO screenshots (reference)
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List

import pandas as pd

from ..errors import SlideError
from .channels import ChannelInfo, parse_channel_filename

DATA_ROOT = Path(os.environ.get("MXDATA_ROOT", "~/Desktop/MedAI/Data/mxdata")).expanduser()
DEFAULT_DATASET = DATA_ROOT / "dataset"                          # every marker, labeled + unlabeled: pretraining and analysis

_HALO_RE = re.compile(r"(SLIDE-\d+)_Region_(\d+)\.geojson$", re.IGNORECASE)       # any export prefix: CellDIVE_…, R1_HALO_…
_QC_RE = re.compile(r"Region_(\d+)(?!\d)", re.IGNORECASE)


@dataclass
class SlidePaths:
    """All files of one slide, resolved. Build with :meth:`discover`."""

    root: Path
    slide: str
    channels: List[ChannelInfo] = field(default_factory=list)
    halo_files: Dict[int, Path] = field(default_factory=dict)      # region id -> geojson
    qc_files: Dict[int, Path] = field(default_factory=dict)        # region id -> csv
    ss_images: List[Path] = field(default_factory=list)
    unparsed_tiffs: List[Path] = field(default_factory=list)

    @classmethod
    def discover(cls, slide: str = "SLIDE-0235", root: "str | Path | None" = None) -> "SlidePaths":
        sdir = (Path(root).expanduser() if root else DATA_ROOT) / slide
        if not sdir.is_dir():
            raise SlideError(f"slide folder not found: {sdir}", hint="set MXDATA_ROOT (or --root) to the folder that holds the SLIDE-* folders")
        sp = cls(root=sdir, slide=slide)
        for p in sorted(sdir.iterdir()):
            if p.is_file() and p.suffix.lower() in (".tif", ".tiff"):
                ci = parse_channel_filename(p)
                (sp.channels.append(ci) if ci else sp.unparsed_tiffs.append(p))
            elif p.is_dir() and p.name.lower().startswith("halo"):
                sp.halo_files = {int(m.group(2)): g for g in sorted(p.glob("*.geojson")) if (m := _HALO_RE.search(g.name))}
            elif p.is_dir() and p.name.lower().startswith("qc"):
                for c in sorted(p.glob("*.csv")):
                    if m := _QC_RE.search(c.name):
                        sp.qc_files.setdefault(int(m.group(1)), c)
            elif p.is_dir() and p.name.lower().startswith("ss"):
                sp.ss_images = sorted(x for x in p.rglob("*") if x.suffix.lower() in (".png", ".jpg", ".jpeg"))
        sp.channels.sort(key=lambda c: (not c.is_dapi, c.round, c.marker))
        if not sp.channels:
            raise SlideError(f"no CellDIVE channel files found in {sdir}", hint="expected names like SLIDE-0235_1.0.4_R000_DAPI__FINAL_F.ome.tif")
        return sp

    @property
    def markers(self) -> List[str]:
        return [c.marker for c in self.channels]

    @property
    def region_ids(self) -> List[int]:
        return sorted(self.halo_files)

    def channel(self, marker: str) -> ChannelInfo:
        for c in self.channels:
            if c.marker.lower() == marker.lower():
                return c
        raise SlideError(f"marker {marker!r} not found for {self.slide}", hint=f"available: {self.markers}")

    def signature(self) -> str:
        """A cheap content fingerprint (sizes + mtimes of every input) for cache invalidation."""
        files = sorted([c.path for c in self.channels] + list(self.halo_files.values()) + list(self.qc_files.values()))
        return f"{self.slide}|" + "|".join(f"{p.name}:{p.stat().st_size}:{int(p.stat().st_mtime)}" for p in files)

    def channels_table(self) -> pd.DataFrame:
        return pd.DataFrame([dict(c.to_dict(), size_MB=round(c.path.stat().st_size / 1e6, 1)) for c in self.channels])

    def summary(self) -> str:
        lines = [f"{self.slide} @ {self.root}",
                 f"  channels ({len(self.channels)}): {', '.join(self.markers)}",
                 f"  HALO regions ({len(self.halo_files)}): {self.region_ids}",
                 f"  QC regions   ({len(self.qc_files)}): {sorted(self.qc_files)}",
                 f"  screenshots: {len(self.ss_images)}"]
        if self.unparsed_tiffs:
            lines.append(f"  unparsed tiffs (ignored): {[p.name for p in self.unparsed_tiffs]}")
        return "\n".join(lines)
