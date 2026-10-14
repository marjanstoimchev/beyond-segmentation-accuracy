"""Parse CellDIVE per-marker file names into channel metadata.

A CellDIVE export names each single-marker OME-TIFF like::

    SLIDE-0235_9.0.4_R000_FITC_PanCK-C11-AF488_FINAL_AFR_F.ome.tif
    SLIDE-0235_1.0.4_R000_DAPI__FINAL_F.ome.tif
    ^slide     ^round   ^rep ^fluor ^marker-clone-dye  ^AFR = autofluorescence removed

The canonical marker is the first ``-`` token of the marker field (``PanCK``, ``CD45``); the DAPI
file has an empty marker field and is canonicalised to ``DAPI``. Parsing is a pure function and
returns ``None`` for names outside the convention (they are reported, not silently used).
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

_NAME_RE = re.compile(
    r"^(?P<slide>SLIDE-\d+)_(?P<round>\d+)\.(?P<sub_a>\d+)\.(?P<sub_b>\d+)_R(?P<rep>\d+)_"
    r"(?P<fluor>DAPI|FITC|Cy3|Cy5|Cy7)_(?P<marker>[^_]*)_FINAL(?P<afr>_AFR)?_F\.ome\.tiff?$",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class ChannelInfo:
    """One single-marker image file of a slide."""

    path: Path
    slide: str
    marker: str          # canonical marker name, e.g. "DAPI", "PanCK", "CD45"
    raw_marker: str      # the full marker field, e.g. "PanCK-C11-AF488"
    fluor: str           # DAPI / FITC / Cy3 / Cy5 / Cy7
    round: int           # staining round
    af_removed: bool     # "_AFR_" in the name: autofluorescence-corrected image

    @property
    def is_dapi(self) -> bool:
        return self.marker.upper() == "DAPI"

    @property
    def clone(self) -> Optional[str]:
        parts = self.raw_marker.split("-")
        return parts[1] if len(parts) > 1 else None

    @property
    def dye(self) -> Optional[str]:
        parts = self.raw_marker.split("-")
        return parts[-1] if len(parts) > 2 else None

    def to_dict(self) -> dict:
        return {"marker": self.marker, "fluor": self.fluor, "round": self.round, "clone": self.clone,
                "dye": self.dye, "af_removed": self.af_removed, "file": self.path.name}


def parse_channel_filename(path: "str | Path") -> Optional[ChannelInfo]:
    """Parse a CellDIVE file name; ``None`` if it does not follow the convention."""
    p = Path(path)
    m = _NAME_RE.match(p.name)
    if not m:
        return None
    raw, fluor = m.group("marker"), m.group("fluor")
    marker = "DAPI" if (fluor.upper() == "DAPI" or raw == "") else raw.split("-")[0]
    return ChannelInfo(path=p, slide=m.group("slide"), marker=marker, raw_marker=raw or "DAPI", fluor=fluor,
                       round=int(m.group("round")), af_removed=m.group("afr") is not None)
