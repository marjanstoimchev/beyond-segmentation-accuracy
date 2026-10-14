"""Validate: refuse or flag anything that would silently corrupt downstream results.

Two kinds of quality (the manual's distinction): *engineering* quality — is the file what it claims,
do boxes fit the slide, do QC rows match contours — is decided here and can stop the run;
*pathology* validity — is a nucleus really wrong — is the biologist's call and travels as data
(``NucleusFlag``), never deleted.
"""
from __future__ import annotations

import logging
from dataclasses import asdict, dataclass, field
from typing import List, Optional, Sequence

from ..annotations.raster import NucleusFlag
from ..annotations.regions import RegionAnnotation, overlap_table, regions_table
from ..config import DataConfig
from ..io.slide import MxSlide

logger = logging.getLogger(__name__)


@dataclass
class Issue:
    severity: str                  # "error" | "warning" | "info"
    code: str
    message: str
    region: Optional[int] = None


@dataclass
class ValidationReport:
    slide: str
    issues: List[Issue] = field(default_factory=list)
    regions: List[dict] = field(default_factory=list)
    overlaps: List[dict] = field(default_factory=list)

    @property
    def n_errors(self) -> int:
        return sum(i.severity == "error" for i in self.issues)

    @property
    def n_warnings(self) -> int:
        return sum(i.severity == "warning" for i in self.issues)

    @property
    def ok(self) -> bool:
        return self.n_errors == 0

    def to_dict(self) -> dict:
        return {"slide": self.slide, "ok": self.ok, "n_errors": self.n_errors, "n_warnings": self.n_warnings,
                "issues": [asdict(i) for i in self.issues], "regions": self.regions, "overlaps": self.overlaps}

    def summary(self, min_severity: str = "info") -> str:
        rank = {"info": 0, "warning": 1, "error": 2}
        head = (f"validation of {self.slide}: {'OK' if self.ok else 'FAILED'} "
                f"({self.n_errors} errors, {self.n_warnings} warnings, {len(self.issues)} issues)")
        lines = [head] + [f"  [{i.severity:7s}] {i.code}" + (f" R{i.region}" if i.region is not None else "") + f": {i.message}"
                          for i in self.issues if rank[i.severity] >= rank[min_severity]]
        return "\n".join(lines)


def validate(slide: MxSlide, regions: Sequence[RegionAnnotation], cfg: Optional[DataConfig] = None) -> ValidationReport:
    rep = ValidationReport(slide=slide.name)
    add = rep.issues.append

    if cfg is not None:
        for m in cfg.markers:
            if not slide.has_marker(m):
                add(Issue("error", "MARKER_MISSING", f"configured marker {m!r} is not a channel (have {slide.markers})"))
        if cfg.level >= slide.n_levels:
            add(Issue("error", "LEVEL_OUT_OF_RANGE", f"level {cfg.level} but slide has {slide.n_levels} levels"))
    if slide.pixel_size_um is None:
        add(Issue("warning", "NO_PIXEL_SIZE", "physical pixel size unknown; micron measures unavailable"))
    for w in slide.warnings:
        add(Issue("warning", "SLIDE", w))
    for ex in slide.excluded:
        add(Issue("warning", "CHANNEL_EXCLUDED", f"{ex['marker']} ({ex['file']}): {ex['reason']}"))
    if not regions:
        add(Issue("error", "NO_REGIONS", "no annotated regions"))

    for r in regions:
        rid = r.region_id
        if len(r) == 0:
            add(Issue("error", "EMPTY_REGION", "no contours in the GeoJSON layer", rid))
            continue
        b = r.box
        if not b.inside(slide.width, slide.height):
            add(Issue("error", "BOX_OUTSIDE_SLIDE", f"FOV box {b.xyxy} exceeds slide {slide.width}x{slide.height}", rid))
        for msg in r.layer.issues:
            add(Issue("warning", "GEOJSON", msg, rid))
        if r.has_qc and r.qc_match:
            m = r.qc_match
            if m["unmatched_qc"]:
                add(Issue("error", "QC_ROWS_UNMATCHED",
                          f"{m['unmatched_qc']} QC row(s) match no contour (bbox tol 1 px) — QC table and layer disagree", rid))
            if m["ambiguous"]:
                add(Issue("warning", "QC_AMBIGUOUS", f"{m['ambiguous']} QC row(s) competed for one contour", rid))
            n_norow = r.count(NucleusFlag.NO_QC_ROW)
            if n_norow:
                add(Issue("info", "CONTOURS_WITHOUT_QC_ROW", f"{n_norow} contour(s) have no QC row", rid))
            add(Issue("info", "QC_WRONG", f"{r.n_wrong} of {len(r)} nuclei flagged Wrong ({100 * r.n_wrong / len(r):.1f}%)", rid))
        n_tiny = r.count(NucleusFlag.TINY)
        if n_tiny:
            add(Issue("info", "TINY_CONTOURS", f"{n_tiny} contour(s) below min_area_px={r.min_area_px} flagged TINY (excluded from labels)", rid))
        seen, dups = set(), 0
        for ring in r.rings:                                   # exact duplicate contours inside one layer
            k = ring.tobytes()
            dups += k in seen
            seen.add(k)
        if dups:
            add(Issue("warning", "DUPLICATE_CONTOURS", f"{dups} exactly duplicated contour(s)", rid))

    if regions and all(len(r) for r in regions):
        rep.regions = regions_table(regions, slide.pixel_size_um).reset_index().to_dict("records")
        rep.overlaps = overlap_table(regions).to_dict("records")
        for row in rep.overlaps:
            add(Issue("info", "FOV_OVERLAP",
                      f"R{row['region_a']} and R{row['region_b']} overlap {row['overlap_w']}x{row['overlap_h']} px "
                      f"({100 * row['overlap_frac_a']:.0f}% of FOV), {row['coincident']} coincident nuclei — keep them in one split"))
    logger.info(rep.summary().splitlines()[0])
    return rep
