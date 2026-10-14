"""Shared stage plumbing: ingest + validate + plan one slide, normalization stats on demand, atomic JSON."""
from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional, Tuple

from ..annotations.regions import RegionAnnotation, load_regions
from ..config import DataConfig
from ..errors import ValidationError
from ..io.paths import SlidePaths
from ..io.slide import MxSlide
from .ingest import ingest
from .plan import RegionUnit, plan_units
from .stats import NormStats, compute_norm_stats
from .validate import validate


def write_json(path: Path, obj) -> None:
    """Temp + rename, so a reader never sees a half-written file."""
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, default=str))
    tmp.replace(path)


def prepare(cfg: DataConfig, slide_name: Optional[str] = None, root=None
            ) -> Tuple[SlidePaths, MxSlide, List[RegionAnnotation], dict, dict, List[RegionUnit]]:
    """Ingest + validate + plan one slide (metadata only, no pixel reads). Raises on validation errors."""
    slide_name = slide_name or cfg.slide
    paths, slide, ingest_report = ingest(slide_name, root)
    regions = load_regions(paths, cfg.regions, cfg.min_area_px)
    report = validate(slide, regions, cfg)
    if not report.ok:
        raise ValidationError(f"{slide_name} failed validation ({report.n_errors} errors)", hint=report.summary("error"))
    return paths, slide, regions, ingest_report.to_dict(), report.to_dict(), plan_units(cfg, slide, regions)


def ensure_stats(cfg: DataConfig, slide: MxSlide, units: List[RegionUnit], paths: SlidePaths,
                 stats_path: Optional[Path]) -> Optional[NormStats]:
    """Current stats from ``stats_path`` if they match the inputs and settings, else computed (and saved)."""
    if cfg.normalization.mode != "slide":
        return None
    sig = paths.signature()
    if stats_path and stats_path.exists():
        existing = NormStats.load(stats_path)
        if existing.is_current(cfg.markers, cfg.normalization, cfg.level, sig):
            return existing
    stats = compute_norm_stats(slide, [u.box for u in units], cfg.markers, cfg.normalization, cfg.level, sig)
    if stats_path:
        stats.save(stats_path)
    return stats
