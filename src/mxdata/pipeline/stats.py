"""Normalization statistics: one (lo, hi) percentile pair **per marker per slide** from a seeded pixel sample
of tissue windows (the annotated FOVs). Not per tile (tiles would become incomparable), not the whole slide
(background would set the range). Saved as JSON with provenance so every consumer normalizes identically.
"""
from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np

from ..config import NormalizationConfig
from ..errors import ValidationError
from ..io.geometry import Box
from ..io.slide import MxSlide

logger = logging.getLogger(__name__)


@dataclass
class NormStats:
    slide: str
    markers: List[str]
    lo: Dict[str, float]
    hi: Dict[str, float]
    pmin: float
    pmax: float
    level: int
    sample_pixels: int
    seed: int
    n_windows: int
    source_signature: str = ""

    def arrays(self, markers: Sequence[str]) -> Tuple[np.ndarray, np.ndarray]:
        if missing := [m for m in markers if m not in self.lo]:
            raise ValidationError(f"normalization stats have no entry for {missing}", hint=f"stats cover {self.markers}")
        return np.array([self.lo[m] for m in markers], np.float32), np.array([self.hi[m] for m in markers], np.float32)

    def save(self, path: "str | Path") -> Path:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(asdict(self), indent=2))
        tmp.replace(p)
        return p

    @classmethod
    def load(cls, path: "str | Path") -> "NormStats":
        d = json.loads(Path(path).read_text())
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})     # ignore keys of older writers

    def is_current(self, markers: Sequence[str], cfg: NormalizationConfig, level: int, source_signature: str) -> bool:
        return (all(m in self.lo for m in markers) and (self.pmin, self.pmax, self.seed, self.sample_pixels, self.level)
                == (cfg.pmin, cfg.pmax, cfg.seed, cfg.sample_pixels, level) and self.source_signature == source_signature)


def compute_norm_stats(slide: MxSlide, boxes: Sequence[Box], markers: Sequence[str], cfg: NormalizationConfig,
                       level: int = 0, source_signature: str = "") -> NormStats:
    """Percentiles of a seeded pixel sample drawn evenly from ``boxes``, per marker."""
    if not boxes:
        raise ValidationError("compute_norm_stats needs at least one window")
    rng = np.random.default_rng(cfg.seed)
    per_box = max(1, cfg.sample_pixels // len(boxes))
    samples: Dict[str, List[np.ndarray]] = {m: [] for m in markers}
    for b in boxes:
        stack = slide.read_region(b, markers, level)
        n = stack.shape[1] * stack.shape[2]
        idx = rng.choice(n, size=per_box, replace=False) if per_box < n else np.arange(n)
        for c, m in enumerate(markers):
            samples[m].append(stack[c].reshape(-1)[idx])
    lo, hi = {}, {}
    for m in markers:
        lo[m], hi[m] = (float(x) for x in np.percentile(np.concatenate(samples[m]), [cfg.pmin, cfg.pmax]))
        if hi[m] <= lo[m]:
            hi[m] = lo[m] + 1.0
            logger.warning("marker %s: flat intensity sample; hi set to lo + 1", m)
    logger.info("normalization stats from %d windows: %s", len(boxes), {m: (round(lo[m]), round(hi[m])) for m in markers})
    return NormStats(slide.name, list(markers), lo, hi, cfg.pmin, cfg.pmax, level, cfg.sample_pixels, cfg.seed, len(boxes), source_signature)
