"""Typed, strict experiment configuration.

Experiment config lives in a versioned YAML and is loaded into these dataclasses; **unknown keys are
rejected** so a typo fails instead of silently taking the default. Environment config (data root, output
root) comes from ``MXDATA_ROOT`` / CLI flags, never from the YAML. The fully resolved config (defaults
included) is saved next to every dataset the pipeline writes.
"""
from __future__ import annotations

import dataclasses
import typing
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import yaml

from .errors import ConfigError

QC_POLICIES = ("keep_all", "ignore_wrong", "drop_wrong")
NORM_MODES = ("slide", "region", "none")
MODES = ("supervised", "self_supervised")


def _check(cond: bool, msg: str) -> None:
    if not cond:
        raise ConfigError(msg)


@dataclass(frozen=True)
class PatchConfig:
    """The fixed segmentation dataset: patches sampled once, tissue-gated, with provenance."""

    size: int = 256                  # patch edge in pixels at the read level
    stride: int = 192                # labeled grid stride; overlap = size - stride (> a nucleus: every border nucleus is whole somewhere)
    min_fg: float = 0.02             # labeled: keep a window if >= this fraction of its pixels are nuclei (the labels say what is there)
    unlabeled: bool = True           # also sample tissue outside the annotated FOVs (image only, for self-supervision)
    unlabeled_stride: Optional[int] = None   # None = size (no overlap: nothing to stitch)
    min_tissue: float = 0.5          # unlabeled: keep a window if >= this fraction is tissue (Otsu mask on a thumbnail)
    tissue_max_dim: int = 2048       # longest side of the thumbnail the tissue mask is computed on
    tissue_marker: str = "DAPI"
    max_unlabeled_per_slide: Optional[int] = None   # seeded cap; None = every tissue window
    rows_per_unit: int = 8           # whole-slide grid rows per resumable unit (one band read each)
    seed: int = 0

    def __post_init__(self):
        _check(self.size > 0, "patches.size must be > 0")
        _check(0 < self.stride <= self.size, f"patches.stride must be in (0, size], got {self.stride}")
        _check(0.0 <= self.min_tissue <= 1.0, "patches.min_tissue must be in [0, 1]")
        _check(0.0 <= self.min_fg < 1.0, "patches.min_fg must be in [0, 1)")
        _check(self.tissue_max_dim >= 64, "patches.tissue_max_dim must be >= 64")
        _check(self.unlabeled_stride is None or 0 < self.unlabeled_stride <= self.size, "patches.unlabeled_stride must be in (0, size]")
        _check(self.max_unlabeled_per_slide is None or self.max_unlabeled_per_slide > 0, "patches.max_unlabeled_per_slide must be > 0")
        _check(self.rows_per_unit > 0, "patches.rows_per_unit must be > 0")

    @property
    def overlap(self) -> int:
        return self.size - self.stride

    @property
    def effective_unlabeled_stride(self) -> int:
        return self.unlabeled_stride or self.size


@dataclass(frozen=True)
class NormalizationConfig:
    mode: str = "slide"              # slide: one (lo, hi) per marker per slide | region: percentiles of the array | none: / dtype max
    pmin: float = 1.0
    pmax: float = 99.5
    sample_pixels: int = 2_000_000   # pixels sampled per marker for slide-level statistics
    seed: int = 0

    def __post_init__(self):
        _check(self.mode in NORM_MODES, f"normalization.mode must be one of {NORM_MODES}, got {self.mode!r}")
        _check(0.0 <= self.pmin < self.pmax <= 100.0, "normalization: need 0 <= pmin < pmax <= 100")
        _check(self.sample_pixels > 0, "normalization.sample_pixels must be > 0")


@dataclass(frozen=True)
class SplitConfig:
    """Train / val / test split over regions that never separates overlapping FOVs.

    ``val_slides`` pins whole slides (leave-slide-out); ``val_regions`` pins region ids — a list applies to every slide, a
    mapping ``{slide: [ids]}`` pins per slide; otherwise overlap groups are drawn with ``seed`` until ``val_frac`` of the
    nuclei are in val. The test set is only ever pinned (``test_regions`` / ``test_slides``): it is reported once and never
    tuned on, so it is never drawn at random.
    """

    val_frac: float = 0.25
    seed: int = 0
    val_regions: Optional[Union[Tuple[int, ...], Dict[str, Tuple[int, ...]]]] = None
    val_slides: Optional[Tuple[str, ...]] = None
    test_regions: Optional[Union[Tuple[int, ...], Dict[str, Tuple[int, ...]]]] = None
    test_slides: Optional[Tuple[str, ...]] = None
    require_qc: bool = False

    def regions_for(self, what: str, slide: str) -> Optional[set]:
        """The pinned region ids of ``what`` ("val" | "test") on ``slide``; None when nothing is pinned."""
        v = getattr(self, f"{what}_regions")
        if v is None:
            return None
        return set(v.get(slide, ())) if isinstance(v, dict) else set(v)

    def __post_init__(self):
        _check(0.0 <= self.val_frac < 1.0, "split.val_frac must be in [0, 1)")
        slides = set()
        for v in (self.val_regions, self.test_regions):
            slides |= set(v) if isinstance(v, dict) else {"*"}
        for s in slides:
            both = (self.regions_for("val", s) or set()) & (self.regions_for("test", s) or set())
            _check(not both, f"split: region(s) {sorted(both)} cannot be in both val_regions and test_regions")
        _check(not (set(self.val_slides or ()) & set(self.test_slides or ())), "split: a slide cannot be in both val_slides and test_slides")


@dataclass(frozen=True)
class AugmentConfig:
    flip: bool = True
    rot90: bool = True


@dataclass(frozen=True)
class LoaderConfig:
    batch_size: int = 8
    num_workers: int = 0
    pin_memory: bool = False
    seed: int = 0

    def __post_init__(self):
        _check(self.batch_size > 0 and self.num_workers >= 0, "loader: batch_size > 0 and num_workers >= 0")


@dataclass(frozen=True)
class DataConfig:
    slides: Tuple[str, ...] = ("SLIDE-0235",)
    markers: Tuple[str, ...] = ("DAPI",)
    level: int = 0
    regions: Optional[Tuple[int, ...]] = None    # None = every HALO region layer
    mode: str = "supervised"                     # supervised: labeled patches with labels | self_supervised: all patches, labels ignored
    qc_policy: str = "ignore_wrong"
    min_area_px: float = 20.0                    # contours below this area are TINY artefacts, never painted
    patches: PatchConfig = field(default_factory=PatchConfig)
    normalization: NormalizationConfig = field(default_factory=NormalizationConfig)
    split: SplitConfig = field(default_factory=SplitConfig)
    augment: AugmentConfig = field(default_factory=AugmentConfig)
    loader: LoaderConfig = field(default_factory=LoaderConfig)

    def __post_init__(self):
        _check(len(self.slides) > 0 and all(self.slides) and len(set(self.slides)) == len(self.slides), "slides must be a non-empty list of unique names")
        _check(len(self.markers) > 0 and len({m.lower() for m in self.markers}) == len(self.markers), "markers must be a non-empty list of unique names")
        _check(self.level >= 0, "level must be >= 0")
        _check(self.mode in MODES, f"mode must be one of {MODES}, got {self.mode!r}")
        _check(self.qc_policy in QC_POLICIES, f"qc_policy must be one of {QC_POLICIES}, got {self.qc_policy!r}")
        _check(self.min_area_px >= 0, "min_area_px must be >= 0")

    @property
    def slide(self) -> str:
        """The first (or only) slide."""
        return self.slides[0]

    @property
    def supervised(self) -> bool:
        return self.mode == "supervised"

    def with_slides(self, *slides: str) -> "DataConfig":
        return dataclasses.replace(self, slides=tuple(slides))


# ---------------------------------------------------------------------------
# strict construction from YAML / dicts
# ---------------------------------------------------------------------------
def _coerce(value: Any, hint: Any, where: str) -> Any:
    origin, args = typing.get_origin(hint), typing.get_args(hint)
    if origin is typing.Union:                                  # Optional[X], or X | Y (first type that accepts the value)
        if value is None:
            return None
        options = [a for a in args if a is not type(None)]
        for opt in options:
            try:
                return _coerce(value, opt, where)
            except ConfigError:
                continue
        raise ConfigError(f"{where}: expected one of {[getattr(o, '__name__', str(o)) for o in options]}, got {value!r}")
    if origin is dict:
        if not isinstance(value, dict):
            raise ConfigError(f"{where}: expected a mapping, got {type(value).__name__}")
        return {str(k): _coerce(v, args[1], f"{where}[{k}]") for k, v in value.items()}
    if dataclasses.is_dataclass(hint):
        return _from_dict(hint, value, where)
    if origin in (tuple, list):
        if isinstance(value, (str, bytes)) or not isinstance(value, (list, tuple)):
            raise ConfigError(f"{where}: expected a list, got {type(value).__name__}")
        return tuple(_coerce(v, args[0], f"{where}[{i}]") for i, v in enumerate(value))
    if hint is float and isinstance(value, int) and not isinstance(value, bool):
        return float(value)
    if hint in (bool, int, float, str) and (not isinstance(value, hint) or (hint is not bool and isinstance(value, bool))):
        raise ConfigError(f"{where}: expected {hint.__name__}, got {value!r}")
    return value


def _from_dict(cls, d: Any, where: str):
    d = {} if d is None else d
    if not isinstance(d, dict):
        raise ConfigError(f"{where}: expected a mapping, got {type(d).__name__}")
    hints, names = typing.get_type_hints(cls), [f.name for f in dataclasses.fields(cls)]
    unknown = sorted(set(d) - set(names))
    if unknown:
        hint = f"allowed keys: {names}"
        if cls is DataConfig and "slide" in unknown:
            hint = "use `slides: [SLIDE-0235]` (a list) — " + hint
        raise ConfigError(f"{where}: unknown key(s) {unknown}", hint=hint)
    try:
        return cls(**{k: _coerce(v, hints[k], f"{where}.{k}") for k, v in d.items()})
    except ConfigError as e:
        raise ConfigError(f"{where}: {e.message}", hint=e.hint) from None


def config_from_dict(d: dict) -> DataConfig:
    return _from_dict(DataConfig, d, "config")


def load_config(path: "str | Path") -> DataConfig:
    p = Path(path)
    if not p.is_file():
        raise ConfigError(f"config file not found: {p}")
    with open(p) as f:
        return config_from_dict(yaml.safe_load(f) or {})


def to_dict(cfg) -> dict:
    """Plain dict (tuples -> lists): the fully resolved config, defaults included."""
    def conv(o):
        if dataclasses.is_dataclass(o):
            return {f.name: conv(getattr(o, f.name)) for f in dataclasses.fields(o)}
        if isinstance(o, dict):
            return {k: conv(v) for k, v in o.items()}
        return [conv(v) for v in o] if isinstance(o, (list, tuple)) else o
    return conv(cfg)


def save_resolved(cfg: DataConfig, path: "str | Path") -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("# Fully resolved mxdata config (defaults included), written by the run that used it.\n" + yaml.safe_dump(to_dict(cfg), sort_keys=False))
    return p
