"""What every network sees — one input for pretraining, training, validation and ``mxseg eval``.

The ``input:`` block of the train and baseline YAMLs assigns one marker of the panel to each of the three channels a
pretrained encoder expects (``red``, ``green``, ``blue``; ``null`` = blank), and the dataset's stored slide-level statistics
scale each marker to [0, 1] (``<dataset>/<slide>/stats.json`` — the single normalization every path uses). The segmenters see
``red: LAP2, green: null, blue: DAPI``: the two channels the reference boundaries were drawn from and nothing else. KODIAK
pretraining fixes red and blue the same way and draws green at random per sample (``kodiak.data``). The resolved input —
channels plus per-slide statistics — is stored with a model's weights, so evaluation reproduces the training input exactly.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from mxdata.data.composite import Channels, render_channels, select_markers
from mxdata.errors import ConfigError

Norms = Dict[str, Dict[str, Tuple[float, float]]]          # slide -> marker -> (lo, hi)


@dataclass(frozen=True)
class InputConfig:
    red: Optional[str] = "LAP2"        # nuclear envelope: with DAPI, the channel the reference boundaries were drawn from
    green: Optional[str] = None        # blank for the segmenters; KODIAK pretraining draws a marker here per sample
    blue: Optional[str] = "DAPI"       # nuclei

    def __post_init__(self):
        if not any(self.channels):
            raise ConfigError("input: at least one of red / green / blue must name a marker")

    @property
    def channels(self) -> Channels:
        return (self.red, self.green, self.blue)

    @property
    def markers(self) -> List[str]:
        """The markers read per image, in channel order, each once."""
        return list(dict.fromkeys(m for m in self.channels if m is not None))


@dataclass(frozen=True)
class InputSpec:
    """The input, fully determined: the channel assignment and the slide-level (lo, hi) each marker is scaled with, per slide."""
    channels: Channels
    norms: Norms

    @property
    def markers(self) -> List[str]:
        return list(dict.fromkeys(m for m in self.channels if m is not None))

    @property
    def n_channels(self) -> int:
        return 3

    def norms_for(self, slide: Optional[str]) -> Dict[str, Tuple[float, float]]:
        if slide in self.norms:
            return self.norms[slide]
        if slide is None and len(self.norms) == 1:                        # one slide: nothing to disambiguate
            return next(iter(self.norms.values()))
        raise ValueError(f"no input statistics for slide {slide!r}; the input knows {sorted(self.norms)}")

    def render(self, stack: np.ndarray, slide: Optional[str] = None) -> np.ndarray:
        """``(len(markers), H, W)`` native stack in ``markers`` order -> the ``(3, H, W)`` network input, float32 in [0, 1]."""
        if stack.shape[0] != len(self.markers):
            raise ValueError(f"input expects {self.markers} ({len(self.markers)} channels), got {stack.shape[0]}")
        return render_channels(stack, self.markers, self.norms_for(slide), self.channels)

    def render_from(self, image: np.ndarray, image_markers: Sequence[str], slide: Optional[str] = None) -> np.ndarray:
        """The network input from a stored stack whose channels are ``image_markers`` (e.g. the dataset's nine)."""
        return render_channels(image, list(image_markers), self.norms_for(slide), self.channels)

    def to_meta(self) -> dict:
        return {"channels": list(self.channels),
                "norms": {s: {m: [float(v) for v in lohi] for m, lohi in n.items()} for s, n in self.norms.items()}}

    @classmethod
    def from_meta(cls, meta) -> Optional["InputSpec"]:
        if not meta or "channels" not in meta:                             # weights without a recipe: the raw stored patch
            return None
        return cls(tuple(meta["channels"]), {s: {m: tuple(v) for m, v in n.items()} for s, n in meta["norms"].items()})

    def describe(self) -> str:
        chan = " · ".join(f"{k} {m or '—'}" for k, m in zip(("R", "G", "B"), self.channels))
        stats = "; ".join(f"{s}: " + ", ".join(f"{m} {lo:.0f}-{hi:.0f}" for m, (lo, hi) in n.items()) for s, n in self.norms.items())
        return f"{chan} ({stats})"


def input_meta(spec: Optional[InputSpec]) -> dict:
    """What to store with a model's weights (and in run records): the spec, or an explicit raw (no spec: the dataset's stored
    channels, normalized per model — the zero-shot baselines)."""
    return spec.to_meta() if spec is not None else {"mode": "raw"}


class SlideReader:
    """Marker stacks read live from the slides at a box (whole-region evaluation); handles opened lazily, once per process
    (and not pickled into dataloader workers, which open their own)."""

    def __init__(self, slide_root=None):
        self.slide_root = slide_root
        self._mx: Dict[str, object] = {}

    def __getstate__(self):
        return {"slide_root": self.slide_root, "_mx": {}}

    def slide(self, name: str):
        if name not in self._mx:
            from mxdata import ingest
            self._mx[name] = ingest(name, self.slide_root)[1]
        return self._mx[name]

    def stack(self, name: str, box, markers: Sequence[str], level: int = 0) -> np.ndarray:
        return self.slide(name).read_region(box, markers=list(markers), level=level)


def dataset_norms(dataset_dir, slides: Sequence[str], markers: Sequence[str]) -> Norms:
    """The stored slide-level statistics of ``markers`` for ``slides`` (``<dataset>/<slide>/stats.json``)."""
    from mxdata.pipeline.stats import NormStats
    norms: Norms = {}
    for s in slides:
        f = Path(dataset_dir) / s / "stats.json"
        if not f.is_file():
            raise ConfigError(f"no normalization statistics for {s} at {f}", hint="run `mxdata sample -c configs/dataset.yaml`")
        st = NormStats.load(f)
        if missing := [m for m in markers if m not in st.lo]:
            raise ConfigError(f"the dataset stores no statistics for marker(s) {missing} of {s}; it has {st.markers}",
                              hint="add the marker to configs/dataset.yaml `markers` and re-run `mxdata sample`")
        norms[s] = {m: (float(st.lo[m]), float(st.hi[m])) for m in markers}
    return norms


def resolve_input(cfg: InputConfig, dataset_dir, slides: Sequence[str]) -> InputSpec:
    """The input recipe with the dataset's stored statistics for its markers — the same numbers the pretraining stream uses."""
    return InputSpec(cfg.channels, dataset_norms(dataset_dir, slides, cfg.markers))


__all__ = ["InputConfig", "InputSpec", "SlideReader", "dataset_norms", "input_meta", "resolve_input", "select_markers"]
