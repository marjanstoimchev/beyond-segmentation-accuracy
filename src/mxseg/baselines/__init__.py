"""Registry: ``get_baseline("cellpose", variant="cpsam_v2")`` or ``from_config("configs/baselines/sam3.yaml")``.
Heavy imports happen inside each baseline."""
from __future__ import annotations

from typing import Dict, Type

from .base import Baseline
from .cellpose import CellposeBaseline
from .instanseg import InstanSegBaseline
from .kodiak import KodiakBaseline
from .sam2 import Sam2Baseline
from .sam3 import Sam3Baseline
from .unet import UNetBaseline

BASELINES: Dict[str, Type[Baseline]] = {"cellpose": CellposeBaseline, "instanseg": InstanSegBaseline, "unet": UNetBaseline,
                                        "kodiak": KodiakBaseline, "sam2": Sam2Baseline, "sam3": Sam3Baseline}


def register(name: str, cls: Type[Baseline]) -> None:
    BASELINES[name] = cls


def from_config(path, **overrides) -> "tuple[Baseline, str]":
    """(baseline, result name) from a ``configs/baselines/*.yaml`` — the file is the run's definition: strict keys
    ``baseline`` (registry name), ``params`` (constructor arguments), optional ``name`` (default: the file stem) and optional
    ``input`` (the model's input recipe, ``mxseg.inputs.InputConfig``; weights that carry their own recipe win).
    ``overrides`` (e.g. from ``--param``) replace individual params."""
    from pathlib import Path

    import yaml
    p = Path(path)
    d = yaml.safe_load(p.read_text()) or {}
    if unknown := set(d) - {"baseline", "name", "params", "input"}:
        raise KeyError(f"{p}: unknown key(s) {sorted(unknown)} — a baseline config has: baseline, params, name, input")
    if "baseline" not in d:
        raise KeyError(f"{p}: missing 'baseline' (one of {sorted(BASELINES)})")
    params = {**(d.get("params") or {}), **{k: v for k, v in overrides.items() if v is not None}}
    baseline = get_baseline(d["baseline"], **params)
    if d.get("input"):
        from ..inputs import InputConfig
        baseline.input_cfg = InputConfig(**d["input"])
    return baseline, str(d.get("name") or p.stem)


def get_baseline(name: str, **params) -> Baseline:
    if name not in BASELINES:
        raise KeyError(f"unknown baseline {name!r}; available: {sorted(BASELINES)}")
    return BASELINES[name](**params)


__all__ = ["Baseline", "BASELINES", "get_baseline", "from_config", "register", "CellposeBaseline", "InstanSegBaseline",
           "UNetBaseline", "KodiakBaseline", "Sam2Baseline", "Sam3Baseline"]
