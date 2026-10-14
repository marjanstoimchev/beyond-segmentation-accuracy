"""Pretrained Cellpose (4.x): ``cpsam_v2`` (Cellpose-SAM, low-contrast retrain — the default), ``cpsam``, ``cpdino``."""
from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional, Sequence

import numpy as np

from .base import Baseline


class CellposeBaseline(Baseline):
    name = "cellpose"

    def __init__(self, variant: str = "cpsam_v2", diameter: Optional[float] = None, flow_threshold: float = 0.4,
                 cellprob_threshold: float = 0.0, gpu: bool = True, batch_size: int = 8):
        self.variant, self.diameter, self.flow_threshold, self.cellprob_threshold = variant, diameter, flow_threshold, cellprob_threshold
        self.gpu, self.batch_size = gpu, batch_size
        self._model = None

    def params(self) -> dict:
        return {"variant": self.variant, "diameter": self.diameter, "flow_threshold": self.flow_threshold, "cellprob_threshold": self.cellprob_threshold}

    def model(self):
        if self._model is None:
            from cellpose import models
            if ("/" in self.variant or self.variant.endswith(".pt")) and not Path(self.variant).is_file():
                raise FileNotFoundError(f"cellpose weights not found: {self.variant} (cellpose would silently fall back to its default model)")
            self._model = models.CellposeModel(gpu=self.gpu, pretrained_model=self.variant)
            side = Path(self.variant).with_suffix(".input.json")          # written by mxseg train next to fine-tuned weights
            if Path(self.variant).is_file() and side.is_file():
                self._input_from_meta({"input": json.loads(side.read_text())})
        return self._model

    def input_markers(self):
        self.model()                                                     # the weights carry the input recipe
        return super().input_markers() or ["DAPI"]                       # zero-shot: the nuclear channel alone, as (H, W)

    def resolve_input(self, dataset_dir, slides):
        self.model()                                                     # fine-tuned weights decide through their sidecar
        return super().resolve_input(dataset_dir, slides)

    @property
    def device(self) -> str:
        return str(getattr(self.model(), "device", "cpu"))

    def predict_batch(self, images: Sequence[np.ndarray], pixel_size_um: Optional[float] = None, slides=None) -> List[np.ndarray]:
        # cellpose takes (H, W) or (H, W, C<=3); it normalizes per image itself (percentiles 1-99)
        if self.input is not None:                                       # the marker image, as (H, W, C)
            slides = slides or [None] * len(images)
            x = [np.moveaxis(self.input.render(im, s), 0, -1) for im, s in zip(images, slides)]
        else:
            x = [im[0] if im.shape[0] == 1 else np.moveaxis(im[:3], 0, -1) for im in images]
        masks = self.model().eval(x, batch_size=self.batch_size, diameter=self.diameter, flow_threshold=self.flow_threshold,
                                  cellprob_threshold=self.cellprob_threshold, normalize=True)[0]
        return [np.asarray(m).astype(np.int32) for m in masks]
