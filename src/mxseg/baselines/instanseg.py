"""Pretrained InstanSeg (``fluorescence_nuclei_and_cells``): a light channel-invariant embedding U-Net that
segments nuclei and whole cells in one pass and rescales by the slide's pixel size."""
from __future__ import annotations

from typing import List, Optional, Sequence

import numpy as np

from .base import Baseline


def _device() -> str:
    import torch
    return "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")


class InstanSegBaseline(Baseline):
    name = "instanseg"

    def __init__(self, model: str = "fluorescence_nuclei_and_cells", target: str = "nuclei", pixel_size_um: Optional[float] = None):
        if target not in ("nuclei", "cells"):
            raise ValueError("target must be 'nuclei' or 'cells'")
        self.model_name, self.target, self.pixel_size_um = model, target, pixel_size_um
        self._model = None

    def params(self) -> dict:
        return {"model": self.model_name, "target": self.target}

    def input_markers(self):
        """Zero-shot on the two channels the reference boundaries were drawn from (InstanSeg takes any channel count)."""
        return super().input_markers() or ["DAPI", "LAP2"]

    def model(self):
        if self._model is None:
            from instanseg import InstanSeg
            self._model = InstanSeg(self.model_name, device=_device(), verbosity=0)
        return self._model

    @property
    def device(self) -> str:
        return _device()

    def predict_batch(self, images: Sequence[np.ndarray], pixel_size_um: Optional[float] = None, slides=None) -> List[np.ndarray]:
        import torch
        px = self.pixel_size_um or pixel_size_um or 0.5
        out = []
        for im in images:
            lab = self.model().eval_small_image(torch.from_numpy(np.ascontiguousarray(im)).float(), pixel_size=px, return_image_tensor=False)
            lab = np.asarray(lab.cpu()).squeeze()
            if lab.ndim == 3:                                   # (2, H, W) = [nuclei, cells]
                lab = lab[0 if self.target == "nuclei" else 1]
            out.append(lab.astype(np.int32))
        return out
