"""One interface for every segmenter: ``predict(image) -> instance label map``."""
from __future__ import annotations

import logging
from typing import List, Optional, Sequence

import numpy as np

logger = logging.getLogger(__name__)


class Baseline:
    """Subclass and implement :meth:`predict` (or :meth:`predict_batch`). ``image`` is ``(C, H, W)`` native dtype
    (uint16), channels in the dataset's marker order; the result is an ``int32`` label map, 0 = background."""

    name: str = "baseline"
    input = None            # InputSpec of a trained model: taken from its weights, else resolved from a config's input: block
    input_cfg = None        # the input: block of a baseline config (InputConfig), resolved on the dataset before predicting
    input_fixed = False     # True once the weights decided (absent from old weights = raw); the config block is then ignored

    def input_markers(self) -> Optional[List[str]]:
        """The markers this baseline needs per image, in order (None = the dataset's stored markers)."""
        return list(self.input.markers) if self.input is not None else None

    def resolve_input(self, dataset_dir, slides):
        """Fix the input recipe before predicting: the one stored with the weights wins (the training input, statistics
        included); else the config's ``input:`` block resolved on the dataset's stored statistics; else raw."""
        if self.input_fixed:
            if self.input is None and self.input_cfg is not None:
                logger.warning("%s: the weights were trained on the stored single-marker input; the config's input block is ignored", self.name)
            # a slide added after training carries no statistics in the weights. The protocol scales every marker by its
            # OWN slide's percentiles, so top the recipe up from that slide's stored stats.json; the trained slides keep
            # the exact values they were trained with.
            if self.input is not None and dataset_dir is not None:
                missing = [s for s in (slides or []) if s is not None and s not in self.input.norms]
                if missing:
                    from ..inputs import InputSpec, dataset_norms
                    extra = dataset_norms(dataset_dir, missing, self.input.markers)
                    logger.warning("%s: weights carry no statistics for %s; scaling it by its own stored slide statistics",
                                   self.name, ", ".join(missing))
                    self.input = InputSpec(self.input.channels, {**self.input.norms, **extra})
            return self.input
        if self.input is None and self.input_cfg is not None:
            from ..inputs import resolve_input
            self.input = resolve_input(self.input_cfg, dataset_dir, slides)
        return self.input

    def _input_from_meta(self, meta: dict) -> None:
        """Weights-backed baselines: the stored input recipe (an old weights file without one trained on raw input)."""
        from ..inputs import InputSpec
        self.input, self.input_fixed = InputSpec.from_meta((meta or {}).get("input")), True

    def params(self) -> dict:
        """What identifies this run (goes into run.json and the default result name)."""
        return {}

    def predict(self, image: np.ndarray, pixel_size_um: Optional[float] = None, slide: Optional[str] = None) -> np.ndarray:
        return self.predict_batch([image], pixel_size_um, [slide])[0]

    def predict_batch(self, images: Sequence[np.ndarray], pixel_size_um: Optional[float] = None,
                      slides: Optional[Sequence[Optional[str]]] = None) -> List[np.ndarray]:
        """``slides`` names each image's slide (its normalization statistics); baselines without an input recipe ignore it."""
        slides = list(slides) if slides is not None else [None] * len(images)
        if self._predict_takes_slide():
            return [self.predict(im, pixel_size_um, s) for im, s in zip(images, slides)]
        return [self.predict(im, pixel_size_um) for im in images]                 # a baseline that never needs the slide

    def _predict_takes_slide(self) -> bool:
        import inspect
        params = list(inspect.signature(type(self).predict).parameters)
        return len(params) >= 4 or "slide" in params

    def describe(self) -> str:
        return self.name + ("(" + ", ".join(f"{k}={v}" for k, v in self.params().items()) + ")" if self.params() else "")

    def slug(self) -> str:
        """Default result name: the baseline name plus every parameter that differs from its constructor default."""
        import inspect
        defaults = {k: p.default for k, p in inspect.signature(type(self).__init__).parameters.items()}
        diff = [f"{k}-{v}" for k, v in self.params().items() if defaults.get(k, object()) != v]
        return "_".join([self.name] + diff)
