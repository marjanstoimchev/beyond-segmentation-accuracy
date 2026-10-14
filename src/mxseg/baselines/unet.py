"""3-class U-Net (background / nucleus / boundary) with watershed instances, trained by ``mxseg train`` (``model: unet``).

Inference is one function shared by training-time validation and ``mxseg eval``: ``segment(net, images)`` normalizes like
training (per-image percentiles 1-99), runs the network, and turns class probabilities into instances
(``instances_from_probs``: nucleus-interior components seed a watershed confined to the foreground, so touching nuclei
split along the predicted boundary). Weights: ``runs/<name>/weights.pt`` -> ``--baseline unet --param weights=...``.
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Sequence

import numpy as np

from ..metrics import relabel
from .base import Baseline

WEIGHTS_FORMAT = 1


def percentile_normalize(image: np.ndarray, lo: float = 1.0, hi: float = 99.0) -> np.ndarray:
    """Per channel ``(x - p_lo) / (p_hi - p_lo)`` as float32 — the same convention Cellpose applies to its inputs."""
    x = np.asarray(image, dtype=np.float32)
    out = np.empty_like(x)
    for c in range(x.shape[0]):
        p0, p1 = np.percentile(x[c], (lo, hi))
        out[c] = (x[c] - p0) / max(float(p1 - p0), 1e-6)
    return out


def build_unet(encoder: str = "resnet34", encoder_weights: Optional[str] = "imagenet", in_channels: int = 1, classes: int = 3):
    import segmentation_models_pytorch as smp
    return smp.Unet(encoder_name=encoder, encoder_weights=encoder_weights, in_channels=in_channels, classes=classes)


def save_unet(path, net, arch: dict, meta: Optional[dict] = None) -> None:
    import torch
    torch.save({"format": WEIGHTS_FORMAT, "arch": dict(arch), "meta": dict(meta or {}), "state_dict": net.state_dict()}, str(path))


def load_unet(path, device="cpu"):
    """The network saved by :func:`save_unet` (architecture included), in eval mode on ``device``."""
    import torch
    ck = torch.load(str(path), map_location="cpu", weights_only=True)
    if not isinstance(ck, dict) or ck.get("format") != WEIGHTS_FORMAT:
        raise ValueError(f"{path}: not a mxseg U-Net weights file (format {WEIGHTS_FORMAT})")
    net = build_unet(**{**ck["arch"], "encoder_weights": None})
    net.load_state_dict(ck["state_dict"])
    return net.to(device).eval(), ck["arch"], ck.get("meta", {})


def instances_from_probs(prob: np.ndarray, min_area: int = 15) -> np.ndarray:
    """``(3, H, W)`` class probabilities -> ``int32`` instances. Nucleus-interior components are the seeds of a watershed on
    the interior probability, confined to the foreground (interior + boundary); objects under ``min_area`` px are dropped."""
    from scipy import ndimage as ndi
    from skimage.segmentation import watershed

    cls = prob.argmax(0)
    markers, n = ndi.label(cls == 1)
    if n == 0:
        return np.zeros(cls.shape, np.int32)
    lab = watershed(-prob[1], markers, mask=cls > 0)
    if min_area > 0:
        small = np.bincount(lab.ravel()) < min_area
        small[0] = False
        lab[small[lab]] = 0
    return relabel(lab)[0].astype(np.int32)


def pick_device() -> str:
    import torch
    return "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")


def render_all(images: Sequence[np.ndarray], input, slides=None) -> List[np.ndarray]:
    """Network inputs for native-dtype marker stacks: the ``InputSpec`` (slide-level scale per channel, ``slides`` naming each
    image's slide), or per-patch percentile normalization when a model has none."""
    if input is None:
        return [percentile_normalize(im) for im in images]
    slides = slides or [None] * len(images)
    return [input.render(im, s) for im, s in zip(images, slides)]


def segment(net, images: Sequence[np.ndarray], device, *, min_area: int = 15, batch_size: int = 16, input=None, slides=None) -> List[np.ndarray]:
    """Instances for ``(C, H, W)`` native-dtype images; sides are reflect-padded to multiples of 32 for the U-Net."""
    import torch
    import torch.nn.functional as F

    xs = render_all(images, input, slides)
    out: List[np.ndarray] = []
    net.eval()
    with torch.inference_mode():
        for i in range(0, len(images), batch_size):
            x = torch.from_numpy(np.stack(xs[i:i + batch_size])).to(device)
            h, w = x.shape[-2:]
            ph, pw = (-h) % 32, (-w) % 32
            if ph or pw:
                x = F.pad(x, (0, pw, 0, ph), mode="reflect")
            prob = torch.softmax(net(x), 1)[..., :h, :w].float().cpu().numpy()
            out.extend(instances_from_probs(p, min_area) for p in prob)
    return out


class UNetBaseline(Baseline):
    """A U-Net trained by ``mxseg train``: ``mxseg eval --baseline unet --param weights=runs/<name>/weights.pt``."""

    name = "unet"

    def __init__(self, weights: str, min_area: int = 15, batch_size: int = 16, device: Optional[str] = None):
        self.weights, self.min_area, self.batch_size = weights, min_area, batch_size
        self._device = device
        self._net = None

    def params(self) -> dict:
        return {"weights": self.weights, "min_area": self.min_area}

    def slug(self) -> str:
        return f"unet_{Path(self.weights).parent.name}" + (f"_min_area-{self.min_area}" if self.min_area != 15 else "")

    @property
    def device(self) -> str:
        return self._device or pick_device()

    def model(self):
        if self._net is None:
            self._net, self.arch, self.meta = load_unet(self.weights, self.device)
            self._input_from_meta(self.meta)
        return self._net

    def input_markers(self):
        self.model()                                                     # the weights carry the input recipe
        return super().input_markers()

    def resolve_input(self, dataset_dir, slides):
        self.model()                                                     # the weights decide
        return super().resolve_input(dataset_dir, slides)

    def predict_batch(self, images: Sequence[np.ndarray], pixel_size_um: Optional[float] = None, slides=None) -> List[np.ndarray]:
        net = self.model()
        if self.meta.get("objective") == "flows":                       # weights trained under the Cellpose protocol
            from .kodiak import segment_flows
            return segment_flows(net, images, self.device, min_area=self.min_area, batch_size=self.batch_size, input=self.input, slides=slides)
        return segment(net, images, self.device, min_area=self.min_area, batch_size=self.batch_size, input=self.input, slides=slides)
