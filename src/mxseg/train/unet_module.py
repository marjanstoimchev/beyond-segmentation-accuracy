"""3-class U-Net training: masked cross-entropy + soft Dice on the nucleus and boundary classes; validation instances come
from the same ``segment`` (softmax -> watershed) that ``mxseg eval --baseline unet`` uses."""
from __future__ import annotations

from typing import Dict

import torch
import torch.nn.functional as F

from ..baselines.unet import build_unet, save_unet, segment
from ..inputs import input_meta
from .config import TrainConfig
from .module import SegModule


def masked_class_loss(logits: torch.Tensor, target: torch.Tensor, ignore: torch.Tensor, dice_weight: float = 1.0) -> Dict[str, torch.Tensor]:
    """``logits`` (B, 3, H, W), ``target`` (B, H, W) classes, ``ignore`` (B, H, W) bool: ignored pixels contribute to neither term."""
    w = (~ignore).to(logits.dtype)
    n = w.sum().clamp(min=1.0)
    ce = (F.cross_entropy(logits, target, reduction="none") * w).sum() / n
    p = torch.softmax(logits, 1) * w[:, None]
    t = F.one_hot(target, logits.shape[1]).permute(0, 3, 1, 2).to(logits.dtype) * w[:, None]
    inter, denom = (p * t).sum((0, 2, 3))[1:], (p.sum((0, 2, 3)) + t.sum((0, 2, 3)))[1:]        # classes 1 (nucleus), 2 (boundary)
    dice = 1.0 - ((2.0 * inter + 1.0) / (denom + 1.0)).mean()
    return {"loss": ce + dice_weight * dice, "ce": ce, "dice": dice}


class UNetModule(SegModule):
    def __init__(self, cfg: TrainConfig, metric_samples=None, in_channels: int = 1):
        super().__init__(cfg, metric_samples)
        u = cfg.unet
        self.arch = {"encoder": u.encoder, "in_channels": in_channels, "classes": 3}
        self.net = build_unet(u.encoder, u.encoder_weights, in_channels, 3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def loss(self, batch):
        if self.cfg.unet.objective == "flows":
            from .cellpose_module import masked_cellpose_loss
            return masked_cellpose_loss(self(batch["image"]), batch["target"], batch["ignore"])
        return masked_class_loss(self(batch["image"]), batch["target"], batch["ignore"], self.cfg.unet.dice_weight)

    def param_layers(self):
        """ResNet-style encoder: stem 0, stages layer1..layer4 -> 1..4 (the ImageNet-pretrained depth), decoder and head 5 (peak).
        An encoder trained from scratch has no pretrained depth: everything at the peak."""
        if self.cfg.unet.encoder_weights is None:
            return super().param_layers()
        stages = [f"encoder.layer{i}." for i in (1, 2, 3, 4)]

        def layer_of(name: str) -> int:
            for i, s in enumerate(stages):
                if name.startswith(s):
                    return i + 1
            return 0 if name.startswith("encoder.") else len(stages) + 1

        return len(stages), [(layer_of(n), p) for n, p in self.net.named_parameters() if p.requires_grad]

    def predict_instances(self, images, slides=None):
        if self.cfg.unet.objective == "flows":
            from ..baselines.kodiak import segment_flows
            return segment_flows(self.net, images, self.device, min_area=self.cfg.unet.min_area, input=self.input, slides=slides)
        return segment(self.net, images, self.device, min_area=self.cfg.unet.min_area, input=self.input, slides=slides)

    def save_weights(self, path) -> None:
        save_unet(path, self.net, self.arch, meta={"objective": self.cfg.unet.objective, "input": input_meta(self.input),
                                                   "boundary_px": self.cfg.unet.boundary_px, "min_area": self.cfg.unet.min_area})
