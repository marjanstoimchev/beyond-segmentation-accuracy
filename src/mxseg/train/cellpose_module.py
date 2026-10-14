"""Cellpose-SAM fine-tuning as a LightningModule.

Same network, targets, loss and optimizer schedule as ``cellpose.train.train_seg`` (AdamW, linear warm-up then
constant lr, flows scaled by 5, ½·MSE + BCE), plus what Cellpose's trainer cannot do: the loss is masked with
the ``ignore`` map, so nuclei flagged wrong are never trained on. The instance metrics on validation run Cellpose's
real inference (flows -> masks) on the fine-tuned net, scored on owned, verified nuclei like ``mxseg eval``.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict

import numpy as np
import torch
import torch.nn.functional as F

from ..inputs import input_meta
from .config import TrainConfig
from .module import SegModule, metrics_due  # noqa: F401  (metrics_due re-exported)

logger = logging.getLogger(__name__)


def masked_cellpose_loss(y: torch.Tensor, target: torch.Tensor, ignore: torch.Tensor) -> Dict[str, torch.Tensor]:
    """``y`` (B, 3, H, W) = [flowY, flowX, cellprob-logit]; ``target`` (B, 3, H, W) = [cellprob, dY, dX]. Pixels with ``ignore``
    contribute nothing. Returns the total and its two parts (means over the counted pixels)."""
    w = (~ignore).to(y.dtype)
    flow = 0.5 * ((y[:, :2] - 5.0 * target[:, 1:]) ** 2).mean(1)                     # cellpose: MSE over 2 channels / 2
    prob = F.binary_cross_entropy_with_logits(y[:, 2], (target[:, 0] > 0.5).to(y.dtype), reduction="none")
    n = w.sum().clamp(min=1.0)
    return {"loss": ((flow + prob) * w).sum() / n, "flow": (flow * w).sum() / n, "prob": (prob * w).sum() / n}


def set_trainable(net, n_blocks: int) -> int:
    """Choose what trains: 0 = the readout head only (whole encoder frozen); N = head + the encoder's top module + the top N ViT
    blocks; all blocks = the whole network. Explicit because the SAM image encoder arrives frozen (``segment_anything.build_sam``
    sets ``requires_grad`` False on every encoder parameter), so ``net.parameters()`` alone would train only patch-embed,
    pos-embed and head. Works for both cellpose backbones: CPSAM (ViT-L/8, 24 blocks, conv ``neck``, learned ``pos_embed``) and
    CPDINO (DINOv3 ViT-B or ViT-L, final LayerNorm ``norm``, RoPE + ``cls_token`` / ``storage_tokens``). Note both forwards read
    ``patch_embed.proj.weight.data``, which detaches: the patch embedding never receives a gradient whatever is unfrozen here, so
    "full fine-tune" means the blocks, the top module and the readout. Cellpose's fixed tensors (``W2``, ``diam_mean``,
    ``diam_labels``) always stay frozen. Returns the trainable count."""
    enc = net.encoder
    blocks = enc.blocks
    full = n_blocks >= len(blocks)
    top = enc.neck if hasattr(enc, "neck") else enc.norm              # CPSAM: the conv neck | CPDINO: the final LayerNorm
    for p in net.parameters():
        p.requires_grad_(False)
    train = (net.out,) + ((top, *blocks[len(blocks) - n_blocks:]) if n_blocks else ()) + ((enc.patch_embed,) if full else ())
    for m in train:
        for p in m.parameters():
            p.requires_grad_(True)
    for name in ("pos_embed", "cls_token", "storage_tokens"):         # whichever this backbone has; absent ones are skipped
        t = getattr(enc, name, None)
        if t is not None:
            t.requires_grad_(full)
    return sum(p.numel() for p in net.parameters() if p.requires_grad)


class CellposeFinetune(SegModule):
    def __init__(self, cfg: TrainConfig, metric_samples=None):
        super().__init__(cfg, metric_samples)
        from cellpose import models
        self.cp = models.CellposeModel(gpu=True, pretrained_model=cfg.cellpose.variant, use_bfloat16=False)   # float32 for training
        self.net = self.cp.net
        set_trainable(self.net, cfg.cellpose.train_blocks)
        self.net.rdrop = cfg.cellpose.rdrop                 # cellpose's layer-drop is only active in train mode

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)[0]

    def loss(self, batch):
        return masked_cellpose_loss(self(batch["image"]), batch["target"], batch["ignore"])

    def param_layers(self):
        """Patch / position / prefix embeddings 0, blocks 1..depth, the encoder's top module (CPSAM's neck, CPDINO's final norm)
        with the top block, the readout head at the peak; whatever ``set_trainable`` froze is absent."""
        depth = len(self.net.encoder.blocks)

        def layer_of(name: str) -> int:
            if name.startswith("encoder.blocks."):
                return int(name.split(".")[2]) + 1
            if name.startswith("encoder.neck") or name.startswith("encoder.norm"):
                return depth
            return 0 if name.startswith("encoder.") else depth + 1

        return depth, [(layer_of(n), p) for n, p in self.net.named_parameters() if p.requires_grad]

    def predict_instances(self, images, slides=None):
        """Cellpose's own inference (flows -> masks) on the fine-tuned net, normalized per image like ``mxseg eval``."""
        slides = slides or [None] * len(images)
        x = [np.moveaxis(self.input.render(im, s), 0, -1) if self.input is not None else im[0] for im, s in zip(images, slides)]   # (H, W, C) or (H, W)
        masks = self.cp.eval(x, batch_size=8, normalize=True)[0]
        return [np.asarray(m).astype(np.int32) for m in masks]

    def save_weights(self, path) -> None:
        """Cellpose-format weights: loadable with ``CellposeModel(pretrained_model=path)`` and ``mxseg eval --param variant=path``.
        The input recipe goes into a ``<weights>.input.json`` sidecar (cellpose's format carries no metadata)."""
        self.net.save_model(str(path))
        Path(path).with_suffix(".input.json").write_text(json.dumps(input_meta(self.input), indent=2))
