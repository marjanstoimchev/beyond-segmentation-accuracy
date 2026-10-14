"""KODIAK segmentation fine-tuning: the pretrained encoder under the identical 3-class protocol as the U-Net — same
targets, same masked cross-entropy + Dice, same watershed instancing (``segment``), same 20-epoch budget — so the row it
adds to the table measures the representation, not the recipe.

The optimizer is the recipe every trained model shares (``OptimConfig``, built in ``SegModule``); this module only declares
which of its parameters sit at which encoder depth, so the layer-wise learning-rate decay lands on the ViT blocks.
"""
from __future__ import annotations

import logging

from ..baselines.kodiak import (
    build_segmenter,
    load_encoder_weights,
    param_layers,
    save_segmenter,
    set_trainable,
)
from ..baselines.unet import segment
from ..inputs import input_meta
from .config import TrainConfig
from .module import SegModule
from .unet_module import masked_class_loss

logger = logging.getLogger(__name__)


class KodiakSegModule(SegModule):
    def __init__(self, cfg: TrainConfig, metric_samples=None, in_channels: int = 1):
        super().__init__(cfg, metric_samples)
        if in_channels not in (1, 3):
            raise ValueError(f"kodiak segmenter takes 1 or 3 markers, got {in_channels}")
        k = cfg.kodiak
        self.net = build_segmenter({"patch_size": k.vit_patch_size, "embed_dim": k.vit_embed_dim, "depth": k.vit_depth,
                                    "num_heads": k.vit_heads, "mlp_ratio": k.vit_mlp_ratio,
                                    "num_storage_tokens": k.num_storage_tokens,
                                    "drop_path_rate": k.drop_path_rate if k.train_blocks else 0.0,
                                    "encoder_upsample": k.encoder_upsample, "objective": k.objective,
                                    "bottleneck": k.bottleneck})
        self.init_from = load_encoder_weights(self.net.backbone, k.checkpoint) if k.checkpoint else "scratch"
        n_train = set_trainable(self.net, k.train_blocks)
        logger.info("kodiak segmenter: encoder from %s (%s), top %d/%d blocks + head trainable (%.1fM parameters)",
                    k.checkpoint, self.init_from, k.train_blocks, len(self.net.backbone.blocks), n_train / 1e6)

    def param_layers(self):
        """ViT blocks 1..12 (patch embedding, RoPE and tokens at 0); the UNETR head and the final norm at 13, the peak."""
        return param_layers(self.net)

    def forward(self, x):
        return self.net(x)

    def loss(self, batch):
        if self.cfg.kodiak.objective == "flows":
            from .cellpose_module import masked_cellpose_loss
            return masked_cellpose_loss(self(batch["image"]), batch["target"], batch["ignore"])
        return masked_class_loss(self(batch["image"]), batch["target"], batch["ignore"], self.cfg.kodiak.dice_weight)

    def predict_instances(self, images, slides=None):
        if self.cfg.kodiak.objective == "flows":
            from ..baselines.kodiak import segment_flows
            return segment_flows(self.net, images, self.device, min_area=self.cfg.kodiak.min_area, input=self.input, slides=slides)
        return segment(self.net, images, self.device, min_area=self.cfg.kodiak.min_area, input=self.input, slides=slides)

    def save_weights(self, path) -> None:
        save_segmenter(path, self.net, meta={"objective": self.cfg.kodiak.objective, "input": input_meta(self.input),
                                             "boundary_px": self.cfg.kodiak.boundary_px, "min_area": self.cfg.kodiak.min_area,
                                             "checkpoint": self.cfg.kodiak.checkpoint, "init_from": self.init_from,
                                             "train_blocks": self.cfg.kodiak.train_blocks})
