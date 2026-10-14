"""KODIAK — self-supervised masked prototype prediction with multi-crop prototype CLS alignment.

The method (model, losses, Lightning learner, multi-crop transforms, block masking) is ported from the reference
implementation at https://github.com/marjanstoimchev/KODIAK with compute-only changes -- the student decoder and
prototype head run on the tokens the mask loss reads (identical values and gradients, pinned by tests) -- one fix to the
masked pass (``mask_padded_keys``: the student attends over its real visible tokens only; False restores the reference,
whose padding let copies of one visible token act as extra keys) and a new data layer feeding the MxIF patches of this
repository: fixed nuclear-envelope / nuclei channels, a context marker drawn into green once per sample, and an MxIF
photometric augmentation in place of the natural-image one (``kodiak.data``, ``kodiak.transforms``). The DINOv3 ViT backbone layers are vendored under ``dinov3`` (Meta, see its LICENSE.md).

    python -m kodiak.pretrain --config configs/kodiak_pretrain.yaml
"""
from .collate import KodiakCollate, MultiCropKodiakCollate
from .data import ChannelPatches, MxIFDataModule
from .learner import KodiakLearner
from .losses import KoLeoLoss, KoLeoLossDistributed, MaskLoss
from .masking import MaskingGenerator
from .model import Kodiak, create_kodiak_small
from .prototype_cls_loss import MultiCropPrototypeCLSLoss
from .transforms import (
           DataAugmentationDINO,
           MultiCropPretrainingTransform,
           MxIFMultiCropTransform,
           make_transform,
)

__all__ = ["Kodiak", "create_kodiak_small", "KodiakLearner", "MxIFDataModule", "ChannelPatches", "MxIFMultiCropTransform", "make_transform",
           "MaskLoss", "KoLeoLoss", "KoLeoLossDistributed", "MultiCropPrototypeCLSLoss", "MaskingGenerator",
           "DataAugmentationDINO", "MultiCropPretrainingTransform", "KodiakCollate", "MultiCropKodiakCollate"]
