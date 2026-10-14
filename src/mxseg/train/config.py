"""Strict training configuration (unknown keys fail), resolved and saved with every run."""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import yaml

from mxdata.config import _from_dict, to_dict
from mxdata.errors import ConfigError

from ..inputs import InputConfig

MODELS = ("cellpose", "unet", "kodiak")


def _check(cond: bool, msg: str) -> None:
    if not cond:
        raise ConfigError(msg)


@dataclass(frozen=True)
class CellposeTrainConfig:
    variant: str = "cpsam_v2"        # weights to start from (a cellpose model name or a .pt path)
    rdrop: float = 0.4               # cellpose's stochastic depth (blocks skipped at random while training); 0 when the encoder is frozen
    train_blocks: int = 24           # ViT blocks (of 24) trained, counted from the top: 24 = full fine-tune like cellpose,
    #                                  4 = last 4 blocks + neck + head, 0 = readout head only (encoder frozen)

    def __post_init__(self):
        _check(0 <= self.train_blocks <= 24, "cellpose.train_blocks must be in 0..24")
        _check(0 <= self.rdrop < 1, "cellpose.rdrop must be in [0, 1)")


@dataclass(frozen=True)
class UNetTrainConfig:
    encoder: str = "resnet34"                    # any segmentation_models_pytorch encoder
    encoder_weights: Optional[str] = "imagenet"  # null = from scratch
    objective: str = "flows"                     # the protocol: "flows" (Cellpose protocol, same 3 output channels) | "threeclass" (classes -> watershed ablation)
    boundary_px: int = 2                         # threeclass only: thickness of the boundary class (outermost pixels of every nucleus)
    dice_weight: float = 1.0                     # loss = masked cross-entropy + dice_weight * masked soft Dice (nucleus + boundary)
    min_area: int = 15                           # instances under this many pixels are dropped after the watershed

    def __post_init__(self):
        _check(self.objective in ("threeclass", "flows"), "unet.objective must be 'threeclass' or 'flows'")
        _check(self.boundary_px >= 1 and self.dice_weight >= 0 and self.min_area >= 0, "unet: boundary_px >= 1, dice_weight >= 0, min_area >= 0")


@dataclass(frozen=True)
class KodiakTrainConfig:
    checkpoint: Optional[str] = "runs/kodiak_mxif/checkpoints/last.ckpt"   # a KODIAK pretraining checkpoint (its EMA
    #                                  teacher is taken) or the raw DINOv3 .pth in weights/ (the no-pretraining ablation); null = scratch
    train_blocks: int = 12           # ViT blocks (of vit_depth) trained, counted from the top: all = full fine-tune, 0 = head only (encoder frozen)
    vit_patch_size: int = 16         # THE ENCODER, named as configs/kodiak_pretrain.yaml names it, so a pretraining config and the
    vit_embed_dim: int = 384         # fine-tuning config that consumes its checkpoint are read side by side. Defaults = DINOv3 ViT-S/16
    vit_depth: int = 12              # (weights/dinov3_vits16_*.pth, 21.6M). ViT-B/16 (weights/dinov3_vitb16_*.pth, 85.7M):
    vit_heads: int = 6               #     vit_embed_dim: 768, vit_heads: 12   (patch 16, depth 12, storage 4, mlp 4.0 are unchanged)
    vit_mlp_ratio: float = 4.0       # The checkpoint must match these exactly — mxseg.baselines.kodiak.load_encoder_weights loads the
    num_storage_tokens: int = 4      # encoder whole or raises, and its message reports the architecture the checkpoint actually holds.
    objective: str = "flows"         # the protocol: "flows" (Cellpose protocol: flow field -> follow flows) | "threeclass" (U-Net-style classes -> watershed ablation)
    boundary_px: int = 2             # threeclass only: same 3-class targets as the U-Net
    drop_path_rate: float = 0.1      # stochastic depth in the encoder while fine-tuning (0 with a frozen encoder)
    encoder_upsample: int = 1        # 2 = encode at 2x input resolution (8 px per token, ~4x encoder compute)
    bottleneck: str = "multi"        # decoder bottleneck reads blocks 5+8+final fused (the final block alone is the worst layer, measured)
    dice_weight: float = 1.0
    min_area: int = 15

    def __post_init__(self):
        _check(0 <= self.train_blocks <= self.vit_depth, f"kodiak.train_blocks must be in 0..{self.vit_depth} (vit_depth)")
        _check(self.vit_embed_dim > 0 and self.vit_embed_dim % self.vit_heads == 0,
               "kodiak.vit_embed_dim must be positive and divisible by kodiak.vit_heads")
        _check(self.vit_depth > 0 and self.vit_patch_size > 0 and self.vit_mlp_ratio > 0 and self.num_storage_tokens >= 0,
               "kodiak: vit_depth, vit_patch_size, vit_mlp_ratio > 0 and num_storage_tokens >= 0")
        _check(0 <= self.drop_path_rate < 1, "kodiak.drop_path_rate must be in [0, 1)")
        _check(self.encoder_upsample in (1, 2), "kodiak.encoder_upsample must be 1 or 2")
        _check(self.objective in ("threeclass", "flows"), "kodiak.objective must be 'threeclass' or 'flows'")
        _check(self.bottleneck in ("final", "multi"), "kodiak.bottleneck must be 'final' or 'multi'")
        _check(self.boundary_px >= 1 and self.dice_weight >= 0 and self.min_area >= 0, "kodiak: boundary_px >= 1, dice_weight >= 0, min_area >= 0")


@dataclass(frozen=True)
class ValConfig:
    every_n_epochs: int = 5          # instance metrics (Cellpose inference on val patches) are slower than the loss
    n_metric_patches: Optional[int] = 40   # None = all val patches
    monitor: str = "val/loss"        # the kept weights and the best checkpoint: the epoch that minimizes this (a loss) or maximizes it
                                     # (anything else, e.g. val/f1). One rule for every trained model.

    def __post_init__(self):
        _check(self.every_n_epochs >= 1, "val.every_n_epochs must be >= 1")
        _check(self.monitor.startswith("val/"), "val.monitor must be a validation quantity (val/loss, val/f1, ...)")

    @property
    def mode(self) -> str:
        """``min`` for a loss, ``max`` for a metric."""
        return "min" if self.monitor.endswith("loss") else "max"


@dataclass(frozen=True)
class OptimConfig:
    """The one optimizer recipe every trained model uses — ViT fine-tuning best practice, applied to all (an undertuned
    protocol makes any pretrained encoder look worse than a frozen one): AdamW, the peak lr at the new decoder / head,
    layer-wise lr decay over a pretrained encoder's depth, linear warm-up into a cosine decay, no weight decay on 1-D
    parameters (biases, norms, tokens). Built once in ``SegModule.configure_optimizers``; models only declare their depth."""
    lr: float = 2e-3                 # PEAK lr: new decoder / head; pretrained layer i of L trains at lr * layerwise_decay^(L + 1 - i)
    layerwise_decay: float = 0.75
    weight_decay: float = 0.05       # on parameters with >= 2 dims only
    warmup_epochs: int = 2           # linear warm-up, then cosine decay towards final_lr_fraction * lr over the remaining epochs
    final_lr_fraction: float = 0.01

    def __post_init__(self):
        _check(self.lr > 0 and self.weight_decay >= 0 and self.warmup_epochs >= 0, "optim: lr > 0, weight_decay >= 0, warmup_epochs >= 0")
        _check(0 < self.layerwise_decay <= 1 and 0 <= self.final_lr_fraction <= 1, "optim: layerwise_decay in (0, 1], final_lr_fraction in [0, 1]")


@dataclass(frozen=True)
class AugmentConfig:
    flip: bool = True
    rot90: bool = True


@dataclass(frozen=True)
class TrainConfig:
    name: str = "cellpose_ft"
    data: str = "configs/dataset.yaml"         # the dataset config (slides, markers, qc_policy, split)
    slides: Optional[List[str]] = None         # train on these slides only (default: every slide of the dataset config; the split
                                               # still decides train / val within them) — `mxseg train --slides SLIDE-0235`
    dataset_dir: Optional[str] = None          # default: $MXDATA_ROOT/dataset
    model: str = "cellpose"
    input: InputConfig = field(default_factory=InputConfig)   # what the network sees: red LAP2, green blank, blue DAPI — the channels the
    #                                            reference boundaries were drawn from, on the dataset's stored slide statistics (mxseg/inputs.py)
    optim: OptimConfig = field(default_factory=OptimConfig)   # the optimizer recipe shared by every trained model
    cellpose: CellposeTrainConfig = field(default_factory=CellposeTrainConfig)
    unet: UNetTrainConfig = field(default_factory=UNetTrainConfig)
    kodiak: KodiakTrainConfig = field(default_factory=KodiakTrainConfig)
    epochs: int = 50
    batch_size: int = 4
    grad_accumulation: int = 1       # optimizer steps see batch_size * grad_accumulation items (standard ViT memory practice)
    num_workers: int = 0
    accelerator: str = "auto"                  # auto | mps | cuda | cpu
    precision: str = "32-true"                 # "bf16-mixed" on CUDA
    seed: int = 0
    augment: AugmentConfig = field(default_factory=AugmentConfig)
    val: ValConfig = field(default_factory=ValConfig)
    out: str = "runs"                          # runs/<name>/

    def __post_init__(self):
        _check(self.model in MODELS, f"model must be one of {MODELS}, got {self.model!r}")
        _check(self.epochs >= 1 and self.batch_size >= 1 and self.num_workers >= 0, "epochs >= 1, batch_size >= 1, num_workers >= 0")
        _check(self.grad_accumulation >= 1, "grad_accumulation must be >= 1")
        _check(bool(self.name), "name must be non-empty")


def load_train_config(path: "str | Path", **overrides) -> TrainConfig:
    p = Path(path)
    if not p.is_file():
        raise ConfigError(f"train config not found: {p}")
    d = yaml.safe_load(p.read_text()) or {}
    for k, v in overrides.items():                  # 'cellpose.lr' -> d['cellpose']['lr']
        if v is not None:
            sec, _, key = k.rpartition(".")
            (d.setdefault(sec, {}) if sec else d)[key] = v
    return _from_dict(TrainConfig, d, "train")


def save_resolved(cfg: TrainConfig, path: Path) -> Path:
    """The run's config, defaults included, with only the active model's block (the other models' blocks are not used)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    d = to_dict(cfg)
    for other in ("cellpose", "unet", "kodiak"):
        if other != cfg.model:
            d.pop(other, None)
    path.write_text(f"# Fully resolved mxseg train config (defaults included) of a `{cfg.model}` run, written by the run that used it.\n"
                    + yaml.safe_dump(d, sort_keys=False))
    return path


def as_dict(cfg: TrainConfig) -> dict:
    return dataclasses.asdict(cfg)
