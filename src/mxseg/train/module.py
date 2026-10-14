"""What every trainable model shares: the masked loss bookkeeping, the one optimizer recipe (``OptimConfig``: AdamW, layer-wise
lr decay over the pretrained depth, warm-up then cosine, no weight decay on 1-D parameters), and validation instance metrics
computed with the model's real inference and scored exactly like ``mxseg eval`` (owned, verified nuclei; Wrong ignored).
The kept weights are chosen by one rule for every model: the epoch that minimizes the validation loss (``val.monitor``,
``val/loss`` by default; a metric such as ``val/f1`` is maximized instead).

A model module sets ``self.net`` and implements ``loss(batch)``, ``predict_instances(images)``, ``save_weights(path)``, and
``param_layers()`` — which of its parameters belong to which depth of a pretrained encoder (the only model-specific input to the
optimizer).
"""
from __future__ import annotations

import logging
import math
from typing import Dict, List, Optional, Tuple

import lightning as L
import numpy as np
import torch

from ..metrics import aggregate, score_patch
from .config import TrainConfig, as_dict

logger = logging.getLogger(__name__)


def metrics_due(epoch: int, max_epochs: int, every_n_epochs: int) -> bool:
    """Instance metrics run every ``every_n_epochs`` epochs and on the final epoch."""
    return (epoch + 1) % every_n_epochs == 0 or epoch + 1 == max_epochs


def is_better(value: float, best, mode: str) -> bool:
    """Whether ``value`` beats ``best`` (None = nothing yet) under ``mode`` (``min`` for a loss, ``max`` for a metric)."""
    return best is None or (value < best if mode == "min" else value > best)


def lr_factor(warmup_epochs: int, epochs: int, final_fraction: float = 0.01):
    """The shared schedule as a per-epoch multiplier of the peak: linear warm-up over ``warmup_epochs``, then cosine from 1 to
    ``final_fraction`` at the last epoch."""
    def factor(epoch: int) -> float:
        if warmup_epochs and epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs
        t = min((epoch - warmup_epochs) / max(epochs - warmup_epochs, 1), 1.0)
        return final_fraction + (1.0 - final_fraction) * 0.5 * (1.0 + math.cos(math.pi * t))
    return factor


class SegModule(L.LightningModule):
    def __init__(self, cfg: TrainConfig, metric_samples=None):
        super().__init__()
        self.cfg = cfg
        self.save_hyperparameters(as_dict(cfg))            # plain dict: loggers cannot sanitize frozen dataclasses
        self._metric_samples = metric_samples              # callable -> iterable of (RegionSample, (labels, ignore, scored, core))
        self._best: Dict[str, float] = {}                  # the kept epoch's validation quantities (+ epoch)
        self._val_acc = [0.0, 0]                           # validation loss accumulated over the epoch: sum(loss * n), n
        self.weights_path = None                           # set by run.py: the kept weights are written here whenever cfg.val.monitor improves
        self.input = None                                  # InputSpec of a composite-input model (set by build_model); None = raw

    # ---- to implement per model ---------------------------------------------------
    def loss(self, batch) -> Dict[str, torch.Tensor]:
        """Masked loss parts for one batch; ``"loss"`` is what is optimized."""
        raise NotImplementedError

    def predict_instances(self, images: List[np.ndarray], slides: Optional[List[str]] = None) -> List[np.ndarray]:
        """Real inference: ``(C, H, W)`` native images (the input's markers) -> int32 instance maps (eval mode, inference_mode);
        ``slides`` names the slide of each image, for its normalization statistics."""
        raise NotImplementedError

    def save_weights(self, path) -> None:
        raise NotImplementedError

    def param_layers(self) -> Tuple[int, List[tuple]]:
        """``(depth, [(layer_id, parameter), ...])`` over the trainable parameters: a pretrained encoder's layers get ids 1..depth
        from input to output (its stem / embeddings 0); everything new (decoder, head) gets depth + 1 and trains at the peak lr.
        Default: nothing pretrained, everything at the peak."""
        return 0, [(1, p) for p in self.net.parameters() if p.requires_grad]

    # ---- shared ------------------------------------------------------------------
    def _step(self, batch, stage: str) -> torch.Tensor:
        parts = self.loss(batch)
        n = batch["image"].shape[0]
        self.log_dict({f"{stage}/{k}": v for k, v in parts.items()}, on_step=stage == "train", on_epoch=True, prog_bar=(stage == "train"),
                      batch_size=n)
        if stage == "val":                                 # the same batch-size-weighted mean Lightning logs as val/loss
            self._val_acc[0] += float(parts["loss"].detach()) * n
            self._val_acc[1] += n
        return parts["loss"]

    def training_step(self, batch, _):
        return self._step(batch, "train")

    def validation_step(self, batch, _):
        self._step(batch, "val")

    def configure_optimizers(self):
        """One recipe for every model (``cfg.optim``): AdamW; lr = peak * decay^(depth + 1 - layer) per parameter group; weight
        decay on >= 2-D parameters only; linear warm-up then cosine decay to ``final_lr_fraction`` of the peak over the budget."""
        o = self.cfg.optim
        depth, layers = self.param_layers()
        groups: Dict[tuple, dict] = {}
        for lid, p in layers:
            wd = 0.0 if p.ndim <= 1 else o.weight_decay
            g = groups.setdefault((lid, wd), {"params": [], "lr": o.lr * o.layerwise_decay ** (depth + 1 - lid), "weight_decay": wd})
            g["params"].append(p)
        opt = torch.optim.AdamW(list(groups.values()), betas=(0.9, 0.999))
        sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_factor(o.warmup_epochs, self.cfg.epochs, o.final_lr_fraction))
        return {"optimizer": opt, "lr_scheduler": {"scheduler": sched, "interval": "epoch"}}

    def on_validation_epoch_end(self) -> None:
        val_loss, n = self._val_acc[0] / max(self._val_acc[1], 1), self._val_acc[1]
        self._val_acc = [0.0, 0]
        if self.trainer.sanity_checking or not n:
            return
        out = {"val/loss": val_loss}                        # already logged per batch by _step (Lightning reduces it the same way)
        if self._metric_samples is not None and metrics_due(self.current_epoch, self.trainer.max_epochs, self.cfg.val.every_n_epochs):
            rows: List[dict] = []
            samples = list(self._metric_samples())
            self.net.eval()
            with torch.inference_mode():
                for i in range(0, len(samples), 8):
                    chunk = samples[i:i + 8]
                    preds = self.predict_instances([rs.image for rs, _ in chunk], [rs.meta.get("slide") for rs, _ in chunk])
                    for (rs, (labels, ignore, scored, core)), pred in zip(chunk, preds):
                        rows.append(score_patch(labels, np.asarray(pred).astype(np.int32), ignore=ignore, gt_scored=scored, core=core))
            m = aggregate(rows)
            inst = {"val/f1": m["f1@0.5"], "val/ap": m["ap@0.5"], "val/ap_50_90": m["ap@0.5:0.9"], "val/pq": m["pq@0.5"],
                    "val/precision": m["precision@0.5"], "val/recall": m["recall@0.5"], "val/count_ratio": m["count_ratio"]}
            self.log_dict(inst, prog_bar=True)
            out.update(inst)
        mon = self.cfg.val.monitor
        if mon in out and is_better(out[mon], self._best.get(mon), self.cfg.val.mode):
            self._best = dict(out, epoch=self.current_epoch)
            if self.weights_path is not None:
                self.save_weights(self.weights_path)            # an interrupted run still leaves its kept weights
        logger.info("epoch %d | %s%s", self.current_epoch, "  ".join(f"{k.split('/')[1]} {v:.3f}" for k, v in out.items()),
                    "  <- kept" if self._best.get("epoch") == self.current_epoch else "")
