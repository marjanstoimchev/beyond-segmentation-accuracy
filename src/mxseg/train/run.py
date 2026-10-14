"""``train(cfg)``: seed, data, model, Lightning trainer, checkpoints, logs, final cellpose-format weights.

    runs/<name>/
      resolved_config.yaml   the train config + the data config it used
      data_manifest.json     dataset stamp, seed, sorted train / val patch keys — every baseline must show the same
      weights.pt             the kept network: the epoch with the minimum validation loss (val.monitor), rewritten whenever it
                             improves (an interrupted run keeps its best so far):
                             cellpose format (`--baseline cellpose --param variant=...`) or mxseg U-Net format
                             (`--baseline unet --param weights=...`) — see `eval_command`; `export_weights` rebuilds it from a checkpoint
      checkpoints/           best (by val.monitor) and last Lightning checkpoints
      metrics.csv, tensorboard/   training curves
      summary.json           best validation metrics, epochs, timing
"""
from __future__ import annotations

import dataclasses
import json
import logging
import time
from pathlib import Path

import lightning as L
import torch
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger, TensorBoardLogger

from mxdata import DEFAULT_DATASET, load_config
from mxdata import save_resolved as save_data_config

from ..inputs import input_meta, resolve_input
from .config import TrainConfig, as_dict, load_train_config, save_resolved
from .data import CellposePatches, FlowPatches, TrainData, UNetPatches

logger = logging.getLogger(__name__)


def _accelerator(name: str) -> str:
    if name != "auto":
        return name
    return "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")


def train(cfg: TrainConfig, fast_dev_run: bool = False) -> Path:
    t0 = time.time()
    L.seed_everything(cfg.seed, workers=True)
    out = Path(cfg.out) / cfg.name
    out.mkdir(parents=True, exist_ok=True)
    data_cfg = load_config(cfg.data)
    if cfg.slides:                                                     # --slides: a subset of the dataset config's slides
        if unknown := set(cfg.slides) - set(data_cfg.slides):
            raise ValueError(f"--slides {sorted(unknown)} are not in {cfg.data} (slides: {list(data_cfg.slides)})")
        data_cfg = dataclasses.replace(data_cfg, slides=tuple(cfg.slides))
    save_resolved(cfg, out / "resolved_config.yaml")
    save_data_config(data_cfg, out / "resolved_data_config.yaml")
    acc = _accelerator(cfg.accelerator)
    flow_device = torch.device(acc) if acc in ("cuda", "mps") else None
    spec = resolve_input(cfg.input, cfg.dataset_dir or DEFAULT_DATASET, data_cfg.slides)
    logger.info("input: %s", spec.describe())

    sub = cfg.kodiak if cfg.model == "kodiak" else cfg.unet
    if cfg.model == "cellpose":
        adapter, kw = CellposePatches, {"flow_device": flow_device}
    elif getattr(sub, "objective", "threeclass") == "flows":           # kodiak or unet under the Cellpose protocol
        adapter, kw = FlowPatches, {"flow_device": flow_device}
    else:                                                              # the 3-class adapter (identical items for unet and kodiak)
        adapter, kw = UNetPatches, {"boundary_px": sub.boundary_px}
    dm = TrainData(data_cfg, cfg.dataset_dir or DEFAULT_DATASET, adapter, batch_size=cfg.batch_size, num_workers=cfg.num_workers,
                   augment=cfg.augment.flip or cfg.augment.rot90, seed=cfg.seed, n_metric_patches=cfg.val.n_metric_patches,
                   input=spec, **kw)
    dm.write_manifest(out / "data_manifest.json")                      # exactly which patches this run sees
    logger.info("preparing targets once (%d train + %d val patches)", len(dm.train), len(dm.val))
    dm.train.precompute()
    dm.val.precompute()

    model = build_model(cfg, in_channels=spec.n_channels if spec else len(data_cfg.markers), metric_samples=dm.metric_samples, input_spec=spec)
    model.weights_path = out / "weights.pt"                             # the kept weights (min val/loss), rewritten whenever they improve
    variant, lr = _variant(cfg)
    ckpt = ModelCheckpoint(dirpath=out / "checkpoints", filename="best-{epoch:03d}-{" + cfg.val.monitor + ":.4f}",
                           monitor=cfg.val.monitor, mode=cfg.val.mode,
                           every_n_epochs=1 if cfg.val.monitor == "val/loss" else cfg.val.every_n_epochs,   # instance metrics exist only on their epochs
                           save_last=True, auto_insert_metric_name=False)
    trainer = L.Trainer(max_epochs=cfg.epochs, accelerator=acc, devices=1, precision=cfg.precision, default_root_dir=str(out),
                        accumulate_grad_batches=cfg.grad_accumulation,
                        logger=[CSVLogger(str(out), name="", version="logs"), TensorBoardLogger(str(out), name="", version="tensorboard")],
                        callbacks=[ckpt, LearningRateMonitor(logging_interval="epoch")], log_every_n_steps=10,
                        check_val_every_n_epoch=1, fast_dev_run=fast_dev_run, enable_model_summary=False, deterministic=False)
    logger.info("training %s: %s %s on %s, %d epochs, batch %d, lr %g", cfg.name, cfg.model, variant, acc, cfg.epochs, cfg.batch_size, lr)
    trainer.fit(model, train_dataloaders=dm.train_dataloader(), val_dataloaders=dm.val_dataloader())

    if not model._best:                                                 # no validation epoch ran: keep the final weights
        model.save_weights(out / "weights.pt")
    summary = {"name": cfg.name, "model": cfg.model, "variant": variant, "epochs": trainer.current_epoch, "accelerator": acc,
               "input": input_meta(spec), "best": model._best, "best_checkpoint": ckpt.best_model_path, "seconds": time.time() - t0,
               "config": as_dict(cfg)}
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    logger.info("done in %.0fs -> %s (best %s)", summary["seconds"], out, model._best)
    return out


def eval_command(cfg: TrainConfig, out: Path) -> str:
    """The `mxseg eval` line that scores this run's weights on the validation split."""
    w = out / "weights.pt"
    param = f"variant={w}" if cfg.model == "cellpose" else f"weights={w}"
    return f"mxseg eval -c {cfg.data} --baseline {cfg.model} --param {param} --name {cfg.name}"


def build_model(cfg: TrainConfig, in_channels: int = 1, metric_samples=None, input_spec=None):
    if cfg.model == "cellpose":
        from .cellpose_module import CellposeFinetune
        model = CellposeFinetune(cfg, metric_samples=metric_samples)
    elif cfg.model == "kodiak":
        from .kodiak_module import KodiakSegModule
        model = KodiakSegModule(cfg, metric_samples=metric_samples, in_channels=in_channels)
    else:
        from .unet_module import UNetModule
        model = UNetModule(cfg, metric_samples=metric_samples, in_channels=in_channels)
    model.input = input_spec                                            # what predict_instances renders and save_weights records
    return model


def _variant(cfg: TrainConfig):
    """(what identifies the run within its model family, its learning rate) — for logs and summary.json."""
    if cfg.model == "cellpose":
        return cfg.cellpose.variant, cfg.optim.lr
    if cfg.model == "kodiak":
        ck = cfg.kodiak.checkpoint
        return (Path(ck).parent.parent.name if ck and ck.endswith(".ckpt") else Path(ck).stem if ck else "scratch"), cfg.optim.lr
    return f"{cfg.unet.encoder}/{cfg.unet.encoder_weights}", cfg.optim.lr


def export_weights(run_dir, checkpoint: str = "best") -> Path:
    """Write ``runs/<name>/weights.pt`` from a Lightning checkpoint of that run: ``best`` (the run's ``val.monitor`` optimum —
    minimum validation loss by default), ``last``, or a path.
    For runs that were interrupted before their weights were written, or to evaluate the last epoch instead of the best."""
    run_dir = Path(run_dir)
    if checkpoint in ("best", "last"):
        found = sorted((run_dir / "checkpoints").glob("best-*.ckpt")) if checkpoint == "best" else [run_dir / "checkpoints" / "last.ckpt"]
        if not found or not found[-1].is_file():
            raise FileNotFoundError(f"no {checkpoint} checkpoint under {run_dir / 'checkpoints'}")
        ckpt = found[-1]
    else:
        ckpt = Path(checkpoint)
        if not ckpt.is_file():
            raise FileNotFoundError(f"checkpoint not found: {ckpt}")
    cfg = load_train_config(run_dir / "resolved_config.yaml")
    data_cfg = load_config(run_dir / "resolved_data_config.yaml")
    spec = resolve_input(cfg.input, cfg.dataset_dir or DEFAULT_DATASET, data_cfg.slides)
    model = build_model(cfg, in_channels=spec.n_channels if spec else len(data_cfg.markers), input_spec=spec)
    model.load_state_dict(torch.load(ckpt, map_location="cpu", weights_only=False)["state_dict"])
    model.save_weights(run_dir / "weights.pt")
    logger.info("exported %s -> %s", ckpt.name, run_dir / "weights.pt")
    return run_dir / "weights.pt"
