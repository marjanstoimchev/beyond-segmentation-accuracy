"""KODIAK self-supervised pretraining on the MxIF patches.

    python -m kodiak.pretrain --config configs/kodiak_pretrain.yaml [--epochs N] [--batch-size B]
                              [--accelerator auto|mps|cuda|cpu] [--fast-dev-run]

The YAML follows the original repository's section layout (model / loss / optimizer / training / data), so its
recommended configurations port over unchanged; runs land in <experiment.out, default runs>/<experiment.name>/.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch
import yaml


def build(cfg: dict):
    from .data import MxIFDataModule
    from .learner import KodiakLearner
    model_c, loss_c, opt_c, train_c, data_c = (cfg.get(k, {}) for k in ("model", "loss", "optimizer", "training", "data"))
    aug = data_c.get("augmentation", {})
    learner = KodiakLearner(
        img_size=model_c.get("image_size", 256), patch_size=model_c.get("vit_patch_size", 16),
        embed_dim=model_c.get("vit_embed_dim", 384), vit_depth=model_c.get("vit_depth", 12),
        vit_heads=model_c.get("vit_heads", 6), mlp_ratio=model_c.get("vit_mlp_ratio", 4.0),
        num_storage_tokens=model_c.get("num_storage_tokens", 4),
        num_prototypes=model_c.get("num_prototypes", 128), projector_dim=model_c.get("projector_dim", 256),
        decoder_embed_dim=model_c.get("decoder_dim", 192), decoder_depth=model_c.get("decoder_depth", 4),
        decoder_num_heads=model_c.get("decoder_heads", 6), drop_path_rate=model_c.get("drop_path_rate", 0.1),
        pretrained_path=model_c.get("pretrained_path"), freeze_backbone=model_c.get("freeze_backbone", False),
        mask_padded_keys=model_c.get("mask_padded_keys", True),
        backbone_lr_scale=model_c.get("backbone_lr_scale", 1.0), compile_model=model_c.get("compile_model", False),
        center_momentum=loss_c.get("center_momentum", 0.9), sinkhorn_iters=loss_c.get("sinkhorn_iters", 3),
        koleo_loss_weight=loss_c.get("koleo_loss_weight", 0.1), koleo_distributed=loss_c.get("koleo_distributed", False),
        prototype_cls_loss_weight=loss_c.get("prototype_cls_loss_weight", 1.0),
        prototype_cls_student_temp=loss_c.get("prototype_cls_student_temp", 0.1),
        prototype_cls_teacher_temp=loss_c.get("prototype_cls_teacher_temp", 0.04),
        n_local_crops=loss_c.get("n_local_crops", aug.get("local_crops_number", 8)),
        lr=opt_c.get("learning_rate", 1e-4), min_lr=opt_c.get("min_lr", 1e-6),
        lr_scaling=opt_c.get("lr_scaling") or "none",
        weight_decay=opt_c.get("weight_decay", 0.04), weight_decay_end=opt_c.get("weight_decay_end", 0.4),
        warmup_epochs=opt_c.get("warmup_epochs", 5), freeze_last_layer_epochs=opt_c.get("freeze_last_layer_epochs", 3),
        layerwise_decay=opt_c.get("layerwise_decay", 0.9), patch_embed_lr_mult=opt_c.get("patch_embed_lr_mult", 0.2),
        adamw_beta1=opt_c.get("adamw_beta1", 0.9), adamw_beta2=opt_c.get("adamw_beta2", 0.999),
        dino_head_wd_multiplier=opt_c.get("dino_head_wd_multiplier", 1.0),
        ema_momentum=opt_c.get("ema_momentum", 0.996), ema_momentum_end=opt_c.get("ema_momentum_end", 1.0),
        teacher_temp_warmup=opt_c.get("teacher_temp_warmup", 0.04), teacher_temp=opt_c.get("teacher_temp", 0.07),
        teacher_temp_warmup_epochs=opt_c.get("teacher_temp_warmup_epochs", 10),
        max_epochs=train_c.get("max_epochs", 50), batch_size=data_c.get("batch_size", 128),
    )
    dm = MxIFDataModule(dataset_dir=data_c.get("dataset_dir"), batch_size=data_c.get("batch_size", 128),
                        num_workers=data_c.get("num_workers", 0), augmentation=aug,
                        input=data_c.get("input"), seed=cfg.get("experiment", {}).get("seed", 42),
                        patch_size=model_c.get("vit_patch_size", 16),
                        mask_ratio_min_max=model_c.get("mask_ratio_min_max", [0.1, 0.5]),
                        mask_sample_probability=model_c.get("mask_sample_probability", 0.5),
                        pin_memory=data_c.get("pin_memory", False))
    return learner, dm


def main(argv=None) -> int:
    import lightning.pytorch as L
    from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
    from lightning.pytorch.loggers import CSVLogger, TensorBoardLogger
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", "-c", required=True)
    ap.add_argument("--epochs", type=int)
    ap.add_argument("--batch-size", type=int, dest="batch_size")
    ap.add_argument("--accelerator", default=None, choices=["auto", "mps", "cuda", "cpu"])
    ap.add_argument("--precision", default=None, choices=["32-true", "bf16-mixed", "16-mixed"],
                    help="override training.precision (bf16-mixed helps on CUDA; measured no faster on MPS, which uses 32-true)")
    ap.add_argument("--out", help="parent folder of the run (default: experiment.out in the config, else 'runs')")
    ap.add_argument("--resume", help="checkpoint to resume from")
    ap.add_argument("--fast-dev-run", action="store_true")
    args = ap.parse_args(argv)
    cfg = yaml.safe_load(Path(args.config).read_text())
    if args.epochs:
        cfg.setdefault("training", {})["max_epochs"] = args.epochs
    if args.batch_size:
        cfg.setdefault("data", {})["batch_size"] = args.batch_size
    if args.out:
        cfg.setdefault("experiment", {})["out"] = args.out
    train_c = cfg.get("training", {})
    L.seed_everything(cfg.get("experiment", {}).get("seed", 42), workers=True)
    learner, dm = build(cfg)
    exp = cfg.get("experiment", {})
    out = Path(exp.get("out", "runs")) / exp.get("name", "kodiak_pretrain")   # runs/<name>; `experiment.out` or --out moves the parent
    out.mkdir(parents=True, exist_ok=True)
    (out / "resolved_config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
    acc = args.accelerator or ("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
    precision = args.precision or ("32-true" if acc == "cpu" else train_c.get("precision", "32-true"))
    if acc == "mps" and precision != "32-true":
        print(f"note: {precision} measured no faster than 32-true on MPS (no low-precision GEMM acceleration); using 32-true for exact numerics")
        precision = "32-true"
    val_every = train_c.get("check_val_every_n_epoch") or 0            # null / 0 = no validation at all (the default)
    ckpt_c = train_c.get("checkpoint", {})
    trainer = L.Trainer(
        max_epochs=train_c.get("max_epochs", 50), accelerator=acc, devices=1,
        precision=precision,
        gradient_clip_val=train_c.get("gradient_clip_val", 3.0),
        accumulate_grad_batches=train_c.get("accumulate_grad_batches", 1),
        check_val_every_n_epoch=val_every or 1, limit_val_batches=1.0 if val_every else 0.0, num_sanity_val_steps=0,
        default_root_dir=str(out), log_every_n_steps=10,
        logger=[CSVLogger(str(out), name="", version="logs"), TensorBoardLogger(str(out), name="", version="tensorboard")],
        callbacks=[ModelCheckpoint(dirpath=out / "checkpoints", save_last=ckpt_c.get("save_last", True),
                                   filename="kodiak-{epoch:03d}", every_n_epochs=ckpt_c.get("every_n_epochs") or 1,
                                   save_top_k=ckpt_c.get("save_top_k", 0)),
                   LearningRateMonitor(logging_interval="epoch")],
        fast_dev_run=args.fast_dev_run, enable_model_summary=False,
    )
    trainer.fit(learner, datamodule=dm, ckpt_path=args.resume)
    print(f"\nrun -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
