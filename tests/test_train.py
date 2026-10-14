"""mxseg.train: the shared training data (same data for every model), D4 flow transforms, config, and the manifest."""
import json

import numpy as np
import pytest
import torch

from mxseg.train.config import load_train_config
from mxseg.train.data import TrainData, TrainingPatches
from mxseg.train.flows import d4_arrays, d4_flows

cellpose = pytest.importorskip("cellpose")


def _blobs():
    import cv2
    lab = np.zeros((64, 64), np.int32)
    for i, (x, y) in enumerate([(12, 12), (40, 14), (20, 44), (48, 46)], 1):
        cv2.ellipse(lab, (x, y), (9, 6), 30 * i, 0, 360, i, -1)
    return lab


def test_d4_flow_transform_matches_recomputed_flows():
    from mxseg.train.flows import labels_to_targets
    lab = _blobs()
    base = labels_to_targets(lab)
    for k, fh, fv in [(1, False, False), (2, False, False), (3, True, False), (0, False, True), (1, True, True)]:
        (lab_t,) = d4_arrays(lab, k=k, fh=fh, fv=fv)
        recomputed = labels_to_targets(lab_t)                       # flows of the transformed labels
        transformed = d4_flows(base, k, fh, fv)                    # transformed flows of the original labels
        assert np.array_equal(transformed[0], recomputed[0])
        inside = recomputed[0] > 0.5
        assert np.abs(transformed[1:, inside] - recomputed[1:, inside]).max() < 0.05, (k, fh, fv)


class TagPatches(TrainingPatches):
    """A second adapter: proves adapters share keys and augmentation draws."""

    def prepare(self, key, image, labels, ignore, aug):
        return {"image": torch.from_numpy(image.astype(np.float32)), "n": torch.tensor(int(len(np.unique(labels)) - 1))}


def test_every_adapter_sees_the_same_data(synth, cfg, dataset):
    out, _ = dataset
    a = TrainData(cfg, out, TrainingPatches, batch_size=4, num_workers=0, augment=True, seed=3)
    b = TrainData(cfg, out, TagPatches, batch_size=4, num_workers=0, augment=True, seed=3)
    assert a.manifest()["train_keys"] == b.manifest()["train_keys"] and a.manifest()["val_keys"] == b.manifest()["val_keys"]
    seq_a = [(k, tuple(g.tolist())) for bt in a.train_dataloader() for k, g in zip(bt["key"], bt["aug"])]
    seq_b = [(k, tuple(g.tolist())) for bt in b.train_dataloader() for k, g in zip(bt["key"], bt["aug"])]
    assert seq_a == seq_b and len(seq_a) == (len(a.train) // 4) * 4                 # same keys, same D4 draws, same batch order
    assert len({g for _, g in seq_a}) > 1                                             # augmentation actually varies
    item = a.train[0]
    assert set(item) >= {"image", "labels", "ignore", "key", "aug"} and item["image"].shape[1:] == item["labels"].shape == item["ignore"].shape
    val_seq = [k for bt in a.val_dataloader() for k in bt["key"]]
    assert val_seq == a.val.keys and all(tuple(g.tolist()) == (0, 0, 0) for bt in a.val_dataloader() for g in bt["aug"])   # val: fixed order, no augmentation
    n_metric = sum(1 for _ in a.metric_samples())
    assert n_metric == len(a.val) and all(len(gt) == 4 for _, gt in a.metric_samples())
    a.write_manifest(out / "data_manifest.json")
    m = json.loads((out / "data_manifest.json").read_text())
    assert m["n_train"] == len(a.train) and m["seed"] == 3 and m["dataset_format"] == 2


def test_train_config_strict(tmp_path):
    p = tmp_path / "t.yaml"
    p.write_text("name: x\nepochs: 3\noptim: {lr: 0.0001}\n")
    c = load_train_config(p, epochs=5)
    assert c.epochs == 5 and c.optim.lr == 1e-4 and c.optim.warmup_epochs == 2 and c.model == "cellpose"
    p.write_text("name: x\nepoch: 3\n")
    with pytest.raises(Exception, match="unknown key"):
        load_train_config(p)


def test_train_config_dotted_override(tmp_path):
    from mxseg.train.config import load_train_config
    cfg_path = tmp_path / "t.yaml"
    cfg_path.write_text("name: x\noptim:\n  lr: 0.001\n")
    cfg = load_train_config(cfg_path, epochs=3, **{"cellpose.train_blocks": 4, "optim.lr": None, "cellpose.rdrop": 0})
    assert (cfg.epochs, cfg.cellpose.train_blocks, cfg.optim.lr, cfg.cellpose.rdrop) == (3, 4, 0.001, 0)   # None = keep YAML value
    from mxseg.cli import _params, build_parser
    a = build_parser().parse_args(["train", "-c", str(cfg_path), "--set", "optim.lr=1e-3", "--set", "val.every_n_epochs=1"])
    cfg = load_train_config(a.config, **_params(a.set))
    assert (cfg.optim.lr, cfg.val.every_n_epochs) == (1e-3, 1)


def test_shared_optimizer_protocol():
    """Every model builds its optimizer from the one recipe: the schedule multiplier, layer-wise groups on the U-Net's ImageNet
    stages, all-at-peak for a frozen ViT encoder, and no weight decay on 1-D parameters anywhere."""
    import math

    from mxseg.train.config import KodiakTrainConfig, OptimConfig, TrainConfig
    from mxseg.train.kodiak_module import KodiakSegModule
    from mxseg.train.module import lr_factor
    from mxseg.train.unet_module import UNetModule
    f = lr_factor(2, 20, 0.01)
    assert (f(0), f(1), f(2)) == (0.5, 1.0, 1.0) and f(10) < f(9)                         # warm-up, then cosine ...
    assert f(19) == pytest.approx(0.01 + 0.99 * 0.5 * (1 + math.cos(math.pi * 17 / 18))) and f(20) == pytest.approx(0.01)   # ... reaching 1 % right after the last epoch
    o = OptimConfig(lr=2e-3, layerwise_decay=0.75, weight_decay=0.05)
    u = UNetModule(TrainConfig(model="unet", optim=o), in_channels=3)
    groups = u.configure_optimizers()["optimizer"].param_groups                        # LambdaLR has already applied the epoch-0 warm-up
    assert {round(g["initial_lr"] / 2e-3, 6) for g in groups} == {round(0.75 ** k, 6) for k in range(6)}   # stem + 4 stages + head
    assert all(g["lr"] == pytest.approx(0.5 * g["initial_lr"]) for g in groups)                            # ... factor 0.5 at epoch 0
    assert all((p.ndim <= 1) == (g["weight_decay"] == 0.0) for g in groups for p in g["params"])
    k = KodiakSegModule(TrainConfig(model="kodiak", optim=o, kodiak=KodiakTrainConfig(checkpoint=None, train_blocks=0)), in_channels=3)
    assert {g["initial_lr"] for g in k.configure_optimizers()["optimizer"].param_groups} == {2e-3}   # frozen encoder: head at peak


def test_set_trainable_selects_top_blocks():
    from cellpose import models

    from mxseg.train.cellpose_module import set_trainable
    if not (models.MODEL_DIR / "cpsam_v2").exists():
        pytest.skip("cpsam_v2 weights not downloaded")
    net = models.CellposeModel(gpu=False, pretrained_model="cpsam_v2").net
    n_all = sum(p.numel() for p in net.parameters())
    fixed = sum(p.numel() for p in (net.W2, net.diam_mean, net.diam_labels))
    assert set_trainable(net, 24) == n_all - fixed                       # full fine-tune: everything cellpose trains
    n4 = set_trainable(net, 4)
    top = sum(p.numel() for m in (*net.encoder.blocks[-4:], net.encoder.neck, net.out) for p in m.parameters())
    assert n4 == top and not net.encoder.blocks[19].attn.qkv.weight.requires_grad and not net.encoder.pos_embed.requires_grad
    assert set_trainable(net, 0) == sum(p.numel() for p in net.out.parameters())          # encoder frozen, head only
    assert not any(p.requires_grad for p in net.encoder.parameters())


def test_hparams_are_loggable(tmp_path):
    """Loggers sanitize hparams with apply_to_collection, which rejects frozen dataclasses (seen with the real trainer)."""
    from lightning.pytorch.loggers import CSVLogger

    from mxseg.train.config import TrainConfig, as_dict
    hp = as_dict(TrainConfig())
    assert hp["cellpose"]["train_blocks"] == 24 and isinstance(hp["val"], dict)
    CSVLogger(tmp_path).log_hyperparams(hp)


def test_kept_weights_follow_the_monitored_loss():
    from mxdata.errors import ConfigError
    from mxseg.train.config import ValConfig
    from mxseg.train.module import is_better
    assert ValConfig().monitor == "val/loss" and ValConfig().mode == "min"           # one rule for every trained model
    assert ValConfig(monitor="val/f1").mode == "max"
    assert is_better(0.5, None, "min") and is_better(0.4, 0.5, "min") and not is_better(0.6, 0.5, "min")
    assert is_better(0.8, 0.7, "max") and not is_better(0.6, 0.7, "max")
    with pytest.raises(ConfigError):
        ValConfig(monitor="train/loss")


def test_metrics_due_every_n_and_final_epoch():
    from mxseg.train.cellpose_module import metrics_due
    assert [metrics_due(e, 7, 3) for e in range(7)] == [False, False, True, False, False, True, True]
    assert metrics_due(0, 1, 5)                      # a 1-epoch run still scores its final epoch


def test_load_history_splits_lightning_csv(tmp_path):
    from mxseg.train.history import list_runs, load_history, runs_table
    run = tmp_path / "r1"
    (run / "logs").mkdir(parents=True)
    (run / "logs" / "metrics.csv").write_text(
        "epoch,lr-AdamW,step,train/loss_epoch,train/loss_step,val/loss,val/f1\n"
        ",1e-6,0,,,,\n"
        "0,,9,,1.4,,\n"
        "0,,19,,1.2,,\n"
        "0,,19,,,0.9,\n"            # val loss (no instance metrics this epoch)
        "0,,19,1.3,,,\n"
        "1,2e-6,39,,1.0,,\n"
        "1,,39,,,0.8,0.7\n"
        "1,,39,1.05,,,\n")
    (run / "resolved_config.yaml").write_text("epochs: 2\nbatch_size: 4\ncellpose:\n  train_blocks: 0\n  lr: 1.0e-05\n")
    h = load_history(run)
    assert list(h.steps["train/loss"]) == [1.4, 1.2, 1.0] and list(h.steps["epoch"]) == [0, 0, 1]
    assert list(h.epochs.columns) == ["epoch", "lr", "train/loss", "val/loss", "val/f1"]
    assert list(h.epochs["train/loss"]) == [1.3, 1.05] and list(h.epochs["val/loss"]) == [0.9, 0.8]
    assert list(h.epochs["lr"]) == [1e-6, 2e-6]                       # lr rows carry no epoch: attached to the epoch starting there
    assert list(h.val_metrics["epoch"]) == [1] and h.summary == {}
    assert h.monitor == "val/loss" and h.best_epoch == 1                  # the kept epoch: minimum validation loss
    pre = tmp_path / "pretrain"                                              # a KODIAK pretraining run: model is a section, not a name
    (pre / "logs").mkdir(parents=True)
    (pre / "logs" / "metrics.csv").write_text("epoch,step,train/loss_step\n0,9,1.0\n")
    (pre / "resolved_config.yaml").write_text("model:\n  mask_padded_keys: true\nepochs: 100\n")
    t = runs_table(list_runs(tmp_path))
    assert list(t.index) == ["r1"]                                          # the pretraining run is skipped, not a crash
    assert t.loc["r1", "best_epoch"] == 1 and t.loc["r1", "epochs_done"] == 2 and not t.loc["r1", "finished"]
    assert t.loc["r1", "best_val_loss"] == 0.8 and t.loc["r1", "best_val_f1"] == 0.7 and t.loc["r1", "max_val_f1"] == 0.7
    assert t.loc["r1", "variant"] == 0 and t.loc["r1", "model"] == "cellpose"
