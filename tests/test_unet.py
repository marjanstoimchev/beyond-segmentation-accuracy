"""3-class U-Net: targets, watershed instances, masked loss, same-data adapter, and train -> weights -> eval end to end."""
import json

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from mxseg.baselines.unet import instances_from_probs
from mxseg.train.threeclass import labels_to_classes
from mxseg.train.unet_module import masked_class_loss


def _touching():
    lab = np.zeros((64, 64), np.int32)
    yy, xx = np.mgrid[:64, :64]
    lab[(yy - 30) ** 2 + (xx - 24) ** 2 <= 100] = 1                          # r = 10, x in 14..34 at y = 30
    lab[((yy - 30) ** 2 + (xx - 42) ** 2 <= 100) & (lab == 0)] = 2           # touches nucleus 1 (x >= 35)
    lab[(yy - 10) ** 2 + (xx - 52) ** 2 <= 25] = 3                           # small isolated nucleus
    return lab


def test_labels_to_classes_boundary_and_contact():
    lab = _touching()
    cls = labels_to_classes(lab, boundary_px=2)
    assert cls.shape == lab.shape and cls.dtype == np.uint8 and set(np.unique(cls)) == {0, 1, 2}
    assert (cls[lab == 0] == 0).all() and cls[30, 24] == 1                    # background stays 0, centre is interior
    assert cls[30, 33] == 2 and cls[30, 34] == 2 and cls[30, 35] == 2         # contact line between 1 and 2 is boundary
    assert cls[30, 14] == 2 and cls[30, 15] == 2 and cls[30, 16] == 1         # 2 px of boundary at the free edge, then interior
    assert (labels_to_classes(lab, boundary_px=1) == 2).sum() < (cls == 2).sum()


def test_instances_from_probs_splits_touching_nuclei():
    lab = _touching()
    cls = labels_to_classes(lab, 2)
    prob = np.stack([(cls == c).astype(np.float32) for c in range(3)]) * 0.9 + 0.05
    inst = instances_from_probs(prob, min_area=15)
    assert inst.dtype == np.int32 and inst.max() == 3
    for g in (1, 2, 3):
        m = lab == g
        ids, counts = np.unique(inst[m], return_counts=True)
        pid = ids[counts.argmax()]
        assert pid > 0 and ((m & (inst == pid)).sum() / (m | (inst == pid)).sum()) > 0.8
    empty = np.stack([np.ones((8, 8)), np.zeros((8, 8)), np.zeros((8, 8))]).astype(np.float32)
    assert instances_from_probs(empty).max() == 0


def test_masked_class_loss_ignores_pixels():
    torch.manual_seed(0)
    target = torch.randint(0, 3, (2, 16, 16))
    logits = F.one_hot(target, 3).permute(0, 3, 1, 2).float() * 10
    none = torch.zeros(2, 16, 16, dtype=torch.bool)
    good = masked_class_loss(logits, target, none)
    assert good["ce"] < 1e-3 and good["dice"] < 0.01
    ign = none.clone()
    ign[:, :8] = True
    bad = logits.clone()
    bad[:, :, :8] = torch.randn_like(bad[:, :, :8]) * 5
    assert torch.allclose(masked_class_loss(logits, target, ign)["loss"], masked_class_loss(bad, target, ign)["loss"])   # ignored pixels: no effect
    assert masked_class_loss(bad, target, none)["loss"] > good["loss"] + 0.5


def test_unet_adapter_sees_same_items(synth, cfg, dataset):
    from mxseg.train.data import TrainData, TrainingPatches, UNetPatches
    out, _ = dataset
    a = TrainData(cfg, out, TrainingPatches, batch_size=4, num_workers=0, augment=True, seed=5)
    b = TrainData(cfg, out, UNetPatches, batch_size=4, num_workers=0, augment=True, seed=5, boundary_px=2)
    seq = lambda d: [(k, tuple(g.tolist())) for bt in d.train_dataloader() for k, g in zip(bt["key"], bt["aug"])]   # noqa: E731
    assert seq(a) == seq(b)                                                     # same keys, same D4 draws, same order
    item = b.train[0]
    assert item["image"].dtype == torch.float32 and item["target"].dtype == torch.int64
    assert set(item["target"].unique().tolist()) <= {0, 1, 2} and item["image"].shape[1:] == item["target"].shape == item["ignore"].shape


def test_unet_train_and_eval_end_to_end(synth, cfg, dataset, tmp_path):
    from mxdata import save_resolved as save_data_config
    from mxdata.data import PatchSource
    from mxseg import get_baseline, run_baseline
    from mxseg.inputs import InputConfig
    from mxseg.train import train
    from mxseg.train.config import TrainConfig, UNetTrainConfig, ValConfig
    from mxseg.train.run import eval_command
    out, _ = dataset
    data_yaml = tmp_path / "data.yaml"
    save_data_config(cfg, data_yaml)
    tcfg = TrainConfig(name="unet_test", data=str(data_yaml), dataset_dir=str(out), model="unet", input=InputConfig(red="PanCK", green=None, blue="DAPI"),
                       unet=UNetTrainConfig(encoder="resnet18", encoder_weights=None, min_area=5), epochs=2, batch_size=4, accelerator="cpu",
                       val=ValConfig(every_n_epochs=1, n_metric_patches=4), out=str(tmp_path / "runs"))
    run_dir = train(tcfg)
    assert (run_dir / "weights.pt").is_file() and (run_dir / "data_manifest.json").is_file()
    summary = json.loads((run_dir / "summary.json").read_text())
    assert summary["model"] == "unet" and "val/f1" in summary["best"] and "val/loss" in summary["best"] and summary["variant"] == "resnet18/None"
    assert "--baseline unet --param weights=" in eval_command(tcfg, run_dir)
    b = get_baseline("unet", weights=str(run_dir / "weights.pt"), device="cpu")
    src = PatchSource(out, split="val", kind="labeled", slides=cfg.slides)
    rs = src.load(src.keys[0])
    from mxseg.inputs import select_markers
    stack = select_markers(rs.image, src.markers, b.input_markers())                # the model's markers out of the stored stack
    pred = b.predict(stack, slide=rs.meta["slide"])
    assert pred.shape == rs.labels.shape and pred.dtype == np.int32 and b.slug() == "unet_unet_test"
    run_baseline(b, cfg, split="val", dataset_dir=out, results_dir=tmp_path / "results", n=3, progress=False, store_predictions=False, slide_root=synth["root"])
    from mxseg.train.run import export_weights
    w = export_weights(run_dir, "last")                                      # rebuild weights.pt from a checkpoint
    assert w == run_dir / "weights.pt" and get_baseline("unet", weights=str(w), device="cpu").predict(stack).shape == rs.labels.shape
    with pytest.raises(FileNotFoundError):
        export_weights(run_dir, "nope.ckpt")
    assert (tmp_path / "results" / "val" / "unet_unet_test" / "metrics.json").is_file()
