"""KODIAK port: the method's pieces fit together at tiny scale, and the shipped config builds the real learner.

The method itself is ported verbatim from the reference implementation, except for compute-only shortcuts (the
masked-token decoder + prototype head, vectorized mask indexing, foreach EMA) that the equivalence test below pins to
the originals; the other tests only guard the wiring — model forward passes, multi-crop transform geometry, block
masking + collate, losses, and the YAML -> learner/datamodule mapping (weights and real data are optional, skipped
when absent).
"""
from pathlib import Path

import pytest
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "kodiak_pretrain.yaml"

TINY = dict(img_size=64, patch_size=16, embed_dim=32, depth=1, num_heads=2, decoder_embed_dim=16,
            decoder_depth=1, decoder_num_heads=2, num_prototypes=8, projector_dim=16, num_storage_tokens=4)
N = (64 // 16) ** 2                                                  # patches per tiny image


def _tiny():
    from kodiak import Kodiak
    return Kodiak(**TINY)


def _mask(b):
    m = torch.zeros(b, N, dtype=torch.bool)
    m[:, :4] = True
    return m


def test_tiny_forward_shapes():
    m = _tiny()
    masks = [_mask(2), _mask(2)]
    masks[1][1, 4:7] = True                                          # views with different masked-token counts
    outs, teacher_logits, s_cls, t_cls, t_patch = m(torch.rand(2, 3, 64, 64), torch.rand(2, 3, 64, 64), masks)
    assert [o.shape for o in outs] == [(int(mk.sum()), TINY["num_prototypes"]) for mk in masks]   # (M_v, K): masked positions only
    assert teacher_logits.shape == (2, N, TINY["num_prototypes"]) and t_cls.shape == (2, TINY["embed_dim"])
    assert s_cls.shape == (4, TINY["embed_dim"])                     # B x views
    assert all(torch.isfinite(t).all() for t in (*outs, teacher_logits, s_cls, t_cls, t_patch))


def test_tiny_multicrop_and_ema_teacher():
    m = _tiny()
    for p in m.teacher_encoder.parameters():
        assert not p.requires_grad                                   # teacher is EMA-only, never trained directly
    out = m.forward_multicrop(global_crops=[torch.rand(2, 3, 64, 64), torch.rand(2, 3, 64, 64)],
                              local_crops=[torch.rand(2, 3, 32, 32) for _ in range(3)],
                              global_masks=[_mask(2), _mask(2)])
    assert len(out["student_patch_outputs"]) == 2 and out["student_patch_outputs"][0].shape == (8, TINY["num_prototypes"])   # M = 2 rows x 4
    assert [t.shape for t in out["teacher_patch_targets"]] == [s.shape for s in out["student_patch_outputs"]]
    assert out["teacher_patch_logits"].shape == (2, 2, N, TINY["num_prototypes"])                  # all tokens: center + metrics
    assert out["teacher_global_cls"].shape == (2, 2, TINY["embed_dim"])
    assert out["student_local_cls"].shape == (3, 2, TINY["embed_dim"])
    with torch.no_grad():
        for p in m.student_encoder.parameters():
            p.add_(torch.randn_like(p))                              # push student away, then EMA must move the teacher
    before = [p.clone() for p in m.teacher_encoder.parameters()]
    m.update_teacher(momentum=0.5)
    assert any(not torch.allclose(a, b) for a, b in zip(m.teacher_encoder.parameters(), before))


def test_transform_and_collate_geometry():
    from kodiak import MaskingGenerator, MultiCropKodiakCollate, MultiCropPretrainingTransform
    aug = yaml.safe_load(CONFIG.read_text())["data"]["augmentation"]
    tf = MultiCropPretrainingTransform(aug)
    out = tf(torch.rand(3, 256, 256))
    assert len(out["global_crops"]) == 2 and out["global_crops"][0].shape == (3, 256, 256)
    assert len(out["local_crops"]) == aug["local_crops_number"] and out["local_crops"][0].shape == (3, 112, 112)
    grid = 256 // 16
    collate = MultiCropKodiakCollate(mask_generator=MaskingGenerator((grid, grid), max_num_patches=128),
                                     mask_ratio_tuple=(0.1, 0.5), mask_probability=0.5, num_patches_total=grid * grid)
    batch = collate([tf(torch.rand(3, 256, 256)) for _ in range(2)])
    assert len(batch["global_crops"]) == 2 and batch["global_crops"][0].shape == (2, 3, 256, 256)
    assert len(batch["local_crops"]) == aug["local_crops_number"] and batch["local_crops"][0].shape == (2, 3, 112, 112)
    assert len(batch["masks"]) == 2 and batch["masks"][0].shape == (2, grid * grid) and batch["masks"][0].dtype == torch.bool


def test_losses_finite_at_tiny_scale():
    from kodiak import KoLeoLoss, MaskLoss, MultiCropPrototypeCLSLoss
    K, D = 8, 16
    s, t, mk = torch.randn(2, N, K), torch.randn(2, N, K), _mask(2)
    ml = MaskLoss(num_prototypes=K)(s, t, mk)
    assert torch.equal(MaskLoss(num_prototypes=K)(s[mk], t[mk]), ml)             # the gathered (M, K) layout the model hands over
    assert MaskLoss(num_prototypes=K)(s.new_zeros(0, K), t.new_zeros(0, K)) == 0   # nothing masked -> zero loss
    kl = KoLeoLoss()(torch.randn(4, D))
    pl, metrics = MultiCropPrototypeCLSLoss(embed_dim=D, num_prototypes=K)(
        torch.randn(2, 2, D), torch.randn(5, 2, D))                  # teacher (2 global, B, D); student (2+3 crops, B, D)
    for v in (ml, kl, pl):
        assert torch.isfinite(v), v
    assert isinstance(metrics, dict)


def test_config_builds_learner_and_datamodule():
    from kodiak.pretrain import build
    cfg = yaml.safe_load(CONFIG.read_text())
    if not (ROOT / cfg["model"]["pretrained_path"]).exists():
        cfg["model"]["pretrained_path"] = None                       # weights are optional for the wiring test
    learner, dm = build(cfg)
    assert learner.model.prototype_layer.layer.weight.shape[0] == 128
    assert learner.model.student_encoder.patch_embed.proj.weight.shape[-1] == 16
    assert dm.collate.mask_probability == pytest.approx(0.5)
    assert dm.transform is not None and len(dm.collate.mask_ratio_tuple) == 2
    assert learner.model.student_encoder.mask_padded_keys is True and learner.hparams["mask_padded_keys"] is True


def test_masked_student_attends_over_real_visible_tokens_only():
    """mask_padded_keys (the default): a padded batch row equals the same row encoded alone from its true visible
    tokens; False keeps the reference behaviour, where the padded copies of one visible token act as extra keys."""
    from kodiak import Kodiak
    torch.manual_seed(0)
    x = torch.rand(3, 3, 64, 64)
    mask = torch.zeros(3, N, dtype=torch.bool)
    mask[0, :8], mask[1, 2:5] = True, True                                    # 50 %, 19 % and 0 % masked rows
    for flag in (True, False):
        m = Kodiak(**TINY, mask_padded_keys=flag).eval()
        with torch.no_grad():                                                 # DINOv3 inits LayerScale at 1e-5, which would hide
            for blk in m.student_encoder.blocks:                              # attention behind the residual; give it full weight
                blk.ls1.gamma.fill_(1.0), blk.ls2.gamma.fill_(1.0)
        vis, _ = m._mask_to_indices(mask)
        cls_b, tok_b = m.student_encoder(x, vis, mask)                        # the padded batch
        for b in range(3):
            n_vis = N - int(mask[b].sum())
            cls_1, tok_1 = m.student_encoder(x[b:b + 1], vis[b:b + 1, :n_vis], mask[b:b + 1])   # alone: nothing to pad
            same = (torch.allclose(cls_b[b], cls_1[0], rtol=1e-4, atol=1e-6)
                    and torch.allclose(tok_b[b, :n_vis], tok_1[0], rtol=1e-4, atol=1e-6))
            assert same == (flag or n_vis == N), (flag, b)                    # reference: only the unpadded row agrees
    # In training, the masked path's stochastic depth is DINOv3's sample-subset index_add in per-sample-factor form:
    # the same draws give the same values.
    m = Kodiak(**TINY, drop_path_rate=0.5).train()
    blk, t = m.student_encoder.blocks[0], torch.rand(6, 5 + N, TINY["embed_dim"])
    rope = m.student_encoder.rope_embed(H=4, W=4)
    torch.manual_seed(3)
    ref = blk(t, rope)
    torch.manual_seed(3)
    new = blk(t, rope, key_padding_mask=torch.ones(6, 1, 1, 5 + N, dtype=torch.bool))
    assert torch.allclose(ref, new, atol=1e-6)


def test_dataset_stream_on_real_data():
    from mxdata import DEFAULT_DATASET
    if not Path(DEFAULT_DATASET).exists():
        pytest.skip(f"{DEFAULT_DATASET} not available")
    from kodiak import MxIFDataModule
    aug = yaml.safe_load(CONFIG.read_text())["data"]["augmentation"]
    data_c = yaml.safe_load(CONFIG.read_text())["data"]
    dm = MxIFDataModule(batch_size=2, num_workers=0, augmentation=aug, input=data_c["input"])
    dm.setup()
    assert len(dm.train_set) > len(dm.val_set) > 0
    batch = next(iter(dm.train_dataloader()))
    assert batch["global_crops"][0].shape == (2, 3, 256, 256) and batch["green"].shape == (2,)
    assert all(g.isfinite().all() for g in batch["global_crops"])


def test_channel_patches_draw_green_per_sample(cfg, dataset):
    """The SSL stream: fixed red / blue markers, green drawn per sample from a weighted set or blank, on the stored stacks."""
    from kodiak.data import ChannelPatches
    from kodiak.transforms import make_transform
    out, _ = dataset
    ds = ChannelPatches(out, "train", red="PanCK", blue="DAPI", green={"PanCK": 0.5, "blank": 0.5}, seed=0)
    x = ds[0]
    assert x.shape == (3, 64, 64) and x.dtype == torch.float32 and 0.0 <= x.min() <= x.max() <= 1.0
    greens = {ds.green_index(ds.draw_green()) for _ in range(40)}
    assert greens == {-1, ds.markers.index("PanCK")}                                # both outcomes occur; -1 = blank
    blank = ChannelPatches(out, "train", red="PanCK", blue="DAPI", green=None)
    assert blank[0][1].max() == 0 and "G drawn per sample (blank 1.00)" in blank.describe()
    with pytest.raises(KeyError, match="NOPE"):
        ChannelPatches(out, "train", red="NOPE", blue="DAPI")
    aug = {"photometric": "mxif", "global_crops_size": 64, "local_crops_size": 32, "local_crops_number": 3, "gain": 0.3, "gamma": [0.7, 1.4], "blur_p": 0.5}
    views = ChannelPatches(out, "train", red="PanCK", blue="DAPI", green={"blank": 1.0}, transform=make_transform(aug))[0]
    assert [v.shape for v in views["global_crops"]] == [(3, 64, 64)] * 2 and [v.shape for v in views["local_crops"]] == [(3, 32, 32)] * 3
    assert views["green"] == -1 and all(v.isfinite().all() for v in views["global_crops"] + views["local_crops"])
    with pytest.raises(ValueError, match="photometric"):
        make_transform({"photometric": "nope"})


# ---- the segmenter (mxseg train model: kodiak) ------------------------------------------------------------------------

SEG_TINY = {"patch_size": 16, "embed_dim": 32, "depth": 1, "num_heads": 2, "num_storage_tokens": 4, "head_channels": 32}


def test_segmenter_forward_freeze_and_roundtrip(tmp_path):
    from mxseg.baselines.kodiak import build_segmenter, load_segmenter, save_segmenter, set_trainable
    net = build_segmenter(SEG_TINY)
    out = net(torch.rand(2, 1, 64, 64))
    assert out.shape == (2, 3, 64, 64) and torch.isfinite(out).all()          # x16 head undoes the /16 patch grid
    head_n = sum(p.numel() for p in net.head.parameters())
    assert set_trainable(net, 0) == head_n                                    # frozen encoder: only the head trains
    assert not any(p.requires_grad for p in net.backbone.parameters())
    assert set_trainable(net, 1) == sum(p.numel() for p in net.parameters())  # depth 1: full fine-tune
    with pytest.raises(ValueError, match="train_blocks"):
        set_trainable(net, 5)
    save_segmenter(tmp_path / "w.pt", net, meta={"min_area": 5})
    net2, arch, meta = load_segmenter(tmp_path / "w.pt")
    assert arch["embed_dim"] == 32 and meta["min_area"] == 5
    x = torch.rand(1, 1, 64, 64)
    assert torch.allclose(net(x), net2(x), atol=1e-6)
    with pytest.raises(ValueError, match="not a mxseg KODIAK segmenter"):
        (tmp_path / "bad.pt").write_bytes(b"")
        torch.save({"format": 0}, tmp_path / "bad.pt")
        load_segmenter(tmp_path / "bad.pt")


def test_encoder_weights_kodiak_teacher_dinov3_and_refusal(tmp_path):
    from kodiak import Kodiak
    from mxseg.baselines.kodiak import build_segmenter, load_encoder_weights
    ck = tmp_path / "kodiak.ckpt"
    teacher = Kodiak(**TINY).teacher_encoder.state_dict()                     # a KODIAK pretraining checkpoint, as Lightning saves it
    torch.save({"state_dict": {f"model.teacher_encoder.{k}": v for k, v in teacher.items()}}, ck)
    net = build_segmenter(SEG_TINY)
    assert load_encoder_weights(net.backbone, ck) == "kodiak-teacher"
    assert torch.equal(dict(net.backbone.state_dict())["norm.weight"], teacher["norm.weight"])
    with pytest.raises(ValueError, match="covers"):                           # wrong architecture: refused whole, never half-loaded
        load_encoder_weights(build_segmenter({}).backbone, ck)
    with pytest.raises(FileNotFoundError, match="pretrain first"):
        load_encoder_weights(net.backbone, tmp_path / "missing.ckpt")
    pth = ROOT / "weights" / "dinov3_vits16_pretrain_lvd1689m-08c60483.pth"
    if pth.exists():
        full = build_segmenter({})
        assert load_encoder_weights(full.backbone, pth) == "dinov3"


def test_kodiak_train_config_and_registry():
    from mxseg.baselines import BASELINES, from_config
    from mxseg.train.config import load_train_config
    cfg = load_train_config(ROOT / "configs" / "train_kodiak.yaml")
    assert cfg.model == "kodiak" and cfg.epochs == 20 and cfg.kodiak.train_blocks == 12   # the shared 20-epoch budget
    assert cfg.kodiak.checkpoint.endswith("last.ckpt")
    assert cfg.batch_size * cfg.grad_accumulation == 16                                    # the shared effective batch
    assert cfg.kodiak.encoder_upsample == 2 and cfg.optim.lr == pytest.approx(2e-3)        # the measured winning recipe, shared by all
    assert cfg.optim.layerwise_decay == 0.75 and cfg.optim.warmup_epochs == 2
    assert cfg.kodiak.objective == "flows"                                                 # the protocol of record
    assert cfg.input.channels == ("LAP2", None, "DAPI")                                      # envelope, blank, nuclei
    assert "kodiak_mxif" in cfg.kodiak.checkpoint
    assert cfg.kodiak.bottleneck == "multi"
    assert "kodiak" in BASELINES
    b, name = from_config(ROOT / "configs" / "baselines" / "kodiak_flows.yaml")
    assert name == "kodiak_flows" and b.name == "kodiak" and b.params()["weights"].endswith("weights.pt")


def test_kodiak_train_and_eval_end_to_end(synth, cfg, dataset, tmp_path):
    import json

    import numpy as np

    from mxdata import save_resolved as save_data_config
    from mxdata.data import PatchSource
    from mxseg import get_baseline, run_baseline
    from mxseg.inputs import InputConfig
    from mxseg.train import train
    from mxseg.train.config import KodiakTrainConfig, TrainConfig, ValConfig
    from mxseg.train.run import eval_command
    out, _ = dataset
    data_yaml = tmp_path / "data.yaml"
    save_data_config(cfg, data_yaml)
    tcfg = TrainConfig(name="kodiak_test", data=str(data_yaml), dataset_dir=str(out), model="kodiak",
                       input=InputConfig(red="PanCK", green=None, blue="DAPI"),    # the synthetic slide has two markers
                       kodiak=KodiakTrainConfig(checkpoint=None, train_blocks=0, min_area=5),
                       epochs=1, batch_size=4, accelerator="cpu", val=ValConfig(every_n_epochs=1, n_metric_patches=2),
                       out=str(tmp_path / "runs"))
    run_dir = train(tcfg)
    assert (run_dir / "weights.pt").is_file() and (run_dir / "data_manifest.json").is_file()
    summary = json.loads((run_dir / "summary.json").read_text())
    assert summary["model"] == "kodiak" and summary["variant"] == "scratch" and "val/f1" in summary["best"] and "val/loss" in summary["best"]
    assert "--baseline kodiak --param weights=" in eval_command(tcfg, run_dir)
    b = get_baseline("kodiak", weights=str(run_dir / "weights.pt"), device="cpu")
    src = PatchSource(out, split="val", kind="labeled", slides=cfg.slides)
    rs = src.load(src.keys[0])
    from mxseg.inputs import select_markers
    stack = select_markers(rs.image, src.markers, b.input_markers())                # the model's markers out of the stored stack
    pred = b.predict(stack, slide=rs.meta["slide"])
    assert pred.shape == rs.labels.shape and pred.dtype == np.int32 and b.slug() == "kodiak_kodiak_test"
    run_baseline(b, cfg, split="val", dataset_dir=out, results_dir=tmp_path / "results", n=2, progress=False, store_predictions=False,
                 slide_root=synth["root"])
    assert (tmp_path / "results" / "val" / "kodiak_kodiak_test" / "metrics.json").is_file()


def test_finetune_protocol_layerwise_decay_groups():
    from mxseg.baselines.kodiak import build_segmenter, layerwise_param_groups, set_trainable
    net = build_segmenter(SEG_TINY)                                           # depth 1: layers embed(0) < block(1) < head+norm(2)
    n_full = set_trainable(net, 1)
    groups = layerwise_param_groups(net, lr=1e-3, layerwise_decay=0.5, weight_decay=0.05)
    assert {round(g["lr"], 10) for g in groups} == {1e-3, 5e-4, 2.5e-4}       # one decay step per layer, head at peak
    for g in groups:                                                          # 1-D params (biases, norms, tokens): no weight decay
        assert all((p.ndim <= 1) == (g["weight_decay"] == 0.0) for p in g["params"])
    assert sum(p.numel() for g in groups for p in g["params"]) == n_full
    set_trainable(net, 0)                                                     # frozen encoder: only head groups, all at peak lr
    groups = layerwise_param_groups(net, 1e-3, 0.5, 0.05)
    assert all(g["lr"] == pytest.approx(1e-3) for g in groups)
    assert sum(p.numel() for g in groups for p in g["params"]) == sum(p.numel() for p in net.head.parameters())


def test_optimized_paths_match_reference_implementations():
    """The vectorized mask indexing and foreach EMA must reproduce the original per-row / per-parameter loops exactly."""
    import copy
    m = _tiny()
    g = torch.Generator().manual_seed(3)
    mask = torch.zeros(6, N, dtype=torch.bool)
    for b, c in enumerate([0, 3, N // 2, 0, 5, 1]):                           # ragged, incl. all-False rows
        mask[b, torch.randperm(N, generator=g)[:c]] = True
    vis, msk = m._mask_to_indices(mask)
    B, n = mask.shape                                                         # the original loop, verbatim
    num_masked = mask.sum(dim=1)
    num_visible = n - num_masked
    sorted_indices = torch.argsort(mask.int(), dim=1)
    vis_ref = torch.zeros(B, int(num_visible.max()), dtype=torch.long)
    msk_ref = torch.zeros(B, int(num_masked.max()), dtype=torch.long)
    for b in range(B):
        n_vis, n_msk = int(num_visible[b]), int(num_masked[b])
        vis_ref[b, :n_vis] = sorted_indices[b, :n_vis]
        if n_vis < vis_ref.shape[1]:
            vis_ref[b, n_vis:] = vis_ref[b, n_vis - 1]
        msk_ref[b, :n_msk] = sorted_indices[b, n - n_msk:]
        if n_msk < msk_ref.shape[1]:
            msk_ref[b, n_msk:] = msk_ref[b, n_msk - 1]
    assert torch.equal(vis, vis_ref) and torch.equal(msk, msk_ref)

    with torch.no_grad():
        for p in m.student_encoder.parameters():
            p.add_(torch.randn_like(p) * 0.1)
    ref = copy.deepcopy(m)
    m.update_teacher(momentum=0.913)
    with torch.no_grad():                                                     # the original per-parameter EMA, verbatim
        for ps, pt in zip(ref.student_encoder.parameters(), ref.teacher_encoder.parameters()):
            pt.data.mul_(0.913).add_(ps.data, alpha=1.0 - 0.913)
        for ps, pt in zip(ref.student_patch_head.parameters(), ref.teacher_patch_head.parameters()):
            pt.data.mul_(0.913).add_(ps.data, alpha=1.0 - 0.913)
    assert all(torch.equal(x, y) for x, y in zip(m.teacher_encoder.parameters(), ref.teacher_encoder.parameters()))
    assert all(torch.equal(x, y) for x, y in zip(m.teacher_patch_head.parameters(), ref.teacher_patch_head.parameters()))

    # The masked-only decoder + prototype head must reproduce the original all-row, all-token path: the same logits
    # at the masked positions and the same gradients (decoder, adapter, projector, prototypes, and into the encoder).
    import torch.nn.functional as F

    from kodiak import MaskLoss
    m = _tiny()
    vis, msk = m._mask_to_indices(mask)                                       # the ragged mask above, incl. all-False rows
    _, s_vis = m.student_encoder(torch.rand(6, 3, 64, 64), vis, mask)
    dec = m.decoder(s_vis, vis, msk, num_patches=N, grid_h=4, grid_w=4)      # the original: every row, every token
    ref = m.prototype_layer(F.normalize(m.student_patch_head(m.student_dec_adapter(dec)), p=2, dim=-1))[mask]
    new = m._masked_patch_logits(s_vis, vis, msk, mask, 4, 4)
    assert new.shape == (int(mask.sum()), TINY["num_prototypes"]) and torch.allclose(new, ref, atol=1e-6)
    targets = torch.randn_like(ref)
    params = [p for mod in (m.decoder, m.student_dec_adapter, m.student_patch_head, m.prototype_layer) for p in mod.parameters()]
    g_ref = torch.autograd.grad(MaskLoss(TINY["num_prototypes"])(ref, targets), params + [s_vis], retain_graph=True)
    g_new = torch.autograd.grad(MaskLoss(TINY["num_prototypes"])(new, targets), params + [s_vis])
    assert all(torch.allclose(a, b, atol=1e-6) for a, b in zip(g_ref, g_new))


def test_learner_step_multicrop_and_legacy():
    """The learner's loss wiring over the (M, K) masked-token outputs, in both batch formats, incl. an unmasked row."""
    from kodiak import KodiakLearner
    learner = KodiakLearner(model=_tiny(), n_local_crops=3, batch_size=2, lr_scaling="none")
    masks = [_mask(2), _mask(2)]
    masks[1][1] = False                                                       # a row the mask loss never reads
    batch = {"global_crops": [torch.rand(2, 3, 64, 64) for _ in range(2)],
             "local_crops": [torch.rand(2, 3, 32, 32) for _ in range(3)], "masks": masks}
    total, mask_loss, koleo, proto, teacher_logits, metrics = learner.shared_step(batch, True)
    assert torch.isfinite(total) and mask_loss > 0 and teacher_logits.shape == (2, N, TINY["num_prototypes"])
    total.backward()
    assert learner.model.prototype_layer.layer.weight.grad is not None
    legacy = {"teacher_images": torch.rand(2, 3, 64, 64), "student_images": torch.rand(2, 3, 64, 64), "masks": masks}
    total, mask_loss, *_ = learner.shared_step(legacy, True)
    assert torch.isfinite(total) and mask_loss > 0


def test_marker_composite_and_colors():
    import numpy as np

    from mxdata.data.composite import compose_rgb
    colors = {"DAPI": "#0000FF", "PanCK": "#FFFFFF", "CD45": "#FFC0CB"}
    stack = np.zeros((3, 8, 8), np.uint16)
    stack[0, :4] = 1000                                        # DAPI top half
    stack[1, 4:] = 800                                         # PanCK bottom half
    norms = {"DAPI": (0, 1000), "PanCK": (0, 800), "CD45": (0, 500)}
    rgb = compose_rgb(stack, ["DAPI", "PanCK", "CD45"], colors, norms)
    assert rgb.shape == (3, 8, 8) and rgb.dtype == np.float32
    assert np.allclose(rgb[:, 0, 0], [0, 0, 1])                # pure DAPI -> blue
    assert np.allclose(rgb[:, 7, 0], [1, 1, 1])                # pure PanCK -> white
    assert rgb.max() <= 1.0 and rgb.min() >= 0.0
    empty = compose_rgb(np.zeros((1, 8, 8), np.uint16), ["CD45"], colors, {"CD45": (0, 500)})
    assert empty.max() == 0.0                                  # slide-level stats keep marker-empty patches dark
    from mxdata.data.composite import render_channels
    x = render_channels(stack, ["DAPI", "PanCK", "CD45"], norms, ("PanCK", None, "dapi"))    # a marker per channel, case-insensitive
    assert np.allclose(x[:, 0, 0], [0, 0, 1]) and np.allclose(x[:, 7, 0], [1, 0, 0])
    with pytest.raises(KeyError, match="NOPE"):
        render_channels(stack, ["DAPI", "PanCK", "CD45"], norms, ("NOPE", None, "DAPI"))
