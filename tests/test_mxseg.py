"""mxseg: instance metrics on synthetic label maps, the evaluation runner + results store on the synthetic dataset."""
import json

import cv2
import numpy as np
import pandas as pd
import pytest

from mxseg import aggregate, compare_results, instance_outcomes, run_baseline, score_patch
from mxseg.baselines import Baseline, get_baseline, register
from mxseg.cli import main


def _blobs(seed=0, n=6):
    lab = np.zeros((64, 64), np.int32)
    rng = np.random.default_rng(seed)
    for i in range(1, n + 1):
        y, x = 8 + (i - 1) // 3 * 24, 8 + (i - 1) % 3 * 20
        cv2.circle(lab, (x + int(rng.integers(0, 3)), y + int(rng.integers(0, 3))), 6, i, -1)
    return lab


def test_score_patch_perfect_and_degraded():
    gt = _blobs()
    s = score_patch(gt, gt)
    assert s["n_gt"] == 6 and s["tp@0.5"] == 6 and s["fp@0.5"] == 0 and s["fn@0.5"] == 0 and s["tp@0.9"] == 6 and s["dice"] == 1.0
    shifted = np.roll(gt, 3, axis=1)                                   # IoU ~0.5: matches at 0.5, not at 0.9
    s = score_patch(gt, shifted)
    assert s["tp@0.5"] >= 4 and s["tp@0.9"] == 0 and s["fn@0.9"] == 6
    extra = gt.copy()
    extra[50:60, 50:60] = 7                                            # a false positive
    missing = np.where(gt == 6, 0, gt)                                 # a false negative
    assert score_patch(gt, extra)["fp@0.5"] == 1 and score_patch(gt, missing)["fn@0.5"] == 1
    assert score_patch(gt, np.zeros_like(gt))["fn@0.5"] == 6 and score_patch(np.zeros_like(gt), gt)["fp@0.5"] == 6
    m = aggregate([score_patch(gt, gt), score_patch(gt, missing)])
    assert m["n_patches"] == 2 and m["recall@0.5"] == 11 / 12 and m["precision@0.5"] == 1.0 and 0.9 < m["ap@0.5:0.9"] <= 1.0
    assert m["pq@0.5"] > 0.9 and m["count_ratio"] == 11 / 12


def test_score_patch_never_scores_wrong_nuclei_without_gt_scored():
    """Whole-region scoring passes no gt_scored: a QC-Wrong nucleus (its pixels in ``ignore``) must still count for nothing."""
    from mxseg.metrics import score_patch
    gt = np.zeros((40, 40), np.int32)
    gt[2:10, 2:10], gt[20:30, 20:30], gt[2:10, 25:35] = 1, 2, 3
    ignore = gt == 1                                                   # nucleus 1 is QC-Wrong
    pred = np.where(gt == 2, 5, 0).astype(np.int32)                   # only nucleus 2 found; the Wrong one is not a miss
    s = score_patch(gt, pred, ignore=ignore)
    assert (s["n_gt"], s["tp@0.5"], s["fp@0.5"], s["fn@0.5"]) == (2, 1, 0, 1)
    pred[gt == 1] = 9                                                  # a prediction on the Wrong nucleus: neither hit nor false positive
    s = score_patch(gt, pred, ignore=ignore)
    assert (s["n_gt"], s["n_pred"], s["tp@0.5"], s["fp@0.5"], s["fn@0.5"]) == (2, 1, 1, 0, 1)
    s = score_patch(gt, pred, ignore=None)                             # no review: every nucleus counts
    assert (s["n_gt"], s["tp@0.5"], s["fn@0.5"]) == (3, 2, 1)


def test_score_patch_owned_and_ignore_rules():
    gt = _blobs()
    scored = np.ones(7, bool)
    scored[6] = False                                                  # nucleus 6 is not owned by this patch
    s = score_patch(gt, np.where(gt == 6, 0, gt), gt_scored=scored)
    assert s["n_gt"] == 5 and s["fn@0.5"] == 0                        # missing an un-owned nucleus is not a miss
    ignore = gt == 1                                                   # nucleus 1 is QC-Wrong
    pred = gt.copy()
    pred[gt == 1] = 0
    assert score_patch(gt, pred, ignore=ignore, gt_scored=np.r_[True, False, [True] * 5])["fn@0.5"] == 0
    pred = gt.copy()
    pred[gt == 1] = 9                                                  # a prediction sitting on the Wrong nucleus is not a false positive
    s = score_patch(gt, pred, ignore=ignore, gt_scored=np.r_[True, False, [True] * 5])
    assert s["fp@0.5"] == 0
    core = (0, 0, 20, 32)                                              # only predictions with centroid in the core count (nucleus 1)
    s = score_patch(gt, gt, core=core, gt_scored=np.array([True, True, False, False, False, False, False]))
    assert s["n_pred"] == 1 and s["tp@0.5"] == 1 and s["fp@0.5"] == 0


class ThresholdBaseline(Baseline):
    """Otsu + connected components: a real (weak) segmenter that needs no weights."""

    name = "threshold"

    def __init__(self, blur: int = 3):
        self.blur = blur

    def params(self):
        return {"blur": self.blur}

    def predict(self, image, pixel_size_um=None):
        u8 = cv2.normalize(image[0].astype(np.float32), None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        u8 = cv2.GaussianBlur(u8, (self.blur, self.blur), 0) if self.blur > 1 else u8
        _, bw = cv2.threshold(u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return cv2.connectedComponents(bw)[1].astype(np.int32)


def test_run_baseline_stores_results_and_compare(cfg, dataset, tmp_path):
    out_dir, _ = dataset
    register("threshold", ThresholdBaseline)
    res = run_baseline(get_baseline("threshold"), cfg, split="val", dataset_dir=out_dir, results_dir=tmp_path / "results", n=8, progress=False,
                       outputs_dir=tmp_path / "outputs", panels=True)                                   # panels only on request
    assert res == tmp_path / "results" / "val" / "threshold"                    # default params -> bare name
    for f in ("run.json", "metrics.json", "per_patch.csv"):
        assert (res / f).exists()
    m = json.loads((res / "metrics.json").read_text())
    assert m["n_patches"] == 8 and 0 < m["f1@0.5"] <= 1 and "ap@0.5:0.9" in m and m["seconds_per_patch"] > 0
    assert m["n_wrong_ignored"] == 0 and m["n_patches_qc"] == 0                    # val = region 3: no QC table
    pf = tmp_path / "outputs" / "figures" / "patch" / "val"
    assert len(list(pf.iterdir())) == 8 and {"dapi.png", "ground_truth.png", "qc_review.png", "threshold.png", "panels.json"} <= {q.name for q in next(iter(sorted(pf.iterdir()))).iterdir()}
    run = json.loads((res / "run.json").read_text())
    assert run["baseline"] == "threshold" and run["params"] == {"blur": 3} and run["split"] == "val" and "mxdata" in run["versions"]
    pp = pd.read_csv(res / "per_patch.csv")
    assert len(pp) == 8 and (pp.n_gt > 0).all()
    qc = run_baseline(get_baseline("threshold"), cfg, split="train", dataset_dir=out_dir, results_dir=tmp_path / "results", require_qc=True, progress=False, store_predictions=False)
    mq = json.loads((qc / "metrics.json").read_text())
    assert mq["n_patches_qc"] == mq["n_patches"] > 0 and mq["n_wrong_ignored"] > 0     # train = regions 1-2 with QC: Wrong nuclei ignored
    assert json.loads((qc / "run.json").read_text())["ground_truth"]["n_wrong_ignored"] == mq["n_wrong_ignored"]
    table = compare_results(tmp_path / "results", split="val")
    assert list(table.name) == ["threshold"] and table["f1@0.5"].iloc[0] == round(m["f1@0.5"], 3)
    import yaml
    c = tmp_path / "cfg.yaml"
    c.write_text(yaml.safe_dump({"slides": list(cfg.slides)}))
    assert main(["eval", "-c", str(c), "--baseline", "threshold", "--param", "blur=1", "--split", "val", "--n", "4",
                 "--dataset", str(out_dir), "--results", str(tmp_path / "results"), "--name", "thr1", "--no-store-predictions"]) == 0
    assert len(compare_results(tmp_path / "results")) == 3
    assert main(["compare", "--results", str(tmp_path / "results")]) == 0
    assert main(["eval", "-c", str(c), "--baseline", "threshold", "--param", "nope", "--dataset", str(out_dir), "--results", str(tmp_path / "r2")]) == 1
    assert main(["eval", "--baseline", "threshold", "--dataset", str(out_dir), "--results", str(tmp_path / "r2")]) == 1   # default slides: not in this dataset
    assert get_baseline("threshold", blur=5).slug() == "threshold_blur-5" and get_baseline("threshold").slug() == "threshold"


def test_instance_outcomes_and_viz():
    from mxseg.viz import overlay_fg, paint_status, pixel_errors
    gt = _blobs()
    pred = np.where(gt == 6, 0, gt)                                    # nucleus 6 missed
    pred[50:60, 50:60] = 9                                             # and one spurious object
    o = instance_outcomes(gt, pred, gt_scored=np.r_[False, [True] * 6])
    assert (o["tp"], o["fp"], o["fn"]) == (5, 1, 1) and list(o["gt_status"][1:]) == ["tp"] * 5 + ["fn"] and "fp" in set(o["pred_status"])
    assert score_patch(gt, pred)["tp@0.5"] == 5                        # same matcher underneath
    rgb = np.zeros((64, 64, 3), np.uint8)
    centre = {i: tuple(np.argwhere(gt == i).mean(0).astype(int)) for i in range(1, 7)}   # (y, x) interior pixels
    a = overlay_fg(rgb, gt, ignore=gt == 1)
    assert a[centre[2]].any() and tuple(a[centre[1]]) == (142, 96, 0)   # nuclei coloured, QC-Wrong amber (0.6 blend)
    ow = instance_outcomes(gt, pred, ignore=gt == 1, gt_scored=np.r_[False, [True] * 6])
    assert ow["gt_status"][1] == "wrong" and ow["fn"] == 1 and ow["tp"] == 4      # Wrong nucleus: displayed, never counted
    b = paint_status(rgb, o["gt"], o["gt_status"], edges=False)
    assert b[centre[6]][2] > b[centre[1]][2] and b[centre[1]][1] > b[centre[6]][1]   # FN violet vs TP green
    c = pixel_errors(rgb, gt, pred)
    assert c[55, 55][0] > 100 and c[centre[6]][2] > 100                 # FP red, FN violet


def test_show_figures_layout():
    import matplotlib
    matplotlib.use("Agg")
    from types import SimpleNamespace

    import matplotlib.pyplot as plt

    from mxseg.viz import show_before_after, show_ground_truth, show_predictions, show_runs
    gt = _blobs()
    before = np.where(gt == 6, 0, gt)
    before[50:60, 50:60] = 9        # misses one nucleus, adds a spurious object
    after = gt.copy()                                                  # perfect
    rs = SimpleNamespace(key="P1", image=np.zeros((1, 64, 64), np.float32), box=SimpleNamespace(x0=0, y0=0),
                         meta={"core": (0, 0, 64, 64), "region_id": 1, "has_qc": True})
    sc, no_ignore = np.r_[False, [True] * 6], np.zeros_like(gt, bool)
    fig = show_before_after([rs, rs], [before, before], [after, after], scored=[sc, sc], labels_ignore=[(gt, no_ignore)] * 2,
                            names=("pretrained", "fine-tuned"), tags=["F1 0.83 → 1.00"] * 2)
    titles = [ax.get_title() for ax in fig.axes if ax.get_title()]
    assert len(titles) == 8 and "BEFORE — pretrained" in titles[2] and "AFTER — fine-tuned" in titles[3]
    assert "F1 0.83" in titles[2] and "F1 1.00" in titles[3] and "[F1 0.83 → 1.00]" in titles[0]
    assert titles[1] == "ground truth: 6 scored here (filled) · 0 not scored (faint)" and len(fig.axes[0].patches) == 1   # core outline
    plt.close(fig)
    fig = show_runs([rs], {"a": [before], "b": [after], "c": [before]}, scored=[sc], labels_ignore=[(gt, no_ignore)])
    titles = [ax.get_title() for ax in fig.axes if ax.get_title()]
    assert len(titles) == 5 and titles[2].startswith("a\n") and titles[4].startswith("c\n") and "F1 1.00" in titles[3]   # DAPI, GT, one column per model
    plt.close(fig)
    fig = show_predictions([rs], [before], scored=[sc], labels_ignore=[(gt, no_ignore)], title="m")
    assert len([ax for ax in fig.axes if ax.get_title()]) == 5
    plt.close(fig)
    wrong = gt == 1                                                    # nucleus 1 flagged Wrong: shown in the QC panel, never scored
    fig = show_ground_truth([rs], scored=[np.r_[False, False, [True] * 5]], labels_ignore=[(gt, wrong)])
    titles = [ax.get_title() for ax in fig.axes if ax.get_title()]
    assert len(titles) == 3 and titles[1].startswith("ground truth: 5 scored here") and titles[2] == "QC review: 5 OK · 1 Wrong (amber, ignored)"
    plt.close(fig)
    from mxseg.viz import show_region_ground_truth
    fig = show_region_ground_truth(rs.image, gt, wrong, True, [(0, 0)], size=64, title="R1")
    titles = [ax.get_title() for ax in fig.axes if ax.get_title()]
    assert len(titles) == 3 and titles[1] == "ground truth: 5 nuclei · 1 QC-Wrong (amber)" and titles[2].startswith("QC review: 5 OK")
    assert not fig.axes[0].patches                                            # region view: no owned-core box
    plt.close(fig)


def test_cellpose_baseline_refuses_missing_weights(tmp_path):
    from mxseg import get_baseline
    b = get_baseline("cellpose", variant=str(tmp_path / "runs" / "x" / "weights.pt"))
    with pytest.raises(FileNotFoundError, match="silently fall back"):
        b.model()


def test_predict_tiled_reassembles_every_object_once():
    from mxseg.metrics import iou_matrix, relabel
    from mxseg.tiling import predict_tiled, tile_grid
    rng = np.random.default_rng(0)
    lab = np.zeros((300, 200), np.int32)
    yy, xx = np.mgrid[:300, :200]
    k = 1
    for cy in range(15, 300, 30):
        for cx in range(15, 200, 25):
            r = 4 + rng.integers(0, 4)                                              # diameter <= 15 < overlap 16
            lab[((yy - cy) ** 2 + (xx - cx) ** 2 <= r * r) & (lab == 0)] = k
            k += 1
    tiles = tile_grid(300, 200, 64, 48)
    assert all(t.y1 - t.y0 == 64 and t.x1 - t.x0 == 64 for t in tiles)
    cover = np.zeros((300, 200), int)
    for t in tiles:
        cover[t.core[0]:t.core[2], t.core[1]:t.core[3]] += 1
    assert (cover == 1).all()                                                       # cores partition the region
    assert all(t.core[0] - t.y0 >= 8 or t.y0 == 0 for t in tiles) and all(t.y1 - t.core[2] >= 8 or t.y1 == 300 for t in tiles)   # margin = overlap / 2
    oracle = lambda ims, px: [relabel(im[0].astype(np.int32))[0] for im in ims]     # noqa: E731  a model that returns the tile's truth
    out = predict_tiled(oracle, lab[None], 64, 48, batch_size=4)
    iou, _, _ = iou_matrix(lab, out)
    assert out.max() == lab.max() and np.allclose(iou.max(1), 1.0)                  # every object recovered whole, exactly once


def test_region_level_eval_on_synthetic_slide(synth, cfg, dataset, tmp_path):
    import json

    import pandas as pd

    from mxseg import compare_results, run_baseline
    from mxseg.baselines import Baseline

    class Bright(Baseline):                                                          # bright blobs -> connected components
        name = "bright"

        def predict(self, image, pixel_size_um=None):
            from scipy import ndimage as ndi
            x = image[0].astype(np.float32)
            return ndi.label(x > x.mean() + x.std())[0].astype(np.int32)

    out, _ = dataset
    res = run_baseline(Bright(), cfg, split="val", dataset_dir=out, results_dir=tmp_path / "results", unit="region", slide_root=synth["root"],
                       progress=False, store_predictions=False)
    assert res == tmp_path / "results" / "region" / "val" / "bright"
    pr = pd.read_csv(res / "per_region.csv")
    met = json.loads((res / "metrics.json").read_text())
    assert list(pr.region_id) == [3] and met["unit"] == "region" and met["n_regions"] == 1 and 0 < met["f1@0.5"] <= 1
    tbl = compare_results(tmp_path / "results", unit="region")
    assert list(tbl.name) == ["bright"] and tbl.n.iloc[0] == 1 and tbl.unit.iloc[0] == "region"
    assert compare_results(tmp_path / "results", unit="patch").empty
    from mxseg import read_prediction, stored_predictions
    from mxseg import run_baseline as rb
    rb(Bright(), cfg, split="val", dataset_dir=out, results_dir=tmp_path / "results", unit="region", slide_root=synth["root"],
       progress=False, outputs_dir=tmp_path / "outputs", panels=True)                            # predictions stored by default, panels on request
    sp = stored_predictions("region", "val", "bright", tmp_path / "outputs")
    figs = tmp_path / "outputs" / "figures" / "region" / "val" / "SLIDE-9999_R03"
    names = {q.name for q in figs.iterdir()}
    assert {"dapi.png", "ground_truth.png", "qc_review.png", "bright.png", "panels.json"} <= names   # panels stored by the eval itself

    class Bright2(Bright):
        name = "bright2"
    rb(Bright2(), cfg, split="val", dataset_dir=out, results_dir=tmp_path / "results", unit="region", slide_root=synth["root"],
       progress=False, outputs_dir=tmp_path / "outputs", panels=True)
    side = json.loads((figs / "panels.json").read_text())
    assert set(side["models"]) == {"bright", "bright2"}                                              # method evals accumulate
    assert len(sp) == 1 and next(iter(sp.values())).suffix == ".npz"                              # data files, not dark images
    assert json.loads((res / "run.json").read_text())["predictions"].endswith("region/val/bright")
    pred = read_prediction(next(iter(sp.values())))
    assert pred.dtype == np.int32 and pred.ndim == 2 and pred.max() > 0                          # lossless roundtrip of the instance map


def test_sam2_masks_to_instances():
    from mxseg.baselines import BASELINES
    from mxseg.baselines.sam2 import masks_to_instances
    assert "sam2" in BASELINES
    a = np.zeros((64, 64), bool)
    a[5:20, 5:20] = True                  # good nucleus
    b = np.zeros((64, 64), bool)
    b[10:25, 10:25] = True                # overlaps a, lower predicted iou
    tissue = np.ones((64, 64), bool)                                    # image-sized mask: dropped
    dust = np.zeros((64, 64), bool)
    dust[0, :3] = True                 # under min_area: dropped
    masks = [{"segmentation": b, "predicted_iou": 0.8}, {"segmentation": a, "predicted_iou": 0.9},
             {"segmentation": tissue, "predicted_iou": 0.99}, {"segmentation": dust, "predicted_iou": 0.95}]
    lab = masks_to_instances(masks, (64, 64), min_area=15, max_area_frac=0.5)
    assert lab.max() == 2 and lab[6, 6] != lab[22, 22] and lab[6, 6] > 0 and lab[22, 22] > 0
    assert (lab[a & ~b] == lab[6, 6]).all() and not (lab == lab[6, 6])[~a].any()      # higher-iou mask kept whole; loser gets the rest
    assert masks_to_instances(masks, (64, 64), min_area=15, max_area_frac=0.5, claimed_frac=0.4).max() == 1   # b: 44% already claimed by a -> dropped


def test_sam3_baseline_registered():
    from mxseg import get_baseline
    from mxseg.baselines import BASELINES
    assert "sam3" in BASELINES
    b = get_baseline("sam3")
    assert b.params()["n_exemplars"] == 8 and b.params()["prompt"] == "exemplars"
    assert get_baseline("sam3", prompt="text").params()["text"] == "cell"
    with pytest.raises(ValueError, match="prompt"):
        get_baseline("sam3", prompt="bogus")


def test_baseline_from_config(tmp_path):
    from pathlib import Path

    from mxseg.baselines import from_config
    y = tmp_path / "m.yaml"
    y.write_text("baseline: cellpose\nparams:\n  variant: cpsam_v2\n  flow_threshold: 0.5\n")
    b, name = from_config(y)
    assert name == "m" and b.name == "cellpose" and b.flow_threshold == 0.5
    b, name = from_config(y, flow_threshold=0.6)                                  # --param overrides the file
    assert b.flow_threshold == 0.6
    y.write_text("baseline: cellpose\nname: custom\nparams: {}\n")
    assert from_config(y)[1] == "custom"
    y.write_text("baseline: cellpose\nbogus: 1\n")
    with pytest.raises(KeyError, match="unknown key"):
        from_config(y)
    for repo_cfg in Path("configs/baselines").glob("*.yaml"):
        if "ft" in repo_cfg.stem or repo_cfg.stem == "unet":
            continue                                                              # need trained weights on disk
        b, name = from_config(repo_cfg)
        assert b.params() and name                                                # every shipped config constructs cleanly


def test_export_region_panels(tmp_path):
    from mxseg.viz import export_region_panels
    gt = _blobs()
    pred = np.where(gt == 6, 0, gt)
    meta = export_region_panels(tmp_path, np.random.rand(1, 64, 64).astype(np.float32), gt, gt == 1, True,
                                {"m1": pred}, px_um=0.325)
    names = {p.name for p in tmp_path.iterdir()}
    assert names == {"dapi.png", "ground_truth.png", "qc_review.png", "m1.png", "panels.json"}          # only visual panels, nothing dark
    import cv2
    img = cv2.imread(str(tmp_path / "m1.png"))
    assert img.shape == (64, 64, 3)                                            # native resolution, no chrome
    assert meta["models"]["m1"]["fn"] == 1 and meta["scale_bar_um"] == 20.0
    export_region_panels(tmp_path, np.random.rand(1, 64, 64).astype(np.float32), gt, None, False, {}, crop=(0, 0, 32), suffix="_z")
    assert (tmp_path / "dapi_z.png").exists() and cv2.imread(str(tmp_path / "dapi_z.png")).shape == (32, 32, 3)


def test_aggregate_seeds_collapses_rows():
    import pandas as pd

    from mxseg.evaluate import aggregate_seeds
    df = pd.DataFrame({"split": ["test"] * 4, "name": ["unet_flows_s0", "unet_flows_s1", "unet_flows_s2", "sam2"], "n": [6] * 4,
                       "f1@0.5": [0.80, 0.82, 0.84, 0.60], "ap@0.5": [0.70, 0.70, 0.70, 0.40]})
    out = aggregate_seeds(df)
    assert list(out.name) == ["unet_flows", "sam2"] and list(out.seeds) == [3, 1]
    assert out.iloc[0]["f1@0.5"] == "0.820 ± 0.016" and out.iloc[0]["ap@0.5"] == "0.700 ± 0.000" and out.iloc[1]["f1@0.5"] == "0.600"
