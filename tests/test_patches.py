"""The sample stage: tissue gate, plans, the dataset on disk (files + tables), resume, multi-slide, labeled/unlabeled."""
import dataclasses
import json

import cv2
import numpy as np
import pandas as pd
import pytest

from mxdata import Box, DatasetError, MxSlide, SlidePaths, ingest, load_regions, sample_patches
from mxdata.config import SplitConfig
from mxdata.pipeline import (
    TissueGate,
    dataset_summary,
    otsu_threshold,
    plan_labeled,
    plan_units,
    read_patch,
    tissue_mask,
)
from mxdata.pipeline.sample import DATASET_FORMAT
from tests.conftest import SLIDE, SLIDE2


def test_otsu_and_tissue_mask():
    rng = np.random.default_rng(0)
    glass, tissue = rng.normal(100, 5, 5000), rng.normal(2000, 100, 5000)
    thr = otsu_threshold(np.concatenate([glass, tissue]))
    assert (glass < thr).all() and (tissue > thr).all()
    thumb = np.full((100, 100), 100.0)
    thumb[20:80, 30:90] = 2000.0
    mask, info = tissue_mask(thumb, closing=1)
    assert info["method"] == "otsu" and mask[50, 60] and not mask[5, 5] and abs(info["tissue_fraction"] - 0.36) < 0.02
    _, info = tissue_mask(np.full((50, 50), 100.0) + rng.normal(0, 0.1, (50, 50)), closing=1)
    assert info["method"].startswith("percentile")                        # near-uniform: Otsu would split noise in half
    gate = TissueGate(mask=mask, downsample=4.0)
    assert gate.fraction(Box(120, 80, 360, 320)) == 1.0 and gate.fraction(Box(0, 0, 40, 40)) == 0.0 and 0 < gate.fraction(Box(0, 0, 400, 400)) < 1


def test_tissue_gate_from_synthetic_slide(synth):
    slide = MxSlide.from_paths(SlidePaths.discover(SLIDE, synth["root"]))
    gate = TissueGate.from_slide(slide, "DAPI", max_dim=256)
    assert gate.downsample == 4 and gate.info["marker"] == "DAPI" and 0.05 < gate.info["tissue_fraction"] < 0.9
    assert gate.fraction(Box(*synth["regions"][1]["bbox"])) > gate.fraction(Box(0, 0, 40, 40))


def test_plan_labeled_is_pure(synth, cfg):
    paths, slide, _ = ingest(SLIDE, synth["root"])
    units = plan_units(cfg, slide, load_regions(paths, min_area_px=20))
    specs = plan_labeled(units, SLIDE, 64, 48)
    assert len(specs) > len(units) and len({s.key for s in specs}) == len(specs) and specs == plan_labeled(units, SLIDE, 64, 48)
    assert specs[0].key == f"{SLIDE}_R01_x{specs[0].tile.box.x0}_y{specs[0].tile.box.y0}"


def test_sample_end_to_end(synth, cfg, dataset):
    out, res = dataset
    assert res.n_failed == 0 and res.n_done == 3 + 3 and 0 < res.n_kept <= res.n_windows and res.n_unlabeled_kept > 0
    for name in ("dataset.json", "patches.csv", "instances.csv", "slides.csv"):
        assert (out / name).exists(), name
    for name in ("stats.json", "tissue.json", "sampling_report.json", "resolved_config.yaml", "ingest_report.json", "validation_report.json", "slide.json", ".done.json"):
        assert (out / SLIDE / name).exists(), name
    meta = json.loads((out / "dataset.json").read_text())
    assert meta["format"] == DATASET_FORMAT and meta["geometry"] == {"size": 64, "stride": 48, "unlabeled_stride": 64, "markers": ["DAPI", "PanCK"], "level": 0}
    assert meta["slides"] == [SLIDE] and meta["counts"]["n_labeled"] == res.n_kept and meta["counts"]["n_unlabeled"] == res.n_unlabeled_kept

    allp = pd.read_csv(out / "patches.csv", dtype={"mask": str, "reject_reason": str})
    lab, unl = allp[allp.kind == "labeled"], allp[allp.kind == "unlabeled"]
    inst, slides = pd.read_csv(out / "instances.csv"), pd.read_csv(out / "slides.csv")
    assert allp.key.is_unique and len(lab) == res.n_windows and lab.kept.sum() == res.n_kept and len(unl) == res.n_unlabeled_windows
    assert (lab[lab.region_id == 3].split == "val").all() and set(lab.group) == {f"{SLIDE}/g0", f"{SLIDE}/g1"}
    g1, g2, g3 = (set(lab[lab.region_id == r].group) for r in (1, 2, 3))
    assert g1 == g2 and len(g1) == 1 and g3 != g1                          # overlapping FOVs share a group
    assert slides.loc[0, "slide"] == SLIDE and slides.loc[0, "n_labeled"] == res.n_kept and slides.loc[0, "n_unlabeled"] == res.n_unlabeled_kept
    assert slides.loc[0, "tissue_method"] == "otsu"

    truth = synth["regions"]
    for _, row in lab[lab.kept].head(12).iterrows():                        # files, ids, flags, ownership agree with the tables
        image, labels = read_patch(out / row.image, out / row["mask"], ["DAPI"])
        assert image.shape == (1, 64, 64) and image.dtype == np.uint16 and labels.shape == (64, 64)
        assert (out / row.image / "DAPI.tif").is_file() and (out / row.image / "mask.png").is_file()   # one folder per patch, one file per marker
        true = cv2.imread(str(out / row.image / "mask_true.png"), cv2.IMREAD_UNCHANGED).astype(np.int32)          # confirmed cells only
        cells = pd.read_csv(out / row.image / "cells.csv")
        wrong_ids = set(cells[cells.flag == "WRONG"].local_id)
        rows = inst[inst.patch_key == row.key].sort_values("local_id")
        assert set(np.unique(true)) - {0} == set(np.unique(labels)) - {0} - wrong_ids and list(cells.local_id) == list(rows.local_id)
        ids = np.unique(labels)
        assert list(ids[ids > 0]) == list(rows.local_id) and row.n_instances == len(rows)
        assert row.n_wrong == (rows.flag == "WRONG").sum() and row.n_owned == rows.owned.sum() and row.n_border == rows.touches_border.sum()
        assert {g + 1 for g in rows[rows.flag == "WRONG"].nucleus_id} <= set(truth[row.region_id]["wrong_labels"]) and "TINY" not in set(rows.flag)
        assert Box(row.core_x0, row.core_y0, row.core_x1, row.core_y1).inside(row.x1, row.y1)
    row = lab[lab.kept].iloc[0]
    slide = MxSlide.from_paths(SlidePaths.discover(SLIDE, synth["root"]))
    np.testing.assert_array_equal(read_patch(out / row.image, out / row["mask"], ["DAPI"])[0], slide.read_region(Box(row.x0, row.y0, row.x1, row.y1), ["DAPI"]))
    assert "kept" in dataset_summary(out)


def test_unlabeled_patches_come_from_tissue_outside_every_fov(synth, cfg, dataset):
    out, res = dataset
    allp = pd.read_csv(out / "patches.csv", dtype={"mask": str, "reject_reason": str})
    unl = allp[allp.kind == "unlabeled"]
    kept = unl[unl.kept.astype(bool)]
    assert len(kept) == res.n_unlabeled_kept and kept.key.str.startswith(f"{SLIDE}_U_x").all() and (kept.region_id == -1).all()
    assert kept["mask"].isna().all() and (kept.tissue_fraction >= cfg.patches.min_tissue).all() and (kept.fov_overlap == 0).all()
    fovs, ub = [Box(*synth["regions"][r]["bbox"]) for r in synth["regions"]], Box(*synth["unannotated_box"])
    for _, row in kept.iterrows():
        b = Box(row.x0, row.y0, row.x1, row.y1)
        assert all(b.intersect(f) is None for f in fovs) and (out / row.image).exists() and b.intersect(ub) is not None
    reasons = set(unl[~unl.kept.astype(bool)].reject_reason)
    assert "fov_overlap" in reasons and any(r.startswith("tissue<") for r in reasons)
    assert not pd.read_csv(out / "instances.csv").patch_key.isin(kept.key).any() and (kept.split == "train").all()


def test_sample_resume_refuses_incompatible_config_and_adds_a_slide(synth, synth2, cfg, dataset):
    out, res = dataset
    again = sample_patches(cfg, out, root=synth["root"])
    assert (again.n_done, again.n_skipped, again.n_kept) == (0, 6, res.n_kept)
    with pytest.raises(DatasetError, match="different --out"):
        sample_patches(dataclasses.replace(cfg, patches=dataclasses.replace(cfg.patches, stride=32)), out, root=synth["root"])
    with pytest.raises(DatasetError):
        sample_patches(dataclasses.replace(cfg, markers=("DAPI",)), out, root=synth["root"])     # a different channel set
    assert sample_patches(cfg, out, root=synth["root"], dry_run=True).n_units == 6

    two = dataclasses.replace(cfg, slides=(SLIDE, SLIDE2), split=SplitConfig(val_slides=(SLIDE2,)))
    res2 = sample_patches(two, out, root=synth["root"])                    # adds SLIDE2, keeps SLIDE
    assert (res2.n_skipped, res2.n_done, res2.n_failed) == (6, 6, 0)
    patches = pd.read_csv(out / "patches.csv")
    assert set(json.loads((out / "dataset.json").read_text())["slides"]) == {SLIDE, SLIDE2}
    assert (patches[patches.slide == SLIDE2].split == "val").all() and (patches[patches.slide == SLIDE].split == "train").all()
    assert set(patches.group) == {f"{s}/{g}" for s in (SLIDE, SLIDE2) for g in ("g0", "g1", "unlabeled")}


def test_unlabeled_cap_is_seeded_and_disable_switch(synth, cfg, tmp_path):
    from mxdata.data import PatchSource
    capped = dataclasses.replace(cfg, patches=dataclasses.replace(cfg.patches, max_unlabeled_per_slide=3))
    a, b = (sample_patches(capped, tmp_path / d, root=synth["root"]) for d in ("a", "b"))
    assert a.n_unlabeled_kept == b.n_unlabeled_kept == 3
    assert PatchSource(tmp_path / "a", kind="unlabeled").keys == PatchSource(tmp_path / "b", kind="unlabeled").keys
    pa = pd.read_csv(tmp_path / "a" / "patches.csv")
    assert (pa[(pa.kind == "unlabeled") & ~pa.kept].reject_reason == "subsampled").sum() > 0
    off = dataclasses.replace(cfg, patches=dataclasses.replace(cfg.patches, unlabeled=False))
    c = sample_patches(off, tmp_path / "c", root=synth["root"])
    assert c.n_unlabeled_windows == 0 and c.n_done == 3 and PatchSource(tmp_path / "c", kind=None).keys == PatchSource(tmp_path / "c").keys


def test_failure_isolation(synth, cfg, tmp_path, monkeypatch):
    import mxdata.pipeline.sample as st
    real = st.sample_labeled_unit

    def boom(unit, *a, **k):
        if unit.region_id == 2:
            raise RuntimeError("simulated corrupt region")
        return real(unit, *a, **k)

    monkeypatch.setattr(st, "sample_labeled_unit", boom)
    res = sample_patches(cfg, tmp_path / "d", root=synth["root"])
    assert res.n_failed == 1 and res.failures[0]["key"] == f"{SLIDE}_R02" and res.n_done == 5
    assert set(pd.read_csv(tmp_path / "d" / "patches.csv").region_id) == {1, 3, -1}
    assert json.loads((tmp_path / "d" / SLIDE / "sampling_report.json").read_text())["failures"][0]["reason"].startswith("RuntimeError")


def test_older_dataset_files_are_refused_with_a_hint(synth, cfg, dataset):
    from mxdata.data import PatchSource
    from mxdata.pipeline import rebuild_tables
    out, _ = dataset
    tp = next((out / SLIDE / "tables").glob("*_R01_patches.csv"))
    pd.read_csv(tp).drop(columns=["kind"]).to_csv(tp, index=False)          # a table from before `kind` existed
    with pytest.raises(DatasetError, match="older mxdata"):
        rebuild_tables(out, cfg)
    meta = json.loads((out / "dataset.json").read_text())
    meta["format"] = 1
    (out / "dataset.json").write_text(json.dumps(meta))
    with pytest.raises(DatasetError, match="format 1"):
        PatchSource(out)
    res = sample_patches(cfg, out, root=synth["root"], resume=False)          # the hinted fix rewrites everything
    assert res.n_done == 6 and PatchSource(out).keys


def test_sample_assigns_test_split(synth, cfg, tmp_path):
    import dataclasses
    import json

    import pytest

    from mxdata import DatasetError, SplitConfig, sample_patches
    from mxdata.data import PatchSource
    three = dataclasses.replace(cfg, split=SplitConfig(val_regions=(3,), test_regions=(1,)))
    out = tmp_path / "ds"
    sample_patches(three, out, root=synth["root"])
    test, val, train = (PatchSource(out, split=s, kind="labeled", slides=cfg.slides) for s in ("test", "val", "train"))
    assert test.keys and val.keys and not train.keys and not (set(test.keys) & set(val.keys))
    assert set(test.table.region_id) == {1, 2} and set(val.table.region_id) == {3}          # region 2 follows its overlap partner 1
    counts = json.loads((out / "dataset.json").read_text())["counts"]["per_kind_split"]
    assert counts["labeled/test"] == len(test.keys) and counts["unlabeled/test"] == 0            # unlabeled patches follow their slide
    with pytest.raises(DatasetError):
        PatchSource(out, split="bogus")
