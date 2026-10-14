import dataclasses

import numpy as np
import pytest
import torch

from mxdata import DataConfig, ValidationError, load_regions, split_regions
from mxdata.config import LoaderConfig, NormalizationConfig, SplitConfig
from mxdata.data import (
    Augment,
    LiveSource,
    Normalizer,
    PatchSource,
    RegionDataset,
    build_datasets,
    build_loaders,
    flip_rot90,
    make_dataloader,
)
from mxdata.pipeline import ingest
from mxdata.splits import overlap_groups
from tests.conftest import SLIDE, SLIDE2

R1, R2, R3 = f"{SLIDE}_R01", f"{SLIDE}_R02", f"{SLIDE}_R03"


@pytest.fixture(scope="module")
def live(synth):
    paths, slide, _ = ingest(SLIDE, synth["root"])
    regions = load_regions(paths, min_area_px=20)
    return regions, LiveSource(slide, regions, ["DAPI"])


def test_split_never_separates_overlapping_regions(live):
    regions, _ = live
    assert overlap_groups(regions) == [[R1, R2], [R3]]
    for seed in range(6):
        s = split_regions(regions, SplitConfig(val_frac=0.3, seed=seed))
        assert sorted(s["train"] + s["val"]) == [R1, R2, R3] and ((R1 in s["val"]) == (R2 in s["val"]))
    assert split_regions(regions, SplitConfig(val_regions=(1,))) == {"train": [R3], "val": [R1, R2], "test": []}
    assert split_regions(regions, SplitConfig(val_regions=(3,), test_regions=(1,))) == {"train": [], "val": [R3], "test": [R1, R2]}   # partners follow
    drawn = split_regions(regions, SplitConfig(val_frac=0.5, seed=0, test_regions=(3,)))
    assert drawn["test"] == [R3] and set(drawn["train"] + drawn["val"]) == {R1, R2}                    # val drawn from what is not test
    with pytest.raises(ValidationError):
        split_regions(regions, SplitConfig(val_regions=(1,), test_regions=(2,)))                       # 2 is 1's overlap partner
    assert split_regions(regions, SplitConfig(require_qc=True, val_regions=(2,)))["train"] == []
    assert split_regions(regions, SplitConfig(val_slides=(SLIDE,)))["train"] == []
    assert split_regions(regions, SplitConfig(val_regions={SLIDE: (3,)}, test_regions={SLIDE: (1,)}))["test"] == [R1, R2]   # pinned per slide
    for bad in (SplitConfig(val_regions=(9,)), SplitConfig(val_slides=("SLIDE-0000",)), SplitConfig(test_regions={"SLIDE-0000": (1,)})):
        with pytest.raises(ValidationError):
            split_regions(regions, bad)


def test_live_region_dataset_and_policies(live, synth):
    regions, src = live
    t = synth["regions"][1]
    for policy in ("keep_all", "ignore_wrong", "drop_wrong"):
        ds = RegionDataset(src, qc_policy=policy, normalizer=Normalizer("region"))
        s = ds[0]
        assert tuple(s["image"].shape[1:]) == tuple(s["labels"].shape) == tuple(s["ignore"].shape) == regions[0].box.shape
        assert s["image"].dtype == torch.float32 and 0 <= float(s["image"].min()) and float(s["image"].max()) <= 1
        ids, wrong = set(torch.unique(s["labels"]).tolist()) - {0}, set(t["wrong_labels"])
        assert (t["tiny_index"] + 1) not in ids
        if policy == "keep_all":
            assert not s["ignore"].any() and wrong <= ids
        elif policy == "ignore_wrong":
            assert wrong <= ids and s["ignore"].sum() > 0
        else:
            assert not (wrong & ids) and s["ignore"].sum() > 0
        assert s["meta"]["region_id"] == 1 and s["meta"]["qc_policy"] == policy and s["meta"]["has_qc"]
    with pytest.raises(ValidationError):
        RegionDataset(src, qc_policy="bogus")


def test_augment_keeps_image_and_masks_aligned():
    rng = np.random.default_rng(0)
    img = rng.random((2, 40, 60), np.float32)
    lab = (np.arange(40 * 60).reshape(40, 60) % 7).astype(np.int32)
    s = {"image": img, "labels": lab, "ignore": lab == 3, "meta": {}}
    for _ in range(8):
        o = flip_rot90(s, rng)
        assert o["image"].shape[1:] == o["labels"].shape == o["ignore"].shape and np.array_equal(o["ignore"], o["labels"] == 3)
        assert np.isclose(o["image"].sum(), img.sum())
    assert Augment(False, False)(s, rng) is s and set(flip_rot90({"image": img, "meta": {}}, rng)) == {"image", "meta"}


def test_normalizer_per_slide():
    from mxdata.pipeline import NormStats
    st = {"A": NormStats("A", ["DAPI"], {"DAPI": 0.0}, {"DAPI": 100.0}, 1, 99.5, 0, 10, 0, 1),
          "B": NormStats("B", ["DAPI"], {"DAPI": 0.0}, {"DAPI": 200.0}, 1, 99.5, 0, 10, 0, 1)}
    n = Normalizer.from_stats(st, ["DAPI"])
    img = np.full((1, 4, 4), 100, np.uint16)
    assert n(img, "A").max() == 1.0 and n(img, "B").max() == 0.5 and "A" in repr(n)
    with pytest.raises(ValidationError):
        n(img, "C")
    with pytest.raises(ValidationError):
        Normalizer("slide")
    assert Normalizer("none")(img).max() == pytest.approx(100 / 65535)


def test_patch_source_kinds_and_datasets(synth, cfg, dataset):
    out, res = dataset
    lab, unl, both = PatchSource(out), PatchSource(out, kind="unlabeled"), PatchSource(out, kind=None)
    assert len(lab.keys) == res.n_kept and len(unl.keys) == res.n_unlabeled_kept and len(both.keys) == len(lab.keys) + len(unl.keys)
    assert lab.markers == ["DAPI", "PanCK"] and SLIDE in lab.stats and all(lab.labeled(k) for k in lab.keys) and not unl.labeled(unl.keys[0])
    rs = lab.load(lab.keys[0])
    assert rs.image.shape == (2, 64, 64) and rs.labels.dtype == np.int32 and len(rs.flags) == rs.labels.max() + 1 and rs.meta["split"] in ("train", "val")
    assert lab.core(lab.keys[0]).inside(*lab.box(lab.keys[0]).xyxy[2:])
    ru = unl.load(unl.keys[0])
    assert ru.image.shape == (2, 64, 64) and not ru.labels.any() and len(ru.flags) == 1 and ru.meta["labeled"] is False
    train, val = PatchSource(out, split="train"), PatchSource(out, split="val")
    assert len(train.keys) + len(val.keys) == len(lab.keys) and all(val.row(k)["region_id"] == 3 for k in val.keys)
    assert PatchSource(out, slides=["SLIDE-0000"]).keys == []
    with pytest.raises(ValidationError, match="unlabeled"):
        RegionDataset(both, with_labels=True)                              # supervised must never see them
    item = RegionDataset(both, with_labels=False)[0]
    assert set(item) == {"image", "meta"} and tuple(item["image"].shape) == (2, 64, 64) and item["meta"]["with_labels"] is False
    s = RegionDataset(PatchSource(out, split="train"), qc_policy="ignore_wrong")[0]
    assert tuple(s["image"].shape) == (2, 64, 64) and s["labels"].dtype == torch.int32 and s["ignore"].dtype == torch.bool


def test_patch_source_reads_only_the_requested_markers(synth, cfg, dataset):
    """The segmenters open DAPI + LAP2 only: a marker subset, in the requested order, case-insensitive; unknown markers are named."""
    out, _ = dataset
    full, sub = PatchSource(out), PatchSource(out, markers=["panck", "DAPI"])
    assert full.markers == list(cfg.markers) and sub.markers == ["PanCK", "DAPI"]
    a, b = full.load(full.keys[0]), sub.load(full.keys[0])
    assert b.image.shape == (2, *a.image.shape[1:]) and b.meta["markers"] == ["PanCK", "DAPI"]
    np.testing.assert_array_equal(b.image[1], a.image[list(cfg.markers).index("DAPI")])
    np.testing.assert_array_equal(b.labels, a.labels)
    with pytest.raises(Exception, match="not stored"):
        PatchSource(out, markers=["CD45"])


def test_build_datasets_supervised_and_self_supervised(synth, cfg, dataset):
    out, res = dataset
    sup = build_datasets(cfg, out)
    assert sup.normalizer.mode == "slide" and SLIDE in sup.stats and sup.counts(sup.train)["unlabeled"] == 0 and "supervised" in sup.summary()
    tl, vl = build_loaders(sup)
    b = next(iter(tl))
    assert tuple(b["image"].shape) == (4, 2, 64, 64) and tuple(b["labels"].shape) == (4, 64, 64) and b["ignore"].dtype == torch.bool and len(b["meta"]) == 4
    assert all(m["split"] == "train" and m["kind"] == "labeled" for m in b["meta"]) and len(list(tl)) == len(sup.train) // 4     # drop_last
    assert all(m["region_id"] == 3 for m in next(iter(vl))["meta"])

    ssl = build_datasets(dataclasses.replace(cfg, mode="self_supervised"), out)
    n_lab = len(PatchSource(out, split="train").keys)
    assert ssl.counts(ssl.train) == {"labeled": n_lab, "unlabeled": res.n_unlabeled_kept} and len(ssl.train) == n_lab + res.n_unlabeled_kept
    stl, _ = build_loaders(ssl)
    assert set(next(iter(stl))) == {"image", "meta"} and {m["kind"] for bt in stl for m in bt["meta"]} == {"labeled", "unlabeled"}

    for bad in (dataclasses.replace(cfg, markers=("PanCK",)), dataclasses.replace(cfg, slides=("SLIDE-0000",))):
        with pytest.raises(ValidationError):
            build_datasets(bad, out)
    region_mode = build_datasets(dataclasses.replace(cfg, normalization=NormalizationConfig(mode="region")), out)
    assert region_mode.normalizer.mode == "region"


def test_workers_get_their_own_seeds(synth, cfg, dataset):
    out, _ = dataset
    bundle = build_datasets(cfg, out)
    loader = make_dataloader(bundle.train, LoaderConfig(batch_size=2, num_workers=2, seed=0), train=True)
    seen = [(m["key"], float(b["image"][j].sum())) for b in loader for j, m in enumerate(b["meta"])]
    assert len(seen) == (len(bundle.train) // 2) * 2 and len({k for k, _ in seen}) == len(seen)


def test_two_slides_leave_slide_out(synth, synth2, cfg, tmp_path):
    from mxdata import sample_patches
    two = dataclasses.replace(cfg, slides=(SLIDE, SLIDE2), split=SplitConfig(val_slides=(SLIDE2,)))
    out = tmp_path / "two"
    sample_patches(two, out, root=synth["root"])
    bundle = build_datasets(two, out)
    assert sorted(bundle.stats) == sorted([SLIDE, SLIDE2]) and set(bundle.normalizer.per_slide) == {SLIDE, SLIDE2}
    assert set(bundle.val.source.table.slide) == {SLIDE2} and set(bundle.train.source.table.slide) == {SLIDE}
    s = bundle.val[0]
    assert s["meta"]["slide"] == SLIDE2 and float(s["image"].max()) <= 1.0
    assert build_datasets(DataConfig(slides=(SLIDE,), markers=cfg.markers, patches=cfg.patches), out).cfg.slides == (SLIDE,)
