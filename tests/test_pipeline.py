import pandas as pd
import pytest

from mxdata import (
    Box,
    DataConfig,
    ValidationError,
    compute_norm_stats,
    ingest,
    load_regions,
    plan_units,
    tile_grid,
    validate,
)
from mxdata.config import NormalizationConfig
from mxdata.pipeline import NormStats, ensure_stats, prepare
from tests.conftest import SLIDE, W


def test_ingest_report(synth):
    paths, slide, rep = ingest(SLIDE, synth["root"])
    d = rep.to_dict()
    assert d["markers"] == ["DAPI", "PanCK"] and d["halo_regions"] == [1, 2, 3] and d["qc_regions"] == [1, 2]
    assert d["width"] == W and d["n_levels"] == 3 and d["pixel_size_um"] == 0.5 and d["unparsed_tiffs"] == ["not_a_channel.tif"]
    assert SLIDE in rep.summary()
    slide.close()


def test_validate_passes_on_good_data_and_reports_facts(synth, cfg):
    paths, slide, _ = ingest(SLIDE, synth["root"])
    rep = validate(slide, load_regions(paths, min_area_px=20), cfg)
    assert rep.ok and {"QC_WRONG", "TINY_CONTOURS", "FOV_OVERLAP"} <= {i.code for i in rep.issues}
    assert len(rep.regions) == 3 and rep.overlaps[0]["region_a"] == 1 and "OK" in rep.summary("warning")


def test_validate_catches_config_slide_mismatch(synth):
    paths, slide, _ = ingest(SLIDE, synth["root"])
    rep = validate(slide, load_regions(paths), DataConfig(slides=(SLIDE,), markers=("DAPI", "CD45"), level=9))
    assert not rep.ok and {"MARKER_MISSING", "LEVEL_OUT_OF_RANGE"} <= {i.code for i in rep.issues if i.severity == "error"}


def test_validate_catches_qc_that_disagrees_with_the_layer(synth_copy):
    paths, slide, _ = ingest(SLIDE, synth_copy)
    df = pd.read_csv(paths.qc_files[2])
    extra = df.iloc[[0]].copy()
    extra[["XMin", "XMax", "YMin", "YMax"]] = [[700, 720, 500, 520]]      # a row with no contour
    pd.concat([df, extra]).to_csv(paths.qc_files[2], index=False)
    rep = validate(slide, load_regions(paths))
    assert not rep.ok and any(i.code == "QC_ROWS_UNMATCHED" and i.region == 2 for i in rep.issues)


def test_prepare_raises_on_validation_errors(synth, tmp_path):
    with pytest.raises(ValidationError, match="failed validation"):
        prepare(DataConfig(slides=(SLIDE,), markers=("CD45",)), root=synth["root"])


def test_plan_units(synth, cfg):
    paths, slide, _ = ingest(SLIDE, synth["root"])
    regions = load_regions(paths, min_area_px=20)
    units = plan_units(cfg, slide, regions)
    assert [u.region_id for u in units] == [1, 2, 3] and units[0].key == f"{SLIDE}_R01" and units[0].has_qc and units[0].box == regions[0].box
    assert [u.region_id for u in plan_units(DataConfig(slides=(SLIDE,), regions=(2,)), slide, regions)] == [2]


@pytest.mark.parametrize("box,size,stride", [(Box(10, 20, 310, 220), 64, 48), (Box(0, 0, 100, 100), 128, 128),
                                             (Box(5, 5, 200, 130), 50, 50), (Box(0, 0, 64, 64), 64, 32)])
def test_tile_grid_cores_partition_the_box_exactly(box, size, stride):
    tiles = tile_grid(box, size, stride)
    assert all(t.box.inside(box.x1, box.y1) and t.box.x0 >= box.x0 and t.box.y0 >= box.y0 for t in tiles)
    assert all(t.core.intersect(t.box) == t.core for t in tiles)
    assert sum(t.core.area for t in tiles) == box.area
    assert all(a.core.intersect(b.core) is None for i, a in enumerate(tiles) for b in tiles[i + 1:])
    with pytest.raises(ValueError):
        tile_grid(box, 64, 65)
    assert len(tile_grid(Box(0, 0, 400, 200), 50, 50, downsample=2)) == 8


def test_norm_stats_are_seeded_and_saved(synth, cfg, tmp_path):
    paths, slide, _ = ingest(SLIDE, synth["root"])
    boxes = [r.box for r in load_regions(paths)]
    ncfg = NormalizationConfig(sample_pixels=10_000, seed=3)
    a = compute_norm_stats(slide, boxes, ["DAPI", "PanCK"], ncfg, 0, paths.signature())
    b = compute_norm_stats(slide, boxes, ["DAPI", "PanCK"], ncfg, 0, paths.signature())
    assert a.lo == b.lo and a.hi == b.hi and a.lo["DAPI"] < a.hi["DAPI"] and a.n_windows == 3
    c = NormStats.load(a.save(tmp_path / "stats.json"))
    assert c == a and c.is_current(["DAPI"], ncfg, 0, paths.signature())
    assert not c.is_current(["DAPI"], NormalizationConfig(sample_pixels=10_000, seed=4), 0, paths.signature())
    lo, hi = c.arrays(["PanCK", "DAPI"])
    assert lo.shape == (2,) and lo[1] == a.lo["DAPI"]
    with pytest.raises(ValidationError):
        c.arrays(["CD45"])
    # ensure_stats reuses a current file and recomputes otherwise
    paths, slide, regions, _, _, units = prepare(cfg, root=synth["root"])
    p = tmp_path / "s.json"
    s1 = ensure_stats(cfg, slide, units, paths, p)
    assert p.exists() and ensure_stats(cfg, slide, units, paths, p) == s1
    assert ensure_stats(DataConfig(slides=(SLIDE,), normalization=NormalizationConfig(mode="region")), slide, units, paths, None) is None
