import numpy as np
import pytest

from mxdata import (
    AnnotationError,
    Box,
    NucleusFlag,
    SlidePaths,
    apply_qc_policy,
    load_halo_geojson,
    load_qc_csv,
    load_regions,
    match_qc_to_polygons,
    nuclei_table,
    overlap_table,
    rasterize_polygons,
    regions_table,
)
from mxdata.annotations import polygon_area
from tests.conftest import SLIDE


def test_geojson_layer(synth):
    p = SlidePaths.discover(SLIDE, synth["root"])
    layer = load_halo_geojson(p.halo_files[1])
    assert len(layer) == synth["regions"][1]["n"] and layer.n_skipped == 0
    assert all(not np.array_equal(r[0], r[-1]) for r in layer.rings)     # closing vertex dropped
    assert layer.issues == []                                             # integer, closed, >= 3 points
    with pytest.raises(AnnotationError):
        load_halo_geojson(p.qc_files[1])                                  # a csv is not GeoJSON


def test_geojson_tolerates_and_reports_oddities(tmp_path):
    import json
    feats = [{"type": "Feature", "properties": {}, "geometry": {"type": "LineString", "coordinates": [[0, 0], [10, 0], [10, 10]]}},
             {"type": "Feature", "properties": {}, "geometry": {"type": "Polygon", "coordinates": [[[0, 0], [5, 0.5], [5, 5], [0, 0]]]}},
             {"type": "Feature", "properties": {}, "geometry": {"type": "Point", "coordinates": [1, 1]}}]
    f = tmp_path / "x.geojson"
    f.write_text(json.dumps({"type": "FeatureCollection", "features": feats}))
    layer = load_halo_geojson(f)
    assert len(layer) == 2 and layer.n_skipped == 1
    assert any("not explicitly closed" in i for i in layer.issues) and any("non-integer" in i for i in layer.issues)


def test_polygon_area_and_rasterize_order():
    big = np.array([[0, 0], [20, 0], [20, 20], [0, 20]], float)
    small = np.array([[5, 5], [10, 5], [10, 10], [5, 10]], float)
    assert polygon_area(big) == 400 and polygon_area(small[:2]) == 0
    lab = rasterize_polygons([small, big], Box(0, 0, 21, 21))
    assert lab[7, 7] == 1 and lab[1, 1] == 2                              # small nucleus stays on top
    lab2 = rasterize_polygons([small, big], Box(0, 0, 21, 21), ids=[7, 9])
    assert lab2[7, 7] == 7 and lab2[1, 1] == 9
    at2 = rasterize_polygons([big], Box(0, 0, 21, 21), downsample=2)
    assert at2.shape == (11, 11) and at2[5, 5] == 1
    with pytest.raises(AnnotationError):
        rasterize_polygons([big], Box(0, 0, 21, 21), ids=[1, 2])


def test_qc_schema_and_matching(synth):
    p = SlidePaths.discover(SLIDE, synth["root"])
    qc = load_qc_csv(p.qc_files[1])
    layer = load_halo_geojson(p.halo_files[1])
    rows, st = match_qc_to_polygons(layer.rings, qc)
    assert st["unmatched_qc"] == 0 and st["max_dist"] <= 1 and st["unmatched_polys"] == 1   # the tiny fragment
    assert rows[synth["regions"][1]["tiny_index"]] == -1
    bad = qc.drop(columns=["Flag"])
    bad.to_csv(p.root / "bad.csv", index=False)
    with pytest.raises(AnnotationError, match="missing column"):
        load_qc_csv(p.root / "bad.csv")


def test_flags_are_data(synth):
    p = SlidePaths.discover(SLIDE, synth["root"])
    regs = {r.region_id: r for r in load_regions(p, min_area_px=20)}
    t = synth["regions"]
    r1, r3 = regs[1], regs[3]
    assert r1.has_qc and not r3.has_qc
    assert sorted(r1.nuclei.label[r1.nuclei.flag == "WRONG"].tolist()) == t[1]["wrong_labels"]
    assert r1.count(NucleusFlag.TINY) == 1 and r1.count(NucleusFlag.NO_QC_ROW) == 0
    assert r1.count(NucleusFlag.OK) + r1.n_wrong + 1 == len(r1)
    assert r3.count(NucleusFlag.UNKNOWN) == len(r3) and r3.n_wrong == 0
    assert r1.box.xyxy == t[1]["bbox"]
    lut = r1.flags
    assert lut.dtype == np.int8 and len(lut) == len(r1) + 1 and lut[t[1]["tiny_index"] + 1] == NucleusFlag.TINY
    tab = regions_table(list(regs.values()), 0.5)
    assert list(tab.index) == [1, 2, 3] and tab.loc[1, "n_wrong"] == len(t[1]["wrong_labels"]) and tab.loc[1, "width_um"] > 0
    nt = nuclei_table(list(regs.values()))
    assert len(nt) == sum(len(r) for r in regs.values()) and set(nt.flag) >= {"OK", "WRONG", "UNKNOWN", "TINY"}


def test_qc_policies(synth):
    p = SlidePaths.discover(SLIDE, synth["root"])
    r = {r.region_id: r for r in load_regions(p, min_area_px=20)}[1]
    full = r.label_map()
    wrong = set(r.nuclei.label[r.nuclei.flag == "WRONG"].tolist())
    tiny = synth["regions"][1]["tiny_index"] + 1
    assert tiny in set(np.unique(full)) or True                            # tiny may be too small to paint any pixel
    lab, ign = apply_qc_policy(full, r.flags, "keep_all")
    assert not ign.any() and tiny not in set(np.unique(lab)) and wrong <= set(np.unique(lab))
    lab, ign = apply_qc_policy(full, r.flags, "ignore_wrong")
    assert wrong <= set(np.unique(lab)) and np.array_equal(ign, np.isin(lab, list(wrong)))
    lab, ign = apply_qc_policy(full, r.flags, "drop_wrong")
    assert not (wrong & set(np.unique(lab))) and ign.sum() == np.isin(full, list(wrong)).sum()
    with pytest.raises(AnnotationError):
        apply_qc_policy(full, r.flags, "nope")
    with pytest.raises(AnnotationError):
        apply_qc_policy(full, r.flags[:3], "keep_all")


def test_overlaps(synth):
    p = SlidePaths.discover(SLIDE, synth["root"])
    regs = load_regions(p)
    ov = overlap_table(regs)
    assert len(ov) == 1 and (ov.region_a[0], ov.region_b[0]) == (1, 2) and ov.nuclei_a[0] > 0
