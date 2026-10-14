import pickle

import numpy as np
import pytest

from mxdata import Box, parse_channel_filename


def test_box_basics():
    b = Box(10, 20, 30, 50)
    assert b.shape == (30, 20) and b.area == 600 and b.xyxy == (10, 20, 30, 50)
    assert b.pad(5).xyxy == (5, 15, 35, 55) and b.pad(50, clamp_to=(40, 60)).xyxy == (0, 0, 40, 60)
    assert b.intersect(Box(25, 40, 100, 100)).xyxy == (25, 40, 30, 50) and b.intersect(Box(30, 50, 40, 60)) is None
    assert b.overlaps(Box(29, 49, 31, 51)) and not b.overlaps(Box(30, 50, 40, 60))
    assert b.contains(10, 20) and not b.contains(30, 20) and b.inside(30, 50) and not b.inside(29, 50)
    assert Box.from_points(np.array([[1, 2], [4, 6]])).xyxy == (1, 2, 5, 7)
    assert Box.union_all([b, Box(0, 0, 1, 1)]).xyxy == (0, 0, 30, 50)
    assert Box(**b.to_dict()) == b and pickle.loads(pickle.dumps(b)) == b


def test_box_downsample_never_loses_pixels():
    b = Box(3, 5, 30, 50)
    x0, y0, x1, y1 = b.at_downsample(4)
    assert (x0, y0, x1, y1) == (0, 1, 8, 13)
    assert x0 * 4 <= b.x0 and y0 * 4 <= b.y0 and x1 * 4 >= b.x1 and y1 * 4 >= b.y1


def test_empty_box_rejected():
    with pytest.raises(ValueError):
        Box(5, 5, 5, 10)
    with pytest.raises(ValueError):
        Box(0, 0, 10, 10).clamp(0, 0) if False else Box(10, 0, 5, 5)


def test_parse_channel_filename():
    ci = parse_channel_filename("SLIDE-0235_9.0.4_R000_FITC_PanCK-C11-AF488_FINAL_AFR_F.ome.tif")
    assert ci.marker == "PanCK" and ci.fluor == "FITC" and ci.round == 9 and ci.af_removed
    assert ci.clone == "C11" and ci.dye == "AF488" and ci.slide == "SLIDE-0235"
    d = parse_channel_filename("SLIDE-0235_1.0.4_R000_DAPI__FINAL_F.ome.tif")
    assert d.is_dapi and d.marker == "DAPI" and not d.af_removed and d.round == 1
    u = parse_channel_filename("SLIDE-0235_13.0.4_R000_Cy3_uPAR-RDPoly-AF555_FINAL_AFR_F.ome.tif")
    assert u.marker == "uPAR" and u.round == 13
    assert parse_channel_filename("random.tif") is None and parse_channel_filename("SLIDE-1_x.ome.tif") is None
