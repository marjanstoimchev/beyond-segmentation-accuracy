import pickle
import shutil

import numpy as np
import pytest

from mxdata import Box, MxSlide, SlideError, SlidePaths
from tests.conftest import SLIDE, H, W, write_ome_pyramid


def test_halo_layer_names_of_both_exports():
    """HALO exports name layers CellDIVE_<slide>_Region_<k>.geojson (SLIDE-0235) or R<k>_HALO_<slide>_Region_<k>.geojson (SLIDE-0236)."""
    from mxdata.io.paths import _HALO_RE
    assert _HALO_RE.search("CellDIVE_SLIDE-0235_Region_12.geojson").groups() == ("SLIDE-0235", "12")
    assert _HALO_RE.search("R4_HALO_SLIDE-0236_Region_4.geojson").groups() == ("SLIDE-0236", "4")
    assert _HALO_RE.search("notes_Region_4.txt") is None


def test_discovery(synth):
    p = SlidePaths.discover(SLIDE, synth["root"])
    assert p.markers == ["DAPI", "PanCK"] and p.region_ids == [1, 2, 3] and sorted(p.qc_files) == [1, 2]
    assert [x.name for x in p.unparsed_tiffs] == ["not_a_channel.tif"] and len(p.ss_images) == 1
    assert p.channel("dapi").is_dapi
    assert "SLIDE-9999" in p.signature() and "channels (2)" in p.summary()
    with pytest.raises(SlideError):
        p.channel("CD45")
    with pytest.raises(SlideError):
        SlidePaths.discover("SLIDE-0000", synth["root"])


def test_geometry_is_read_not_assumed(synth):
    s = MxSlide.from_paths(SlidePaths.discover(SLIDE, synth["root"]))
    assert (s.height, s.width) == (H, W) and s.n_levels == 3 and s.dtype == np.uint16
    assert np.allclose(s.downsamples, [1, 2, 4]) and s.pixel_size_um == pytest.approx(0.5)
    assert s.tile == (64, 64) and s.compression == "LZW" and s.excluded == []
    assert s.level_for_max_dim(300) == 2 and s.level_for_max_dim(10_000) == 0 and s.has_marker("panck") and not s.has_marker("CD45")


def test_reads_are_level0_boxes_scaled_to_the_level(synth):
    s = MxSlide.from_paths(SlidePaths.discover(SLIDE, synth["root"]))
    dapi = np.load(synth["arrays"]["DAPI"])
    b = Box(101, 51, 300, 180)
    a = s.read_region(b, ["DAPI"], level=0)
    assert a.shape == (1, 129, 199) and a.dtype == np.uint16
    np.testing.assert_array_equal(a[0], dapi[51:180, 101:300])          # lossless: bit-exact
    a2 = s.read_region(b, ["DAPI", "PanCK"], level=2)
    x0, y0, x1, y1 = b.at_downsample(4)
    assert a2.shape == (2, y1 - y0, x1 - x0)
    assert s.read_channel(b, "PanCK").shape == (129, 199)
    thumb, ds = s.thumbnail(max_dim=300)
    assert ds == 4 and thumb.shape == (2, H // 4, W // 4)
    s.close()


def test_refusals_name_themselves(synth):
    s = MxSlide.from_paths(SlidePaths.discover(SLIDE, synth["root"]))
    with pytest.raises(SlideError, match="not a channel"):
        s.read_region(Box(0, 0, 10, 10), ["CD45"])
    with pytest.raises(SlideError, match="outside"):
        s.read_region(Box(W + 10, 0, W + 20, 10))
    with pytest.raises(SlideError, match="level"):
        s.read_region(Box(0, 0, 10, 10), level=7)
    partial = s.read_region(Box(W - 5, H - 5, W + 50, H + 50))          # clamped, not empty
    assert partial.shape == (2, 5, 5)


def test_picklable_and_reads_after_unpickle(synth):
    s = MxSlide.from_paths(SlidePaths.discover(SLIDE, synth["root"]))
    s.read_region(Box(0, 0, 8, 8))                                       # open a handle first
    s2 = pickle.loads(pickle.dumps(s))
    assert s2._files == {} and s2.markers == s.markers
    np.testing.assert_array_equal(s2.read_region(Box(0, 0, 8, 8)), s.read_region(Box(0, 0, 8, 8)))


def test_mismatched_grid_and_unreadable_channels_are_excluded(synth, tmp_path):
    root = tmp_path / "root"
    shutil.copytree(synth["root"] / SLIDE, root / SLIDE)
    # a channel on another resolution grid, and one that is not a TIFF at all
    write_ome_pyramid(root / SLIDE / f"{SLIDE}_5.0.4_R000_Cy5_CD4-RM4-AF647_FINAL_AFR_F.ome.tif",
                      np.zeros((H // 2, W // 2), np.uint16))
    (root / SLIDE / f"{SLIDE}_9.0.4_R000_Cy5_CD45-CST-AF647_FINAL_AFR_F.ome.tif").write_bytes(b"garbage")
    s = MxSlide.from_paths(SlidePaths.discover(SLIDE, root))
    assert s.markers == ["DAPI", "PanCK"]
    reasons = {e["marker"]: e["reason"] for e in s.excluded}
    assert "grid" in reasons["CD4"] and "unreadable" in reasons["CD45"]
