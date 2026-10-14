"""Checks against the real SLIDE-0235 folder; skipped when it is not present (CI never has it)."""
import pytest

from mxdata import DATA_ROOT, load_regions, validate
from mxdata.pipeline import ingest

pytestmark = pytest.mark.realdata
REAL = DATA_ROOT / "SLIDE-0235"


@pytest.fixture(scope="module")
def real():
    if not REAL.is_dir():
        pytest.skip(f"{REAL} not available")
    paths, slide, rep = ingest("SLIDE-0235")
    return paths, slide, rep


def test_slide_0235_facts(real):
    paths, slide, rep = real
    assert (slide.width, slide.height, slide.n_levels) == (20437, 13048, 6)
    assert slide.pixel_size_um == pytest.approx(0.325, abs=1e-3) and slide.markers == ["DAPI", "FOXP3", "CD3", "CD8", "LAP2", "CD4", "CD45", "PanCK", "uPAR"]
    assert paths.region_ids == list(range(1, 17)) and sorted(paths.qc_files) == list(range(1, 17))


def test_slide_0235_annotations_validate(real):
    paths, slide, _ = real
    regions = load_regions(paths, min_area_px=20)
    assert sum(len(r) for r in regions) == 21690
    rep = validate(slide, regions)
    assert rep.ok, rep.summary("error")
    assert all(r.qc_match["unmatched_qc"] == 0 for r in regions if r.has_qc)
