"""The unified phenotype-label store: loaders validate the schema and filter by area / phenotype."""
import json

import pandas as pd
import pytest

from mxdata.errors import DatasetError
from mxdata.phenotyping import (
    AREA_COLUMNS,
    LABEL_COLUMNS,
    NUCLEI_COLUMNS,
    load_areas,
    load_geojson,
    load_labels,
    load_tagged_nuclei,
    phenotyping_dir,
    summary,
    tags_of,
)


@pytest.fixture
def store(tmp_path):
    d = phenotyping_dir("SLIDE-TEST", tmp_path)
    (d / "geojson").mkdir(parents=True)
    pd.DataFrame([{"slide": "SLIDE-TEST", "area": "R1", "kind": "region", "x0": 0, "y0": 0, "x1": 100, "y1": 100, "frame": "R0", "images_available": True,
                   "nucleus_id_space": "pipeline", "n_nuclei": 3, "n_flagged": 1, "sessions": "T cell;Epithelial", "source": "test"}], columns=AREA_COLUMNS).to_csv(d / "areas.csv", index=False)
    rows = [dict(slide="SLIDE-TEST", area="R1", set="B", phenotype=ph, label_id=f"R1_{i}", kind="box", x0=i, y0=i, x1=i + 10, y1=i + 10, cx=i + 5, cy=i + 5, nucleus_id=n,
                 nucleus_cx=i + 5, nucleus_cy=i + 5, nucleus_flagged=False, n_nuclei_inside=1, other_nuclei="", negatives_defined=True, source_file="f")
            for i, (ph, n) in enumerate([("T cell", 1), ("T cell", 2), ("Epithelial", 3), ("Epithelial", 0)])]
    pd.DataFrame(rows, columns=LABEL_COLUMNS).to_csv(d / "labels.csv", index=False)
    pd.DataFrame([dict(slide="SLIDE-TEST", area="R1", nucleus_id=1, cx=5, cy=5, area_px=40, flagged=False, tags="T cell", n_labels=1),
                  dict(slide="SLIDE-TEST", area="R1", nucleus_id=3, cx=7, cy=7, area_px=40, flagged=True, tags="Epithelial;T cell", n_labels=2)], columns=NUCLEI_COLUMNS).to_csv(d / "tagged_nuclei.csv", index=False)
    (d / "geojson" / "R1_T_cell.geojson").write_text(json.dumps({"type": "FeatureCollection", "features": [{"type": "Feature", "geometry": {"type": "Polygon", "coordinates": [[[0, 0], [10, 0], [10, 10], [0, 10], [0, 0]]]}, "properties": {"classification": {"name": "T cell"}}}]}))
    return tmp_path


def test_loaders_filter(store):
    assert len(load_areas("SLIDE-TEST", store)) == 1
    L = load_labels("SLIDE-TEST", store)
    assert len(L) == 4 and list(L.columns) == LABEL_COLUMNS
    assert len(load_labels("SLIDE-TEST", store, phenotype="T cell")) == 2 and len(load_labels("SLIDE-TEST", store, area="R9")) == 0
    N = load_tagged_nuclei("SLIDE-TEST", store, area="R1")
    assert len(N) == 2 and len(tags_of(N)) == 3
    assert len(load_geojson("SLIDE-TEST", "R1", "T cell", store)["features"]) == 1


def test_summary(store):
    t = summary("SLIDE-TEST", store)
    assert t.loc["R1", "T cell"] == 2 and t.loc["R1", "Epithelial"] == 2 and t.loc["R1", "Treg"] == 0 and t.loc["R1", "nuclei"] == 3


def test_missing_and_schema(store, tmp_path):
    with pytest.raises(DatasetError):
        load_labels("SLIDE-NONE", tmp_path)
    (phenotyping_dir("SLIDE-TEST", store) / "labels.csv").write_text("slide,area\nx,y\n")
    with pytest.raises(DatasetError):
        load_labels("SLIDE-TEST", store)
