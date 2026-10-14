"""Human phenotype labels of a slide, unified: ``<root>/<slide>/phenotyping/``.

The biologist labels cells in two ways — bounding boxes drawn in HALO around cells of one phenotype per session, and tags on the
annotation tool's own cell polygons — and none of it is segmentation ground truth: a label *locates* a cell. This module reads the
one layout every slide shares (built by ``scripts/build_phenotyping.py`` from ``phenotyping/original_delivery/``)::

    <slide>/phenotyping/
        areas.csv            one row per annotated area (a HALO region, a tool window, a field of a pending slide)
        labels.csv           one row per human label (box or polygon), with the HALO nucleus it lands on and its HALO region
        tagged_nuclei.csv    one row per HALO nucleus that carries at least one tag (the phenotype reference)
        geojson/             <area>_<phenotype>.geojson — the labels as polygons, classification.name = phenotype
        original_delivery/   everything as delivered, unchanged
        README.txt

Frames: ``areas.frame`` is ``R0`` when the coordinates are those of the slide's images the pipeline reads, ``HALO-R1`` when the
annotations live in a HALO scan region whose images are pending (SLIDE-0233); ``nucleus_id_space`` says what ``nucleus_id`` means
(``pipeline`` = the label id the pipeline rasterizes from the HALO layer of that region; ``export`` = the index in the delivered
HALO export); ``region_id`` is the HALO region that nucleus belongs to (0 = no nucleus), since a tool window can span two regions
(SLIDE-0236 W1 lies over regions 3 and 4). Phenotype names are those of ``configs/panel.yaml``.

    from mxdata.phenotyping import load_areas, load_labels, load_tagged_nuclei, summary
    labels = load_labels("SLIDE-0235")                          # every label of the slide
    t = load_tagged_nuclei("SLIDE-0235", area="R12")             # nuclei of region 12 with their tags
    summary("SLIDE-0235")                                        # labels per area x phenotype
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional, Sequence

import pandas as pd

from .errors import DatasetError
from .io import DATA_ROOT

PHENOTYPING = "phenotyping"
AREA_COLUMNS = ["slide", "area", "kind", "x0", "y0", "x1", "y1", "frame", "images_available", "nucleus_id_space", "n_nuclei", "n_flagged",
                "sessions", "source"]
LABEL_COLUMNS = ["slide", "area", "set", "phenotype", "label_id", "kind", "x0", "y0", "x1", "y1", "cx", "cy", "nucleus_id", "nucleus_cx",
                 "nucleus_cy", "nucleus_flagged", "n_nuclei_inside", "other_nuclei", "negatives_defined", "source_file", "region_id"]
NUCLEI_COLUMNS = ["slide", "area", "nucleus_id", "cx", "cy", "area_px", "flagged", "tags", "n_labels", "region_id"]
PHENOTYPES = ["Epithelial", "T cell", "CD4+ T cell", "CD8+ T cell", "Treg"]


def phenotyping_dir(slide: str, root: "str | Path | None" = None) -> Path:
    """``<root>/<slide>/phenotyping`` (root defaults to ``$MXDATA_ROOT``)."""
    return (Path(root).expanduser() if root else DATA_ROOT) / slide / PHENOTYPING


def _read(slide: str, name: str, columns: Sequence[str], root=None) -> pd.DataFrame:
    f = phenotyping_dir(slide, root) / name
    if not f.is_file():
        raise DatasetError(f"no {name} for {slide} at {f}", hint="run `python scripts/build_phenotyping.py --slides <slide>` on phenotyping/original_delivery/")
    df = pd.read_csv(f)
    if missing := [c for c in columns if c not in df.columns]:
        raise DatasetError(f"{f} lacks column(s) {missing}", hint="rebuild with scripts/build_phenotyping.py")
    return df


def load_areas(slide: str, root=None) -> pd.DataFrame:
    """The annotated areas of a slide (areas.csv)."""
    return _read(slide, "areas.csv", AREA_COLUMNS, root)


def load_labels(slide: str, root=None, area: Optional[str] = None, phenotype: Optional[str] = None) -> pd.DataFrame:
    """Every human label of a slide (labels.csv), optionally one area and/or one phenotype."""
    df = _read(slide, "labels.csv", LABEL_COLUMNS, root)
    if area is not None:
        df = df[df.area == area]
    if phenotype is not None:
        df = df[df.phenotype == phenotype]
    return df.reset_index(drop=True)


def load_tagged_nuclei(slide: str, root=None, area: Optional[str] = None) -> pd.DataFrame:
    """The HALO nuclei that carry tags (tagged_nuclei.csv); ``tags`` is ';'-joined. Optionally one area."""
    df = _read(slide, "tagged_nuclei.csv", NUCLEI_COLUMNS, root)
    if area is not None:
        df = df[df.area == area]
    return df.reset_index(drop=True)


def load_geojson(slide: str, area: str, phenotype: str, root=None) -> dict:
    """The labels of one area and phenotype as a GeoJSON FeatureCollection (Polygon features, classification.name = phenotype)."""
    f = phenotyping_dir(slide, root) / "geojson" / f"{area}_{phenotype.replace(' ', '_').replace('+', 'pos')}.geojson"
    if not f.is_file():
        raise DatasetError(f"no labels for {slide} {area} {phenotype!r} at {f}")
    return json.load(open(f))


def summary(slide: str, root=None) -> pd.DataFrame:
    """Labels per area x phenotype, with the area's nuclei count and frame (what the notebooks print first)."""
    a = load_areas(slide, root).set_index("area")
    L = load_labels(slide, root)
    t = L.pivot_table(index="area", columns="phenotype", values="label_id", aggfunc="count", fill_value=0).reindex(columns=[p for p in PHENOTYPES], fill_value=0)
    t = t.reindex(a.index, fill_value=0)
    for i, (name, col) in enumerate([("kind", a.kind), ("nuclei", a.n_nuclei), ("flagged", a.n_flagged), ("frame", a.frame), ("images", a.images_available)]):
        t.insert(i, name, col)
    t.columns.name = None
    return t


def tags_of(nuclei: pd.DataFrame) -> pd.DataFrame:
    """Long form of ``tagged_nuclei``: one row per (nucleus, tag)."""
    d = nuclei.assign(tag=nuclei.tags.astype(str).str.split(";")).explode("tag")
    return d[d.tag.astype(str) != ""].reset_index(drop=True)


__all__ = ["AREA_COLUMNS", "LABEL_COLUMNS", "NUCLEI_COLUMNS", "PHENOTYPES", "PHENOTYPING", "load_areas", "load_geojson", "load_labels",
           "load_tagged_nuclei", "phenotyping_dir", "summary", "tags_of"]
