#!/usr/bin/env python
"""Build the unified phenotype-label store of a slide, <slide>/phenotyping/{areas,labels,tagged_nuclei}.csv + geojson/ + README.txt,
from what the biologist delivered (kept unchanged in <slide>/phenotyping/original_delivery/). Idempotent; nothing is deleted.

Sources it understands:
  * HALO project exports with one box session per file ('Layer 2x' Polygon features)              -> SLIDE-0235 regions 11, 12
  * the annotation tool's window CSVs (tool polygons as WKT, a 'tag' column per session)         -> SLIDE-0235 window in region 1
  * HALO box sessions over a field of a slide whose images are pending ('Layer 10' / 'Layer 8')  -> SLIDE-0233 field 4, area 5
  * the annotation tool's per-session exports of 1024^2 windows (only the tagged cells, WKT polygons)   -> SLIDE-0236 windows W1-W4
A label is attached to the HALO nucleus whose centroid lies in it (if several, the one the label covers most; else the nucleus the
label covers at least a fifth of). For slides the pipeline can rasterize, nucleus ids are the pipeline's label ids of that region;
``region_id`` says which region (a window can span two).

    conda activate dl
    python scripts/build_phenotyping.py --slides SLIDE-0235 SLIDE-0233 SLIDE-0236
"""
from __future__ import annotations

import argparse
import collections
import json
import re
import shutil
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from scipy import ndimage as ndi
from shapely import wkt
from shapely.geometry import Polygon, box as sbox, mapping, shape

from mxdata import DATA_ROOT, apply_qc_policy, ingest, load_config, load_regions
from mxdata.data import LiveSource
from mxdata.phenotyping import AREA_COLUMNS, LABEL_COLUMNS, NUCLEI_COLUMNS, PHENOTYPES, phenotyping_dir

REPO = Path(__file__).resolve().parents[1]
PHENO_OF_NAME = [("CD4", "CD4+ T cell"), ("CD8", "CD8+ T cell"), ("FoxP3", "Treg"), ("FOXP3", "Treg"), ("PanCK", "Epithelial"), ("Tcell", "T cell")]
WINDOW_SESSIONS = {"_PanCK.csv": "Epithelial", ".Tcell.csv": "T cell", ".CD4+Tcell.csv": "CD4+ T cell", ".CD8+Tcell.csv": "CD8+ T cell"}
SESSION_RE_0236 = re.compile(r"SLIDE-0236_Final_x(\d+)_y(\d+)_l0__(\w+)\.csv$")   # one tool export per (window, phenotype session)
WINDOW_PX_0236 = 1024
FOV_233 = {"R2": (10833, 29163, 11842, 29914),                      # the fields of SLIDE-0233 with boxes, in the HALO frame
           "R4": (10278, 27756, 11287, 28507)}
BOX_LAYERS_233 = {"region 4 slide 233 cd3 t cells only annotations.geojson": ("R4", "T cell", "Layer 10"),
                  "region 4 slide 233 cd3 and cd4 t cells annotations.geojson": ("R4", "CD4+ T cell", "Layer 10"),
                  "region 4 slide 233 cd3 and cd8 t cells annotations.geojson": ("R4", "CD8+ T cell", "Layer 10"),
                  "region 4 slide 233 treg t cells annotations.geojson": ("R4", "Treg", "Layer 10"),
                  "region 4 slide 233 panck+ cells annotations.geojson": ("R4", "Epithelial", "Layer 10"),
                  "slide 233 t cells cd3+ only.geojson": ("A5", "T cell", "Layer 8"),
                  # the second delivery: field 2, its PanCK session drawn in a different layer than its T-cell sessions
                  "SLIDE-0233_Final_region_2_x10843_y29162_I0.Epithelial_PanCK.geojson": ("R2", "Epithelial", "Layer 11"),
                  "SLIDE-0233_Final_region_2_x10843_y29162_I0.Tcells.geojson": ("R2", "T cell", "Layer 8"),
                  "SLIDE-0233_Final_region_2_x10843_y29162_I0.Tcells_CD4.geojson": ("R2", "CD4+ T cell", "Layer 8"),
                  "SLIDE-0233_Final_region_2_x10843_y29162_I0.Tcells_CD8.geojson": ("R2", "CD8+ T cell", "Layer 8"),
                  "SLIDE-0233_Final_region_2_x10843_y29162_I0.Tcells_FoxP3.geojson": ("R2", "Treg", "Layer 8")}


def pheno_of(name: str) -> str:
    return next(v for k, v in PHENO_OF_NAME if k in name)


def fname(area: str, phenotype: str) -> str:
    return f"{area}_{phenotype.replace(' ', '_').replace('+', 'pos')}.geojson"


class Nuclei:
    """The nuclei of one area: centroids, areas, flags, and a way to find which nucleus a label lands on."""

    def __init__(self, ids, cx, cy, area, flagged, labels=None, origin=(0, 0), polys=None, space="pipeline", region=0, parts=None):
        self.ids, self.cx, self.cy, self.area, self.flagged = np.asarray(ids), np.asarray(cx), np.asarray(cy), np.asarray(area), np.asarray(flagged)
        self.labels, self.origin, self.polys, self.space, self.parts = labels, origin, polys, space, parts
        self.region = np.broadcast_to(np.asarray(region), self.ids.shape).copy()

    @classmethod
    def concat(cls, parts):
        """The nuclei of several regions as one set (a tool window may lie over two HALO regions)."""
        cat = lambda a: np.concatenate([getattr(p, a) for p in parts])
        return cls(cat("ids"), cat("cx"), cat("cy"), cat("area"), cat("flagged"), space=parts[0].space, region=cat("region"), parts=list(parts))

    def coverage(self, g):
        """Fraction of each nucleus covered by polygon ``g``."""
        if self.parts is not None:
            return np.concatenate([p.coverage(g) for p in self.parts])
        if self.labels is not None:
            m = np.zeros(self.labels.shape, np.uint8)
            cv2.fillPoly(m, [np.round(np.array(g.exterior.coords) - self.origin).astype(np.int32)], 1)
            return np.bincount(self.labels[m.astype(bool)], minlength=len(self.ids) + 1)[1:] / np.maximum(self.area, 1)
        return np.array([g.intersection(p).area / max(p.area, 1e-6) for p in self.polys])

    def attach(self, g):
        """(chosen nucleus id or 0, n inside, others) for label polygon ``g``."""
        x0, y0, x1, y1 = g.bounds
        inside = np.nonzero((self.cx >= x0) & (self.cx <= x1) & (self.cy >= y0) & (self.cy <= y1))[0]
        if len(inside) == 0:
            cov = self.coverage(g); inside = np.array([int(cov.argmax())]) if len(cov) and cov.max() > 0.2 else np.array([], int)
        elif len(inside) > 1:
            cov = self.coverage(g); inside = inside[np.argsort(-cov[inside])]
        chosen = int(self.ids[inside[0]]) if len(inside) else 0
        return chosen, int(len(inside)), ";".join(str(int(self.ids[i])) for i in inside[1:])


def pipeline_nuclei(mx, region, markers, level) -> Nuclei:
    """The nuclei of a HALO region as the pipeline rasterizes them (label ids), with the QC flags."""
    rs = LiveSource(mx, [region], markers, level).load(f"{mx.slide if hasattr(mx, 'slide') else region.key.split('_R')[0]}_R{region.region_id:02d}")
    labels, ignore = apply_qc_policy(rs.labels, rs.flags, "ignore_wrong")
    ids = np.arange(1, rs.labels.max() + 1); cen = np.array(ndi.center_of_mass(rs.labels > 0, rs.labels, ids)); area = np.bincount(rs.labels.ravel(), minlength=len(ids) + 1)[1:]
    flagged = ndi.mean(ignore.astype(float), rs.labels, ids) > 0.5
    b = region.box
    return Nuclei(ids, cen[:, 1] + b.x0, cen[:, 0] + b.y0, area, flagged, labels=rs.labels, origin=(b.x0, b.y0), region=region.region_id)


class Store:
    def __init__(self, slide: str, root: Path):
        self.slide, self.dir = slide, phenotyping_dir(slide, root)
        (self.dir / "geojson").mkdir(parents=True, exist_ok=True); (self.dir / "original_delivery").mkdir(exist_ok=True)
        self.areas, self.labels, self.nuclei = [], [], {}

    def add_area(self, area, kind, box, frame, images, nuc: Nuclei | None, sessions, source):
        self.areas.append({"slide": self.slide, "area": area, "kind": kind, "x0": box[0], "y0": box[1], "x1": box[2], "y1": box[3], "frame": frame,
                           "images_available": images, "nucleus_id_space": nuc.space if nuc else "", "n_nuclei": len(nuc.ids) if nuc else 0,
                           "n_flagged": int(nuc.flagged.sum()) if nuc else 0, "sessions": ";".join(sessions), "source": source})
        if nuc is not None:
            self.nuclei[area] = nuc

    def add_labels(self, area, set_, phenotype, geoms, kind, negatives, source_file):
        nuc = self.nuclei.get(area); feats = []
        for k, g in enumerate(geoms):
            x0, y0, x1, y1 = g.bounds; lid = f"{area}_{phenotype.replace(' ', '').replace('+', 'pos')}_{k:04d}"
            chosen, n_in, others = nuc.attach(g) if nuc is not None else (0, 0, "")
            j = int(np.nonzero(nuc.ids == chosen)[0][0]) if chosen else -1
            self.labels.append({"slide": self.slide, "area": area, "set": set_, "phenotype": phenotype, "label_id": lid, "kind": kind,
                                "x0": x0, "y0": y0, "x1": x1, "y1": y1, "cx": g.centroid.x, "cy": g.centroid.y, "nucleus_id": chosen,
                                "nucleus_cx": float(nuc.cx[j]) if chosen else np.nan, "nucleus_cy": float(nuc.cy[j]) if chosen else np.nan,
                                "nucleus_flagged": bool(nuc.flagged[j]) if chosen else None, "n_nuclei_inside": n_in, "other_nuclei": others,
                                "negatives_defined": negatives, "source_file": source_file, "region_id": int(nuc.region[j]) if chosen else 0})
            feats.append({"type": "Feature", "geometry": mapping(g), "properties": {"classification": {"name": phenotype}, "label_id": lid, "area": area, "nucleus_id": chosen}})
        (self.dir / "geojson" / fname(area, phenotype)).write_text(json.dumps({"type": "FeatureCollection", "features": feats}))
        print(f"  {self.slide} {area:3s} {phenotype:12s} {len(geoms):5d} {'boxes' if kind == 'box' else 'polygons'}")

    def write(self):
        A = pd.DataFrame(self.areas, columns=AREA_COLUMNS); L = pd.DataFrame(self.labels, columns=LABEL_COLUMNS)
        rows = []
        for area, nuc in self.nuclei.items():
            la = L[(L.area == area) & (L.nucleus_id > 0)]
            for nid, g in la.groupby("nucleus_id"):
                j = int(np.nonzero(nuc.ids == nid)[0][0])
                rows.append({"slide": self.slide, "area": area, "nucleus_id": int(nid), "cx": float(nuc.cx[j]), "cy": float(nuc.cy[j]), "area_px": float(nuc.area[j]),
                             "flagged": bool(nuc.flagged[j]), "tags": ";".join(sorted(set(g.phenotype))), "n_labels": len(g), "region_id": int(nuc.region[j])})
        N = pd.DataFrame(rows, columns=NUCLEI_COLUMNS)
        A.to_csv(self.dir / "areas.csv", index=False); L.to_csv(self.dir / "labels.csv", index=False); N.to_csv(self.dir / "tagged_nuclei.csv", index=False)
        for stale in ("boxes",):                                            # the pre-unification layout of 2026-09-07 is superseded
            if (self.dir / stale).is_dir():
                shutil.rmtree(self.dir / stale)
        return A, L, N


def build_0235(root: Path) -> Store:
    slide = "SLIDE-0235"; st = Store(slide, root); od = st.dir / "original_delivery"
    cfg = load_config(REPO / "configs" / "dataset.yaml"); paths, mx, _ = ingest(slide, root); markers = list(cfg.markers)
    regions = {r.region_id: r for r in load_regions(paths, [1, 11, 12], cfg.min_area_px)}
    # -- boxes on regions 11 and 12 (HALO project exports; the boxes are the 'Layer 2x' polygons) --
    files = sorted(od.glob("SLIDE-0235_Final_region_*.geojson"))
    for rid in (11, 12):
        r = regions[rid]; nuc = pipeline_nuclei(mx, r, markers, cfg.level); b = r.box
        sessions = sorted({pheno_of(f.name.split("I0.")[1]) for f in files if f"region_{rid}_" in f.name})
        st.add_area(f"R{rid}", "region", (b.x0, b.y0, b.x1, b.y1), "R0", True, nuc, sessions, "HALO_layers_0235 + phenotyping/original_delivery")
        for f in files:
            if f"region_{rid}_" not in f.name:
                continue
            ph = pheno_of(f.name.split("I0.")[1]); d = json.load(open(f))
            geoms = [shape(ft["geometry"]) for ft in d["features"] if ft["geometry"]["type"] == "Polygon" and str((ft["properties"].get("classification") or {}).get("name", "")).startswith("Layer")]
            st.add_labels(f"R{rid}", "B", ph, geoms, "box", True, f.name)
    # -- the annotation-tool window in region 1 (moved from <root>/annotatedCells on first build) --
    win_dir = od / "window_R1"
    legacy = root / "annotatedCells"
    if legacy.is_dir() and not win_dir.is_dir():
        shutil.move(str(legacy), str(win_dir)); print(f"  moved {legacy} -> {win_dir}")
    if win_dir.is_dir():
        stem = next(iter(win_dir.glob("*_l0.csv"))).name[:-4]; m = re.search(r"_x(\d+)_y(\d+)_l(\d+)$", stem); wx, wy = int(m.group(1)), int(m.group(2)); size = 1024
        r1 = regions[1]; nuc = pipeline_nuclei(mx, r1, markers, cfg.level)
        st.add_area("W1", "window", (wx, wy, wx + size, wy + size), "R0", True, nuc, list(WINDOW_SESSIONS.values()), "phenotyping/original_delivery/window_R1 (annotation tool)")
        for suf, ph in WINDOW_SESSIONS.items():
            d = pd.read_csv(win_dir / f"{stem}{suf}"); tagged = d[d.tag.astype(str).str.lower() != "untagged"]
            geoms = [wkt.loads(p) for p in tagged.polygon]
            st.add_labels("W1", "A", ph, geoms, "polygon", True, f"{stem}{suf}")
    return st


def build_0233(root: Path) -> Store:
    slide = "SLIDE-0233"; st = Store(slide, root); od = st.dir / "original_delivery"
    first = root / slide / "original_delivery"                                # the first delivery of the slide (HALO export, QC tables, boxes)
    big = next(f for f in (first / "CellDIVE_SLIDE-0233_R2_IMMUNE_RICH.geojson", od / "CellDIVE_SLIDE-0233_R1_Region_4_Annotations_Immune_Rich.geojson") if f.is_file())
    export = json.load(open(big))
    polys = []
    for ft in export["features"]:
        if (ft["properties"].get("classification") or {}).get("name") == "Nuclei":
            g = Polygon(ft["geometry"]["coordinates"]); g = g if g.is_valid else g.buffer(0)
            if not g.is_empty:
                polys.append(g)
    QC_233 = {"R2": "Slide_233_Region_2_Immune_Rich - pranav 233 region 2 v1.csv",
              "R4": "Slide_233_Region_4_Immune_Rich - pranav 233 region 4 v1 (1).csv"}
    have_images = any((root / slide).glob("*_R001_*.ome.tif"))       # the annotated scan region arrived 2026-09-07
    for area, box in FOV_233.items():
        qc = pd.read_csv(first / QC_233[area]); qc.columns = [c.strip() for c in qc.columns]
        gcol = next(c for c in qc.columns if c.lower().replace(" ", "_") in ("ground_truths", "flag"))
        qb = qc[["XMin", "YMin", "XMax", "YMax"]].values.astype(float)
        qflag = qc[gcol].fillna("").astype(str).str.strip().str.lower().eq("wrong").values
        fov = sbox(*box); nucp = [g for g in polys if fov.contains(g.centroid)]
        def flag(g):
            d = np.abs(qb - np.array(g.bounds)).sum(1); j = int(d.argmin()); return bool(qflag[j]) if d[j] < 8 else False
        nuc = Nuclei(np.arange(1, len(nucp) + 1), [g.centroid.x for g in nucp], [g.centroid.y for g in nucp], [g.area for g in nucp], [flag(g) for g in nucp], polys=nucp, space="export", region=int(area[1:]))
        sessions = sorted({ph for a, ph, _ in BOX_LAYERS_233.values() if a == area})
        st.add_area(area, "field", box, "R1", have_images, nuc, sessions,
                    f"{big.name} + {QC_233[area]} (HALO scan region 1; its images verified aligned 2026-09-07)")
        (st.dir / "geojson" / f"{area}_HALO_nuclei.geojson").write_text(json.dumps({"type": "FeatureCollection", "features": [
            {"type": "Feature", "geometry": mapping(g), "properties": {"classification": {"name": "Nuclei"}, "nucleus_id": i + 1, "flagged": bool(nuc.flagged[i])}} for i, g in enumerate(nucp)]}))
    a5 = None
    for f, (area, ph, layer) in BOX_LAYERS_233.items():
        src = first / f if (first / f).is_file() else od / f
        d = json.load(open(src))
        geoms = []
        for ft in d["features"]:                                   # HALO exports a box session as Polygon in some files and as a
            if (ft["properties"].get("classification") or {}).get("name") != layer:   # closed LineString ring in others; both are
                continue                                           # the same rectangle, so take either
            g = ft["geometry"]
            if g["type"] == "Polygon":
                geoms.append(shape(g))
            elif g["type"] == "LineString" and len(g["coordinates"]) >= 4:
                ring = g["coordinates"] + ([g["coordinates"][0]] if g["coordinates"][0] != g["coordinates"][-1] else [])
                poly = Polygon(ring)
                geoms.append(poly if poly.is_valid else poly.buffer(0))
        if area == "A5" and a5 is None:
            xs = [g.bounds for g in geoms]; a5 = (min(b[0] for b in xs), min(b[1] for b in xs), max(b[2] for b in xs), max(b[3] for b in xs))
            st.add_area("A5", "area", a5, "R1", have_images, None, ["T cell"], f"{f} (no HALO nuclei here)")
        st.add_labels(area, "C", ph, geoms, "box", True, f)
    return st


def build_0236(root: Path) -> Store:
    """The test slide's phenotype reference: the annotation tool's sessions over four 1024^2 windows (delivered 2026-10-13 as
    slide236regions1_4/<timestamp>_<annotator>/SLIDE-0236_Final_x<X>_y<Y>_l0__<session>.csv — only the tagged cells are exported, as
    the tool's own cell polygons). Windows are named W1..W4 left to right; labels are attached to the HALO nuclei of every region the
    window overlaps (W1: regions 3 and 4; W4: region 2; W2 and W3 lie outside the HALO regions -> nucleus_id 0)."""
    slide = "SLIDE-0236"; st = Store(slide, root); od = st.dir / "original_delivery"
    cfg = load_config(REPO / "configs" / "dataset.yaml"); paths, mx, _ = ingest(slide, root); markers = list(cfg.markers)
    files = sorted(f for f in od.rglob("SLIDE-0236_Final_x*_l0__*.csv") if SESSION_RE_0236.search(f.name))
    tiles = sorted({(int(m.group(1)), int(m.group(2))) for f in files if (m := SESSION_RE_0236.search(f.name))})
    regions = load_regions(paths, None, cfg.min_area_px); cache = {}
    for k, (wx, wy) in enumerate(tiles, 1):
        area, box = f"W{k}", (wx, wy, wx + WINDOW_PX_0236, wy + WINDOW_PX_0236)
        over = [r for r in regions if r.box.x0 < box[2] and r.box.x1 > box[0] and r.box.y0 < box[3] and r.box.y1 > box[1]]
        for r in over:
            cache.setdefault(r.region_id, pipeline_nuclei(mx, r, markers, cfg.level))
        nuc = Nuclei.concat([cache[r.region_id] for r in over]) if over else None
        sess = [(f, pheno_of(SESSION_RE_0236.search(f.name).group(3))) for f in files if SESSION_RE_0236.search(f.name).groups()[:2] == (str(wx), str(wy))]
        st.add_area(area, "window", box, "R0", True, nuc, sorted({ph for _, ph in sess}),
                    "phenotyping/original_delivery/slide236regions1_4 (annotation tool; HALO " + (", ".join(f"R{r.region_id}" for r in over) or "none") + ")")
        for f, ph in sess:
            d = pd.read_csv(f)
            d = d[(d.status.astype(str) == "labeled") & (d.tag.astype(str) == "accepted")]
            st.add_labels(area, "A", ph, [wkt.loads(p) for p in d.polygon], "polygon", True, str(f.relative_to(od)))
    return st


def readme(st: Store, A, L, N):
    per = L.groupby(["area", "phenotype"]).size().unstack(fill_value=0).reindex(columns=PHENOTYPES, fill_value=0)
    lines = [f"{st.slide}/phenotyping — the biologist's phenotype labels, unified (built by scripts/build_phenotyping.py; originals untouched in original_delivery/).", "",
             "areas.csv           one row per annotated area: kind (region | window | field | area), its box (x0 y0 x1 y1), frame (R0 = the slide's images the pipeline",
             "                    reads; HALO-R1 = a HALO scan region whose images are pending), images_available, nucleus_id_space (pipeline = label ids the pipeline",
             "                    rasterizes from the HALO layer; export = index in the delivered HALO export), n_nuclei / n_flagged, sessions (phenotypes with an",
             "                    exhaustive session there: an unlabeled cell is a negative for them).",
             "labels.csv          one row per human label: set (A window tags | B region boxes | C field boxes), phenotype (configs/panel.yaml names), kind",
             "                    (box | polygon), its bounds and centroid, the HALO nucleus it lands on (nucleus_id, 0 = none; centroid, flag) with",
             "                    n_nuclei_inside and other_nuclei when several are inside, negatives_defined, source_file, region_id (the HALO",
             "                    region of that nucleus, 0 = none).",
             "tagged_nuclei.csv   one row per HALO nucleus with at least one label: tags (';'-joined), n_labels, its QC flag, region_id — the phenotype reference.",
             "geojson/            <area>_<phenotype>.geojson: the labels as Polygon features (classification.name = phenotype, nucleus_id in properties).",
             "original_delivery/  as delivered.", "",
             "A label locates a cell; it never defines a boundary. Labels on flagged (merged) nuclei and boxes holding two nuclei are kept and marked.", "",
             "Areas: " + "; ".join(f"{r.area} ({r.kind}, {r.frame}, {r.n_nuclei} nuclei, {r.n_flagged} flagged, sessions: {r.sessions})" for r in A.itertuples()), "",
             "Labels per area:", per.to_string(), "", f"Tagged nuclei: {len(N)} ({int(N.flagged.sum())} flagged); labels without a nucleus: {int((L.nucleus_id == 0).sum())} of {len(L)}.", ""]
    if st.slide == "SLIDE-0235":
        lines += ["Notes: region 11's CD4 session was delivered without boxes (re-export needed). Regions 11 and 12 overlap (x 9317-10132, y 8906-9574).",
                  "The window W1 (1024^2 px at (8192, 3072), 74 % inside region 1) carries the annotation tool's polygons; its labels are attached to the",
                  "HALO nuclei of region 1 where the window overlaps it, and have nucleus_id 0 in the strip below region 1. Region 1 is",
                  "expert-reviewed since the QC delivery of 2026-10-13 (every region of the slide now is), so W1's nuclei carry QC flags."]
    if st.slide == "SLIDE-0236":
        lines += ["Notes: four 1024^2 windows of the annotation tool at y 7168 (x 6144, 7168, 8192, 9216 -> W1..W4), delivered 2026-10-13 as",
                  "slide236regions1_4/<timestamp>_<annotator>/ (sureshp1: 18 sessions, dimitrn: W2 T cell and W3 CD4+ T cell). Each file holds",
                  "only the cells tagged in that session (status 'labeled', tag 'accepted'), as the tool's own polygons (model 'instanseg');",
                  "cell_id is per session, so the same cell has different ids across sessions -> phenotypes are related by geometry only.",
                  "negatives_defined = True assumes each session tagged every cell of its type in the window (the untagged cells are not exported).",
                  "The per-cell marker means in the files are the tool's; verified against the slide's R0 images 2026-10-13 (CD3 x17-81 inside the",
                  "T-cell polygons, PanCK x1.4-2.9 inside the epithelial ones). W1 lies over HALO regions 3 and 4, W4 over region 2; W2 and W3",
                  "have no HALO nuclei (nucleus_id 0). The slide is the TEST slide: these labels are for scoring, never for calibration."]
    if st.slide == "SLIDE-0233":
        lines += ["!! All coordinates are in the HALO frame of scan region 1 (CellDIVE_SLIDE-0233_R1.afi); the delivered *_R000_* images do not contain",
                  "   these areas. Field 2 (announced) will be in the same frame. Needed: the nine SLIDE-0233_*_R001_*_FINAL_AFR_F.ome.tif files or a",
                  "   re-export of the layers on the R000 frame; then rebuild with pipeline nucleus ids. See ../README.txt."]
    (st.dir / "README.txt").write_text("\n".join(lines) + "\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--slides", nargs="+", default=["SLIDE-0235", "SLIDE-0233", "SLIDE-0236"]); ap.add_argument("--root", default=None)
    args = ap.parse_args(); root = Path(args.root).expanduser() if args.root else DATA_ROOT
    for s in args.slides:
        print(f"== {s}")
        st = {"SLIDE-0235": build_0235, "SLIDE-0233": build_0233, "SLIDE-0236": build_0236}[s](root)
        A, L, N = st.write(); readme(st, A, L, N)
        print(f"  wrote {st.dir}: {len(A)} areas, {len(L)} labels, {len(N)} tagged nuclei")


if __name__ == "__main__":
    main()
