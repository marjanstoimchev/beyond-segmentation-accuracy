#!/usr/bin/env python
"""Panels of Figure 1 of the paper (LaTex/figures/slides/), composed by LaTeX (main.tex): the three benchmark slides with
their annotated regions colored by role under the leave-slide-out protocol (slide1-3.pdf), annotated cells per region with
the expert's flags (cells.pdf), full-resolution crops of a validation region of the development slide and of a test region
of the test slide with the expert annotation (val_crop.png, test_crop.png), the two channels the segmenters read (lap2.png,
dapi.png) and the network input they form, bare and with the annotation (input.png, input_ann.png). Everything is read from
the slides and the dataset; no model is involved. Panel letters, headers and the legend are typeset in LaTeX; only region
labels and scale bars live in the pixels.

    conda activate dl
    python scripts/figure_slides.py                     # -> LaTex/figures/slides/
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage as ndi
import pandas as pd
from matplotlib import patheffects as pe
from matplotlib.collections import LineCollection
import cv2

from mxdata import DATA_ROOT, DEFAULT_DATASET, apply_qc_policy, ingest, load_config, load_regions
from mxdata.data import LiveSource
from mxdata.data.composite import render_channels
from mxdata.io import Box
from mxdata.viz import label_contours
from mxseg.inputs import dataset_norms

REPO = Path(__file__).resolve().parents[1]
SLIDES = {1: "SLIDE-0235", 2: "SLIDE-0236", 3: "SLIDE-0233"}                  # paper label -> CellDIVE slide
OVERVIEW = ("CD8", "PanCK", "DAPI")   # a-c: the slide overviews carry the phenotype view of Figure 6, not the input
OVER_GAIN = 1.7                       # printed small and viewed from a distance: exposed harder than the crops
INPUT = ("LAP2", None, "DAPI")                                                 # what the segmenters see: envelope, blank, nuclei
MARKERS = [m for m in INPUT if m]                                              # the overview panels show that same image, so the
#                                                                                figure never displays a channel no method reads
ROLE_COLOR = {"train": "#AEB7C0", "val": "#FFB000", "test": "#FF5FD0"}       # the same colors as the LaTeX legend --
#                                                                              one colour per role, outline on the slides
#                                                                              and bar fill in (d), so one swatch serves both
BAR_COLOR = {"train": "#AEB7C0", "val": "#FFB000", "test": "#FF5FD0"}         # magenta, so no outline is mistaken for cyan DAPI
WRONG = "#EDA100"
CROPS = {"val": ("SLIDE-0235", 15, (120, 140)), "test": ("SLIDE-0236", 2, (40, 40))}   # region-local (y, x) of the 512² crops
CROP = 512
PANEL_H = 2.4                                                                  # inches drawn; printed at ~1.4 in height
CELLS_PANEL = (1.7, 2.75)                                                      # inches; printed at 0.3 textwidth


ASINH = 8.0                     # display curve: fluorescence is sparse -- most tissue pixels hold no nucleus and a few hold a
#                                 bright one -- so a linear or gamma ramp prints the whole midrange black. asinh expands the
#                                 faint end and compresses the bright one, the standard transfer for this kind of signal.


LAP2_GAIN, LAP2_K = 1.2, 7.0   # red tops out at 21\% luminance, so the envelope needs a harder curve than the nuclei


def soft(x: np.ndarray, k: float = ASINH) -> np.ndarray:
    """[0, 1] -> [0, 1] through asinh: faint structure lifted, bright structure held below saturation."""
    return np.arcsinh(k * np.clip(x, 0, 1)) / np.arcsinh(k)


def tint(v: np.ndarray, rgb) -> np.ndarray:
    """A single marker as an (h, w, 3) image in its own display colour, so the separate panels and the merged input are read
    in one glance: red plus cyan is exactly what the composite shows."""
    return np.clip(v, 0, 1)[..., None] * np.asarray(rgb, np.float32)


CYAN_MIX = 0.0                 # DAPI is drawn blue everywhere in the paper, as the network input itself puts it;
#                                mirroring it into green made the same marker cyan here and blue in (a-c) and Figure 6.
DAPI_GAIN = 1.50               # blue carries little luminance, so the nuclei are exposed rather than recoloured
#                                 of perceived luminance against red's 21\%, so DAPI on blue vanishes; full cyan carries 79\%
#                                 and then DAPI can swamp LAP2, the channel that separates touching nuclei -- so LAP2 is given a
#                                 stronger display curve below (LAP2_GAIN, LAP2_K) instead, since red's luminance ceiling is low.




def cyan_dapi(rgb: np.ndarray, mix: float = CYAN_MIX) -> np.ndarray:
    """The nuclear (blue) channel partly mirrored into green. Values are not changed, only the colour they are drawn in; the
    network still receives DAPI on blue with green empty."""
    out = rgb.copy()
    out[..., 1] = np.maximum(out[..., 1], mix * out[..., 2])
    return out


def display(rgb: np.ndarray, gain: float = 1.0, k: float = 5.0) -> np.ndarray:
    """Display transform of a network input (which lives on the slide-level [0, 1] scale and prints dark). Gentler than the
    thumbnails' curve: the crops start from slide-level normalization rather than raw counts, so ``k = 5`` at unit gain puts
    the nuclei near 0.83 at the 90th percentile with under 1\% of pixels clipped."""
    return soft(np.clip(rgb * gain, 0, 1), k)


def stretch(thumb: np.ndarray, markers=None) -> np.ndarray:
    """``(len(MARKERS), h, w)`` native thumbnail -> ``(h, w, 3)`` display image. Each marker goes to the channel ``INPUT``
    gives it (so the overview panels carry the network input's colours, the unused channel left black), windowed on the 2nd
    and 99th percentile of its own TISSUE pixels -- taking them over the whole frame put the low end on background and the
    high end on the few brightest nuclei, which printed 80\% of the tissue below 0.1 -- and passed through ``soft``."""
    ref = thumb.astype(np.float32).max(0)
    tissue = ref > np.percentile(ref[ref > 0], 35) if (ref > 0).any() else ref > 0
    out = np.zeros((thumb.shape[1], thumb.shape[2], 3), np.float32)
    markers = list(markers or MARKERS)
    for c, marker in enumerate(INPUT):
        if marker is None or marker not in markers:
            continue
        ch = thumb[markers.index(marker)].astype(np.float32)
        v = ch[tissue]
        lo, hi = (np.percentile(v, [2, 99]) if v.size else (0.0, 1.0))
        w = np.clip((ch - lo) / max(hi - lo, 1e-6), 0, 1)
        # the red channel carries the envelope and tops out near 21% luminance: it needs the same harder curve here
        # that display() gives it in the crops, or the overview prints the tissue cyan with the envelope invisible
        out[..., c] = soft(np.clip(w * LAP2_GAIN, 0, 1), LAP2_K) if c == 0 else soft(w)
    return out


def scalebar(ax, npx_w, npx_h, um, px_um, color="white", size=7.5, x0=0.0, y0=0.0, xfrac=0.96):
    L = um / px_um
    x1, y = x0 + npx_w * xfrac, y0 + npx_h * 0.93
    ax.plot([x1 - L, x1], [y, y], color=color, lw=1.6, solid_capstyle="butt")
    ax.text(x1 - L / 2, y - npx_h * 0.03, f"{um:g} µm" if um < 1000 else f"{um / 1000:g} mm", color=color, ha="center", va="bottom", fontsize=size,
            path_effects=[pe.withStroke(linewidth=1.2, foreground="black")])


PHENO_COLOR = {"Epithelial": "#5AFF5A", "T cell": "#FFA500", "CD4+ T cell": "#FFE14A",
               "CD8+ T cell": "#B36BFF", "Treg": "#FF1493"}                 # configs/panel.yaml's own marker colours


def phenotype_panel(out: Path, slide: str, area: str, markers=("PanCK", "DAPI"), size=430):
    """The expert's phenotype labels on the tissue that carries them: a crop of one annotated field with every label drawn in
    the colour configs/panel.yaml gives its marker. This is the level above segmentation -- what the cells are, not where."""
    lab = pd.read_csv(DATA_ROOT / slide / "phenotyping" / "labels.csv")
    lab = lab[lab.area == area]
    ar = pd.read_csv(DATA_ROOT / slide / "phenotyping" / "areas.csv")
    ar = ar[ar.area == area].iloc[0]
    cx, cy = (lab.cx.min() + lab.cx.max()) / 2, (lab.cy.min() + lab.cy.max()) / 2
    x0, y0 = int(cx - size / 2), int(cy - size / 2)
    _, mx, _ = ingest(slide, None)
    have = [m for m in markers if any((DATA_ROOT / slide).glob(f"*[-_]{m}[-_]*.ome.tif"))]
    st = mx.read_region(Box(x0, y0, x0 + size, y0 + size), markers=have, level=0).astype(np.float32)
    grey = np.zeros((size, size), np.float32)                               # the tissue stays neutral so the label colours,
    for ch in st:                                                           # which are the point of the panel, carry alone
        lo, hi = np.percentile(ch, (2, 99.5))
        grey = np.maximum(grey, soft(np.clip((ch - lo) / max(hi - lo, 1e-6), 0, 1), 6.0))
    img = np.repeat((0.62 * grey)[..., None], 3, axis=2)
    fig = plt.figure(figsize=(2.0, 2.0)); a = fig.add_axes([0, 0, 1, 1])
    a.imshow(img, interpolation="nearest")
    sub = lab[(lab.cx >= x0) & (lab.cx < x0 + size) & (lab.cy >= y0) & (lab.cy < y0 + size)].copy()
    depth = {"Epithelial": 1, "T cell": 1, "CD4+ T cell": 2, "CD8+ T cell": 2, "Treg": 3}   # the sessions are hierarchical:
    sub["d"] = sub.phenotype.map(depth)                                      # a CD4+ T cell also carries a T-cell box, so keep
    sub = (sub.sort_values("d").groupby(np.where(sub.nucleus_id > 0, sub.nucleus_id, -sub.index - 1), as_index=False)
              .last())                                                       # one box per cell, at its deepest class
    for r in sub.itertuples():
        a.add_patch(plt.Rectangle((r.x0 - x0, r.y0 - y0), r.x1 - r.x0, r.y1 - r.y0, fill=False,
                                  ec=PHENO_COLOR.get(r.phenotype, "#FFFFFF"), lw=1.0))
    a.set_xlim(0, size); a.set_ylim(size, 0); a.set_axis_off()
    scalebar(a, size, size, 50, 0.325, size=8)
    fig.savefig(out, dpi=size / 2.0); plt.close(fig)
    return {"slide": slide, "area": area, "box": [x0, y0, x0 + size, y0 + size],
            "labels": {k: int(v) for k, v in sub.phenotype.value_counts().items()}}


def extra_fields(slide: str):
    """(name, box) of every annotated field of a slide the dataset does not carry yet, taken from the QC tables of its
    delivery (each lists one row per nucleus of one field, so the field is their extent). Lets the overview panel show where
    the annotation sits before the slide enters configs/dataset.yaml."""
    out = []
    for f in sorted((DATA_ROOT / slide / "original_delivery").glob("*Region_*.csv")):
        m = re.search(r"Region_(\d+)", f.name)
        d = pd.read_csv(f)
        d.columns = [c.strip() for c in d.columns]
        if not {"XMin", "XMax", "YMin", "YMax"} <= set(d.columns):
            continue
        out.append((f"R{m.group(1)}", (d.XMin.min(), d.YMin.min(), d.XMax.max(), d.YMax.max())))
    return out


def image_panel(img: np.ndarray, out: Path, um: float | None, px_um: float, cmap=None) -> None:
    """A square crop written edge to edge at its native resolution (512 px), with an optional scale bar."""
    fig = plt.figure(figsize=(2.0, 2.0))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.imshow(img, cmap=cmap, vmin=0 if cmap else None, vmax=1 if cmap else None, interpolation="nearest")
    ax.set_axis_off()
    if um:
        scalebar(ax, img.shape[1], img.shape[0], um, px_um)
    fig.savefig(out, dpi=img.shape[1] / 2.0)
    plt.close(fig)


def cell_contours(labels):
    """One closed polyline per instance, as vectors -- a raster boundary subsamples to speckle at print size."""
    out = []
    for i, sl in enumerate(ndi.find_objects(labels), start=1):
        if sl is None:
            continue
        m = (labels[sl] == i).astype(np.uint8)
        for c in cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]:
            if len(c) > 2:
                out.append(np.vstack([c[:, 0, :], c[:1, 0, :]]) + [sl[1].start, sl[0].start])
    return out


def annotated_panel(img, labels, ignore, out: Path, um, px_um):
    """The annotated crop with vector contours: white per confirmed cell, amber per flagged one, each over a dark halo
    so the boundary reads on bright tissue and on dark. Same treatment as the method panels of Figure 8."""
    base = img.copy()
    base[ignore] = 0.45 * base[ignore] + 0.55 * np.array([0.93, 0.63, 0.0])
    fig = plt.figure(figsize=(2.0, 2.0))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.imshow(np.clip(base * 0.90, 0, 1), interpolation="lanczos")
    halo = [pe.withStroke(linewidth=1.05, foreground="black", alpha=0.85)]
    ax.add_collection(LineCollection(cell_contours(labels), colors="white", linewidths=0.38, alpha=0.96, path_effects=halo))
    ax.add_collection(LineCollection(cell_contours(ndi.label(ignore)[0]), colors="#FFC742", linewidths=0.38,
                                     alpha=0.96, path_effects=halo))
    ax.set_xlim(0, img.shape[1]); ax.set_ylim(img.shape[0], 0); ax.set_axis_off()
    if um:
        scalebar(ax, img.shape[1], img.shape[0], um, px_um)
    fig.savefig(out); plt.close(fig)


def overlay(img: np.ndarray, labels: np.ndarray, ignore: np.ndarray) -> np.ndarray:
    """Expert cells as white contours; flagged (ignored) cells tinted amber with an amber contour. The contour is darkened
    underneath before it is drawn: a white line over bright tissue is nearly invisible, and these boundaries are the reference
    the whole benchmark is scored against."""
    img = img.copy()
    img[ignore] = 0.45 * img[ignore] + 0.55 * np.array([0.93, 0.63, 0.0])
    halo = label_contours(labels, thickness=3) | label_contours(ignore.astype(np.int32), thickness=3)
    img[halo] *= 0.25                                                        # a dark rim so the contour reads on any tissue
    img[label_contours(labels, thickness=2)] = 1.0
    img[label_contours(ignore.astype(np.int32), thickness=2)] = np.array([1.0, 0.78, 0.25])
    return img


def main(out_dir: Path):
    matplotlib.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
                                "font.size": 6.5, "axes.linewidth": 0.5, "pdf.fonttype": 42})
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg = load_config(REPO / "configs" / "dataset.yaml")
    norms = dataset_norms(DEFAULT_DATASET, cfg.slides, sorted({*MARKERS, *[m for m in INPUT if m], *OVERVIEW}))

    # ---- per-region counts from the dataset: annotated cells and the expert's flags ---------------------------------------
    pt = pd.read_csv(DEFAULT_DATASET / "patches.csv", dtype={"split": str}, low_memory=False)
    lab = pt[(pt.kind == "labeled") & pt.kept.astype(bool)]
    role_of = lab.groupby(["slide", "region_id"]).split.first().to_dict()
    inst = pd.read_csv(DEFAULT_DATASET / "instances.csv")
    own = inst[inst.owned.astype(bool)].merge(lab[["key", "region_id"]].rename(columns={"region_id": "lab_region"}),
                                             left_on="patch_key", right_on="key")
    own["region_id"], own["slide"] = own.lab_region, own.patch_key.str.slice(0, 10)
    stats = own.groupby(["slide", "region_id"]).flag.value_counts().unstack(fill_value=0)
    for c in ("OK", "WRONG", "UNKNOWN", "NO_QC_ROW"):
        stats[c] = stats.get(c, 0)
    stats["cells"] = stats[["OK", "WRONG", "UNKNOWN", "NO_QC_ROW"]].sum(axis=1)
    stats["unreviewed"] = stats.UNKNOWN + stats.NO_QC_ROW
    stats["reviewed"] = stats.UNKNOWN == 0
    stats["role"] = [role_of[k] for k in stats.index]

    # ---- a-c: the three slides, regions by role ----------------------------------------------------------------------------
    layout = {}
    for k, slide in SLIDES.items():
        paths, mx, _ = ingest(slide, None)
        px_um = float(mx.pixel_size_um or 0.325)
        regions = load_regions(paths, None, cfg.min_area_px) if slide in cfg.slides else []
        have = [m for m in OVERVIEW if any((DATA_ROOT / slide).glob(f"*[-_]{m}[-_]*.ome.tif"))]
        thumb, ds = mx.thumbnail(max_dim=2000, markers=have)
        # one rendering for every whole-slide view in the paper: the same three markers, the same slide-level norms and
        # the same curve as the region panels, so a locator and its zoom are one image at two scales (Figure 6a)
        if slide in norms:                                                  # Slide 3 has no stats.json yet: window it on
            nm = norms[slide]                                               # its own tissue pixels, same curve either way
        else:
            t_ = thumb.astype(np.float32); ref_ = t_.max(0)
            tis_ = ref_ > np.percentile(ref_[ref_ > 0], 35) if (ref_ > 0).any() else ref_ > 0
            nm = {m: tuple(np.percentile(t_[i][tis_], [2, 99.9])) for i, m in enumerate(have)}   # 99 blows up sparse CD8
        tc = render_channels(thumb.astype(np.float32), have, nm, OVERVIEW)
        img = np.clip(np.stack([display(tc[i], gain=OVER_GAIN) for i in range(3)], -1), 0, 1)
        H, W = img.shape[:2]
        # the empty slide carries per-tile offsets that print as coloured blocks: mask it to true black and frame the
        # tissue, so a tall narrow section (Slide 3) is not mostly background with oversized annotation on top of it
        lum = img.max(-1)
        msk = ndi.binary_fill_holes(ndi.binary_closing(lum > 0.14, np.ones((7, 7))))
        lb, n = ndi.label(msk)
        if n:
            msk = ndi.binary_dilation(lb == (np.bincount(lb.ravel())[1:].argmax() + 1), np.ones((5, 5)))
        img = img.copy(); img[~msk] = 0.0
        ys_, xs_ = np.where(msk)
        pad = int(0.015 * max(H, W))
        cy0, cy1 = max(int(ys_.min()) - pad, 0), min(int(ys_.max()) + pad + 1, H)
        cx0, cx1 = max(int(xs_.min()) - pad, 0), min(int(xs_.max()) + pad + 1, W)
        CW, CH = cx1 - cx0, cy1 - cy0
        fs = float(np.clip(PANEL_H * CW / CH / 2.2, 0.5, 1.0))     # 1.0 on a wide panel, smaller as it narrows
        fig = plt.figure(figsize=(PANEL_H * CW / CH, PANEL_H), facecolor="black")
        a = fig.add_axes([0, 0, 1, 1]); a.set_facecolor("black")
        a.imshow(img, interpolation="lanczos")
        fields = extra_fields(slide)
        if fields:
            for _, b in fields:
                a.add_patch(plt.Rectangle((b[0] / ds, b[1] / ds), (b[2] - b[0]) / ds, (b[3] - b[1]) / ds,
                                          fill=False, ec=ROLE_COLOR["test"], lw=1.0))
            gx0 = min(b[0] for _, b in fields) / ds; gx1 = max(b[2] for _, b in fields) / ds
            gy0 = min(b[1] for _, b in fields) / ds; gy1 = max(b[3] for _, b in fields) / ds
            a.add_patch(plt.Rectangle((gx0, gy0), gx1 - gx0, gy1 - gy0, fill=False, ec=ROLE_COLOR["test"],
                                      lw=0.7, ls=(0, (3, 2)), alpha=0.8))
            a.annotate(f"{len(fields)} annotated fields" if fs > 0.8 else f"{len(fields)} fields",
                       xy=((gx0 + gx1) / 2, gy0),
                       xytext=((gx0 + gx1) / 2, max(cy0 + 0.02 * CH, gy0 - 0.07 * CH)), color=ROLE_COLOR["test"], fontsize=7 * fs,
                       fontweight="bold", ha="center", va="bottom",
                       arrowprops=dict(arrowstyle="-", color=ROLE_COLOR["test"], lw=0.8),
                       path_effects=[pe.withStroke(linewidth=1.6, foreground="black")])
        for r in regions:
            role, b = role_of.get((slide, r.region_id), "train"), r.box
            x0, y0, w, h = b.x0 / ds, b.y0 / ds, b.width / ds, b.height / ds
            a.add_patch(plt.Rectangle((x0, y0), w, h, fill=False, ec=ROLE_COLOR[role], lw=1.1))
            above = y0 > cy0 + 0.06 * CH
            a.text(x0 + w / 2, (y0 - 0.008 * CH) if above else (y0 + h + 0.008 * CH), f"R{r.region_id}{'*' if r.has_qc else ''}",
                   color=ROLE_COLOR[role], fontsize=9 * fs, fontweight="bold", ha="center", va="bottom" if above else "top",
                   path_effects=[pe.withStroke(linewidth=1.8, foreground="black")])
        a.set_xlim(cx0, cx1); a.set_ylim(cy1, cy0); a.set_axis_off()
        xf = 0.96                                                   # keep the bar away from the annotated fields
        if fields and (min(b[0] for _, b in fields) / ds + max(b[2] for _, b in fields) / ds) / 2 > cx0 + 0.5 * CW:
            xf = 0.04 + (1000 / (px_um * ds)) / CW
        scalebar(a, CW, CH, 1000, px_um * ds, size=9.5 * fs, x0=cx0, y0=cy0, xfrac=xf)
        fig.savefig(out_dir / f"slide{k}.pdf", dpi=300, facecolor="black")
        plt.close(fig)
        layout[f"slide{k}"] = {"slide": slide, "width_px": int(round(W * ds)), "height_px": int(round(H * ds)), "aspect": W / H,
                               "mm": [round(W * ds * px_um / 1000, 2), round(H * ds * px_um / 1000, 2)], "regions": len(regions)}

    # ---- slide 3: per-field counts straight from the delivery (it is not in the dataset yet) ------------------------------
    class Row:                                                                 # the same fields the dataset rows expose
        def __init__(self, ok, wrong, unrev, role, reviewed):
            self.OK, self.WRONG, self.unreviewed, self.role, self.reviewed = ok, wrong, unrev, role, reviewed
            self.cells = ok + wrong + unrev
    s3 = []
    od3 = DATA_ROOT / SLIDES[3] / "original_delivery"
    big3 = od3 / "CellDIVE_SLIDE-0233_R2_IMMUNE_RICH.geojson"
    if big3.is_file():
        cent3 = np.array([np.array(ft["geometry"]["coordinates"], float).mean(0)
                          for ft in json.load(open(big3))["features"]
                          if (ft["properties"].get("classification") or {}).get("name") == "Nuclei"])
        for name, b in extra_fields(SLIDES[3]):
            rid = int(name[1:])
            n = int(((cent3[:, 0] >= b[0]) & (cent3[:, 0] <= b[2]) & (cent3[:, 1] >= b[1]) & (cent3[:, 1] <= b[3])).sum())
            q = next((f for f in od3.glob(f"*Region_{rid}_*.csv")), None)
            d = pd.read_csv(q); d.columns = [c.strip() for c in d.columns]
            col = next((c for c in d.columns if c.lower().replace(" ", "_") in ("ground_truths", "flag")), None)
            wrong = int(d[col].astype(str).str.strip().str.lower().eq("wrong").sum()) if col else 0
            partial = len(d) < n * 0.9                                         # field 1 was reviewed only halfway and is left out
            s3.append((3, rid, Row(0 if partial else len(d) - wrong, 0 if partial else wrong,
                                   n if partial else max(n - len(d), 0), "test", not partial)))

    # ---- d: cells per region, all three slides -------------------------------------------------------------------------------
    rows = [(k, rid, stats.loc[(SLIDES[k], rid)]) for k in (1, 2) for rid in sorted(stats.loc[SLIDES[k]].index)] + s3
    y, cur = [], 0.0
    for i, (k, _, _) in enumerate(rows):
        if i and rows[i - 1][0] != k:
            cur += 1.3
        y.append(cur); cur += 1.0
    y = max(y) - np.array(y)                                                   # Slide 1, region 1 on top
    fig = plt.figure(figsize=CELLS_PANEL)
    b_ = fig.add_axes([0.28, 0.125, 0.70, 0.865])
    for (k, rid, s), yy in zip(rows, y):
        col = BAR_COLOR[s.role]
        b_.barh(yy, s.OK / 1000, color=col, height=0.74, zorder=3)
        b_.barh(yy, s.unreviewed / 1000, left=s.OK / 1000, color=col, alpha=0.38, height=0.74, zorder=3)
        b_.barh(yy, s.WRONG / 1000, left=(s.OK + s.unreviewed) / 1000, color=WRONG, height=0.74, zorder=3, hatch="////", edgecolor="white", lw=0)
    b_.set_yticks(y, [f"R{rid}{'*' if s.reviewed else ''}" for _, rid, s in rows], fontsize=6)
    b_.tick_params(axis="y", length=0, pad=2); b_.tick_params(axis="x", labelsize=6.5, width=0.5, length=2)
    b_.set_xlabel("annotated cells (×10³)", fontsize=7, labelpad=2)
    for k in sorted({kk for kk, _, _ in rows}):
        ys = [yy for (kk, _, _), yy in zip(rows, y) if kk == k]
        b_.annotate(f"Slide {k}", xy=(0, float(np.mean(ys))), xycoords=("axes fraction", "data"), xytext=(-27, 0), textcoords="offset points",
                    rotation=90, ha="center", va="center", fontsize=7, fontweight="bold")
    b_.set_xlim(0, max(stats.cells.max(), max((r.cells for _, _, r in s3), default=0)) / 1000 * 1.06); b_.set_ylim(-0.6, max(y) + 0.6)
    for s_ in ("top", "right"):
        b_.spines[s_].set_visible(False)
    b_.grid(axis="x", color="#E4E6E9", lw=0.5, zorder=0); b_.set_axisbelow(True)
    fig.savefig(out_dir / "cells.pdf")
    plt.close(fig)

    # ---- e-j: full-resolution crops with the expert annotation, the input channels and the network input -------------------
    for name, (slide, rid, (y0, x0)) in CROPS.items():
        paths, mx, _ = ingest(slide, None)
        px_um = float(mx.pixel_size_um or 0.325)
        regions = load_regions(paths, [rid], cfg.min_area_px)
        rs = LiveSource(mx, regions, cfg.markers, cfg.level).load(f"{slide}_R{rid:02d}")
        labels, ignore = apply_qc_policy(rs.labels, rs.flags, "ignore_wrong")
        r = regions[0]
        y0, x0 = int(min(y0, labels.shape[0] - CROP)), int(min(x0, labels.shape[1] - CROP))
        box = Box(r.box.x0 + x0, r.box.y0 + y0, r.box.x0 + x0 + CROP, r.box.y0 + y0 + CROP)
        L, Wm = labels[y0:y0 + CROP, x0:x0 + CROP], ignore[y0:y0 + CROP, x0:x0 + CROP]
        stack = mx.read_region(box, markers=MARKERS, level=0)
        _c = render_channels(stack, MARKERS, norms[slide], INPUT)
        img = np.stack([display(_c[0], gain=LAP2_GAIN, k=LAP2_K), display(_c[1]), display(_c[2], gain=DAPI_GAIN)], -1)
        image_panel(overlay(img, L, Wm), out_dir / f"{name}_crop.png", 50, px_um)
        if name == "test":
            st = mx.read_region(box, markers=[m for m in INPUT if m], level=0)
            comp = render_channels(st, [m for m in INPUT if m], norms[slide], INPUT)          # R = LAP2, G = blank, B = DAPI
            image_panel(tint(display(comp[0], gain=LAP2_GAIN, k=LAP2_K), (1.0, 0.0, 0.0)), out_dir / "lap2.png", None, px_um)
            image_panel(tint(display(comp[2], gain=DAPI_GAIN), (0.0, 0.0, 1.0)), out_dir / "dapi.png", None, px_um)
            inp = np.stack([display(comp[0], gain=LAP2_GAIN, k=LAP2_K), display(comp[1]), display(comp[2], gain=DAPI_GAIN)], axis=-1)
            image_panel(inp, out_dir / "input.png", None, px_um)
            annotated_panel(inp, L, Wm, out_dir / "input_ann.pdf", 50, px_um)
        layout[f"{name}_crop"] = {"slide": slide, "region": rid, "y0": y0, "x0": x0, "cells_in_crop": int(len(np.unique(L)) - 1)}

    layout["phenotypes"] = phenotype_panel(out_dir / "phenotypes.png", SLIDES[3], "R2")
    (out_dir / "layout.json").write_text(json.dumps(layout, indent=1))
    print(f"wrote panels to {out_dir}")
    print(stats[["role", "reviewed", "cells", "OK", "WRONG", "unreviewed"]].to_string())
    for slide in cfg.slides:
        s = stats.loc[slide]
        print(f"{slide}: {len(s)} regions ({int(s.reviewed.sum())} reviewed), {int(s.cells.sum())} cells: {int(s.OK.sum())} confirmed, "
              f"{int(s.WRONG.sum())} flagged, {int(s.unreviewed.sum())} unreviewed; reviewed regions hold {int(s[s.reviewed].cells.sum())} cells")
    print(json.dumps(layout, indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=str(REPO / "LaTex" / "figures" / "slides"))
    args = ap.parse_args()
    main(Path(args.out))
