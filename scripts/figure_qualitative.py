#!/usr/bin/env python
"""What each method predicts, and what its extra predictions actually are.

Two figures, both on the test slide and both consistent with Table 2 by construction:

  qual_*.pdf   one window, every reference cell filled with the error class the taxonomy of scripts/figure_errors.py
               counted it as, so the colours here are the numbers there (Slide 2 = SLIDE-0236 only).
  gained_*.pdf the predictions that cover no reference cell, ranked by how many independent methods place a cell at
               the same spot, each drawn on the tissue with the expert's own contours -- the direct way to ask whether
               a "false positive" is a real nucleus the reference lacks.

    conda activate dl
    python scripts/figure_qualitative.py          # -> LaTex/figures/qual/
"""
from __future__ import annotations
from pathlib import Path
import json, sys
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import ndimage as ndi
from scipy.spatial import cKDTree
from skimage.segmentation import find_boundaries
import matplotlib.patheffects as pe
from matplotlib.collections import LineCollection
import cv2

from mxdata import apply_qc_policy, ingest, load_config, load_regions
from mxdata.data import LiveSource
from mxdata.data.composite import render_channels
from mxdata.io import Box
from mxseg.metrics import instance_outcomes, iou_matrix
from mxseg.inputs import dataset_norms

sys.path.insert(0, str(Path(__file__).resolve().parent))
from figure_errors import taxonomy                                  # the function behind Figure 7
from figure_slides import display, LAP2_GAIN, LAP2_K, DAPI_GAIN, INPUT, MARKERS   # the paper's one display blend

REPO = Path(__file__).resolve().parents[1]
PRED = REPO / "outputs" / "region" / "test"
OUT = REPO / "LaTex" / "figures" / "qual"
KEY, CROP, PANEL = "SLIDE-0236_R02", 384, 1.70
ZOOM, NZOOM = 104, 6                                                 # px per gained-cell crop, and how many are shown

KODIAK = "MxIF-SSL"                                                  # what \kodiak expands to in LaTex/main.tex
SHOW = [("cellpose_ft_s0", "Cellpose"), ("kodiakb_flows_s0", f"{KODIAK} B/16"), ("dinov3b_flows_s0", "DINOv3 B/16"),
        ("unet_flows_s0", "U-Net"), ("kodiakb_frozen_s0", f"{KODIAK} B/16, frozen"), ("dinov3b_frozen_s0", "DINOv3 B/16, frozen"),
        ("sam3", "SAM 3"), ("cellpose", "Cellpose-SAM")]
SHOW = [(run, label) for run, label in SHOW if (PRED / run / f"{KEY}.npz").is_file()]   # only the rows that exist in outputs/
FILL = {"correct": "#CBE5D8", "missed": "#7C4DBF", "merged": "#1E7FE0", "split": "#F2820C", "boundary": "#E8B21A"}
UNSCORED, GAINED, GAINED_FILL = "#EDEFF2", "#E12D39", "#F9D2D6"

matplotlib.rcParams.update({"font.family": "sans-serif",
    "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 7, "pdf.fonttype": 42, "legend.frameon": False,
    "figure.facecolor": "white", "savefig.facecolor": "white"})
hexrgb = lambda h: np.array([int(h[i:i + 2], 16) / 255 for i in (1, 3, 5)], np.float32)


def cell_contours(labels):
    """One closed polyline per instance, as vectors: raster boundaries turn to speckle at print size."""
    out = []
    for i, sl in enumerate(ndi.find_objects(labels), start=1):
        if sl is None:
            continue
        m = (labels[sl] == i).astype(np.uint8)
        for c in cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]:
            if len(c) > 2:
                out.append(np.vstack([c[:, 0, :], c[:1, 0, :]]) + [sl[1].start, sl[0].start])
    return out


def draw_cells(ax, labels, color="white", lw=0.30, halo=0.80):
    ax.add_collection(LineCollection(cell_contours(labels), colors=color, linewidths=lw, alpha=0.95,
                                     path_effects=[pe.withStroke(linewidth=halo, foreground="black", alpha=0.8)]))


def bare(ax, edge="#C6CBD2"):
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_edgecolor(edge); s.set_linewidth(0.5)


def tissue(mx, box, norms):
    """The network input, in the paper's own display blend -- identical to Figure 1e-g."""
    comp = render_channels(mx.read_region(box, markers=MARKERS, level=0), MARKERS, norms, INPUT)
    return np.stack([display(comp[0], gain=LAP2_GAIN, k=LAP2_K), display(comp[1]), display(comp[2], gain=DAPI_GAIN)], -1)


def outcomes(gt, ignore, pred):
    o = instance_outcomes(gt, pred, ignore=ignore, thr=0.5)
    iou, ag, ap = iou_matrix(o["gt"], o["pred"]) if (o["gt"].max() and o["pred"].max()) else (np.zeros((0, 0)), np.zeros(0), np.zeros(0))
    inter = iou * (ag[:, None] + ap[None, :]) / (1 + iou)
    o.update(iou=iou, area_g=ag, area_p=ap, inter=inter,
             cover_g=inter / np.maximum(ag, 1)[:, None], cover_p=inter / np.maximum(ap, 1)[None, :])
    return o, *taxonomy(o)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    cfg = load_config(REPO / "configs" / "dataset.yaml")
    slide = KEY.split("_R")[0]
    paths, mx, _ = ingest(slide, None)
    px = float(mx.pixel_size_um or 0.325)
    norms = dataset_norms(REPO / "data" / "processed" if False else __import__("mxdata").DEFAULT_DATASET, cfg.slides, MARKERS)[slide]
    regions = {r.key: r for r in load_regions(paths, None, cfg.min_area_px) if r.key.startswith(slide + "_")}   # Slide 2 only
    src = LiveSource(mx, list(regions.values()), MARKERS, cfg.level)

    # ================= figure 1: the error map, one window ==========================================================
    rs = src.load(KEY)
    gt, ignore = apply_qc_policy(rs.labels, rs.flags, "ignore_wrong")
    d = ndi.uniform_filter((gt > 0).astype(np.float32), CROP)
    inner = d[CROP // 2:-CROP // 2, CROP // 2:-CROP // 2]
    y0, x0 = (int(v) for v in np.unravel_index(np.argmax(inner), inner.shape))
    sl = (slice(y0, y0 + CROP), slice(x0, x0 + CROP))
    print(f"{KEY}: window ({y0},{x0}) = {CROP * px:.0f} um", flush=True)

    r = regions[KEY]
    box = Box(r.box.x0 + x0, r.box.y0 + y0, r.box.x0 + x0 + CROP, r.box.y0 + y0 + CROP)
    img = tissue(mx, box, norms)
    gcrop = gt[sl]
    fig = plt.figure(figsize=(PANEL, PANEL)); ax = fig.add_axes([0, 0, 1, 1])
    ax.imshow(np.clip(img, 0, 1) * 0.88, interpolation="lanczos")             # headroom so the contours sit above it
    draw_cells(ax, gcrop)                                                     # every cell separately outlined
    bare(ax)
    L = 25 / px
    ax.plot([CROP * 0.95 - L, CROP * 0.95], [CROP * 0.94, CROP * 0.94], color="white", lw=1.8, solid_capstyle="butt")
    ax.text(CROP * 0.95 - L / 2, CROP * 0.94 - 7, "25 µm", color="white", ha="center", va="bottom", fontsize=5.8)
    fig.savefig(OUT / "qual_input.pdf"); plt.close(fig)

    for run, label in SHOW:
        f = PRED / run / f"{KEY}.npz"
        if not f.is_file():
            print("MISSING", f); continue
        o, g_cls, p_cls = outcomes(gt, ignore, np.load(f)["pred"].astype(np.int32))
        lut = np.ones((int(o["gt"].max()) + 1, 3), np.float32)
        for gid in range(1, int(o["gt"].max()) + 1):
            c = g_cls[gid - 1]
            lut[gid] = hexrgb(FILL[c]) if c in FILL else hexrgb(UNSCORED)
        gl = o["gt"][sl]
        gm = np.isin(o["pred"], np.nonzero(p_cls == "gained")[0] + 1)[sl] & (gl == 0)
        panel = np.ones(gl.shape + (3,), np.float32)
        panel[gm] = hexrgb(GAINED_FILL)                                       # extra objects: filled, so they read at once
        panel[find_boundaries(gm, mode="inner") & gm] = hexrgb(GAINED)
        cell = gl > 0
        panel[cell] = lut[gl][cell]
        panel[find_boundaries(gl, mode="inner") & cell] = 1.0                 # white separation between touching cells
        fig = plt.figure(figsize=(PANEL, PANEL)); ax = fig.add_axes([0, 0, 1, 1])
        ax.imshow(panel, interpolation="nearest"); bare(ax)
        fig.savefig(OUT / f"qual_{run}.pdf"); plt.close(fig)
        print(f"  {label:22s} drawn", flush=True)

    lg = plt.figure(figsize=(5.3, 0.26))
    items = [("correct", FILL["correct"]), ("missed", FILL["missed"]), ("merged", FILL["merged"]),
             ("split", FILL["split"]), ("poor boundary", FILL["boundary"])]
    h = [plt.Line2D([], [], marker="s", ls="", ms=4.6, color=c, mec="#B9C2CC", mew=0.4) for _, c in items]
    h.append(plt.Line2D([], [], marker="s", ls="", ms=4.6, mfc="none", mec=GAINED, mew=1.1))
    lg.legend(h, [t for t, _ in items] + ["gained (no reference cell)"], loc="center", ncol=6, fontsize=6.3,
              handlelength=1.0, handletextpad=0.4, columnspacing=1.3, borderpad=0.0, borderaxespad=0.0, frameon=False)
    lg.savefig(OUT / "qual_legend.pdf", bbox_inches="tight", pad_inches=0.01); plt.close(lg)

    # ================= figure 2: are the gained predictions real cells? =============================================
    cand = []
    for key in sorted(regions):
        rs2 = src.load(key)
        g2, ig2 = apply_qc_policy(rs2.labels, rs2.flags, "ignore_wrong")
        per = []
        for run, label in SHOW:
            f = PRED / run / f"{key}.npz"
            if not f.is_file():
                continue
            o, _, p_cls = outcomes(g2, ig2, np.load(f)["pred"].astype(np.int32))
            ids = np.nonzero(p_cls == "gained")[0] + 1
            if not len(ids):
                continue
            ar = np.array(ndi.sum(np.ones_like(o["pred"]), o["pred"], ids))
            cen = np.array(ndi.center_of_mass(o["pred"] > 0, o["pred"], ids))
            keep = ar >= 40                                                   # a cell, not a speck
            for c_, a_, i_ in zip(cen[keep], ar[keep], ids[keep]):
                per.append(dict(label=label, run=run, yx=c_, area=float(a_), pid=int(i_)))
        if not per:
            continue
        allc = np.vstack([e["yx"] for e in per])
        used = np.zeros(len(allc), bool)
        tree = cKDTree(allc)
        for i in np.argsort([-e["area"] for e in per]):
            if used[i]:
                continue
            grp = np.array(tree.query_ball_point(allc[i], 8.0))
            grp = grp[~used[grp]]
            used[grp] = True
            names = sorted({per[j]["label"] for j in grp})
            if len(names) >= 2:
                cand.append(dict(key=key, yx=allc[grp].mean(0), n=len(names), names=names,
                                 run=per[i]["run"], pid=per[i]["pid"]))
    cand.sort(key=lambda c: -c["n"])
    hist = np.bincount([c["n"] for c in cand], minlength=len(SHOW) + 1)
    print(f"gained locations by >=2 methods: {len(cand)} | by all {len(SHOW)}: {hist[len(SHOW)]} | "
          + " ".join(f"{k}:{hist[k]}" for k in range(2, len(SHOW) + 1)), flush=True)

    shown, seen_keys = [], {}
    for c in cand:
        if len(shown) >= NZOOM:
            break
        if sum(1 for s in shown if s["key"] == c["key"]) >= 2:                # spread over regions
            continue
        shown.append(c)
    shown_meta = []
    for i, c in enumerate(shown):
        key = c["key"]
        if key not in seen_keys:
            rs2 = src.load(key)
            seen_keys[key] = apply_qc_policy(rs2.labels, rs2.flags, "ignore_wrong")
        g2 = seen_keys[key][0]
        cy, cx = (int(v) for v in c["yx"])
        yy = int(np.clip(cy - ZOOM // 2, 0, g2.shape[0] - ZOOM)); xx = int(np.clip(cx - ZOOM // 2, 0, g2.shape[1] - ZOOM))
        rr = regions[key]
        b = Box(rr.box.x0 + xx, rr.box.y0 + yy, rr.box.x0 + xx + ZOOM, rr.box.y0 + yy + ZOOM)
        im = np.clip(tissue(mx, b, norms), 0, 1)
        gz = g2[yy:yy + ZOOM, xx:xx + ZOOM]
        o2, _, _ = outcomes(g2, seen_keys[key][1], np.load(PRED / c["run"] / f"{key}.npz")["pred"].astype(np.int32))
        pm = (o2["pred"] == c["pid"])[yy:yy + ZOOM, xx:xx + ZOOM]
        fig = plt.figure(figsize=(0.86, 0.86)); ax = fig.add_axes([0, 0, 1, 1])
        ax.imshow(im * 0.88, interpolation="lanczos")
        draw_cells(ax, gz, lw=0.5, halo=1.1)                                  # the expert's own cells
        draw_cells(ax, pm.astype(np.int32), color=GAINED, lw=0.9, halo=1.6)   # what the methods drew
        bare(ax)
        fig.savefig(OUT / f"gained_{i}.pdf"); plt.close(fig)
        print(f"  gained {i}: {key} at ({cy},{cx}) by {c['n']} methods, contour from {c['run']}", flush=True)
        shown_meta.append({"panel": i, "key": key, "n_methods": c["n"], "run": c["run"],
                           "label": dict(SHOW).get(c["run"], c["run"])})
    (OUT / "gained.json").write_text(json.dumps(shown_meta, indent=1))
    print("done ->", OUT, flush=True)


if __name__ == "__main__":
    main()
