#!/usr/bin/env python
"""Panels of Figure 1 of the paper (LaTex/figures/slide/{a..j}), composed by LaTeX (main.tex): the benchmark slide as the
networks see it with its sixteen annotated regions by split (a), annotated cells per region with the expert's flags (b),
full-resolution crops of the four held-out test regions with the expert annotation (c-f), and the marker channels behind the
composite with the resulting network input (g-j). Everything is read from the slide and the dataset; no model is involved.
Panel letters, headers and the legend are typeset in LaTeX; only region labels and scale bars live in the pixels.

    conda activate dl
    python scripts/figure_slide.py                      # -> LaTex/figures/slide/a.pdf, b.pdf, c.png ... j.png
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import patheffects as pe

from mxdata import DEFAULT_DATASET, apply_qc_policy, ingest, load_config, load_regions
from mxdata.data import LiveSource
from mxdata.data.composite import render_channels
from mxdata.io import Box
from mxdata.viz import label_contours
from mxseg.inputs import dataset_norms

REPO = Path(__file__).resolve().parents[1]
MARKERS = ["CD45", "PanCK", "DAPI"]                                            # R, G, B of the display composite (figures only)
INPUT = ("LAP2", None, "DAPI")                                                 # what the segmenters see: envelope, blank, nuclei
SPLIT_COLOR = {"train": "#FFFFFF", "val": "#FFB000", "test": "#35D6FF"}      # the same colors as the LaTeX legend
BAR_COLOR = {"train": "#9AA3AD", "val": "#FFB000", "test": "#35D6FF"}
WRONG = "#EDA100"
CROPS = {2: (40, 40), 10: (150, 250), 13: (400, 900), 16: (200, 200)}         # region-local (y, x) of the 512² test crops
CHANNEL_CROP = (15, (120, 140))                                                # the reviewed villous region for panels g-j
CROP = 512
PANEL_A_W = 5.0                                                                # inches; printed at 0.7445 textwidth (4.1 in)
PANEL_B = (1.5, 3.05)                                                          # inches; the same aspect as panel a's height


def display(rgb: np.ndarray, gain: float = 1.0, gamma: float = 0.85) -> np.ndarray:
    """Display transform of a network input (which lives on the slide-level [0, 1] scale and prints dark)."""
    return np.clip(rgb * gain, 0, 1) ** gamma


def scalebar(ax, npx_w, npx_h, um, px_um, color="white", size=7.5):
    L = um / px_um
    x1, y = npx_w * 0.96, npx_h * 0.93
    ax.plot([x1 - L, x1], [y, y], color=color, lw=1.6, solid_capstyle="butt")
    ax.text(x1 - L / 2, y - npx_h * 0.03, f"{um:g} µm" if um < 1000 else f"{um / 1000:g} mm", color=color, ha="center", va="bottom", fontsize=size,
            path_effects=[pe.withStroke(linewidth=1.2, foreground="black")])


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


def main(out_dir: Path):
    matplotlib.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
                                "font.size": 6.5, "axes.linewidth": 0.5, "pdf.fonttype": 42})
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg = load_config(REPO / "configs" / "dataset.yaml")
    paths, mx, _ = ingest(cfg.slides[0], None)
    px_um = float(mx.pixel_size_um or 0.325)
    regions = load_regions(paths, None, cfg.min_area_px)
    norms = dataset_norms(DEFAULT_DATASET, cfg.slides, sorted({*MARKERS, *[m for m in INPUT if m]}))[cfg.slides[0]]   # the dataset's statistics
    composite = lambda stack: render_channels(stack, MARKERS, norms, tuple(MARKERS))

    pt = pd.read_csv(DEFAULT_DATASET / "patches.csv", dtype={"split": str})
    lab = pt[(pt.kind == "labeled") & pt.kept.astype(bool)]
    split_of = lab.groupby("region_id").split.first().to_dict()

    # ---- per-region counts: annotated cells and the expert's wrong flags -----------------------------------------------
    src = LiveSource(mx, regions, cfg.markers, cfg.level)
    rows, cache = [], {}
    for r in regions:
        rs = src.load(f"{cfg.slides[0]}_R{r.region_id:02d}")
        labels, ignore = apply_qc_policy(rs.labels, rs.flags, "ignore_wrong")
        n_wrong = int((np.unique(rs.labels[ignore]) > 0).sum()) if ignore.any() else 0
        rows.append({"region": r.region_id, "split": split_of.get(r.region_id, "train"), "cells": int(len(np.unique(labels)) - 1),
                     "wrong": n_wrong, "qc": bool(r.has_qc)})
        cache[r.region_id] = (r, labels, ignore)
    stats = pd.DataFrame(rows).set_index("region").sort_index()

    # ---- a: the slide, regions by split ---------------------------------------------------------------------------------
    thumb, ds = mx.thumbnail(max_dim=2000, markers=MARKERS)
    slide = display(composite(thumb).transpose(1, 2, 0), gain=2.4, gamma=0.8)
    H, W = slide.shape[:2]
    fig = plt.figure(figsize=(PANEL_A_W, PANEL_A_W * H / W))
    a = fig.add_axes([0, 0, 1, 1])
    a.imshow(slide, interpolation="lanczos")
    for r in regions:
        b, split = r.box, split_of.get(r.region_id, "train")
        x0, y0, w, h = b.x0 / ds, b.y0 / ds, b.width / ds, b.height / ds
        a.add_patch(plt.Rectangle((x0, y0), w, h, fill=False, ec=SPLIT_COLOR[split], lw=0.9))
        above = y0 > 0.06 * H
        a.text(x0 + w / 2, (y0 - 0.008 * H) if above else (y0 + h + 0.008 * H), f"R{r.region_id}{'*' if r.has_qc else ''}",
               color=SPLIT_COLOR[split], fontsize=7.6, fontweight="bold", ha="center", va="bottom" if above else "top",
               path_effects=[pe.withStroke(linewidth=1.6, foreground="black")])
    a.set_xlim(0, W); a.set_ylim(H, 0); a.set_axis_off()
    scalebar(a, W, H, 1000, px_um * ds, size=8)
    fig.savefig(out_dir / "a.pdf", dpi=300)
    plt.close(fig)

    # ---- b: cells per region --------------------------------------------------------------------------------------------------
    fig = plt.figure(figsize=PANEL_B)
    b_ = fig.add_axes([0.2, 0.09, 0.78, 0.9])
    y = np.arange(len(stats))[::-1]                                            # region 1 on top
    kept = stats.cells - stats.wrong
    b_.barh(y, kept / 1000, color=[BAR_COLOR[s] for s in stats.split], height=0.72, zorder=3)
    b_.barh(y, stats.wrong / 1000, left=kept / 1000, color=WRONG, height=0.72, zorder=3, hatch="////", edgecolor="white", lw=0)
    b_.set_yticks(y, [f"{r}{'*' if q else ''}" for r, q in zip(stats.index, stats.qc)], fontsize=7)
    b_.tick_params(axis="y", length=0, pad=2); b_.tick_params(axis="x", labelsize=7, width=0.5, length=2)
    b_.set_xlabel("annotated cells (×10³)", fontsize=7.5, labelpad=2)
    b_.set_ylabel("region", fontsize=7.5, labelpad=2)
    b_.set_xlim(0, stats.cells.max() / 1000 * 1.06); b_.set_ylim(-0.6, len(stats) - 0.4)
    for s_ in ("top", "right"):
        b_.spines[s_].set_visible(False)
    b_.grid(axis="x", color="#E4E6E9", lw=0.5, zorder=0); b_.set_axisbelow(True)
    fig.savefig(out_dir / "b.pdf")
    plt.close(fig)

    # ---- c-f: the four test regions, full-resolution crops with the expert annotation ------------------------------------
    for k, rid in enumerate(sorted(CROPS)):
        r, labels, ignore = cache[rid]
        y0, x0 = CROPS[rid]
        y0, x0 = int(min(y0, labels.shape[0] - CROP)), int(min(x0, labels.shape[1] - CROP))
        stack = mx.read_region(Box(r.box.x0 + x0, r.box.y0 + y0, r.box.x0 + x0 + CROP, r.box.y0 + y0 + CROP), markers=MARKERS, level=0)
        img = display(composite(stack).transpose(1, 2, 0), gain=1.6)
        L, W_ = labels[y0:y0 + CROP, x0:x0 + CROP], ignore[y0:y0 + CROP, x0:x0 + CROP]
        img[W_] = 0.45 * img[W_] + 0.55 * np.array([0.93, 0.63, 0.0])
        img[label_contours(L, thickness=2)] = 1.0
        img[label_contours(W_.astype(np.int32), thickness=2)] = np.array([1.0, 0.75, 0.2])
        image_panel(img, out_dir / f"{'cdef'[k]}.png", 50 if k == len(CROPS) - 1 else None, px_um)

    # ---- g-j: the marker channels behind the composite, and the network input ---------------------------------------------
    rid, (y0, x0) = CHANNEL_CROP
    r = cache[rid][0]
    stack = mx.read_region(Box(r.box.x0 + x0, r.box.y0 + y0, r.box.x0 + x0 + CROP, r.box.y0 + y0 + CROP), markers=MARKERS, level=0)
    comp = composite(stack)
    for k, m in enumerate(MARKERS):
        image_panel(display(comp[k], gain=1.6), out_dir / f"{'ghi'[k]}.png", None, px_um, cmap="gray")
    image_panel(display(comp.transpose(1, 2, 0), gain=1.6), out_dir / "j.png", 50, px_um)

    print(f"wrote panels a-j to {out_dir}")
    print(stats.to_string())
    print(f"cells: train {int(stats[stats.split == 'train'].cells.sum())}, val {int(stats[stats.split == 'val'].cells.sum())}, "
          f"test {int(stats[stats.split == 'test'].cells.sum())} (flagged {int(stats[stats.split == 'test'].wrong.sum())}); "
          f"reviewed regions {int(stats.qc.sum())} with {int(stats.wrong.sum())} flagged of {int(stats[stats.qc].cells.sum())}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=str(REPO / "LaTex" / "figures" / "slide"))
    args = ap.parse_args()
    main(Path(args.out))
