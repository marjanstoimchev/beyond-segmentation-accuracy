#!/usr/bin/env python
"""Cell maps of chosen regions as tissue archetypes, across the slides the benchmark already reports.

Every cell of a region is drawn in its gated phenotype colour, beside a depth profile that gives the immune fraction in
horizontal bands, so a top-to-bottom gradient is legible rather than asserted. The gates are the calibrated ones of
configs/panel.yaml, converted to this slide's raw counts through its own stats.json, and the pipeline is the one of
notebooks/analyses/10_tumor_immune_spatial.ipynb.

    conda activate dl
    python scripts/figure_regions_0235.py                              # -> LaTex/figures/regions/arch_*.pdf
    python scripts/figure_regions_0235.py --regions 235:R13 236:R05    # <slide>:<region>, in the order given

Reviewed regions carry the expert's flags and flagged cells are dropped; the rest are HALO contours as delivered, so
their cell-level detail is unreviewed even where the tissue architecture is clear. Slide 233 is deliberately absent: it
is the second test slide, still to be scored unchanged, so its cells are not looked at here.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.collections import LineCollection
from scipy import ndimage as ndi

import cv2
from mxdata import DEFAULT_DATASET, apply_qc_policy, ingest, load_config, load_regions
from mxdata.data import LiveSource
from mxdata.pipeline.stats import NormStats
from mxseg.analysis import assign_phenotypes, load_panel, phenotype_gates

REPO = Path(__file__).resolve().parents[1]
CLASS = {"Tumor": "#3FA96B", "CD8 T": "#5B2BD9", "Treg": "#E11D74", "CD4 T": "#C9A227", "T other": "#E08A3C",
         "Other immune": "#E0B4B4", "Other": "#E7EAEF"}
IMMUNE = ("CD8 T", "Treg", "CD4 T", "T other", "Other immune")
PANEL_TO_CLASS = {"Epithelial": "Tumor", "uPAR+ epithelial": "Tumor", "CD8+ T cell": "CD8 T", "Treg": "Treg",
                  "CD4+ T cell": "CD4 T", "T cell": "T other", "Immune": "Other immune", "Other": "Other"}
INK, MUTED, GRID = "#1A1C1F", "#6B7075", "#E4E6E9"
PROFILE_COLORS = ["#5B2BD9", "#E08A3C", "#2A78D6", "#C43E4E"]   # one per region shown, in order
SLIDE_NO = {"SLIDE-0235": "1", "SLIDE-0236": "2", "SLIDE-0233": "3"}
BANDS = 14                                                     # horizontal bands of the depth profile


def hexrgb(h):
    return np.array([int(h[i:i + 2], 16) / 255 for i in (1, 3, 5)], np.float32)


def main(keys, out):
    matplotlib.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
        "font.size": 7.5, "axes.labelsize": 7.5, "xtick.labelsize": 7, "ytick.labelsize": 7,
        "axes.linewidth": 0.5, "xtick.major.width": 0.5, "ytick.major.width": 0.5,
        "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False,
        "figure.facecolor": "white", "savefig.facecolor": "white", "pdf.fonttype": 42})

    cfg = load_config(REPO / "configs" / "dataset.yaml")
    markers = list(cfg.markers)
    panel = load_panel(REPO / "configs" / "panel.yaml")
    px_um = None
    per_slide, order = {}, []
    for spec in keys:                                          # "<slide>:<region>", e.g. 235:R13
        sl, rg = spec.split(":") if ":" in spec else ("235", spec)
        order.append((f"SLIDE-0{sl}", rg))
    for sl in dict.fromkeys(s for s, _ in order):
        st = NormStats.load(Path(DEFAULT_DATASET) / sl / "stats.json")   # this slide's own normalization
        thr = phenotype_gates(pd.DataFrame(), panel, {m: (float(st.lo[m]), float(st.hi[m])) for m in markers})
        paths, mx, _ = ingest(sl, None)
        px_um = float(mx.pixel_size_um or 0.325)
        regs = {r.key: r for r in load_regions(paths, None, cfg.min_area_px)}
        want = [f"{sl}_{rg}" for s_, rg in order if s_ == sl]
        missing = [k for k in want if k not in regs]
        if missing:
            raise SystemExit(f"no such region: {missing}; have {sorted(regs)}")
        per_slide[sl] = (thr, regs, LiveSource(mx, [regs[k] for k in want], markers, cfg.level))

    rows = []
    for sl, rg in order:
        thr, regions, src = per_slide[sl]
        key = f"{sl}_{rg}"
        rs = src.load(key)
        labels, _ = apply_qc_policy(rs.labels, rs.flags, "drop_wrong")    # a picture should not show objects the expert rejected
        ids = np.unique(labels)[1:]
        t = pd.DataFrame({"id": ids})
        cen = np.array(ndi.center_of_mass(labels > 0, labels, ids))
        t["cy"], t["cx"] = cen[:, 0], cen[:, 1]
        for m in panel.gate_markers:
            t[m] = ndi.mean(rs.image[markers.index(m)].astype(np.float32), labels, ids)
        ph, _lin = assign_phenotypes(t.set_index("id"), panel, thr)
        t = t.set_index("id")
        t["cls"] = ph.map(PANEL_TO_CLASS).values
        rows.append({"key": key, "slide": sl, "region": rg, "labels": labels, "t": t,
                     "reviewed": regions[key].has_qc, "n_flagged": regions[key].n_wrong})
        share = t.cls.value_counts(normalize=True)
        print(f"{key}: {len(t):5d} cells | tumor {100 * share.get('Tumor', 0):4.1f} % | "
              f"immune {100 * sum(share.get(c, 0) for c in IMMUNE):4.1f} % | "
              f"CD8 {100 * share.get('CD8 T', 0):3.1f} % | Treg {100 * share.get('Treg', 0):3.1f} % | "
              f"reviewed {regions[key].has_qc}")

    # ---- the paper's panels: one PDF each at final print size (2.60 x 1.63 in), composed by main.tex ------------------
    h, w = rows[0]["labels"].shape
    PW, PH = 2.42, 2.42 * h / w
    edges = np.linspace(0, h, BANDS + 1)
    mid_um = (edges[:-1] + edges[1:]) / 2 * px_um
    prof = {}
    for r in rows:
        b = np.clip(np.digitize(r["t"].cy.to_numpy(), edges) - 1, 0, BANDS - 1)
        prof[r["key"]] = np.array([np.nan if (b == k).sum() < 8 else r["t"][b == k].cls.isin(IMMUNE).mean()
                                   for k in range(BANDS)])

    def save(fig, name):
        fig.savefig(out.parent / f"{name}.pdf")
        plt.close(fig)

    out.parent.mkdir(parents=True, exist_ok=True)
    for r in rows:
        fig = plt.figure(figsize=(PW, PH))
        ax = fig.add_axes([0, 0, 1, 1])
        lut = np.ones((int(r["labels"].max()) + 1, 3), np.float32)
        for cid, cl in r["t"].cls.items():
            lut[cid] = hexrgb(CLASS[cl])
        ax.imshow(lut[r["labels"]], interpolation="nearest")
        cnts = cv2.findContours((r["labels"] > 0).astype(np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[0]
        ax.add_collection(LineCollection([np.vstack([c[:, 0, :], c[:1, 0, :]]) for c in cnts],
                                         colors="#5D646D", linewidths=0.12, alpha=0.45))
        ax.set_xlim(0, w); ax.set_ylim(h, 0); ax.set_axis_off()
        ax.add_patch(plt.Rectangle((0.5, 0.5), w - 1, h - 1, fill=False, ec="#C6CBD2", lw=0.5, zorder=20))
        if r is rows[0]:                                       # one scale bar for the set, on a white plate
            L = 100 / px_um
            ax.add_patch(plt.Rectangle((w * 0.975 - L - 7, h * 0.955 - 36), L + 14, 42, fc="white", ec="none",
                                       alpha=0.86, zorder=8))
            ax.plot([w * 0.975 - L, w * 0.975], [h * 0.955 - 7, h * 0.955 - 7], color=INK, lw=1.5,
                    solid_capstyle="butt", zorder=9)
            ax.text(w * 0.975 - L / 2, h * 0.955 - 12, "100 µm", ha="center", va="bottom", fontsize=5.6,
                    color=INK, zorder=9)
        save(fig, "arch_" + r["region"].lower() + "_" + SLIDE_NO[r["slide"]])

    # the gradient panel: immune share against depth, the three regions on one axis
    fig = plt.figure(figsize=(PW, PH))
    ax = fig.add_axes([0.155, 0.255, 0.825, 0.715])   # room for the x label: 0.175 clipped it
    for r in rows:
        y = prof[r["key"]]
        c = PROFILE_COLORS[rows.index(r) % len(PROFILE_COLORS)]
        ok = ~np.isnan(y)
        ax.fill_between(mid_um[ok], 0, y[ok], color=c, alpha=0.11, lw=0, zorder=2)
        ax.plot(mid_um[ok], y[ok], "-", color=c, lw=1.5, solid_capstyle="round", zorder=3)
        ax.plot(mid_um[ok], y[ok], "o", ms=2.6, color=c, mec="white", mew=0.5, zorder=4)
        ax.text(mid_um[ok][-1] + 3, y[ok][-1], f"S{SLIDE_NO[r['slide']]} {r['region']}", color=c,
                fontsize=6.2, fontweight="bold", va="center", ha="left")
    ax.set_xlim(0, h * px_um * 1.10)
    ax.set_ylim(0, 0.9)
    ax.set_xticks([0, 100, 200, 300])
    ax.set_yticks([0, 0.25, 0.5, 0.75])
    ax.set_xlabel("depth from the top edge (µm)", labelpad=2)
    ax.set_ylabel("immune share of cells", labelpad=2)
    ax.tick_params(pad=1.8)
    ax.grid(axis="y", color=GRID, lw=0.5)
    ax.set_axisbelow(True)
    ax.spines["bottom"].set_bounds(0, 300)
    ax.spines["left"].set_bounds(0, 0.75)
    save(fig, "arch_profile")

    lg = plt.figure(figsize=(2 * PW + 0.13, 0.30))
    lg.legend([plt.Line2D([], [], marker="s", ls="", ms=4.2, color=CLASS[c]) for c in CLASS], list(CLASS),
              loc="center", ncol=7, fontsize=6.2, handlelength=1.0, handletextpad=0.4, columnspacing=1.1,
              borderpad=0.0, borderaxespad=0.0)
    lg.savefig(out.parent / "arch_legend.pdf", bbox_inches="tight", pad_inches=0.012)
    plt.close(lg)
    print(f"wrote arch_* panels ({PW:.2f} x {PH:.2f} in) to {out.parent}")
    return
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out.with_suffix(".pdf"))
    fig.savefig(out.with_suffix(".png"), dpi=300)
    print("wrote", out.with_suffix(".pdf"), "and", out.with_suffix(".png"))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--regions", nargs="+", default=["235:R13", "235:R06", "236:R05"])
    ap.add_argument("--out", default=str(REPO / "LaTex" / "figures" / "regions" / "arch"))
    a = ap.parse_args()
    main(a.regions, Path(a.out))
