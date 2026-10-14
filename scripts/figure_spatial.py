#!/usr/bin/env python
"""The two tumor--immune figures of the paper, from the results table of notebooks/analyses/10_tumor_immune_spatial.ipynb
(outputs/analysis/tumor_immune_fidelity.csv: one row per run x test region).

Every panel is a separate file drawn at its FINAL PRINT SIZE in inches, so main.tex includes it with no scaling and the
type stays at the paper's size -- the convention of scripts/figure_errors.py (LaTex/figures/errors/{a,b,c,d}.pdf). The
notebook's own wide figures are for reading the notebook; cutting them up and shrinking them to a column is what made the
earlier paper copies illegible. Legends are drawn as their own files and composed under the panels by main.tex.

    LaTex/figures/spatial/contacts_{a,b,c}.pdf          where the expert's tumor-CD8 contacts are lost   (2.35 + 1.51 + 1.51 in)
    LaTex/figures/spatial/contacts_legend_{a,b,c}.pdf   one legend under each of those panels
    LaTex/figures/spatial/pqbio_{a,b}.pdf               what predicts the tumor-CD8 endpoint          (2.62 x 2.10 in)
    LaTex/figures/spatial/pqbio_{c,d,e}.pdf             infiltration, exclusion, regulation           (1.72 x 1.80 in)
    LaTex/figures/spatial/pqbio_legend.pdf              the methods, under the paired panels
    LaTex/figures/spatial/descriptive_legend.pdf        the same methods, under the descriptive panels
    LaTex/figures/spatial/spatial.json                  every number the caption and Section 5.5 quote

    conda activate dl
    python scripts/figure_spatial.py                    # add --all-methods to include the prompted / zero-shot rows

Methods: the six rows trained on Slide 1 and carried through the downstream pipeline (U-Net, MxIF-SSL B/16, DINOv3 B/16,
Cellpose fine-tuned, and the two frozen B/16 probes; three seeds each, runs <row>_s<seed>). The prompted and zero-shot
rows of Table 2 are excluded by default -- they are not candidates for a spatial-pathology pipeline, and SAM 2.1 alone
sets the scale of every axis here. Display names follow the macros of main.tex (kodiak -> MxIF-SSL, cpdino -> Cellpose).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FuncFormatter, MultipleLocator
from scipy.stats import spearmanr

REPO = Path(__file__).resolve().parents[1]
FID_CSV = REPO / "outputs" / "analysis" / "tumor_immune_fidelity.csv"        # one row per run x region
EXPERT_CSV = REPO / "outputs" / "analysis" / "tumor_immune_expert_regions.csv"   # the expert's reading of each region

# ---- shared with scripts/figure_errors.py and \definecolor{m...} in main.tex -----------------------------------------
ORDER = ["unet_flows", "kodiakb_flows", "dinov3b_flows", "cellpose_ft", "dinov3b_frozen", "kodiakb_frozen",   # trained on Slide 1
         "cellpose", "cpdino", "instanseg", "sam3", "sam3_text", "sam2"]                                        # zero-shot / prompted
DOWNSTREAM = ORDER[:6]                                                     # the rows Section 5.5 reports
LABEL = {"unet_flows": "U-Net", "kodiakb_flows": "MxIF-SSL B/16", "dinov3b_flows": "DINOv3 B/16", "cellpose_ft": "Cellpose",
         "dinov3b_frozen": "DINOv3 B/16, frozen", "kodiakb_frozen": "MxIF-SSL B/16, frozen",
         "cellpose": "Cellpose-SAM", "cpdino": "Cellpose (DINOv3)", "instanseg": "InstanSeg", "sam3": "SAM 3", "sam3_text": "SAM 3, text", "sam2": "SAM 2.1"}
# frozen rows and the zero-shot Cellpose (DINOv3) are the 55% tint of their full fine-tune colour, as m<name>!55 in main.tex
COLOR = {"unet_flows": "#2A78D6", "kodiakb_flows": "#7E1F2E", "dinov3b_flows": "#6E5314", "cellpose_ft": "#EB6834",
         "dinov3b_frozen": "#AFA07E", "kodiakb_frozen": "#B8848C",
         "cellpose": "#8A8A8A", "cpdino": "#F4AC8F", "instanseg": "#2AA7C9", "sam3": "#1BAF7A", "sam3_text": "#E87BA4", "sam2": "#9A86E0"}
INK, MUTED, GRID = "#1A1C1F", "#6B7075", "#E4E6E9"
tick = FuncFormatter(lambda v, _: f"{v:g}")        # 0.5 not 0.50, 0.6 not 0.60

WHY = ["why_recovered", "why_cell_not_matched", "why_class_changed", "why_adjacency_lost"]
WHY_LABEL = ["recovered", "a cell not matched", "a class changed", "adjacency lost"]
WHY_COLOR = ["#1BAF7A", "#6F3CE0", "#E3B505", "#9AA3AD"]
MERGE = ["merge_tumor_tumor", "merge_tumor_T_cell", "merge_T_cell_T_cell", "merge_other"]
MERGE_LABEL = ["tumor + tumor", "tumor + T cell", "T cell + T cell", "unclassified"]
MERGE_COLOR = ["#3DB55E", "#E0218A", "#6F3CE0", "#D7DBE0"]
IFACE = ["E_homotypic", "E_interface"]
IFACE_LABEL = ["tumor–tumor", "tumor–T cell"]
IFACE_COLOR = ["#3DB55E", "#E0218A"]
REGMARK = ["o", "s", "^", "D", "v", "P"]

RC = {"font.family": "sans-serif", "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
      "font.size": 6.5, "axes.labelsize": 6.5, "xtick.labelsize": 6, "ytick.labelsize": 6, "legend.fontsize": 6.2,
      "axes.linewidth": 0.5, "xtick.major.width": 0.5, "ytick.major.width": 0.5, "xtick.major.size": 1.9, "ytick.major.size": 1.9,
      "xtick.direction": "out", "ytick.direction": "out", "axes.spines.top": False, "axes.spines.right": False,
      "legend.frameon": False, "figure.facecolor": "white", "savefig.facecolor": "white", "pdf.fonttype": 42, "ps.fonttype": 42}

# Every canvas is trimmed to its content: about 0.02 in of white outside the labels, so the panels sit flush in the grid.
CONTACT_W = (2.15, 1.50, 1.50)                 # panel widths in inches; + two 0.066 in gaps = 5.30 in, included unscaled
CONTACT_H = 1.94
CONTACT_RECT = ([0.415, 0.119, 0.547, 0.866], [0.035, 0.119, 0.905, 0.866])   # with method names ("MxIF-SSL B/16, frozen" needs 0.89 in), without
PQBIO_W, PQBIO_H = 2.40, 1.78                  # (a) carries the shared y axis
PQBIO_W2 = 2.12                                # (b) shares it: no y label or tick labels; 2.72 + 2.42 + 0.13 = 5.27 in
PQBIO_SW, PQBIO_SH = 1.52, 1.50                # bottom row: 3 x 1.72 + two 0.09 in gaps = 5.34 in
PQBIO_RECT = [0.128, 0.135, 0.845, 0.850]
PQBIO_RECT2 = [0.032, 0.135, 0.950, 0.850]     # (b): almost no left margin, the axis is (a)'s
PQBIO_SRECT = [0.190, 0.160, 0.795, 0.825]


def style(ax, axis="x"):
    """The paper's axis: two spines, a grid on the value axis only, ticks under the data."""
    ax.grid(axis="y" if axis == "y" else "x", color=GRID, lw=0.55, zorder=0)
    ax.set_axisbelow(True)
    ax.tick_params(pad=1.8)


def rows(ax, methods, labelled):
    """The shared method rows of the contact figure: newest at the top, names only on the left-most panel."""
    y = np.arange(len(methods))[::-1]
    ax.set_ylim(-0.62, len(methods) - 0.38)
    ax.set_yticks(y)
    ax.set_yticklabels([LABEL[m] for m in methods] if labelled else [])
    if labelled:
        for t, m in zip(ax.get_yticklabels(), methods):
            t.set_color(COLOR[m])
    ax.tick_params(axis="y", length=0, pad=2)
    return y


def legend_file(path, handles, labels, width, ncol, height=0.36):
    """One legend as its own file, cut to its content: main.tex centres it in a \\makebox as wide as its panel."""
    fig = plt.figure(figsize=(width, height))
    fig.legend(handles, labels, loc="center", ncol=ncol, handlelength=1.15, handletextpad=0.42,
               columnspacing=1.1, labelspacing=0.5, borderpad=0.0, borderaxespad=0.0)
    fig.savefig(path, bbox_inches="tight", pad_inches=0.012)
    plt.close(fig)


def patch(color):
    return matplotlib.patches.Patch(facecolor=color, edgecolor="none")


def dot(color, marker="o", ms=3.0, **kw):
    return plt.Line2D([], [], ls="", marker=marker, color=color, ms=ms, **kw)


def main(out_dir: Path, methods: list[str]) -> None:
    matplotlib.rcParams.update(RC)
    out_dir.mkdir(parents=True, exist_ok=True)
    fid = pd.read_csv(FID_CSV)
    fid = fid[fid.method.isin(methods)].copy()
    methods = [m for m in methods if m in set(fid.method)]
    keys = sorted(fid.region.unique())
    mark = dict(zip(keys, REGMARK))
    n_runs = fid.run.nunique()
    print(f"{n_runs} runs of {len(methods)} methods on {len(keys)} regions, {len(fid)} run x region points")

    def save(fig, name):
        fig.savefig(out_dir / f"{name}.pdf")
        fig.savefig(out_dir / f"{name}.png", dpi=400)
        plt.close(fig)

    # ================= Figure: where the expert's tumor-CD8 contacts are lost =============================================
    per_run = fid.groupby(["method", "run"])[WHY + MERGE].sum().reset_index()
    why = per_run.groupby("method")[WHY].mean().reindex(methods)
    why = 100 * why.div(why.sum(1), axis=0)                                # share of the expert's contacts, mean over seeds
    merges = per_run.groupby("method")[MERGE].mean().reindex(methods) / len(keys)
    iface = 100 * fid.groupby("method")[IFACE].mean().reindex(methods)

    # ---- (a) the fate of every expert tumor-CD8 contact ------------------------------------------------------------------
    fig = plt.figure(figsize=(CONTACT_W[0], CONTACT_H))
    ax = fig.add_axes(CONTACT_RECT[0])
    y = rows(ax, methods, labelled=True)
    left = np.zeros(len(methods))
    for c, col in zip(WHY, WHY_COLOR):
        ax.barh(y, why[c].values, left=left, color=col, height=0.68, lw=0, zorder=3)
        left += why[c].values
    for i, m in enumerate(methods):                                        # the recovered share, printed where it ends
        ax.text(why.loc[m, "why_recovered"] - 1.5, y[i], f"{why.loc[m, 'why_recovered']:.0f}", color="white",
                fontsize=5.2, fontweight="bold", ha="right", va="center", zorder=5)
    ax.set_xlim(0, 100)
    ax.xaxis.set_major_locator(MultipleLocator(25))
    ax.set_xlabel("expert tumor–CD8 contacts (%)", labelpad=2)
    style(ax, "x")
    save(fig, "contacts_a")
    legend_file(out_dir / "contacts_legend_a.pdf", [patch(c) for c in WHY_COLOR], WHY_LABEL, CONTACT_W[0], ncol=2, height=0.32)

    # ---- (b) what the merged predictions fuse ----------------------------------------------------------------------------
    fig = plt.figure(figsize=(CONTACT_W[1], CONTACT_H))
    ax = fig.add_axes(CONTACT_RECT[1])
    y = rows(ax, methods, labelled=False)
    left = np.zeros(len(methods))
    for c, col in zip(MERGE, MERGE_COLOR):
        ax.barh(y, merges[c].values, left=left, color=col, height=0.68, lw=0, zorder=3)
        left += merges[c].values
    for i, m in enumerate(methods):                                        # the sliver that matters, as a number
        ax.text(left[i] - 1.5, y[i], f"{merges.loc[m, 'merge_tumor_T_cell']:.1f}", color=MERGE_COLOR[1],
                fontsize=5.2, fontweight="bold", ha="right", va="center", zorder=5)
    ax.set_xlim(0, 30 * np.ceil(left.max() / 30))
    ax.xaxis.set_major_locator(MultipleLocator(30))
    ax.set_xlabel("merged predictions per region", labelpad=2)
    style(ax, "x")
    save(fig, "contacts_b")
    legend_file(out_dir / "contacts_legend_b.pdf", [patch(c) for c in MERGE_COLOR], MERGE_LABEL, CONTACT_W[1], ncol=2, height=0.32)

    # ---- (c) how often a contact of each kind is touched by a segmentation error -----------------------------------------
    fig = plt.figure(figsize=(CONTACT_W[2], CONTACT_H))
    ax = fig.add_axes(CONTACT_RECT[1])
    y = rows(ax, methods, labelled=False)
    h = 0.30
    for k, (c, col) in enumerate(zip(IFACE, IFACE_COLOR)):
        off = (0.5 - k) * h
        ax.barh(y + off, iface[c].values, height=h, color=col, lw=0, zorder=3)
        per = 100 * fid.groupby(["method", "region"])[c].mean().unstack("region").reindex(methods)
        for i in range(len(methods)):
            ax.plot(per.iloc[i].values, np.full(per.shape[1], y[i] + off), ".", ms=1.9, color=INK, alpha=0.85, zorder=4)
    ax.set_xlim(0, 100)
    ax.xaxis.set_major_locator(MultipleLocator(25))
    ax.set_xlabel("contacts touched by an error (%)", labelpad=2)
    style(ax, "x")
    save(fig, "contacts_c")
    legend_file(out_dir / "contacts_legend_c.pdf",
                [patch(IFACE_COLOR[0]), patch(IFACE_COLOR[1]), dot(INK, ms=2.2)],
                IFACE_LABEL + ["one region"], CONTACT_W[2], ncol=2, height=0.32)

    # ================= Figure: segmentation quality against biological fidelity ===========================================
    expert = pd.read_csv(EXPERT_CSV, index_col=0)                           # the expert's reading of each region
    for c in ("cd8_touching_frac", "cd8_distant_frac", "cd8_regulated_frac"):
        fid[c + "_ref"] = fid.region.map(expert[c])                         # the fidelity table stores run - expert
        fid[c + "_run"] = fid[c + "_ref"] + fid[c + "_err"]

    def scatter(name, x, yv, xlabel, ylabel, size, identity=False, means=False, xlim=None, ylim=None, xstep=0.5,
                rho_at="upper left", jitter=0.0, dodge=False, share_y=False):
        """One correlation panel. 90-108 points in two inches: they read as a cloud, so every run x region point is one
        small translucent dot in the method's colour (a second, shape-based encoding of region was illegible at this
        size and only added clutter), the method means sit on top as large white-filled rings, and a median-of-quantile-
        and the two clouds are left to make the contrast themselves: no fitted line, since over the narrow PQ range a
        least-squares slope looks like a relationship even at rho = 0.20."""
        fig = plt.figure(figsize=size)
        ax = fig.add_axes(PQBIO_RECT2 if share_y else PQBIO_RECT if size[0] > 2 else PQBIO_SRECT)
        for i, m in enumerate(methods):
            d = fid[fid.method == m]
            xs = d[x].to_numpy(float)
            if dodge:                                                      # one slot per method around its expert value
                xs = xs + (i - (len(methods) - 1) / 2) * (2 * jitter / max(len(methods) - 1, 1))
            ax.plot(xs, d[yv], ls="", marker="o", ms=2.9, color=COLOR[m], mew=0, alpha=0.62, zorder=3)
        ok = fid[[x, yv]].dropna()
        rho = float(spearmanr(ok[x], ok[yv])[0])
        for m in methods:
            if means:
                d = fid[fid.method == m]
                ax.plot(d[x].mean(), d[yv].mean(), ls="", marker="o", ms=9.0, mfc="white", mec="white", mew=2.4, zorder=6)
                ax.plot(d[x].mean(), d[yv].mean(), ls="", marker="o", ms=8.0, mfc="white", mec=COLOR[m], mew=1.6, zorder=7)
        if identity:
            ax.plot([-0.02, 1.02], [-0.02, 1.02], color=MUTED, lw=0.6, ls=(0, (3, 2)), zorder=1)
            ax.set_xlim(-0.04, 1.04)
            ax.set_ylim(-0.04, 1.04)
        ax.set_xlim(*xlim) if xlim else None
        ax.set_ylim(*ylim) if ylim else None
        ax.xaxis.set_major_locator(MultipleLocator(xstep))
        ax.xaxis.set_minor_locator(MultipleLocator(xstep / 2))
        ax.yaxis.set_major_locator(MultipleLocator(0.5))
        ax.yaxis.set_minor_locator(MultipleLocator(0.25))
        ax.xaxis.set_major_formatter(tick)
        ax.yaxis.set_major_formatter(tick)
        lo_x, hi_x = ax.get_xlim(); lo_y, hi_y = ax.get_ylim()
        ax.spines["bottom"].set_bounds(np.ceil(lo_x / xstep) * xstep, np.floor(hi_x / xstep) * xstep)
        ax.spines["left"].set_bounds(max(0.0, np.ceil(lo_y * 2) / 2), min(1.0, np.floor(hi_y * 2) / 2))
        ax.tick_params(which="minor", length=1.1, width=0.5)
        ax.tick_params(pad=1.8)
        ax.set_xlabel(xlabel, labelpad=2)
        if share_y:
            ax.set_yticklabels([])
        else:
            ax.set_ylabel(ylabel, labelpad=2)
        big = size[0] > 2                                                  # the two paired panels carry the headline number
        x0, ha = (0.03, "left") if rho_at == "upper left" else (0.97, "right")
        ax.text(x0, 0.995, f"$\\rho$ = {rho:+.2f}", transform=ax.transAxes, va="top", ha=ha,
                fontsize=9.5 if big else 7.0, fontweight="bold", color=INK)
        ax.text(x0, 0.995 - (0.085 if big else 0.105), f"n = {len(ok)}", transform=ax.transAxes, va="top", ha=ha,
                fontsize=6.2 if big else 5.6, color=MUTED)
        save(fig, name)
        return rho

    big, small = (PQBIO_W, PQBIO_H), (PQBIO_SW, PQBIO_SH)
    # ---- (a, b) the key test: does an overlap metric predict the endpoint, or does the interface error rate? -------------
    r_pq = scatter("pqbio_a", "PQ", "contact_f1", "Panoptic Quality of the region", "tumor–CD8 contact F1", big,
                   means=True, xlim=(0.588, 0.772), ylim=(-0.04, 1.06), xstep=0.05)
    r_ei = scatter("pqbio_b", "E_interface", "contact_f1", "tumor–immune interface error rate", "", (PQBIO_W2, PQBIO_H), share_y=True,
                   means=True, xlim=(-0.04, 1.04), ylim=(-0.04, 1.06), rho_at="upper right")
    # ---- (c-e) the three descriptive endpoints against the expert's own value on the same region -------------------------
    r_c = scatter("pqbio_c", "cd8_touching_frac_ref", "cd8_touching_frac_run", "expert", "method", small, identity=True, jitter=0.055, dodge=True)
    r_d = scatter("pqbio_d", "cd8_distant_frac_ref", "cd8_distant_frac_run", "expert", "method", small, identity=True, jitter=0.055, dodge=True)
    r_e = scatter("pqbio_e", "cd8_regulated_frac_ref", "cd8_regulated_frac_run", "expert", "method", small, identity=True, jitter=0.055, dodge=True)

    # ---- the shared legend: methods on two rows, then the six regions ----------------------------------------------------
    fig = plt.figure(figsize=(PQBIO_W + PQBIO_W2 + 0.13, 0.40))
    h_m = [dot(COLOR[m], ms=3.4) for m in methods] + [plt.Line2D([], [], ls="", marker="o", ms=3.8, mfc="none", mec=INK, mew=1.0)]
    lg = fig.legend(h_m, [LABEL[m] for m in methods] + ["method mean"], loc="upper center", bbox_to_anchor=(0.5, 1.0),
                    ncol=5, handlelength=0.9, handletextpad=0.42, columnspacing=1.4, labelspacing=0.55,
                    borderpad=0.0, borderaxespad=0.0)
    fig.add_artist(lg)
    fig.savefig(out_dir / "pqbio_legend.pdf", bbox_inches="tight", pad_inches=0.012)
    plt.close(fig)

    # the descriptive panels (c-e) plot no method means, so they get their own key without that entry
    fig = plt.figure(figsize=(3 * PQBIO_SW + 0.18, 0.34))
    fig.legend([dot(COLOR[m], ms=3.4) for m in methods], [LABEL[m] for m in methods], loc="upper center",
               bbox_to_anchor=(0.5, 1.0), ncol=4, handlelength=0.9, handletextpad=0.42, columnspacing=1.4,
               labelspacing=0.55, borderpad=0.0, borderaxespad=0.0)
    fig.savefig(out_dir / "descriptive_legend.pdf", bbox_inches="tight", pad_inches=0.012)
    plt.close(fig)

    # ---- the numbers the caption and Section 5.5 quote --------------------------------------------------------------------
    per_method = fid.groupby("method")
    numbers = {
        "methods": methods, "n_runs": n_runs, "n_regions": len(keys),
        "n_expert_contacts": int(fid[fid.run == fid.run.iloc[0]].n_expert_contacts.sum()),
        "contact_f1": per_method.contact_f1.mean().reindex(methods).round(3).to_dict(),
        "PQ": per_method.PQ.mean().reindex(methods).round(3).to_dict(),
        "F1": per_method.F1.mean().reindex(methods).round(3).to_dict(),
        "why_pct": why.round(1).to_dict(orient="index"),
        "merges_per_region": merges.round(1).to_dict(orient="index"),
        "interface_error_pct": iface.round(1).to_dict(orient="index"),
        "spearman_with_contact_f1": {"PQ": round(r_pq, 3),
                                     "F1": round(float(spearmanr(*fid[["F1", "contact_f1"]].dropna().values.T)[0]), 3),
                                     "Dice": round(float(spearmanr(*fid[["Dice", "contact_f1"]].dropna().values.T)[0]), 3),
                                     "E_interface": round(float(spearmanr(*fid[["E_interface", "contact_f1"]].dropna().values.T)[0]), 3),
                                     # E_interface counts the endpoint's own contacts; the independent variant excludes them
                                     "E_interface_other": round(float(spearmanr(*fid[["E_interface_other", "contact_f1"]].dropna().values.T)[0]), 3),
                                     "E_homotypic": round(float(spearmanr(*fid[["E_homotypic", "contact_f1"]].dropna().values.T)[0]), 3)},
        "identity_spearman": {"touching": round(r_c, 3), "distant": round(r_d, 3), "regulated": round(r_e, 3)},
        "spearman_interface_error": round(r_ei, 3),
        "call_agreement": per_method.call_agree.mean().reindex(methods).round(3).to_dict(),
    }
    (out_dir / "spatial.json").write_text(json.dumps(numbers, indent=1, default=str))
    print(why.round(1).to_string())
    print(merges.round(1).to_string())
    print(iface.round(1).to_string())
    print("contact F1:", {LABEL[m]: round(v, 3) for m, v in numbers["contact_f1"].items()})
    print("Spearman with contact F1:", numbers["spearman_with_contact_f1"])
    print(f"wrote contacts_{{a,b,c}}, pqbio_{{a,b,c,d,e}} and their legends to {out_dir}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=str(REPO / "LaTex" / "figures" / "spatial"))
    ap.add_argument("--all-methods", action="store_true", help="include the prompted and zero-shot rows of Table 2")
    a = ap.parse_args()
    main(Path(a.out), ORDER if a.all_methods else DOWNSTREAM)
