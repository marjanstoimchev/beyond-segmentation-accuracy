#!/usr/bin/env python
"""The README's results-at-a-glance figure, light and dark variants: every method as a row, the two held-out slides' segmentation F1
side by side, then the four levels of the hierarchy as small multiples with the value printed beside each mark.
Reads outputs/analysis/{benchmark_region, benchmark_region_SLIDE-0233, phenotype_summary, neighborhood_summary, niche_summary,
tumor_immune_fidelity}.csv (the chain 02 -> 05 + notebook 10) and writes media/results_light.png, media/results_dark.png.

    python scripts/figure_readme_results.py
"""
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

REPO = Path(__file__).resolve().parents[1]
A = REPO / "outputs" / "analysis"
OUT = REPO / "media"

ORDER = ["cellpose_ft", "kodiakb_flows", "unet_flows", "dinov3b_flows", "dinov3b_frozen", "kodiakb_frozen",
         "sam3", "sam3_text", "sam2", "cellpose", "cpdino", "instanseg"]
LABEL = {"cellpose_ft": "Cellpose (DINOv3 B/16), fine-tuned", "kodiakb_flows": "KODIAK B/16, fine-tuned", "unet_flows": "U-Net",
         "dinov3b_flows": "DINOv3 B/16, fine-tuned", "dinov3b_frozen": "DINOv3 B/16, frozen", "kodiakb_frozen": "KODIAK B/16, frozen",
         "sam3": "SAM 3, exemplar prompts", "sam3_text": "SAM 3, text prompt", "sam2": "SAM 2.1, prototype prompts",
         "cellpose": "Cellpose-SAM, zero-shot", "cpdino": "Cellpose (DINOv3), zero-shot", "instanseg": "InstanSeg, zero-shot"}
GROUP = {"cellpose_ft": "fine-tuned", "kodiakb_flows": "fine-tuned", "unet_flows": "fine-tuned", "dinov3b_flows": "fine-tuned",
         "dinov3b_frozen": "frozen", "kodiakb_frozen": "frozen", "sam3": "prompted", "sam3_text": "prompted", "sam2": "prompted",
         "cellpose": "released", "cpdino": "released", "instanseg": "released"}
# the four supervision groups: blue / yellow / magenta / green, the one four-hue subset of the reference palette that passes
# the all-pairs colour-vision checks on both surfaces (scripts/validate_palette.js of the dataviz skill)
THEME = {
    "light": dict(surface="#fcfcfb", ink="#0b0b0b", muted="#52514e", grid="#e6e5e1",
                  group={"fine-tuned": "#2a78d6", "frozen": "#eda100", "prompted": "#008300", "released": "#e87ba4"}),
    "dark": dict(surface="#0d1117", ink="#ffffff", muted="#c3c2b7", grid="#30363d",
                 group={"fine-tuned": "#3987e5", "frozen": "#c98500", "prompted": "#008300", "released": "#d55181"}),
}


def load():
    b2 = pd.read_csv(A / "benchmark_region.csv", index_col=0)
    b3 = pd.read_csv(A / "benchmark_region_SLIDE-0233.csv", index_col=0)
    ph = pd.read_csv(A / "phenotype_summary.csv", index_col=0)
    nb = pd.read_csv(A / "neighborhood_summary.csv", index_col=0)
    ni = pd.read_csv(A / "niche_summary.csv", index_col=0)
    ti = pd.read_csv(A / "tumor_immune_fidelity.csv").groupby("method").contact_f1.mean()
    d = pd.DataFrame({"f1_s2": b2["f1@0.5"], "f1_s3": b3["f1@0.5"], "pheno": ph.type_agreement, "edges": nb.edge_recall_all,
                      "niche": ni.niche_agreement, "contact": ti})
    return d.reindex(ORDER)


def draw(d, mode):
    t = THEME[mode]
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 13, "axes.edgecolor": t["grid"], "axes.linewidth": 0.8,
                         "xtick.color": t["muted"], "ytick.color": t["ink"], "axes.labelcolor": t["muted"], "text.color": t["ink"]})
    cols = [("f1_s2", "1 · Segmentation\nF1 at IoU 0.5", (0.15, 0.95), "{:.3f}"), ("pheno", "2 · Phenotype\nagreement, %", (82, 98), "{:.1f}"),
            ("edges", "3 · Neighborhood\nedge recall", (0.55, 0.97), "{:.2f}"), ("niche", "4a · Niche\nagreement, %", (83, 94), "{:.1f}"),
            ("contact", "4b · Tumor–CD8\ncontact F1", (-0.04, 0.68), "{:.2f}")]
    fig = plt.figure(figsize=(14.5, 8.6), dpi=170); fig.patch.set_facecolor(t["surface"])
    widths = [2.1, 1, 1, 1, 1]; left = 0.265; gap = 0.03; avail = 0.99 - left - gap * (len(cols) - 1); tot = sum(widths)
    y = np.arange(len(d))[::-1]
    x0 = left
    for k, (col, title, lim, fmt) in enumerate(cols):
        w = avail * widths[k] / tot
        ax = fig.add_axes([x0, 0.115, w, 0.675]); x0 += w + gap
        ax.set_facecolor(t["surface"])
        for sp in ("top", "right", "left"): ax.spines[sp].set_visible(False)
        ax.grid(axis="x", color=t["grid"], lw=0.8, zorder=0); ax.set_axisbelow(True)
        span = lim[1] - lim[0]; pad = 0.62 if col == "f1_s2" else 0.34                     # a value column at the right of each panel
        ax.set_xlim(lim[0], lim[1] + pad * span); ax.set_xticks([t for t in ax.get_xticks() if lim[0] <= t <= lim[1]])
        ax.set_ylim(-0.7, len(d) - 0.3); ax.set_yticks(y)
        ax.set_yticklabels([LABEL[m] for m in d.index] if k == 0 else [""] * len(d), fontsize=13)
        ax.tick_params(axis="y", length=0); ax.tick_params(axis="x", length=3, labelsize=11)
        ax.set_title(title, loc="left", fontsize=13, fontweight="bold", color=t["ink"], pad=10, linespacing=1.15)
        colors = [t["group"][GROUP[m]] for m in d.index]
        v = d[col].values
        if col == "f1_s2":                                              # slide 2 filled, slide 3 hollow, joined by a hairline
            v3 = d["f1_s3"].values
            for yi, a, b, c in zip(y, v, v3, colors):
                ax.plot([a, b], [yi, yi], color=c, lw=1.4, alpha=0.55, zorder=2)
            ax.scatter(v3, y, s=95, facecolor=t["surface"], edgecolor=colors, linewidth=2.0, zorder=3)
            ax.scatter(v, y, s=95, color=colors, edgecolor=t["surface"], linewidth=1.2, zorder=4)
            for yi, a, b in zip(y, v, v3):
                ax.text(lim[1] + pad * span, yi, f"{a:.3f}  {b:.3f}", va="center", ha="right", fontsize=11, color=t["muted"])
            ax.text(lim[1] + pad * span, len(d) - 0.25, "S2       S3", va="bottom", ha="right", fontsize=10, color=t["muted"])
        else:
            ax.scatter(v, y, s=95, color=colors, edgecolor=t["surface"], linewidth=1.2, zorder=4)
            for yi, a in zip(y, v):
                ax.text(lim[1] + pad * span, yi, fmt.format(a), va="center", ha="right", fontsize=11, color=t["muted"])
    # legend: the four groups + the slide markers
    g = t["group"]
    handles = [Line2D([], [], marker="o", ls="", ms=10, color=g[k], label=k) for k in ("fine-tuned", "frozen", "prompted", "released")]
    handles += [Line2D([], [], marker="o", ls="", ms=10, color=t["muted"], label="Slide 2 (test)"),
                Line2D([], [], marker="o", ls="", ms=10, markerfacecolor=t["surface"], markeredgecolor=t["muted"], markeredgewidth=2.0, label="Slide 3 (reviewed fields)")]
    leg = fig.legend(handles=handles, loc="lower left", bbox_to_anchor=(0.265, 0.0), ncol=6, frameon=False, fontsize=12,
                     handletextpad=0.4, columnspacing=1.3)
    for txt in leg.get_texts(): txt.set_color(t["muted"])
    fig.text(0.01, 0.955, "Twelve segmenters on the held-out slides, mean over three seeds where a row has them", fontsize=15,
             fontweight="bold", color=t["ink"], ha="left", va="center")
    fig.text(0.01, 0.915, "Level 1 on both held-out slides (S2 = Slide 2, S3 = the reviewed fields of Slide 3), levels 2–4 on Slide 2: the levels do not order the methods alike.",
             fontsize=12, color=t["muted"], ha="left", va="center")
    fig.savefig(OUT / f"results_{mode}.png", facecolor=t["surface"]); plt.close(fig)


if __name__ == "__main__":
    d = load(); OUT.mkdir(exist_ok=True)
    for mode in ("light", "dark"): draw(d, mode)
    print(d.round(3).to_string()); print("written:", sorted(p.name for p in OUT.glob("results_*.png")))
