#!/usr/bin/env python
"""The error-analysis figure of the paper (LaTex/figures/errors/{a,b,c,d}.pdf) from every finished run on the test slide:
panels a-d as separate files composed by main.tex: (a) reference cells by error class per method, (b, c) recall by
reference cell area and by distance to the nearest neighboring cell, (d) how many methods find each confirmed cell. Definitions are those of notebooks/07_test_inference.ipynb
(instance verdicts at IoU 0.5 with the expert ignore mask; Mesmer-style classes). Numbers quoted in the text are written to
errors.json next to the figure.

    conda activate dl
    python scripts/figure_errors.py                     # -> LaTex/figures/errors/{a,b,c,d}.pdf, errors.json
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import ndimage as ndi
from scipy.spatial import cKDTree

from mxdata import apply_qc_policy, ingest, load_config, load_regions
from mxdata.data import LiveSource
from mxseg.metrics import instance_outcomes, iou_matrix

REPO = Path(__file__).resolve().parents[1]
RESULTS, PRED = REPO / "results" / "region" / "test", REPO / "outputs" / "region" / "test"
ORDER = ["unet_flows", "kodiak_flows", "dinov3_flows", "kodiakb_flows", "dinov3b_flows", "cellpose_ft", "dinov3_frozen", "kodiak_frozen", "cellpose", "sam3", "sam2", "sam3_text"]
LABEL = {"unet_flows": "U-Net", "kodiak_flows": "MxIF-SSL S/16", "dinov3_flows": "DINOv3 S/16", "kodiakb_flows": "MxIF-SSL B/16", "dinov3b_flows": "DINOv3 B/16", "cellpose_ft": "Cellpose", "dinov3_frozen": "DINOv3 S/16, frozen",
         "kodiak_frozen": "MxIF-SSL S/16, frozen", "cellpose": "Cellpose, zero-shot", "sam3": "SAM 3", "sam2": "SAM 2.1", "sam3_text": "SAM 3, text"}
COLOR = {"unet_flows": "#2A78D6", "kodiak_flows": "#C43E4E", "dinov3_flows": "#B08A2E", "kodiakb_flows": "#7E1F2E", "dinov3b_flows": "#6E5314", "cellpose_ft": "#EB6834", "dinov3_frozen": "#D3BF8C",
         "kodiak_frozen": "#DF959E", "cellpose": "#8A8A8A", "sam3": "#1BAF7A", "sam2": "#9A86E0", "sam3_text": "#E87BA4"}
TAXO = {"missed": "#8E5BE0", "merged": "#1E90FF", "split": "#FF7F0E", "boundary": "#F2C230"}
TAXO_LABEL = {"missed": "missed", "merged": "merged", "split": "split", "boundary": "poor boundary"}
AREA_BINS = [0, 15, 25, 35, 50, 70, 200]
NN_BINS = [0, 5, 7, 9, 12, 100]
INK, MUTED, GRID = "#1A1C1F", "#6B7075", "#E4E6E9"
SEED_RE = re.compile(r"^(.*)_s(\d+)$")


def taxonomy(o):
    """Mesmer-style classes for every scored true cell and every scored prediction (as in notebook 07)."""
    G, P = len(o["area_g"]), len(o["area_p"])
    g_cls = np.array(["unscored"] * G, dtype=object); p_cls = np.array(["unscored"] * P, dtype=object)
    gs, ps = o["gt_status"][1:], o["pred_status"][1:]
    scored_g = np.isin(gs, ["tp", "fn"]); scored_p = np.isin(ps, ["tp", "fp"])
    if G and P:
        cg, cp = o["cover_g"], o["cover_p"]
        merged_by = {p_ for p_ in range(P) if ((cg[:, p_] >= 0.5) & scored_g).sum() >= 2}
        for g in np.nonzero(scored_g)[0]:
            if gs[g] == "tp":
                g_cls[g] = "correct"; continue
            big = np.nonzero((cg[g] >= 0.2) & scored_p)[0]
            g_cls[g] = "missed" if len(big) == 0 else "split" if len(big) >= 2 else "merged" if big[0] in merged_by else "boundary"
        for p_ in np.nonzero(scored_p)[0]:
            p_cls[p_] = "correct" if ps[p_] == "tp" else "part" if (cp[:, p_] >= 0.2).any() else "gained"
    else:
        g_cls[scored_g] = "missed"; p_cls[scored_p] = "gained"
    return g_cls, p_cls


def main(out_dir: Path):
    matplotlib.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
                                "font.size": 6.5, "axes.labelsize": 6.5, "xtick.labelsize": 6, "ytick.labelsize": 6, "legend.fontsize": 5.8,
                                "axes.linewidth": 0.5, "xtick.major.width": 0.5, "ytick.major.width": 0.5, "xtick.major.size": 1.9, "ytick.major.size": 1.9,
                                "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False, "pdf.fonttype": 42})
    out_dir.mkdir(parents=True, exist_ok=True)
    runs = []
    for m in sorted(RESULTS.glob("*/metrics.json")):
        name = m.parent.name; mm = SEED_RE.match(name)
        runs.append({"run": name, "method": mm.group(1) if mm else name, "seed": int(mm.group(2)) if mm else None})
    methods = [m for m in ORDER if any(r["method"] == m for r in runs)]
    by = {m: sorted([r for r in runs if r["method"] == m], key=lambda r: (r["seed"] is None, r["seed"])) for m in methods}

    cfg = load_config(REPO / "configs" / "dataset.yaml")
    slide = cfg.split.test_slides[0] if getattr(cfg, "split", None) and cfg.split.test_slides else "SLIDE-0236"
    paths, mx, _ = ingest(slide, None)
    px = float(mx.pixel_size_um or 0.325)
    regions = load_regions(paths, None, cfg.min_area_px)
    src = LiveSource(mx, regions, ["DAPI"], cfg.level)
    gt = {}
    for r in regions:
        rs = src.load(r.key); labels, ignore = apply_qc_policy(rs.labels, rs.flags, "ignore_wrong"); gt[r.key] = (labels, ignore)
    keys = sorted(gt)

    # ---- verdicts of every run on every region; per-cell tables ------------------------------------------------------------
    cells = []                                                                 # one row per scored reference cell per run
    tax = []                                                                   # taxonomy counts per run
    for m in methods:
        for r in by[m]:
            for k in keys:
                path = PRED / r["run"] / f"{k}.npz"
                if not path.is_file():
                    continue
                o = instance_outcomes(gt[k][0], np.load(path)["pred"].astype(np.int32), ignore=gt[k][1], thr=0.5)
                iou, area_g, area_p = iou_matrix(o["gt"], o["pred"]) if (o["gt"].max() and o["pred"].max()) else (np.zeros((0, 0)), np.zeros(0), np.zeros(0))
                inter = iou * (area_g[:, None] + area_p[None, :]) / (1 + iou)                     # IoU = inter / (a_g + a_p - inter)
                o.update(iou=iou, area_g=area_g, area_p=area_p, inter=inter, cover_g=inter / np.maximum(area_g, 1)[:, None],
                         cover_p=inter / np.maximum(area_p, 1)[None, :])
                g_cls, p_cls = taxonomy(o)
                scored = np.isin(o["gt_status"][1:], ["tp", "fn"])
                cen = np.array(ndi.center_of_mass(o["gt"] > 0, o["gt"], np.arange(1, len(o["area_g"]) + 1))) if len(o["area_g"]) else np.zeros((0, 2))
                allc = cen                                                     # all reference cells incl. flagged: neighbors exist regardless of review
                d, _ = cKDTree(allc).query(cen, k=2) if len(cen) > 1 else (np.full((len(cen), 2), np.inf), None)
                idx = np.nonzero(scored)[0]
                cells.append(pd.DataFrame({"method": m, "run": r["run"], "seed": r["seed"], "key": k, "gid": idx + 1,
                                           "hit": o["gt_status"][1:][idx] == "tp", "area_um2": o["area_g"][idx] * px * px, "nn_um": d[idx, 1] * px}))
                n = {c: int((g_cls == c).sum()) for c in ("correct", "missed", "merged", "split", "boundary")}
                n["gained"] = int((p_cls == "gained").sum()); n["scored"] = int(scored.sum())
                tax.append({"method": m, "run": r["run"], "key": k, **n})
    cells = pd.concat(cells, ignore_index=True); tax = pd.DataFrame(tax)
    print(f"{len(runs)} runs, {len(methods)} methods, {len(keys)} regions, {int(tax[tax.run == tax.run.iloc[0]].scored.sum()):,} scored cells")

    # ---- (a) taxonomy: fractions per run, then mean over seeds ---------------------------------------------------------------
    per_run = tax.groupby(["method", "run"])[["correct", "missed", "merged", "split", "boundary", "gained", "scored"]].sum().reset_index()
    for c in ("correct", "missed", "merged", "split", "boundary"):
        per_run[c + "_pct"] = 100 * per_run[c] / per_run.scored
    frac = per_run.groupby("method")[[c + "_pct" for c in ("correct", "missed", "merged", "split", "boundary")] + ["gained"]].mean().reindex(methods)
    nseeds = per_run.groupby("method").run.nunique().reindex(methods)

    # ---- (b, c) recall by area and by nearest-neighbor distance: per run, then mean over seeds --------------------------------
    cells["abin"] = pd.cut(cells.area_um2, AREA_BINS, right=False); cells["nbin"] = pd.cut(cells.nn_um, NN_BINS, right=False)
    def curve(col):
        cats = list(cells[col].cat.categories)
        per = cells.groupby(["method", "run", col], observed=False).hit.mean().unstack(col)      # one curve per run
        n = cells[cells.run == cells.run.iloc[0]].groupby(col, observed=False).size().reindex(cats, fill_value=0)
        keep = [c for c in cats if n[c] > 0]                                    # bins without cells are dropped from the axis
        g = per[keep].groupby(level="method")
        return g.mean().reindex(methods), g.min().reindex(methods), g.max().reindex(methods), n[keep]
    rec_area, lo_area, hi_area, n_area = curve("abin"); rec_nn, lo_nn, hi_nn, n_nn = curve("nbin")

    # ---- (d) consensus: seed 0 (or the only run) of every method ------------------------------------------------------------
    seeds_all = sorted({r["seed"] for m in methods for r in by[m] if r["seed"] is not None}) or [None]
    n_methods = len(methods); cons_by_seed = []
    for sd in seeds_all:                                                        # seed s of every method where it exists, else its only run
        rep = {m: next((r["run"] for r in by[m] if r["seed"] == sd), by[m][0]["run"]) for m in methods}
        hits = cells[cells.run.isin(rep.values())].groupby(["key", "gid"]).hit.sum()
        cons_by_seed.append([(hits == j).mean() for j in range(n_methods + 1)])
    cons_by_seed = np.array(cons_by_seed); cons = cons_by_seed.mean(0); cons_lo, cons_hi = cons_by_seed.min(0), cons_by_seed.max(0)
    rep = {m: by[m][0]["run"] for m in methods}
    all_runs = cells.groupby(["key", "gid"]).hit.agg(["sum", "size"])
    never_any = float((all_runs["sum"] == 0).mean())

    # ---- panels a-d as separate files; letters, headers and legends are typeset in LaTeX (like Figure 1) -------------------
    H = 1.42
    def save(fig, name):
        fig.savefig(out_dir / f"{name}.pdf"); fig.savefig(out_dir / f"{name}.png", dpi=250); plt.close(fig)

    fig = plt.figure(figsize=(2.15, H)); a = fig.add_axes([0.430, 0.2, 0.555, 0.78])   # room for "MxIF-SSL S/16, frozen"
    y = np.arange(len(methods))[::-1]; left = np.zeros(len(methods))
    for cls in ("missed", "merged", "split", "boundary"):
        v = frac[cls + "_pct"].values
        a.barh(y, v, left=left, color=TAXO[cls], height=0.7, zorder=3); left += v
    for i, m in enumerate(methods):
        tot = 100 - per_run[per_run.method == m].correct_pct.values                # total error of each seed
        a.plot(tot, np.full(len(tot), y[i]), "o", ms=2.0, color=INK, mec="white", mew=0.35, zorder=5)
        a.text(max(tot.max(), left[i]) + 1.2, y[i], f"{frac.loc[m, 'correct_pct']:.1f}% correct, {frac.loc[m, 'gained']:.0f} gained", va="center", fontsize=5.0, color=MUTED)
    a.set_yticks(y, [LABEL[m] for m in methods])
    for t, m in zip(a.get_yticklabels(), methods):
        t.set_color(COLOR[m])
    a.tick_params(axis="y", length=0, pad=2); a.set_ylim(-0.6, len(methods) - 0.4)
    a.set_xlim(0, max(max(left), (100 - frac.correct_pct).max()) * 3.45)   # room for the longest annotation
    a.set_xlabel("cells in an error class (%)", labelpad=2)
    a.grid(axis="x", color=GRID, lw=0.6, zorder=0); a.set_axisbelow(True)
    save(fig, "a")

    def curves(rec, lo, hi, xlabel, unit, name):
        fig = plt.figure(figsize=(1.05, H)); ax = fig.add_axes([0.350, 0.30, 0.605, 0.68])
        labels = [f"{iv.left:g}–{iv.right:g}" if (iv.right < 100 or unit == "um2") else f">{iv.left:g}" for iv in rec.columns]
        x = np.arange(len(labels))
        for m in methods:
            ax.fill_between(x, lo.loc[m].values, hi.loc[m].values, color=COLOR[m], alpha=0.18, lw=0, zorder=2)   # min-max over seeds
            ax.plot(x, rec.loc[m].values, "-o", color=COLOR[m], ms=2.0, lw=0.9, mec="white", mew=0.3, zorder=3)
        ax.set_xticks(x, labels, rotation=45, ha="right", rotation_mode="anchor")
        ax.set_ylim(0.35, 1.02); ax.set_yticks([0.4, 0.6, 0.8, 1.0]); ax.set_xlabel(xlabel, labelpad=2, fontsize=6); ax.set_ylabel("recall at IoU 0.5", labelpad=2)
        ax.grid(axis="y", color=GRID, lw=0.6, zorder=0); ax.set_axisbelow(True)
        save(fig, name)
    curves(rec_area, lo_area, hi_area, "cell area (µm²)", "um2", "b")
    curves(rec_nn, lo_nn, hi_nn, "nearest cell (µm)", "um", "c")

    fig = plt.figure(figsize=(1.15, H)); d = fig.add_axes([0.285, 0.30, 0.680, 0.68])
    x = np.arange(n_methods + 1); cols = ["#8E5BE0"] + ["#C7CBD1"] * (n_methods - 1) + ["#1BAF7A"]
    d.bar(x, 100 * cons, color=cols, width=0.72, zorder=3)
    d.errorbar(x, 100 * cons, yerr=[100 * (cons - cons_lo), 100 * (cons_hi - cons)], fmt="none", ecolor=INK, elinewidth=0.6, capsize=1.2, zorder=4)
    # 13 bars in 1.2 in: label only the three the text quotes, or the numbers collide into one another
    for xi in sorted({0, int(np.argmax(cons)), n_methods}):
        d.text(xi, 100 * cons_hi[xi] + 1.4, f"{100 * cons[xi]:.1f}", ha="center", va="bottom", fontsize=5.0,
               fontweight="bold" if xi in (0, n_methods) else "normal", color=INK if xi in (0, n_methods) else MUTED)
    d.set_xticks(x)
    d.set_xticklabels([f"{k}\nnone" if k == 0 else f"{k}\nall" if k == n_methods else (str(k) if k % 2 == 0 else "")
                       for k in x], fontsize=5.5)
    d.tick_params(axis="x", length=1.6)
    d.set_xlabel("methods finding the cell", labelpad=1, fontsize=5.5); d.set_ylabel("confirmed cells (%)", labelpad=2)
    d.set_ylim(0, 100 * cons_hi.max() * 1.22); d.grid(axis="y", color=GRID, lw=0.6, zorder=0); d.set_axisbelow(True)
    save(fig, "d")

    numbers = {"methods": methods, "seeds": nseeds.to_dict(), "taxonomy_pct": frac.round(2).to_dict(orient="index"),
               "recall_by_area": {m: dict(zip([str(i) for i in rec_area.columns], rec_area.loc[m].round(3))) for m in methods}, "n_area": {str(k): int(v) for k, v in n_area.items()},
               "recall_by_nn": {m: dict(zip([str(i) for i in rec_nn.columns], rec_nn.loc[m].round(3))) for m in methods}, "n_nn": {str(k): int(v) for k, v in n_nn.items()},
               "consensus_pct": [round(100 * v, 2) for v in cons], "consensus_pct_by_seed": (100 * cons_by_seed).round(2).tolist(), "never_found_any_run_pct": round(100 * never_any, 2),
               "n_runs": len(runs), "n_scored": int(tax[tax.run == tax.run.iloc[0]].scored.sum())}
    (out_dir / "errors.json").write_text(json.dumps(numbers, indent=1, default=str))
    print(frac.round(1).to_string()); print("recall by area:\n", rec_area.round(2).to_string()); print("n per area bin:", n_area.to_dict())
    print("recall by NN distance:\n", rec_nn.round(2).to_string()); print("n per NN bin:", n_nn.to_dict())
    print("consensus (% of confirmed cells found by 0..N methods, mean over seed sets):", [round(100 * v, 1) for v in cons], "min-max of none:", (100 * cons_lo[0]).round(1), (100 * cons_hi[0]).round(1), "all:", (100 * cons_lo[-1]).round(1), (100 * cons_hi[-1]).round(1), "| never found by any of", len(runs), "runs:", round(100 * never_any, 1))
    print(f"wrote panels a-d to {out_dir}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=str(REPO / "LaTex" / "figures" / "errors"))
    main(Path(ap.parse_args().out))
