#!/usr/bin/env python
"""Calibrate the panel gates (configs/panel.yaml) on the biologist's phenotype labels of the development slide, read from the unified
store (<slide>/phenotyping/, mxdata.phenotyping). For every gated phenotype the marker's positivity threshold, in slide-normalized
units, is the value that maximizes F1 over the tagged nuclei against the untagged nuclei of the same session scope; children of
T cell (CD4+, CD8+, Treg) are calibrated within the T-cell compartment, as the hierarchical rule applies them. Leave-one-area-out
gives the honest F1 of each gate on an area it was not fitted on. Writes outputs/analysis/gates_calibration.csv and rewrites the
`gates:` values of configs/panel.yaml for the calibrated markers (uPAR and CD45, which have no tags, are left as they are).

    conda activate dl
    python scripts/calibrate_gates.py                 # SLIDE-0235 areas R11, R12, W1
"""
from __future__ import annotations

import argparse
import datetime as _dt
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import ndimage as ndi

from mxdata import apply_qc_policy, ingest, load_config, load_regions
from mxdata.data import LiveSource
from mxdata.phenotyping import load_areas, load_tagged_nuclei
from mxdata.pipeline.stats import NormStats
from mxseg.analysis import load_panel

REPO = Path(__file__).resolve().parents[1]
CHILDREN_OF_T = {"CD4+ T cell", "CD8+ T cell", "Treg"}
EXCLUDE = {"PanCK": {"W1"}}          # the window's crypt epithelium barely stains for PanCK (00c): a robustness check, not a calibration set
T_LINEAGE = {"T cell"} | CHILDREN_OF_T


def nucleus_table(mx, region, markers, level, norms) -> pd.DataFrame:
    """One row per confirmed nucleus of a HALO region: label id, centroid, mean of every marker in slide-normalized units."""
    rs = LiveSource(mx, [region], markers, level).load(region.key)
    labels, ignore = apply_qc_policy(rs.labels, rs.flags, "ignore_wrong")
    ids = np.arange(1, rs.labels.max() + 1)
    flagged = ndi.mean(ignore.astype(float), rs.labels, ids) > 0.5
    cen = np.array(ndi.center_of_mass(rs.labels > 0, rs.labels, ids))
    t = pd.DataFrame({"nucleus_id": ids, "cx": cen[:, 1] + region.box.x0, "cy": cen[:, 0] + region.box.y0, "flagged": flagged})
    for m in markers:
        lo, hi = norms[m]; t[m] = (ndi.mean(rs.image[markers.index(m)].astype(np.float32), rs.labels, ids) - lo) / (hi - lo)
    return t


def best_threshold(x: np.ndarray, y: np.ndarray):
    """The threshold on x that maximizes F1 of (x > t) against y, and that F1."""
    order = np.argsort(-x); xs, ys = x[order], y[order]
    tp = np.cumsum(ys); fp = np.cumsum(~ys); fn = ys.sum() - tp
    f1 = 2 * tp / np.maximum(2 * tp + fp + fn, 1)
    k = int(f1.argmax()); t = (xs[k] + xs[k + 1]) / 2 if k + 1 < len(xs) else xs[k]
    return float(t), float(f1[k])


def scores(x, y, t):
    p = x > t; tp = int((p & y).sum()); fp = int((p & ~y).sum()); fn = int((~p & y).sum())
    prec, rec = tp / max(tp + fp, 1), tp / max(tp + fn, 1)
    from scipy.stats import mannwhitneyu
    au = mannwhitneyu(x[y], x[~y], alternative="greater").statistic / (y.sum() * (~y).sum()) if y.sum() > 2 and (~y).sum() > 2 else np.nan
    return dict(precision=prec, recall=rec, f1=2 * prec * rec / max(prec + rec, 1e-9), auroc=au, n_pos=int(y.sum()), n_neg=int((~y).sum()))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--slide", default="SLIDE-0235"); ap.add_argument("--areas", nargs="+", default=["R11", "R12", "W1"])
    ap.add_argument("--write-panel", action="store_true", default=True); ap.add_argument("--no-write-panel", dest="write_panel", action="store_false")
    args = ap.parse_args()
    cfg = load_config(REPO / "configs" / "dataset.yaml"); panel = load_panel(REPO / "configs" / "panel.yaml"); markers = list(cfg.markers)
    paths, mx, _ = ingest(args.slide, None); regions = {r.region_id: r for r in load_regions(paths, None, cfg.min_area_px)}
    st = NormStats.load(Path(cfg.out if hasattr(cfg, "out") else REPO / "dataset") / args.slide / "stats.json") if False else None
    from mxdata import DEFAULT_DATASET
    st = NormStats.load(Path(DEFAULT_DATASET) / args.slide / "stats.json"); norms = {m: (float(st.lo[m]), float(st.hi[m])) for m in markers}
    areas = load_areas(args.slide).set_index("area"); tagged = load_tagged_nuclei(args.slide)
    tables = {}
    for a in args.areas:
        row = areas.loc[a]
        rid = int(a[1:]) if row.kind == "region" else next(r.region_id for r in regions.values() if r.box.x0 <= row.x0 < r.box.x1 and r.box.y0 <= row.y0 < r.box.y1)
        t = nucleus_table(mx, regions[rid], markers, cfg.level, norms)
        if row.kind == "window":                                                # only nuclei inside the window are in the sessions' scope
            t = t[(t.cx >= row.x0) & (t.cx < row.x1) & (t.cy >= row.y0) & (t.cy < row.y1)]
        tg = tagged[tagged.area == a].set_index("nucleus_id").tags
        t = t.assign(tags=t.nucleus_id.map(tg).fillna(""), area=a)
        t = t[~t.flagged]; tables[a] = t
        print(f"{a}: {len(t)} confirmed nuclei in scope, {int((t.tags != '').sum())} tagged")
    allt = pd.concat(tables.values(), ignore_index=True)
    rows, gates = [], {}
    for ph in [p for p in panel.phenotypes if p.marker in ("PanCK", "CD3", "CD4", "CD8", "FOXP3")]:
        m = ph.marker
        def scope(t):
            has_session = t.area.map(lambda a: ph.name in str(areas.loc[a].sessions).split(";") and a not in EXCLUDE.get(m, set()))
            t = t[has_session]
            if ph.name in CHILDREN_OF_T:                                          # gated within T cells: negatives are other T-lineage tagged nuclei
                t = t[t.tags.map(lambda s: any(x in T_LINEAGE for x in s.split(";")))]
            return t
        s = scope(allt)
        if s.empty or (s.tags.str.contains(ph.name, regex=False)).sum() < 5:
            print(f"{ph.name}: too few tags ({int(s.tags.str.contains(ph.name, regex=False).sum())}) — gate left unchanged"); continue
        y = s.tags.str.contains(ph.name, regex=False).values; x = s[m].values.astype(float)
        t_all, f1_all = best_threshold(x, y); gates[m] = t_all
        for a in sorted(s.area.unique()):                                        # leave-one-area-out
            tr, te = s[s.area != a], s[s.area == a]
            if te.tags.str.contains(ph.name, regex=False).sum() < 3 or tr.tags.str.contains(ph.name, regex=False).sum() < 5:
                continue
            t_tr, _ = best_threshold(tr[m].values.astype(float), tr.tags.str.contains(ph.name, regex=False).values)
            sc = scores(te[m].values.astype(float), te.tags.str.contains(ph.name, regex=False).values, t_tr)
            rows.append({"phenotype": ph.name, "marker": m, "held_out": a, "threshold_fit_elsewhere": t_tr, **sc})
        sc = scores(x, y, t_all); rows.append({"phenotype": ph.name, "marker": m, "held_out": "none (fit on all)", "threshold_fit_elsewhere": t_all, **sc})
        print(f"{ph.name:12s} {m:6s} gate {t_all:.4f} (norm. units) | fit-on-all F1 {f1_all:.2f}, AUROC {sc['auroc']:.2f}, n+ {sc['n_pos']}, n- {sc['n_neg']}")
    out = REPO / "outputs" / "analysis"; out.mkdir(parents=True, exist_ok=True)
    R = pd.DataFrame(rows); R.to_csv(out / "gates_calibration.csv", index=False); print(f"wrote {out / 'gates_calibration.csv'}")
    print(R.round(3).to_string())
    if args.write_panel and gates:
        p = REPO / "configs" / "panel.yaml"; s = p.read_text()
        for m, g in gates.items():
            cv = R[(R.marker == m) & (R.held_out != "none (fit on all)")]
            note = f"calibrated {_dt.date.today().isoformat()} on the R11/R12 boxes + W1 tags of SLIDE-0235 ({int(R[(R.marker == m)].n_pos.max())} tagged nuclei); leave-one-area-out F1 {cv.f1.mean():.2f}" if len(cv) else "calibrated on the development-slide labels"
            s, n = re.subn(rf"^(  {m}:\s+)[0-9.]+(\s*#.*)?$", lambda mm: f"{mm.group(1)}{g:.4f}            # {note}", s, count=1, flags=re.M)
            assert n == 1, m
        s = re.sub(r"# two-Gaussian mixture gate on the test slide's expert cells \(label-free, identical for every method\)\n", "", s)
        s = re.sub(r"\n\s+#   \(0\.018\) sits below the median cell of the test slide; until PanCK is tagged on a bright region this is a\n", "\n", s)
        p.write_text(s); print("configs/panel.yaml gates updated:", {m: round(g, 4) for m, g in gates.items()})


if __name__ == "__main__":
    main()
