"""Downstream analysis of stored predictions: error taxonomy, per-cell marker tables, spatial graphs.

The machinery behind Experiments 2--4 of the paper (notebooks 09--12): everything consumes a ground-truth label map and
a stored predicted instance map (``outputs/region/<split>/<method>/<key>.npz``), so no model runs here. QC-``Wrong``
nuclei are excluded from every analysis, mirroring the evaluation protocol.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

# Error taxonomy following the DeepCell / Mesmer evaluation framework (Greenwald et al., Nature Biotechnology 2022):
# ground truth and predictions form a bipartite overlap graph (edges at IoU >= edge_iou); one-to-one pairs with
# IoU >= tp_iou are "correct"; remaining connected components are classified by their unique highest-degree node --
# a ground-truth node -> "split", a prediction node -> "merge", no unique one -> "catastrophe"; isolated ground
# truth -> "missed", isolated predictions -> "gained" (the false positives).
TAXONOMY = ("correct", "split", "merge", "catastrophe", "missed")


def overlap_fractions(gt: np.ndarray, pred: np.ndarray):
    """(iou, frac_gt, ids_gt, ids_pred): pairwise IoU plus |g∩p| / |g| for label maps with ids 1..G / 1..P.
    Rows for absent ids (holes in the numbering, e.g. after dropping QC-wrong nuclei) have zero area and are skipped
    by the callers."""
    G, P = int(gt.max()), int(pred.max())
    inter = np.bincount((gt.astype(np.int64) * (P + 1) + pred).ravel(), minlength=(G + 1) * (P + 1)).reshape(G + 1, P + 1)
    area_g = inter.sum(1)
    union = area_g[:, None] + inter.sum(0)[None, :] - inter
    iou = inter[1:, 1:] / np.maximum(union[1:, 1:], 1)
    frac_g = inter[1:, 1:] / np.maximum(area_g[1:, None], 1)
    return iou, frac_g, np.arange(1, G + 1), np.arange(1, P + 1)


def classify_errors(gt: np.ndarray, pred: np.ndarray, ignore: Optional[np.ndarray] = None,
                    tp_iou: float = 0.6, edge_iou: float = 0.1):
    """DeepCell-style error classification (Greenwald et al. 2022). Returns ``(df, gained)``:

    ``df`` has one row per scored ground-truth nucleus — its taxonomy class, best-overlapping prediction id
    (0 = none) and that IoU, centroid and area; ``gained`` is the list of predicted ids counted as false positives.
    QC-``Wrong`` nuclei are removed from the ground truth first, and predictions lying mostly (>= 50 %) on their
    pixels are excluded from ``gained`` — mirroring the benchmark's ignore convention."""
    from scipy.optimize import linear_sum_assignment

    gt = gt.copy()
    wrong_mask = None
    if ignore is not None and ignore.any():
        wrong_mask = ignore
        for g in np.unique(gt[ignore]):
            if g:
                gt[gt == g] = 0
    iou, frac_g, ids_g, ids_p = overlap_fractions(gt, pred)
    present = {int(g) for g in np.unique(gt) if g}

    correct = {}
    if iou.size:
        ri, ci = linear_sum_assignment(-iou)
        for r, c in zip(ri, ci):
            if iou[r, c] >= tp_iou:
                correct[int(ids_g[r])] = int(ids_p[c])

    # bipartite graph over everything not already a correct pair: gt -> preds and pred -> gts degree maps
    gdeg, pdeg = {}, {}
    for r, g in enumerate(ids_g):
        if int(g) not in present or int(g) in correct:
            continue
        for c, pid in enumerate(ids_p):
            if int(pid) in correct.values():
                continue
            if iou[r, c] >= edge_iou:
                gdeg.setdefault(int(g), set()).add(int(pid))
                pdeg.setdefault(int(pid), set()).add(int(g))

    cls = {g: "correct" for g in correct}
    for g in present - set(cls):
        preds = gdeg.get(g, set())
        if not preds:
            cls[g] = "missed"
        elif len(preds) >= 2 and all(len(pdeg[p]) == 1 for p in preds):
            cls[g] = "split"
        elif len(preds) == 1 and len(pdeg[next(iter(preds))]) >= 2:
            cls[g] = "merge"
        elif len(preds) == 1 and len(pdeg[next(iter(preds))]) == 1:
            cls[g] = "missed"                              # weak one-to-one overlap below tp_iou: FN + FP, per DeepCell
        else:
            cls[g] = "catastrophe"

    rows = []
    for r, g in enumerate(ids_g):
        g = int(g)
        if g not in present:
            continue
        ys, xs = np.nonzero(gt == g)
        best = int(np.argmax(iou[r])) if iou.size and iou[r].max() >= edge_iou else -1
        rows.append({"gt_id": g, "cls": cls[g], "pred_id": int(ids_p[best]) if best >= 0 else 0,
                     "iou": float(iou[r].max()) if iou.size else 0.0,
                     "cy": float(ys.mean()), "cx": float(xs.mean()), "area": int(len(ys))})
    df = pd.DataFrame(rows)

    claimed = set(correct.values()) | {p for ps in gdeg.values() for p in ps}
    gained = []
    for c, pid in enumerate(ids_p):
        pid = int(pid)
        if pid in claimed or not (pred == pid).any():
            continue
        if wrong_mask is not None:
            pm = pred == pid
            if wrong_mask[pm].mean() >= 0.5:               # sits on an expert-flagged nucleus: not a false positive
                continue
        gained.append(pid)
    return df, gained


def boundary_f1(gt: np.ndarray, pred: np.ndarray, pairs: Sequence[Tuple[int, int]], tol_px: int = 2) -> float:
    """Boundary F1 over matched pairs: boundary pixels of one map within ``tol_px`` of the other's."""
    from scipy import ndimage as ndi

    def boundary(mask):
        return mask & ~ndi.binary_erosion(mask)

    tp_p = tp_g = n_p = n_g = 0
    for g, p in pairs:
        bg, bp = boundary(gt == g), boundary(pred == p)
        if not bg.any() or not bp.any():
            continue
        dg = ndi.distance_transform_edt(~bg)
        dp = ndi.distance_transform_edt(~bp)
        tp_p += (dg[bp] <= tol_px).sum()
        n_p += bp.sum()
        tp_g += (dp[bg] <= tol_px).sum()
        n_g += bg.sum()
    prec = tp_p / max(n_p, 1)
    rec = tp_g / max(n_g, 1)
    return 2 * prec * rec / max(prec + rec, 1e-9)


def cell_marker_table(labels: np.ndarray, markers: np.ndarray, names: Sequence[str],
                      ids: Optional[Sequence[int]] = None) -> pd.DataFrame:
    """Mean marker intensity per labeled cell (vectorized): one row per id, one column per marker."""
    from scipy import ndimage as ndi

    ids = np.asarray(sorted(ids if ids is not None else [i for i in np.unique(labels) if i]))
    out = {"id": ids}
    for c, name in enumerate(names):
        out[name] = ndi.mean(markers[c].astype(np.float64), labels=labels, index=ids)
    com = ndi.center_of_mass(np.ones_like(labels, dtype=np.float64), labels=labels, index=ids)
    out["cy"] = [c[0] for c in com]
    out["cx"] = [c[1] for c in com]
    return pd.DataFrame(out)


def gate_phenotypes(cells: pd.DataFrame, thresholds: Dict[str, float]) -> pd.Series:
    """Priority gating: Epithelial if PanCK above threshold, else Immune if CD45 above, else Other."""
    pheno = pd.Series("Other", index=cells.index)
    pheno[cells["CD45"] > thresholds["CD45"]] = "Immune"
    pheno[cells["PanCK"] > thresholds["PanCK"]] = "Epithelial"          # PanCK wins ties (priority gate)
    return pheno


def otsu_threshold(values: np.ndarray) -> float:
    """Otsu's threshold on log-intensities — a simple, reproducible marker gate."""
    from skimage.filters import threshold_otsu

    v = np.log1p(np.asarray(values, float))
    return float(np.expm1(threshold_otsu(v)))


def knn_edges(cy: np.ndarray, cx: np.ndarray, k: int = 6) -> set:
    """Undirected k-nearest-neighbor edge set over cell centroids, as frozen (i, j) index pairs."""
    from scipy.spatial import cKDTree

    pts = np.column_stack([cy, cx])
    _, nn = cKDTree(pts).query(pts, k=min(k + 1, len(pts)))
    return {tuple(sorted((i, int(j)))) for i, row in enumerate(nn) for j in row[1:]}


def neighborhood_composition(pheno: Sequence[str], edges: set, classes: Sequence[str]) -> np.ndarray:
    """Per cell: the fraction of each phenotype among its graph neighbors ((n, len(classes)), rows sum to 1)."""
    idx = {c: i for i, c in enumerate(classes)}
    n = len(pheno)
    comp = np.zeros((n, len(classes)))
    for i, j in edges:
        comp[i, idx[pheno[j]]] += 1
        comp[j, idx[pheno[i]]] += 1
    return comp / np.maximum(comp.sum(1, keepdims=True), 1)


# ---------------------------------------------------------------------------------------------------------------------
# The marker panel (configs/panel.yaml): roles, display colors, and the biologist's phenotype hierarchy.
# ---------------------------------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class Phenotype:
    name: str
    marker: str
    parent: Optional[str] = None


@dataclass(frozen=True)
class Panel:
    """The panel: marker roles and colors, the nuclear / envelope channels, and the phenotype hierarchy (one marker each)."""
    markers: Dict[str, dict]
    nuclear: str
    envelope: Optional[str]
    phenotypes: Tuple[Phenotype, ...]
    gates: object = "otsu"

    @property
    def colors(self) -> Dict[str, str]:
        return {m: d["color"] for m, d in self.markers.items()}

    @property
    def names(self) -> List[str]:
        return [p.name for p in self.phenotypes]

    def chain(self, name: str) -> List[Phenotype]:
        """The phenotype and its ancestors, root first."""
        by = {p.name: p for p in self.phenotypes}
        out, cur = [], by[name]
        while cur is not None:
            out.append(cur)
            cur = by[cur.parent] if cur.parent else None
        return out[::-1]

    def lineage(self, name: str) -> str:
        return self.chain(name)[0].name

    @property
    def lineages(self) -> List[str]:
        return list(dict.fromkeys(self.lineage(p.name) for p in self.phenotypes))

    @property
    def gate_markers(self) -> List[str]:
        return list(dict.fromkeys(p.marker for p in self.phenotypes))


def load_panel(path="configs/panel.yaml") -> Panel:
    import yaml
    d = yaml.safe_load(open(path))
    phen = tuple(Phenotype(str(p["name"]), str(p["marker"]), p.get("parent")) for p in d.get("phenotypes", []))
    names = {p.name for p in phen}
    for p in phen:
        if p.parent is not None and p.parent not in names:
            raise ValueError(f"panel: phenotype {p.name!r} has unknown parent {p.parent!r}")
        if p.marker not in d.get("markers", {}):
            raise ValueError(f"panel: phenotype {p.name!r} gates on {p.marker!r}, which is not a panel marker")
    return Panel({m: dict(v) for m, v in d.get("markers", {}).items()}, d.get("nuclear", "DAPI"), d.get("envelope"), phen, d.get("gates", "otsu"))


def phenotype_gates(cells: pd.DataFrame, panel: Panel, norms: Optional[Dict[str, Tuple[float, float]]] = None) -> Dict[str, float]:
    """One positivity threshold per phenotype marker, in the units of ``cells`` (raw marker means).

    The panel's ``gates`` mapping holds the thresholds in *slide-normalized* units — the fraction of the slide's stored
    intensity window (``stats.json``: lo = 1st, hi = 99.7th percentile), the scale every network input uses — so one set
    of gates, calibrated on a biologist's tags, applies to any slide; ``norms`` = ``{marker: (lo, hi)}`` of the slide the
    cells come from converts them. Without ``norms`` the values are taken as raw. ``gates: otsu`` (an unvalidated
    histogram heuristic, kept for reference only) picks the thresholds from the cells themselves."""
    if isinstance(panel.gates, dict):
        out = {}
        for m in panel.gate_markers:
            g = float(panel.gates[m])
            out[m] = g if norms is None else float(norms[m][0] + g * (norms[m][1] - norms[m][0]))
        return out
    return {m: otsu_threshold(cells[m]) for m in panel.gate_markers}


def assign_phenotypes(cells: pd.DataFrame, panel: Panel, thresholds: Dict[str, float]):
    """Hierarchical gating: a cell takes the deepest phenotype whose whole ancestor chain is positive (marker mean above its
    threshold), ties broken by the panel's order; nothing positive -> ``Other``. Returns ``(phenotype, lineage)`` Series."""
    positive = {m: cells[m] > thresholds[m] for m in panel.gate_markers}
    pheno = pd.Series("Other", index=cells.index)
    depth = pd.Series(0, index=cells.index)
    for p in panel.phenotypes:
        chain = panel.chain(p.name)
        m = np.logical_and.reduce([positive[c.marker].values for c in chain])
        take = m & (len(chain) > depth.values)
        pheno[take], depth[take] = p.name, len(chain)
    lineage = pheno.map(lambda n: "Other" if n == "Other" else panel.lineage(n))
    return pheno, lineage
