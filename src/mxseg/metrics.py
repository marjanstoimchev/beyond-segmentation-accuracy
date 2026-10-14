"""Instance-level scoring for nucleus segmentation (this is an instance problem: score objects, not pixels).

Per patch, predicted and true instances are matched one-to-one by IoU (Hungarian assignment), then counted at
each IoU threshold. Two rules make scores honest on this dataset:

* only **owned** true nuclei (centroid in the patch's stride core) and predictions whose centroid lies in that
  core are counted, so nuclei in the 64 px overlap between patches are counted exactly once;
* true nuclei flagged QC-``Wrong`` are **ignored**: they are never false negatives, and a prediction that mostly
  covers one is never a false positive (the biologist said the truth there is unreliable).

Aggregate metrics: precision / recall / F1, AP = TP / (TP + FP + FN) (the StarDist convention), PQ = SQ x RQ,
mean matched IoU, count ratio, plus pixel Dice of the foreground as a sanity number.
"""
from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

IOU_THRESHOLDS = (0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9)


def relabel(labels: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Compact ids to ``1..K``; returns ``(relabeled, original id of each new id (index 0 unused))``."""
    ids = np.unique(labels)
    ids = ids[ids > 0]
    lut = np.zeros(int(labels.max()) + 1 if labels.size else 1, dtype=np.int32)
    lut[ids] = np.arange(1, len(ids) + 1)
    return lut[labels], np.concatenate([[0], ids])


def iou_matrix(gt: np.ndarray, pred: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``(iou (G, P), gt areas (G,), pred areas (P,))`` for label maps with ids ``1..G`` / ``1..P``."""
    G, P = int(gt.max()), int(pred.max())
    inter = np.bincount((gt.astype(np.int64) * (P + 1) + pred).ravel(), minlength=(G + 1) * (P + 1)).reshape(G + 1, P + 1)
    area_g, area_p = inter.sum(1), inter.sum(0)
    union = area_g[:, None] + area_p[None, :] - inter
    return inter[1:, 1:] / np.maximum(union[1:, 1:], 1), area_g[1:], area_p[1:]


def centroids(labels: np.ndarray) -> np.ndarray:
    """``(K, 2)`` (x, y) centroids of ids ``1..K``."""
    K = int(labels.max())
    if K == 0:
        return np.zeros((0, 2))
    ys, xs = np.nonzero(labels)
    ids = labels[ys, xs]
    n = np.bincount(ids, minlength=K + 1)[1:]
    return np.stack([np.bincount(ids, weights=xs, minlength=K + 1)[1:] / n, np.bincount(ids, weights=ys, minlength=K + 1)[1:] / n], 1)


def match_at(iou: np.ndarray, thr: float) -> List[Tuple[int, int]]:
    """One-to-one Hungarian matching of rows (GT) to columns (pred) with IoU >= ``thr``; 0-based index pairs."""
    if iou.size == 0:
        return []
    rows, cols = linear_sum_assignment(np.where(iou >= thr, -iou, 0.0))
    return [(int(r), int(c)) for r, c in zip(rows, cols) if iou[r, c] >= thr]


def instance_outcomes(gt: np.ndarray, pred: np.ndarray, *, ignore: Optional[np.ndarray] = None, gt_scored: Optional[np.ndarray] = None,
                      core: Optional[Tuple[int, int, int, int]] = None, thr: float = 0.5) -> dict:
    """Per-instance verdicts at one threshold, for error maps: ``gt_status[id]`` in {tp, fn, wrong, unscored}, ``pred_status[id]``
    in {tp, fp, unscored}, over the relabeled maps returned as ``gt`` / ``pred`` (ids 1..K). ``wrong`` = a QC-Wrong nucleus
    (most of its pixels in ``ignore``): kept for display, never counted."""
    gt_r, gt_ids = relabel(gt.astype(np.int32))
    pred_r, _ = relabel(pred.astype(np.int32))
    G, P = int(gt_r.max()), int(pred_r.max())
    iou, area_g, area_p = iou_matrix(gt_r, pred_r) if (G and P) else (np.zeros((G, P)), np.zeros(G), np.zeros(P))
    g_ok = np.ones(G, bool) if gt_scored is None else gt_scored[gt_ids[1:]].astype(bool)
    p_ok = np.ones(P, bool)
    if core is not None and P:
        c = centroids(pred_r)
        p_ok &= (c[:, 0] >= core[0]) & (c[:, 0] < core[2]) & (c[:, 1] >= core[1]) & (c[:, 1] < core[3])
    if ignore is not None and ignore.any() and P:
        p_ok &= np.bincount(pred_r[ignore & (pred_r > 0)], minlength=P + 1)[1:] / np.maximum(area_p, 1) < 0.5
    pairs = match_at(iou, thr)
    gt_status = np.array(["unscored"] * (G + 1), dtype=object)
    pred_status = np.array(["unscored"] * (P + 1), dtype=object)
    g_wrong = np.zeros(G, bool)
    if ignore is not None and ignore.any() and G:
        g_wrong = np.bincount(gt_r[ignore & (gt_r > 0)], minlength=G + 1)[1:] / np.maximum(area_g, 1) >= 0.5
    for g in range(G):
        gt_status[g + 1] = "wrong" if g_wrong[g] else ("fn" if g_ok[g] else "unscored")
    for p in range(P):
        pred_status[p + 1] = "fp" if p_ok[p] else "unscored"
    for g, p in pairs:
        if g_wrong[g]:
            pred_status[p + 1] = "unscored"                       # matched a Wrong nucleus: neither right nor wrong
        elif g_ok[g] or p_ok[p]:
            gt_status[g + 1], pred_status[p + 1] = "tp", "tp"
        else:
            gt_status[g + 1], pred_status[p + 1] = "unscored", "unscored"
    return {"gt": gt_r, "pred": pred_r, "gt_status": gt_status, "pred_status": pred_status, "pairs": pairs, "iou": iou,
            "tp": int((pred_status == "tp").sum()), "fp": int((pred_status == "fp").sum()), "fn": int((gt_status == "fn").sum())}


def score_patch(gt: np.ndarray, pred: np.ndarray, *, ignore: Optional[np.ndarray] = None, gt_scored: Optional[np.ndarray] = None,
                core: Optional[Tuple[int, int, int, int]] = None, thresholds: Sequence[float] = IOU_THRESHOLDS) -> dict:
    """Match one patch. ``gt_scored[id]`` says which true ids count (owned, not Wrong); ``core`` (x0, y0, x1, y1, patch-local)
    selects the predictions that count; ``ignore`` is the pixel mask of Wrong nuclei. A nucleus lying mostly (>= 50 %) in
    ``ignore`` is never scored — not a hit when matched, not a miss when not — and a prediction matched to it is neither a
    hit nor a false positive, with or without ``gt_scored`` (whole-region scoring passes none)."""
    gt, gt_ids = relabel(gt.astype(np.int32))
    pred, _ = relabel(pred.astype(np.int32))
    G, P = int(gt.max()), int(pred.max())
    iou, area_g, area_p = iou_matrix(gt, pred) if (G and P) else (np.zeros((G, P)), np.zeros(G), np.zeros(P))
    g_ok = np.ones(G, bool) if gt_scored is None else gt_scored[gt_ids[1:]].astype(bool)
    g_wrong = np.zeros(G, bool)
    if ignore is not None and ignore.any() and G:
        g_wrong = np.bincount(gt[ignore & (gt > 0)], minlength=G + 1)[1:] / np.maximum(area_g, 1) >= 0.5
        g_ok &= ~g_wrong                                         # a QC-Wrong nucleus is never scored
    p_ok = np.ones(P, bool)
    if core is not None and P:
        c = centroids(pred)
        p_ok &= (c[:, 0] >= core[0]) & (c[:, 0] < core[2]) & (c[:, 1] >= core[1]) & (c[:, 1] < core[3])
    if ignore is not None and ignore.any() and P:
        ign_frac = np.bincount(pred[ignore & (pred > 0)], minlength=P + 1)[1:] / np.maximum(area_p, 1)
        p_ok &= ign_frac < 0.5                                   # a prediction sitting on a Wrong nucleus is not a false positive
    keep = ~ignore if ignore is not None else np.ones(gt.shape, dtype=bool)
    fg_gt, fg_pr = (gt > 0) & keep, (pred > 0) & keep
    out = {"n_gt": int(g_ok.sum()), "n_pred": int(p_ok.sum()),
           "dice": float(2 * (fg_gt & fg_pr).sum() / max((fg_gt.sum() + fg_pr.sum()), 1))}
    for thr in thresholds:
        tp = fp = fn = 0
        iou_sum = 0.0
        pairs = match_at(iou, thr) if (G and P) else []
        matched_g, matched_p = {g for g, _ in pairs}, {p for _, p in pairs}
        for g, p in pairs:
            if g_wrong[g]:                                       # matched a Wrong nucleus: neither right nor wrong
                continue
            if g_ok[g] or p_ok[p]:
                tp += 1
                iou_sum += float(iou[g, p])
        fn = int(sum(g_ok[g] for g in range(G) if g not in matched_g))
        fp = int(sum(p_ok[p] for p in range(P) if p not in matched_p))
        out[f"tp@{thr:g}"], out[f"fp@{thr:g}"], out[f"fn@{thr:g}"], out[f"iou_sum@{thr:g}"] = tp, fp, fn, iou_sum
    return out


def aggregate(rows: Iterable[dict], thresholds: Sequence[float] = IOU_THRESHOLDS) -> Dict[str, float]:
    """Sum the per-patch counts and derive precision / recall / F1 / AP / PQ per threshold, plus AP averaged over thresholds."""
    df = pd.DataFrame(list(rows))
    if df.empty:
        return {}
    out: Dict[str, float] = {"n_patches": int(len(df)), "n_gt": int(df.n_gt.sum()), "n_pred": int(df.n_pred.sum()),
                             "count_ratio": float(df.n_pred.sum() / max(df.n_gt.sum(), 1)), "dice": float(df.dice.mean())}
    aps = []
    for thr in thresholds:
        tp, fp, fn, s = (float(df[f"{k}@{thr:g}"].sum()) for k in ("tp", "fp", "fn", "iou_sum"))
        p, r = tp / max(tp + fp, 1), tp / max(tp + fn, 1)
        out[f"precision@{thr:g}"], out[f"recall@{thr:g}"] = p, r
        out[f"f1@{thr:g}"] = 2 * p * r / max(p + r, 1e-9)
        out[f"ap@{thr:g}"] = tp / max(tp + fp + fn, 1)
        out[f"pq@{thr:g}"] = (s / max(tp, 1)) * (tp / max(tp + 0.5 * fp + 0.5 * fn, 1))
        out[f"mean_iou@{thr:g}"] = s / max(tp, 1)
        aps.append(out[f"ap@{thr:g}"])
    out["ap@0.5:0.9"] = float(np.mean(aps))
    return out


SUMMARY_KEYS: List[str] = ["f1@0.5", "ap@0.5", "ap@0.5:0.9", "pq@0.5", "precision@0.5", "recall@0.5", "mean_iou@0.5", "count_ratio", "dice"]
