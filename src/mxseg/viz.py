"""Visualizing a segmentation *against the ground truth* (for notebooks).

Nuclei are one colour — this is foreground vs background, instances separated by thin edges — and errors are
painted from the actual instance matching: **TP green, FP red, FN violet**; **QC-Wrong nuclei amber** (kept for
display, never trained on or counted); objects that are merely not scored in this patch grey. A pixel-level error
map (FP / FN pixels) shows where boundaries disagree.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence

import numpy as np

from mxdata.viz import gray_to_rgb, label_contours, to_uint8

from .metrics import instance_outcomes

COLORS = {"gt": (27, 175, 122), "pred": (42, 120, 214), "tp": (27, 175, 122), "fp": (227, 73, 72), "fn": (144, 100, 255),
          "wrong": (237, 161, 0), "ignore": (237, 161, 0), "unscored": (150, 150, 150)}


def _blend(rgb: np.ndarray, mask: np.ndarray, color, alpha: float) -> None:
    rgb[mask] = (rgb[mask] * (1 - alpha) + np.asarray(color, np.float32) * alpha)


def overlay_fg(rgb: np.ndarray, labels: np.ndarray, color=COLORS["gt"], alpha: float = 0.4, ignore: Optional[np.ndarray] = None,
               edges: bool = True) -> np.ndarray:
    """All nuclei in one colour (foreground vs background), QC-Wrong (``ignore``) pixels amber, thin edges between instances."""
    out = rgb.astype(np.float32)
    fg = labels > 0
    if ignore is not None and ignore.any():
        _blend(out, fg & ~ignore, color, alpha)
        _blend(out, ignore, COLORS["ignore"], 0.6)
    else:
        _blend(out, fg, color, alpha)
    if edges:
        out[label_contours(labels)] = np.asarray(color, np.float32) * 0.55
    return np.clip(out, 0, 255).astype(np.uint8)


def overlay_gt(rgb: np.ndarray, labels: np.ndarray, scored: np.ndarray, wrong_ids: Optional[np.ndarray] = None) -> np.ndarray:
    """Ground truth: scored nuclei filled green with crisp instance boundaries; annotated-but-not-scored nuclei outline only
    (owned by a neighbouring patch); QC-Wrong nuclei amber."""
    out = rgb.astype(np.float32)
    own = np.isin(labels, np.nonzero(scored)[0]) & (labels > 0)
    _blend(out, own, COLORS["gt"], 0.42)
    wrong = np.isin(labels, wrong_ids[wrong_ids > 0]) if wrong_ids is not None and len(wrong_ids) else np.zeros(labels.shape, bool)
    _blend(out, wrong, COLORS["wrong"], 0.55)
    edges = label_contours(labels)
    out[edges & own] = np.asarray(COLORS["gt"], np.float32) * 0.5
    out[edges & wrong] = np.asarray(COLORS["wrong"], np.float32) * 0.6
    out[edges & ~own & ~wrong] = 230.0                             # not scored here: white outline only
    return np.clip(out, 0, 255).astype(np.uint8)


def qc_panel(rgb: np.ndarray, labels: np.ndarray, ignore: Optional[np.ndarray], has_qc: bool) -> np.ndarray:
    """The biologist's review: nuclei confirmed OK (green) and flagged Wrong (amber, ignored everywhere). Regions without a
    QC table show outlines only — every contour counts as correct there."""
    out = rgb.astype(np.float32)
    fg = labels > 0
    if has_qc:
        ign = ignore if ignore is not None else np.zeros(labels.shape, bool)
        _blend(out, fg & ~ign, COLORS["gt"], 0.3)
        _blend(out, ign, COLORS["wrong"], 0.65)
    out[label_contours(labels)] = np.asarray(COLORS["gt"], np.float32) * 0.55
    return np.clip(out, 0, 255).astype(np.uint8)


def dim_outside(rgb: np.ndarray, core, factor: float = 0.35) -> np.ndarray:
    """Darken everything outside the owned core (local ``(x0, y0, x1, y1)``): what is scored here stays bright."""
    out = rgb.astype(np.float32) * factor
    x0, y0, x1, y1 = core
    out[y0:y1, x0:x1] = rgb[y0:y1, x0:x1]
    return np.clip(out, 0, 255).astype(np.uint8)


def draw_scale_bar(ax, px_um: Optional[float], um: float = 20.0) -> None:
    """A white scale bar (bottom right) when the pixel size is known."""
    if not px_um:
        return
    import matplotlib.patheffects as pe
    x1, y1 = ax.get_xlim()[1], max(ax.get_ylim())
    L = um / px_um
    ax.plot([x1 - L - 12, x1 - 12], [y1 - 14, y1 - 14], color="white", lw=3, solid_capstyle="butt")
    ax.text(x1 - 12 - L / 2, y1 - 24, f"{um:g} µm", color="white", ha="center", va="bottom", fontsize=8,
            path_effects=[pe.withStroke(linewidth=2, foreground="black")])


def draw_core(ax, core) -> None:
    """Outline the owned core: nuclei whose centroid lies inside are scored in this patch."""
    import matplotlib.patches as mpatches
    x0, y0, x1, y1 = core
    ax.add_patch(mpatches.Rectangle((x0 - 0.5, y0 - 0.5), x1 - x0, y1 - y0, fill=False, ec="white", ls="--", lw=1.0, alpha=0.9))


def paint_status(rgb: np.ndarray, labels: np.ndarray, status: np.ndarray, alpha: float = 0.5, edges: bool = True) -> np.ndarray:
    """Colour every instance by its verdict (``status[id]`` in tp / fp / fn / unscored)."""
    out = rgb.astype(np.float32)
    for verdict in ("unscored", "wrong", "tp", "fn", "fp"):
        ids = np.nonzero(status == verdict)[0]
        ids = ids[ids > 0]
        if ids.size:
            _blend(out, np.isin(labels, ids), COLORS[verdict], alpha if verdict != "unscored" else 0.35)
    if edges:
        out[label_contours(labels)] = out[label_contours(labels)] * 0.5
    return np.clip(out, 0, 255).astype(np.uint8)


def pixel_errors(rgb: np.ndarray, gt: np.ndarray, pred: np.ndarray, ignore: Optional[np.ndarray] = None, alpha: float = 0.55) -> np.ndarray:
    """Foreground pixels: agreement green, predicted-only (FP) red, missed (FN) violet; QC-Wrong pixels left untouched."""
    out = rgb.astype(np.float32)
    keep = ~ignore if ignore is not None else np.ones(gt.shape, bool)
    g, p = (gt > 0) & keep, (pred > 0) & keep
    _blend(out, g & p, COLORS["tp"], alpha)
    _blend(out, p & ~g, COLORS["fp"], alpha)
    _blend(out, g & ~p, COLORS["fn"], alpha)
    return np.clip(out, 0, 255).astype(np.uint8)


def error_map(rgb: np.ndarray, o: dict, core, alpha: float = 0.5) -> np.ndarray:
    """Verdicts only: TP (green, true shape), FN (violet, true shape), FP (red, predicted shape). Every predicted object is
    outlined; an outline without fill is a prediction that is not scored here (on a QC-Wrong nucleus, or outside the owned
    core, which is dimmed)."""
    out = dim_outside(rgb, core).astype(np.float32)
    for verdict, lab, status in (("tp", o["gt"], o["gt_status"]), ("fn", o["gt"], o["gt_status"]), ("fp", o["pred"], o["pred_status"])):
        ids = np.nonzero(status == verdict)[0]
        ids = ids[ids > 0]
        if ids.size:
            _blend(out, np.isin(lab, ids), COLORS[verdict], alpha)
    out[label_contours(o["pred"])] = 225.0
    return np.clip(out, 0, 255).astype(np.uint8)


def _error_panel(rgb: np.ndarray, labels: np.ndarray, pred: np.ndarray, ignore: np.ndarray, sc: np.ndarray, core, thr: float):
    o = instance_outcomes(labels, pred, ignore=ignore, gt_scored=sc, core=core, thr=thr)
    o["f1"] = 2 * o["tp"] / max(2 * o["tp"] + o["fp"] + o["fn"], 1)
    o["n_scored"] = int(np.isin(o["gt_status"], ["tp", "fn"]).sum())
    return error_map(rgb, o, core), o


def _core(rs):
    return tuple(v - o for v, o in zip(rs.meta["core"], (rs.box.x0, rs.box.y0, rs.box.x0, rs.box.y0)))


def _n_scored(labels: np.ndarray, scored: np.ndarray) -> int:
    ids = np.unique(labels)
    ids = ids[(ids > 0) & (ids < len(scored))]
    return int(scored[ids].sum())


def _gt_title(labels: np.ndarray, scored: np.ndarray) -> str:
    n_all, n_sc = int(len(np.unique(labels)) - 1), _n_scored(labels, scored)
    return f"ground truth: {n_sc} scored here (filled) · {n_all - n_sc} not scored (faint)"


def _legend(fig, n_rows: int, verdicts: bool = True, core: bool = True):
    import matplotlib.lines as mlines
    import matplotlib.patches as mpatches
    if verdicts:
        handles = [mpatches.Patch(color=np.array(COLORS[k]) / 255, label=lab) for k, lab in
                   (("tp", "true positive"), ("fp", "false positive"), ("fn", "false negative"))]
        handles += [mlines.Line2D([], [], color="0.88", lw=1.5, label="outline only: predicted, not scored" + (" here" if core else " (on a QC-Wrong nucleus)"))]
        if core:
            handles += [mlines.Line2D([], [], color="white", ls="--", lw=1, label="dashed box: owned core (outside is dimmed, scored by a neighbour)")]
    else:
        handles = [mpatches.Patch(color=np.array(COLORS["gt"]) / 255, label="annotated nucleus"),
                   mpatches.Patch(color=np.array(COLORS["wrong"]) / 255, label="QC-Wrong (ignored in training and scoring)")]
        if core:
            handles += [mlines.Line2D([], [], color="white", ls="--", lw=1, label="dashed box: owned core — centroids inside are scored here")]
    top = 1 - 0.55 / fig.get_size_inches()[1]                       # a fixed 0.55-inch band for the legend, whatever the height
    fig.tight_layout(rect=(0, 0, 1, top))
    fig.legend(handles=handles, loc="upper center", ncol=len(handles), frameon=False, fontsize=10, bbox_to_anchor=(0.5, 1.0))


def _row_title(rs, tag: str) -> str:
    return f"{rs.key}{tag}\nR{rs.meta['region_id']} · QC table: {rs.meta['has_qc']}"


def show_ground_truth(samples: Sequence, *, scored: Sequence[np.ndarray], labels_ignore: Sequence, tags: Optional[Sequence[str]] = None):
    """How a patch is scored, without any model: DAPI with the owned core | ground truth (scored filled, rest faint) |
    the QC review (OK / Wrong) or "no QC table". Returns the figure."""
    import matplotlib.pyplot as plt

    n = len(samples)
    fig, axes = plt.subplots(n, 3, figsize=(14.4, 4.8 * n), squeeze=False)
    for i, (row, rs, (labels, ignore), sc) in enumerate(zip(axes, samples, labels_ignore, scored)):
        rgb = gray_to_rgb(to_uint8(rs.image[0]))
        core = _core(rs)
        tag = f" [{tags[i]}]" if tags else ""
        n_wrong = int(len(np.unique(labels[ignore])) - int((labels[ignore] == 0).any())) if ignore is not None and ignore.any() else 0
        qc_title = (f"QC review: {int(len(np.unique(labels)) - 1 - n_wrong)} OK · {n_wrong} Wrong (amber, ignored)" if rs.meta["has_qc"]
                    else "no QC table for this region: every contour counts as correct")
        panels = [(rgb, _row_title(rs, tag)), (overlay_gt(rgb, labels, sc), _gt_title(labels, sc)),
                  (qc_panel(rgb, labels, ignore, rs.meta["has_qc"]), qc_title)]
        for ax, (img, ttl) in zip(row, panels):
            ax.imshow(img)
            ax.set_title(ttl, fontsize=9)
            ax.set_axis_off()
            draw_core(ax, core)
    _legend(fig, n, verdicts=False)
    return fig


def show_runs(samples: Sequence, preds: "dict", *, scored: Sequence[np.ndarray], labels_ignore: Sequence, thr: float = 0.5,
              tags: Optional[Sequence[str]] = None):
    """Several models on the same patches, one row per patch: DAPI | ground truth | one error map per model (dict order).
    Error maps show verdicts only (TP / FP / FN); predictions that are not scored here are outlined, the outside of the
    owned core is dimmed. Titles carry objects, TP / FP / FN and F1. Returns the figure."""
    import matplotlib.pyplot as plt

    names = list(preds)
    n, m = len(samples), len(names)
    fig, axes = plt.subplots(n, 2 + m, figsize=(4.8 * (2 + m), 4.8 * n), squeeze=False)
    for i, (row, rs, (labels, ignore), sc) in enumerate(zip(axes, samples, labels_ignore, scored)):
        rgb = gray_to_rgb(to_uint8(rs.image[0]))
        core = _core(rs)
        tag = f" [{tags[i]}]" if tags else ""
        panels = [(rgb, _row_title(rs, tag)), (overlay_gt(rgb, labels, sc), _gt_title(labels, sc))]
        for name in names:
            pred = preds[name][i]
            img, o = _error_panel(rgb, labels, pred, ignore, sc, core, thr)
            panels.append((img, f"{name}\n{int(pred.max())} objects · TP {o['tp']}  FP {o['fp']}  FN {o['fn']}  ·  F1 {o['f1']:.2f}"))
        for ax, (img, ttl) in zip(row, panels):
            ax.imshow(img)
            ax.set_title(ttl, fontsize=9)
            ax.set_axis_off()
            draw_core(ax, core)
    _legend(fig, n)
    return fig


def show_before_after(samples: Sequence, preds_before: Sequence[np.ndarray], preds_after: Sequence[np.ndarray], *,
                      scored: Sequence[np.ndarray], labels_ignore: Sequence, names: Sequence[str] = ("before", "after"),
                      thr: float = 0.5, tags: Optional[Sequence[str]] = None):
    """Two models side by side: DAPI | ground truth | BEFORE errors | AFTER errors (see :func:`show_runs`)."""
    return show_runs(samples, {f"BEFORE — {names[0]}": preds_before, f"AFTER — {names[1]}": preds_after},
                     scored=scored, labels_ignore=labels_ignore, thr=thr, tags=tags)


def show_predictions(samples: Sequence, preds: Sequence[np.ndarray], *, scored: Sequence[np.ndarray], labels_ignore: Sequence,
                     title: str = "prediction", thr: float = 0.5, tags: Optional[Sequence[str]] = None):
    """One model in detail, one row per patch: DAPI | ground truth | all predicted objects | error map | pixel errors.
    Returns the figure."""
    import matplotlib.pyplot as plt

    n = len(samples)
    fig, axes = plt.subplots(n, 5, figsize=(24, 4.8 * n), squeeze=False)
    for i, (row, rs, pred, (labels, ignore), sc) in enumerate(zip(axes, samples, preds, labels_ignore, scored)):
        rgb = gray_to_rgb(to_uint8(rs.image[0]))
        core = _core(rs)
        dim = dim_outside(rgb, core)
        err, o = _error_panel(rgb, labels, pred, ignore, sc, core, thr)
        tag = f" [{tags[i]}]" if tags else ""
        panels = [(rgb, _row_title(rs, tag)), (overlay_gt(rgb, labels, sc), _gt_title(labels, sc)),
                  (overlay_fg(dim, pred, COLORS["pred"]), f"{title}: {int(pred.max())} predicted objects"),
                  (err, f"verdicts @IoU {thr}: TP {o['tp']}  FP {o['fp']}  FN {o['fn']}  ·  F1 {o['f1']:.2f}"),
                  (pixel_errors(dim, labels, pred, ignore), "pixels: agreement / predicted only / missed")]
        for ax, (img, ttl) in zip(row, panels):
            ax.imshow(img)
            ax.set_title(ttl, fontsize=9)
            ax.set_axis_off()
            draw_core(ax, core)
    _legend(fig, n)
    return fig


def show_region_crops(image: np.ndarray, labels: np.ndarray, ignore: Optional[np.ndarray], preds: "dict", crops: Sequence,
                      *, size: Optional[int] = 512, thr: float = 0.5, title: str = "", px_um: Optional[float] = None):
    """Whole-region scoring shown on crops: DAPI | ground truth | one verdict map per model. The verdicts come from matching the
    region's complete maps (one ground-truth map, one prediction map per model); the crops only choose where to look, so
    nothing here is "owned by a neighbour". Returns the figure."""
    import matplotlib.pyplot as plt

    outs = {n: instance_outcomes(labels, p, ignore=ignore, thr=thr) for n, p in preds.items()}
    wrong_ids = np.unique(labels[ignore]) if ignore is not None and ignore.any() else np.zeros(0, int)
    scored = np.ones(int(labels.max()) + 1, bool)
    scored[0] = False
    scored[wrong_ids] = False
    n, m = len(crops), len(preds)
    H, W = labels.shape
    h, w = (H, W) if size is None else (size, size)
    fig, axes = plt.subplots(n, 2 + m, figsize=((2 + m) * 4.8 * (w / h if size is None else 1), 4.8 * n), squeeze=False, dpi=140)
    for row, (y, x) in zip(axes, crops):
        sl = (slice(y, y + h), slice(x, x + w))
        rgb = gray_to_rgb(to_uint8(image[0][sl]))
        lab = labels[sl]
        ids = np.unique(lab)
        ids = ids[ids > 0]
        panels = [(rgb, title if size is None else f"{title}\ncrop x {x}–{x + w}, y {y}–{y + h}"),
                  (overlay_gt(rgb, lab, scored, wrong_ids), f"ground truth: {int(scored[ids].sum())} nuclei · {int((~scored[ids]).sum())} QC-Wrong (amber)")]
        for name, o in outs.items():
            g, p = o["gt"][sl], o["pred"][sl]
            gi, pi = np.unique(g), np.unique(p)
            gi, pi = gi[gi > 0], pi[pi > 0]
            tp, fn, fp = int((o["gt_status"][gi] == "tp").sum()), int((o["gt_status"][gi] == "fn").sum()), int((o["pred_status"][pi] == "fp").sum())
            f1 = 2 * tp / max(2 * tp + fp + fn, 1)
            img = error_map(rgb, {"gt": g, "pred": p, "gt_status": o["gt_status"], "pred_status": o["pred_status"]}, (0, 0, h, w))
            panels.append((img, f"{name}\nTP {tp}  FP {fp}  FN {fn}  ·  F1 {f1:.2f}"))
        for ax, (img, ttl) in zip(row, panels):
            ax.imshow(img)
            ax.set_title(ttl, fontsize=9)
            ax.set_axis_off()
        draw_scale_bar(row[0], px_um)
    _legend(fig, n, core=False)
    return fig


def show_region_ground_truth(image: np.ndarray, labels: np.ndarray, ignore: Optional[np.ndarray], has_qc: bool, crops: Sequence,
                             *, size: Optional[int] = 512, title: str = "", px_um: Optional[float] = None):
    """The annotation at region level: DAPI | annotated nuclei (QC-Wrong faint) | the QC review. No ownership — a region is
    one ground-truth map and every nucleus counts exactly once. Returns the figure."""
    import matplotlib.pyplot as plt

    wrong_ids = np.unique(labels[ignore]) if ignore is not None and ignore.any() else np.zeros(0, int)
    scored = np.ones(int(labels.max()) + 1, bool)
    scored[0] = False
    scored[wrong_ids] = False
    n = len(crops)
    H, W = labels.shape
    h, w = (H, W) if size is None else (size, size)
    fig, axes = plt.subplots(n, 3, figsize=(3 * 4.8 * w / h if size is None else 14.4, 4.8 * n), squeeze=False, dpi=140)
    for row, (y, x) in zip(axes, crops):
        sl = (slice(y, y + h), slice(x, x + w))
        rgb = gray_to_rgb(to_uint8(image[0][sl]))
        lab, ign = labels[sl], (ignore[sl] if ignore is not None else None)
        ids = np.unique(lab)
        ids = ids[ids > 0]
        n_wrong = int((~scored[ids]).sum())
        qc_title = (f"QC review: {len(ids) - n_wrong} OK · {n_wrong} Wrong (amber, ignored)" if has_qc
                    else "no QC table for this region: every contour counts as correct")
        panels = [(rgb, title if size is None else f"{title}\ncrop x {x}–{x + w}, y {y}–{y + h}"),
                  (overlay_gt(rgb, lab, scored, wrong_ids), f"ground truth: {len(ids) - n_wrong} nuclei · {n_wrong} QC-Wrong (amber)"),
                  (qc_panel(rgb, lab, ign, has_qc), qc_title)]
        for ax, (img, ttl) in zip(row, panels):
            ax.imshow(img)
            ax.set_title(ttl, fontsize=9)
            ax.set_axis_off()
        draw_scale_bar(row[0], px_um)
    _legend(fig, n, verdicts=False, core=False)
    return fig


def save_panel(path, rgb: np.ndarray) -> None:
    """One clean panel: the raw RGB array at native resolution — no titles, legends or axes (LaTeX-ready)."""
    import cv2
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), rgb[..., ::-1]):
        raise IOError(f"could not write {path}")


def export_region_panels(out_dir, image: np.ndarray, labels: np.ndarray, ignore: Optional[np.ndarray], has_qc: bool,
                         preds: "dict", *, thr: float = 0.5, px_um: Optional[float] = None, crop=None, suffix: str = "") -> dict:
    """Publication panels for one region, each saved individually and unannotated:

    ``dapi``, ``ground_truth``, ``qc_review``, and one verdict map per model —
    plus ``panels{suffix}.json`` with everything a caption needs (pixel size, bar length, per-model TP / FP / FN / F1).
    ``crop = (y, x, size)`` exports that window; verdicts always come from matching the complete region maps."""
    import json

    out_dir = Path(out_dir)
    sl = (slice(None), slice(None)) if crop is None else (slice(crop[0], crop[0] + crop[2]), slice(crop[1], crop[1] + crop[2]))
    rgb = gray_to_rgb(to_uint8(image[0][sl]))
    lab = labels[sl]
    wrong_ids = np.unique(labels[ignore]) if ignore is not None and ignore.any() else np.zeros(0, int)
    scored = np.ones(int(labels.max()) + 1, bool)
    scored[0] = False
    scored[wrong_ids] = False
    import json as _json
    meta = {"pixel_size_um": px_um, "scale_bar_um": 20.0 if px_um else None, "crop": crop, "thr": thr,
            "n_nuclei": int(len(np.unique(lab)) - 1), "n_wrong": int((~scored[np.unique(lab)[1:]]).sum()) if lab.any() else 0, "models": {}}
    sidecar = out_dir / f"panels{suffix}.json"
    if sidecar.exists():                                     # evals run method by method: accumulate their caption numbers
        meta["models"] = _json.loads(sidecar.read_text()).get("models", {})
    save_panel(out_dir / f"dapi{suffix}.png", rgb)
    save_panel(out_dir / f"ground_truth{suffix}.png", overlay_gt(rgb, lab, scored, wrong_ids))
    save_panel(out_dir / f"qc_review{suffix}.png", qc_panel(rgb, lab, ignore[sl] if ignore is not None else None, has_qc))
    for name, pred in preds.items():
        o = instance_outcomes(labels, pred, ignore=ignore, thr=thr)
        g, pr = o["gt"][sl], o["pred"][sl]
        img = error_map(rgb, {"gt": g, "pred": pr, "gt_status": o["gt_status"], "pred_status": o["pred_status"]},
                        (0, 0, lab.shape[0], lab.shape[1]))
        save_panel(out_dir / f"{name}{suffix}.png", img)
        gi, pi = np.unique(g), np.unique(pr)
        gi, pi = gi[gi > 0], pi[pi > 0]
        tp, fn = int((o["gt_status"][gi] == "tp").sum()), int((o["gt_status"][gi] == "fn").sum())
        fp = int((o["pred_status"][pi] == "fp").sum())
        meta["models"][name] = {"tp": tp, "fp": fp, "fn": fn, "f1": round(2 * tp / max(2 * tp + fp + fn, 1), 4)}
    sidecar.write_text(json.dumps(meta, indent=2))
    return meta
