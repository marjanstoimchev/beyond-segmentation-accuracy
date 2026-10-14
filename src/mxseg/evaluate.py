"""Run a baseline over a split of the segmentation dataset and store its **metrics** (never images).

    results/<split>/<name>/
      run.json          baseline + params, dataset, split, patches used, QC accounting, versions, device, timing
      metrics.json      aggregate metrics per IoU threshold — what `mxseg compare` reads
      per_patch.csv     per-patch counts (tp/fp/fn per threshold), dice, seconds

Ground truth is the biologist-verified annotation: nuclei flagged QC-``Wrong`` are ignored (never a miss, and a
prediction on them is never a false positive); ``require_qc=True`` further restricts to patches from regions that
have a QC table, so every counted nucleus was actually reviewed. Only *owned* nuclei count (patches overlap).
Predictions are not stored — the notebook renders them on demand.
"""
from __future__ import annotations

import dataclasses
import json
import logging
import platform
import time
from pathlib import Path
from typing import List, Optional, Sequence

import numpy as np
import pandas as pd
from tqdm import tqdm

from mxdata import DEFAULT_DATASET, DataConfig, apply_qc_policy, ingest, load_regions
from mxdata.data import LiveSource, PatchSource
from mxdata.pipeline import read_dataset_json

from ._version import __version__
from .baselines import Baseline
from .inputs import SlideReader, input_meta
from .metrics import SUMMARY_KEYS, aggregate, score_patch
from .tiling import predict_tiled

logger = logging.getLogger(__name__)
RESULTS_ROOT = Path("results")
OUTPUTS_ROOT = Path("outputs")


def _predictions_dir(unit: str, split: str, name: str, outputs_dir=None) -> Path:
    return (Path(outputs_dir) if outputs_dir else OUTPUTS_ROOT) / unit / split / name


def save_prediction(pred_dir: Path, key: str, pred: np.ndarray) -> None:
    """One predicted instance map as a compressed ``.npz`` — data, not an image (the visual panels are the PNGs)."""
    pred_dir.mkdir(parents=True, exist_ok=True)
    tmp = pred_dir / f"{key}.tmp.npz"
    np.savez_compressed(tmp, pred=pred.astype(np.int32))
    tmp.rename(pred_dir / f"{key}.npz")


def stored_predictions(unit: str, split: str, name: str, outputs_dir=None) -> dict:
    """``{key: path}`` of the predictions an eval saved (empty dict if none)."""
    d = _predictions_dir(unit, split, name, outputs_dir)
    return {p.name.removesuffix(".npz"): p for p in sorted(d.glob("*.npz"))} if d.is_dir() else {}


def read_prediction(path) -> np.ndarray:
    return np.load(path)["pred"].astype(np.int32)


def owned_luts(root: Path, keys: Sequence[str]) -> dict:
    """``{patch key: bool LUT by local id}`` — which nuclei each patch owns (from instances.csv)."""
    inst = pd.read_csv(Path(root) / "instances.csv")
    inst = inst[inst.patch_key.isin(keys)]
    out = {}
    for pk, g in inst.groupby("patch_key", sort=False):
        lut = np.zeros(int(g.local_id.max()) + 1, dtype=bool)
        lut[g.local_id.to_numpy()] = g.owned.to_numpy()
        out[pk] = lut
    return out


def verified_gt(sample, owned: dict):
    """The ground truth a patch is scored against: ``(labels, ignore, scored, core)`` — all nuclei painted (TINY removed),
    ``ignore`` = QC-Wrong pixels, ``scored[id]`` = owned and not Wrong, ``core`` = the owned core in patch coordinates."""
    labels, ignore = apply_qc_policy(sample.labels, sample.flags, "ignore_wrong")
    scored = owned.get(sample.key, np.zeros(1, bool)).copy()
    wrong = np.unique(labels[ignore])
    scored[wrong[wrong < len(scored)]] = False
    core = tuple(v - o for v, o in zip(sample.meta["core"], (sample.box.x0, sample.box.y0, sample.box.x0, sample.box.y0)))
    return labels, ignore, scored, core


def _versions() -> dict:
    v = {"mxseg": __version__, "python": platform.python_version(), "platform": platform.platform()}
    for mod in ("mxdata", "torch", "cellpose", "instanseg"):
        try:
            m = __import__(mod)
            v[mod] = str(getattr(m, "__version__", None) or getattr(m, "version", "?"))
        except Exception:
            pass
    return v


def run_baseline(baseline: Baseline, cfg: DataConfig, *, split: str = "val", dataset_dir: "str | Path | None" = None,
                 results_dir: "str | Path | None" = None, name: Optional[str] = None, n: Optional[int] = None,
                 require_qc: bool = False, batch_size: int = 8, progress: bool = True, unit: str = "patch",
                 slide_root=None, store_predictions: bool = True, outputs_dir=None, panels: bool = False) -> Path:
    """``unit="patch"``: the stored patches, owned nuclei scored (fast proxy) -> results/<split>/<name>/.
    ``unit="region"``: whole annotated regions of the split, predicted from tiles of the training geometry and scored as one
    ground-truth map each (the paper number) -> results/region/<split>/<name>/; with ``panels=True`` the LaTeX-ready panels (dapi,
    ground truth, QC review, this method's verdict map + a merged panels.json of caption numbers) land in outputs/figures/<split>/<region>/.
    Every predicted instance map is also saved to outputs/<unit>/<split>/<name>/<key>.png (uint16) so notebooks plot
    without re-running inference (``store_predictions=False`` to skip)."""
    if unit == "region":
        return run_baseline_regions(baseline, cfg, split=split, dataset_dir=dataset_dir, results_dir=results_dir, name=name, n=n,
                                    require_qc=require_qc, batch_size=batch_size, progress=progress, slide_root=slide_root,
                                    store_predictions=store_predictions, outputs_dir=outputs_dir, panels=panels)
    if unit != "patch":
        raise ValueError(f"unit must be 'patch' or 'region', got {unit!r}")
    root = Path(dataset_dir) if dataset_dir else DEFAULT_DATASET
    info = read_dataset_json(root)
    baseline.resolve_input(root, cfg.slides)
    markers = baseline.input_markers()                                 # a model with an input recipe: its markers, and only those
    in_store = bool(markers) and {m.lower() for m in markers} <= {m.lower() for m in info["markers"]}
    src = PatchSource(root, split=split, kind="labeled", slides=cfg.slides, markers=markers if in_store else None)   # opens only its files
    keys = [k for k in src.keys if (src.row(k)["has_qc"] or not require_qc)][: n or None]
    if not keys:
        raise ValueError(f"no labeled {split} patches for {list(cfg.slides)} in {root}" + (" with a QC table" if require_qc else ""))
    owned = owned_luts(root, keys)
    reader = SlideReader(slide_root) if markers and not in_store else None   # a marker the dataset lacks: live from the slide
    name = name or baseline.slug()
    out = (Path(results_dir) if results_dir else RESULTS_ROOT) / split / name
    out.mkdir(parents=True, exist_ok=True)
    n_qc = sum(bool(src.row(k)["has_qc"]) for k in keys)
    logger.info("%s | %s split, %d patches (%d from QC'd regions%s) | regions %s | slides %s",
                baseline.describe(), split, len(keys), n_qc, ", require_qc" if require_qc else "",
                sorted({int(src.row(k)["region_id"]) for k in keys}), list(cfg.slides))

    rows: List[dict] = []
    n_wrong_ignored = 0
    t0 = time.time()
    bar = tqdm(total=len(keys), unit="patch", desc=name, disable=not progress, dynamic_ncols=True)
    for b0 in range(0, len(keys), batch_size):
        samples = [src.load(k) for k in keys[b0:b0 + batch_size]]
        if reader is not None:                                          # the model's markers read live (labels unchanged)
            samples = [dataclasses.replace(s, image=reader.stack(s.meta["slide"], src.box(s.key), markers, src.level)) for s in samples]
        t1 = time.time()
        preds = baseline.predict_batch([s.image for s in samples], samples[0].meta.get("pixel_size_um"), [s.meta["slide"] for s in samples])
        dt = (time.time() - t1) / len(samples)
        for s, pred in zip(samples, preds):
            labels, ignore, scored, core = verified_gt(s, owned)                     # TINY gone; Wrong kept in GT but ignored
            if store_predictions:
                save_prediction(_predictions_dir("patch", split, name, outputs_dir), s.key, pred)
                if panels:                                            # PNG panels only on request (--panels)
                    from .viz import export_region_panels
                    export_region_panels((Path(outputs_dir) if outputs_dir else OUTPUTS_ROOT) / "figures" / "patch" / split / s.key,
                                     s.image, labels, ignore, s.meta["has_qc"], {name: pred}, px_um=s.meta.get("pixel_size_um"))
            own = owned.get(s.key, np.zeros(1, bool))
            wrong = np.unique(labels[ignore])
            n_wrong_ignored += int(own[wrong[wrong < len(own)]].sum())               # owned nuclei that were Wrong
            rows.append({"key": s.key, "region_id": s.meta["region_id"], "has_qc": s.meta["has_qc"], "seconds": dt,
                         **score_patch(labels, pred, ignore=ignore, gt_scored=scored, core=core)})
        bar.update(len(samples))
        bar.set_postfix(f1=f"{aggregate(rows).get('f1@0.5', 0):.3f}", s_per_patch=f"{dt:.2f}")
    bar.close()
    seconds = time.time() - t0

    metrics = aggregate(rows)
    metrics.update(seconds_per_patch=seconds / len(keys), n_patches_qc=n_qc, n_wrong_ignored=n_wrong_ignored, unit="patch")
    pd.DataFrame(rows).to_csv(out / "per_patch.csv", index=False)
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2))
    (out / "run.json").write_text(json.dumps({
        "name": name, "baseline": baseline.name, "params": baseline.params(), "device": getattr(baseline, "device", None),
        "input": input_meta(baseline.input),
        "split": split, "unit": "patch", "require_qc": require_qc, "n_patches": len(keys), "n_patches_qc": n_qc, "slides": list(cfg.slides),
        "regions": sorted({int(src.row(k)["region_id"]) for k in keys}),
        "ground_truth": {"scored": "owned nuclei only (patch overlaps counted once)", "ignored": "QC-Wrong nuclei", "n_gt_scored": metrics["n_gt"],
                         "n_wrong_ignored": n_wrong_ignored},
        "dataset": str(root), "dataset_created": info.get("created"), "dataset_format": info.get("format"),
        "predictions": str(_predictions_dir("patch", split, name, outputs_dir)) if store_predictions else None,
        "figures": str((Path(outputs_dir) if outputs_dir else OUTPUTS_ROOT) / "figures" / "patch" / split) if store_predictions else None,
        "seconds": seconds, "finished_at": time.strftime("%Y-%m-%dT%H:%M:%S"), "versions": _versions()}, indent=2, default=str))
    logger.info("done in %.0fs (%.2f s/patch): %d scored nuclei, %d Wrong ignored | %s -> %s", seconds, seconds / len(keys),
                metrics["n_gt"], n_wrong_ignored, "  ".join(f"{k} {metrics[k]:.3f}" for k in SUMMARY_KEYS if k in metrics), out)
    return out


def split_regions_of(root: Path, split: str, slides: Sequence[str]) -> dict:
    """``{slide: [region ids]}`` of the labeled patches of one split (regions were split as wholes, so this is exact)."""
    pt = pd.read_csv(root / "patches.csv", dtype={"split": str, "group": str, "kind": str})
    pt = pt[(pt.kind == "labeled") & pt.kept.astype(bool) & (pt.split == split) & pt.slide.isin(list(slides))]
    return {sl: sorted(int(r) for r in g.region_id.unique()) for sl, g in pt.groupby("slide")}


def run_baseline_regions(baseline: Baseline, cfg: DataConfig, *, split: str = "val", dataset_dir=None, results_dir=None,
                         name: Optional[str] = None, n: Optional[int] = None, require_qc: bool = False, batch_size: int = 8,
                         progress: bool = True, slide_root=None, store_predictions: bool = True, outputs_dir=None,
                         panels: bool = False) -> Path:
    root = Path(dataset_dir) if dataset_dir else DEFAULT_DATASET
    info = read_dataset_json(root)
    size, stride = int(info["geometry"]["size"]), int(info["geometry"]["stride"])
    per_slide = split_regions_of(root, split, cfg.slides)
    if not per_slide:
        raise ValueError(f"no labeled {split} regions for {list(cfg.slides)} in {root}")
    name = name or baseline.slug()
    out = (Path(results_dir) if results_dir else RESULTS_ROOT) / "region" / split / name
    out.mkdir(parents=True, exist_ok=True)
    baseline.resolve_input(root, cfg.slides)
    markers = baseline.input_markers() or list(cfg.markers)           # a model with an input recipe reads its own markers
    samples = []                                                       # (LiveSource, key) in a fixed order
    for slide, ids in per_slide.items():
        paths, mx, _ = ingest(slide, slide_root)
        regions = [r for r in load_regions(paths, ids, cfg.min_area_px) if r.has_qc or not require_qc]
        src = LiveSource(mx, regions, markers, cfg.level)
        samples += [(src, k) for k in src.keys]
    samples = samples[: n or None]
    if not samples:
        raise ValueError(f"no {split} regions left" + (" with a QC table" if require_qc else ""))
    logger.info("%s | %s split, %d whole regions %s | tiles %d px on a %d-px grid (training geometry)", baseline.describe(), split,
                len(samples), [k for _, k in samples], size, stride)

    rows: List[dict] = []
    n_wrong_ignored = 0
    t0 = time.time()
    bar = tqdm(total=len(samples), unit="region", desc=name, disable=not progress, dynamic_ncols=True)
    for src, key in samples:
        rs = src.load(key)
        labels, ignore = apply_qc_policy(rs.labels, rs.flags, "ignore_wrong")
        t1 = time.time()
        slide = rs.meta["slide"]
        pred = predict_tiled(lambda ims, px: baseline.predict_batch(ims, px, [slide] * len(ims)), rs.image, size, stride, batch_size,
                             rs.meta.get("pixel_size_um"), progress=progress, desc=f"{key} tiles")
        dt = time.time() - t1
        if store_predictions:
            save_prediction(_predictions_dir("region", split, name, outputs_dir), key, pred)
            if panels:                                            # PNG panels only on request (--panels)
                from .viz import export_region_panels
                export_region_panels((Path(outputs_dir) if outputs_dir else OUTPUTS_ROOT) / "figures" / "region" / split / key,
                                 rs.image, labels, ignore, rs.meta["has_qc"], {name: pred}, px_um=rs.meta.get("pixel_size_um"))
        wrong = np.unique(labels[ignore]) if ignore.any() else np.zeros(0, int)
        n_wrong_ignored += int((wrong > 0).sum())
        rows.append({"key": key, "region_id": rs.meta["region_id"], "has_qc": rs.meta["has_qc"], "height": labels.shape[0], "width": labels.shape[1],
                     "seconds": dt, **score_patch(labels, pred, ignore=ignore)})
        bar.update(1)
        bar.set_postfix(f1=f"{aggregate(rows).get('f1@0.5', 0):.3f}", s_per_region=f"{dt:.1f}")
    bar.close()
    seconds = time.time() - t0

    metrics = aggregate(rows)
    n_qc = sum(bool(r["has_qc"]) for r in rows)
    metrics.update(seconds_per_region=seconds / len(rows), n_regions=len(rows), n_regions_qc=n_qc, n_wrong_ignored=n_wrong_ignored, unit="region")
    pd.DataFrame(rows).to_csv(out / "per_region.csv", index=False)
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2))
    (out / "run.json").write_text(json.dumps({
        "name": name, "baseline": baseline.name, "params": baseline.params(), "device": getattr(baseline, "device", None),
        "input": input_meta(baseline.input),
        "split": split, "unit": "region", "require_qc": require_qc, "n_regions": len(rows), "n_regions_qc": n_qc, "slides": list(cfg.slides),
        "regions": sorted(int(r["region_id"]) for r in rows), "tile": {"size": size, "stride": stride, "stitching": "objects owned by the tile whose core holds their centroid"},
        "ground_truth": {"scored": "every nucleus of the region (one ground-truth map per region)", "ignored": "QC-Wrong nuclei",
                         "n_gt_scored": metrics["n_gt"], "n_wrong_ignored": n_wrong_ignored},
        "dataset": str(root), "dataset_created": info.get("created"), "dataset_format": info.get("format"),
        "predictions": str(_predictions_dir("region", split, name, outputs_dir)) if store_predictions else None,
        "figures": str((Path(outputs_dir) if outputs_dir else OUTPUTS_ROOT) / "figures" / "region" / split) if store_predictions else None,
        "seconds": seconds, "finished_at": time.strftime("%Y-%m-%dT%H:%M:%S"), "versions": _versions()}, indent=2, default=str))
    logger.info("done in %.0fs (%.1f s/region): %d scored nuclei, %d Wrong ignored | %s -> %s", seconds, seconds / len(rows),
                metrics["n_gt"], n_wrong_ignored, "  ".join(f"{k} {metrics[k]:.3f}" for k in SUMMARY_KEYS if k in metrics), out)
    return out


def aggregate_seeds(df: pd.DataFrame) -> pd.DataFrame:
    """Runs named ``<row>_s<seed>`` collapsed to one row per ``<row>`` and split: ``mean ± std`` (population std) over the
    seeds of every summary metric, ``seeds`` = how many; rows without the suffix pass through as a single seed."""
    if not len(df):
        return df
    keys = [k for k in SUMMARY_KEYS if k in df.columns]
    rows = []
    for (split, name), g in df.assign(_row=df.name.str.replace(r"_s\d+$", "", regex=True)).groupby(["split", "_row"], sort=False):
        row = {"split": split, "name": name, "seeds": len(g), "n": g.n.iloc[0], "_f1": float(g["f1@0.5"].mean()) if "f1@0.5" in g else 0.0}
        for k in keys:
            row[k] = f"{g[k].mean():.3f} ± {g[k].std(ddof=0):.3f}" if len(g) > 1 else f"{g[k].iloc[0]:.3f}"
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["split", "_f1"], ascending=[True, False]).drop(columns="_f1")


def compare_results(results_dir: "str | Path | None" = None, split: Optional[str] = None, unit: str = "patch") -> pd.DataFrame:
    """One row per stored run of one unit (``patch`` or ``region``): name, params, split, n, the summary metrics, seconds per unit."""
    root = Path(results_dir) if results_dir else RESULTS_ROOT
    if unit == "region":
        root = root / "region"
    elif unit != "patch":
        raise ValueError(f"unit must be 'patch' or 'region', got {unit!r}")
    rows = []
    for m in sorted(root.glob("*/*/metrics.json")):
        if unit == "patch" and m.parent.parent.parent != root:           # results/region/... are listed with unit="region"
            continue
        run = json.loads((m.parent / "run.json").read_text()) if (m.parent / "run.json").exists() else {}
        if split and run.get("split", m.parent.parent.name) != split:
            continue
        met = json.loads(m.read_text())
        rows.append({"split": run.get("split", m.parent.parent.name), "name": m.parent.name, "unit": met.get("unit", unit),
                     "params": ",".join(f"{k}={v}" for k, v in run.get("params", {}).items()),
                     "n": met.get("n_regions", met.get("n_patches")), "qc_only": run.get("require_qc", False),
                     **{k: round(met[k], 3) for k in SUMMARY_KEYS if k in met},
                     "s/unit": round(met.get("seconds_per_region", met.get("seconds_per_patch", float("nan"))), 3), "finished": run.get("finished_at", "")})
    return pd.DataFrame(rows).sort_values(["split", "f1@0.5"], ascending=[True, False]) if rows else pd.DataFrame(columns=["split", "name"])
