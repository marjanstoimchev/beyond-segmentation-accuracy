"""Sample: build (or extend) the fixed **segmentation dataset** from one or more slides.

Layout — one folder per slide, tables at the top level spanning all slides::

    <dataset>/
      dataset.json          format, patch geometry, kinds, markers, level, slides, counts
      patches.csv           one row per grid window, kept AND rejected (with the reason): provenance per patch
      instances.csv         one row per (labeled patch, nucleus): local id -> region nucleus id, QC flag, owned, border
      slides.csv            one row per slide: geometry, tissue gate, counts, normalization stats, source signature
      <SLIDE>/<R01_x8073_y2896 | U_x10368_y9728>/<marker>.tif [mask.png]   one folder per patch, one image per marker
      <SLIDE>/tables/<unit>_*.csv  stats.json  tissue.json  reports  resolved_config.yaml  .done.json

Resumable: a unit's tables are written before it is recorded in ``.done.json``; a failed unit is recorded with
its reason and the run continues; the top-level tables are rebuilt from every unit table at the end of every run
(splits are assigned then, so adding a slide re-splits deterministically under the same config).
"""
from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, List, Optional, Sequence

import pandas as pd

from .._version import __version__
from ..config import DataConfig, save_resolved, to_dict
from ..errors import DatasetError
from ..io.paths import DEFAULT_DATASET
from ..splits import SPLITS, assign_splits
from .patches import (
    INSTANCE_COLUMNS,
    KINDS,
    PATCH_COLUMNS,
    UnlabeledUnit,
    plan_labeled,
    plan_unlabeled,
    sample_labeled_unit,
    sample_unlabeled_unit,
)
from .prepare import ensure_stats, prepare, write_json
from .tissue import TissueGate

logger = logging.getLogger(__name__)
DATASET_FORMAT = 2          # bump when the tables' schema changes; readers refuse other formats
TEXT_COLUMNS = ("group", "split", "region_key", "reject_reason", "image", "mask")


@dataclass
class SampleResult:
    out_dir: Path
    slides: List[str]
    n_units: int
    n_done: int = 0
    n_skipped: int = 0
    n_failed: int = 0
    n_windows: int = 0              # labeled grid windows
    n_kept: int = 0                 # labeled patches kept
    n_unlabeled_windows: int = 0
    n_unlabeled_kept: int = 0
    failures: List[dict] = field(default_factory=list)
    dry_run: bool = False
    seconds: float = 0.0

    def summary(self) -> str:
        head = "dry run: " if self.dry_run else f"sample -> {self.out_dir}: "
        s = (f"{head}{len(self.slides)} slide(s), {self.n_units} units; labeled grid {self.n_windows} windows"
             f"{'' if self.dry_run else f' -> {self.n_kept} kept'}; whole-slide grid {self.n_unlabeled_windows} windows -> {self.n_unlabeled_kept} unlabeled tissue patches")
        return s if self.dry_run else s + f"; {self.n_done} units written, {self.n_skipped} already done, {self.n_failed} failed in {self.seconds:.1f}s"


def write_table(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    df.to_csv(tmp, index=False)
    tmp.replace(path)


def _geometry(cfg: DataConfig) -> dict:
    return {"size": cfg.patches.size, "stride": cfg.patches.stride, "unlabeled_stride": cfg.patches.effective_unlabeled_stride,
            "markers": list(cfg.markers), "level": cfg.level}


def read_dataset_json(out: Path) -> dict:
    """``dataset.json`` of a dataset directory, refusing other formats with a named error."""
    p = out / "dataset.json"
    if not p.exists():
        raise DatasetError(f"no dataset.json in {out}", hint="run `mxdata sample --out <dir>` first")
    d = json.loads(p.read_text())
    if d.get("format") != DATASET_FORMAT:
        raise DatasetError(f"{p} has format {d.get('format')}, this mxdata reads format {DATASET_FORMAT}",
                           hint=f"re-run `mxdata sample --out {out} --no-resume` to rewrite the dataset")
    return d


def _check_compatible(out: Path, cfg: DataConfig, resume: bool) -> None:
    """A dataset directory has one patch geometry and one channel set; refuse to mix."""
    if not (out / "dataset.json").exists():
        return
    if not resume:                                              # a full rewrite may change format and geometry
        return
    have = read_dataset_json(out)["geometry"]
    if have != _geometry(cfg):
        raise DatasetError(f"dataset at {out} was built with {have}, config asks {_geometry(cfg)}",
                           hint="use a different --out directory for a different patch geometry / channel set")


def sample_patches(cfg: DataConfig, out_dir: "str | Path | None" = None, *, root=None, slides: Optional[Sequence[str]] = None,
                   resume: bool = True, dry_run: bool = False, progress: Optional[Callable[[int, int, str], None]] = None) -> SampleResult:
    t0 = time.time()
    out = Path(out_dir) if out_dir else DEFAULT_DATASET
    res = SampleResult(out, list(slides or cfg.slides), 0, dry_run=dry_run)
    if not dry_run:
        _check_compatible(out, cfg, resume)
        out.mkdir(parents=True, exist_ok=True)

    for slide_name in res.slides:
        paths, slide, regions, ingest_d, valid_d, units = prepare(cfg, slide_name, root)
        gate = TissueGate.from_slide(slide, cfg.patches.tissue_marker, cfg.patches.tissue_max_dim)
        specs = plan_labeled(units, slide.name, cfg.patches.size, cfg.patches.stride, slide.downsamples[cfg.level])
        uunits = plan_unlabeled(slide, [r.box for r in regions], gate, cfg) if cfg.patches.unlabeled else []
        res.n_units += len(units) + len(uunits)
        res.n_windows += len(specs)
        res.n_unlabeled_windows += sum(len(u.windows) for u in uunits)
        res.n_unlabeled_kept += sum(not s.reject for u in uunits for s in u.windows)
        if dry_run:
            slide.close()
            continue

        sdir = out / slide.name
        (sdir / "tables").mkdir(parents=True, exist_ok=True)
        save_resolved(cfg, sdir / "resolved_config.yaml")
        write_json(sdir / "ingest_report.json", ingest_d)
        write_json(sdir / "validation_report.json", valid_d)
        write_json(sdir / "tissue.json", gate.info)
        stats = ensure_stats(cfg, slide, units, paths, sdir / "stats.json")
        done_path = sdir / ".done.json"
        done = json.loads(done_path.read_text()) if (resume and done_path.exists()) else {}
        done = {k: v for k, v in done.items() if (sdir / "tables" / f"{k}_patches.csv").exists()}   # trust only what is on disk
        by_id = {r.region_id: r for r in regions}
        by_unit: dict = {}
        for s in specs:
            by_unit.setdefault(f"{slide.name}_R{s.region_id:02d}", []).append(s)
        all_units = list(units) + list(uunits)
        try:
            for i, u in enumerate(all_units):
                if u.key in done:
                    res.n_skipped += 1
                    continue
                try:
                    if isinstance(u, UnlabeledUnit):
                        prows, irows = sample_unlabeled_unit(u, cfg, slide, sdir), []
                    else:
                        prows, irows = sample_labeled_unit(u, by_id[u.region_id], by_unit.get(u.key, []), cfg, slide, gate, sdir)
                    write_table(pd.DataFrame(prows, columns=PATCH_COLUMNS), sdir / "tables" / f"{u.key}_patches.csv")
                    write_table(pd.DataFrame(irows, columns=INSTANCE_COLUMNS), sdir / "tables" / f"{u.key}_instances.csv")
                except Exception as exc:                                   # structural: record, continue
                    logger.error("unit %s failed: %s", u.key, exc)
                    res.failures.append({"slide": slide.name, "key": u.key, "reason": f"{type(exc).__name__}: {exc}"})
                    continue
                done[u.key] = {"n_windows": len(prows), "n_kept": int(sum(r["kept"] for r in prows))}
                write_json(done_path, done)                                   # write, THEN record
                res.n_done += 1
                if progress:
                    progress(i + 1, len(all_units), u.key)
        finally:
            write_json(sdir / "sampling_report.json", {"slide": slide.name, "n_units": len(all_units), "units_done": len(done),
                                                       "n_windows_labeled": len(specs), "n_windows_unlabeled": res.n_unlabeled_windows,
                                                       "tissue": gate.info, "failures": [f for f in res.failures if f["slide"] == slide.name]})
            write_json(sdir / "slide.json", {"slide": slide.name, "root": str(paths.root), "width": slide.width, "height": slide.height,
                                             "pixel_size_um": slide.pixel_size_um, "markers": list(cfg.markers), "level": cfg.level,
                                             "n_regions": len(regions), "source_signature": paths.signature(),
                                             "sampled_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
                                             "lo": stats.lo if stats else None, "hi": stats.hi if stats else None})
            slide.close()

    res.n_failed = len(res.failures)
    res.seconds = time.time() - t0
    if not dry_run:
        counts = rebuild_tables(out, cfg)
        res.n_kept, res.n_unlabeled_kept = counts["n_labeled"], counts["n_unlabeled"]
        logger.info(res.summary())
    return res


def rebuild_tables(out: "str | Path", cfg: DataConfig) -> dict:
    """Concatenate every slide's unit tables, assign groups + splits, write patches/instances/slides + dataset.json."""
    out = Path(out)
    pframes, iframes, srows = [], [], []
    for sdir in sorted(d for d in out.iterdir() if (d / "slide.json").exists()):
        info = json.loads((sdir / "slide.json").read_text())
        tissue = json.loads((sdir / "tissue.json").read_text()) if (sdir / "tissue.json").exists() else {}
        pf = []
        for tp in sorted((sdir / "tables").glob("*_patches.csv")):
            d = pd.read_csv(tp, dtype={c: str for c in TEXT_COLUMNS})
            if missing := [c for c in PATCH_COLUMNS if c not in d.columns]:
                raise DatasetError(f"{tp} was written by an older mxdata (missing columns {missing})",
                                   hint=f"re-run `mxdata sample --out {out} --no-resume` to rewrite the dataset")
            if len(d):
                pf.append(d)
        inf = [d for d in (pd.read_csv(p) for p in sorted((sdir / "tables").glob("*_instances.csv"))) if len(d)]
        pdf = pd.concat(pf, ignore_index=True) if pf else pd.DataFrame(columns=PATCH_COLUMNS)
        pframes.append(pdf)
        iframes += inf
        kept = pdf[pdf.kept.astype(bool)] if len(pdf) else pdf
        srows.append({"slide": info["slide"], "root": info["root"], "width": info["width"], "height": info["height"],
                      "pixel_size_um": info["pixel_size_um"], "markers": ",".join(info["markers"]), "level": info["level"],
                      "n_regions": info["n_regions"], "n_windows": len(pdf), "n_kept": len(kept), "n_rejected": len(pdf) - len(kept),
                      "n_labeled": int((kept.kind == "labeled").sum()), "n_unlabeled": int((kept.kind == "unlabeled").sum()),
                      "tissue_method": tissue.get("method"), "tissue_threshold": tissue.get("threshold_intensity"),
                      "tissue_fraction_slide": tissue.get("tissue_fraction"), "source_signature": info["source_signature"],
                      "sampled_at": info["sampled_at"], "lo": json.dumps(info.get("lo")), "hi": json.dumps(info.get("hi"))})
    patches = pd.concat([f for f in pframes if len(f)], ignore_index=True) if any(len(f) for f in pframes) else pd.DataFrame(columns=PATCH_COLUMNS)
    instances = pd.concat(iframes, ignore_index=True) if iframes else pd.DataFrame(columns=INSTANCE_COLUMNS)
    if len(patches):
        patches = patches.drop_duplicates("key", keep="last").reset_index(drop=True)
        for col in TEXT_COLUMNS:
            patches[col] = patches[col].fillna("").astype(str)
        lab = patches.kind == "labeled"
        if lab.any():
            groups, splits = assign_splits(patches[lab], cfg.split)
            patches.loc[lab, "group"] = patches.loc[lab, "region_key"].map(groups)
            patches.loc[lab, "split"] = patches.loc[lab, "region_key"].map(splits)
        val_slides, test_slides = set(cfg.split.val_slides or ()), set(cfg.split.test_slides or ())   # unlabeled patches follow their slide
        patches.loc[~lab, "group"] = patches.loc[~lab, "slide"] + "/unlabeled"
        patches.loc[~lab, "split"] = patches.loc[~lab, "slide"].map(lambda s: "test" if s in test_slides else ("val" if s in val_slides else "train"))
        bad = int((lab & patches.kept.astype(bool) & (patches["mask"] == "")).sum() + (~lab & (patches["mask"] != "")).sum())
        if bad:
            raise DatasetError(f"inconsistent dataset tables: {bad} patch rows whose kind and mask disagree")
    write_table(patches[PATCH_COLUMNS], out / "patches.csv")
    write_table(instances[INSTANCE_COLUMNS], out / "instances.csv")
    write_table(pd.DataFrame(srows), out / "slides.csv")
    kept = patches[patches.kept.astype(bool)] if len(patches) else patches
    counts = {"n_slides": len(srows), "n_windows": len(patches), "n_kept": len(kept),
              "n_labeled": int((kept.kind == "labeled").sum()), "n_unlabeled": int((kept.kind == "unlabeled").sum()),
              "n_instances": len(instances),
              "per_kind_split": {f"{k}/{s}": int(((kept.kind == k) & (kept.split == s)).sum()) for k in KINDS for s in SPLITS}}
    write_json(out / "dataset.json", {
        "format": DATASET_FORMAT, "created": time.strftime("%Y-%m-%dT%H:%M:%S"), "mxdata_version": __version__,
        "geometry": _geometry(cfg), "patch": to_dict(cfg.patches), "kinds": list(KINDS), "markers": list(cfg.markers), "level": cfg.level,
        "qc_policy_default": cfg.qc_policy, "min_area_px": cfg.min_area_px, "split": to_dict(cfg.split), "slides": [r["slide"] for r in srows],
        "counts": counts,
        "conventions": {"coordinates": "level-0 pixels, half-open boxes",
                        "mask": "<patch folder>/mask.png: every HALO cell, uint16 ids local to the patch (inspection); mask_true.png: the same without the cells flagged Wrong (training / scoring); cells.csv: id, flag, ownership per cell",
                        "image": "<patch folder>/<marker>.tif: uint16, zlib, one file per marker; the (C, H, W) stack follows dataset.json markers order",
                        "flags": "instances.csv: flag per (patch, local_id)",
                        "kind": "labeled = inside an annotated FOV with a complete mask; unlabeled = tissue outside every FOV, image only"}})
    return counts


def dataset_summary(out: "str | Path") -> str:
    out = Path(out)
    d = read_dataset_json(out)
    patches = pd.read_csv(out / "patches.csv", dtype={c: str for c in TEXT_COLUMNS})
    kept, rej = patches[patches.kept.astype(bool)], patches[~patches.kept.astype(bool)]
    g = d["geometry"]
    lines = [f"segmentation dataset {out} (format {d['format']}, mxdata {d['mxdata_version']}, created {d['created']})",
             f"  patch {g['size']}² · labeled stride {g['stride']} (overlap {g['size'] - g['stride']}) · unlabeled stride {g['unlabeled_stride']} · markers {g['markers']} · level {g['level']}",
             f"  slides: {d['slides']}",
             f"  windows {len(patches)} -> kept {len(kept)}: {d['counts']['n_labeled']} labeled + {d['counts']['n_unlabeled']} unlabeled; rejected: "
             + (", ".join(f"{k}: {v}" for k, v in rej.reject_reason.fillna("").value_counts().items()) or "none"),
             f"  instances: {d['counts']['n_instances']}; per kind/split: {d['counts']['per_kind_split']}"]
    if len(kept):
        lines.append(kept.groupby(["slide", "kind", "split"]).agg(patches=("key", "count"), regions=("region_id", "nunique"), instances=("n_instances", "sum"),
                                                                  wrong=("n_wrong", "sum"), tissue=("tissue_fraction", "mean")).round(3).to_string())
    return "\n".join(lines)
