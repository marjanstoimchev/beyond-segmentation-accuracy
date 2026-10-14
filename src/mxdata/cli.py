"""``mxdata`` command line — one sub-command per thing you do with a slide:

    mxdata report   -c CONFIG            what is there: geometry, regions, QC, overlaps, validation summary
    mxdata validate -c CONFIG            full validation report; exit code 1 on errors
    mxdata sample   -c CONFIG [-o DIR]   build / extend the segmentation dataset (--dry-run counts, writes nothing)
    mxdata dataset  [-o DIR]             summary of a dataset directory
    mxdata config   -c CONFIG [-o FILE]  print (or write) the fully resolved config

Environment config: --root / MXDATA_ROOT (slide folders); -o defaults to $MXDATA_ROOT/dataset.
"""
from __future__ import annotations

import argparse
import sys

import pandas as pd
import yaml

from ._version import __version__
from .config import DataConfig, load_config, save_resolved, to_dict
from .errors import MxDataError
from .io.paths import DEFAULT_DATASET
from .log import setup_logging


def _cfg(args) -> DataConfig:
    cfg = load_config(args.config) if args.config else DataConfig()
    return cfg.with_slides(*args.slides) if getattr(args, "slides", None) else cfg


def cmd_report(args) -> int:
    from .annotations import load_regions, overlap_table, regions_table
    from .pipeline import ingest, validate
    from .splits import overlap_groups
    cfg = _cfg(args)
    pd.set_option("display.width", 220)
    pd.set_option("display.max_columns", 40)
    for name in cfg.slides:
        paths, slide, rep = ingest(name, args.root)
        print(paths.summary(), "\n" + rep.summary(), "\n")
        regions = load_regions(paths, cfg.regions, cfg.min_area_px)
        tab = regions_table(regions, slide.pixel_size_um)
        print(tab.to_string(), "\n")
        print(f"total nuclei: {int(tab.n_nuclei.sum())} in {len(tab)} regions; QC tables for {int(tab.has_qc.sum())} regions "
              f"({int(tab.n_wrong.sum())} flagged Wrong)\n")
        ov = overlap_table(regions)
        print("overlapping FOV pairs:\n" + (ov.to_string(index=False) if len(ov) else "  none"), "\n")
        print("overlap groups (kept together when splitting):", overlap_groups(regions), "\n")
        print(validate(slide, regions, cfg).summary("warning"), "\n")
        slide.close()
    return 0


def cmd_validate(args) -> int:
    from .annotations import load_regions
    from .pipeline import ingest, validate
    cfg = _cfg(args)
    ok = True
    for name in cfg.slides:
        paths, slide, _ = ingest(name, args.root)
        rep = validate(slide, load_regions(paths, cfg.regions, cfg.min_area_px), cfg)
        print(rep.summary("info" if args.verbose else "warning"))
        ok &= rep.ok
        slide.close()
    return 0 if ok else 1


def cmd_sample(args) -> int:
    from .pipeline import dataset_summary, sample_patches
    res = sample_patches(_cfg(args), args.out, root=args.root, resume=not args.no_resume, dry_run=args.dry_run,
                         progress=lambda i, n, key: print(f"  [{i}/{n}] {key}", flush=True))
    print(res.summary())
    for f in res.failures:
        print(f"  FAILED {f['key']}: {f['reason']}")
    if not args.dry_run:
        print(dataset_summary(res.out_dir))
    return 0 if res.n_failed == 0 else 2


def cmd_dataset(args) -> int:
    from .pipeline import dataset_summary
    print(dataset_summary(args.out or DEFAULT_DATASET))
    return 0


def cmd_config(args) -> int:
    cfg = _cfg(args)
    print(f"wrote {save_resolved(cfg, args.out)}" if args.out else yaml.safe_dump(to_dict(cfg), sort_keys=False))
    return 0


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="mxdata", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--version", action="version", version=f"mxdata {__version__}")
    sub = ap.add_subparsers(dest="cmd", required=True)

    def add(name, help, fn, *, slide_args=True, out=None):
        p = sub.add_parser(name, help=help)
        if slide_args:
            p.add_argument("--config", "-c", help="experiment config YAML (default: built-in defaults)")
            p.add_argument("--slides", nargs="+", help="override the slide list from the config")
            p.add_argument("--root", help="folder holding SLIDE-*/ (default: $MXDATA_ROOT)")
        if out:
            p.add_argument("--out", "-o", help=out)
        p.add_argument("-v", "--verbose", action="store_true")
        p.set_defaults(fn=fn)
        return p

    add("report", "what is there: geometry, regions, QC, overlaps, validation summary", cmd_report)
    add("validate", "full validation report (exit 1 on errors)", cmd_validate)
    p = add("sample", "build / extend the dataset", cmd_sample, out="dataset directory (default: $MXDATA_ROOT/dataset)")
    p.add_argument("--no-resume", action="store_true", help="redo every unit")
    p.add_argument("--dry-run", action="store_true", help="count windows, write nothing")
    add("dataset", "summarize a dataset directory", cmd_dataset, slide_args=False, out="dataset directory (default: $MXDATA_ROOT/dataset)")
    add("config", "print (or write) the fully resolved config", cmd_config, out="write the resolved YAML here")
    return ap


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    setup_logging("DEBUG" if args.verbose else "INFO")
    try:
        return int(args.fn(args))
    except MxDataError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
