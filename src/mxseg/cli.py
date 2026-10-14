"""``mxseg`` — run and compare segmentation baselines on the mxdata dataset.

    mxseg eval    -c configs/slide-0235.yaml --baseline cellpose [--param variant=cpsam_v2 --param diameter=25]
                  [--split val] [--require-qc] [--n 20] [--name NAME] [--dataset DIR] [--results DIR]
    mxseg compare [--results DIR] [--split val]
    mxseg export  runs/<name> [--checkpoint best|last|PATH]
    mxseg train   -c configs/train_cellpose.yaml [--epochs N] [--name NAME] [--train-blocks N] [--set KEY=VALUE]* [--fast-dev-run]

``--param key=value`` passes any constructor argument of the chosen baseline (values parsed as Python literals),
so switching baselines or sweeping their settings never needs new flags.
"""
from __future__ import annotations

import argparse
import ast
import dataclasses
import sys
from pathlib import Path

import pandas as pd

from mxdata import DataConfig, MxDataError, load_config, setup_logging

from ._version import __version__
from .baselines import BASELINES, from_config, get_baseline
from .evaluate import aggregate_seeds, compare_results, run_baseline


def _params(items) -> dict:
    out = {}
    for it in items or []:
        if "=" not in it:
            raise ValueError(f"expected key=value, got {it!r}")
        k, v = it.split("=", 1)
        try:
            out[k] = ast.literal_eval(v)
        except (ValueError, SyntaxError):
            out[k] = v
    return out


def cmd_eval(args) -> int:
    default_cfg = Path("configs/dataset.yaml")
    cfg = load_config(args.config) if args.config else (load_config(default_cfg) if default_cfg.is_file() else DataConfig())
    if args.slides:                                                     # a subset of the config's slides
        if unknown := set(args.slides) - set(cfg.slides):
            raise ValueError(f"--slides {sorted(unknown)} are not in the dataset config (slides: {list(cfg.slides)})")
        cfg = dataclasses.replace(cfg, slides=tuple(args.slides))
    if args.baseline.endswith((".yaml", ".yml")) or Path(args.baseline).is_file():
        baseline, cfg_name = from_config(args.baseline, **_params(args.param))
        args.name = args.name or cfg_name
    else:
        baseline = get_baseline(args.baseline, **_params(args.param))
    print(f"baseline {baseline.describe()} | split {args.split} | unit {args.unit} | slides {list(cfg.slides)}")
    out = run_baseline(baseline, cfg, split=args.split, dataset_dir=args.dataset, results_dir=args.results, name=args.name,
                       n=args.n, require_qc=args.require_qc, unit=args.unit, slide_root=args.root,
                       store_predictions=args.store_predictions, panels=args.panels)
    print(f"\nresults -> {out}\n")
    pd.set_option("display.width", 220)
    print(compare_results(args.results, split=args.split, unit=args.unit).to_string(index=False))
    return 0


def cmd_train(args) -> int:
    from .train import load_train_config, train
    from .train.run import eval_command
    cfg = load_train_config(args.config, epochs=args.epochs, name=args.name, batch_size=args.batch_size, accelerator=args.accelerator,
                            slides=args.slides, **{"cellpose.train_blocks": args.train_blocks, **_params(args.set)})
    out = train(cfg, fast_dev_run=args.fast_dev_run)
    print(f"\nrun -> {out}\nevaluate it with:\n  {eval_command(cfg, out)}")
    return 0


def cmd_export(args) -> int:
    from .train import load_train_config
    from .train.run import eval_command, export_weights
    out = export_weights(args.run, args.checkpoint)
    cfg = load_train_config(Path(args.run) / "resolved_config.yaml")
    print(f"weights -> {out}\nevaluate them with:\n  {eval_command(cfg, Path(args.run))}")
    return 0


def cmd_compare(args) -> int:
    pd.set_option("display.width", 220)
    df = compare_results(args.results, split=args.split, unit=args.unit)
    if args.seeds:
        df = aggregate_seeds(df)
    print(df.to_string(index=False) if len(df) else "no results yet")
    return 0


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="mxseg", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--version", action="version", version=f"mxseg {__version__}")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("eval", help="run a baseline over a split and store the results")
    p.add_argument("--config", "-c", help="mxdata experiment config (slides, markers)")
    p.add_argument("--baseline", "-b", required=True,
                   help=f"a configs/baselines/*.yaml (the run's full definition) or a registry name: {', '.join(sorted(BASELINES))}")
    p.add_argument("--param", "-p", action="append", help="baseline constructor argument, key=value (repeatable)")
    p.add_argument("--split", default="val", choices=["train", "val", "test"], help="test = the held-out set: evaluate once, never tune on it")
    p.add_argument("--slides", nargs="+", metavar="SLIDE", help="only these slides of the dataset config (default: all; the split still applies)")
    p.add_argument("--unit", default="patch", choices=["patch", "region"],
                   help="patch = stored patches, owned nuclei (fast proxy) | region = whole annotated regions predicted from training-size tiles (the paper number)")
    p.add_argument("--root", help="folder holding the SLIDE-* folders, for --unit region (default: $MXDATA_ROOT)")
    p.add_argument("--store-predictions", action="store_true", dest="store_predictions", default=True,
                   help="(default) save every predicted instance map to outputs/<unit>/<split>/<name>/ (uint16 PNG) — notebooks plot without re-running inference")
    p.add_argument("--no-store-predictions", action="store_false", dest="store_predictions", help="metrics only, skip outputs/")
    p.add_argument("--panels", action="store_true", help="also export PNG panels per region / patch under outputs/figures/ (off by default)")
    p.add_argument("--n", type=int, help="only the first N patches (quick check)")
    p.add_argument("--name", help="result name (default: baseline + params)")
    p.add_argument("--require-qc", action="store_true", help="score only patches from regions with a QC table (fully verified GT)")
    p.add_argument("--dataset", help="dataset directory (default: $MXDATA_ROOT/dataset)")
    p.add_argument("--results", help="results root (default: ./results)")
    p.add_argument("-v", "--verbose", action="store_true")
    p.set_defaults(fn=cmd_eval)
    p = sub.add_parser("train", help="fine-tune a model with Lightning (configs/train_*.yaml)")
    p.add_argument("--config", "-c", required=True, help="train config YAML")
    p.add_argument("--epochs", type=int, help="override epochs")
    p.add_argument("--name", help="override run name (runs/<name>)")
    p.add_argument("--batch-size", type=int, dest="batch_size", help="override batch size")
    p.add_argument("--accelerator", choices=["auto", "mps", "cuda", "cpu"], help="override accelerator")
    p.add_argument("--train-blocks", type=int, dest="train_blocks", metavar="N",
                   help="train only the last N ViT blocks (+ neck + head); 24 = full fine-tune, 0 = head only, encoder frozen")
    p.add_argument("--slides", nargs="+", metavar="SLIDE",
                   help="train on these slides of the dataset config only (default: all; the split still decides train / val)")
    p.add_argument("--set", action="append", metavar="KEY=VALUE",
                   help="override any config key, dotted for nested (repeatable): --set optim.lr=1e-3 --set val.every_n_epochs=1")
    p.add_argument("--fast-dev-run", action="store_true", help="one train + one val batch, to check the pipeline")
    p.add_argument("-v", "--verbose", action="store_true")
    p.set_defaults(fn=cmd_train)
    p = sub.add_parser("export", help="write runs/<name>/weights.pt from a checkpoint (best | last | path), e.g. after an interrupted run")
    p.add_argument("run", help="run directory, e.g. runs/cellpose_head_lr1e-3")
    p.add_argument("--checkpoint", default="best", help="best (the run's monitored optimum: minimum val/loss by default) | last | path to a .ckpt")
    p.add_argument("-v", "--verbose", action="store_true")
    p.set_defaults(fn=cmd_export)

    p = sub.add_parser("compare", help="table of all stored runs")
    p.add_argument("--seeds", action="store_true", help="collapse runs named <row>_s<seed> to one row per <row>: mean ± std over seeds")
    p.add_argument("--results", help="results root (default: ./results)")
    p.add_argument("--split", choices=["train", "val", "test"])
    p.add_argument("--unit", default="patch", choices=["patch", "region"])
    p.add_argument("-v", "--verbose", action="store_true")
    p.set_defaults(fn=cmd_compare)
    return ap


def _quiet_imports() -> None:
    """Silence third-party import noise (timm deprecations, segment_anything re-registering tiny_vit models)."""
    import warnings
    warnings.filterwarnings("ignore", category=FutureWarning, module=r"timm\..*")
    warnings.filterwarnings("ignore", message=r"Overwriting tiny_vit_.*", category=UserWarning)


def main(argv=None) -> int:
    _quiet_imports()
    args = build_parser().parse_args(argv)
    setup_logging("DEBUG" if args.verbose else "INFO")
    try:
        return int(args.fn(args))
    except (MxDataError, KeyError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
