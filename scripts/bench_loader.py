#!/usr/bin/env python
"""Inspect what the loader returns and measure how fast it samples batches from the segmentation dataset.

    conda activate dl
    python scripts/bench_loader.py -c configs/slide-0235.yaml --workers 0 4 [--mode self_supervised]
"""
from __future__ import annotations

import argparse
import dataclasses
import os
import time

import torch

from mxdata import load_config, setup_logging
from mxdata.data import build_datasets, make_dataloader


def describe_batch(b: dict) -> None:
    for k, v in b.items():
        if torch.is_tensor(v):
            extra = f"  min {v.min():.3f} max {v.max():.3f}" if v.dtype == torch.float32 else ""
            print(f"  {k:8s} {str(tuple(v.shape)):22s} {str(v.dtype):14s} {v.numel() * v.element_size() / 1e6:6.2f} MB{extra}")
        else:
            print(f"  {k:8s} list[{len(v)}] of dict  e.g. {{key: {v[0]['key']!r}, kind: {v[0]['kind']!r}, region: {v[0]['region_id']}}}")
    assert 0.0 <= float(b["image"].min()) and float(b["image"].max()) <= 1.0
    if "labels" in b:
        assert (b["ignore"] & (b["labels"] == 0)).sum() == 0, "ignore pixels must lie on nuclei"


def time_epochs(loader, epochs: int = 2) -> list:
    out = []
    for _ in range(epochs):
        t0 = time.perf_counter()
        n = sum(b["image"].shape[0] for b in loader)
        dt = time.perf_counter() - t0
        out.append((dt, n / dt))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("-c", "--config", default="configs/slide-0235.yaml")
    ap.add_argument("--dataset", default=None, help="dataset dir (default: $MXDATA_ROOT/dataset)")
    ap.add_argument("--workers", type=int, nargs="+", default=[0, 4])
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--mode", default=None, choices=["supervised", "self_supervised"])
    args = ap.parse_args()
    setup_logging("WARNING")
    cfg = load_config(args.config)
    if args.mode:
        cfg = dataclasses.replace(cfg, mode=args.mode)
    bundle = build_datasets(cfg, args.dataset)
    print(f"machine: {os.cpu_count()} CPUs, torch {torch.__version__}, mps={torch.backends.mps.is_available()}, cuda={torch.cuda.is_available()}")
    print(bundle.summary())
    describe_batch(next(iter(make_dataloader(bundle.train, cfg.loader, train=True, batch_size=args.batch_size, num_workers=0))))
    for w in args.workers:
        loader = make_dataloader(bundle.train, cfg.loader, train=True, batch_size=args.batch_size, num_workers=w)
        (t1, _), (t2, rate) = time_epochs(loader)
        print(f"  workers={w}: epoch1 {t1:.2f}s (incl. worker start-up) | epoch2 {t2:.2f}s = {rate:.0f} patches/s")
        del loader


if __name__ == "__main__":
    main()
