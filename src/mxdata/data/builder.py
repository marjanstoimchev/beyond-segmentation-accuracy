"""Everything from one config: dataset directory -> per-slide normalizer -> train/val datasets -> loaders.

    bundle = build_datasets(cfg)                 # from $MXDATA_ROOT/dataset (or dataset_dir=...)
    train_loader, val_loader = build_loaders(bundle)
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict

from torch.utils.data import DataLoader

from ..config import DataConfig
from ..errors import ValidationError
from ..io.paths import DEFAULT_DATASET
from ..pipeline.stats import NormStats
from .datasets import RegionDataset
from .loaders import make_dataloader
from .sources import PatchSource
from .transforms import Augment, Normalizer

logger = logging.getLogger(__name__)


@dataclass
class DataBundle:
    cfg: DataConfig
    train: RegionDataset
    val: RegionDataset
    normalizer: Normalizer
    stats: Dict[str, NormStats]
    dataset_dir: Path

    def counts(self, ds: RegionDataset) -> dict:
        t = ds.source.table
        return {k: int((t.kind == k).sum()) for k in ("labeled", "unlabeled")}

    def summary(self) -> str:
        return (f"{'+'.join(self.cfg.slides)} [{self.cfg.mode}] from {self.dataset_dir}: train {len(self.train)} patches {self.counts(self.train)}, "
                f"val {len(self.val)} patches {self.counts(self.val)}; markers {list(self.cfg.markers)}, qc_policy {self.cfg.qc_policy}, {self.normalizer}")


def build_datasets(cfg: DataConfig, dataset_dir: "str | Path | None" = None) -> DataBundle:
    root = Path(dataset_dir) if dataset_dir else DEFAULT_DATASET
    kind = "labeled" if cfg.supervised else None            # self-supervised: labeled + unlabeled, labels ignored
    train_src = PatchSource(root, split="train", kind=kind, slides=cfg.slides)
    val_src = PatchSource(root, split="val", kind=kind, slides=cfg.slides)
    if [m.lower() for m in train_src.markers] != [m.lower() for m in cfg.markers]:
        raise ValidationError(f"dataset at {root} holds markers {train_src.markers}, config asks {list(cfg.markers)}",
                              hint="re-sample into another directory or change cfg.markers")
    if missing := [s for s in cfg.slides if s not in set(train_src.info.get("slides", []))]:
        raise ValidationError(f"dataset at {root} has no slide(s) {missing}", hint=f"it holds {train_src.info.get('slides')}")
    if not train_src.keys:
        raise ValidationError(f"dataset at {root} has no {kind or ''} training patches for slides {list(cfg.slides)}")
    n = cfg.normalization
    if n.mode == "slide":
        if missing := [s for s in cfg.slides if s not in train_src.stats]:
            raise ValidationError(f"no normalization stats for {missing} in {root}", hint="re-run `mxdata sample` or set normalization.mode: region")
        normalizer = Normalizer.from_stats(train_src.stats, cfg.markers, n.pmin, n.pmax)
    else:
        normalizer = Normalizer(n.mode, pmin=n.pmin, pmax=n.pmax)
    common = dict(qc_policy=cfg.qc_policy, normalizer=normalizer, with_labels=cfg.supervised, seed=cfg.loader.seed)
    bundle = DataBundle(cfg, RegionDataset(train_src, augment=Augment(cfg.augment.flip, cfg.augment.rot90), **common),
                        RegionDataset(val_src, augment=None, **common), normalizer, train_src.stats, root)
    logger.info(bundle.summary())
    return bundle


def build_loaders(bundle: DataBundle) -> tuple[DataLoader, DataLoader]:
    return make_dataloader(bundle.train, bundle.cfg.loader, train=True), make_dataloader(bundle.val, bundle.cfg.loader, train=False)
