"""One training dataset for every model — the guarantee that all baselines see the same data.

``TrainingPatches`` decides *what* is seen: the labeled patches of the stored split, the label policy (TINY removed, nuclei flagged wrong
kept in the labels but returned as ``ignore``), and the augmentation draw (D4: rotation + flips from a seeded RNG).
A model adapter subclasses it and only adds its own inputs/targets from the identical item (``CellposePatches``:
cellpose-normalized image + cached flow targets transformed with the same D4 parameters; ``UNetPatches``: 3-class map). With the same seed and the
same ``make_loader``, two adapters iterate the same keys, the same augmentations, in the same batch order.
"""
from __future__ import annotations

import dataclasses
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from mxdata import DataConfig, apply_qc_policy
from mxdata.data import PatchSource

from ..baselines.unet import percentile_normalize
from ..evaluate import owned_luts, verified_gt
from ..inputs import InputSpec, input_meta, select_markers
from .flows import cellpose_normalize, d4_arrays, d4_flows, labels_to_targets
from .threeclass import labels_to_classes

logger = logging.getLogger(__name__)


class TrainingPatches(Dataset):
    """Items: ``image (C,H,W) uint16``, ``labels (H,W) int32`` (all nuclei, TINY removed), ``ignore (H,W) bool`` (nuclei flagged wrong),
    ``key``, ``aug = (k, fh, fv)``. Subclasses add model-specific tensors via :meth:`prepare`."""

    def __init__(self, source: PatchSource, augment: bool, seed: int = 0, input: Optional[InputSpec] = None):
        self.source, self.augment, self.seed = source, augment, seed
        self.input = input                                            # the network's channels, rendered from the stored stack
        self.keys: List[str] = list(source.keys)
        self.rng = np.random.default_rng(seed)
        self._raw: Dict[str, tuple] = {}

    def __len__(self) -> int:
        return len(self.keys)

    def raw(self, key: str) -> tuple:
        if key not in self._raw:
            rs = self.source.load(key)
            labels, ignore = apply_qc_policy(rs.labels, rs.flags, "ignore_wrong")
            image = self.input.render_from(rs.image, self.source.markers, rs.meta["slide"]) if self.input is not None else rs.image
            self._raw[key] = (image, labels, ignore)
        return self._raw[key]

    def model_input(self, image: np.ndarray) -> np.ndarray:
        """A rendered marker image already is the network's [0, 1] input; the stored single marker is percentile-normalized here."""
        return image if self.input is not None else percentile_normalize(image)

    def draw(self) -> tuple:
        """The augmentation for the next item — identical for every model given the same seed and order."""
        if not self.augment:
            return (0, False, False)
        return (int(self.rng.integers(4)), bool(self.rng.random() < 0.5), bool(self.rng.random() < 0.5))

    def prepare(self, key: str, image: np.ndarray, labels: np.ndarray, ignore: np.ndarray, aug: tuple) -> Dict[str, object]:
        """Model-specific inputs/targets for one (already augmented) item. Base: the arrays as tensors."""
        return {"image": torch.from_numpy(image.astype(np.float32)), "labels": torch.from_numpy(labels), "ignore": torch.from_numpy(ignore)}

    def __getitem__(self, i: int) -> Dict[str, object]:
        key = self.keys[i]
        image, labels, ignore = self.raw(key)
        aug = self.draw()
        image, labels, ignore = d4_arrays(image, labels, ignore, k=aug[0], fh=aug[1], fv=aug[2])
        return {**self.prepare(key, image, labels, ignore, aug), "key": key, "aug": torch.tensor(aug, dtype=torch.int64)}

    def precompute(self, progress: bool = True) -> None:
        from tqdm import tqdm
        for k in tqdm(self.keys, desc="targets", unit="patch", disable=not progress, dynamic_ncols=True):
            self.raw(k)
            self.cache_targets(k)

    def cache_targets(self, key: str) -> None:            # hook for adapters with expensive targets
        pass


class CellposePatches(TrainingPatches):
    """Cellpose adapter: ``image`` cellpose-normalized (1, H, W); ``target`` (3, H, W) = [cellprob, dY, dX] from the cached,
    un-augmented flows, transformed with the item's D4 parameters (vectors included)."""

    def __init__(self, source: PatchSource, augment: bool, seed: int = 0, flow_device=None, **kw):
        super().__init__(source, augment, seed, **kw)
        self.flow_device = flow_device
        self._targets: Dict[str, np.ndarray] = {}

    def cache_targets(self, key: str) -> None:
        if key not in self._targets:
            self._targets[key] = labels_to_targets(self.raw(key)[1], self.flow_device)

    def prepare(self, key, image, labels, ignore, aug):
        self.cache_targets(key)
        target = d4_flows(self._targets[key], *aug)
        return {"image": torch.from_numpy(cellpose_normalize(image)), "target": torch.from_numpy(target), "ignore": torch.from_numpy(ignore)}


class FlowPatches(CellposePatches):
    """Flow adapter for the trained dense segmenters (KODIAK, U-Net): their own input convention (percentile-normalized)
    with Cellpose's cached flow targets — same keys, same D4 draws, same batch order as every other adapter."""

    def prepare(self, key, image, labels, ignore, aug):
        self.cache_targets(key)
        target = d4_flows(self._targets[key], *aug)
        return {"image": torch.from_numpy(self.model_input(image)), "target": torch.from_numpy(target), "ignore": torch.from_numpy(ignore)}


class UNetPatches(TrainingPatches):
    """U-Net adapter: ``image`` percentile-normalized (C, H, W) float32; ``target`` (H, W) int64 classes 0 / 1 / 2 computed from the
    augmented labels (cheap, so nothing is cached)."""

    def __init__(self, source: PatchSource, augment: bool, seed: int = 0, boundary_px: int = 2, **kw):
        super().__init__(source, augment, seed, **kw)
        self.boundary_px = boundary_px

    def prepare(self, key, image, labels, ignore, aug):
        return {"image": torch.from_numpy(self.model_input(image)),
                "target": torch.from_numpy(labels_to_classes(labels, self.boundary_px).astype(np.int64)), "ignore": torch.from_numpy(ignore)}


def _seed_worker(worker_id: int) -> None:
    info = torch.utils.data.get_worker_info()
    if info is not None:
        info.dataset.rng = np.random.default_rng(torch.initial_seed() % 2**32)


def make_loader(ds: Dataset, batch_size: int, num_workers: int, seed: int, train: bool) -> DataLoader:
    """Seeded batch order: identical across models for the same seed."""
    gen = torch.Generator()
    gen.manual_seed(seed)
    return DataLoader(ds, batch_size=batch_size, shuffle=train, drop_last=train, num_workers=num_workers, generator=gen,
                      worker_init_fn=_seed_worker, persistent_workers=num_workers > 0)


class TrainData:
    """Train / val datasets of one adapter + the verified ground truth for validation instance metrics."""

    def __init__(self, cfg: DataConfig, dataset_dir, adapter=TrainingPatches, *, batch_size: int, num_workers: int, augment: bool,
                 seed: int, n_metric_patches: Optional[int] = None, input: Optional[InputSpec] = None, **adapter_kwargs):
        self.root = Path(dataset_dir)
        markers = list(input.markers) if input is not None else None       # read only the model's markers out of the patch folders
        self.train_src = PatchSource(self.root, split="train", kind="labeled", slides=cfg.slides, markers=markers)
        self.val_src = PatchSource(self.root, split="val", kind="labeled", slides=cfg.slides, markers=markers)
        if not self.train_src.keys or not self.val_src.keys:
            raise ValueError(f"need labeled train and val patches for {list(cfg.slides)} in {self.root}")
        self.input = input
        self.train = adapter(self.train_src, augment=augment, seed=seed, input=input, **adapter_kwargs)
        self.val = adapter(self.val_src, augment=False, seed=seed, input=input, **adapter_kwargs)
        self.batch_size, self.num_workers, self.seed = batch_size, num_workers, seed
        self.metric_keys: List[str] = self.val_src.keys[: n_metric_patches or None]
        self.owned = owned_luts(self.root, self.metric_keys)
        logger.info("train %d patches, val %d patches (%d scored for instance metrics)", len(self.train), len(self.val), len(self.metric_keys))

    def manifest(self) -> dict:
        """What this run trains and validates on — write it next to the run so two runs can be diffed."""
        info = self.train_src.info
        return {"dataset": str(self.root), "dataset_created": info.get("created"), "dataset_format": info.get("format"),
                "slides": sorted(set(self.train_src.table.slide) | set(self.val_src.table.slide)),   # the slides this run trains / validates on
                "dataset_slides": info.get("slides"), "seed": self.seed, "batch_size": self.batch_size, "augment": self.train.augment,
                "n_train": len(self.train), "n_val": len(self.val), "n_metric": len(self.metric_keys),
                "input": input_meta(self.input), "train_keys": sorted(self.train.keys), "val_keys": sorted(self.val.keys)}

    def write_manifest(self, path: Path) -> None:
        path.write_text(json.dumps(self.manifest(), indent=2))

    def metric_samples(self):
        for k in self.metric_keys:
            rs = self.val_src.load(k)
            if self.input is not None:                                  # the model's markers out of the stored stack (labels unchanged)
                rs = dataclasses.replace(rs, image=select_markers(rs.image, self.val_src.markers, self.input.markers))
            yield rs, verified_gt(rs, self.owned)

    def train_dataloader(self) -> DataLoader:
        return make_loader(self.train, self.batch_size, self.num_workers, self.seed, train=True)

    def val_dataloader(self) -> DataLoader:
        return make_loader(self.val, self.batch_size, self.num_workers, self.seed, train=False)
