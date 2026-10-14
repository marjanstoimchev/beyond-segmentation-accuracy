"""The MxIF patches as a KODIAK pretraining stream.

Every kept patch of a split — labeled and unlabeled alike, labels never read — becomes the three-channel image the encoder
expects: ``red`` and ``blue`` are fixed markers (the nuclear envelope and the nuclei, the channels the reference boundaries
were drawn from), and ``green`` is drawn **once per sample** from a weighted set of the other markers, or left blank. The
draw is shared by every crop of the sample, so the method's cross-view objectives compare views that agree on content;
the variety enters across samples and epochs, and the encoder learns to read any marker in green without being asked to
ignore one. Intensities are scaled with the dataset's stored per-slide statistics — the same numbers the segmenters use.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

import lightning.pytorch as L
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from mxdata.data.composite import marker_index, render_channels
from mxdata.pipeline.patches import read_image
from mxdata.pipeline.sample import TEXT_COLUMNS, read_dataset_json

from .collate import MultiCropKodiakCollate
from .masking import MaskingGenerator
from .transforms import make_transform

BLANK = "blank"


class ChannelPatches(Dataset):
    """Patches of one split as (red, green, blue) marker images. ``green`` maps marker names (and ``"blank"``) to sampling
    weights; None = always blank. Slide handles are never needed: the stacks are stored per patch."""

    def __init__(self, dataset_dir=None, split: str = "train", red: Optional[str] = "LAP2", blue: Optional[str] = "DAPI",
                 green: Optional[Dict[str, float]] = None, transform=None, seed: int = 0):
        from mxdata import DEFAULT_DATASET
        from mxseg.inputs import dataset_norms
        self.root = Path(dataset_dir or DEFAULT_DATASET)
        self.markers = list(read_dataset_json(self.root)["markers"])
        df = pd.read_csv(self.root / "patches.csv", dtype={c: str for c in TEXT_COLUMNS})
        df = df[df.kept.astype(bool) & (df.split == split)].reset_index(drop=True)
        self.rows = df[["key", "slide", "image", "kind"]].to_dict("records")
        green = {str(k): float(v) for k, v in (green or {BLANK: 1.0}).items()}
        if not green or any(v < 0 for v in green.values()) or sum(green.values()) <= 0:
            raise ValueError(f"green must map markers (or 'blank') to non-negative weights that do not all vanish, got {green}")
        self.green_choices = list(green)
        w = np.array([green[k] for k in self.green_choices], np.float64)
        self.green_weights = w / w.sum()
        self.red, self.blue = red, blue
        used = [m for m in [red, blue, *self.green_choices] if m and m != BLANK]
        for m in used:
            marker_index(self.markers, m)                                   # a clear KeyError for a marker the dataset lacks
        self.norms = dataset_norms(self.root, sorted(df.slide.unique()), list(dict.fromkeys(used))) if len(df) else {}
        self.transform, self.seed = transform, seed
        self._rng: Optional[np.random.Generator] = None

    def __len__(self) -> int:
        return len(self.rows)

    def rng(self) -> np.random.Generator:
        """Per worker and per epoch (Lightning reseeds the workers each epoch): the draws differ across epochs, and a run is
        reproducible under its seed."""
        if self._rng is None:
            info = torch.utils.data.get_worker_info()
            self._rng = np.random.default_rng([self.seed, torch.initial_seed() % 2**32, info.id if info else 0])
        return self._rng

    def draw_green(self) -> Optional[str]:
        g = self.green_choices[int(self.rng().choice(len(self.green_choices), p=self.green_weights))]
        return None if g == BLANK else g

    def green_index(self, g: Optional[str]) -> int:
        """Position of the drawn marker in the dataset's marker list (-1 = blank) — travels with the batch for bookkeeping."""
        return -1 if g is None else marker_index(self.markers, g)

    def __getitem__(self, i: int):
        r = self.rows[i]
        image = read_image(self.root / r["image"], self.markers)
        g = self.draw_green()
        x = torch.from_numpy(render_channels(image, self.markers, self.norms[r["slide"]], (self.red, g, self.blue)))
        out = self.transform(x) if self.transform is not None else x
        if isinstance(out, dict):
            out["green"] = self.green_index(g)
        return out

    def describe(self) -> str:
        draw = ", ".join(f"{k} {w:.2f}" for k, w in zip(self.green_choices, self.green_weights))
        return f"{len(self)} patches | R {self.red or '—'} · G drawn per sample ({draw}) · B {self.blue or '—'} | slides {sorted(self.norms)}"


class MxIFDataModule(L.LightningDataModule):
    """Multi-crop pretraining loaders over the MxIF patches (method-identical masking and collate; the transform and the
    channel draw are the data side of this repository)."""

    def __init__(self, dataset_dir=None, batch_size: int = 128, num_workers: int = 0, augmentation: Optional[dict] = None,
                 patch_size: int = 16, mask_ratio_min_max=(0.1, 0.5), mask_sample_probability: float = 0.5,
                 pin_memory: bool = False, input: Optional[dict] = None, seed: int = 0):
        super().__init__()
        inp = dict(input or {})
        self.channels = {"red": inp.get("red", "LAP2"), "blue": inp.get("blue", "DAPI"), "green": inp.get("green")}
        aug = dict(augmentation or {})
        self.transform = make_transform(aug)
        grid = int(aug.get("global_crops_size", 256)) // patch_size
        self.collate = MultiCropKodiakCollate(
            mask_generator=MaskingGenerator(input_size=(grid, grid), max_num_patches=int(0.5 * grid * grid)),
            mask_ratio_tuple=tuple(mask_ratio_min_max),
            mask_probability=mask_sample_probability,
            num_patches_total=grid * grid,
        )
        self.dataset_dir, self.batch_size, self.num_workers, self.pin_memory, self.seed = dataset_dir, batch_size, num_workers, pin_memory, seed

    def setup(self, stage=None):
        self.train_set = ChannelPatches(self.dataset_dir, "train", transform=self.transform, seed=self.seed, **self.channels)
        self.val_set = ChannelPatches(self.dataset_dir, "val", transform=self.transform, seed=self.seed, **self.channels)

    def _loader(self, ds, shuffle):
        return DataLoader(ds, batch_size=self.batch_size, shuffle=shuffle, num_workers=self.num_workers,
                          collate_fn=self.collate, pin_memory=self.pin_memory, drop_last=shuffle,
                          persistent_workers=self.num_workers > 0)

    def train_dataloader(self):
        return self._loader(self.train_set, True)

    def val_dataloader(self):
        return self._loader(self.val_set, False)
