"""DataLoader construction with the boring things done right: per-worker seeding of numpy / python / the dataset
RNG (otherwise every forked worker draws identical augmentations), persistent workers when there are any,
``drop_last`` and a seeded shuffle on training loaders, ``meta`` collated as a list."""
from __future__ import annotations

import random
from typing import Optional

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, get_worker_info

from ..config import LoaderConfig
from .datasets import collate_samples


def worker_init_fn(worker_id: int) -> None:
    seed = torch.initial_seed() % 2**32                 # unique per worker and per epoch
    np.random.seed(seed)
    random.seed(seed)
    info = get_worker_info()
    if info is not None and hasattr(info.dataset, "rng"):
        info.dataset.rng = np.random.default_rng(seed)


def make_dataloader(ds: Dataset, cfg: Optional[LoaderConfig] = None, *, train: bool = False,
                    batch_size: Optional[int] = None, num_workers: Optional[int] = None) -> DataLoader:
    cfg = cfg or LoaderConfig()
    nw = cfg.num_workers if num_workers is None else num_workers
    gen = torch.Generator()
    gen.manual_seed(cfg.seed)
    kw = dict(batch_size=batch_size or cfg.batch_size, shuffle=train, drop_last=train, num_workers=nw, collate_fn=collate_samples,
              pin_memory=cfg.pin_memory, worker_init_fn=worker_init_fn, generator=gen)
    if nw > 0:
        kw["persistent_workers"] = True
    return DataLoader(ds, **kw)
