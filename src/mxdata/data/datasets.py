"""``RegionDataset``: one patch (or whole FOV) per item, as tensors.

``with_labels=True`` (supervised): items carry ``labels`` + ``ignore`` under the QC policy, and every key of the
source must be labeled. ``with_labels=False`` (self-supervised): items carry ``image`` + ``meta`` only — labels are
ignored even where they exist. Samples are normalized (per slide), cached per process, then augmented with the
per-worker ``rng`` (reseeded by ``worker_init_fn``).
"""
from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import torch
from torch.utils.data import Dataset

from ..annotations.raster import QC_POLICIES, apply_qc_policy
from ..errors import ValidationError
from .sources import RegionSource
from .transforms import Augment, Normalizer, Sample


class RegionDataset(Dataset):
    def __init__(self, source: RegionSource, *, qc_policy: str = "ignore_wrong", normalizer: Optional[Normalizer] = None,
                 augment: Optional[Augment] = None, with_labels: bool = True, seed: int = 0, cache: bool = True):
        if qc_policy not in QC_POLICIES:
            raise ValidationError(f"qc_policy must be one of {QC_POLICIES}, got {qc_policy!r}")
        if with_labels and (unl := [k for k in source.keys if not source.labeled(k)]):
            raise ValidationError(f"{len(unl)} unlabeled patch(es) in a supervised dataset (e.g. {unl[0]})",
                                  hint="use PatchSource(kind='labeled') or RegionDataset(with_labels=False)")
        self.source, self.keys = source, list(source.keys)
        self.qc_policy, self.normalizer, self.augment, self.with_labels = qc_policy, normalizer or Normalizer("none"), augment, with_labels
        self.rng = np.random.default_rng(seed)
        self._cache: Dict[str, Sample] = {} if cache else None

    def __len__(self) -> int:
        return len(self.keys)

    def load_region(self, i: int) -> Sample:
        """Normalized numpy sample (labels/ignore under the QC policy), cached; no augmentation."""
        key = self.keys[i]
        if self._cache is not None and key in self._cache:
            return self._cache[key]
        rs = self.source.load(key)
        s: Sample = {"image": self.normalizer(rs.image, rs.meta.get("slide")),
                     "meta": {**rs.meta, "key": key, "box": rs.box.xyxy, "with_labels": self.with_labels}}
        if self.with_labels:
            s["labels"], s["ignore"] = apply_qc_policy(rs.labels, rs.flags, self.qc_policy)
            s["meta"].update(qc_policy=self.qc_policy, n_labels=int(len(np.unique(s["labels"])) - 1))
        if self._cache is not None:
            self._cache[key] = s
        return s

    def __getitem__(self, i: int) -> Dict[str, object]:
        s = self.load_region(i)
        if self.augment is not None:
            s = self.augment(s, self.rng)
        out: Dict[str, object] = {"meta": s["meta"], "image": torch.from_numpy(np.ascontiguousarray(s["image"], dtype=np.float32))}
        if self.with_labels:
            out["labels"] = torch.from_numpy(np.ascontiguousarray(s["labels"], dtype=np.int32))
            out["ignore"] = torch.from_numpy(np.ascontiguousarray(s["ignore"], dtype=bool))
        return out


def collate_samples(batch: List[Dict[str, object]]) -> Dict[str, object]:
    """Stack tensors when every item has the same shape, otherwise keep a list; ``meta`` is always a list."""
    out: Dict[str, object] = {}
    for key in batch[0]:
        vals = [b[key] for b in batch]
        out[key] = torch.stack(vals) if (key != "meta" and all(v.shape == vals[0].shape for v in vals)) else vals
    return out
