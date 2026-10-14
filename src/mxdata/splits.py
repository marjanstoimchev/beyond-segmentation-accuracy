"""Overlap-aware train / val / test splits over annotated regions (torch-free).

FOVs of a slide can overlap and then carry near-duplicate annotations, so splitting is done on connected
components of the per-slide overlap graph, never on single regions. With several slides the honest
protocol is leave-slide-out (``val_slides``). Regions are addressed by key ``<slide>_R<id:02d>``.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from .config import SplitConfig
from .errors import ValidationError
from .io.geometry import Box


@dataclass(frozen=True)
class RegionInfo:
    """What the splitter needs to know about a region (a ``RegionAnnotation`` provides the same attributes)."""

    key: str
    slide: str
    region_id: int
    box: Box
    n_nuclei: int
    has_qc: bool

    def __len__(self) -> int:
        return self.n_nuclei


def overlap_groups(regions: Sequence) -> List[List[str]]:
    """Connected components of "same slide and boxes overlap" -> lists of region keys (largest first)."""
    n = len(regions)
    parent = list(range(n))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i in range(n):
        for j in range(i + 1, n):
            if regions[i].slide == regions[j].slide and regions[i].box.overlaps(regions[j].box):
                parent[find(i)] = find(j)
    comps: Dict[int, List[str]] = {}
    for i in range(n):
        comps.setdefault(find(i), []).append(regions[i].key)
    return sorted((sorted(c) for c in comps.values()), key=lambda c: (-len(c), c))


SPLITS = ("train", "val", "test")


def _pinned(groups, by_key, regs, region_ids, slides, what: str):
    """Group indices pinned by ``<what>_slides`` (whole slides) or ``<what>_regions`` (a list: ids in every slide; a mapping:
    ids per slide; overlap partners are pulled in); ``None`` when nothing is pinned."""
    if slides:
        if unknown := set(slides) - {r.slide for r in regs}:
            raise ValidationError(f"{what}_slides {sorted(unknown)} are not among the slides {sorted({r.slide for r in regs})}")
        return {i for i, g in enumerate(groups) if by_key[g[0]].slide in set(slides)}
    if region_ids is not None:
        per_slide = dict(region_ids) if isinstance(region_ids, dict) else {s: region_ids for s in {r.slide for r in regs}}
        for s, ids in per_slide.items():
            have = {r.region_id for r in regs if r.slide == s}
            if not have:
                raise ValidationError(f"{what}_regions names slide {s!r}, which has no regions here ({sorted({r.slide for r in regs})})")
            if unknown := set(ids) - have:
                raise ValidationError(f"{what}_regions {sorted(unknown)} are not available regions of {s}")
        return {i for i, g in enumerate(groups) if any(by_key[k].region_id in set(per_slide.get(by_key[k].slide, ())) for k in g)}
    return None


def split_regions(regions: Sequence, cfg: Optional[SplitConfig] = None) -> Dict[str, List[str]]:
    """``{"train": [keys], "val": [keys], "test": [keys]}`` without separating overlapping regions (test empty unless pinned)."""
    cfg = cfg or SplitConfig()
    regs = [r for r in regions if r.has_qc or not cfg.require_qc]
    if not regs:
        raise ValidationError("no regions left to split" + (" (require_qc=True but none has a QC table)" if cfg.require_qc else ""))
    groups = overlap_groups(regs)
    by_key = {r.key: r for r in regs}
    test_g = _pinned(groups, by_key, regs, cfg.test_regions, cfg.test_slides, "test") or set()
    val_g = _pinned(groups, by_key, regs, cfg.val_regions, cfg.val_slides, "val")
    if val_g is None:                                             # draw val from what is not test
        weight = [sum(len(by_key[k]) for k in g) for g in groups]
        free = [i for i in range(len(groups)) if i not in test_g]
        target = cfg.val_frac * sum(weight[i] for i in free)
        val_g, acc = set(), 0.0
        for i in np.random.default_rng(cfg.seed).permutation(free):
            if acc >= target:
                break
            if acc > 0 and acc + weight[i] > 1.5 * target:      # don't overshoot wildly
                continue
            val_g.add(int(i))
            acc += weight[i]
    if both := val_g & test_g:
        raise ValidationError(f"regions in both val and test (overlap partners included): {sorted(k for i in both for k in groups[i])}")
    pick = lambda idx: sorted(k for i in idx for k in groups[i])   # noqa: E731
    return {"train": pick(i for i in range(len(groups)) if i not in val_g | test_g), "val": pick(val_g), "test": pick(test_g)}


def assign_splits(patches: pd.DataFrame, cfg: Optional[SplitConfig] = None):
    """Overlap group + split per region key, from a ``patches.csv``-like table of *labeled* windows.

    Returns ``(group_of_region_key, split_of_region_key)``; group ids are ``<slide>/g<k>``.
    """
    regs = []
    for rk, g in patches.groupby("region_key", sort=True):
        kept = g[g.kept.astype(bool)]
        n = int((kept if len(kept) else g).n_owned.sum())
        regs.append(RegionInfo(rk, str(g.slide.iloc[0]), int(g.region_id.iloc[0]),
                               Box(g.x0.min(), g.y0.min(), g.x1.max(), g.y1.max()), max(n, 1), bool(g.has_qc.iloc[0])))
    group_of: Dict[str, str] = {}
    counter: Dict[str, int] = {}
    for comp in overlap_groups(regs):
        slide = comp[0].rsplit("_R", 1)[0]
        counter[slide] = counter.get(slide, 0) + 1
        group_of.update({k: f"{slide}/g{counter[slide] - 1}" for k in comp})
    split = split_regions(regs, cfg)
    split_of = {k: side for side in SPLITS for k in split[side]}
    for r in regs:
        split_of.setdefault(r.key, "unused")           # e.g. require_qc dropped it
    return group_of, split_of
