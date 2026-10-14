"""Whole-region prediction from tiles of the training geometry (256² on a 192-px grid): every tile is normalized and predicted
exactly as a training patch, then predicted objects are stitched by ownership — an object is taken, whole, from the tile whose
core holds its centroid. The result is one instance map per region, scored like any ground-truth map."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Optional

import numpy as np

from .metrics import centroids


@dataclass(frozen=True)
class Tile:
    y0: int
    x0: int
    y1: int
    x1: int
    core: tuple            # (y0, x0, y1, x1) in region coordinates: cores partition the region


def _starts(n: int, size: int, stride: int) -> List[int]:
    if n <= size:
        return [0]
    s = list(range(0, n - size, stride))
    return s + [n - size]                                   # last tile shifted to the edge


def _bounds(starts: List[int], size: int, n: int) -> List[int]:
    """Core boundaries: the region edges and the midpoint of every overlap, so each core sits inside its tile with a margin of
    half the overlap (32 px for 256 / 192) on both sides — an object owned by a tile is whole in that tile."""
    return [0] + [(a + size + b) // 2 for a, b in zip(starts[:-1], starts[1:])] + [n]


def tile_grid(height: int, width: int, size: int = 256, stride: int = 192) -> List[Tile]:
    """Tiles covering the region (last row / column shifted to the edge) with their owned cores, which partition the region."""
    ys, xs = _starts(height, size, stride), _starts(width, size, stride)
    by, bx = _bounds(ys, size, height), _bounds(xs, size, width)
    return [Tile(y0, x0, min(y0 + size, height), min(x0 + size, width), (by[i], bx[j], by[i + 1], bx[j + 1]))
            for i, y0 in enumerate(ys) for j, x0 in enumerate(xs)]


def predict_tiled(predict_batch: Callable, image: np.ndarray, size: int = 256, stride: int = 192, batch_size: int = 8,
                  pixel_size_um: Optional[float] = None, progress: bool = False, desc: str = "tiles") -> np.ndarray:
    """``image`` (C, H, W) native dtype -> int32 instance map (H, W). ``predict_batch(images, pixel_size_um)`` is any baseline's."""
    from tqdm import tqdm
    _, H, W = image.shape
    tiles = tile_grid(H, W, size, stride)
    out = np.zeros((H, W), np.int32)
    next_id = 1
    bar = tqdm(total=len(tiles), unit="tile", desc=desc, disable=not progress, leave=False, dynamic_ncols=True)
    for b in range(0, len(tiles), batch_size):
        chunk = tiles[b:b + batch_size]
        preds = predict_batch([image[:, t.y0:t.y1, t.x0:t.x1] for t in chunk], pixel_size_um)
        for t, pred in zip(chunk, preds):
            pred = np.asarray(pred).astype(np.int32)
            if not pred.any():
                continue
            c = centroids(pred)                                          # (K, 2) = (x, y) in tile coordinates
            cy, cx = c[:, 1] + t.y0, c[:, 0] + t.x0
            cy0, cx0, cy1, cx1 = t.core
            owned = np.nonzero((cy >= cy0) & (cy < cy1) & (cx >= cx0) & (cx < cx1))[0] + 1
            if not owned.size:
                continue
            lut = np.zeros(int(pred.max()) + 1, np.int32)
            lut[owned] = np.arange(next_id, next_id + owned.size)
            next_id += owned.size
            target = out[t.y0:t.y1, t.x0:t.x1]
            sel = (lut[pred] > 0) & (target == 0)                          # paste whole objects; never overwrite a neighbour's
            target[sel] = lut[pred][sel]
        bar.update(len(chunk))
    bar.close()
    return out
