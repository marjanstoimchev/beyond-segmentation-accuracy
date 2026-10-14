"""Where samples come from — one small interface, two backends:

* :class:`PatchSource` — the fixed segmentation dataset (training);
* :class:`LiveSource`  — whole annotated FOVs read from the slide on demand (inspection / EDA).

Both are picklable (paths + metadata only) and open handles lazily, so they work inside DataLoader workers.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Protocol, Sequence, runtime_checkable

import numpy as np
import pandas as pd

from ..annotations.raster import NucleusFlag
from ..annotations.regions import RegionAnnotation
from ..errors import DatasetError
from ..io.geometry import Box
from ..io.slide import MxSlide
from ..pipeline.patches import read_image, read_patch
from ..pipeline.sample import TEXT_COLUMNS, read_dataset_json
from ..pipeline.stats import NormStats


@dataclass
class RegionSample:
    key: str
    image: np.ndarray        # (C, H, W) native dtype
    labels: np.ndarray       # (H, W) int32, every contour painted; QC policy not yet applied (zeros if unlabeled)
    flags: np.ndarray        # (n_labels + 1,) int8 NucleusFlag per label id
    box: Box
    meta: dict = field(default_factory=dict)


@runtime_checkable
class RegionSource(Protocol):
    keys: List[str]
    markers: List[str]
    stats: Dict[str, NormStats]

    def load(self, key: str) -> RegionSample: ...
    def labeled(self, key: str) -> bool: ...


class LiveSource:
    """Whole annotated FOVs of one slide, read and rasterized on demand."""

    def __init__(self, slide: MxSlide, regions: Sequence[RegionAnnotation], markers: Sequence[str] = ("DAPI",),
                 level: int = 0, stats: Optional[NormStats] = None):
        self.slide, self.markers, self.level = slide, list(markers), level
        self.stats = {slide.name: stats} if stats else {}
        self._regions = {r.key: r for r in sorted(regions, key=lambda r: r.region_id)}
        self.keys = list(self._regions)

    def labeled(self, key: str) -> bool:
        return True

    def region(self, key: str) -> RegionAnnotation:
        return self._regions[key]

    def load(self, key: str) -> RegionSample:
        r, ds = self._regions[key], self.slide.downsamples[self.level]
        image = self.slide.read_region(r.box, self.markers, self.level)
        labels = r.label_map(r.box, ds)
        h, w = min(image.shape[1], labels.shape[0]), min(image.shape[2], labels.shape[1])   # 1-px rounding at the edge
        return RegionSample(key, image[:, :h, :w], labels[:h, :w], r.flags, r.box,
                            {"slide": self.slide.name, "kind": "labeled", "labeled": True, "region_id": r.region_id, "has_qc": r.has_qc,
                             "markers": self.markers, "level": self.level, "downsample": ds, "pixel_size_um": self.slide.pixel_size_um})


class PatchSource:
    """Patches of the fixed segmentation dataset. ``split`` = "train" | "val" | "test" | None; ``kind`` = "labeled" (mask +
    QC flags) | "unlabeled" (image only) | None (both, for self-supervision); ``slides`` restricts to some slides."""

    def __init__(self, root: "str | Path", split: Optional[str] = None, kind: Optional[str] = "labeled",
                 slides: Optional[Sequence[str]] = None, markers: Optional[Sequence[str]] = None):
        """``markers`` = read only these (in this order) out of every patch folder; default all the dataset's markers."""
        if kind not in (None, "labeled", "unlabeled"):
            raise DatasetError(f"kind must be 'labeled', 'unlabeled' or None, got {kind!r}")
        if split not in (None, "train", "val", "test"):
            raise DatasetError(f"split must be 'train', 'val', 'test' or None, got {split!r}")
        self.root = Path(root)
        self.info = read_dataset_json(self.root)
        pt = pd.read_csv(self.root / "patches.csv", dtype={c: str for c in TEXT_COLUMNS})
        pt = pt[pt.kept.astype(bool)]
        if split is not None:
            pt = pt[pt.split == split]
        if kind is not None:
            pt = pt[pt.kind == kind]
        if slides is not None:
            pt = pt[pt.slide.isin(list(slides))]
        self.table = pt.reset_index(drop=True)
        self._rows: Dict[str, dict] = {r["key"]: r for r in self.table.to_dict("records")}
        self.keys = list(self._rows)
        self.markers, self.level = list(self.info["markers"]), int(self.info["level"])
        if markers is not None:
            stored = {m.lower(): m for m in self.markers}
            missing = [m for m in markers if m.lower() not in stored]
            if missing:
                raise DatasetError(f"{self.root}: markers {missing} are not stored (dataset has {self.markers})")
            self.markers = [stored[m.lower()] for m in markers]
        self._flags: Dict[str, np.ndarray] = {}
        if self.keys and (self.root / "instances.csv").exists():
            inst = pd.read_csv(self.root / "instances.csv")
            inst = inst[inst.patch_key.isin(self.keys)]
            code = {n: int(f) for n, f in NucleusFlag.__members__.items()}
            for pk, g in inst.groupby("patch_key", sort=False):
                lut = np.zeros(int(g.local_id.max()) + 1, dtype=np.int8)
                lut[g.local_id.to_numpy()] = [code[f] for f in g.flag]
                self._flags[pk] = lut
        self.stats = {s: NormStats.load(self.root / s / "stats.json") for s in self.info.get("slides", []) if (self.root / s / "stats.json").exists()}

    def row(self, key: str) -> dict:
        return self._rows[key]

    def labeled(self, key: str) -> bool:
        return self._rows[key]["kind"] == "labeled"

    def box(self, key: str) -> Box:
        r = self._rows[key]
        return Box(r["x0"], r["y0"], r["x1"], r["y1"])

    def core(self, key: str) -> Box:
        r = self._rows[key]
        return Box(r["core_x0"], r["core_y0"], r["core_x1"], r["core_y1"])

    def load(self, key: str) -> RegionSample:
        r = self._rows[key]
        meta = {"slide": r["slide"], "kind": r["kind"], "labeled": self.labeled(key), "region_id": int(r["region_id"]), "has_qc": bool(r["has_qc"]),
                "markers": self.markers, "level": self.level, "downsample": float(r["downsample"]), "pixel_size_um": r["pixel_size_um"],
                "split": r["split"], "group": r["group"], "core": self.core(key).xyxy, "row": int(r["row"]), "col": int(r["col"]),
                "tissue_fraction": float(r["tissue_fraction"])}
        if self.labeled(key):
            if not r["mask"]:
                raise DatasetError(f"{key}: labeled patch without a mask path in patches.csv")
            image, labels = read_patch(self.root / r["image"], self.root / r["mask"], self.markers)
            flags = self._flags.get(key, np.zeros(1, np.int8))
            if labels.size and int(labels.max()) >= len(flags):
                raise DatasetError(f"{key}: mask has label {int(labels.max())} but instances.csv lists {len(flags) - 1}")
        else:
            image = read_image(self.root / r["image"], self.markers)
            labels, flags = np.zeros(image.shape[1:], np.int32), np.zeros(1, np.int8)
        return RegionSample(key, image, labels, flags, self.box(key), meta)
