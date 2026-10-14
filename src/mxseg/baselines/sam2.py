"""SAM 2.1 prompted with exemplar nuclei — few-shot segmentation, no training anywhere.

A prototype embedding is averaged from ``n_exemplars`` ground-truth nuclei of the training split (built once,
deterministically); each patch is encoded once by SAM2, the cosine similarity between its features and the prototype is
taken, and the similarity peaks become box prompts (``box_scale`` x the exemplars' median radius). Decoded masks below
``pred_iou_thresh``, outside a nucleus-plausible area window, or mostly duplicating a kept mask are dropped. CLAHE contrast
by default — without it SAM sees almost nothing on raw DAPI. The only supervision is the few exemplars.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional, Sequence

import numpy as np

from ..metrics import relabel
from .base import Baseline

logger = logging.getLogger(__name__)

CONFIGS = {"sam2.1_hiera_tiny": "configs/sam2.1/sam2.1_hiera_t.yaml", "sam2.1_hiera_small": "configs/sam2.1/sam2.1_hiera_s.yaml",
           "sam2.1_hiera_base_plus": "configs/sam2.1/sam2.1_hiera_b+.yaml", "sam2.1_hiera_large": "configs/sam2.1/sam2.1_hiera_l.yaml"}


def masks_to_instances(masks: Sequence[dict], shape, min_area: int = 15, max_area_frac: float = 0.1,
                       claimed_frac: float = 1.0) -> np.ndarray:
    """Mask dicts -> int32 instance map: masks painted in descending predicted IoU (first wins), nucleus-implausible masks
    dropped (smaller than ``min_area`` px or larger than ``max_area_frac`` of the image — tissue, not a nucleus). A mask whose
    pixels are already claimed beyond ``claimed_frac`` is a duplicate of a kept mask and is skipped entirely."""
    out = np.zeros(shape, np.int32)
    next_id = 1
    lim = max_area_frac * shape[0] * shape[1]
    for m in sorted(masks, key=lambda m: -float(m.get("predicted_iou", 0.0))):
        seg = np.asarray(m["segmentation"], bool)
        area = seg.sum()
        if not (min_area <= area <= lim) or (seg & (out > 0)).sum() > claimed_frac * area:
            continue
        sel = seg & (out == 0)
        if sel.sum() >= min_area:
            out[sel] = next_id
            next_id += 1
    return relabel(out)[0].astype(np.int32)


def pick_sam2_device() -> str:
    import torch
    return "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")


class Sam2Baseline(Baseline):
    """``mxseg eval --baseline sam2`` — few-shot: prompted by a prototype of ``n_exemplars`` training nuclei."""

    name = "sam2"

    def __init__(self, checkpoint: str = "sam2.1_hiera_small", n_exemplars: int = 8, sim_threshold: float = 0.7,
                 peak_min_distance: int = 6, box_scale: float = 2.4, pred_iou_thresh: float = 0.7, min_area: int = 15,
                 max_mask_area: int = 3000, max_area_frac: float = 0.1, claimed_frac: float = 0.5, contrast: str = "clahe",
                 exemplar_seed: int = 0, exemplar_dataset: Optional[str] = None, points_per_batch: int = 128,
                 device: Optional[str] = None):
        if contrast not in ("clahe", "gamma", "none"):
            raise ValueError(f"contrast must be 'clahe', 'gamma' or 'none', got {contrast!r}")
        self.checkpoint, self.n_exemplars, self.sim_threshold, self.peak_min_distance = checkpoint, n_exemplars, sim_threshold, peak_min_distance
        self.box_scale, self.pred_iou_thresh, self.min_area = box_scale, pred_iou_thresh, min_area
        self.max_mask_area, self.max_area_frac, self.claimed_frac, self.contrast = max_mask_area, max_area_frac, claimed_frac, contrast
        self.exemplar_seed, self.exemplar_dataset, self.points_per_batch = exemplar_seed, exemplar_dataset, points_per_batch
        self._device = device
        self._pred = None
        self._proto = None
        self.exemplars_used: List[dict] = []                   # filled by _prototype: the few nuclei that supervise everything

    def params(self) -> dict:
        return {"checkpoint": self.checkpoint, "n_exemplars": self.n_exemplars, "sim_threshold": self.sim_threshold,
                "pred_iou_thresh": self.pred_iou_thresh, "box_scale": self.box_scale, "min_area": self.min_area,
                "contrast": self.contrast, "exemplar_seed": self.exemplar_seed}

    @property
    def device(self) -> str:
        return self._device or pick_sam2_device()

    def predictor(self):
        """One ``SAM2ImagePredictor``, built once — rebuilding per call re-creates its jit-scripted transforms every time."""
        if self._pred is None:
            import sam2 as sam2_pkg
            from sam2.build_sam import build_sam2
            from sam2.sam2_image_predictor import SAM2ImagePredictor
            ckpt = Path(self.checkpoint)
            if not ckpt.is_file():
                if self.checkpoint not in CONFIGS:
                    raise ValueError(f"unknown sam2 checkpoint {self.checkpoint!r}: use one of {sorted(CONFIGS)} or a .pt path")
                ckpt = Path(sam2_pkg.__file__).parent.parent / "checkpoints" / f"{self.checkpoint}.pt"
                if not ckpt.is_file():
                    raise FileNotFoundError(f"sam2 checkpoint not found: {ckpt}")
            self._pred = SAM2ImagePredictor(build_sam2(CONFIGS.get(self.checkpoint, self.checkpoint), str(ckpt), device=self.device))
        return self._pred

    def input_markers(self):
        """The markers of the ``input:`` block if the config has one — the protocol's shared marker image, the same one the
        trained models see — else the single nuclear channel this baseline used before."""
        return super().input_markers() or ["DAPI"]

    def _rgb(self, image: np.ndarray, slide: Optional[str] = None) -> np.ndarray:
        """The 8-bit RGB SAM is shown. With an ``input:`` block the image is the protocol's marker composite on the slide's
        stored statistics — byte for byte the input the trained models receive, so every row of the benchmark sees the same
        picture. Without one it falls back to the single nuclear channel, per-image percentile-normalized. ``contrast``
        applies to either, but note that equalizing a composite rescales its channels independently and so changes the
        marker ratios the image encodes: leave it ``none`` whenever an ``input:`` block is in play."""
        if self.input is not None:
            x = np.clip(self.input.render(image, slide), 0, 1)            # (3, H, W) float in [0, 1]
        else:
            from .unet import percentile_normalize
            x = np.repeat(np.clip(percentile_normalize(image[:1]), 0, 1), 3, axis=0)
        if self.contrast == "clahe":                          # SAM expects natural-image contrast; raw DAPI is too dark for it
            from skimage import exposure
            x = np.stack([exposure.equalize_adapthist(c, clip_limit=0.02) for c in x])
        elif self.contrast == "gamma":
            x = x ** 0.5
        return np.moveaxis((np.clip(x, 0, 1) * 255).astype(np.uint8), 0, -1)

    def _embed(self, rgb: np.ndarray):
        """Set the image on the predictor and return its L2-normalized encoder embedding (256, 64, 64)."""
        import torch.nn.functional as F
        pred = self.predictor()
        pred.set_image(rgb)
        return F.normalize(pred._features["image_embed"][0].float(), dim=0)

    def _prototype(self):
        """(prototype embedding, prompt sigma): the mean embedding of ``n_exemplars`` ground-truth nuclei of the training
        split, chosen deterministically (``exemplar_seed``). ``exemplars_used`` records exactly which nuclei supervise."""
        if self._proto is None:
            logger.info("building the exemplar prototype: %d nuclei from the training split (seed %d) ...", self.n_exemplars, self.exemplar_seed)
            import torch
            import torch.nn.functional as F

            from mxdata import DEFAULT_DATASET, apply_qc_policy
            from mxdata.data import PatchSource
            src = PatchSource(self.exemplar_dataset or DEFAULT_DATASET, split="train", kind="labeled", markers=self.input_markers())
            rng = np.random.default_rng(self.exemplar_seed)
            protos, radii = [], []
            for key in src.keys[:: max(1, len(src.keys) // max(self.n_exemplars // 2, 1))]:
                if len(protos) >= self.n_exemplars:
                    break
                rs = src.load(key)
                labels, ignore = apply_qc_policy(rs.labels, rs.flags, "ignore_wrong")
                fm = self._embed(self._rgb(rs.image, rs.meta.get("slide")))
                ids, counts = np.unique(labels[(labels > 0) & ~ignore], return_counts=True)
                good = ids[(counts > 150) & (counts < 1500)]
                for i in rng.choice(good, size=min(2, len(good)), replace=False):
                    m = torch.from_numpy((labels == i).astype(np.float32))[None, None]
                    md = F.interpolate(m, size=fm.shape[-2:], mode="area")[0, 0] > 0.25
                    if md.sum():
                        protos.append(F.normalize(fm[:, md.to(fm.device)].mean(1), dim=0))
                        radii.append(float(np.sqrt((labels == i).sum() / np.pi)))
                        self.exemplars_used.append({"patch": key, "label": int(i), "radius_px": radii[-1]})
            if not protos:
                raise RuntimeError("no usable exemplar nuclei found in the training split")
            self._proto = (F.normalize(torch.stack(protos).mean(0), dim=0), float(np.median(radii)) / np.sqrt(2.0))
            logger.info("prototype ready: %d exemplars, median radius %.1f px", len(protos), float(np.median(radii)))
        return self._proto

    def similarity(self, rgb: np.ndarray) -> np.ndarray:
        """Cosine similarity of every pixel to the exemplar prototype (the image stays set for decoding)."""
        import torch
        import torch.nn.functional as F
        proto, _ = self._prototype()
        fm = self._embed(rgb)
        sim = torch.einsum("chw,c->hw", fm, proto.to(fm.device))
        return F.interpolate(sim[None, None], size=rgb.shape[:2], mode="bilinear")[0, 0].cpu().numpy()

    def _blobs(self, sim: np.ndarray) -> np.ndarray:
        """Candidate nuclei ``(x, y, sigma)``: peaks of the similarity map, at the exemplars' scale."""
        from skimage.feature import peak_local_max
        _, sigma = self._prototype()
        peaks = peak_local_max(sim, min_distance=self.peak_min_distance, threshold_abs=self.sim_threshold)
        return np.stack([peaks[:, 1], peaks[:, 0], np.full(len(peaks), sigma)], 1) if len(peaks) else np.zeros((0, 3))

    def _box_masks(self, rgb: np.ndarray, blobs: np.ndarray) -> List[dict]:
        """One decoded mask per candidate, prompted with a box of ``box_scale`` x its radius; keep it when nucleus-sized and
        above ``pred_iou_thresh``. The image must already be set (``similarity`` sets it)."""
        import torch
        pred = self.predictor()
        H, W = rgb.shape[:2]
        out: List[dict] = []
        for b in range(0, len(blobs), self.points_per_batch):
            chunk = blobs[b:b + self.points_per_batch]
            pts = torch.as_tensor(chunk[:, :2], dtype=torch.float32, device=pred.device)
            r = torch.as_tensor(chunk[:, 2], dtype=torch.float32, device=pred.device) * self.box_scale
            boxes = torch.stack([pts[:, 0] - r, pts[:, 1] - r, pts[:, 0] + r, pts[:, 1] + r], 1)
            tb = pred._transforms.transform_boxes(boxes, normalize=True, orig_hw=(H, W))
            masks, ious, _ = pred._predict(None, None, boxes=tb, multimask_output=False, return_logits=False)
            masks, ious = masks.cpu().numpy() > 0, ious.cpu().numpy()
            areas = masks.sum((2, 3))
            keep = (areas[:, 0] >= self.min_area) & (areas[:, 0] <= self.max_mask_area) & (ious[:, 0] >= self.pred_iou_thresh)
            out.extend({"segmentation": masks[k, 0], "predicted_iou": float(ious[k, 0])} for k in np.nonzero(keep)[0])
        return out

    def predict(self, image: np.ndarray, pixel_size_um: Optional[float] = None, slide=None) -> np.ndarray:
        rgb = self._rgb(image, slide)
        blobs = self._blobs(self.similarity(rgb))
        masks = self._box_masks(rgb, blobs) if len(blobs) else []
        if self.device == "mps":
            import torch
            torch.mps.empty_cache()                           # per-patch prompt counts vary; without this the MPS cache pool
        return masks_to_instances(masks, rgb.shape[:2], self.min_area, self.max_area_frac, self.claimed_frac)   # grows unbounded and stalls the machine
