"""SAM 3 prompted with exemplar nuclei — few-shot concept detection, no training anywhere.

SAM 3's detector natively supports geometric exemplar prompts: boxes drawn on reference images become prompt tokens, and the
DETR-style head then detects *every* matching instance in a target image in one pass (no per-nucleus prompting). The prompt —
``n_exemplars`` ground-truth nuclei boxes from the training split, spread over several reference patches — is encoded once and
reused for every patch. ``prompt="text"`` uses a bare phrase (default "cell") instead: fully zero-shot. Model access goes
self-contained in ``_sam3_model`` (HF ``facebook/sam3``, MPS-patched); CLAHE contrast as for the other SAM baselines.
"""
from __future__ import annotations

import logging
from typing import List, Optional

import numpy as np

from .base import Baseline
from .sam2 import masks_to_instances

logger = logging.getLogger(__name__)


class Sam3Baseline(Baseline):
    """``mxseg eval --baseline sam3`` — few-shot: one cached prompt from ``n_exemplars`` training nuclei boxes."""

    name = "sam3"

    def __init__(self, prompt: str = "exemplars", n_exemplars: int = 8, detection_threshold: float = 0.5,
                 mask_threshold: float = 0.5, text: str = "cell", min_area: int = 15, max_area_frac: float = 0.1,
                 claimed_frac: float = 0.5, contrast: str = "clahe", exemplar_seed: int = 0,
                 exemplar_dataset: Optional[str] = None):
        if prompt not in ("exemplars", "text"):
            raise ValueError(f"prompt must be 'exemplars' (few-shot boxes) or 'text' (zero-shot), got {prompt!r}")
        if contrast not in ("clahe", "gamma", "none"):
            raise ValueError(f"contrast must be 'clahe', 'gamma' or 'none', got {contrast!r}")
        self.prompt, self.n_exemplars, self.detection_threshold, self.mask_threshold = prompt, n_exemplars, detection_threshold, mask_threshold
        self.text, self.min_area, self.max_area_frac, self.claimed_frac = text, min_area, max_area_frac, claimed_frac
        self.contrast, self.exemplar_seed, self.exemplar_dataset = contrast, exemplar_seed, exemplar_dataset
        self._prompt_feats = None
        self.exemplars_used: List[dict] = []                   # filled when the prompt is built: the boxes that supervise everything

    def params(self) -> dict:
        out = {"prompt": self.prompt, "detection_threshold": self.detection_threshold, "mask_threshold": self.mask_threshold,
               "min_area": self.min_area, "contrast": self.contrast}
        out.update({"n_exemplars": self.n_exemplars, "exemplar_seed": self.exemplar_seed} if self.prompt == "exemplars"
                   else {"text": self.text})
        return out

    @property
    def device(self) -> str:
        from ._sam3_model import device
        return device()

    def input_markers(self):
        """The markers of the ``input:`` block if the config has one — the protocol's shared marker image — else DAPI alone."""
        return super().input_markers() or ["DAPI"]

    def _rgb(self, image: np.ndarray, slide: Optional[str] = None) -> np.ndarray:
        from .sam2 import Sam2Baseline
        return Sam2Baseline._rgb(self, image, slide)           # identical preprocessing across the SAM baselines

    def _build_prompt(self):
        """[text] + geometry tokens of every exemplar box, encoded once from the training split (deterministic)."""
        import torch

        from mxdata import DEFAULT_DATASET, apply_qc_policy
        from mxdata.data import PatchSource

        from ._sam3_model import DUMMY_TEXT, as_pil, box_to_cxcywh_norm, load_sam3
        model, proc = load_sam3()
        DEVICE, DTYPE = self.device, torch.float32
        src = PatchSource(self.exemplar_dataset or DEFAULT_DATASET, split="train", kind="labeled", markers=self.input_markers())
        rng = np.random.default_rng(self.exemplar_seed)
        logger.info("building the SAM3 exemplar prompt: %d nuclei boxes from the training split (seed %d) ...", self.n_exemplars, self.exemplar_seed)
        feats = masks_ = None
        with torch.inference_mode():
            for key in src.keys[:: max(1, len(src.keys) // max(self.n_exemplars // 2, 1))]:
                if len(self.exemplars_used) >= self.n_exemplars:
                    break
                rs = src.load(key)
                labels, ignore = apply_qc_policy(rs.labels, rs.flags, "ignore_wrong")
                ids, counts = np.unique(labels[(labels > 0) & ~ignore], return_counts=True)
                good = ids[(counts > 150) & (counts < 1500)]
                if not good.size:
                    continue
                H, W = labels.shape
                boxes = []
                for i in rng.choice(good, size=min(2, len(good)), replace=False):
                    ys, xs = np.nonzero(labels == i)
                    boxes.append(box_to_cxcywh_norm((xs.min(), ys.min(), xs.max() + 1, ys.max() + 1), W, H))
                    self.exemplars_used.append({"patch": key, "label": int(i), "area_px": int((labels == i).sum())})
                ref = proc(images=as_pil(self._rgb(rs.image, rs.meta.get("slide"))), text=DUMMY_TEXT, return_tensors="pt").to(DEVICE)
                vo = model.vision_encoder(ref.pixel_values.to(DTYPE))
                if feats is None:
                    to = model.get_text_features(input_ids=ref.input_ids, attention_mask=ref.attention_mask, return_dict=True)
                    feats, masks_ = to.pooler_output, ref.attention_mask.bool()
                arr = np.asarray(boxes, np.float32).reshape(1, -1, 4)
                go = model.geometry_encoder(box_embeddings=torch.tensor(arr, dtype=DTYPE, device=DEVICE),
                                            box_mask=torch.ones(1, arr.shape[1], dtype=torch.bool, device=DEVICE),
                                            box_labels=torch.ones(1, arr.shape[1], dtype=torch.long, device=DEVICE),
                                            img_feats=vo.fpn_hidden_states[:-1], img_pos_embeds=vo.fpn_position_encoding[:-1])
                gm = go.attention_mask if go.attention_mask is not None else torch.ones(1, go.last_hidden_state.shape[1], dtype=torch.bool, device=DEVICE)
                feats = torch.cat([feats, go.last_hidden_state], 1)
                masks_ = torch.cat([masks_, gm], 1)
        if feats is None:
            raise RuntimeError("no usable exemplar nuclei found in the training split")
        logger.info("prompt ready: %d exemplar boxes over %d reference patches", len(self.exemplars_used), len({e['patch'] for e in self.exemplars_used}))
        return feats, masks_

    def _detect(self, rgb: np.ndarray):
        """One detection pass on a target patch with the cached prompt -> (masks (N, H, W) bool, scores (N,))."""
        import torch
        import torch.nn.functional as TF

        from ._sam3_model import DUMMY_TEXT, as_pil, load_sam3
        model, proc = load_sam3()
        DEVICE, DTYPE = self.device, torch.float32
        if self._prompt_feats is None:
            self._prompt_feats = self._build_prompt()
        combined, cmask = self._prompt_feats
        H, W = rgb.shape[:2]
        with torch.inference_mode():
            tgt = proc(images=as_pil(rgb), text=DUMMY_TEXT, return_tensors="pt").to(DEVICE)
            tv = model.vision_encoder(tgt.pixel_values.to(DTYPE))
            tfpn, tfpn_p = tv.fpn_hidden_states[:-1], tv.fpn_position_encoding[:-1]
            eo = model.detr_encoder(vision_features=[tfpn[-1]], text_features=combined, vision_pos_embeds=[tfpn_p[-1]], text_mask=cmask)
            do = model.detr_decoder(vision_features=eo.last_hidden_state, text_features=eo.text_features,
                                    vision_pos_encoding=eo.pos_embeds_flattened, text_mask=cmask, spatial_shapes=eo.spatial_shapes)
            logits = model.dot_product_scoring(decoder_hidden_states=do.intermediate_hidden_states,
                                               text_features=eo.text_features, text_mask=cmask).squeeze(-1)[-1]
            scores = logits.sigmoid() * do.presence_logits[-1].sigmoid()
            keep = scores[0] >= self.detection_threshold
            if int(keep.sum()) == 0:
                return np.zeros((0, H, W), bool), np.zeros((0,))
            mo = model.mask_decoder(decoder_queries=do.intermediate_hidden_states[-1], backbone_features=list(tfpn),
                                    encoder_hidden_states=eo.last_hidden_state, prompt_features=combined, prompt_mask=cmask)
            mu = TF.interpolate(mo.pred_masks[0, keep].unsqueeze(0).float(), size=(H, W), mode="bilinear", align_corners=False).squeeze(0)
            return (mu.sigmoid() > self.mask_threshold).cpu().numpy(), scores[0, keep].float().cpu().numpy()

    def predict(self, image: np.ndarray, pixel_size_um: Optional[float] = None, slide=None) -> np.ndarray:
        rgb = self._rgb(image, slide)
        if self.prompt == "text":
            from ._sam3_model import text_segment
            masks = text_segment(rgb, self.text, self.detection_threshold, self.mask_threshold)
            scores = np.linspace(1.0, 0.5, len(masks))
        else:
            masks, scores = self._detect(rgb)
        md = [{"segmentation": np.asarray(m, bool), "predicted_iou": float(s)} for m, s in zip(masks, scores)]
        if self.device == "mps":
            import torch
            torch.mps.empty_cache()
        return masks_to_instances(md, rgb.shape[:2], self.min_area, self.max_area_frac, self.claimed_frac)
