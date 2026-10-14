"""Standalone SAM 3 access (HF transformers): model loading, text segmentation, box helpers.

Self-contained so this repository does not depend on any other project (adapted from the author's ``sam3_utils``).
Weights: ``facebook/sam3`` from the local HF cache (offline mode enforced). On Apple Silicon the unsupported
``roi_align`` op inside the geometry encoder is patched to run on CPU; everything else stays on MPS, float32.
"""
from __future__ import annotations

import os

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.0")

import numpy as np

SAM3_MODEL_ID = "facebook/sam3"
DUMMY_TEXT = "visual"        # canonical sentinel meaning "rely on the geometric prompt"

_model = None
_proc = None


def device() -> str:
    import torch
    return "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")


def as_pil(img):
    from PIL import Image
    return img if isinstance(img, Image.Image) else Image.fromarray(np.asarray(img))


def box_to_cxcywh_norm(box_xyxy, W: int, H: int) -> list:
    """``(x1, y1, x2, y2)`` px -> normalized ``[cx, cy, w, h]`` for SAM 3 geometry prompts."""
    x1, y1, x2, y2 = box_xyxy
    return [((x1 + x2) / 2) / W, ((y1 + y2) / 2) / H, (x2 - x1) / W, (y2 - y1) / H]


def load_sam3():
    """Load ``Sam3Model`` + ``Sam3Processor`` once; on MPS, patch the geometry encoder's roi_align to run on CPU."""
    global _model, _proc
    if _model is not None:
        return _model, _proc
    import torch
    import torchvision
    from transformers import Sam3Model, Sam3Processor
    from transformers.models.sam3.modeling_sam3 import box_cxcywh_to_xyxy

    dev = device()
    m = Sam3Model.from_pretrained(SAM3_MODEL_ID, dtype=torch.float32).to(dev).eval()
    p = Sam3Processor.from_pretrained(SAM3_MODEL_ID)

    if dev == "mps":
        def _encode_boxes_cpu_roi(self, boxes, boxes_mask, boxes_labels, vision_features):
            bsz, nb = boxes.shape[:2]
            h, w = vision_features.shape[-2:]
            emb = self.boxes_direct_project(boxes)
            bx = box_cxcywh_to_xyxy(boxes) * torch.tensor([w, h, w, h], dtype=boxes.dtype, device=boxes.device).view(1, 1, 4)
            vf = vision_features.detach().to("cpu", dtype=torch.float32)
            sampled = torchvision.ops.roi_align(vf, list(bx.detach().to("cpu", torch.float32).unbind(0)), self.roi_size)
            sampled = sampled.to(vision_features.device, dtype=vision_features.dtype)
            emb = emb + self.boxes_pool_project(sampled).view(bsz, nb, self.hidden_size)
            cx, cy, bw, bh = boxes.unbind(-1)
            pe = self._encode_box_coordinates(cx.flatten(), cy.flatten(), bw.flatten(), bh.flatten())
            emb = emb + self.boxes_pos_enc_project(pe.view(bsz, nb, pe.shape[-1]))
            return self.label_embed(boxes_labels.long()) + emb, boxes_mask

        m.geometry_encoder._encode_boxes = _encode_boxes_cpu_roi.__get__(m.geometry_encoder, type(m.geometry_encoder))
    _model, _proc = m, p
    return m, p


def text_segment(rgb, text: str = "cell", detection_threshold: float = 0.4, mask_threshold: float = 0.5) -> np.ndarray:
    """Concept/text-prompt segmentation: instance masks ``(N, H, W)`` uint8 for the phrase ``text``.

    A bare noun works best on DAPI ("cell"); "a cell" and "nucleus" return zero detections (measured)."""
    import torch
    model, proc = load_sam3()
    pil = as_pil(rgb)
    W, H = pil.size
    with torch.inference_mode():
        inp = proc(images=pil, text=text, return_tensors="pt").to(device())
        out = model(**inp)
        r = proc.post_process_instance_segmentation(out, threshold=detection_threshold, mask_threshold=mask_threshold,
                                                    target_sizes=[(H, W)])[0]
    if r["scores"] is None or len(r["scores"]) == 0:
        return np.zeros((0, H, W), np.uint8)
    return r["masks"].detach().cpu().numpy().astype(np.uint8)
