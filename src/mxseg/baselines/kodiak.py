"""KODIAK segmenter: the pretrained ViT-S/16 encoder + a light upsampling head, trained by ``mxseg train`` (``model: kodiak``).

The encoder is KODIAK's own ``DinoV3Encoder`` (the exact DINOv3 backbone the method pretrains); the head turns its patch
tokens into the same 3-class map (background / nucleus / boundary) the U-Net predicts, so instances come from the
identical watershed and the comparison isolates the representation. ``load_encoder_weights`` accepts either a KODIAK
pretraining checkpoint (the EMA teacher is taken, the standard DINO evaluation choice) or the raw DINOv3 ``.pth`` — and
refuses anything that does not cover the encoder completely, so weights are never silently half-loaded. The input is the
marker image of the pretraining (``mxseg.inputs``: CD45, PanCK, DAPI as R, G, B, stored with the weights) — or, for older
weights, a percentile-normalized single marker replicated to 3 channels; the network applies the ImageNet statistics KODIAK
pretrains with. Weights: ``runs/<name>/weights.pt`` -> ``--baseline kodiak --param weights=...``.
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Sequence

import numpy as np

from .base import Baseline
from .unet import pick_device, segment

WEIGHTS_FORMAT = 3                              # 3: UNETR head (multi-depth ViT taps + image stem; a ViT/16 grid alone cannot draw 2 px boundaries)
KIND = "kodiak_segmenter"
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
DEFAULT_ARCH = {"patch_size": 16, "embed_dim": 384, "depth": 12, "num_heads": 6, "num_storage_tokens": 4,   # DINOv3 ViT-S/16
                "mlp_ratio": 4.0,                                            # ViT-B/16: embed_dim 768, num_heads 12 (rest unchanged)
                "head_channels": 192, "stem_channels": 32, "classes": 3, "drop_path_rate": 0.0,
                "encoder_upsample": 1,                                       # 2 = encode at 2x resolution (8 px/token, ~4x compute)
                "objective": "threeclass",                                   # "threeclass" -> watershed | "flows" -> Cellpose decoding
                "bottleneck": "final"}                                       # "multi" = blocks 5+8+final fused (the final block is the
#                                                                              worst instance-separability layer in every encoder, measured)


def build_segmenter(arch: Optional[dict] = None):
    """The network: KODIAK's DINOv3 ViT encoder + a UNETR head (Hatamizadeh et al., WACV 2022) — the standard decoder
    for plain-ViT dense prediction. Intermediate transformer depths (1/4, 1/2, 3/4) are reassembled into a feature
    pyramid with deconv chains, the final tokens form the bottleneck, and a conv stem on the input image supplies the
    full-resolution edge skip a 16 px token grid cannot carry (nuclei are 10-20 px, the boundary class 2 px). Measured
    without any skips: boundary recall 19 %, touching nuclei fuse; final-layer + image stem alone: 48 %."""
    import torch
    import torch.nn as nn

    from kodiak.components import DinoV3Encoder

    a = {**DEFAULT_ARCH, **(arch or {})}

    def block(c_in, c_out):
        return nn.Sequential(nn.Conv2d(c_in, c_out, 3, padding=1), nn.GroupNorm(8, c_out), nn.GELU())

    def deconv(c_in, c_out):
        return nn.Sequential(nn.ConvTranspose2d(c_in, c_out, 2, stride=2), nn.GroupNorm(8, c_out), nn.GELU())

    class Head(nn.Module):
        """UNETR decoder: 3 reassembled ViT depths + image stem as skips, 4 fused upsampling stages (~1.6M parameters)."""

        def __init__(self):
            super().__init__()
            sc, chans = a["stem_channels"], []
            self.stem = block(3, sc // 2)                                    # full-resolution edges from the image itself
            skip_ch = [max(a["head_channels"] >> i, 16) for i in range(3)]   # z(3/4) -> H/8, z(1/2) -> H/4, z(1/4) -> H/2
            self.reassemble = nn.ModuleList(
                nn.Sequential(nn.Conv2d(a["embed_dim"], c, 1), *[deconv(c, c) for _ in range(n)])
                for c, n in zip(skip_ch, (1, 2, 3)))
            c_in = a["embed_dim"]
            for i in range(4):
                c_out = max(a["head_channels"] >> i, 16)
                chans.append((c_in, c_out))
                c_in = c_out
            skip = skip_ch + [sc // 2]
            self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
            self.stages = nn.ModuleList(block(ci + skip[i], co) for i, (ci, co) in enumerate(chans))
            self.out = nn.Conv2d(c_in, a["classes"], 1)
            D = a["embed_dim"]                                               # multi-bottleneck: read the mid-network layers too,
            self.bfuse = (nn.Sequential(nn.Conv2d(3 * D, D, 1), nn.GroupNorm(8, D), nn.GELU())
                          if a["bottleneck"] == "multi" else None)           # not just the collapsed final block

        def forward(self, f, taps, x):
            if self.bfuse is not None:
                f = self.bfuse(torch.cat([taps[1], taps[2], f], dim=1))      # blocks 1/2 and 3/4 depth + final tokens
            skips = [r(t) for r, t in zip(self.reassemble, taps[::-1])] + [self.stem(x)]   # deepest tap fuses first
            for stage, s in zip(self.stages, skips):
                if f.shape[-1] < s.shape[-1]:
                    f = self.up(f)
                f = stage(torch.cat([f, s], dim=1))
            return self.out(f)

    class KodiakSegmenter(nn.Module):
        def __init__(self):
            super().__init__()
            self.arch = dict(a)
            self.backbone = DinoV3Encoder(patch_size=a["patch_size"], embed_dim=a["embed_dim"], depth=a["depth"],
                                          num_heads=a["num_heads"], mlp_ratio=a["mlp_ratio"],
                                          num_storage_tokens=a["num_storage_tokens"],
                                          drop_path_rate=a["drop_path_rate"])
            d = a["depth"]
            self.taps = sorted({max(0, d * k // 4 - 1) for k in (1, 2, 3)})  # depth 12: blocks 2, 5, 8 (UNETR's z3, z6, z9)
            self.head = Head()
            self.register_buffer("in_mean", torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1), persistent=False)
            self.register_buffer("in_std", torch.tensor(IMAGENET_STD).view(1, 3, 1, 1), persistent=False)

        def _encode(self, x):
            """The backbone's own forward pass (identical math to forward_full), returning the tapped depths too."""
            bb = self.backbone
            t, (h, w) = bb.prepare_tokens(x)
            rope = bb.rope_embed(H=h, W=w)
            n_prefix = 1 + (bb.n_storage_tokens if bb.storage_tokens is not None else 0)
            taps = []
            for i, blk in enumerate(bb.blocks):
                t = blk(t, rope)
                if i in self.taps:
                    taps.append(t[:, n_prefix:])
            while len(taps) < 3:                                             # tiny test depths: reuse the single tap
                taps.append(taps[-1])
            return bb.norm(t)[:, n_prefix:], taps, (h, w)

        def forward(self, x):
            if x.shape[1] == 1:
                x = x.repeat(1, 3, 1, 1)                                     # single marker -> the 3 channels of pretraining
            elif x.shape[1] != 3:
                raise ValueError(f"kodiak segmenter takes 1- or 3-channel images, got {x.shape[1]}")
            x = (x.clamp(0.0, 1.0) - self.in_mean) / self.in_std             # exactly the pretraining input statistics
            B, _, H, W = x.shape
            up = self.arch["encoder_upsample"]
            xe = torch.nn.functional.interpolate(x, scale_factor=up, mode="bilinear", align_corners=False) if up > 1 else x
            tokens, taps, (h, w) = self._encode(xe)                          # dynamic RoPE: any grid size works
            D = self.arch["embed_dim"]                                       # .contiguous(): MPS conv rejects strided views
            f = tokens.transpose(1, 2).reshape(B, D, h, w).contiguous()
            taps = [t.transpose(1, 2).reshape(B, D, h, w).contiguous() for t in taps]
            return self.head(f, taps, x)

    return KodiakSegmenter()


def describe_encoder(state) -> str:
    """The architecture a checkpoint's encoder tensors hold, phrased like the ``kodiak:`` keys that configure it."""
    pe = state.get("patch_embed.proj.weight")
    if pe is None:
        return "no recognizable DINOv3 encoder tensors"
    depth = 1 + max((int(k.split(".")[1]) for k in state if k.startswith("blocks.")), default=-1)
    st = state.get("storage_tokens")
    dim = int(pe.shape[0])
    return (f"embed_dim {dim}, patch_size {int(pe.shape[-1])}, depth {depth}, num_heads {dim // 64} (assuming 64-wide heads), "
            f"storage tokens {0 if st is None else int(st.shape[1])}")


def load_encoder_weights(backbone, path) -> str:
    """Initialize the encoder from ``path`` — a KODIAK pretraining checkpoint (its EMA teacher) or raw DINOv3 weights.
    Every encoder tensor must be covered, or this raises: weights are loaded whole or not at all."""
    import torch
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"encoder weights not found: {p} — pretrain first (python -m kodiak.pretrain) or point "
                                "kodiak.checkpoint at the DINOv3 .pth in weights/")
    ck = torch.load(str(p), map_location="cpu", weights_only=False)
    state = ck.get("state_dict", ck.get("model", ck)) if isinstance(ck, dict) else ck
    teacher = {k.split("model.teacher_encoder.", 1)[1]: v for k, v in state.items() if k.startswith("model.teacher_encoder.")}
    src, kind = (teacher, "kodiak-teacher") if teacher else (state, "dinov3")
    own = backbone.state_dict()
    usable = {k: v for k, v in src.items() if k in own and own[k].shape == v.shape}
    if missing := sorted(set(own) - set(usable)):
        raise ValueError(f"{p}: covers {len(usable)}/{len(own)} encoder tensors (first missing: {missing[:3]}); "
                         f"it holds {describe_encoder(src)}, the segmenter is built as "
                         f"embed_dim {backbone.embed_dim}, depth {len(backbone.blocks)}, num_heads {backbone.num_heads}, "
                         f"storage tokens {backbone.num_storage_tokens} — set kodiak.vit_embed_dim / vit_depth / vit_heads to match, "
                         "or point kodiak.checkpoint at the weights for this architecture (loaded whole or not at all)")
    backbone.load_state_dict(usable, strict=False)                           # extra source keys (heads, student) are fine
    return kind


def set_trainable(net, train_blocks: int) -> int:
    """Freeze the encoder except its top ``train_blocks`` ViT blocks (12 = full fine-tune, 0 = head only); the final norm
    trains whenever any block does, patch embedding and RoPE only in the full fine-tune. The head always trains.
    Returns the number of trainable parameters."""
    depth = len(net.backbone.blocks)
    if not 0 <= train_blocks <= depth:
        raise ValueError(f"kodiak.train_blocks must be in 0..{depth}, got {train_blocks}")
    for p in net.backbone.parameters():
        p.requires_grad = False
    for blk in list(net.backbone.blocks)[depth - train_blocks:]:
        for p in blk.parameters():
            p.requires_grad = True
    if train_blocks > 0:
        for p in net.backbone.norm.parameters():
            p.requires_grad = True
    if train_blocks == depth:
        for p in net.backbone.parameters():                                  # full fine-tune: patch embed, RoPE, cls + storage tokens too
            p.requires_grad = True
    for p in net.head.parameters():
        p.requires_grad = True
    return sum(p.numel() for p in net.parameters() if p.requires_grad)


def param_layers(net) -> tuple:
    """``(depth, [(layer_id, parameter)])`` over the trainable parameters, the depth ids the shared optimizer scales the learning
    rate by (``mxseg.train.module``): encoder block i -> i + 1, patch embedding / RoPE / cls + storage tokens -> 0 (deepest), the
    head and the final norm -> depth + 1 (the peak). Apply :func:`set_trainable` first."""
    depth = len(net.backbone.blocks)

    def layer_of(name: str) -> int:
        if name.startswith("backbone.blocks."):
            return int(name.split(".")[2]) + 1
        if name.startswith("backbone.") and not name.startswith("backbone.norm"):
            return 0                                                         # patch embed, RoPE, cls + storage tokens
        return depth + 1                                                     # head and the final norm

    return depth, [(layer_of(n), p) for n, p in net.named_parameters() if p.requires_grad]


def layerwise_param_groups(net, lr: float, layerwise_decay: float, weight_decay: float) -> list:
    """Fine-tuning parameter groups, ViT best practice (BEiT / DINOv2-3 recipes): the head and final norm train at the
    peak ``lr``, each encoder block one decay step below the one above it (block i: ``lr * decay^(depth - i)``), the patch
    embedding and tokens deepest of all; 1-D parameters (biases, norms, cls / storage tokens) get no weight decay.
    The same grouping every trained model gets from ``SegModule.configure_optimizers``, built here for one network."""
    depth, layers = param_layers(net)
    groups: dict = {}
    for lid, p in layers:
        wd = 0.0 if p.ndim <= 1 else weight_decay
        g = groups.setdefault((lid, wd), {"params": [], "lr": lr * layerwise_decay ** (depth + 1 - lid), "weight_decay": wd})
        g["params"].append(p)
    return list(groups.values())


def save_segmenter(path, net, meta: Optional[dict] = None) -> None:
    import torch
    torch.save({"format": WEIGHTS_FORMAT, "kind": KIND, "arch": dict(net.arch), "meta": dict(meta or {}),
                "state_dict": net.state_dict()}, str(path))


def load_segmenter(path, device="cpu"):
    """The network saved by :func:`save_segmenter` (architecture included), in eval mode on ``device``."""
    import torch
    ck = torch.load(str(path), map_location="cpu", weights_only=True)
    if not isinstance(ck, dict) or ck.get("format") != WEIGHTS_FORMAT or ck.get("kind") != KIND:
        raise ValueError(f"{path}: not a mxseg KODIAK segmenter weights file (format {WEIGHTS_FORMAT})")
    net = build_segmenter(ck["arch"])
    net.load_state_dict(ck["state_dict"])
    return net.to(device).eval(), ck["arch"], ck.get("meta", {})


def flow_instances(y: np.ndarray, min_area: int = 15) -> np.ndarray:
    """``(3, H, W)`` network output [flowY, flowX, cellprob-logit] (Cellpose scale: flows are 5x the unit field) ->
    ``int32`` instances via Cellpose's own decoding: pixels follow the flow and coalesce at the centers."""
    import torch
    from cellpose.dynamics import compute_masks

    from ..metrics import relabel
    out = compute_masks(np.ascontiguousarray(y[:2]), np.ascontiguousarray(y[2]), min_size=min_area,
                        device=torch.device("cpu"))
    masks = out[0] if isinstance(out, tuple) else out
    return relabel(np.asarray(masks).astype(np.int32))[0].astype(np.int32)


def segment_flows(net, images: Sequence[np.ndarray], device, *, min_area: int = 15, batch_size: int = 16, input=None, slides=None) -> List[np.ndarray]:
    """Instances for ``(C, H, W)`` native-dtype images through the flow objective (the Cellpose protocol); ``input`` / ``slides``
    as in :func:`mxseg.baselines.unet.render_all`."""
    import torch
    import torch.nn.functional as F

    from .unet import render_all
    xs = render_all(images, input, slides)
    out: List[np.ndarray] = []
    net.eval()
    with torch.inference_mode():
        for i in range(0, len(images), batch_size):
            x = torch.from_numpy(np.stack(xs[i:i + batch_size])).to(device)
            h, w = x.shape[-2:]
            ph, pw = (-h) % 32, (-w) % 32
            if ph or pw:
                x = F.pad(x, (0, pw, 0, ph), mode="reflect")
            y = net(x)[..., :h, :w].float().cpu().numpy()
            out.extend(flow_instances(p, min_area) for p in y)
    return out


class KodiakBaseline(Baseline):
    """A KODIAK segmenter trained by ``mxseg train``: ``mxseg eval --baseline kodiak --param weights=runs/<name>/weights.pt``."""

    name = "kodiak"

    def __init__(self, weights: str, min_area: int = 15, batch_size: int = 16, device: Optional[str] = None):
        self.weights, self.min_area, self.batch_size = weights, min_area, batch_size
        self._device = device
        self._net = None

    def params(self) -> dict:
        return {"weights": self.weights, "min_area": self.min_area}

    def slug(self) -> str:
        return f"kodiak_{Path(self.weights).parent.name}" + (f"_min_area-{self.min_area}" if self.min_area != 15 else "")

    @property
    def device(self) -> str:
        return self._device or pick_device()

    def model(self):
        if self._net is None:
            self._net, self.arch, self.meta = load_segmenter(self.weights, self.device)
            self._input_from_meta(self.meta)
        return self._net

    def input_markers(self):
        self.model()                                                     # the weights carry the input recipe
        return super().input_markers()

    def resolve_input(self, dataset_dir, slides):
        self.model()                                                     # the weights decide
        return super().resolve_input(dataset_dir, slides)

    def predict_batch(self, images: Sequence[np.ndarray], pixel_size_um: Optional[float] = None, slides=None) -> List[np.ndarray]:
        net = self.model()
        seg = segment_flows if self.arch.get("objective") == "flows" else segment
        return seg(net, images, self.device, min_area=self.min_area, batch_size=self.batch_size, input=self.input, slides=slides)
