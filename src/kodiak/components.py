# models/components.py

"""
Shared components for KODIAK models:
- Encoders, Decoders, Projectors, and utility layers

Encoder and decoder components are borrowed from the DINOv3 architecture.
"""

from __future__ import annotations

import logging
from functools import partial
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# DINOv3 imports - exact DINOv3 components, vendored as the ``dinov3`` package
# ---------------------------------------------------------------------------
from dinov3.layers import (
    LayerScale,
    Mlp,
    PatchEmbed,
    RopePositionEmbedding,
)
from dinov3.layers.attention import SelfAttention
from dinov3.layers.block import SelfAttentionBlock
from dinov3.utils import named_apply

logger = logging.getLogger("kodiak")


# ---------------------------------------------------------------------------
# DINOv3 Weight Initialization (exact copy from DINOv3)
# ---------------------------------------------------------------------------

def init_weights_vit(module: nn.Module, name: str = ""):
    """Exact DINOv3 weight initialization."""
    if isinstance(module, nn.Linear):
        torch.nn.init.trunc_normal_(module.weight, std=0.02)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
        if hasattr(module, "bias_mask") and module.bias_mask is not None:
            o = module.out_features
            module.bias_mask.fill_(1)
            module.bias_mask[o // 3 : 2 * o // 3].fill_(0)
    if isinstance(module, nn.LayerNorm):
        module.reset_parameters()
    if isinstance(module, LayerScale):
        module.reset_parameters()
    if isinstance(module, PatchEmbed):
        module.reset_parameters()


# ---------------------------------------------------------------------------
# Helper Modules
# ---------------------------------------------------------------------------

class MLP(nn.Module):
    def __init__(self, dim: int, mlp_ratio: float = 4.0, dropout: float = 0.0):
        super().__init__()
        hidden_dim = int(dim * mlp_ratio)
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class TransformerDecoderBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int, mlp_ratio: float = 4.0,
                 attn_dropout: float = 0.0, proj_dropout: float = 0.0, mlp_dropout: float = 0.0, use_sdpa: bool = True):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, dropout=attn_dropout, batch_first=True)
        self.dropout1 = nn.Dropout(proj_dropout)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLP(dim, mlp_ratio=mlp_ratio, dropout=mlp_dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = x
        x = self.norm1(x)
        x, _ = self.attn(x, x, x, need_weights=False)
        x = self.dropout1(x)
        x = x + h
        h = x
        x = self.norm2(x)
        x = self.mlp(x)
        x = x + h
        return x


class PrototypeLayer(nn.Module):
    """
    Prototype prediction layer (EXACT DINOv3 last_layer match).

    DINOv3 uses a simple Linear(bottleneck_dim, n_prototypes, bias=False)
    with trunc_normal_ initialization. No weight normalization.

    Input should be L2-normalized BEFORE calling this layer.
    """
    def __init__(self, in_dim: int, num_prototypes: int):
        super().__init__()
        # EXACT DINOv3: Linear with bias=False
        self.layer = nn.Linear(in_dim, num_prototypes, bias=False)
        self._init_weights()

    def _init_weights(self):
        # EXACT DINOv3: trunc_normal_ initialization
        torch.nn.init.trunc_normal_(self.layer.weight, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # EXACT DINOv3: Input should already be normalized, just apply linear
        # NO weight normalization (DINOv3 doesn't do this)
        # NO input normalization here (done before calling this layer)
        return self.layer(x)


# ---------------------------------------------------------------------------
# Encoder (borrowed from DINOv3)
# ---------------------------------------------------------------------------

class KeyMaskedSelfAttention(SelfAttention):
    """DINOv3's SelfAttention, with ``attn_bias`` accepted as a boolean key mask (B, 1, 1, L; True = may be attended).
    Without one this is the vendored attention, op for op."""

    def compute_attention(self, qkv: torch.Tensor, attn_bias=None, rope=None) -> torch.Tensor:
        if attn_bias is None:
            return super().compute_attention(qkv, attn_bias=None, rope=rope)
        B, N, _ = qkv.shape
        C = self.qkv.in_features
        qkv = qkv.reshape(B, N, 3, self.num_heads, C // self.num_heads)
        q, k, v = [t.transpose(1, 2) for t in torch.unbind(qkv, 2)]
        if rope is not None:
            q, k = self.apply_rope(q, k, rope)
        x = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_bias)
        return x.transpose(1, 2).reshape(B, N, C)


class KeyMaskedSelfAttentionBlock(SelfAttentionBlock):
    """DINOv3's block, plus an optional ``key_padding_mask`` for the padded student batch (see DinoV3Encoder.forward).

    Without a mask it is the vendored block, path for path. With one, the residual branches run on the whole batch
    and stochastic depth is applied as a per-sample factor (0 or b / kept) -- the same draws and the same values as
    DINOv3's sample-subset ``index_add`` form (tests pin it), without the gathers the mask would otherwise follow.
    """

    def __init__(self, *args, **kwargs):
        kwargs.setdefault("attn_class", KeyMaskedSelfAttention)
        super().__init__(*args, **kwargs)

    def forward(self, x, rope=None, key_padding_mask=None):
        if key_padding_mask is None:
            return super().forward(x, rope)
        x = x + self._drop_path(self.ls1(self.attn(self.norm1(x), attn_bias=key_padding_mask, rope=rope)))
        return x + self._drop_path(self.ls2(self.mlp(self.norm2(x))))

    def _drop_path(self, residual: torch.Tensor) -> torch.Tensor:
        if not (self.training and self.sample_drop_ratio > 0.0):
            return residual
        b = residual.shape[0]
        kept = max(int(b * (1 - self.sample_drop_ratio)), 1)
        keep = torch.zeros(b, device=residual.device, dtype=residual.dtype)
        keep[torch.randperm(b, device=residual.device)[:kept]] = b / kept
        return keep[:, None, None] * residual


class DinoV3Encoder(nn.Module):
    """
    Vision Transformer encoder for KODIAK (borrowed from DINOv3).

    This is a faithful implementation of DinoVisionTransformer from DINOv3,
    adapted to support KODIAK's masked training paradigm while maintaining
    architectural equivalence.

    Key features (matching DINOv3):
    - SelfAttentionBlock with qkv_bias=True
    - LayerScale initialization
    - RoPE position encoding
    - Storage tokens (registers)
    - Proper weight initialization
    - Optional mask_k_bias for checkpoint compatibility

    The only difference from DINOv3 is the forward pass:
    - DINOv3: replaces masked tokens with mask_token
    - KODIAK: removes masked tokens for memory efficiency (MAE-style)

    Both approaches are mathematically equivalent for the visible tokens -- provided the padding that ragged masks
    need never enters attention, which ``mask_padded_keys`` guarantees (see ``forward``).
    """

    def __init__(
        self,
        img_size: int = 256,
        patch_size: int = 16,
        in_chans: int = 3,
        embed_dim: int = 384,
        depth: int = 12,
        num_heads: int = 6,
        mlp_ratio: float = 4.0,
        num_storage_tokens: int = 4,
        # DINOv3 defaults
        qkv_bias: bool = True,  # DINOv3 default
        proj_bias: bool = True,
        ffn_bias: bool = True,
        drop_path_rate: float = 0.0,
        layerscale_init: float = 1e-5,  # DINOv3 default
        norm_layer: str = "layernorm",
        # RoPE parameters (DINOv3 defaults)
        rope_theta: float = 100.0,
        rope_normalize_coords: str = "separate",
        # mask_k_bias for checkpoint compatibility
        mask_k_bias: bool = True,
        # masked forward: exclude the padded (duplicated) visible tokens from attention; False = the reference behaviour
        mask_padded_keys: bool = True,
        **kwargs,
    ):
        super().__init__()

        # Store dimensions
        self.embed_dim = embed_dim
        self.mask_padded_keys = mask_padded_keys
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.num_storage_tokens = num_storage_tokens
        self.n_storage_tokens = num_storage_tokens  # DINOv3 naming
        self.patch_size = patch_size
        self.n_blocks = depth
        self.mask_k_bias = mask_k_bias

        # Default grid size (actual computed dynamically)
        self.default_grid_size = img_size // patch_size
        self.default_num_patches = self.default_grid_size ** 2

        # Norm layer
        norm_layer_cls = partial(nn.LayerNorm, eps=1e-6)

        # Patch embedding (DINOv3 style - flatten_embedding=False)
        self.patch_embed = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
            flatten_embedding=False,
        )

        # Learnable tokens (DINOv3 uses torch.empty)
        self.cls_token = nn.Parameter(torch.empty(1, 1, embed_dim))
        if num_storage_tokens > 0:
            self.storage_tokens = nn.Parameter(torch.empty(1, num_storage_tokens, embed_dim))
        else:
            self.register_parameter('storage_tokens', None)

        # Mask token (DINOv3 style - for checkpoint compatibility only)
        # NOTE: KODIAK uses MAE-style masking (removes tokens) not DINOv3-style (replaces with mask_token)
        # Registered as buffer (not parameter) to avoid DDP unused parameter error
        self.register_buffer('mask_token', torch.zeros(1, embed_dim))

        # RoPE position encoding (exact DINOv3 parameters)
        self.rope_embed = RopePositionEmbedding(
            embed_dim=embed_dim,
            num_heads=num_heads,
            base=rope_theta,
            normalize_coords=rope_normalize_coords,
        )

        # Transformer blocks (exact DINOv3 SelfAttentionBlock; the subclass only adds the key mask of the masked pass)
        self.blocks = nn.ModuleList([
            KeyMaskedSelfAttentionBlock(
                dim=embed_dim,
                num_heads=num_heads,
                ffn_ratio=mlp_ratio,
                qkv_bias=qkv_bias,  # DINOv3 default: True
                proj_bias=proj_bias,
                ffn_bias=ffn_bias,
                drop_path=drop_path_rate,
                norm_layer=norm_layer_cls,
                act_layer=nn.GELU,
                ffn_layer=Mlp,
                init_values=layerscale_init,
                mask_k_bias=mask_k_bias,
            )
            for _ in range(depth)
        ])

        # Final norm
        self.norm = norm_layer_cls(embed_dim)

        # Initialize weights (DINOv3 style)
        self.init_weights()

    def init_weights(self):
        """DINOv3-style weight initialization."""
        self.rope_embed._init_weights()
        nn.init.normal_(self.cls_token, std=0.02)
        if self.storage_tokens is not None:
            nn.init.normal_(self.storage_tokens, std=0.02)
        # mask_token is a buffer (initialized to zeros in __init__), no need to reinit
        named_apply(init_weights_vit, self)

    def prepare_tokens(self, x: torch.Tensor) -> Tuple[torch.Tensor, Tuple[int, int]]:
        """
        Prepare tokens for transformer (DINOv3 style).

        Args:
            x: Input images (B, C, H, W)

        Returns:
            tokens: (B, 1 + n_storage + N, D) prepared tokens
            hw_tuple: (H, W) grid dimensions
        """
        x = self.patch_embed(x)  # (B, H, W, D)
        B, H, W, _ = x.shape
        x = x.flatten(1, 2)  # (B, N, D)

        # Concat CLS + storage + patches
        cls_token = self.cls_token.expand(B, -1, -1)
        if self.storage_tokens is not None:
            storage = self.storage_tokens.expand(B, -1, -1)
            x = torch.cat([cls_token, storage, x], dim=1)
        else:
            x = torch.cat([cls_token, x], dim=1)

        return x, (H, W)

    def forward_features(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Full forward pass (DINOv3 style interface).

        Args:
            x: Input images (B, C, H, W)

        Returns:
            Dictionary with normalized outputs
        """
        tokens, (H, W) = self.prepare_tokens(x)

        # Get RoPE
        rope_sincos = self.rope_embed(H=H, W=W)

        # Forward through blocks
        for block in self.blocks:
            tokens = block(tokens, rope_sincos)

        # Normalize
        x_norm = self.norm(tokens)

        n_prefix = 1 + (self.n_storage_tokens if self.storage_tokens is not None else 0)

        return {
            "x_norm_clstoken": x_norm[:, 0],
            "x_storage_tokens": x_norm[:, 1:n_prefix] if self.storage_tokens is not None else None,
            "x_norm_patchtokens": x_norm[:, n_prefix:],
            "x_prenorm": tokens,
        }

    def forward_full(self, x: torch.Tensor, return_attention: bool = False):
        """
        Full forward pass without masking (for teacher encoder and inference).
        Compatible with KODIAK interface.

        Args:
            x: Input images (B, C, H, W)
            return_attention: Whether to return attention maps (not implemented for DINOv3 blocks)

        Returns:
            cls_token: (B, D) CLS token output
            patch_tokens: (B, N, D) all patch token outputs
            attn_maps: None (attention maps not supported with DINOv3 blocks)
        """
        output = self.forward_features(x)
        return output["x_norm_clstoken"], output["x_norm_patchtokens"], None

    def forward(
        self,
        x: torch.Tensor,
        visible_indices: torch.Tensor,
        mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass with masking (for student encoder during training).

        This is the MAE-style forward where only visible patches are processed.
        More memory efficient than DINOv3's mask token replacement approach.

        Ragged masks are padded to the batch's largest visible count by repeating a row's last visible token
        (``Kodiak._mask_to_indices``). With ``mask_padded_keys`` those padded slots are excluded as attention keys,
        so every token attends over the row's real visible tokens only -- the MAE-style encoding the method
        describes. The reference implementation (``mask_padded_keys=False``) lets the duplicates act as extra keys,
        which skews attention toward one arbitrary patch: measured with the DINOv3 weights, that moves a 30 %-masked
        row's CLS token about as far as swapping in a different image (cosine 0.72) and its visible tokens by ~30 %.

        Args:
            x: Input images (B, C, H, W)
            visible_indices: (B, N_visible) indices of visible patches (padded rows: real entries first)
            mask: (B, N) boolean mask (True = masked)

        Returns:
            cls_token: (B, D) CLS token output
            patch_tokens: (B, N_visible, D) visible patch token outputs
        """
        B, C, H, W = x.shape
        n_prefix = 1 + (self.n_storage_tokens if self.storage_tokens is not None else 0)

        # Compute grid dimensions
        grid_h = H // self.patch_size
        grid_w = W // self.patch_size

        # Get patches
        patches = self.patch_embed(x)  # (B, H, W, D)
        patches = patches.flatten(1, 2)  # (B, N, D)

        # Gather visible patches
        batch_idx = torch.arange(B, device=x.device).unsqueeze(1)
        visible_patches = patches[batch_idx, visible_indices]  # (B, N_vis, D)

        # Concat CLS + storage + visible patches
        cls_token = self.cls_token.expand(B, -1, -1)
        if self.storage_tokens is not None:
            storage = self.storage_tokens.expand(B, -1, -1)
            tokens = torch.cat([cls_token, storage, visible_patches], dim=1)
        else:
            tokens = torch.cat([cls_token, visible_patches], dim=1)

        # Compute RoPE for full grid, then gather for visible positions
        rope_sin, rope_cos = self.rope_embed(H=grid_h, W=grid_w)  # (N, head_dim)

        # Gather RoPE for visible indices
        # rope_sin/cos shape: (N, head_dim) -> expand to (B, N, head_dim) -> gather
        rope_sin = rope_sin.unsqueeze(0).expand(B, -1, -1)  # (B, N, head_dim)
        rope_cos = rope_cos.unsqueeze(0).expand(B, -1, -1)

        gather_idx = visible_indices.unsqueeze(-1).expand(-1, -1, self.head_dim)
        rope_sin_vis = torch.gather(rope_sin, 1, gather_idx)  # (B, N_vis, head_dim)
        rope_cos_vis = torch.gather(rope_cos, 1, gather_idx)

        # DINOv3 attention expects RoPE with shape that broadcasts with [B, heads, N, head_dim]
        # Add head dimension: (B, N_vis, head_dim) -> (B, 1, N_vis, head_dim)
        rope_sin_vis = rope_sin_vis.unsqueeze(1)  # (B, 1, N_vis, head_dim)
        rope_cos_vis = rope_cos_vis.unsqueeze(1)  # (B, 1, N_vis, head_dim)

        rope_sincos = (rope_sin_vis, rope_cos_vis)

        # Key mask: prefix tokens and each row's first num_visible slots may be attended, the padded repeats may not
        key_padding_mask = None
        if self.mask_padded_keys:
            num_visible = (~mask).sum(dim=1, keepdim=True)                                              # (B, 1)
            valid = torch.arange(visible_indices.shape[1], device=x.device).unsqueeze(0) < num_visible  # (B, N_vis)
            key_padding_mask = torch.cat([valid.new_ones(B, n_prefix), valid], dim=1)[:, None, None, :]  # (B, 1, 1, L)

        # Forward through blocks
        # Note: DINOv3 blocks apply RoPE only to patch tokens (not prefix tokens)
        for block in self.blocks:
            tokens = block(tokens, rope_sincos, key_padding_mask=key_padding_mask)

        # Normalize
        tokens = self.norm(tokens)

        return tokens[:, 0], tokens[:, n_prefix:]


# Alias for backward compatibility
MAEStyleDinoV3Encoder = DinoV3Encoder


# ---------------------------------------------------------------------------
# Decoder (Student Only) - Supports Variable Input Sizes
# ---------------------------------------------------------------------------

class MAEStyleDecoder(nn.Module):
    """
    MAE-style decoder that supports variable input sizes via position embedding interpolation.
    """
    def __init__(self, embed_dim, decoder_embed_dim, decoder_depth, decoder_num_heads, mlp_ratio, num_patches, use_sdpa=True):
        super().__init__()
        self.default_num_patches = num_patches
        self.default_grid_size = int(num_patches ** 0.5)
        self.decoder_embed_dim = decoder_embed_dim
        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))
        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, num_patches, decoder_embed_dim))

        self.decoder_blocks = nn.ModuleList([
            TransformerDecoderBlock(decoder_embed_dim, decoder_num_heads, mlp_ratio, use_sdpa=use_sdpa)
            for _ in range(decoder_depth)
        ])
        self.decoder_norm = nn.LayerNorm(decoder_embed_dim)
        self._init_weights()

    def _init_weights(self):
        nn.init.trunc_normal_(self.mask_token, std=0.02)
        nn.init.trunc_normal_(self.decoder_pos_embed, std=0.02)

    def _get_pos_embed(self, num_patches: int, grid_h: int = None, grid_w: int = None):
        """Get position embeddings, interpolating if size differs from default."""
        if num_patches == self.default_num_patches:
            return self.decoder_pos_embed

        if grid_h is None or grid_w is None:
            grid_h = grid_w = int(num_patches ** 0.5)

        pos_embed = self.decoder_pos_embed.reshape(
            1, self.default_grid_size, self.default_grid_size, self.decoder_embed_dim
        ).permute(0, 3, 1, 2)

        pos_embed = F.interpolate(
            pos_embed,
            size=(grid_h, grid_w),
            mode='bilinear',
            align_corners=False
        )

        pos_embed = pos_embed.permute(0, 2, 3, 1).reshape(1, num_patches, self.decoder_embed_dim)
        return pos_embed

    def forward(self, visible_features, visible_indices, masked_indices, num_patches=None, grid_h=None, grid_w=None):
        """Decode visible features and predict masked positions."""
        B = visible_features.shape[0]

        if num_patches is None:
            num_patches = self.default_num_patches

        visible_dec = self.decoder_embed(visible_features)

        full = torch.zeros(B, num_patches, self.decoder_embed_dim, device=visible_dec.device, dtype=visible_dec.dtype)

        batch_idx = torch.arange(B, device=visible_dec.device).unsqueeze(1)
        full[batch_idx, visible_indices] = visible_dec

        num_masked = masked_indices.shape[1]
        mask_tokens = self.mask_token.to(visible_dec.dtype).expand(B, num_masked, -1)
        full[batch_idx, masked_indices] = mask_tokens

        pos_embed = self._get_pos_embed(num_patches, grid_h, grid_w)
        full = full + pos_embed.to(full.dtype)

        for block in self.decoder_blocks:
            full = block(full)
        return self.decoder_norm(full)


# ---------------------------------------------------------------------------
# DINO Projector
# ---------------------------------------------------------------------------

class DINOProjector(nn.Module):
    """
    DINO-style projector head (EXACT DINOv3 _build_mlp match).

    DINOv3 DINOHead structure:
    - mlp: in_dim → hidden_dim → ... → bottleneck_dim (with GELU, bias=True)
    - Normalization after mlp
    - last_layer: bottleneck_dim → n_prototypes (handled by PrototypeLayer)

    This projector implements the mlp part only. Output should be normalized
    before passing to PrototypeLayer.
    """
    def __init__(self, in_dim, out_dim, hidden_dim=2048, n_layers=3, bias=True):
        super().__init__()
        # EXACT DINOv3 _build_mlp structure
        if n_layers == 1:
            self.mlp = nn.Linear(in_dim, out_dim, bias=bias)
        else:
            layers = []
            # First layer: in_dim → hidden_dim
            layers.append(nn.Linear(in_dim, hidden_dim, bias=bias))
            layers.append(nn.GELU())
            # Middle layers: hidden_dim → hidden_dim
            for _ in range(n_layers - 2):
                layers.append(nn.Linear(hidden_dim, hidden_dim, bias=bias))
                layers.append(nn.GELU())
            # Last layer: hidden_dim → out_dim (bottleneck)
            layers.append(nn.Linear(hidden_dim, out_dim, bias=bias))
            self.mlp = nn.Sequential(*layers)

        self._init_weights()

    def _init_weights(self):
        # EXACT DINOv3: trunc_normal_ with std=0.02, zeros for bias
        for m in self.modules():
            if isinstance(m, nn.Linear):
                torch.nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.mlp(x)
