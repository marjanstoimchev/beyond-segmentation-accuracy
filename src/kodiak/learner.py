# learners/pretraining.py

"""
PyTorch Lightning module for training KODIAK with KoLeo.
OPTIMIZED VERSION with DINOv3-style scheduling (borrowed from DINOv3).

Features (EXACT match to DINOv3 ssl_default_config.yaml):
- CosineScheduler with warmup + cosine decay
- Separate last_layer_lr_schedule (frozen during freeze_last_layer_epochs)
- LR scaling: sqrt_wrt_1024 rule
- Layerwise LR decay (default: 0.9)
- Patch embedding LR multiplier (default: 0.2)
- No weight decay on bias, norm, gamma (EXACT DINOv3 match)
- KoLeo loss for collapse prevention
- Teacher temperature warmup schedule (0.04 → 0.07 over 30 epochs)
- EMA momentum cosine schedule (0.992 → 1.0)
- Weight decay cosine schedule (0.04 → 0.4)
- AdamW with betas=(0.9, 0.999)
"""

from __future__ import annotations

import math
import re
from typing import Any, Dict, List, Optional

import lightning.pytorch as L
import numpy as np
import torch
from torch.optim import AdamW

from .losses import KoLeoLoss, KoLeoLossDistributed, MaskLoss
from .model import Kodiak
from .prototype_cls_loss import MultiCropPrototypeCLSLoss


class CosineScheduler:
    """
    DINOv3-style cosine scheduler with freeze phase and warmup.

    Schedule order: freeze_iters → warmup_iters → cosine_decay

    This matches the exact implementation in:
    DinoV3LightningTraining/dinov3/dinov3/train/cosine_lr_scheduler.py
    """

    def __init__(
        self,
        base_value: float,
        final_value: float,
        total_iters: int,
        warmup_iters: int = 0,
        start_warmup_value: float = 0.0,
        freeze_iters: int = 0,
    ):
        self.final_value = np.float64(final_value)
        self.total_iters = total_iters

        # Phase 1: Freeze (LR = 0)
        freeze_schedule = np.zeros((freeze_iters,))

        # Phase 2: Linear warmup (0 → base_value)
        warmup_schedule = np.linspace(start_warmup_value, base_value, warmup_iters)

        # Phase 3: Cosine decay (base_value → final_value)
        cosine_iters = total_iters - warmup_iters - freeze_iters
        if cosine_iters > 0:
            iters = np.arange(cosine_iters)
            cosine_schedule = final_value + 0.5 * (base_value - final_value) * (
                1 + np.cos(np.pi * iters / cosine_iters)
            )
        else:
            cosine_schedule = np.array([])

        self.schedule = np.concatenate(
            (freeze_schedule, warmup_schedule, cosine_schedule),
            dtype=np.float64
        )

        # Ensure schedule length matches total_iters
        if len(self.schedule) != self.total_iters:
            # Pad or truncate if needed
            if len(self.schedule) < self.total_iters:
                padding = np.full(self.total_iters - len(self.schedule), self.final_value)
                self.schedule = np.concatenate((self.schedule, padding))
            else:
                self.schedule = self.schedule[:self.total_iters]

    def __getitem__(self, it: int) -> float:
        if it >= self.total_iters:
            return float(self.final_value)
        return float(self.schedule[it])

    def __len__(self) -> int:
        return self.total_iters


def cosine_schedule(base_value: float, final_value: float, epochs: int, warmup_epochs: int = 0) -> callable:
    """
    Create a cosine schedule function for epoch-level schedules (EMA, weight decay, temp).

    Args:
        base_value: Starting value
        final_value: Ending value
        epochs: Total number of epochs
        warmup_epochs: Number of warmup epochs (linear warmup to base_value)

    Returns:
        Function that takes epoch and returns scheduled value
    """
    def schedule(epoch: int) -> float:
        if epoch < warmup_epochs:
            return base_value
        else:
            progress = (epoch - warmup_epochs) / max(1, epochs - warmup_epochs)
            return final_value + 0.5 * (base_value - final_value) * (1 + math.cos(math.pi * progress))
    return schedule


def get_vit_lr_decay_rate(name: str, lr_decay_rate: float, num_layers: int) -> float:
    """
    Calculate LR decay rate for different ViT blocks (DINOv3 style).

    This matches the exact implementation in:
    DinoV3LightningTraining/dinov3/dinov3/train/param_groups.py

    Args:
        name: Parameter name
        lr_decay_rate: Base LR decay rate (e.g., 0.9)
        num_layers: Number of transformer blocks (e.g., 12)

    Returns:
        LR multiplier for this parameter
    """
    layer_id = num_layers + 1  # Default: highest layer (full LR)

    # Check if this is a backbone parameter
    if "student_encoder" in name or "backbone" in name:
        if (
            ".pos_embed" in name
            or ".patch_embed" in name
            or ".mask_token" in name
            or ".cls_token" in name
            or ".storage_tokens" in name
            or "pos_embed" in name
            or "patch_embed" in name
            or "cls_token" in name
            or "storage_tokens" in name
        ):
            layer_id = 0
        elif ".blocks." in name:
            # Extract block number: "student_encoder.blocks.5.attn..." → 5
            match = re.search(r"\.blocks\.(\d+)\.", name)
            if match:
                layer_id = int(match.group(1)) + 1
        elif "blocks." in name:
            # Handle case without leading dot
            match = re.search(r"blocks\.(\d+)\.", name)
            if match:
                layer_id = int(match.group(1)) + 1

    return lr_decay_rate ** (num_layers + 1 - layer_id)


def should_skip_weight_decay(name: str) -> bool:
    """
    Determine if a parameter should skip weight decay (EXACT DINOv3 match).

    From DINOv3 param_groups.py:
    No weight decay on: bias, norm parameters, layer scale gamma, fourier_w.

    NOTE: DINOv3 does NOT skip weight decay on learned tokens (cls_token,
    storage_tokens, pos_embed, mask_token) - only on bias/norm/gamma.
    """
    return (
        name.endswith("bias")
        or "norm" in name
        or "gamma" in name  # Layer scale
        or "fourier_w" in name  # DINOv3 fourier features
    )


def scale_lr(base_lr: float, batch_size: int, world_size: int, scaling: str = "sqrt_wrt_1024") -> float:
    """
    Scale learning rate based on effective batch size (DINOv3 style).

    Args:
        base_lr: Base learning rate
        batch_size: Batch size per GPU
        world_size: Number of GPUs/processes
        scaling: Scaling rule ('none', 'sqrt_wrt_1024')

    Returns:
        Scaled learning rate
    """
    if scaling is None or scaling == "none":
        return base_lr
    elif scaling == "sqrt_wrt_1024":
        effective_batch_size = batch_size * world_size
        return base_lr * 4.0 * math.sqrt(effective_batch_size / 1024.0)
    else:
        raise ValueError(f"Unknown LR scaling rule: {scaling}")


def get_layer_id_for_vit(name: str, num_layers: int) -> int:
    """
    Assign a layer ID to each parameter for layerwise LR decay.

    Layer 0 = patch_embed, cls_token, storage_tokens (lowest LR)
    Layer 1 to num_layers = transformer blocks
    Layer num_layers + 1 = norm, heads, etc. (highest LR)

    Args:
        name: Parameter name
        num_layers: Number of transformer blocks (e.g., 12)

    Returns:
        Layer ID (0 to num_layers + 1)
    """
    if "patch_embed" in name:
        return 0
    elif "cls_token" in name or "storage_tokens" in name or "pos_embed" in name:
        return 0
    elif "blocks." in name:
        match = re.search(r"blocks\.(\d+)\.", name)
        if match:
            block_id = int(match.group(1))
            return block_id + 1
    return num_layers + 1


class KodiakLearner(L.LightningModule):
    """
    PyTorch Lightning module for KODIAK training with DINOv3-style scheduling.

    Features (matched to DinoV3LightningTraining):
        - CosineScheduler with freeze phase + linear warmup + cosine decay
        - LR scaling: sqrt_wrt_1024 rule
        - Layerwise LR decay (default: 0.9)
        - Patch embedding LR multiplier (default: 0.2)
        - No weight decay on bias, norm, gamma, learned tokens
        - KoLeo loss for collapse prevention
        - Teacher temperature warmup (0.04 → 0.07 over 30 epochs)
        - EMA momentum cosine schedule (0.992 → 1.0)
        - Weight decay cosine schedule (0.04 → 0.4)
    """

    def __init__(
        self,
        # Dependency injection - SOLID principle
        model: Optional[torch.nn.Module] = None,
        loss_fn: Optional[torch.nn.Module] = None,
        koleo_loss: Optional[torch.nn.Module] = None,
        # Training config (DINOv3-aligned defaults)
        lr: float = 1e-3,  # DINOv3 default base LR
        min_lr: float = 1e-6,  # DINOv3 default
        weight_decay: float = 0.04,  # DINOv3 default start
        weight_decay_end: float = 0.4,  # DINOv3 cosine schedule endpoint
        warmup_epochs: int = 10,  # DINOv3 default
        freeze_last_layer_epochs: int = 1,  # DINOv3 default (freeze last layer at start)
        max_epochs: int = 100,
        freeze_backbone: bool = False,
        backbone_lr_scale: float = 1.0,  # 1.0 for pretraining, 0.1 for finetuning
        # EMA momentum schedule (DINOv3 style)
        ema_momentum: float = 0.992,  # DINOv3 default start value
        ema_momentum_end: float = 1.0,  # DINOv3 default end value
        # Teacher temperature schedule (DINOv3 style)
        teacher_temp_warmup: float = 0.04,  # DINOv3 default start temperature
        teacher_temp: float = 0.07,  # DINOv3 default end temperature
        teacher_temp_warmup_epochs: int = 30,  # DINOv3 default
        # KoLeo loss (DINOv3 style)
        koleo_loss_weight: float = 0.1,  # DINOv3 default
        koleo_distributed: bool = False,
        # Multi-Crop Prototype CLS Loss
        # CLS matching in prototype space (K-dim) instead of feature space (D-dim)
        prototype_cls_loss_weight: float = 1.0,  # Weight for prototype CLS loss
        prototype_cls_student_temp: float = 0.1,  # Student temperature (softer)
        prototype_cls_teacher_temp: float = 0.04,  # Teacher temperature (sharper)
        n_local_crops: int = 8,  # Number of local crops (96x96)
        compile_model: bool = False,
        # Layerwise LR decay (DINOv3 style)
        layerwise_decay: float = 0.9,  # DINOv3 default (was 0.95)
        patch_embed_lr_mult: float = 0.2,  # DINOv3 default
        # LR scaling rule
        lr_scaling: str = "sqrt_wrt_1024",  # DINOv3 default scaling
        batch_size: int = 64,  # For LR scaling calculation
        # AdamW betas (EXACT DINOv3 defaults)
        adamw_beta1: float = 0.9,  # DINOv3 default
        adamw_beta2: float = 0.999,  # DINOv3 default
        # DINO head weight decay multiplier (DINOv3 style)
        dino_head_wd_multiplier: float = 1.0,  # DINOv3 default
        # Legacy support - will be deprecated
        img_size: Optional[int] = None,
        patch_size: Optional[int] = None,
        embed_dim: Optional[int] = None,
        vit_depth: Optional[int] = None,
        vit_heads: Optional[int] = None,
        mlp_ratio: Optional[float] = None,
        num_storage_tokens: Optional[int] = None,
        num_prototypes: Optional[int] = None,
        projector_dim: Optional[int] = None,
        decoder_embed_dim: Optional[int] = None,
        decoder_depth: Optional[int] = None,
        decoder_num_heads: Optional[int] = None,
        drop_path_rate: float = 0.1,  # Conservative for low-data regime
        pretrained_path: Optional[str] = None,
        mask_padded_keys: bool = True,  # student attends over its real visible tokens only; False = reference behaviour
        center_momentum: Optional[float] = None,
        sinkhorn_iters: Optional[int] = None,
        # Ablation flags for controlled experiments
        use_sinkhorn: bool = True,  # If False, use softmax instead of Sinkhorn-Knopp
        **kwargs,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=['model', 'loss_fn', 'koleo_loss'])

        # Dependency injection: Use provided instances or create defaults
        if model is None:
            # Legacy mode: Create model from individual parameters
            if img_size is None:
                raise ValueError(
                    "ERROR: Either provide 'model' instance or individual model parameters\n"
                    "\n"
                    "Option 1 (Recommended - Dependency Injection):\n"
                    "  model = Kodiak(...)\n"
                    "  learner = KodiakLearner(model=model, loss_fn=loss_fn, ...)\n"
                    "\n"
                    "Option 2 (Legacy):\n"
                    "  learner = KodiakLearner(img_size=256, embed_dim=384, ...)\n"
                )

            model = Kodiak(
                num_prototypes=num_prototypes or 128,
                projector_dim=projector_dim or 256,
                img_size=img_size,
                patch_size=patch_size or 16,
                embed_dim=embed_dim,
                depth=vit_depth or 12,
                num_heads=vit_heads or 6,
                mlp_ratio=mlp_ratio or 4.0,
                num_storage_tokens=num_storage_tokens if num_storage_tokens is not None else 4,
                decoder_embed_dim=decoder_embed_dim or 192,
                decoder_depth=decoder_depth or 4,
                decoder_num_heads=decoder_num_heads or 6,
                drop_path_rate=drop_path_rate,
                pretrained_path=pretrained_path,
                mask_padded_keys=mask_padded_keys,
            )

        self.model = model

        # Compile optimization (PyTorch 2.0+)
        if compile_model and hasattr(torch, "compile"):
            print("Compiling model with torch.compile()...")
            self.model.student_encoder = torch.compile(self.model.student_encoder)
            self.model.teacher_encoder = torch.compile(self.model.teacher_encoder)
            self.model.decoder = torch.compile(self.model.decoder)

        if freeze_backbone:
            print("Freezing backbone weights...")
            for p in self.model.student_encoder.parameters():
                p.requires_grad = False

        # Store ablation flags
        self.use_sinkhorn = use_sinkhorn

        # Loss functions (dependency injection)
        # Note: teacher_temp will be updated via schedule, start with warmup value
        if loss_fn is None:
            num_protos = getattr(self.model, 'num_prototypes', num_prototypes or 128)
            loss_fn = MaskLoss(
                num_prototypes=num_protos,
                teacher_temp=teacher_temp_warmup,  # Start with warmup temperature
                center_momentum=center_momentum or 0.9,
                sinkhorn_iters=sinkhorn_iters or 3,
                use_sinkhorn=use_sinkhorn,  # A1 ablation
            )

        # KoLeo loss for collapse prevention
        if koleo_loss is None:
            if koleo_distributed:
                koleo_loss = KoLeoLossDistributed(topk=1)
            else:
                koleo_loss = KoLeoLoss()

        self.loss_fn = loss_fn
        self.koleo_loss = koleo_loss
        self.koleo_loss_weight = koleo_loss_weight
        self.backbone_lr_scale = backbone_lr_scale

        # Multi-Crop Prototype CLS Loss CONTRIBUTION (replaces DINO loss)
        # CLS tokens match in prototype space across global + local crops
        # This is what differentiates us from DINOv3
        self.prototype_cls_loss_weight = prototype_cls_loss_weight

        # Only create the module if weight > 0 to avoid DDP unused parameter errors
        if prototype_cls_loss_weight > 0:
            num_protos = getattr(self.model, 'num_prototypes', num_prototypes or 128)
            embed_dim_val = getattr(self.model.student_encoder, 'embed_dim', embed_dim or 384)
            self.prototype_cls_loss = MultiCropPrototypeCLSLoss(
                embed_dim=embed_dim_val,
                num_prototypes=num_protos,
                student_temp=prototype_cls_student_temp,
                teacher_temp=prototype_cls_teacher_temp,
                center_momentum=center_momentum or 0.9,
                n_global_crops=2,
            )
        else:
            self.prototype_cls_loss = None
        self.n_local_crops = n_local_crops

        # Store DINOv3-aligned schedule parameters
        self.ema_momentum_start = ema_momentum
        self.ema_momentum_end = ema_momentum_end
        self.teacher_temp_warmup = teacher_temp_warmup
        self.teacher_temp_final = teacher_temp
        self.teacher_temp_warmup_epochs = teacher_temp_warmup_epochs
        self.weight_decay_start = weight_decay
        self.weight_decay_end = weight_decay_end

        # DINOv3-specific parameters
        self.freeze_last_layer_epochs = freeze_last_layer_epochs
        self.layerwise_decay = layerwise_decay
        self.patch_embed_lr_mult = patch_embed_lr_mult
        self.lr_scaling = lr_scaling
        self.batch_size = batch_size
        self.min_lr = min_lr
        self.adamw_beta1 = adamw_beta1
        self.adamw_beta2 = adamw_beta2
        self.dino_head_wd_multiplier = dino_head_wd_multiplier

        # Create schedules (will be initialized in on_fit_start)
        self._momentum_schedule = None
        self._temp_schedule = None
        self._wd_schedule = None
        self._lr_schedule = None  # CosineScheduler for normal params
        self._last_layer_lr_schedule = None  # CosineScheduler for last layer (frozen during freeze epochs)

    def on_fit_start(self):
        """Initialize schedules when training starts."""
        max_epochs = self.hparams.get('max_epochs', 100)

        # Momentum schedule: linear warmup from start to end
        self._momentum_schedule = cosine_schedule(
            base_value=self.ema_momentum_start,
            final_value=self.ema_momentum_end,
            epochs=max_epochs,
        )

        # Temperature schedule: warmup then constant
        def temp_schedule(epoch):
            if epoch < self.teacher_temp_warmup_epochs:
                # Linear warmup
                return self.teacher_temp_warmup + (self.teacher_temp_final - self.teacher_temp_warmup) * (epoch / self.teacher_temp_warmup_epochs)
            return self.teacher_temp_final

        self._temp_schedule = temp_schedule

        # Weight decay schedule
        self._wd_schedule = cosine_schedule(
            base_value=self.weight_decay_start,
            final_value=self.weight_decay_end,
            epochs=max_epochs,
        )

    def on_train_epoch_start(self):
        """Update epoch-level schedules (temperature only - WD is per-step in DINOv3)."""
        epoch = self.current_epoch

        # Update teacher temperature (epoch-level schedule)
        if self._temp_schedule is not None:
            new_temp = self._temp_schedule(epoch)
            self.loss_fn.teacher_temp = new_temp
            # Also update Prototype CLS loss teacher temperature for consistency
            if self.prototype_cls_loss is not None:
                self.prototype_cls_loss.teacher_temp = new_temp

    def on_train_batch_start(self, batch, batch_idx):
        """
        Apply optimizer schedules at each step (EXACT DINOv3 match).

        This matches DINOv3's apply_optim_scheduler() function exactly:
        - lr_schedule for normal parameters
        - last_layer_lr_schedule for last layer (frozen during freeze epochs)
        - wd_schedule for weight decay
        """
        # Get current global step
        it = self.global_step

        # Get scheduled values
        lr = self._lr_schedule[it] if self._lr_schedule is not None else self._scaled_lr
        last_layer_lr = self._last_layer_lr_schedule[it] if self._last_layer_lr_schedule is not None else lr
        wd = self._wd_schedule(self.current_epoch) if self._wd_schedule is not None else self.weight_decay_start

        # Apply to all parameter groups (EXACT DINOv3 apply_optim_scheduler logic)
        for param_group in self.trainer.optimizers[0].param_groups:
            is_last_layer = param_group.get("is_last_layer", False)
            lr_multiplier = param_group.get("lr_multiplier", 1.0)
            wd_multiplier = param_group.get("wd_multiplier", 1.0)

            # Update weight decay
            param_group["weight_decay"] = wd * wd_multiplier

            # Update learning rate (different for last layer)
            if is_last_layer:
                param_group["lr"] = last_layer_lr * lr_multiplier
            else:
                param_group["lr"] = lr * lr_multiplier

    def _get_current_momentum(self) -> float:
        """Get current EMA momentum based on schedule."""
        if self._momentum_schedule is not None:
            return self._momentum_schedule(self.current_epoch)
        return self.ema_momentum_start

    def shared_step(self, batch, is_train: bool):
        """
        Shared training/validation step with Multi-Crop Prototype Learning.

        Supports two modes:
        1. Multi-crop mode (with local crops): Uses forward_multicrop
        2. Legacy mode (global crops only): Uses original forward

        Loss computation:
        - Mask Loss: Patch-level masked prototype prediction
        - Prototype CLS Loss: CLS-level prototype matching across crops
        - KoLeo Loss: Uniformity regularization
        """
        # Check if we have local crops (multi-crop mode)
        has_local_crops = "local_crops" in batch and batch["local_crops"] is not None

        if has_local_crops:
            # ============================================
            # Multi-Crop Mode (Global + Local crops)
            # ============================================
            global_crops = batch["global_crops"]  # List of 2 tensors
            local_crops = batch["local_crops"]    # List of N tensors
            global_masks = batch["masks"]         # List of 2 mask tensors

            # Optimization: Channels Last
            if global_crops[0].device.type == 'cuda':
                global_crops = [g.contiguous(memory_format=torch.channels_last) for g in global_crops]
                local_crops = [lc.contiguous(memory_format=torch.channels_last) for lc in local_crops]

            # Move masks to device
            global_masks = [m.to(self.device, non_blocking=True) for m in global_masks]

            # Multi-Crop Forward Pass
            outputs = self.model.forward_multicrop(global_crops, local_crops, global_masks)

            student_patch_outputs = outputs["student_patch_outputs"]
            teacher_patch_logits = outputs["teacher_patch_logits"]
            teacher_global_cls = outputs["teacher_global_cls"]
            student_global_cls = outputs["student_global_cls"]
            student_local_cls = outputs["student_local_cls"]

            # 1. Mask Loss (on global crops only): student vs teacher rows at the masked positions, (M, K) each
            mask_loss = 0.0
            for s_logits, t_targets in zip(student_patch_outputs, outputs["teacher_patch_targets"]):
                mask_loss += self.loss_fn(s_logits, t_targets)
            mask_loss /= len(student_patch_outputs)

            # Update centers
            self.loss_fn.update_center(teacher_patch_logits.mean(dim=0))

            # 2. Multi-Crop Prototype CLS Loss CONTRIBUTION
            # Combines all student CLS (global + local) predicting teacher global CLS prototypes
            proto_cls_loss = torch.tensor(0.0, device=self.device)
            proto_cls_metrics = {}
            if self.prototype_cls_loss_weight > 0:
                # Combine student CLS tokens: (2, B, D) and (N, B, D) → (2+N, B, D)
                if student_local_cls is not None:
                    all_student_cls = torch.cat([student_global_cls, student_local_cls], dim=0)
                else:
                    all_student_cls = student_global_cls

                proto_cls_loss, proto_cls_metrics = self.prototype_cls_loss(
                    teacher_cls_tokens=teacher_global_cls,
                    student_cls_tokens=all_student_cls,
                )

            # 3. KoLeo Loss (on all student CLS tokens)
            koleo_loss = torch.tensor(0.0, device=self.device)
            if self.koleo_loss_weight > 0:
                # Flatten all student CLS: (2+N, B, D) → ((2+N)*B, D)

                # Compute KoLeo per crop type and average
                n_global = student_global_cls.shape[0]
                B = student_global_cls.shape[1]
                global_cls_flat = student_global_cls.reshape(n_global * B, -1)
                koleo_loss = self.koleo_loss(global_cls_flat)

                if student_local_cls is not None:
                    n_local = student_local_cls.shape[0]
                    local_cls_flat = student_local_cls.reshape(n_local * B, -1)
                    koleo_loss = (koleo_loss + self.koleo_loss(local_cls_flat)) / 2

            # For logging compatibility
            teacher_logits = teacher_patch_logits[0]  # Use first global crop for metrics

        else:
            # ============================================
            # Legacy Mode (Global crops only, backward compatible)
            # ============================================
            if "teacher_images" in batch:
                teacher_images = batch["teacher_images"]
                student_images = batch["student_images"]
            else:
                teacher_images = batch["images"]
                student_images = batch["images"]

            masks = batch["masks"]

            if teacher_images.device.type == 'cuda':
                teacher_images = teacher_images.contiguous(memory_format=torch.channels_last)
                student_images = student_images.contiguous(memory_format=torch.channels_last)

            masks = [m.to(self.device, non_blocking=True) for m in masks]

            # Legacy Forward Pass
            student_outputs, teacher_logits, student_cls_tokens, teacher_cls_token, _ = self.model(
                teacher_images, student_images, masks
            )

            # 1. Mask Loss
            mask_loss = 0.0
            for i, s_logits in enumerate(student_outputs):           # (M_v, K) vs the teacher rows at the same positions
                mask_loss += self.loss_fn(s_logits, teacher_logits[masks[i]])
            mask_loss /= len(masks)
            self.loss_fn.update_center(teacher_logits)

            # 2. Prototype CLS Loss (simplified for legacy mode)
            proto_cls_loss = torch.tensor(0.0, device=self.device)
            proto_cls_metrics = {}
            if self.prototype_cls_loss_weight > 0:
                num_views = len(masks)
                # Reshape for CLS loss: (B*views, D) → (views, B, D)
                B = teacher_logits.shape[0]
                student_cls_reshaped = student_cls_tokens.reshape(num_views, B, -1)
                teacher_cls_reshaped = teacher_cls_token.unsqueeze(0)  # (1, B, D)

                proto_cls_loss, proto_cls_metrics = self.prototype_cls_loss(
                    teacher_cls_tokens=teacher_cls_reshaped,
                    student_cls_tokens=student_cls_reshaped,
                )

            # 3. KoLeo Loss
            koleo_loss = torch.tensor(0.0, device=self.device)
            num_views = len(masks)
            if self.koleo_loss_weight > 0:
                cls_per_view = torch.chunk(student_cls_tokens, num_views, dim=0)
                koleo_loss = sum(self.koleo_loss(cls_v) for cls_v in cls_per_view) / num_views

        # Total loss
        total_loss = (
            mask_loss +
            self.prototype_cls_loss_weight * proto_cls_loss +
            self.koleo_loss_weight * koleo_loss
        )

        return total_loss, mask_loss, koleo_loss, proto_cls_loss, teacher_logits, proto_cls_metrics

    def training_step(self, batch, batch_idx):
        total, mask, koleo, proto_cls, teacher_logits, proto_cls_metrics = self.shared_step(batch, is_train=True)

        # Calculate prototype utilization (unique prototypes used) — logging only. torch.unique forces a
        # host sync, so compute it every 10th step and reuse the last value in between.
        if batch_idx % 10 == 0 or not hasattr(self, "_last_util"):
            self._last_util = torch.unique(teacher_logits.argmax(dim=-1)).numel()
        unique = self._last_util

        # Get current schedule values for logging
        it = self.global_step
        current_lr = self._lr_schedule[it] if self._lr_schedule is not None else 0.0
        current_last_layer_lr = self._last_layer_lr_schedule[it] if self._last_layer_lr_schedule is not None else 0.0
        current_wd = self._wd_schedule(self.current_epoch) if self._wd_schedule is not None else self.weight_decay_start

        # Losses — shown on progress bar
        loss_dict = {
            "train_loss": total,
            "train_mask": mask,
            "train_koleo": koleo,
            "train_proto_cls": proto_cls,
        }
        if proto_cls_metrics:
            if "cls_global_loss" in proto_cls_metrics:
                loss_dict["train_cls_global"] = proto_cls_metrics["cls_global_loss"]
            if "cls_local_loss" in proto_cls_metrics:
                loss_dict["train_cls_local"] = proto_cls_metrics["cls_local_loss"]

        self.log_dict(loss_dict, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)

        # Schedule values + extras — logged to file only
        self.log_dict({
            "train_util": float(unique),
            "lr": current_lr,
            "last_layer_lr": current_last_layer_lr,
            "wd": current_wd,
            "mom": self._get_current_momentum(),
            "teacher_temp": self.loss_fn.teacher_temp,
        }, on_step=True, on_epoch=True, prog_bar=False, sync_dist=True)
        return total

    @torch.no_grad()
    def on_train_batch_end(self, outputs, batch, batch_idx):
        # Use scheduled momentum
        momentum = self._get_current_momentum()
        self.model.update_teacher(momentum)

    def validation_step(self, batch, batch_idx):
        total, mask, koleo, proto_cls, teacher_logits, proto_cls_metrics = self.shared_step(batch, is_train=False)
        unique = torch.unique(teacher_logits.argmax(dim=-1)).numel()

        log_dict = {
            "val_loss": total,
            "val_mask": mask,
            "val_koleo": koleo,
            "val_proto_cls": proto_cls,
            "val_util": float(unique)
        }

        # Add prototype CLS metrics if available (DINOv3-style: global + local separately)
        if proto_cls_metrics:
            if "cls_global_loss" in proto_cls_metrics:
                log_dict["val_cls_global"] = proto_cls_metrics["cls_global_loss"]
            if "cls_local_loss" in proto_cls_metrics:
                log_dict["val_cls_local"] = proto_cls_metrics["cls_local_loss"]
            if "cls_proto_util" in proto_cls_metrics:
                log_dict["val_cls_proto_util"] = proto_cls_metrics["cls_proto_util"]

        self.log_dict(log_dict, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)

    def configure_optimizers(self):
        """
        Configure optimizer and scheduler with EXACT DINOv3 dynamics.

        DINOv3 features implemented (EXACT match):
        1. LR scaling: sqrt_wrt_1024 rule based on effective batch size
        2. Layerwise LR decay (default: 0.9)
        3. Patch embedding LR multiplier (default: 0.2)
        4. No weight decay on bias, norm, gamma (EXACT DINOv3 match)
        5. Two separate schedules: lr_schedule and last_layer_lr_schedule
        6. last_layer_lr_schedule is 0 during freeze_last_layer_epochs
        7. AdamW with betas=(0.9, 0.999)
        """
        base_lr = self.hparams.get('lr', 1e-3)
        weight_decay = self.hparams.get('weight_decay', 0.04)
        warmup_epochs = self.hparams.get('warmup_epochs', 10)
        max_epochs = self.hparams.get('max_epochs', 100)
        vit_depth = self.hparams.get('vit_depth', 12)

        # Get world size for LR scaling
        world_size = self.trainer.world_size if self.trainer else 1

        # Scale LR based on effective batch size (DINOv3 style)
        scaled_lr = scale_lr(
            base_lr,
            self.batch_size,
            world_size,
            self.lr_scaling
        )

        print(f"[DINOv3 LR Scaling] base_lr={base_lr}, batch_size={self.batch_size}, "
              f"world_size={world_size}, scaling={self.lr_scaling} → scaled_lr={scaled_lr:.6f}")

        # Fused optimizer is incompatible with AMP gradient clipping
        # Only use fused when no gradient clipping is configured
        gradient_clip_val = self.trainer.gradient_clip_val if self.trainer else 0.0
        use_fused = (self.device.type == 'cuda') and (gradient_clip_val is None or gradient_clip_val == 0.0)

        # Build parameter groups with DINOv3-style layerwise decay and weight decay filtering
        param_groups = self._build_param_groups(
            scaled_lr=scaled_lr,
            weight_decay=weight_decay,
            num_layers=vit_depth,
        )

        # EXACT DINOv3: AdamW with explicit betas
        optimizer = AdamW(
            param_groups,
            betas=(self.adamw_beta1, self.adamw_beta2),
            fused=use_fused,
        )
        if not use_fused and self.device.type == 'cuda':
            print("[DINOv3 Optimizer] Fused AdamW disabled (gradient clipping enabled)")
        print(f"[DINOv3 Optimizer] AdamW betas=({self.adamw_beta1}, {self.adamw_beta2})")

        # Calculate total steps and phase durations
        total_steps = self.trainer.estimated_stepping_batches
        steps_per_epoch = total_steps // max_epochs

        freeze_steps = self.freeze_last_layer_epochs * steps_per_epoch
        warmup_steps = warmup_epochs * steps_per_epoch

        print(f"[DINOv3 Scheduler] total_steps={total_steps}, steps_per_epoch={steps_per_epoch}")
        print(f"[DINOv3 Scheduler] freeze_last_layer_steps={freeze_steps}, warmup_steps={warmup_steps}")

        # Create DINOv3-style CosineScheduler for normal parameters
        # NO freeze phase - freeze only applies to last layer
        self._lr_schedule = CosineScheduler(
            base_value=scaled_lr,
            final_value=self.min_lr,
            total_iters=total_steps,
            warmup_iters=warmup_steps,
            start_warmup_value=0.0,
            freeze_iters=0,  # No freeze for normal params
        )

        # Create separate schedule for last_layer with freeze phase
        # EXACT DINOv3: last_layer_lr_schedule has 0 during freeze epochs
        self._last_layer_lr_schedule = CosineScheduler(
            base_value=scaled_lr,
            final_value=self.min_lr,
            total_iters=total_steps,
            warmup_iters=warmup_steps,
            start_warmup_value=0.0,
            freeze_iters=0,  # Same warmup structure
        )
        # EXACT DINOv3: Zero out the first freeze_steps for last layer
        self._last_layer_lr_schedule.schedule[:freeze_steps] = 0.0

        # Store scaled_lr for manual schedule application
        self._scaled_lr = scaled_lr

        # We return optimizer without a scheduler - we'll apply schedules manually
        # in on_train_batch_start to match DINOv3's apply_optim_scheduler exactly
        return optimizer

    def _build_param_groups(
        self,
        scaled_lr: float,
        weight_decay: float,
        num_layers: int,
    ) -> List[Dict[str, Any]]:
        """
        Build parameter groups with EXACT DINOv3-style layerwise LR decay and weight decay filtering.

        EXACT match to DINOv3 param_groups.py:
        - Layerwise LR decay (lower LR for earlier layers)
        - Patch embedding LR multiplier
        - No weight decay on bias, norm, gamma (NOT learned tokens)
        - dino_head_wd_multiplier for DINO head
        - Last layer tracking for freeze phase
        """
        # Group parameters by their properties
        param_groups_dict = {}

        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue

            # Calculate LR multiplier (DINOv3 style)
            lr_mult = 1.0

            if "student_encoder" in name or "backbone" in name:
                # Apply layerwise decay
                lr_mult = get_vit_lr_decay_rate(name, self.layerwise_decay, num_layers)
                lr_mult *= self.backbone_lr_scale

                # Apply patch embedding multiplier (DINOv3 style)
                if "patch_embed" in name:
                    lr_mult *= self.patch_embed_lr_mult

            # Determine weight decay multiplier (EXACT DINOv3 match)
            wd_mult = 1.0

            # EXACT DINOv3: dino_head gets special wd multiplier
            if "dino_head" in name:
                wd_mult = self.dino_head_wd_multiplier

            # EXACT DINOv3: No weight decay on bias, norm, gamma, fourier_w
            if should_skip_weight_decay(name):
                wd_mult = 0.0

            # EXACT DINOv3: is_last_layer for prototype/last_layer params
            is_last_layer = "last_layer" in name or "prototype" in name

            # Create group key based on lr_mult, wd_mult, and is_last_layer
            group_key = f"lr_{lr_mult:.6f}_wd_{wd_mult:.2f}_last_{is_last_layer}"

            if group_key not in param_groups_dict:
                param_groups_dict[group_key] = {
                    "params": [],
                    "lr": scaled_lr * lr_mult,
                    "weight_decay": weight_decay * wd_mult,
                    "is_last_layer": is_last_layer,
                    "lr_multiplier": lr_mult,
                    "wd_multiplier": wd_mult,
                }

            param_groups_dict[group_key]["params"].append(param)

        # Log parameter group summary
        print(f"[DINOv3 Param Groups] Created {len(param_groups_dict)} parameter groups:")
        for key, group in param_groups_dict.items():
            n_params = len(group["params"])
            is_last = group["is_last_layer"]
            print(f"  {key}: {n_params} params, lr={group['lr']:.6f}, wd={group['weight_decay']:.4f}, is_last_layer={is_last}")

        return list(param_groups_dict.values())


def create_motif_learner_dinov3(pretrained_path: Optional[str] = None, **kwargs) -> MotifLearner:
    """
    Factory function to create a MotifLearner with EXACT DINOv3-style defaults.

    All parameters match DINOv3 ssl_default_config.yaml EXACTLY:
    - LR: 1e-3 with sqrt_wrt_1024 scaling
    - Warmup: 10 epochs
    - Freeze last layer: 1 epoch (ONLY last layer frozen, not all params)
    - Layerwise decay: 0.9
    - Patch embed LR mult: 0.2
    - Weight decay: 0.04 → 0.4 (cosine)
    - EMA momentum: 0.992 → 1.0 (cosine)
    - Teacher temp: 0.04 → 0.07 (linear warmup over 30 epochs)
    - AdamW betas: (0.9, 0.999)
    - Gradient clip: 3.0
    """
    defaults = dict(
        # Model architecture
        img_size=256,  # DINOv3 official default
        patch_size=16,
        embed_dim=384,
        vit_depth=12,
        vit_heads=6,
        mlp_ratio=4.0,
        num_storage_tokens=4,  # DINOv3 register tokens
        num_prototypes=128,
        projector_dim=256,
        decoder_embed_dim=192,
        decoder_depth=4,
        decoder_num_heads=6,
        drop_path_rate=0.1,  # Conservative for low-data regime
        pretrained_path=pretrained_path,
        # Training config (EXACT DINOv3 defaults from ssl_default_config.yaml)
        lr=1e-3,  # DINOv3 base LR (optim.lr)
        min_lr=1e-6,  # DINOv3 default (optim.min_lr)
        lr_scaling="sqrt_wrt_1024",  # DINOv3 scaling rule (optim.scaling_rule)
        warmup_epochs=10,  # DINOv3 default (optim.warmup_epochs)
        freeze_last_layer_epochs=1,  # DINOv3 default (optim.freeze_last_layer_epochs)
        max_epochs=100,  # DINOv3 default (optim.epochs)
        # Backbone settings
        freeze_backbone=False,
        backbone_lr_scale=1.0,  # 1.0 for pretraining
        layerwise_decay=0.9,  # DINOv3 default (optim.layerwise_decay)
        patch_embed_lr_mult=0.2,  # DINOv3 default (optim.patch_embed_lr_mult)
        # Weight decay (cosine schedule)
        weight_decay=0.04,  # DINOv3 start (optim.weight_decay)
        weight_decay_end=0.4,  # DINOv3 end (optim.weight_decay_end)
        # EMA momentum (cosine schedule)
        ema_momentum=0.992,  # DINOv3 start (teacher.momentum_teacher)
        ema_momentum_end=1.0,  # DINOv3 end (teacher.final_momentum_teacher)
        # Teacher temperature (linear warmup)
        teacher_temp_warmup=0.04,  # DINOv3 start (teacher.warmup_teacher_temp)
        teacher_temp=0.07,  # DINOv3 end (teacher.teacher_temp)
        teacher_temp_warmup_epochs=30,  # DINOv3 default (teacher.warmup_teacher_temp_epochs)
        # AdamW optimizer (EXACT DINOv3 defaults)
        adamw_beta1=0.9,  # DINOv3 default (optim.adamw_beta1)
        adamw_beta2=0.999,  # DINOv3 default (optim.adamw_beta2)
        # Weight decay multipliers
        dino_head_wd_multiplier=1.0,  # DINOv3 default (optim.dino_head_wd_multiplier)
        # KoLeo loss
        koleo_loss_weight=0.1,  # DINOv3 default (dino.koleo_loss_weight)
        koleo_distributed=False,
        # Multi-Crop Prototype CLS Loss
        prototype_cls_loss_weight=1.0,
        prototype_cls_student_temp=0.1,
        prototype_cls_teacher_temp=0.04,
        n_local_crops=8,
    )
    defaults.update(kwargs)
    return MotifLearner(**defaults)


# The class was called MotifLearner in the original repository.
MotifLearner = KodiakLearner
