"""
Multi-Crop Prototype CLS Loss

CLS tokens learn to match in prototype space (K-dimensional discrete concepts)
rather than feature space (D-dimensional continuous features).

Architecture:
    Teacher (2 global crops):
        Global Crop 1 -> Encoder -> CLS -> Prototype Head -> Softmax -> Target 1
        Global Crop 2 -> Encoder -> CLS -> Prototype Head -> Softmax -> Target 2

    Student (2 global + N local crops):
        Global Crop 1 -> Encoder -> CLS -> Prototype Head -> Log-Softmax -> Pred 1
        Global Crop 2 -> Encoder -> CLS -> Prototype Head -> Log-Softmax -> Pred 2
        Local Crop i  -> Encoder -> CLS -> Prototype Head -> Log-Softmax -> Pred 2+i

Loss Structure:
    - Global-to-Global: Student global predicts OTHER teacher global (cross-view)
    - Local-to-Global: Student local predicts ALL teacher globals
"""

from typing import Optional, Tuple

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F


class MultiCropPrototypeCLSLoss(nn.Module):
    """
    Multi-Crop Prototype CLS Loss.

    CLS tokens match in prototype space instead of feature space.

    Args:
        embed_dim: CLS token embedding dimension (e.g., 384)
        num_prototypes: Number of prototypes K (e.g., 4096)
        student_temp: Student temperature (higher = softer, default: 0.1)
        teacher_temp: Teacher temperature (lower = sharper, default: 0.04)
        center_momentum: EMA momentum for center (default: 0.9)
        n_global_crops: Number of global crops (default: 2)
    """

    def __init__(
        self,
        embed_dim: int,
        num_prototypes: int,
        student_temp: float = 0.1,
        teacher_temp: float = 0.04,
        center_momentum: float = 0.9,
        n_global_crops: int = 2,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_prototypes = num_prototypes
        self.student_temp = student_temp
        self.teacher_temp = teacher_temp
        self.center_momentum = center_momentum
        self.n_global_crops = n_global_crops

        # CLS -> Prototype projection head
        self.cls_prototype_head = nn.Sequential(
            nn.Linear(embed_dim, 2048),
            nn.GELU(),
            nn.Linear(2048, 2048),
            nn.GELU(),
            nn.Linear(2048, num_prototypes, bias=False),
        )

        # Center for prototype logits (prevents collapse)
        self.register_buffer("center", torch.zeros(1, num_prototypes))

        self._init_weights()

    def _init_weights(self):
        """Initialize weights."""
        for m in self.cls_prototype_head.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    @torch.no_grad()
    def update_center(self, teacher_logits: torch.Tensor):
        """
        Update center with EMA (prevents collapse to single prototype).

        Args:
            teacher_logits: (B, K) or (n_crops, B, K) teacher prototype logits
        """
        if teacher_logits.dim() == 3:
            teacher_logits = teacher_logits.mean(dim=0)

        batch_center = teacher_logits.mean(dim=0, keepdim=True)

        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(batch_center)
            batch_center /= dist.get_world_size()

        self.center.mul_(self.center_momentum).add_(
            batch_center, alpha=1 - self.center_momentum
        )

    def forward(
        self,
        teacher_cls_tokens: torch.Tensor,
        student_cls_tokens: torch.Tensor,
        epoch: Optional[int] = None,
    ) -> Tuple[torch.Tensor, dict]:
        """
        Compute Multi-Crop Prototype CLS Loss.

        Args:
            teacher_cls_tokens: Teacher CLS from global crops (n_global, B, D) or (B, D)
            student_cls_tokens: Student CLS from all crops (n_crops, B, D)
            epoch: Current epoch (optional)

        Returns:
            loss: Scalar loss value
            metrics: Dict with logging metrics
        """
        # Handle input shapes
        if teacher_cls_tokens.dim() == 2:
            teacher_cls_tokens = teacher_cls_tokens.unsqueeze(0)
        if student_cls_tokens.dim() == 2:
            student_cls_tokens = student_cls_tokens.unsqueeze(0)

        n_teacher_crops, B, D = teacher_cls_tokens.shape
        n_student_crops = student_cls_tokens.shape[0]
        n_local_crops = n_student_crops - self.n_global_crops

        # Project CLS tokens to prototype space
        teacher_logits = torch.stack([
            self.cls_prototype_head(teacher_cls_tokens[i])
            for i in range(n_teacher_crops)
        ], dim=0)

        student_logits = torch.stack([
            self.cls_prototype_head(student_cls_tokens[i])
            for i in range(n_student_crops)
        ], dim=0)

        # Compute teacher targets (sharpened, centered)
        with torch.no_grad():
            teacher_centered = teacher_logits - self.center
            teacher_probs = F.softmax(teacher_centered / self.teacher_temp, dim=-1)
            self.update_center(teacher_logits)

        # Compute student log-probabilities
        student_log_probs = F.log_softmax(student_logits / self.student_temp, dim=-1)

        # Compute cross-entropy loss
        global_loss = 0.0
        local_loss = 0.0
        n_global_terms = 0
        n_local_terms = 0

        # Global-to-Global: Student global predicts OTHER teacher global
        for s_idx in range(self.n_global_crops):
            for t_idx in range(n_teacher_crops):
                if s_idx == t_idx:
                    continue

                loss_term = -torch.sum(
                    teacher_probs[t_idx] * student_log_probs[s_idx],
                    dim=-1
                ).mean()

                global_loss += loss_term
                n_global_terms += 1

        # Local-to-Global: Student local predicts ALL teacher globals
        for s_idx in range(self.n_global_crops, n_student_crops):
            for t_idx in range(n_teacher_crops):
                loss_term = -torch.sum(
                    teacher_probs[t_idx] * student_log_probs[s_idx],
                    dim=-1
                ).mean()

                local_loss += loss_term
                n_local_terms += 1

        # Average each component
        global_loss = global_loss / max(n_global_terms, 1)
        local_loss = local_loss / max(n_local_terms, 1) if n_local_terms > 0 else torch.tensor(0.0, device=student_cls_tokens.device)

        # Total loss
        n_total_terms = n_global_terms + n_local_terms
        total_loss = (global_loss * n_global_terms + local_loss * n_local_terms) / max(n_total_terms, 1)

        # Compute metrics
        with torch.no_grad():
            teacher_assignments = teacher_probs.argmax(dim=-1)
            unique_prototypes = torch.unique(teacher_assignments).numel()

            teacher_entropy = -torch.sum(
                teacher_probs * torch.log(teacher_probs + 1e-8), dim=-1
            ).mean()

        metrics = {
            "cls_loss": total_loss.detach(),
            "cls_global_loss": global_loss.detach(),
            "cls_local_loss": local_loss.detach() if n_local_terms > 0 else 0.0,
            "cls_proto_util": float(unique_prototypes),
            "cls_teacher_entropy": teacher_entropy.detach(),
            "n_global_terms": n_global_terms,
            "n_local_terms": n_local_terms,
            "n_local_crops": n_local_crops,
        }

        return total_loss, metrics

    def get_prototype_distribution(self, cls_token: torch.Tensor) -> torch.Tensor:
        """
        Get prototype distribution for a CLS token.

        Args:
            cls_token: (B, D) CLS token

        Returns:
            distribution: (B, K) prototype probability distribution
        """
        logits = self.cls_prototype_head(cls_token)
        return F.softmax(logits / self.student_temp, dim=-1)

    def get_top_prototypes(
        self,
        cls_token: torch.Tensor,
        top_k: int = 10
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get top-k prototypes for a CLS token.

        Args:
            cls_token: (B, D) CLS token
            top_k: Number of top prototypes

        Returns:
            indices: (B, top_k) prototype indices
            probs: (B, top_k) prototype probabilities
        """
        dist = self.get_prototype_distribution(cls_token)
        probs, indices = torch.topk(dist, k=top_k, dim=-1)
        return indices, probs
