# losses.py

from typing import Optional

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F

# ============================================================
#  KoLeo Loss (from DINOv3)
#  Kozachenko-Leonenko entropic loss regularizer
#  Reference: Sablayrolles et al. - 2018 - Spreading vectors for similarity search
# ============================================================

class KoLeoLoss(nn.Module):
    """
    Kozachenko-Leonenko entropic loss regularizer.

    Encourages uniform spreading of representations in the embedding space
    by maximizing the distance to the nearest neighbor.

    This helps prevent representation collapse without requiring
    multiple views or histogram matching.
    """

    def __init__(self):
        super().__init__()
        self.pdist = nn.PairwiseDistance(2, eps=1e-8)

    def pairwise_NNs_inner(self, x: torch.Tensor) -> torch.Tensor:
        """
        Find nearest neighbors using inner product for L2-normalized vectors.
        Uses Torch rather than Faiss to remain on GPU.

        Args:
            x: (B, D) L2-normalized features

        Returns:
            indices: (B,) index of nearest neighbor for each sample
        """
        # Pairwise dot products (= inverse distance for normalized vectors)
        dots = torch.mm(x, x.t())
        n = x.shape[0]
        dots.view(-1)[:: (n + 1)].fill_(-1)  # Fill diagonal with -1 to exclude self
        _, indices = torch.max(dots, dim=1)  # max inner prod -> min distance
        return indices

    def forward(self, student_output: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
        """
        Compute KoLeo loss.

        Args:
            student_output: (B, D) backbone output features (pre-head)
            eps: small constant for numerical stability

        Returns:
            scalar loss value
        """
        with torch.autocast(student_output.device.type, enabled=False):
            student_output = F.normalize(student_output.float(), eps=eps, p=2, dim=-1)
            indices = self.pairwise_NNs_inner(student_output)
            distances = self.pdist(student_output, student_output[indices])  # B
            loss = -torch.log(distances + eps).mean()
        return loss


class KoLeoLossDistributed(nn.Module):
    """
    Distributed version of KoLeo loss for multi-GPU training.

    Gathers features from all GPUs before computing nearest neighbors,
    providing a larger pool for more accurate entropy estimation.
    """

    def __init__(self, topk: int = 1):
        super().__init__()
        self.pdist = nn.PairwiseDistance(2, eps=1e-8)
        self.topk = topk

    def pairwise_NNs_inner(self, x: torch.Tensor, all_x: torch.Tensor, rank: int) -> torch.Tensor:
        """
        Find nearest neighbors across all gathered features.

        Args:
            x: (local_B, D) local features
            all_x: (global_B, D) gathered features from all GPUs
            rank: current GPU rank

        Returns:
            indices: (local_B, topk) indices of nearest neighbors in all_x
        """
        # Pairwise dot products (= inverse distance)
        dots = torch.mm(x, all_x.t())  # local_B x global_B
        local_B, global_B = dots.shape
        dots.view(-1)[rank * local_B :: (global_B + 1)].fill_(-1)  # Fill diagonal with -1
        _, indices = torch.topk(dots, dim=1, k=self.topk)  # max inner prod -> min distance
        return indices

    def forward(self, student_output: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
        """
        Compute distributed KoLeo loss.

        Args:
            student_output: (B, D) backbone output features (pre-head)
            eps: small constant for numerical stability

        Returns:
            scalar loss value
        """
        with torch.autocast(student_output.device.type, enabled=False):
            student_output = F.normalize(student_output.float(), eps=eps, p=2, dim=-1)  # local_B x D

            if dist.is_available() and dist.is_initialized():
                all_student_outputs = torch.cat(
                    [torch.zeros_like(student_output) for _ in range(dist.get_world_size())],
                    dim=0
                )
                dist.all_gather_into_tensor(all_student_outputs, student_output)
                # Make sure gradients flow back
                all_student_outputs = torch.cat(
                    dist.nn.all_gather(student_output), dim=0
                )
                rank = dist.get_rank()
            else:
                all_student_outputs = student_output
                rank = 0

            with torch.no_grad():
                indices = self.pairwise_NNs_inner(student_output, all_student_outputs, rank)  # local_B x topk

            student_output_expanded = (
                student_output.unsqueeze(1).repeat(1, self.topk, 1).flatten(0, 1)
            )  # (local_B * topk) x D
            distances = self.pdist(
                student_output_expanded,
                all_student_outputs[indices].flatten(0, 1)
            )  # (local_B * topk)
            loss = -torch.log(distances.float() + eps).mean()

        return loss


# ============================================================
#  Sinkhorn-Knopp (balanced assignments)
# ============================================================

@torch.no_grad()
def sinkhorn_knopp(
    logits: torch.Tensor,
    epsilon: float = 0.05,
    niters: int = 3,
) -> torch.Tensor:
    """
    Balanced assignments as in SwAV / DINO.

    Args:
        logits:  (M, K) or (..., K)
        epsilon: temperature / entropic regularization
        niters:  number of Sinkhorn iterations
    """
    if logits.dim() > 2:
        logits = logits.view(-1, logits.shape[-1])  # (M, K)

    logits = logits.float()
    logits = logits / epsilon
    logits = logits - logits.max(dim=1, keepdim=True).values

    Q = torch.exp(logits).T  # (K, M)
    K_, M = Q.shape
    eps = 1e-8

    for _ in range(niters):
        Q = Q / (Q.sum(dim=1, keepdim=True) + eps)
        Q = Q / (K_ * (Q.sum(dim=0, keepdim=True) + eps))

    Q = Q.T
    Q = Q / (Q.sum(dim=1, keepdim=True) + eps)
    return Q


# ============================================================
#  Mask Loss (Masked Prototype Prediction)
# ============================================================

class MaskLoss(nn.Module):
    """
    Mask Loss (Masked Prototype Prediction):

    - Teacher logits (full image, unmasked) are centered
      and passed through Sinkhorn to obtain balanced targets.
    - Student predicts prototypes only on MASKED patches.

    Args:
        use_sinkhorn: If False, use softmax instead of Sinkhorn-Knopp
        use_centering: If False, skip EMA centering of teacher logits
    """

    def __init__(
        self,
        num_prototypes: int,
        teacher_temp: float = 0.04,
        center_momentum: float = 0.9,
        sinkhorn_iters: int = 3,
        use_sinkhorn: bool = True,
        use_centering: bool = True,
    ):
        super().__init__()
        self.teacher_temp = float(teacher_temp)
        self.center_momentum = float(center_momentum)
        self.sinkhorn_iters = int(sinkhorn_iters)
        self.use_sinkhorn = use_sinkhorn
        self.use_centering = use_centering

        # Keep center as shape (1, 1, K)
        self.register_buffer("patch_center", torch.zeros(1, 1, num_prototypes))

    # -------------------------------
    # Center update (EMA)
    # -------------------------------
    @torch.no_grad()
    def _update_patch_center(self, teacher_logits: torch.Tensor):
        """
        teacher_logits: (B, N, K)
        """
        logits_f = teacher_logits.detach().float()
        batch_center = logits_f.mean(dim=(0, 1), keepdim=True)  # (1, 1, K)

        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(batch_center)
            batch_center /= dist.get_world_size()

        self.patch_center.mul_(self.center_momentum).add_(
            batch_center, alpha=1 - self.center_momentum
        )

    def update_center(self, teacher_logits: torch.Tensor):
        self._update_patch_center(teacher_logits)

    # -------------------------------
    # Forward loss
    # -------------------------------
    def forward(
        self,
        student_logits: torch.Tensor,  # (M, K) rows at the masked positions, or (B, N, K) with ``mask``
        teacher_logits: torch.Tensor,  # same layout as student_logits
        mask: Optional[torch.Tensor] = None,   # (B, N) True = masked; only for the (B, N, K) layout
    ) -> torch.Tensor:
        """
        Cross-entropy between the balanced teacher targets and the student predictions on the masked patches.
        The model already hands over the (M, K) rows at the masked positions (student and teacher in the same
        order); the full (B, N, K) grids plus ``mask`` are accepted too and gathered here.
        """
        K = student_logits.shape[-1]
        device = student_logits.device

        if mask is not None:
            mask_flat = mask.reshape(-1)
            student_m = student_logits.reshape(-1, K)[mask_flat]  # (M, K)
            teacher_m = teacher_logits.reshape(-1, K)[mask_flat]  # (M, K)
        else:
            student_m = student_logits.reshape(-1, K)
            teacher_m = teacher_logits.reshape(-1, K)
        if student_m.shape[0] == 0:
            return student_logits.new_zeros(())

        # --- teacher balanced targets ---
        with torch.no_grad():
            if self.use_centering:
                center = self.patch_center.to(device)  # (1, 1, K)
                teacher_centered = teacher_m.float() - center  # (M, K)
            else:
                teacher_centered = teacher_m.float()

            if self.use_sinkhorn:
                # Sinkhorn-Knopp for balanced assignments (default)
                teacher_probs = sinkhorn_knopp(
                    teacher_centered,
                    epsilon=self.teacher_temp,
                    niters=self.sinkhorn_iters,
                ).detach()
            else:
                # A1 ablation: use softmax instead of Sinkhorn
                teacher_probs = F.softmax(teacher_centered / self.teacher_temp, dim=-1).detach()

        # Student log-probs
        student_log_probs = F.log_softmax(student_m.float(), dim=-1)

        # Cross-entropy between teacher targets and student predictions
        loss = -(teacher_probs * student_log_probs).sum(dim=-1).mean()
        return loss


# Backward compatibility alias
MPPLoss = MaskLoss
