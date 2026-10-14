# =============================================================================
# Collate Functions
# =============================================================================
# Collate functions for pretraining and classification.
# Handles masking and multi-crop batching.
# =============================================================================

import random
from typing import List, Tuple

import numpy as np
import torch

from .masking import MaskingGenerator


class KodiakCollate:
    """
    Collate function implementing DINOv3-style VARIABLE masking with cross-view support.

    Cross-View Mode (pretraining):
    - Input batch has 'teacher_image' and 'student_image' (different augmentations)
    - Teacher sees full image (crop1), Student sees masked image (crop2)
    - Returns {"teacher_images": ..., "student_images": ..., "masks": ...}

    Single-View Mode (legacy):
    - Input batch has 'images' only
    - Same image used for both teacher and student
    - Returns {"images": ..., "masks": ...}

    Masking:
    - For each batch, a fraction `mask_probability` of samples are masked.
    - For those, the mask ratio is linearly sampled between mask_ratio_tuple[0/1].
    """
    def __init__(
        self,
        mask_generator: MaskingGenerator,
        mask_ratio_tuple: Tuple[float, float],
        mask_probability: float,
        num_masks: int = 2,
        num_patches_total: int = 196
    ):
        self.mask_generator = mask_generator
        self.mask_ratio_tuple = mask_ratio_tuple
        self.mask_probability = mask_probability
        self.num_masks = num_masks
        self.N = num_patches_total

    def __call__(self, batch):
        # Check if cross-view mode (pretraining with two crops)
        is_cross_view = "teacher_image" in batch[0]

        if is_cross_view:
            teacher_images = torch.stack([x["teacher_image"] for x in batch])
            student_images = torch.stack([x["student_image"] for x in batch])
        else:
            # Legacy single-view mode
            images = torch.stack([x["images"] for x in batch])
            teacher_images = images
            student_images = images

        B = teacher_images.shape[0]

        masks_list: List[torch.Tensor] = []

        for _ in range(self.num_masks):
            n_samples_masked = int(B * self.mask_probability)
            probs = np.linspace(self.mask_ratio_tuple[0], self.mask_ratio_tuple[1], n_samples_masked + 1)

            batch_masks_np = []

            # Variable masks
            for i in range(n_samples_masked):
                prob_max = probs[i + 1]
                num_tokens_to_mask = int(self.N * prob_max)
                m = self.mask_generator(num_tokens_to_mask)  # boolean 2D array
                batch_masks_np.append(m.flatten())

            # Unmasked samples (0 masked tokens)
            if n_samples_masked < B:
                for _ in range(n_samples_masked, B):
                    m = self.mask_generator(0)
                    batch_masks_np.append(m.flatten())

            random.shuffle(batch_masks_np)
            np_stack = np.stack(batch_masks_np)
            masks_list.append(torch.from_numpy(np_stack))

        # Build result dict
        if is_cross_view:
            result = {
                "teacher_images": teacher_images,
                "student_images": student_images,
                "masks": masks_list
            }
        else:
            # Legacy format for backward compatibility
            result = {"images": teacher_images, "masks": masks_list}

        # Preserve labels if present (needed for classification)
        if "label" in batch[0]:
            labels = torch.tensor([x["label"] for x in batch])
            result["labels"] = labels

        return result


class MultiCropKodiakCollate:
    """
    Collate function for Multi-Crop KODIAK pretraining with Prototype CLS Loss.

    Handles the format for forward_multicrop:
    - global_crops: List of 2 tensors for teacher + student encoding
    - local_crops: List of N tensors for student CLS encoding only
    - masks: List of 2 mask tensors for global crops only

    Input batch format (from MultiCropPretrainingTransform):
    {
        "global_crops": List of 2 tensors (3, 224, 224),
        "local_crops": List of N tensors (3, 96, 96),
    }

    Output format (for forward_multicrop):
    {
        "global_crops": List of 2 tensors (B, 3, 224, 224),
        "local_crops": List of N tensors (B, 3, 96, 96),
        "masks": List of 2 tensors (B, N_patches),
    }
    """
    def __init__(
        self,
        mask_generator: MaskingGenerator,
        mask_ratio_tuple: Tuple[float, float],
        mask_probability: float,
        num_patches_total: int = 196,
    ):
        self.mask_generator = mask_generator
        self.mask_ratio_tuple = mask_ratio_tuple
        self.mask_probability = mask_probability
        self.N = num_patches_total

    def __call__(self, batch):
        B = len(batch)

        # ============================================================
        # Stack global crops: List[List[tensor]] -> List[tensor(B, ...)]
        # ============================================================
        n_global = len(batch[0]["global_crops"])
        global_crops = []
        for g_idx in range(n_global):
            stacked = torch.stack([x["global_crops"][g_idx] for x in batch])
            global_crops.append(stacked)

        # ============================================================
        # Stack local crops: List[List[tensor]] -> List[tensor(B, ...)]
        # ============================================================
        local_crops = None
        if batch[0]["local_crops"] and len(batch[0]["local_crops"]) > 0:
            n_local = len(batch[0]["local_crops"])
            local_crops = []
            for l_idx in range(n_local):
                stacked = torch.stack([x["local_crops"][l_idx] for x in batch])
                local_crops.append(stacked)

        # ============================================================
        # Generate masks for global crops only (2 masks)
        # ============================================================
        masks_list: List[torch.Tensor] = []

        for _ in range(n_global):
            n_samples_masked = int(B * self.mask_probability)
            probs = np.linspace(self.mask_ratio_tuple[0], self.mask_ratio_tuple[1], n_samples_masked + 1)

            batch_masks_np = []

            for i in range(n_samples_masked):
                prob_max = probs[i + 1]
                num_tokens_to_mask = int(self.N * prob_max)
                m = self.mask_generator(num_tokens_to_mask)
                batch_masks_np.append(m.flatten())

            if n_samples_masked < B:
                for _ in range(n_samples_masked, B):
                    m = self.mask_generator(0)
                    batch_masks_np.append(m.flatten())

            random.shuffle(batch_masks_np)
            np_stack = np.stack(batch_masks_np)
            masks_list.append(torch.from_numpy(np_stack))

        # Build result
        result = {
            "global_crops": global_crops,  # List of (B, 3, H, W)
            "local_crops": local_crops,    # List of (B, 3, h, w) or None
            "green": torch.tensor([int(x.get("green", -1)) for x in batch]),   # the marker drawn into green per sample (-1 = blank)
            "masks": masks_list,           # List of (B, N)
        }

        # Preserve labels if present
        if "label" in batch[0]:
            labels = torch.tensor([x["label"] for x in batch])
            result["labels"] = labels

        return result

# Backward compatibility
MPPCollate = KodiakCollate
MultiCropMPPCollate = MultiCropKodiakCollate
