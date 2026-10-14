# =============================================================================
# DINOv3 Mask Generator (Block Masking)
# =============================================================================
# Block masking generator for Masked Prototype Prediction.
# Uses NumPy for speed (CPU-side).
# =============================================================================

import math
import random
from typing import Optional, Tuple

import numpy as np


class MaskingGenerator:
    """
    DINOv3 Block Masking Generator.
    Uses NumPy for speed (CPU-side).
    """
    def __init__(
        self,
        input_size: Tuple[int, int],
        num_masking_patches: Optional[int] = None,
        min_num_patches: int = 4,
        max_num_patches: Optional[int] = None,
        min_aspect: float = 0.3,
        max_aspect: Optional[float] = None,
    ):
        if not isinstance(input_size, tuple):
            input_size = (input_size,) * 2
        self.height, self.width = input_size

        self.num_patches = self.height * self.width
        self.num_masking_patches = num_masking_patches

        self.min_num_patches = min_num_patches

        # Ensure max_num_patches is not None
        if max_num_patches is None:
            self.max_num_patches = num_masking_patches if num_masking_patches is not None else self.num_patches
        else:
            self.max_num_patches = max_num_patches

        max_aspect = max_aspect or 1 / min_aspect
        self.log_aspect_ratio = (math.log(min_aspect), math.log(max_aspect))

    def get_shape(self):
        return self.height, self.width

    def _mask(self, mask, max_mask_patches):
        delta = 0
        for _ in range(10):
            target_area = random.uniform(self.min_num_patches, max_mask_patches)
            aspect_ratio = math.exp(random.uniform(*self.log_aspect_ratio))
            h = int(round(math.sqrt(target_area * aspect_ratio)))
            w = int(round(math.sqrt(target_area / aspect_ratio)))
            if w < self.width and h < self.height:
                top = random.randint(0, self.height - h)
                left = random.randint(0, self.width - w)

                num_masked = mask[top : top + h, left : left + w].sum()
                # Overlap check
                if 0 < h * w - num_masked <= max_mask_patches:
                    for i in range(top, top + h):
                        for j in range(left, left + w):
                            if mask[i, j] == 0:
                                mask[i, j] = 1
                                delta += 1

                if delta > 0:
                    break
        return delta

    def __call__(self, num_masking_patches=0):
        mask = np.zeros(shape=self.get_shape(), dtype=bool)
        mask_count = 0
        while mask_count < num_masking_patches:
            max_mask_patches = num_masking_patches - mask_count
            max_mask_patches = min(max_mask_patches, self.max_num_patches)

            delta = self._mask(mask, max_mask_patches)
            if delta == 0:
                break
            else:
                mask_count += delta

        return self.complete_mask_randomly(mask, num_masking_patches)

    def complete_mask_randomly(self, mask, num_masking_patches):
        shape = mask.shape
        m2 = mask.flatten()

        current_masked = m2.sum()
        if current_masked < num_masking_patches:
            to_add = np.random.choice(np.where(~m2)[0], size=num_masking_patches - current_masked, replace=False)
            m2[to_add] = True

        return m2.reshape(shape)
