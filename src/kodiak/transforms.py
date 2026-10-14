# augmentations.py
"""
EXACT DINOv3 Augmentation Pipeline.

This module replicates the exact augmentation pipeline from DINOv3
using torchvision transforms (NOT Kornia) to ensure identical behavior.

Key differences from the previous Kornia implementation:
- Uses torchvision transforms (same library as DINOv3)
- Two DIFFERENT global crop pipelines (transfo1 vs transfo2)
- Correct GaussianBlur with kernel_size=9
- RandomSolarize for global crop 2 only
- Proper PIL-based workflow matching DINOv3

Reference: DinoV3LightningTraining/dinov3/dinov3/data/augmentations.py
"""

import logging
from typing import Any, Dict, Tuple

import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF

logger = logging.getLogger("mpp_dinov3_augmentations")

# ImageNet default normalization
IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)


class GaussianBlur(transforms.RandomApply):
    """
    Apply Gaussian Blur to the PIL image (EXACT DINOv3 implementation).

    DINOv3 uses kernel_size=9 and sigma=(0.1, 2.0).

    IMPORTANT: DINOv3 uses keep_p = 1 - p, meaning p is the probability
    of APPLYING the blur, not skipping it.
    """
    def __init__(self, *, p: float = 0.5, radius_min: float = 0.1, radius_max: float = 2.0):
        # NOTE: torchvision RandomApply applies transform with probability p
        # DINOv3 passes keep_p = 1 - p to get the OPPOSITE behavior
        # So if we want blur with probability p, we pass 1-p to RandomApply
        keep_p = 1 - p
        transform = transforms.GaussianBlur(kernel_size=9, sigma=(radius_min, radius_max))
        super().__init__(transforms=[transform], p=keep_p)


def make_normalize_transform(
    mean: Tuple[float, float, float] = IMAGENET_DEFAULT_MEAN,
    std: Tuple[float, float, float] = IMAGENET_DEFAULT_STD,
) -> transforms.Normalize:
    """Create ImageNet normalization transform."""
    return transforms.Normalize(mean=mean, std=std)


class DataAugmentationDINO:
    """
    EXACT DINOv3 Data Augmentation Pipeline.

    This class replicates the exact augmentation pipeline from DINOv3:
    - 2 Global crops (224x224) with DIFFERENT distortions
    - Optional local crops (96x96) - disabled for KODIAK (global-only)

    Global Crop 1 (transfo1):
        - RandomResizedCrop (scale=[0.32, 1.0])
        - RandomHorizontalFlip (p=0.5)
        - ColorJitter (brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1, p=0.8)
        - RandomGrayscale (p=0.2)
        - GaussianBlur (p=1.0) - ALWAYS applied
        - Normalize

    Global Crop 2 (transfo2):
        - RandomResizedCrop (scale=[0.32, 1.0])
        - RandomHorizontalFlip (p=0.5)
        - ColorJitter (same as above)
        - RandomGrayscale (p=0.2)
        - GaussianBlur (p=0.1) - LOW probability
        - RandomSolarize (threshold=128, p=0.2)
        - Normalize

    Reference: DinoV3LightningTraining/dinov3/dinov3/data/augmentations.py
    """

    def __init__(
        self,
        global_crops_scale: Tuple[float, float] = (0.32, 1.0),
        global_crops_size: int = 256,  # DINOv3 official default
        local_crops_scale: Tuple[float, float] = (0.05, 0.32),
        local_crops_size: int = 112,  # DINOv3 official default
        local_crops_number: int = 0,  # Disabled by default for KODIAK
        share_color_jitter: bool = False,
        horizontal_flips: bool = True,
        mean: Tuple[float, float, float] = IMAGENET_DEFAULT_MEAN,
        std: Tuple[float, float, float] = IMAGENET_DEFAULT_STD,
    ):
        self.global_crops_scale = global_crops_scale
        self.global_crops_size = global_crops_size
        self.local_crops_scale = local_crops_scale
        self.local_crops_size = local_crops_size
        self.local_crops_number = local_crops_number
        self.share_color_jitter = share_color_jitter
        self.horizontal_flips = horizontal_flips
        self.mean = mean
        self.std = std

        flip_p = 0.5 if horizontal_flips else 0.0

        # ============================================================
        # GEOMETRIC AUGMENTATION FOR GLOBAL CROPS
        # ============================================================
        self.geometric_augmentation_global = transforms.Compose([
            transforms.RandomResizedCrop(
                global_crops_size,
                scale=global_crops_scale,
                interpolation=InterpolationMode.BICUBIC,
            ),
            transforms.RandomHorizontalFlip(p=flip_p),
        ])

        # ============================================================
        # GEOMETRIC AUGMENTATION FOR LOCAL CROPS (if enabled)
        # ============================================================
        self.geometric_augmentation_local = transforms.Compose([
            transforms.RandomResizedCrop(
                local_crops_size,
                scale=local_crops_scale,
                interpolation=InterpolationMode.BICUBIC,
            ),
            transforms.RandomHorizontalFlip(p=flip_p),
        ])

        # ============================================================
        # COLOR JITTERING (DINOv3 exact parameters)
        # ============================================================
        color_jittering = transforms.Compose([
            transforms.RandomApply(
                [transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1)],
                p=0.8,
            ),
            transforms.RandomGrayscale(p=0.2),
        ])

        # ============================================================
        # ADDITIONAL DISTORTIONS
        # ============================================================
        # Global Crop 1: Strong blur (p=1.0), no solarize
        global_transfo1_extra = GaussianBlur(p=1.0)

        # Global Crop 2: Weak blur (p=0.1) + solarize (p=0.2)
        global_transfo2_extra = transforms.Compose([
            GaussianBlur(p=0.1),
            transforms.RandomSolarize(threshold=128, p=0.2),
        ])

        # Local Crops: Medium blur (p=0.5)
        local_transfo_extra = GaussianBlur(p=0.5)

        # ============================================================
        # NORMALIZATION
        # ============================================================
        self.normalize = transforms.Compose([
            transforms.ToTensor(),
            make_normalize_transform(mean=mean, std=std),
        ])

        # ============================================================
        # BUILD COMPLETE PIPELINES
        # ============================================================
        if self.share_color_jitter:
            # Color jitter applied once to full image before cropping
            self.color_jittering = color_jittering
            self.global_transfo1 = transforms.Compose([global_transfo1_extra, self.normalize])
            self.global_transfo2 = transforms.Compose([global_transfo2_extra, self.normalize])
            self.local_transfo = transforms.Compose([local_transfo_extra, self.normalize])
        else:
            # Color jitter applied after cropping (default, more diverse)
            self.global_transfo1 = transforms.Compose([
                color_jittering, global_transfo1_extra, self.normalize
            ])
            self.global_transfo2 = transforms.Compose([
                color_jittering, global_transfo2_extra, self.normalize
            ])
            self.local_transfo = transforms.Compose([
                color_jittering, local_transfo_extra, self.normalize
            ])

        logger.info("###################################")
        logger.info("Using EXACT DINOv3 Augmentations:")
        logger.info(f"  global_crops_size: {global_crops_size}")
        logger.info(f"  global_crops_scale: {global_crops_scale}")
        logger.info(f"  local_crops_number: {local_crops_number}")
        logger.info(f"  local_crops_size: {local_crops_size}")
        logger.info(f"  local_crops_scale: {local_crops_scale}")
        logger.info(f"  horizontal_flips: {horizontal_flips}")
        logger.info(f"  share_color_jitter: {share_color_jitter}")
        logger.info("  Global Crop 1: ColorJitter + Grayscale + GaussianBlur(p=1.0)")
        logger.info("  Global Crop 2: ColorJitter + Grayscale + GaussianBlur(p=0.1) + Solarize(p=0.2)")
        logger.info(f"  normalize_mean: {mean}")
        logger.info(f"  normalize_std: {std}")
        logger.info("###################################")

    def __call__(self, image) -> Dict[str, Any]:
        """
        Apply DINOv3 augmentations to an image.

        Args:
            image: PIL Image or tensor

        Returns:
            dict with:
                - 'global_crops': List of 2 augmented global crop tensors
                - 'local_crops': List of N augmented local crop tensors (if enabled)
        """
        # Convert tensor to PIL if needed
        if isinstance(image, torch.Tensor):
            # Handle grayscale
            if image.dim() == 2:
                image = image.unsqueeze(0)
            if image.size(0) == 1:
                image = image.repeat(3, 1, 1)
            elif image.size(0) > 3:
                image = image[:3]
            # Convert to PIL for torchvision transforms
            image = TF.to_pil_image(image)

        output = {}

        # Apply shared color jitter if enabled
        if self.share_color_jitter:
            image = self.color_jittering(image)

        # ============================================================
        # GLOBAL CROPS (2 crops with DIFFERENT distortions)
        # ============================================================
        # Global Crop 1: geometric + transfo1 (strong blur)
        im1_base = self.geometric_augmentation_global(image)
        global_crop_1 = self.global_transfo1(im1_base)

        # Global Crop 2: geometric + transfo2 (weak blur + solarize)
        im2_base = self.geometric_augmentation_global(image)
        global_crop_2 = self.global_transfo2(im2_base)

        output["global_crops"] = [global_crop_1, global_crop_2]

        # ============================================================
        # LOCAL CROPS (if enabled)
        # ============================================================
        if self.local_crops_number > 0:
            local_crops = [
                self.local_transfo(self.geometric_augmentation_local(image))
                for _ in range(self.local_crops_number)
            ]
            output["local_crops"] = local_crops
        else:
            output["local_crops"] = []

        return output


class DINOv3AugmentationModule(nn.Module):
    """
    PyTorch Module wrapper for DINOv3 augmentations.

    This provides a simple interface that returns a single augmented image
    for compatibility with the existing TransformDataset.

    For pretraining with two views, use DataAugmentationDINO directly.
    """

    def __init__(self, aug_cfg: dict, view_index: int = 0):
        """
        Args:
            aug_cfg: Augmentation config dict
            view_index: Which global crop to return (0 or 1)
                        0 = transfo1 (strong blur)
                        1 = transfo2 (weak blur + solarize)
        """
        super().__init__()

        self.dino_aug = DataAugmentationDINO(
            global_crops_scale=tuple(aug_cfg.get("global_crops_scale", [0.32, 1.0])),
            global_crops_size=int(aug_cfg.get("global_crops_size", 256)),
            local_crops_scale=tuple(aug_cfg.get("local_crops_scale", [0.05, 0.32])),
            local_crops_size=int(aug_cfg.get("local_crops_size", 112)),
            local_crops_number=int(aug_cfg.get("local_crops_number", 0)),
            share_color_jitter=bool(aug_cfg.get("share_color_jitter", False)),
            horizontal_flips=bool(aug_cfg.get("horizontal_flips", True)),
            mean=tuple(aug_cfg.get("normalize_mean", IMAGENET_DEFAULT_MEAN)),
            std=tuple(aug_cfg.get("normalize_std", IMAGENET_DEFAULT_STD)),
        )
        self.view_index = view_index
        self.image_size = int(aug_cfg.get("global_crops_size", 256))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply DINOv3 augmentation and return a single view.

        Args:
            x: (C, H, W) tensor in [0, 1]

        Returns:
            (3, H', W') normalized tensor
        """
        output = self.dino_aug(x)
        return output["global_crops"][self.view_index]


class DINOv3PretrainingTransform:
    """
    DINOv3 transform for pretraining that returns both global crops.

    Returns a dict with both global crops for cross-view masked prediction:
    - crop1 (transfo1): Strong blur, used as TEACHER view
    - crop2 (transfo2): Weak blur + solarize, used as STUDENT view (masked)

    This enables cross-view learning where student predicts teacher's
    prototype assignments from a differently augmented (and masked) view.
    """

    def __init__(self, aug_cfg: dict):
        self.dino_aug = DataAugmentationDINO(
            global_crops_scale=tuple(aug_cfg.get("global_crops_scale", [0.32, 1.0])),
            global_crops_size=int(aug_cfg.get("global_crops_size", 256)),
            local_crops_number=0,  # KODIAK uses global crops only
            share_color_jitter=bool(aug_cfg.get("share_color_jitter", False)),
            horizontal_flips=bool(aug_cfg.get("horizontal_flips", True)),
            mean=tuple(aug_cfg.get("normalize_mean", IMAGENET_DEFAULT_MEAN)),
            std=tuple(aug_cfg.get("normalize_std", IMAGENET_DEFAULT_STD)),
        )
        self.image_size = int(aug_cfg.get("global_crops_size", 256))

    def __call__(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Apply DINOv3 augmentation and return both global crops.

        Args:
            x: (C, H, W) tensor in [0, 1]

        Returns:
            dict with:
                - 'teacher_crop': (3, 224, 224) crop1 (strong blur) for teacher
                - 'student_crop': (3, 224, 224) crop2 (weak blur + solarize) for student
        """
        output = self.dino_aug(x)
        return {
            "teacher_crop": output["global_crops"][0],  # transfo1: strong blur
            "student_crop": output["global_crops"][1],  # transfo2: weak blur + solarize
        }


class MultiCropPretrainingTransform:
    """
    Multi-Crop Pretraining Transform for Multi-Crop Prototype CLS Loss.

    Uses EXACT DINOv3 augmentation pipeline:
    - 2 Global crops (224x224) with different distortions
    - N Local crops (96x96) for CLS-level learning

    Returns a dict with:
    - global_crops: List of 2 tensors (for teacher + student encoding)
    - local_crops: List of N tensors (for student encoding only)
    """

    def __init__(self, aug_cfg: dict):
        self.dino_aug = DataAugmentationDINO(
            global_crops_scale=tuple(aug_cfg.get("global_crops_scale", [0.32, 1.0])),
            global_crops_size=int(aug_cfg.get("global_crops_size", 256)),
            local_crops_scale=tuple(aug_cfg.get("local_crops_scale", [0.05, 0.32])),
            local_crops_size=int(aug_cfg.get("local_crops_size", 112)),
            local_crops_number=int(aug_cfg.get("local_crops_number", 8)),  # Enable local crops!
            share_color_jitter=bool(aug_cfg.get("share_color_jitter", False)),
            horizontal_flips=bool(aug_cfg.get("horizontal_flips", True)),
            mean=tuple(aug_cfg.get("normalize_mean", IMAGENET_DEFAULT_MEAN)),
            std=tuple(aug_cfg.get("normalize_std", IMAGENET_DEFAULT_STD)),
        )
        self.global_crops_size = int(aug_cfg.get("global_crops_size", 256))
        self.local_crops_size = int(aug_cfg.get("local_crops_size", 112))
        self.local_crops_number = int(aug_cfg.get("local_crops_number", 8))

        logger.info("###################################")
        logger.info("Using Multi-Crop Pretraining Transform (for Prototype CLS Loss):")
        logger.info(f"  global_crops_size: {self.global_crops_size}")
        logger.info(f"  local_crops_number: {self.local_crops_number}")
        logger.info(f"  local_crops_size: {self.local_crops_size}")
        logger.info("###################################")

    def __call__(self, x: torch.Tensor) -> Dict[str, Any]:
        """
        Apply multi-crop augmentation.

        Args:
            x: (C, H, W) tensor in [0, 1]

        Returns:
            dict with:
                - 'global_crops': List of 2 tensors (3, 224, 224)
                - 'local_crops': List of N tensors (3, 96, 96)
        """
        output = self.dino_aug(x)

        return {
            "global_crops": output["global_crops"],  # List of 2 tensors
            "local_crops": output["local_crops"],    # List of N tensors
        }




class MxIFMultiCropTransform:
    """Multi-crop views of a marker image — ``(3, H, W)`` float in [0, 1] — for KODIAK's objectives. The geometry is DINO's
    (two global crops, N local crops, flips) plus 90° rotations (tissue has no orientation); the photometric part is what
    varies in MxIF instead of photographs: per-channel gain and gamma (staining and exposure) and mild blur. No hue rotation
    (it would swap marker roles), no grayscale (it erases the DAPI anchor), no solarization (meaningless for fluorescence).
    Which marker sits in green is NOT decided here: the dataset draws it once per sample, so every crop of a sample agrees."""

    def __init__(self, aug_cfg: dict):
        import math
        self.global_size = int(aug_cfg.get("global_crops_size", 256))
        self.local_size = int(aug_cfg.get("local_crops_size", 112))
        self.n_local = int(aug_cfg.get("local_crops_number", 8))
        self.gain = float(aug_cfg.get("gain", 0.3))
        lo, hi = aug_cfg.get("gamma", (0.7, 1.4))
        self.log_gamma = (math.log(float(lo)), math.log(float(hi)))
        self.blur_p = float(aug_cfg.get("blur_p", 0.3))
        self.flips = bool(aug_cfg.get("flips", aug_cfg.get("horizontal_flips", True)))
        self.rot90 = bool(aug_cfg.get("rot90", True))
        self.mean = torch.tensor(aug_cfg.get("normalize_mean", IMAGENET_DEFAULT_MEAN)).view(3, 1, 1)
        self.std = torch.tensor(aug_cfg.get("normalize_std", IMAGENET_DEFAULT_STD)).view(3, 1, 1)
        self.crop_global = transforms.RandomResizedCrop(self.global_size, scale=tuple(aug_cfg.get("global_crops_scale", (0.32, 1.0))),
                                                        interpolation=InterpolationMode.BICUBIC, antialias=True)
        self.crop_local = transforms.RandomResizedCrop(self.local_size, scale=tuple(aug_cfg.get("local_crops_scale", (0.05, 0.32))),
                                                       interpolation=InterpolationMode.BICUBIC, antialias=True)
        self.blur = transforms.GaussianBlur(kernel_size=9, sigma=(0.1, 2.0))
        logger.info("MxIF multi-crop transform: global %d x2, local %d x%d, gain ±%.2f, gamma %s, blur p=%.2f, flips=%s, rot90=%s",
                    self.global_size, self.local_size, self.n_local, self.gain, aug_cfg.get("gamma", (0.7, 1.4)), self.blur_p, self.flips, self.rot90)

    def _view(self, x: torch.Tensor, crop) -> torch.Tensor:
        v = crop(x)
        if self.flips:
            if torch.rand(1).item() < 0.5:
                v = TF.hflip(v)
            if torch.rand(1).item() < 0.5:
                v = TF.vflip(v)
        if self.rot90:
            v = torch.rot90(v, int(torch.randint(0, 4, (1,)).item()), dims=(1, 2))
        if self.gain > 0:
            v = v * (1.0 + (torch.rand(3, 1, 1) * 2 - 1) * self.gain)
        gamma = torch.exp(torch.empty(3, 1, 1).uniform_(*self.log_gamma))
        v = v.clamp(0.0, 1.0) ** gamma
        if self.blur_p > 0 and torch.rand(1).item() < self.blur_p:
            v = self.blur(v)
        return (v.clamp(0.0, 1.0) - self.mean) / self.std

    def __call__(self, x) -> Dict[str, Any]:
        if not isinstance(x, torch.Tensor):
            x = TF.to_tensor(x)
        if x.dim() == 2:
            x = x.unsqueeze(0)
        if x.size(0) == 1:
            x = x.repeat(3, 1, 1)
        x = x.float()
        return {"global_crops": [self._view(x, self.crop_global) for _ in range(2)],
                "local_crops": [self._view(x, self.crop_local) for _ in range(self.n_local)]}


def make_transform(aug_cfg: dict):
    """The multi-crop transform the config asks for: ``photometric: mxif`` (default, marker images) or ``dino`` (the
    natural-image recipe of the reference implementation, kept for ablation)."""
    kind = str(aug_cfg.get("photometric", "mxif")).lower()
    if kind == "dino":
        return MultiCropPretrainingTransform(aug_cfg)
    if kind != "mxif":
        raise ValueError(f"augmentation.photometric must be 'mxif' or 'dino', got {kind!r}")
    return MxIFMultiCropTransform(aug_cfg)
