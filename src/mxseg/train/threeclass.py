"""3-class targets for the U-Net: 0 background, 1 nucleus interior, 2 boundary — the ``boundary_px`` outermost pixels of every
nucleus, which includes the contact line between touching nuclei (DSB-2018 / Caicedo et al. 2019 style). At inference the
watershed in ``mxseg.baselines.unet`` turns interior + boundary back into instances."""
from __future__ import annotations

import numpy as np


def labels_to_classes(labels: np.ndarray, boundary_px: int = 2) -> np.ndarray:
    """``(H, W)`` instance labels -> ``(H, W)`` uint8 classes. The image border is not a boundary (labels are edge-replicated)."""
    from scipy import ndimage as ndi

    lab = np.asarray(labels)
    fg = lab > 0
    p = np.pad(lab, 1, mode="edge")
    edge = ((p[:-2, 1:-1] != lab) | (p[2:, 1:-1] != lab) | (p[1:-1, :-2] != lab) | (p[1:-1, 2:] != lab)) & fg
    if boundary_px > 1:
        edge = ndi.binary_dilation(edge, iterations=boundary_px - 1) & fg
    out = fg.astype(np.uint8)
    out[edge] = 2
    return out
