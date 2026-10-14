import numpy as np

from mxdata.viz import colorize_labels, gray_to_rgb, label_contours, overlay_labels, overlay_masks, to_uint8


def test_label_contours_are_complete_on_every_side():
    lab = np.zeros((20, 20), np.int32)
    lab[5:15, 5:15] = 1                                   # a square nucleus on background
    b = label_contours(lab)
    assert b[5, 5:15].all() and b[14, 5:15].all() and b[5:15, 5].all() and b[5:15, 14].all()   # top, bottom, left, right
    assert not b[6:14, 6:14].any() and not b[lab == 0].any()
    lab[5:15, 15:18] = 2                                  # a touching neighbour: both sides of the shared edge
    b = label_contours(lab)
    assert b[5:15, 14].all() and b[5:15, 15].all()
    assert label_contours(lab, thickness=2).sum() > b.sum()


def test_overlay_and_helpers():
    img = np.random.default_rng(0).integers(0, 4000, (20, 20)).astype(np.uint16)
    u8 = to_uint8(img)
    assert u8.dtype == np.uint8 and gray_to_rgb(u8).shape == (20, 20, 3)
    lab = np.zeros((20, 20), np.int32)
    lab[2:8, 2:8] = 1
    out = overlay_labels(gray_to_rgb(u8), lab, color=(0, 255, 0), ignore=lab == 1)
    assert out.shape == (20, 20, 3) and (out[2, 2] == (255, 60, 60)).all()      # ignored contour takes the ignore colour
    assert colorize_labels(lab)[0, 0].sum() == 0 and colorize_labels(lab)[4, 4].sum() > 0


def test_overlay_masks_tints_exactly_the_label_pixels():
    rgb = np.zeros((20, 20, 3), np.uint8)
    lab = np.zeros((20, 20), np.int32)
    lab[2:8, 2:8] = 1
    lab[10:15, 10:15] = 2
    out = overlay_masks(rgb, lab, ignore=lab == 2, alpha=0.5)
    assert out[lab == 0].sum() == 0                      # background untouched
    assert out[4, 4].sum() > 0 and tuple(out[12, 12]) == (127, 30, 30)   # nucleus 1 coloured; nucleus 2 = ignore tint
    lut = np.zeros((3, 3), np.uint8)
    lut[1] = (0, 200, 0)
    assert tuple(overlay_masks(rgb, lab, alpha=1.0, colors=lut)[4, 4]) == (0, 200, 0)
