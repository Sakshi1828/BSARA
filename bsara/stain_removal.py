"""Background-only control images: blood removed, everything else kept.

If a classifier still identifies the pattern type after every stain pixel has
been inpainted away, it is reading the *dataset source* (paper texture, scanner
colour, lighting, image framing) rather than the bloodstain. This is the
shortcut test reported in the README.
"""
import cv2
import numpy as np


def stain_mask(img_bgr, dark_thresh=200, dilate_px=9):
    """Pixels darker than the paper (or strongly red) are treated as blood."""
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    # Adaptive to each image's paper brightness: anything clearly darker than
    # the median background counts as stain.
    bg = np.median(gray)
    dark = gray < min(dark_thresh, bg - 25)
    b, g, r = cv2.split(img_bgr.astype(np.int16))
    red = (r - np.maximum(g, b)) > 40
    mask = (dark | red).astype(np.uint8) * 255
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilate_px, dilate_px))
    return cv2.dilate(mask, k)


def remove_stains(img_bgr):
    mask = stain_mask(img_bgr)
    # Inpaint at reduced resolution for speed, then paste back.
    small = cv2.resize(img_bgr, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
    small_mask = cv2.resize(mask, (small.shape[1], small.shape[0]), interpolation=cv2.INTER_NEAREST)
    filled = cv2.inpaint(small, small_mask, 5, cv2.INPAINT_TELEA)
    filled = cv2.resize(filled, (img_bgr.shape[1], img_bgr.shape[0]))
    out = img_bgr.copy()
    out[mask > 0] = filled[mask > 0]
    return out, mask
