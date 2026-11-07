"""Hand segmentation using only OpenCV colour spaces and morphology."""
from __future__ import annotations

import cv2
import numpy as np

import config as cfg


def _ellipse(k: int) -> np.ndarray:
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))


def clean_mask(mask: np.ndarray) -> np.ndarray:
    """Denoise a binary mask and keep only the significant blobs."""
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, _ellipse(3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, _ellipse(7))
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if n <= 1:
        return mask
    areas = stats[1:, cv2.CC_STAT_AREA]
    biggest = areas.max()
    keep = np.zeros(n, dtype=bool)
    keep[1:] = areas >= max(30, cfg.MIN_COMPONENT_FRAC * biggest)
    return (keep[labels].astype(np.uint8)) * 255


def trim_wrist(mask: np.ndarray) -> np.ndarray:
    """Cut the forearm when the arm enters the image from the bottom edge.

    The palm centre is the maximum of the distance transform; everything more
    than WRIST_TRIM_FACTOR palm radii below it is removed.
    """
    h = mask.shape[0]
    if not mask[h - max(3, int(0.05 * h)):, :].any():   # arm does not reach the bottom edge
        return mask
    dt = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
    _, radius, _, (_, cy) = cv2.minMaxLoc(dt)
    cut = int(cy + cfg.WRIST_TRIM_FACTOR * radius)
    if cut >= h - 1:
        return mask
    out = mask.copy()
    out[cut:, :] = 0
    return clean_mask(out)


class SkinSegmenter:
    """Skin-colour segmenter that can be calibrated and given a background."""

    def __init__(self) -> None:
        self.reset()

    # -- state -------------------------------------------------------------
    def reset(self) -> None:
        self.cr = cfg.CR_RANGE
        self.cb = cfg.CB_RANGE
        self.s_min = cfg.HSV_S_MIN
        self.v_min = cfg.HSV_V_MIN
        self.background: np.ndarray | None = None
        self.calibrated = False

    @property
    def has_background(self) -> bool:
        return self.background is not None

    def calibrate(self, patch_bgr: np.ndarray) -> None:
        """Learn the skin colour range from a patch that contains only skin."""
        blur = cv2.GaussianBlur(patch_bgr, (5, 5), 0)
        ycc = cv2.cvtColor(blur, cv2.COLOR_BGR2YCrCb).reshape(-1, 3)
        hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV).reshape(-1, 3)
        m = cfg.CALIB_MARGIN
        cr_lo, cr_hi = np.percentile(ycc[:, 1], [3, 97])
        cb_lo, cb_hi = np.percentile(ycc[:, 2], [3, 97])
        self.cr = (int(max(110, cr_lo - m)), int(min(190, cr_hi + m)))
        self.cb = (int(max(60, cb_lo - m)), int(min(140, cb_hi + m)))
        self.s_min = int(max(8, np.percentile(hsv[:, 1], 3) - 15))
        self.v_min = int(max(25, np.percentile(hsv[:, 2], 3) - 35))
        self.calibrated = True

    def capture_background(self, roi_bgr: np.ndarray) -> None:
        gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
        self.background = cv2.GaussianBlur(gray, (7, 7), 0)

    def clear_background(self) -> None:
        self.background = None

    # -- segmentation ------------------------------------------------------
    def segment(self, roi_bgr: np.ndarray) -> np.ndarray:
        """Return a cleaned 0/255 hand mask for the ROI."""
        blur = cv2.GaussianBlur(roi_bgr, (5, 5), 0)
        ycc = cv2.cvtColor(blur, cv2.COLOR_BGR2YCrCb)
        hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)

        lo = np.array([0, self.cr[0], self.cb[0]], np.uint8)
        hi = np.array([255, self.cr[1], self.cb[1]], np.uint8)
        mask = cv2.inRange(ycc, lo, hi)
        mask &= cv2.inRange(hsv, np.array([0, self.s_min, self.v_min], np.uint8),
                            np.array([180, 255, 255], np.uint8))

        if self.background is not None and self.background.shape == mask.shape:
            gray = cv2.GaussianBlur(cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY), (7, 7), 0)
            moved = cv2.absdiff(gray, self.background) > cfg.BG_DIFF_THRESHOLD
            moved = cv2.dilate(moved.astype(np.uint8) * 255, _ellipse(9))
            mask &= moved

        return trim_wrist(clean_mask(mask))
