"""Hand geometry analysis and descriptor extraction (OpenCV + NumPy only)."""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import cv2
import numpy as np

import config as cfg


# --------------------------------------------------------------------------
# Data containers
# --------------------------------------------------------------------------
@dataclass
class Finger:
    tip: tuple[int, int]
    angle: float        # degrees, 0 = right, 90 = up
    length: float       # distance from palm centre in palm radii
    units: int          # 1 for a single finger, more if several are pressed together


@dataclass
class HandShape:
    contour: np.ndarray
    area: float
    bbox: tuple[int, int, int, int]
    palm_center: tuple[int, int]
    palm_radius: float
    hull: np.ndarray
    solidity: float
    extent: float
    aspect: float
    compactness: float
    holes: int
    fingers: list[Finger] = field(default_factory=list)

    @property
    def finger_units(self) -> int:
        return int(min(5, sum(f.units for f in self.fingers)))


@dataclass
class Descriptor:
    hog_gray: np.ndarray
    hog_mask: np.ndarray
    radial: np.ndarray
    hu: np.ndarray
    geom: np.ndarray
    shape: HandShape | None = None


# --------------------------------------------------------------------------
# Shape analysis
# --------------------------------------------------------------------------
def analyze_shape(mask: np.ndarray) -> HandShape | None:
    """Measure palm, fingers and global shape statistics of a binary hand mask."""
    if mask is None or not mask.any():
        return None
    cnts, hier = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    if not cnts or hier is None:
        return None
    hier = hier[0]
    outer = [i for i in range(len(cnts)) if hier[i][3] == -1]
    main = max(outer, key=lambda i: cv2.contourArea(cnts[i]))
    contour = cnts[main]
    area = cv2.contourArea(contour)
    if area < cfg.MIN_HAND_AREA_FRAC * mask.size:
        return None

    holes = sum(1 for i in range(len(cnts))
                if hier[i][3] == main and cv2.contourArea(cnts[i]) > 0.012 * area)

    x, y, w, h = cv2.boundingRect(contour)
    hull = cv2.convexHull(contour)
    hull_area = max(cv2.contourArea(hull), 1.0)
    perimeter = max(cv2.arcLength(contour, True), 1.0)

    filled = np.zeros_like(mask)
    cv2.drawContours(filled, [contour], -1, 255, -1)
    dt = cv2.distanceTransform(filled, cv2.DIST_L2, 5)
    _, radius, _, (px, py) = cv2.minMaxLoc(dt)
    radius = max(float(radius), 1.0)

    shape = HandShape(
        contour=contour, area=area, bbox=(x, y, w, h),
        palm_center=(int(px), int(py)), palm_radius=radius, hull=hull,
        solidity=area / hull_area, extent=area / max(w * h, 1),
        aspect=w / max(h, 1), compactness=4 * math.pi * area / perimeter ** 2,
        holes=holes,
    )
    shape.fingers = _find_fingers(filled, shape)
    return shape


def _circular_runs(fg: np.ndarray) -> list[tuple[int, int]]:
    """(start, length) of every run of True in a circular boolean array."""
    n = len(fg)
    if fg.all() or not fg.any():
        return []
    i0 = int(np.argmin(fg))                      # a background sample
    rolled = np.roll(fg, -i0)
    d = np.diff(np.concatenate([[0], rolled.astype(np.int8), [0]]))
    starts, ends = np.nonzero(d == 1)[0], np.nonzero(d == -1)[0]
    return [((s + i0) % n, int(e - s)) for s, e in zip(starts, ends)]


def _find_fingers(filled: np.ndarray, s: HandShape) -> list[Finger]:
    """Circular scan: walk circles around the palm centre and count the arcs of
    hand they cross. Each arc is one finger (or several pressed together, judged
    by the arc width). The radius that exposes the most fingers is used."""
    px, py = s.palm_center
    r = s.palm_radius
    h, w = filled.shape
    steps = 360
    theta = np.radians(np.arange(steps))
    cos, sin = np.cos(theta), np.sin(theta)

    best: list[tuple[int, int]] = []
    best_rho = 0.0
    for extra in (0.0, 0.2, 0.4, 0.6):
        rho = (cfg.FINGER_MIN_LENGTH + extra) * r
        xs = np.round(px + rho * cos).astype(int)
        ys = np.round(py - rho * sin).astype(int)
        ok = (xs >= 0) & (xs < w) & (ys >= 0) & (ys < h)
        fg = np.zeros(steps, bool)
        fg[ok] = filled[ys[ok], xs[ok]] > 0
        runs = [run for run in _circular_runs(fg) if run[1] >= 3]
        if len(runs) > len(best):
            best, best_rho = runs, rho

    fingers: list[Finger] = []
    for start, length in best:
        mid = (start + length / 2.0) % steps
        width_px = length / steps * 2 * math.pi * best_rho
        units = int(np.clip(round(width_px / (cfg.FINGER_UNIT_WIDTH * r)), 1, 4))
        a = math.radians(mid)
        tip, reach = (int(px + best_rho * math.cos(a)), int(py - best_rho * math.sin(a))), best_rho
        for rr in np.arange(best_rho, best_rho * 2.6, 1.0):          # march outwards to the tip
            x, y = int(round(px + rr * math.cos(a))), int(round(py - rr * math.sin(a)))
            if not (0 <= x < w and 0 <= y < h) or filled[y, x] == 0:
                break
            tip, reach = (x, y), rr
        fingers.append(Finger(tip, mid if mid <= 180 else mid - 360, float(reach / r), units))
    fingers.sort(key=lambda f: f.angle, reverse=True)
    return fingers


def _radial_signature(s: HandShape) -> np.ndarray:
    """Max contour distance from the palm centre per angle bin, scale-normalised."""
    pts = s.contour.reshape(-1, 2).astype(np.float32)
    dx = pts[:, 0] - s.palm_center[0]
    dy = pts[:, 1] - s.palm_center[1]
    ang = (np.degrees(np.arctan2(-dy, dx)) % 360.0)
    dist = np.hypot(dx, dy) / math.sqrt(s.area)
    bins = (ang / 360.0 * cfg.RADIAL_BINS).astype(int) % cfg.RADIAL_BINS
    sig = np.zeros(cfg.RADIAL_BINS, np.float32)
    np.maximum.at(sig, bins, dist)
    padded = np.concatenate([sig[-1:], sig, sig[:1]])
    return np.convolve(padded, [0.25, 0.5, 0.25], mode="valid").astype(np.float32)


# --------------------------------------------------------------------------
# Descriptors
# --------------------------------------------------------------------------
_HOG = cv2.HOGDescriptor((cfg.NORM_SIZE,) * 2, (16, 16), (8, 8), (8, 8), 9)


def _hog(img: np.ndarray) -> np.ndarray:
    v = _HOG.compute(img).ravel().astype(np.float32)
    n = float(np.linalg.norm(v))
    return v / n if n > 0 else v


def _square_crop(img: np.ndarray, bbox, pad: float, border: int = 0) -> np.ndarray:
    x, y, w, h = bbox
    side = max(int(round(max(w, h) * (1 + 2 * pad))), 8)
    x0 = int(round(x + w / 2 - side / 2))
    y0 = int(round(y + h / 2 - side / 2))
    M = np.float32([[1, 0, -x0], [0, 1, -y0]])
    return cv2.warpAffine(img, M, (side, side), flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_CONSTANT, borderValue=border)


def describe(mask: np.ndarray, gray: np.ndarray,
             shape: HandShape | None = None) -> Descriptor | None:
    """Build the full descriptor of the hand in `mask` (gray = same-size grayscale)."""
    shape = shape or analyze_shape(mask)
    if shape is None:
        return None

    bbox = cv2.boundingRect(cv2.findNonZero(mask))
    n = cfg.NORM_SIZE
    mask_sq = _square_crop(mask, bbox, 0.08)
    gray_sq = _square_crop(gray, bbox, 0.08, border=int(gray.mean()))
    nm = cv2.resize(mask_sq, (n, n), interpolation=cv2.INTER_AREA)
    ng = cv2.resize(gray_sq, (n, n), interpolation=cv2.INTER_AREA)

    # Illumination-robust interior texture: normalise grey levels inside the hand only.
    inside = nm > 127
    g = ng.astype(np.float32)
    if inside.sum() > 20:
        mu, sd = g[inside].mean(), g[inside].std() + 1e-3
        g = (g - mu) / sd * 40.0 + 128.0
    g[~inside] = 128.0
    g = cv2.GaussianBlur(np.clip(g, 0, 255).astype(np.uint8), (3, 3), 0)

    hog_gray = _hog(g)
    hog_mask = _hog(cv2.GaussianBlur(nm, (3, 3), 0))

    hu = cv2.HuMoments(cv2.moments(nm, binaryImage=True)).ravel()
    hu = -np.sign(hu) * np.log10(np.abs(hu) + 1e-30)

    geom = np.array([
        shape.solidity,
        shape.extent,
        min(shape.aspect, 2.0) / 2.0,
        min(shape.holes, 2) / 2.0,
        shape.finger_units / 5.0,
        min(len(shape.fingers), 5) / 5.0,
        shape.compactness,
    ], np.float32)

    return Descriptor(hog_gray, hog_mask, _radial_signature(shape),
                      hu.astype(np.float32), geom, shape)
