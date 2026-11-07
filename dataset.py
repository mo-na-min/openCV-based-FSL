"""Load DATASETS/<label>/*.jpg into reference descriptors (no training involved).

Each image is segmented and described with exactly the same code path the live
camera uses. The results are cached so start-up is instant after the first run.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import cv2
import numpy as np

import config as cfg
from features import Descriptor, analyze_shape, describe
from segmentation import SkinSegmenter, clean_mask, trim_wrist

USE_GRABCUT = True   # OpenCV's classic GrabCut, only as a fallback for odd images


@dataclass
class ReferenceSet:
    labels: np.ndarray            # (N,) '<U1'
    src: np.ndarray               # (N,) source image id (mirror copy shares the id)
    hog_gray: np.ndarray
    hog_mask: np.ndarray
    radial: np.ndarray
    hu: np.ndarray
    geom: np.ndarray
    skipped: list[str] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.labels)

    def class_counts(self) -> dict[str, int]:
        vals, counts = np.unique(self.labels, return_counts=True)
        return {str(v): int(c) for v, c in zip(vals, counts)}


# --------------------------------------------------------------------------
# Segmenting a single reference photo
# --------------------------------------------------------------------------
def _usable(mask: np.ndarray | None) -> bool:
    if mask is None:
        return False
    frac = float((mask > 0).mean())
    return 0.05 <= frac <= 0.80 and analyze_shape(mask) is not None


def _grabcut(img: np.ndarray) -> np.ndarray | None:
    h, w = img.shape[:2]
    m = np.zeros((h, w), np.uint8)
    rect = (int(w * 0.08), int(h * 0.08), int(w * 0.84), int(h * 0.84))
    try:
        cv2.grabCut(img, m, rect, np.zeros((1, 65)), np.zeros((1, 65)), 4,
                    cv2.GC_INIT_WITH_RECT)
    except cv2.error:
        return None
    return trim_wrist(clean_mask(np.where((m == 1) | (m == 3), 255, 0).astype(np.uint8)))


def _otsu(img: np.ndarray) -> np.ndarray | None:
    gray = cv2.GaussianBlur(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), (5, 5), 0)
    _, b = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if (b > 0).mean() > 0.5:            # the hand is the minority region
        b = cv2.bitwise_not(b)
    return trim_wrist(clean_mask(b))


def segment_reference(img: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
    """Return (resized image, hand mask) or None if no hand could be found."""
    h, w = img.shape[:2]
    scale = cfg.DATASET_MAX_SIDE / max(h, w)
    if scale < 1.0:
        img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    h, w = img.shape[:2]

    # 1) generic skin colour
    mask = SkinSegmenter().segment(img)
    if _usable(mask):
        return img, mask

    # 2) skin colour calibrated on the centre of this very image
    seg = SkinSegmenter()
    seg.calibrate(img[h * 2 // 5: h * 3 // 5, w * 2 // 5: w * 3 // 5])
    mask = seg.segment(img)
    if _usable(mask):
        return img, mask

    # 3) GrabCut / 4) Otsu threshold (already-segmented or plain-background photos)
    if USE_GRABCUT:
        mask = _grabcut(img)
        if _usable(mask):
            return img, mask
    mask = _otsu(img)
    if _usable(mask):
        return img, mask
    return None


def _read_image(path: Path) -> np.ndarray | None:
    try:                                  # np.fromfile keeps non-ASCII Windows paths working
        data = np.fromfile(str(path), np.uint8)
        return cv2.imdecode(data, cv2.IMREAD_COLOR)
    except Exception:
        return None


# --------------------------------------------------------------------------
# Scanning + caching
# --------------------------------------------------------------------------
def scan_dataset(dataset_dir: Path) -> list[tuple[str, Path]]:
    items: list[tuple[str, Path]] = []
    if not dataset_dir.is_dir():
        return items
    for sub in sorted(dataset_dir.iterdir()):
        label = sub.name.upper()
        if not sub.is_dir() or label not in cfg.VALID_LABELS:
            continue
        for f in sorted(sub.iterdir()):
            if f.suffix.lower() in cfg.IMAGE_EXTS:
                items.append((label, f))
    return items


def _signature(items: list[tuple[str, Path]]) -> str:
    h = hashlib.md5(f"v{cfg.CACHE_VERSION}|{cfg.MIRROR_AUGMENT}|{cfg.NORM_SIZE}".encode())
    for label, p in items:
        st = p.stat()
        h.update(f"{label}|{p.name}|{st.st_size}|{st.st_mtime_ns}".encode())
    return h.hexdigest()


def _stack(rows: list[Descriptor], dim_src: dict[str, int]) -> dict[str, np.ndarray]:
    if not rows:
        return {k: np.zeros((0, d), np.float32) for k, d in dim_src.items()}
    return {
        "hog_gray": np.stack([r.hog_gray for r in rows]),
        "hog_mask": np.stack([r.hog_mask for r in rows]),
        "radial": np.stack([r.radial for r in rows]),
        "hu": np.stack([r.hu for r in rows]),
        "geom": np.stack([r.geom for r in rows]),
    }


def load_references(dataset_dir: Path = cfg.DATASET_DIR,
                    progress: Callable[[int, int], None] | None = None,
                    use_cache: bool = True) -> ReferenceSet:
    items = scan_dataset(dataset_dir)
    sig = _signature(items) if items else ""

    if use_cache and items and cfg.CACHE_PATH.exists():
        try:
            z = np.load(cfg.CACHE_PATH, allow_pickle=False)
            if str(z["signature"]) == sig:
                return ReferenceSet(z["labels"], z["src"], z["hog_gray"], z["hog_mask"],
                                    z["radial"], z["hu"], z["geom"],
                                    list(z["skipped"]))
        except Exception:
            pass   # stale or corrupt cache -> rebuild

    labels: list[str] = []
    src: list[int] = []
    rows: list[Descriptor] = []
    skipped: list[str] = []

    for idx, (label, path) in enumerate(items):
        if progress:
            progress(idx, len(items))
        img = _read_image(path)
        if img is None:
            skipped.append(f"{label}/{path.name}: unreadable")
            continue
        seg = segment_reference(img)
        if seg is None:
            skipped.append(f"{label}/{path.name}: no hand found")
            continue
        img, mask = seg
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        variants = [(mask, gray)]
        if cfg.MIRROR_AUGMENT:
            variants.append((cv2.flip(mask, 1), cv2.flip(gray, 1)))
        for m, g in variants:
            d = describe(m, g)
            if d is None:
                continue
            rows.append(d)
            labels.append(label)
            src.append(idx)
    if progress:
        progress(len(items), len(items))

    dims = {"hog_gray": 1764, "hog_mask": 1764, "radial": cfg.RADIAL_BINS, "hu": 7, "geom": 7}
    arrays = _stack(rows, dims)
    ref = ReferenceSet(np.array(labels, dtype="<U1"), np.array(src, dtype=np.int32),
                       skipped=skipped, **arrays)

    if items and len(ref):
        cfg.CACHE_DIR.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cfg.CACHE_PATH, signature=np.array(sig), labels=ref.labels,
                            src=ref.src, hog_gray=ref.hog_gray, hog_mask=ref.hog_mask,
                            radial=ref.radial, hu=ref.hu, geom=ref.geom,
                            skipped=np.array(skipped, dtype=str))
    return ref
