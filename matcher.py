"""Nearest-reference matching. No model, no fitting, no weights: just distances."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

import config as cfg
from dataset import ReferenceSet
from features import Descriptor

_GEOM_W = np.array([1.0, 1.0, 1.0, 1.5, 2.5, 1.0, 1.0], np.float32)


@dataclass
class Prediction:
    label: str
    confidence: float
    ranking: list[tuple[str, float]] = field(default_factory=list)  # (label, confidence-like score)
    source: str = "dataset"


class ReferenceMatcher:
    def __init__(self, refs: ReferenceSet) -> None:
        self.refs = refs
        self.classes = sorted(set(refs.labels.tolist()))
        self._idx = {c: np.nonzero(refs.labels == c)[0] for c in self.classes}

    # ----------------------------------------------------------------------
    def distances(self, d: Descriptor) -> np.ndarray:
        """Combined 0..1 distance from the query to every reference."""
        r = self.refs
        dg = np.clip((1.0 - r.hog_gray @ d.hog_gray) / cfg.HOG_COS_SCALE, 0, 1)
        dm = np.clip((1.0 - r.hog_mask @ d.hog_mask) / cfg.HOG_COS_SCALE, 0, 1)
        dr = np.clip(np.abs(r.radial - d.radial).mean(1) / cfg.RADIAL_SCALE, 0, 1)
        dh = np.clip(np.abs(r.hu - d.hu)[:, :6].mean(1) / cfg.HU_SCALE, 0, 1)
        dq = np.clip((np.abs(r.geom - d.geom) * _GEOM_W).sum(1) / _GEOM_W.sum() * 2.5, 0, 1)
        return (cfg.W_HOG_GRAY * dg + cfg.W_HOG_MASK * dm + cfg.W_RADIAL * dr
                + cfg.W_HU * dh + cfg.W_GEOM * dq)

    def predict(self, d: Descriptor, allowed: set[str] | None = None,
                exclude_src: int | None = None) -> Prediction:
        if not len(self.refs):
            return Prediction("?", 0.0, source="dataset")
        dist = self.distances(d)
        if exclude_src is not None:
            dist = np.where(self.refs.src == exclude_src, np.inf, dist)

        scores: list[tuple[str, float]] = []
        for c in self.classes:
            if allowed is not None and c not in allowed:
                continue
            v = dist[self._idx[c]]
            v = v[np.isfinite(v)]
            if v.size == 0:
                continue
            k = min(cfg.K_NEAREST, v.size)
            scores.append((c, float(np.sort(v)[:k].mean())))
        if not scores:
            return Prediction("?", 0.0, source="dataset")
        scores.sort(key=lambda t: t[1])

        d1 = scores[0][1]
        d2 = scores[1][1] if len(scores) > 1 else 1.0
        absolute = float(np.clip(1.0 - d1 / cfg.CONF_ABS_SCALE, 0, 1))
        margin = float(np.clip((d2 - d1) / max(d2, 1e-6) / cfg.CONF_MARGIN_SCALE, 0, 1))
        conf = 0.55 * absolute + 0.45 * margin
        ranking = [(c, float(np.clip(1.0 - dc / cfg.CONF_ABS_SCALE, 0, 1)))
                   for c, dc in scores[:3]]
        if d1 > cfg.REJECT_DISTANCE:
            return Prediction("?", conf * 0.5, ranking, "dataset")
        return Prediction(scores[0][0], conf, ranking, "dataset")
