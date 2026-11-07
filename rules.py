"""Geometry-only fallback used when DATASETS is missing.

Honest limitation: a silhouette cannot separate look-alike fist signs
(A / E / S / M / N / T) or pairs such as V / 2, W / 3, B / 4. This module only
gives a coarse guess from finger count, finger directions and holes; the dataset
matcher is the accurate path.
"""
from __future__ import annotations

from features import HandShape
from matcher import Prediction


def classify_rules(shape: HandShape | None, allowed: set[str] | None = None) -> Prediction:
    if shape is None:
        return Prediction("?", 0.0, source="rules")

    n = shape.finger_units
    f = shape.fingers
    angles = [x.angle for x in f]
    spread = (max(angles) - min(angles)) if len(angles) >= 2 else 0.0
    up = [a for a in angles if 55 <= a <= 125]
    side = [a for a in angles if a < 35 or a > 145]

    guesses: list[tuple[str, float]]
    if n == 0:
        if shape.holes >= 1:
            guesses = [("O", 0.55), ("0", 0.50)]
        elif shape.solidity >= 0.88:
            guesses = [("A", 0.50), ("S", 0.45), ("E", 0.40)]
        else:
            guesses = [("C", 0.45), ("S", 0.40)]
    elif n == 1:
        if up:
            guesses = [("1", 0.60), ("D", 0.50), ("I", 0.45)]
        else:
            guesses = [("G", 0.50), ("Y", 0.40)]
    elif n == 2:
        if up and side:
            guesses = [("L", 0.60)]
        elif spread > 25:
            guesses = [("V", 0.55), ("2", 0.55)]
        else:
            guesses = [("U", 0.50), ("H", 0.40)]
    elif n == 3:
        guesses = [("W", 0.50), ("3", 0.50)]
    elif n == 4:
        guesses = [("B", 0.50), ("4", 0.50)]
    else:
        guesses = [("5", 0.60)]

    if allowed is not None:
        filtered = [g for g in guesses if g[0] in allowed]
        guesses = filtered or guesses
    label, conf = guesses[0]
    return Prediction(label, conf, guesses[:3], source="rules")
