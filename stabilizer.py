"""Temporal smoothing and hold-to-commit logic for static signs."""
from __future__ import annotations

import time
from collections import deque

import config as cfg


class Stabilizer:
    """Votes over the last frames and commits a sign after it is held steady."""

    def __init__(self) -> None:
        self.history: deque[tuple[str | None, float]] = deque(maxlen=cfg.SMOOTH_WINDOW)
        self.text: list[str] = []
        self._held: str | None = None
        self._held_since = 0.0
        self._last_commit = 0.0
        self._released = True

    # ----------------------------------------------------------------------
    def update(self, label: str | None, conf: float) -> tuple[str | None, float, float, bool]:
        """Feed one frame. Returns (stable_label, stable_conf, progress 0..1, committed)."""
        if label is None or label == "?" or conf < cfg.MIN_CONFIDENCE:
            self.history.append((None, 0.0))
        else:
            self.history.append((label, conf))

        weights: dict[str, float] = {}
        for lab, c in self.history:
            if lab is not None:
                weights[lab] = weights.get(lab, 0.0) + c
        total = sum(c for _, c in self.history) or 1.0

        stable, stable_conf = None, 0.0
        if weights:
            best = max(weights, key=weights.get)
            share = weights[best] / max(total, 1e-6)
            n = sum(1 for lab, _ in self.history if lab == best)
            if share >= cfg.MIN_VOTE_SHARE and n >= 3:
                stable = best
                stable_conf = weights[best] / n * share

        now = time.time()
        if stable is None:
            self._held = None
            self._released = True
            return None, 0.0, 0.0, False

        if stable != self._held:
            self._held, self._held_since = stable, now
        held = now - self._held_since
        progress = min(held / cfg.DWELL_SECONDS, 1.0)

        committed = False
        if held >= cfg.DWELL_SECONDS:
            repeat_ok = now - self._last_commit >= cfg.REPEAT_SECONDS
            if self._released or repeat_ok:
                self.text.append(stable)
                self._last_commit = now
                self._released = False
                self._held_since = now          # restart the bar for a possible repeat
                committed = True
        return stable, stable_conf, progress, committed

    # ----------------------------------------------------------------------
    def reset_votes(self) -> None:
        self.history.clear()
        self._held = None
        self._released = True

    def add_space(self) -> None:
        self.text.append(" ")

    def backspace(self) -> None:
        if self.text:
            self.text.pop()

    def clear(self) -> None:
        self.text.clear()

    @property
    def string(self) -> str:
        return "".join(self.text)
