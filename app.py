#!/usr/bin/env python3
"""Static FSL translator (letters A-Z, numbers 0-9): pure OpenCV + NumPy.

Run:  python app.py            (use --help for options)
"""
from __future__ import annotations

import argparse
import math
import sys
import time

import cv2
import numpy as np

import config as cfg
import gui
from dataset import load_references
from features import analyze_shape, describe
from matcher import Prediction, ReferenceMatcher
from rules import classify_rules
from segmentation import SkinSegmenter
from stabilizer import Stabilizer

WINDOW = "FSL Translator"
SCOPES = {"ALL": None, "LETTERS": cfg.LETTERS, "NUMBERS": cfg.NUMBERS}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Static FSL translator (OpenCV only)")
    p.add_argument("--camera", type=int, default=cfg.CAMERA_INDEX, help="camera index")
    p.add_argument("--no-dataset", action="store_true", help="geometry rules only")
    p.add_argument("--rebuild", action="store_true", help="ignore the feature cache")
    return p.parse_args()


def load_matcher(args: argparse.Namespace) -> ReferenceMatcher | None:
    if args.no_dataset:
        print("[INFO] --no-dataset: using geometry rules only")
        return None
    if not cfg.DATASET_DIR.is_dir():
        print(f"[WARN] {cfg.DATASET_DIR} not found: falling back to geometry rules only")
        return None

    def progress(i: int, n: int) -> None:
        print(f"\r[INFO] Building reference features {i}/{n}", end="", flush=True)

    t0 = time.time()
    refs = load_references(progress=progress, use_cache=not args.rebuild)
    print(f"\r[INFO] {len(refs)} reference samples ready in {time.time() - t0:.1f}s" + " " * 20)
    if refs.skipped:
        print(f"[WARN] {len(refs.skipped)} images skipped (e.g. {refs.skipped[0]})")
    if not len(refs):
        print("[WARN] DATASETS has no usable images: falling back to geometry rules only")
        return None
    return ReferenceMatcher(refs)


class App:
    def __init__(self, args: argparse.Namespace) -> None:
        self.matcher = load_matcher(args)
        self.seg = SkinSegmenter()
        self.stab = Stabilizer()
        self.ui = gui.Gui()
        self.scope = "ALL"
        self.task: tuple[str, float] | None = None     # ("calib" | "bg", fire time)
        self.running = True
        self.fps = 0.0
        self.last_t = time.time()
        self.snapshot: np.ndarray | None = None

        self.cap = cv2.VideoCapture(args.camera)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, cfg.FRAME_W)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cfg.FRAME_H)

    # ----------------------------------------------------------------------
    def roi_rect(self, fw: int, fh: int) -> tuple[int, int, int, int]:
        side = int(fh * cfg.ROI_SIZE_FRAC)
        cx, cy = int(fw * cfg.ROI_CENTER_X_FRAC), int(fh * cfg.ROI_CENTER_Y_FRAC)
        x1 = max(0, min(fw - side, cx - side // 2))
        y1 = max(0, min(fh - side, cy - side // 2))
        return x1, y1, x1 + side, y1 + side

    def on_mouse(self, event: int, x: int, y: int, *_: object) -> None:
        self.ui.mouse = (x, y)
        if event == cv2.EVENT_LBUTTONUP:
            bid = self.ui.hit(x, y)
            if bid:
                self.action(bid)

    def action(self, bid: str) -> None:
        if bid.startswith("scope_"):
            self.scope = bid.split("_", 1)[1]
            self.stab.reset_votes()
        elif bid == "calib":
            self.task = ("calib", time.time() + 3.0)
        elif bid == "bg":
            self.task = ("bg", time.time() + 3.0)
        elif bid == "reset":
            self.seg.reset()
            self.stab.reset_votes()
            self.task = None
        elif bid == "space":
            self.stab.add_space()
        elif bid == "back":
            self.stab.backspace()
        elif bid == "clear":
            self.stab.clear()

    def on_key(self, key: int) -> None:
        if key in (ord("q"), 27):
            self.running = False
        elif key == ord("k"):
            self.action("calib")
        elif key == ord("b"):
            self.action("bg")
        elif key == ord("r"):
            self.action("reset")
        elif key == ord(" "):
            self.action("space")
        elif key in (8, 127):
            self.action("back")
        elif key == ord("x"):
            self.action("clear")
        elif key in (ord("1"), ord("2"), ord("3")):
            self.action("scope_" + ("ALL", "LETTERS", "NUMBERS")[key - ord("1")])

    # ----------------------------------------------------------------------
    def predict(self, desc, shape) -> Prediction:
        allowed = SCOPES[self.scope]
        if self.matcher is not None:
            return self.matcher.predict(desc, allowed)
        return classify_rules(shape, allowed)

    def step(self) -> bool:
        ok, raw = self.cap.read()
        if not ok:
            return False
        frame = cv2.flip(raw, 1)                       # mirror view, like a mirror
        fh, fw = frame.shape[:2]
        x1, y1, x2, y2 = self.roi_rect(fw, fh)
        roi_raw = frame[y1:y2, x1:x2].copy()

        # --- calibration / background tasks (3 second countdown) -----------
        banner = None
        calib_box = None
        if self.task:
            kind, fire = self.task
            remaining = fire - time.time()
            if kind == "calib":
                calib_box = gui.draw_calibration_box(frame, (x1, y1, x2, y2))
                banner = f"Cover the blue box with your palm: {max(1, math.ceil(remaining))}"
            else:
                banner = f"Remove your hand from view: {max(1, math.ceil(remaining))}"
            if remaining <= 0:
                if kind == "calib":
                    bx1, by1, bx2, by2 = calib_box
                    self.seg.calibrate(frame[by1:by2, bx1:bx2])
                else:
                    self.seg.capture_background(roi_raw)
                self.task = None
                self.stab.reset_votes()

        # --- vision pipeline ---------------------------------------------------
        gray = cv2.cvtColor(roi_raw, cv2.COLOR_BGR2GRAY)
        mask = self.seg.segment(roi_raw)
        shape = analyze_shape(mask)
        desc = describe(mask, gray, shape) if shape else None

        pred = self.predict(desc, shape) if desc is not None else Prediction("?", 0.0)
        label, conf, progress, committed = self.stab.update(
            pred.label if desc is not None else None, pred.confidence)
        if committed:
            self.snapshot = frame.copy()
            cfg.HISTORY_DIR.mkdir(exist_ok=True)
            cv2.imwrite(str(cfg.HISTORY_DIR / f"fsl_{time.strftime('%Y%m%d_%H%M%S')}.jpg"), frame)

        edges = cv2.Canny(cv2.GaussianBlur(gray, (5, 5), 0), 40, 120)
        edges = cv2.bitwise_and(edges, cv2.erode(mask, np.ones((5, 5), np.uint8)))

        # --- overlays + render -------------------------------------------------
        roi_view = frame[y1:y2, x1:x2]
        gui.draw_hand(roi_view, shape)
        gui.draw_roi(frame, (x1, y1, x2, y2), active=shape is not None)

        now = time.time()
        self.fps = 0.9 * self.fps + 0.1 / max(now - self.last_t, 1e-3)
        self.last_t = now

        state = gui.UiState(
            frame=frame, hand_found=shape is not None,
            label=label, confidence=conf, ranking=pred.ranking if label else [],
            source=pred.source if desc is not None else ("dataset" if self.matcher else "rules"),
            progress=progress, mask=mask, edges=edges,
            fingers=shape.finger_units if shape else 0, holes=shape.holes if shape else 0,
            solidity=shape.solidity if shape else 0.0,
            text=self.stab.string, fps=self.fps,
            ref_count=len(self.matcher.refs) if self.matcher else 0,
            scope=self.scope, skin_calibrated=self.seg.calibrated,
            bg_captured=self.seg.has_background, banner=banner,
        )
        cv2.imshow(WINDOW, self.ui.render(state))
        return True

    def run(self) -> int:
        if not self.cap.isOpened():
            print("[ERROR] Could not open the camera. Try --camera 1 (or another index).")
            return 1
        cv2.namedWindow(WINDOW, cv2.WINDOW_AUTOSIZE)
        cv2.setMouseCallback(WINDOW, self.on_mouse)
        print("[INFO] Running. Keys: K calibrate skin | B background | R reset | "
              "1/2/3 scope | Space / Backspace / X text | Q quit")
        while self.running:
            if not self.step():
                print("[ERROR] Lost the camera feed.")
                break
            self.on_key(cv2.waitKey(1) & 0xFF)
            if cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
                break
        self.cap.release()
        cv2.destroyAllWindows()
        return 0


def main() -> int:
    return App(parse_args()).run()


if __name__ == "__main__":
    sys.exit(main())
