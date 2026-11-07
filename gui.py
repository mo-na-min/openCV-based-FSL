"""Light-mode, card-based GUI drawn entirely with OpenCV primitives."""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import cv2
import numpy as np

from features import HandShape

# --------------------------------------------------------------------------
# Palette (BGR)
# --------------------------------------------------------------------------
BG = (250, 248, 246)
CARD = (255, 255, 255)
BORDER = (232, 226, 221)
TEXT = (46, 38, 33)
MUTED = (125, 112, 104)
ACCENT = (235, 99, 59)          # blue
ACCENT_DARK = (200, 78, 40)
ACCENT_SOFT = (252, 239, 230)
TRACK = (240, 236, 233)
GREEN = (94, 160, 34)
AMBER = (11, 158, 245)
RED = (68, 68, 220)
WHITE = (255, 255, 255)

FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_B = cv2.FONT_HERSHEY_DUPLEX
AA = cv2.LINE_AA

W, H = 1280, 720

# Layout rectangles (x1, y1, x2, y2)
CAM_CARD = (24, 80, 808, 696)
CAM_IMG = (36, 92, 796, 684)
PRED_CARD = (832, 80, 1256, 288)
VIS_CARD = (832, 298, 1256, 444)
TEXT_CARD = (832, 454, 1256, 580)
CTRL_CARD = (832, 590, 1256, 696)


# --------------------------------------------------------------------------
# Drawing helpers
# --------------------------------------------------------------------------
def rrect(img, p1, p2, color, r=14):
    x1, y1 = p1
    x2, y2 = p2
    r = int(min(r, (x2 - x1) // 2, (y2 - y1) // 2))
    cv2.rectangle(img, (x1 + r, y1), (x2 - r, y2), color, -1, AA)
    cv2.rectangle(img, (x1, y1 + r), (x2, y2 - r), color, -1, AA)
    for cx, cy in ((x1 + r, y1 + r), (x2 - r, y1 + r), (x1 + r, y2 - r), (x2 - r, y2 - r)):
        cv2.circle(img, (cx, cy), r, color, -1, AA)


def card(img, rect, fill=CARD, border=BORDER, r=16):
    x1, y1, x2, y2 = rect
    rrect(img, (x1, y1), (x2, y2), border, r)
    rrect(img, (x1 + 1, y1 + 1), (x2 - 1, y2 - 1), fill, r - 1)


def text(img, s, org, scale=0.5, color=TEXT, thick=1, font=FONT, align="left"):
    (w, _), _ = cv2.getTextSize(s, font, scale, thick)
    x, y = org
    if align == "center":
        x -= w // 2
    elif align == "right":
        x -= w
    cv2.putText(img, s, (int(x), int(y)), font, scale, color, thick, AA)
    return w


def text_center(img, s, center, scale=0.5, color=TEXT, thick=1, font=FONT):
    (w, h), _ = cv2.getTextSize(s, font, scale, thick)
    cv2.putText(img, s, (int(center[0] - w / 2), int(center[1] + h / 2)),
                font, scale, color, thick, AA)


def blend_rect(img, p1, p2, color, alpha, r=12):
    layer = img.copy()
    rrect(layer, p1, p2, color, r)
    cv2.addWeighted(layer, alpha, img, 1 - alpha, 0, img)


# --------------------------------------------------------------------------
# Camera overlays (drawn on raw frames before the GUI crops them)
# --------------------------------------------------------------------------
def draw_roi(frame: np.ndarray, rect: tuple[int, int, int, int], active: bool) -> None:
    x1, y1, x2, y2 = rect
    color = (110, 200, 90) if active else (255, 255, 255)
    L, t = 34, 5
    for (cx, cy, dx, dy) in ((x1, y1, 1, 1), (x2, y1, -1, 1), (x1, y2, 1, -1), (x2, y2, -1, -1)):
        cv2.line(frame, (cx, cy), (cx + dx * L, cy), color, t, AA)
        cv2.line(frame, (cx, cy), (cx, cy + dy * L), color, t, AA)
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 1, AA)


def draw_calibration_box(frame: np.ndarray, rect: tuple[int, int, int, int]) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = rect
    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
    half = int((x2 - x1) * 0.09)
    box = (cx - half, cy - half, cx + half, cy + half)
    cv2.rectangle(frame, box[:2], box[2:], (59, 99, 235), 3, AA)
    return box


def draw_hand(roi_view: np.ndarray, shape: HandShape | None) -> None:
    if shape is None:
        return
    cv2.drawContours(roi_view, [shape.contour], -1, (235, 130, 70), 2, AA)
    cv2.circle(roi_view, shape.palm_center, int(shape.palm_radius), (255, 255, 255), 1, AA)
    cv2.circle(roi_view, shape.palm_center, 4, (255, 255, 255), -1, AA)
    for f in shape.fingers:
        cv2.line(roi_view, shape.palm_center, f.tip, (255, 255, 255), 1, AA)
        cv2.circle(roi_view, f.tip, 8, (11, 158, 245), -1, AA)
        cv2.circle(roi_view, f.tip, 8, (255, 255, 255), 2, AA)


# --------------------------------------------------------------------------
# UI state + buttons
# --------------------------------------------------------------------------
@dataclass
class UiState:
    frame: np.ndarray                         # mirrored camera frame with overlays
    hand_found: bool = False
    label: str | None = None
    confidence: float = 0.0
    ranking: list[tuple[str, float]] = field(default_factory=list)
    source: str = "dataset"
    progress: float = 0.0
    mask: np.ndarray | None = None
    edges: np.ndarray | None = None
    fingers: int = 0
    holes: int = 0
    solidity: float = 0.0
    text: str = ""
    fps: float = 0.0
    ref_count: int = 0
    scope: str = "ALL"
    skin_calibrated: bool = False
    bg_captured: bool = False
    banner: str | None = None                 # calibration countdown message


@dataclass
class Button:
    id: str
    label: str
    rect: tuple[int, int, int, int]
    kind: str = "secondary"                   # secondary | primary | segment


class Gui:
    def __init__(self) -> None:
        self.mouse = (-1, -1)
        self.buttons: list[Button] = self._make_buttons()
        self._base = self._build_base()
        self._cam_mask = np.zeros((CAM_IMG[3] - CAM_IMG[1], CAM_IMG[2] - CAM_IMG[0]), np.uint8)
        rrect(self._cam_mask, (0, 0), (self._cam_mask.shape[1] - 1, self._cam_mask.shape[0] - 1), 255, 12)

    # -- interaction ---------------------------------------------------------
    def hit(self, x: int, y: int) -> str | None:
        for b in self.buttons:
            x1, y1, x2, y2 = b.rect
            if x1 <= x <= x2 and y1 <= y <= y2:
                return b.id
        return None

    @staticmethod
    def _make_buttons() -> list[Button]:
        bs: list[Button] = []
        # scope segmented control (top bar)
        x = 884
        for bid, lab in (("scope_ALL", "All"), ("scope_LETTERS", "Letters"), ("scope_NUMBERS", "Numbers")):
            bs.append(Button(bid, lab, (x, 18, x + 118, 52), "segment"))
            x += 122
        # text card
        x = 856
        for bid, lab in (("space", "Space"), ("back", "Backspace"), ("clear", "Clear")):
            bs.append(Button(bid, lab, (x, 540, x + 118, 570)))
            x += 129
        # setup card
        x = 856
        for bid, lab, kind in (("calib", "Calibrate skin", "primary"),
                               ("bg", "Capture background", "secondary"),
                               ("reset", "Reset", "secondary")):
            w = {"calib": 130, "bg": 150, "reset": 76}[bid]
            bs.append(Button(bid, lab, (x, 630, x + w, 662), kind))
            x += w + 10
        return bs

    # -- static layer -----------------------------------------------------------
    def _build_base(self) -> np.ndarray:
        base = np.full((H, W, 3), BG, np.uint8)
        cards = (CAM_CARD, PRED_CARD, VIS_CARD, TEXT_CARD, CTRL_CARD)

        shadow = np.zeros((H, W), np.float32)
        for x1, y1, x2, y2 in cards:
            cv2.rectangle(shadow, (x1 + 2, y1 + 6), (x2 - 2, y2 + 6), 1.0, -1)
        shadow = cv2.GaussianBlur(shadow, (0, 0), 9)
        base = (base.astype(np.float32) * (1 - 0.09 * shadow[..., None])).astype(np.uint8)
        for c in cards:
            card(base, c)

        # top bar
        rrect(base, (24, 16), (58, 50), ACCENT, 10)
        text_center(base, "F", (41, 33), 0.8, WHITE, 2, FONT_B)
        text(base, "FSL Translator", (72, 32), 0.72, TEXT, 1, FONT_B)
        text(base, "Static Filipino Sign Language  |  letters A-Z, numbers 0-9", (72, 49), 0.40, MUTED)
        rrect(base, (880, 14), (1256, 56), TRACK, 20)

        # card titles
        text(base, "DETECTED SIGN", (856, 112), 0.42, MUTED, 1)
        text(base, "WHAT THE CAMERA SEES", (856, 322), 0.42, MUTED, 1)
        text(base, "TEXT", (856, 480), 0.42, MUTED, 1)
        text(base, "SETUP", (856, 618), 0.42, MUTED, 1)
        text(base, "Confidence", (1016, 146), 0.42, MUTED)
        text(base, "Also considered", (1016, 196), 0.42, MUTED)
        text(base, "Hold steady to add", (1016, 252), 0.42, MUTED)
        return base

    # -- per-frame rendering ----------------------------------------------------
    def render(self, s: UiState) -> np.ndarray:
        img = self._base.copy()
        self._camera(img, s)
        self._prediction(img, s)
        self._vision(img, s)
        self._text_card(img, s)
        self._setup(img, s)
        self._top_info(img, s)
        for b in self.buttons:
            self._button(img, b, s)
        return img

    def _camera(self, img, s: UiState) -> None:
        x1, y1, x2, y2 = CAM_IMG
        w, h = x2 - x1, y2 - y1
        fh, fw = s.frame.shape[:2]
        target = w / h
        if fw / fh > target:
            nw = int(fh * target)
            x0 = (fw - nw) // 2
            crop = s.frame[:, x0:x0 + nw]
        else:
            nh = int(fw / target)
            y0 = (fh - nh) // 2
            crop = s.frame[y0:y0 + nh]
        view = cv2.resize(crop, (w, h), interpolation=cv2.INTER_AREA)

        if s.banner:
            blend_rect(view, (w // 2 - 250, 16), (w // 2 + 250, 62), WHITE, 0.88, 22)
            text_center(view, s.banner, (w // 2, 39), 0.62, TEXT, 1)
        else:
            hint = "Place your hand inside the box" if not s.hand_found else "Hand detected"
            tw = cv2.getTextSize(hint, FONT, 0.5, 1)[0][0]
            blend_rect(view, (w // 2 - tw // 2 - 18, h - 52), (w // 2 + tw // 2 + 18, h - 18), WHITE, 0.85, 17)
            text_center(view, hint, (w // 2, h - 35), 0.5, GREEN if s.hand_found else MUTED, 1)

        region = img[y1:y2, x1:x2]
        region[self._cam_mask > 0] = view[self._cam_mask > 0]

    def _prediction(self, img, s: UiState) -> None:
        shown = s.label is not None
        tile = (856, 122, 996, 262)
        rrect(img, tile[:2], tile[2:], ACCENT_SOFT if shown else TRACK, 22)
        cx, cy = (tile[0] + tile[2]) // 2, (tile[1] + tile[3]) // 2
        if shown:
            text_center(img, s.label, (cx, cy), 4.6, ACCENT, 8, FONT_B)
        else:
            for dx in (-18, 0, 18):
                cv2.circle(img, (cx + dx, cy - 10), 5, (190, 180, 172), -1, AA)
            text_center(img, "waiting", (cx, cy + 26), 0.5, MUTED, 1)

        # source pill
        pill = "Dataset match" if s.source == "dataset" else "Rules only"
        pw = cv2.getTextSize(pill, FONT, 0.4, 1)[0][0] + 22
        rrect(img, (1232 - pw, 94), (1232, 118), TRACK, 11)
        text_center(img, pill, (1232 - pw // 2, 106), 0.4, MUTED)

        conf = s.confidence if shown else 0.0
        text(img, f"{int(conf * 100)}%", (1232, 146), 0.46, TEXT, 1, align="right")
        self._bar(img, (1016, 156, 1232, 166), conf, GREEN if conf >= 0.65 else AMBER if conf >= 0.45 else RED)

        x = 1016
        for lab, sc in (s.ranking[:3] if shown or s.ranking else []):
            rrect(img, (x, 206), (x + 66, 234), TRACK, 12)
            text_center(img, f"{lab} {int(sc * 100)}%", (x + 33, 220), 0.42, TEXT, 1)
            x += 72
        self._bar(img, (1016, 262, 1232, 270), s.progress if shown else 0.0, ACCENT)

    def _vision(self, img, s: UiState) -> None:
        for x, label, src, kind in ((856, "Hand mask", s.mask, "mask"), (964, "Edges", s.edges, "edges")):
            rrect(img, (x, 336), (x + 100, 436), TRACK, 10)
            if src is not None and src.size:
                th = cv2.resize(src, (96, 96), interpolation=cv2.INTER_AREA)
                canvas = np.full((96, 96, 3), 250, np.uint8)
                if kind == "mask":
                    canvas[th > 127] = (235, 175, 140)
                else:
                    canvas[th > 60] = (110, 80, 60)
                m = np.zeros((96, 96), np.uint8)
                rrect(m, (0, 0), (95, 95), 255, 9)
                img[338:434, x + 2:x + 98][m > 0] = canvas[m > 0]
            text(img, label, (x + 6, 430), 0.36, MUTED)

        mx = 1084
        for i, (k, v) in enumerate((("Fingers", str(s.fingers)), ("Holes", str(s.holes)),
                                    ("Solidity", f"{s.solidity:.2f}"))):
            y = 358 + i * 28
            text(img, k, (mx, y), 0.45, MUTED)
            text(img, v if s.hand_found else "-", (1232, y), 0.5, TEXT, 1, align="right")

    def _text_card(self, img, s: UiState) -> None:
        rrect(img, (856, 490), (1232, 530), BG, 10)
        tail = s.text[-20:]
        if tail:
            w = text(img, tail, (868, 518), 0.8, TEXT, 1)
        else:
            w = 0
            text(img, "Held signs appear here", (868, 516), 0.5, MUTED)
        if int(time.time() * 2) % 2 == 0:
            cv2.line(img, (870 + w, 498), (870 + w, 522), ACCENT, 2, AA)

    def _setup(self, img, s: UiState) -> None:
        skin = "calibrated" if s.skin_calibrated else "default"
        bg = "captured" if s.bg_captured else "off"
        text(img, f"Skin: {skin}   |   Background: {bg}   |   Refs: {s.ref_count}", (856, 686), 0.4, MUTED)

    def _top_info(self, img, s: UiState) -> None:
        text(img, f"{s.fps:.0f} FPS", (860, 38), 0.5, MUTED, 1, align="right")

    # -- widgets --------------------------------------------------------------
    @staticmethod
    def _bar(img, rect, value, color) -> None:
        x1, y1, x2, y2 = rect
        rrect(img, (x1, y1), (x2, y2), TRACK, (y2 - y1) // 2)
        w = int((x2 - x1) * float(np.clip(value, 0, 1)))
        if w > (y2 - y1):
            rrect(img, (x1, y1), (x1 + w, y2), color, (y2 - y1) // 2)

    def _button(self, img, b: Button, s: UiState) -> None:
        x1, y1, x2, y2 = b.rect
        hover = self.hit(*self.mouse) == b.id
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        if b.kind == "segment":
            active = b.id == f"scope_{s.scope}"
            if active:
                rrect(img, (x1 + 2, y1 + 2), (x2 - 2, y2 - 2), WHITE, 16)
            text_center(img, b.label, (cx, cy), 0.5, ACCENT if active else (TEXT if hover else MUTED),
                        1 if not active else 2)
            return
        if b.kind == "primary":
            fill, border, fg = (ACCENT_DARK if hover else ACCENT), None, WHITE
        else:
            fill, border, fg = (ACCENT_SOFT if hover else WHITE), BORDER, TEXT
        if border:
            rrect(img, (x1, y1), (x2, y2), border, 10)
            rrect(img, (x1 + 1, y1 + 1), (x2 - 1, y2 - 1), fill, 9)
        else:
            rrect(img, (x1, y1), (x2, y2), fill, 10)
        text_center(img, b.label, (cx, cy), 0.45, fg, 1)
