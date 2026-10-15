"""Debug view — live camera feed with gesture overlays (dev harness).

Draws, per frame:
  - the camera feed
  - a bounding box around each detected hand/palm (both hands when two are up)
  - the 21 hand landmarks + connections per hand
  - an overlay label showing the current recognition state
  - a scroll direction arrow when scrolling
  - during registration: a motion trail of the last N hand positions and a
    signature overlay showing the landmark pairs the gesture was built from

Colors follow the PRD §9.1 scheme:
  green  — gesture recognized
  yellow — tracking active
  red    — gesture not recognized / registration prompt
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Optional

import cv2
import numpy as np

from axis.gestures.recorder import MotionRecorder
from axis.gestures.recognizer import Recognition
from axis.landmarks import HandData

# PRD §9.1 colors (BGR)
GREEN = (0, 180, 0)
YELLOW = (0, 200, 255)
RED = (0, 0, 220)
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)

LANDMARK_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),          # thumb
    (0, 5), (5, 6), (6, 7), (7, 8),          # index
    (5, 9), (9, 10), (10, 11), (11, 12),     # middle
    (9, 13), (13, 14), (14, 15), (15, 16),   # ring
    (13, 17), (17, 18), (18, 19), (19, 20),  # pinky
    (0, 17),                                 # palm
]

TRAIL_LEN = 24  # frames of palm-position history for the registration trail

TUTORIAL_STEPS: list[tuple[str, str]] = [
    (
        "1/6 MOVE — show an OPEN PALM and drift it",
        "Open palm (all fingers out). Drift slowly: cursor follows. Ring = green.",
    ),
    (
        "2/6 CLICK — PINCH index + middle to thumb",
        "Quick pinch = left-click. Pinch twice = double-click. Hand stays still.",
    ),
    (
        "3/6 SCROLL — TWO FINGERS (index + middle, pinky folded) + move",
        "Hold two fingers up and stroke. Up strokes scroll up, down strokes scroll down.",
    ),
    (
        "4/6 DRAG — PINCH, HOLD 0.3s, then move",
        "Pinch and hold until the ring turns blue, then drag. Release to drop.",
    ),
    (
        "5/6 CLUTCH — make a FIST to freeze",
        "Fist = freeze (yellow ring). Cursor holds while you re-center your hand.",
    ),
    (
        "6/6 DOCK — open palm to the screen bottom",
        "Move down slowly with an open palm to reach the Dock. Press T to exit.",
    ),
]


def recognition_color(rec: Recognition) -> tuple[int, int, int]:
    if rec.kind in ("pinch", "scroll"):
        return GREEN
    if rec.kind == "unknown":
        return RED
    if rec.kind == "none":
        return YELLOW
    return YELLOW


def draw_landmarks(
    frame: np.ndarray, hand_data: HandData, mirror_x: bool = False
) -> None:
    """Draw landmarks + connections + hand bbox for every detected hand.

    When the preview is shown as a selfie view (mirrored), the landmark
    x-coords must be mirrored too — otherwise the skeleton draws on the
    opposite side from the hand the user sees.
    """
    h, w = frame.shape[:2]
    for hand in hand_data.hands:
        pts = hand.landmarks.copy()
        if mirror_x:
            pts[:, 0] = 1.0 - pts[:, 0]
        for a, b in LANDMARK_CONNECTIONS:
            p1 = (int(pts[a][0] * w), int(pts[a][1] * h))
            p2 = (int(pts[b][0] * w), int(pts[b][1] * h))
            cv2.line(frame, p1, p2, GREEN, 1)
        for p in pts:
            cv2.circle(frame, (int(p[0] * w), int(p[1] * h)), 3, WHITE, -1)
        xs = pts[:, 0] * w
        ys = pts[:, 1] * h
        x0, y0 = int(xs.min()), int(ys.min())
        x1, y1 = int(xs.max()), int(ys.max())
        cv2.rectangle(frame, (x0, y0), (x1, y1), GREEN, 2)


def draw_label(frame: np.ndarray, text: str, color: tuple[int, int, int]) -> None:
    """Overlay a status label at the top of the frame."""
    draw_label_at(frame, text, color, y_top=10)


def draw_label_at(frame: np.ndarray, text: str, color: tuple[int, int, int], y_top: int) -> None:
    """Overlay a status label at an explicit vertical offset."""
    h = frame.shape[0]
    cv2.rectangle(frame, (10, y_top), (frame.shape[1] - 10, y_top + 40), BLACK, -1)
    cv2.putText(
        frame,
        text,
        (20, y_top + 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        color,
        2,
        cv2.LINE_AA,
    )


def draw_scroll_arrow(frame: np.ndarray, direction: str) -> None:
    """Draw an arrow indicating scroll direction (up/down)."""
    h, w = frame.shape[:2]
    cx, cy = w // 2, h // 2
    tip = (cx, cy - 40) if direction == "up" else (cx, cy + 40)
    base = (cx, cy + 20) if direction == "up" else (cx, cy - 20)
    cv2.arrowedLine(frame, base, tip, GREEN, 3, cv2.LINE_AA)


def draw_trail(frame: np.ndarray, trail: list[tuple[int, int]]) -> None:
    """Draw the recent palm-position trail as a fading polyline."""
    if len(trail) < 2:
        return
    for i in range(1, len(trail)):
        alpha = i / len(trail)
        color = (int(0 * alpha), int(180 * alpha), int(220 * (1 - alpha)))
        cv2.line(frame, trail[i - 1], trail[i], color, 2, cv2.LINE_AA)


def draw_recording(frame: np.ndarray, recorder: MotionRecorder) -> None:
    """Overlay the recording progress bar + live trail + hint text."""
    h, w = frame.shape[:2]
    # Progress bar along the bottom.
    frac = min(1.0, recorder.elapsed / recorder.duration_s)
    bar_w = int(w * 0.7)
    x0, y0 = (w - bar_w) // 2, h - 40
    cv2.rectangle(frame, (x0, y0), (x0 + bar_w, y0 + 12), (60, 60, 60), -1)
    cv2.rectangle(frame, (x0, y0), (x0 + int(bar_w * frac), y0 + 12), GREEN, -1)
    # Hint text above the bar.
    cv2.putText(
        frame,
        "Recording — perform the gesture a few times",
        (x0, y0 - 12),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        YELLOW,
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        f"{recorder.remaining:.1f}s remaining — press ESC to cancel",
        (x0, y0 + 34),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        WHITE,
        1,
        cv2.LINE_AA,
    )


def draw_menu(frame: np.ndarray, options: list[str], title: str) -> None:
    """Overlay a choose-an-option menu (classify the recorded gesture)."""
    h, w = frame.shape[:2]
    lines = [title] + [f"  {i + 1}. {opt}" for i, opt in enumerate(options)]
    box_h = 30 + 24 * len(lines)
    y0 = (h - box_h) // 2
    cv2.rectangle(frame, (w // 2 - 220, y0 - 10), (w // 2 + 220, y0 + box_h), (0, 0, 0), -1)
    cv2.rectangle(frame, (w // 2 - 220, y0 - 10), (w // 2 + 220, y0 + box_h), WHITE, 1)
    for i, txt in enumerate(lines):
        cv2.putText(
            frame,
            txt,
            (w // 2 - 200, y0 + 24 + i * 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            GREEN if i == 0 else WHITE,
            1,
            cv2.LINE_AA,
        )


def draw_signature(frame: np.ndarray, rec: Recognition) -> None:
    """Overlay the registered gesture's signature (landmark pairs + ratios).

    Shows which landmark pairs the new gesture was captured from, so the
    user can see *what* was recognized.
    """
    cand = rec.registration_candidate
    if cand is None:
        return
    h, w = frame.shape[:2]
    lines = [
        f"New gesture: {cand.name}",
        f"pairs: {[(a, b) for a, b in cand.pairs]}",
        f"ratios: {[round(r, 2) for r in cand.ratios]}",
    ]
    y0 = h - 90
    cv2.rectangle(frame, (10, y0 - 10), (w - 10, y0 + 60), BLACK, -1)
    for i, txt in enumerate(lines):
        cv2.putText(
            frame,
            txt,
            (20, y0 + i * 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            WHITE,
            1,
            cv2.LINE_AA,
        )


@dataclass
class NavButton:
    """A clickable navbar button."""

    label: str
    x: int
    y: int
    w: int
    h: int
    active: bool = False  # highlighted (e.g. current selection / armed)

    def contains(self, px: int, py: int) -> bool:
        return self.x <= px <= self.x + self.w and self.y <= py <= self.y + self.h


NAVBAR_H = 56


@dataclass
class DebugView:
    """Owns the debug window and per-frame rendering."""

    window_name: str = "Axis — debug view"
    _trail: list[tuple[int, int]] = field(default_factory=list)
    _hand_tracked: bool = False
    # Gesture selection for training.
    gesture_names: list[str] = field(
        default_factory=lambda: [
            "Scroll Down",
            "Scroll Up",
            "Scroll Left",
            "Scroll Right",
            "Zoom In",
            "Zoom Out",
            "Custom",
        ]
    )
    gesture_idx: int = 0
    training: bool = False  # TRAIN pressed → recording loop active
    dropdown_open: bool = False
    toast: str = ""
    toast_until: float = 0.0
    record_count: int = 0  # completed 20s windows (for the 60s prompt)
    tutorial_active: bool = False
    tutorial_step: int = 0
    mirror_preview: bool = True  # selfie view: flip the feed horizontally
    _flipbuf: Optional[np.ndarray] = None  # reused flip destination buffer
    _continue_done: bool = True
    _buttons: list[NavButton] = field(default_factory=list)
    _last_frame_wh: tuple[int, int] = (640, 480)

    @property
    def selected_gesture(self) -> str:
        return self.gesture_names[self.gesture_idx]

    def _layout_buttons(self, w: int) -> None:
        """Position the navbar buttons for the current frame width."""
        self._buttons = []
        gap = 8
        y = 8
        h = 40
        x = 10
        # Gesture selector (opens dropdown). ASCII only — OpenCV's built-in
        # font can't render '▼', it would show as '???'.
        sel_w = 200
        self._buttons.append(
            NavButton(self.selected_gesture + " [+]", x, y, sel_w, h, active=True)
        )
        x += sel_w + gap
        # TRAIN / STOP toggle.
        train_w = 90
        label = "STOP" if self.training else "TRAIN"
        self._buttons.append(
            NavButton(label, x, y, train_w, h, active=self.training)
        )
        x += train_w + gap
        # Recording counter (e.g. "REC 2").
        rec_w = 90
        rec_label = f"REC {self.record_count}" if self.training else "REC 0"
        self._buttons.append(NavButton(rec_label, x, y, rec_w, h, active=False))
        x += rec_w + gap
        # Hand-tracked indicator (right-aligned).
        track_label = "HAND: YES" if self._hand_tracked else "HAND: NO"
        track_w = 110
        x = w - track_w - 52 - 44
        self._buttons.append(
            NavButton(track_label, x, y, track_w, h, active=self._hand_tracked)
        )
        # HELP (?) — toggles the gesture tutorial.
        help_w = 34
        x = w - track_w - 52
        self._buttons.append(
            NavButton("?", x, y, help_w, h, active=self.tutorial_active)
        )
        # Close (X) — left of the HAND button, not overlapping.
        close_w = 34
        x = w - close_w - 10
        self._buttons.append(NavButton("X", x, y, close_w, h))
        self._buttons[-1].active = False

    def _draw_dropdown(self, frame: np.ndarray) -> None:
        """Draw the gesture dropdown below the selector button."""
        if not self.dropdown_open or not self._buttons:
            return
        b = self._buttons[0]
        item_h = 26
        x0, y0 = b.x, b.y + b.h + 2
        w = b.w
        for i, name in enumerate(self.gesture_names):
            yi = y0 + i * item_h
            selected = i == self.gesture_idx
            bg = (0, 100, 0) if selected else (50, 50, 50)
            cv2.rectangle(frame, (x0, yi), (x0 + w, yi + item_h), bg, -1)
            cv2.rectangle(frame, (x0, yi), (x0 + w, yi + item_h), WHITE, 1)
            cv2.putText(
                frame,
                name,
                (x0 + 8, yi + 18),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                WHITE,
                1,
                cv2.LINE_AA,
            )

    def _draw_navbar(self, frame: np.ndarray) -> None:
        h, w = frame.shape[:2]
        cv2.rectangle(frame, (0, 0), (w, NAVBAR_H), (40, 40, 40), -1)
        self._layout_buttons(w)
        for b in self._buttons:
            color = (0, 120, 0) if b.active else (90, 90, 90)
            cv2.rectangle(frame, (b.x, b.y), (b.x + b.w, b.y + b.h), color, -1)
            cv2.rectangle(frame, (b.x, b.y), (b.x + b.w, b.y + b.h), WHITE, 1)
            cv2.putText(
                frame,
                b.label,
                (b.x + 8, b.y + 26),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                WHITE,
                1,
                cv2.LINE_AA,
            )
        self._draw_dropdown(frame)

    def _draw_toast(self, frame: np.ndarray) -> None:
        if not self.toast or time.monotonic() > self.toast_until:
            return
        h, w = frame.shape[:2]
        (tw, th), _ = cv2.getTextSize(self.toast, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
        x0 = (w - tw) // 2 - 10
        y0 = h - 50
        cv2.rectangle(frame, (x0, y0), (x0 + tw + 20, y0 + th + 16), BLACK, -1)
        cv2.putText(
            frame,
            self.toast,
            (x0 + 10, y0 + th + 6),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            GREEN,
            1,
            cv2.LINE_AA,
        )

    def _draw_tutorial(self, frame: np.ndarray) -> None:
        """Draw the tutorial step card (center) when the tutorial is on."""
        if not self.tutorial_active:
            return
        idx = min(self.tutorial_step, len(TUTORIAL_STEPS) - 1)
        title, body = TUTORIAL_STEPS[idx]
        h, w = frame.shape[:2]
        lines = [title, body, "T = next step, HELP (?) = exit"]
        box_w, box_h = 520, 30 + 24 * len(lines)
        x0 = max((w - box_w) // 2, 6)
        y0 = (h - box_h) // 2
        cv2.rectangle(frame, (x0, y0), (x0 + box_w, y0 + box_h), (0, 0, 0), -1)
        cv2.rectangle(frame, (x0, y0), (x0 + box_w, y0 + box_h), WHITE, 1)
        for i, txt in enumerate(lines):
            cv2.putText(
                frame,
                txt[:72],
                (x0 + 12, y0 + 24 + i * 24),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                GREEN if i == 0 else WHITE,
                1,
                cv2.LINE_AA,
            )

    def tutorial_advance(self) -> None:
        """Step forward; completing the last step exits the tutorial."""
        if self.tutorial_step + 1 >= len(TUTORIAL_STEPS):
            self.tutorial_active = False
            self.tutorial_step = 0
            self._toast("Tutorial done")
        else:
            self.tutorial_step += 1
            self._toast(f"Tutorial {self.tutorial_step + 1}/{len(TUTORIAL_STEPS)}")

    def toggle_tutorial(self) -> None:
        self.tutorial_active = not self.tutorial_active
        self.tutorial_step = 0
        self._toast(
            "Tutorial ON — press T for next step"
            if self.tutorial_active
            else "Tutorial off"
        )

    def tutorial_observe(self, rec: Recognition, control_state: str = "") -> None:
        """Auto-advance on success: the step completes when the live
        recognition + control state show the gesture working."""
        if not self.tutorial_active:
            return
        step = self.tutorial_step
        done = False
        if step == 0:
            done = control_state == "tracking"
        elif step == 1:
            done = rec.kind == "pinch"
        elif step == 2:
            done = rec.kind == "scroll"
        elif step == 3:
            done = control_state == "dragging"
        elif step == 4:
            done = rec.kind == "fist" or control_state == "clutched"
        elif step == 5:
            done = control_state == "tracking"
        if done:
            self.tutorial_advance()

    def show(
        self,
        frame: np.ndarray,
        rec: Recognition,
        hand: Optional[HandData],
        recorder: Optional[MotionRecorder] = None,
        menu_options: Optional[list[str]] = None,
        menu_title: str = "",
        cursor_pos: Optional[tuple[int, int]] = None,
    ) -> None:
        """Render one frame and present it. Returns immediately.

        Key handling is done by the caller (main loop), not here — a
        waitKey() inside this method would swallow keyboard input.

        `cursor_pos` is the virtual-cursor position in screen points
        (top-left origin) when the overlay is active — drawn as a mirror
        dot on the frame for diagnostics.
        """
        self._hand_tracked = hand is not None and len(hand.hands) > 0
        color = recognition_color(rec)
        if self.mirror_preview:
            # Selfie view: show the flipped pixels AND flip the landmark
            # x-coords before drawing, so the skeleton/trail sit exactly
            # on the hand the user sees (not mirrored to the wrong side).
            # The destination buffer is reused — no full-frame copy/frame.
            if self._flipbuf is None or self._flipbuf.shape != frame.shape:
                self._flipbuf = np.empty_like(frame)
            cv2.flip(frame, 1, dst=self._flipbuf)
            frame = self._flipbuf
        if hand is not None:
            draw_landmarks(frame, hand, mirror_x=self.mirror_preview)
            if rec.kind == "unknown" and rec.registration_candidate is not None:
                primary = hand.primary
                if primary is not None:
                    px = 1.0 - primary.landmarks[9][0] if self.mirror_preview else primary.landmarks[9][0]
                    p = (px, primary.landmarks[9][1])
                    self._trail.append((int(p[0] * frame.shape[1]), int(p[1] * frame.shape[0])))
                    if len(self._trail) > TRAIL_LEN:
                        self._trail.pop(0)
                draw_trail(frame, self._trail)
                draw_signature(frame, rec)
            else:
                self._trail.clear()

        if recorder is not None and recorder.active:
            draw_recording(frame, recorder)
        elif menu_options is not None:
            draw_menu(frame, menu_options, menu_title)

        # Shift the status label below the navbar.
        draw_label_at(frame, rec.label, color, y_top=NAVBAR_H + 6)
        if rec.kind == "scroll" and rec.scroll_direction:
            draw_scroll_arrow(frame, rec.scroll_direction)
        elif rec.kind == "zoom" and rec.zoom_direction:
            h, w = frame.shape[:2]
            cx, cy = w // 2, h // 2
            if rec.zoom_direction == "in":
                cv2.arrowedLine(frame, (cx - 60, cy), (cx - 20, cy), GREEN, 3, cv2.LINE_AA)
                cv2.arrowedLine(frame, (cx + 60, cy), (cx + 20, cy), GREEN, 3, cv2.LINE_AA)
            else:
                cv2.arrowedLine(frame, (cx - 20, cy), (cx - 60, cy), GREEN, 3, cv2.LINE_AA)
                cv2.arrowedLine(frame, (cx + 20, cy), (cx + 60, cy), GREEN, 3, cv2.LINE_AA)

        self._last_frame_wh = (int(frame.shape[1]), int(frame.shape[0]))
        if cursor_pos is not None:
            # Mirror dot: normalized screen position → frame image coords.
            h, w = frame.shape[:2]
            from axis.cursor.controller import RelativePointer  # lazy: avoid cycle

            # The caller passes screen points; normalize against a nominal
            # screen. We only know the frame, so scale by frame dims.
            mx = int((cursor_pos[0] / 1440.0) * w)
            my = int((cursor_pos[1] / 900.0) * h)
            mx = min(max(mx, 8), w - 8)
            my = min(max(my, 8), h - 8)
            cv2.circle(frame, (mx, my), 7, GREEN, 2)
            cv2.circle(frame, (mx, my), 3, GREEN, -1)
        self._draw_navbar(frame)
        self._draw_toast(frame)
        self._draw_tutorial(frame)
        cv2.imshow(self.window_name, frame)

    def map_coords(self, x: int, y: int) -> tuple[int, int]:
        """Map window (mouse) coords to image coords.

        getWindowImageRect returns the window's screen rect (x,y is the
        screen position, rw/rh is the window size). The mouse callback
        already reports coords in window-content space (0,0 at the content
        origin), so we must NOT subtract rx/ry — we only scale when the
        window has been resized bigger than the image.
        """
        fw, fh = self._last_frame_wh
        try:
            _rx, _ry, rw, rh = cv2.getWindowImageRect(self.window_name)
        except cv2.error:
            return x, y
        if rw <= 0 or rh <= 0:
            return x, y
        if abs(rw - fw) <= 2 and abs(rh - fh) <= 2:
            return x, y
        return int(x * fw / rw), int(y * fh / rh)

    def handle_click(self, x: int, y: int) -> Optional[str]:
        """Route a navbar click; returns a command or None.

        Commands: "open_dropdown", "select_gesture:<name>", "toggle_train",
        "toggle_tutorial", "close". Dropdown item clicks are handled first.
        """
        # If the dropdown is open, check item clicks first.
        if self.dropdown_open and self._buttons:
            b = self._buttons[0]
            item_h = 26
            x0, y0 = b.x, b.y + b.h + 2
            if x0 <= x <= x0 + b.w:
                for i, name in enumerate(self.gesture_names):
                    yi = y0 + i * item_h
                    if yi <= y <= yi + item_h:
                        self.gesture_idx = i
                        self.dropdown_open = False
                        self._toast(f"Gesture: {name}")
                        return f"select_gesture:{name}"
        self.dropdown_open = False

        for i, b in enumerate(self._buttons):
            if not b.contains(x, y):
                continue
            if i == 0:
                # Gesture selector → open/close dropdown.
                self.dropdown_open = not self.dropdown_open
                if self.dropdown_open:
                    self._toast("Pick one — or press 1-7")
                return "open_dropdown"
            if i == 1:
                self.training = not self.training
                self._toast(
                    "Recording — press STOP to end"
                    if self.training
                    else "Recording stopped"
                )
                return "toggle_train"
            if b.label == "X":
                return "close"
            if b.label == "?":
                self.toggle_tutorial()
                return "toggle_tutorial"
        return None

    def cycle_gesture(self, delta: int = 1) -> str:
        """Move the gesture selection by delta (keyboard fallback)."""
        n = len(self.gesture_names)
        self.gesture_idx = (self.gesture_idx + delta) % n
        self._toast(f"Gesture: {self.selected_gesture}")
        return self.selected_gesture

    def select_gesture_by_index(self, idx: int) -> Optional[str]:
        """Select gesture by index (keyboard number shortcut)."""
        if 0 <= idx < len(self.gesture_names):
            self.gesture_idx = idx
            self._toast(f"Gesture: {self.selected_gesture}")
            return self.selected_gesture
        return None

    def _toast(self, msg: str) -> None:
        self.toast = msg
        self.toast_until = time.monotonic() + 3.0

    def close(self) -> None:
        cv2.destroyWindow(self.window_name)
