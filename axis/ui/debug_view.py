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


def recognition_color(rec: Recognition) -> tuple[int, int, int]:
    if rec.kind in ("pinch", "scroll"):
        return GREEN
    if rec.kind == "unknown":
        return RED
    if rec.kind == "none":
        return YELLOW
    return YELLOW


def draw_landmarks(frame: np.ndarray, hand_data: HandData) -> None:
    """Draw landmarks + connections + hand bbox for every detected hand."""
    h, w = frame.shape[:2]
    for hand in hand_data.hands:
        pts = hand.landmarks
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
    cv2.rectangle(frame, (10, 10), (frame.shape[1] - 10, 50), BLACK, -1)
    cv2.putText(
        frame,
        text,
        (20, 38),
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
class DebugView:
    """Owns the debug window and per-frame rendering."""

    window_name: str = "Axis — debug view"
    _trail: list[tuple[int, int]] = field(default_factory=list)

    def show(
        self,
        frame: np.ndarray,
        rec: Recognition,
        hand: Optional[HandData],
        recorder: Optional[MotionRecorder] = None,
        menu_options: Optional[list[str]] = None,
        menu_title: str = "",
    ) -> None:
        """Render one frame and present it. Returns immediately."""
        color = recognition_color(rec)
        if hand is not None:
            draw_landmarks(frame, hand)

            # Record palm trail (primary hand) for registration visibility.
            if rec.kind == "unknown" and rec.registration_candidate is not None:
                primary = hand.primary
                if primary is not None:
                    p = primary.landmarks[9]  # middle MCP as palm center
                    self._trail.append((int(p[0] * frame.shape[1]), int(p[1] * frame.shape[0])))
                    if len(self._trail) > TRAIL_LEN:
                        self._trail.pop(0)
                draw_trail(frame, self._trail)
                draw_signature(frame, rec)
            else:
                self._trail.clear()

        if recorder is not None and recorder.active:
            # Recording: show the live path + progress bar instead of the
            # static recognition label.
            draw_recording(frame, recorder)
        elif menu_options is not None:
            draw_menu(frame, menu_options, menu_title)

        draw_label(frame, rec.label, color)
        if rec.kind == "scroll" and rec.scroll_direction:
            draw_scroll_arrow(frame, rec.scroll_direction)
        elif rec.kind == "zoom" and rec.zoom_direction:
            # Zoom indicator: converging arrows for out, diverging for in.
            h, w = frame.shape[:2]
            cx, cy = w // 2, h // 2
            if rec.zoom_direction == "in":
                cv2.arrowedLine(frame, (cx - 60, cy), (cx - 20, cy), GREEN, 3, cv2.LINE_AA)
                cv2.arrowedLine(frame, (cx + 60, cy), (cx + 20, cy), GREEN, 3, cv2.LINE_AA)
            else:
                cv2.arrowedLine(frame, (cx - 20, cy), (cx - 60, cy), GREEN, 3, cv2.LINE_AA)
                cv2.arrowedLine(frame, (cx + 20, cy), (cx + 60, cy), GREEN, 3, cv2.LINE_AA)
        cv2.imshow(self.window_name, frame)
        cv2.waitKey(1)

    def close(self) -> None:
        cv2.destroyWindow(self.window_name)
