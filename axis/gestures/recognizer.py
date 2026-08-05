"""Gesture engine — recognize and register gestures.

The recognizer combines *static* pose classification (pinch shape vs.
open palm vs. other) with a *dynamic* scroll detector (directional
motion of the hand, projected into the camera plane).

Rules (Phase 1, deliberately simple):
  - pinch: index/middle tips close to thumb tip (ratio < 0.40)
  - scroll: an "extended" hand (index+middle out) moving vertically —
    covers full palm, two-finger, three-finger, and angled hands
  - unknown: a stable pose that matches no known gesture and is held
    for a dwell period → surfaced to the caller as a registration
    candidate.

Motion templates: learned motions (Scroll Down/Up) recorded via the
MotionRecorder. The recognizer tracks each hand's palm-center path and
matches it against stored templates (direction + finger pattern + speed),
so user-specific scroll styles (2 fingers, angled hand, fast/slow) are
recognized without retuning the built-in rules.

Two-hand behavior:
  - both hands are tracked (HandLandmarker num_hands=2)
  - static gestures classify the primary (highest-confidence) hand
  - scroll fires on any extended hand; a moving fist/pinch never scrolls
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

from axis.gestures.definitions import (
    FINGER_INDEX,
    FINGER_MIDDLE,
    GestureDefinition,
    GestureRegistry,
    MotionRegistry,
    PINCH_PAIRS,
    SCROLL_PAIRS,
    distance_to,
    finger_pattern,
    matches,
    normalized_signature,
    palm_size,
)
from axis.landmarks import MIDDLE_MCP, HandData, Hand

# --- built-in gesture builders -----------------------------------------

DEFAULT_TOLERANCE = 0.10
MAX_REGISTRATION_DEVIATION = 0.10
REGISTRATION_DWELL_FRAMES = 10  # ~0.33s at 30fps

# Finger-spread check for open palm (tips away from thumb tip).
SPREAD_MIN_RATIO = 0.50
# Pinch: tip distance / palm size below this.
PINCH_MAX_RATIO = 0.40
# Scroll: vertical velocity of the palm, in palm-sizes per frame.
SCROLL_MIN_SPEED = 0.05
# Stroke latch: minimum net displacement (in palm-sizes) before direction
# is declared. Kills the up/down flapping on near-zero motion.
STROKE_MIN_DISPLACEMENT = 0.25
# Stroke latch: frames the hand must stay below SCROLL_MIN_SPEED before
# the latched direction resets.
STROKE_RESET_FRAMES = 6
# Scroll: require consistent direction across these many consecutive frames.
SCROLL_CONSECUTIVE = 3
# Hand-size gate: ignore hands smaller than this fraction of the frame.
MIN_HAND_SIZE_RATIO = 0.04
# Debounce: require the same static gesture for this many consecutive
# frames before it counts (reduces flicker between pinch/fist/open-palm).
STATIC_DEBOUNCE_FRAMES = 3
# A hand can scroll if index+middle are both extended (covers full palm,
# two/three-finger scrolls, angled hands).
SCROLL_MIN_FINGERS = FINGER_INDEX | FINGER_MIDDLE
# Continuous pinch-zoom: thumb-index distance ratio for the "spread"
# (zoom in) vs. "pinch" (zoom out) states.
ZOOM_SPREAD_RATIO = 0.60   # above this = fingers spread
ZOOM_PINCH_RATIO = 0.30    # below this = fingers pinched


def make_pinch_definition() -> GestureDefinition:
    """Built-in pinch: index + middle tips close to thumb tip."""
    return GestureDefinition(
        id="pinch",
        name="Pinch",
        pairs=PINCH_PAIRS,
        ratios=[0.22, 0.22],
        tolerance=DEFAULT_TOLERANCE,
    )


def make_open_palm_definition() -> GestureDefinition:
    """Built-in open palm: all four fingertips spread from thumb tip."""
    return GestureDefinition(
        id="open_palm",
        name="Open Palm",
        pairs=PINCH_PAIRS,
        ratios=[0.85, 0.85],
        tolerance=DEFAULT_TOLERANCE,
    )


def is_spread(lm: np.ndarray) -> bool:
    """True if index/middle/ring/pinky tips are all far from the thumb tip."""
    size = palm_size(lm)
    if size <= 1e-6:
        return False
    tips = lm[[8, 12, 16, 20], :2]
    thumb = lm[4, :2]
    return bool(np.all(np.linalg.norm(tips - thumb, axis=1) / size > SPREAD_MIN_RATIO))


def is_pinch_shape(lm: np.ndarray) -> bool:
    """True if index + middle tips are both close to the thumb tip.

    A fist also closes the index/middle onto the thumb, so additionally
    require that ring + pinky stay relatively far from the thumb — a
    pinch closes two fingers, a fist closes all four.
    """
    size = palm_size(lm)
    if size <= 1e-6:
        return False
    idx = np.linalg.norm(lm[8, :2] - lm[4, :2]) / size
    mid = np.linalg.norm(lm[12, :2] - lm[4, :2]) / size
    ring = np.linalg.norm(lm[16, :2] - lm[4, :2]) / size
    pinky = np.linalg.norm(lm[20, :2] - lm[4, :2]) / size
    if not (idx < PINCH_MAX_RATIO and mid < PINCH_MAX_RATIO):
        return False
    # Ring + pinky must stay away from the thumb (not a full fist).
    return ring > 0.35 and pinky > 0.35


def is_fist_shape(lm: np.ndarray) -> bool:
    """True if all four fingertips are tucked close to the palm (fist-like).

    Distinct from pinch: fingers together *and* close to the palm center.
    """
    size = palm_size(lm)
    if size <= 1e-6:
        return False
    palm_center = np.mean(lm[[0, 5, 9, 13, 17], :2], axis=0)
    tips = lm[[8, 12, 16, 20], :2]
    return bool(np.mean(np.linalg.norm(tips - palm_center, axis=1)) / size < 0.32)


def can_scroll(lm: np.ndarray) -> bool:
    """True if the hand shape is scroll-compatible (index+middle extended).

    Relaxed: accepts a full open palm, two-finger (index+middle), three-
    finger, and angled hands — any shape where index+middle are out.
    """
    mask = finger_pattern(lm)
    return (mask & SCROLL_MIN_FINGERS) == SCROLL_MIN_FINGERS


# --- recognizer ---------------------------------------------------------

@dataclass
class Recognition:
    """Output of the gesture recognizer for one frame."""

    kind: str          # "pinch" | "scroll" | "open_palm" | "fist" | "none" | "zoom"
    label: str         # human-readable
    confidence: float  # 0..1
    registration_candidate: Optional[GestureDefinition] = None
    scroll_direction: str = ""  # "up" | "down" | "left" | "right" | ""
    zoom_direction: str = ""    # "in" | "out" | ""


class GestureRecognizer:
    def __init__(
        self,
        registry: GestureRegistry,
        motions: Optional[MotionRegistry] = None,
    ):
        self.registry = registry
        self.motions = motions or MotionRegistry()
        # Per-hand scroll tracking state (keyed by hand identity).
        self._scroll: Dict[str, dict] = {}
        # Per-hand zoom state (thumb-index ratio tracking).
        self._zoom: Dict[str, dict] = {}
        self._unknown_frames = 0
        self._last_registration: Optional[GestureDefinition] = None
        self._static_state = ""
        self._static_count = 0
        self._scroll_direction = ""

    # --- static classification -----------------------------------------

    def _classify_static(self, lm: np.ndarray) -> Optional[tuple[str, float]]:
        """Return (kind, confidence) for a static pose, or None if nothing."""
        if is_pinch_shape(lm):
            return "pinch", 0.9
        if is_fist_shape(lm):
            return "fist", 0.8
        if is_spread(lm):
            return "open_palm", 0.8
        return None

    def _debounced_static(self, kind: Optional[tuple[str, float]]) -> Optional[tuple[str, float]]:
        """Require the same static kind for STATIC_DEBOUNCE_FRAMES frames."""
        if kind is None:
            self._static_state = ""
            self._static_count = 0
            return None
        k, conf = kind
        if k == self._static_state:
            self._static_count += 1
        else:
            self._static_state = k
            self._static_count = 1
        if self._static_count >= STATIC_DEBOUNCE_FRAMES:
            return k, conf
        return None

    # --- dynamic scroll -------------------------------------------------

    def _hand_key(self, hand: Hand) -> str:
        """Stable identity for a hand across frames.

        MediaPipe tracks at most 2 hands and reports stable handedness
        (Left/Right) per physical hand, so the handedness label is a
        reliable key. (A positional hash would break as the hand moves.)
        """
        return hand.handedness

    def _update_scroll(self, hand: Hand) -> Tuple[str, float, str]:
        """Track one hand's palm-center motion with a stroke latch.

        Returns (kind, confidence, direction). Direction is only declared
        once net displacement over the window exceeds STROKE_MIN_DISPLACEMENT,
        then *latched* until the hand stops (speed below threshold for
        STROKE_RESET_FRAMES). This prevents the up/down flapping that comes
        from per-frame velocity sign.

        Supports both vertical (up/down) and horizontal (left/right) strokes.
        """
        lm = hand.landmarks
        if not can_scroll(lm):
            self._scroll.pop(self._hand_key(hand), None)
            return "none", 0.0, ""

        size = palm_size(lm)
        if size <= 1e-6:
            return "none", 0.0, ""

        key = self._hand_key(hand)
        st = self._scroll.setdefault(
            key, {"buffer": [], "latched": "", "still_frames": 0}
        )

        # Palm center: mean of wrist + the 4 MCPs (stable under rotation).
        center = np.mean(lm[[0, 5, 9, 13, 17], :2], axis=0)
        st["buffer"].append(center.copy())
        if len(st["buffer"]) > 12:
            st["buffer"].pop(0)
        if len(st["buffer"]) < 4:
            return "none", 0.0, ""

        first = st["buffer"][0]
        last = st["buffer"][-1]
        dx = (last[0] - first[0]) / size   # net X displacement in palm-sizes
        dy = (last[1] - first[1]) / size   # net Y displacement in palm-sizes
        speed = np.hypot(dx, dy) / len(st["buffer"]) * 10.0  # palm-sizes/frame

        # If the hand has stopped, count down to resetting the latch.
        if speed < SCROLL_MIN_SPEED:
            st["still_frames"] += 1
            if st["still_frames"] >= STROKE_RESET_FRAMES:
                st["latched"] = ""
                st["buffer"].clear()
            return "none", 0.0, ""

        st["still_frames"] = 0

        # Direction from net displacement — only if it exceeds the latch
        # threshold (avoids declaring a direction on jitter).
        direction = ""
        disp = np.hypot(dx, dy)
        if disp >= STROKE_MIN_DISPLACEMENT:
            if abs(dy) >= abs(dx):
                direction = "down" if dy > 0 else "up"
            else:
                direction = "right" if dx > 0 else "left"
            st["latched"] = direction

        if not st["latched"]:
            return "none", 0.0, ""

        # Latched direction persists until the hand stops.
        return "scroll", min(1.0, speed / 0.3), st["latched"]

    def _update_zoom(self, hand: Hand) -> Tuple[str, float, str]:
        """Continuous pinch-zoom: thumb-index distance over time.

        Tracks the ratio r = dist(thumb,index) / palm_size. When the hand
        spreads (r crosses ZOOM_SPREAD_RATIO going up) it's zoom-in; when
        it pinches (r crosses ZOOM_PINCH_RATIO going down) it's zoom-out.

        Fires *once per crossing* (edge-triggered): a held pinch at rest
        does not keep zooming — that's a click, not a zoom. Re-arm happens
        when the ratio moves back across the opposite threshold.
        """
        lm = hand.landmarks
        size = palm_size(lm)
        if size <= 1e-6:
            return "none", 0.0, ""
        key = self._hand_key(hand)
        st = self._zoom.setdefault(key, {"last_ratio": None, "armed": True})
        ratio = float(np.linalg.norm(lm[8, :2] - lm[4, :2]) / size)

        direction = ""
        if st["last_ratio"] is not None:
            prev = st["last_ratio"]
            if st["armed"] and ratio > ZOOM_SPREAD_RATIO and prev <= ZOOM_SPREAD_RATIO:
                direction = "in"
            elif st["armed"] and ratio < ZOOM_PINCH_RATIO and prev >= ZOOM_PINCH_RATIO:
                direction = "out"
        st["last_ratio"] = ratio

        if direction:
            st["armed"] = False  # fired — re-arm after returning to neutral
        elif ratio > ZOOM_PINCH_RATIO and ratio < ZOOM_SPREAD_RATIO:
            st["armed"] = True  # back to neutral range → can fire again

        if direction:
            return "zoom", 0.8, direction
        return "none", 0.0, ""

    def _reset_scroll(self) -> None:
        self._scroll.clear()

    # --- motion template matching ---------------------------------------

    def _match_motion(self, hands: List[Hand]) -> Optional[Recognition]:
        """Check live motion against learned templates (e.g. Scroll Down).

        Uses the primary hand's latched stroke direction + finger pattern +
        speed, matched against MotionRegistry templates with tolerance.
        """
        if not hands:
            return None
        primary = max(hands, key=lambda h: h.score)
        mask = finger_pattern(primary.landmarks)
        st = self._scroll.get(self._hand_key(primary))
        if st is None or not st.get("latched"):
            return None
        direction = st["latched"]
        # Speed from the recent path.
        buf = st.get("buffer", [])
        if len(buf) < 2:
            return None
        size = max(palm_size(primary.landmarks), 1e-6)
        disp = np.linalg.norm(buf[-1] - buf[0]) / size
        speed = disp / len(buf) * 10.0
        tpl = self.motions.find(direction, mask, speed)
        if tpl is None:
            return None
        return Recognition(
            kind="scroll",
            label=tpl.name,
            confidence=0.85,
            scroll_direction=direction,
        )

    # --- registration ---------------------------------------------------

    def _update_registration(self, lm: np.ndarray) -> Optional[GestureDefinition]:
        """Detect a held, deliberate, unknown pose → registration candidate."""
        if self._classify_static(lm) is not None:
            self._unknown_frames = 0
            return None
        if is_pinch_shape(lm):
            self._unknown_frames = 0
            return None
        for g in self.registry.definitions.values():
            if distance_to(lm, g) < MAX_REGISTRATION_DEVIATION:
                self._unknown_frames = 0
                return None
        self._unknown_frames += 1
        if self._unknown_frames >= REGISTRATION_DWELL_FRAMES:
            self._unknown_frames = 0
            candidate = GestureDefinition(
                id="custom_" + str(len(self.registry.definitions)),
                name="Unnamed Gesture",
                pairs=PINCH_PAIRS,
                ratios=normalized_signature(lm, PINCH_PAIRS),
                tolerance=DEFAULT_TOLERANCE,
            )
            self._last_registration = candidate
            return candidate
        return None

    # --- main entry -----------------------------------------------------

    def recognize(self, hands: List[Hand]) -> Recognition:
        """Classify one frame's hands into a Recognition result."""
        if not hands:
            self._reset_scroll()
            self._zoom.clear()
            self._unknown_frames = 0
            self._static_state = ""
            self._static_count = 0
            return Recognition(kind="none", label="No hand", confidence=0.0)

        primary = max(hands, key=lambda h: h.score)
        lm = primary.landmarks
        size = palm_size(lm)
        if size <= 1e-6 or size < MIN_HAND_SIZE_RATIO:
            self._reset_scroll()
            self._zoom.clear()
            self._unknown_frames = 0
            self._static_state = ""
            self._static_count = 0
            return Recognition(kind="none", label="Tracking...", confidence=0.3)

        # Continuous pinch-zoom: a pinching/spreading hand is zooming, not
        # clicking. Takes priority over static pinch.
        zoom_kind, zoom_conf, zoom_dir = "none", 0.0, ""
        for h in hands:
            k, c, d = self._update_zoom(h)
            if k == "zoom" and c > zoom_conf:
                zoom_kind, zoom_conf, zoom_dir = k, c, d
        if zoom_kind == "zoom":
            self._static_state = ""
            self._static_count = 0
            label = "Zooming in" if zoom_dir == "in" else "Zooming out"
            return Recognition(
                kind="zoom",
                label=label,
                confidence=zoom_conf,
                zoom_direction=zoom_dir,
            )

        # Learned motion templates get priority (user-specific styles).
        tpl_rec = self._match_motion(hands)
        if tpl_rec is not None:
            return tpl_rec

        # Scroll: any scroll-compatible hand moving in a stroke scrolls
        # (vertical or horizontal, direction latched).
        scroll_kind, scroll_conf, scroll_dir = "none", 0.0, ""
        for h in hands:
            k, c, d = self._update_scroll(h)
            if k == "scroll" and c > scroll_conf:
                scroll_kind, scroll_conf, scroll_dir = k, c, d
        if scroll_kind == "scroll":
            self._static_state = ""
            self._static_count = 0
            return Recognition(
                kind="scroll",
                label="You are trying to scroll",
                confidence=scroll_conf,
                scroll_direction=scroll_dir,
            )

        static = self._debounced_static(self._classify_static(lm))
        if static is not None:
            kind, conf = static
            labels = {
                "pinch": "You are trying to click",
                "fist": "Fist",
                "open_palm": "Open palm",
            }
            return Recognition(kind=kind, label=labels[kind], confidence=conf)

        candidate = self._update_registration(lm)
        if candidate is not None:
            return Recognition(
                kind="unknown",
                label="New gesture detected",
                confidence=0.6,
                registration_candidate=candidate,
            )

        return Recognition(kind="none", label="Tracking...", confidence=0.3)
