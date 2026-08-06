"""Motion recorder — captures a gesture's path for learning.

When the recognizer can't classify a motion, the app records a short
window of hand positions (the "start → middle → end" the user asked for),
computes the dominant direction, finger pattern, and speed, then offers
to save it as a named motion template (Scroll Down, Scroll Up, or custom).

Recording is manual-mode: it runs for a fixed duration (default 8s) so
the user can perform the gesture a few times. Every frame updates the
recording state; the debug view renders a progress bar + live trail.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

from axis.gestures.definitions import (
    FINGER_INDEX,
    FINGER_MIDDLE,
    FINGER_PINKY,
    FINGER_RING,
    FINGER_THUMB,
    MotionTemplate,
    finger_pattern,
    palm_size,
)
from axis.gestures.embedding import path_embedding
from axis.landmarks import MIDDLE_MCP, Hand

MIN_VELOCITY_FOR_DIRECTION = 0.005  # normalized units/frame; below = stationary
SPEED_PALM_SIZE_FACTOR = 10.0       # velocity in palm-sizes/frame


@dataclass
class MotionSample:
    """One frame's hand state during recording."""

    x: float      # normalized palm x (0..1)
    y: float      # normalized palm y (0..1)
    mask: int     # finger pattern bitmask
    speed: float  # velocity in palm-sizes/frame (smoothed)
    palm_size: float = 0.0  # palm size (normalized) at this sample
    t: float = 0.0          # seconds since recording start


@dataclass
class MotionRecording:
    """The captured motion, ready to become a MotionTemplate."""

    samples: List[MotionSample] = field(default_factory=list)
    direction: str = ""       # dominant direction from the samples
    finger_mask: int = 0      # most common finger pattern
    speed: float = 0.0        # mean speed in palm-sizes/frame
    valid: bool = False       # True if enough motion captured
    embedding: np.ndarray = None  # fixed-length path embedding

    def to_template(self, name: str, gesture_id: str) -> MotionTemplate:
        return MotionTemplate(
            id=gesture_id,
            name=name,
            direction=self.direction,
            finger_mask=self.finger_mask,
            speed=self.speed,
            embedding=self.embedding,
        )


class MotionRecorder:
    """Records N seconds of palm motion and summarizes it."""

    def __init__(self, duration_s: float = 8.0):
        self.duration_s = duration_s
        self._started_at: Optional[float] = None
        self._last: Optional[Hand] = None
        self._last_pos: Optional[np.ndarray] = None
        self._samples: List[MotionSample] = []
        self._mask_counts: dict[int, int] = {}
        self._speed_ema: float = 0.0

    @property
    def active(self) -> bool:
        return self._started_at is not None

    @property
    def elapsed(self) -> float:
        if self._started_at is None:
            return 0.0
        return time.monotonic() - self._started_at

    @property
    def remaining(self) -> float:
        return max(0.0, self.duration_s - self.elapsed)

    @property
    def done(self) -> bool:
        return self.active and self.elapsed >= self.duration_s

    def start(self) -> None:
        self._started_at = time.monotonic()
        self._last = None
        self._last_pos = None
        self._samples = []
        self._mask_counts = {}
        self._speed_ema = 0.0

    def cancel(self) -> None:
        self._started_at = None
        self._samples = []

    def update(self, hand: Optional[Hand]) -> None:
        """Record one frame's hand state (no-op if not recording)."""
        if not self.active or hand is None:
            return

        pos = hand.landmarks[MIDDLE_MCP, :2].copy()
        t = self.elapsed

        # Speed from palm-center displacement, normalized by palm size.
        if self._last_pos is not None and self._last is not None:
            prev_size = palm_size(self._last.landmarks)
            if prev_size > 1e-6:
                dx = np.linalg.norm(pos - self._last_pos)
                vel = dx / prev_size * SPEED_PALM_SIZE_FACTOR
                # EMA smoothing of speed (reduces single-frame spikes).
                self._speed_ema = self._speed_ema * 0.8 + vel * 0.2

        self._last = hand
        self._last_pos = pos

        mask = finger_pattern(hand.landmarks)
        self._mask_counts[mask] = self._mask_counts.get(mask, 0) + 1
        size = palm_size(hand.landmarks)
        self._samples.append(
            MotionSample(
                x=float(pos[0]),
                y=float(pos[1]),
                mask=mask,
                speed=self._speed_ema,
                palm_size=float(size),
                t=t,
            )
        )

    def finish(self) -> Optional[MotionRecording]:
        """Summarize the captured samples into a recording; reset state."""
        self._started_at = None
        if len(self._samples) < 4:
            return None

        # Dominant direction from net displacement of the palm path.
        xs = np.array([s.x for s in self._samples])
        ys = np.array([s.y for s in self._samples])
        dx = xs[-1] - xs[0]
        dy = ys[-1] - ys[0]
        # Use total path length for direction too — a "down-up" wobble
        # cancels net displacement but has clear vertical motion.
        total_v = np.sum(np.abs(np.diff(ys)))
        total_h = np.sum(np.abs(np.diff(xs)))

        direction = ""
        if abs(dy) > abs(dx) and abs(dy) > MIN_VELOCITY_FOR_DIRECTION * len(self._samples):
            direction = "down" if dy > 0 else "up"
        elif abs(dx) > abs(dy) and abs(dx) > MIN_VELOCITY_FOR_DIRECTION * len(self._samples):
            direction = "right" if dx > 0 else "left"
        elif total_v > total_h and total_v > MIN_VELOCITY_FOR_DIRECTION * len(self._samples) * 2:
            direction = "down" if np.mean(np.sign(np.diff(ys))) > 0 else "up"

        # Most common finger pattern.
        mask = max(self._mask_counts, key=self._mask_counts.get)

        # Mean speed.
        speed = float(np.mean([s.speed for s in self._samples[1:]]))

        rec = MotionRecording(
            samples=self._samples,
            direction=direction,
            finger_mask=mask,
            speed=speed,
            valid=bool(direction) and speed > 0,
        )
        self._samples = []
        self._mask_counts = {}
        return rec

    def finish_variations(self) -> List[MotionRecording]:
        """Split the captured window into stroke variations.

        A 20s recording of "do scroll-down at different angles" produces
        several strokes. Each stroke (a contiguous run of motion with a
        consistent direction) becomes its own MotionRecording — so every
        angle/finger-style variation is stored as a separate template.
        """
        self._started_at = None
        if len(self._samples) < 4:
            self._samples = []
            self._mask_counts = {}
            return []

        strokes: List[List[MotionSample]] = []
        current: List[MotionSample] = []
        prev_dir = ""

        # Rebuild with explicit direction per sample using consecutive deltas.
        strokes = []
        current = []
        prev_x, prev_y = None, None
        for s in self._samples:
            if prev_x is None:
                prev_x, prev_y = s.x, s.y
                current.append(s)
                continue
            dx = s.x - prev_x
            dy = s.y - prev_y
            prev_x, prev_y = s.x, s.y
            # Stationary frame → the stroke ends (a pause separates strokes).
            if abs(dx) < 1e-4 and abs(dy) < 1e-4:
                if current:
                    strokes.append(current)
                    current = []
                continue
            d = ""
            if abs(dy) >= abs(dx):
                d = "down" if dy > 0 else "up"
            else:
                d = "right" if dx > 0 else "left"

            if current and d != prev_dir:
                # Direction changed → stroke boundary. End this stroke.
                strokes.append(current)
                current = []
            current.append(s)
            prev_dir = d

        if current:
            strokes.append(current)

        # Keep only strokes with enough samples + net displacement.
        variations: List[MotionRecording] = []
        for stroke in strokes:
            if len(stroke) < 3:
                continue
            xs = np.array([s.x for s in stroke])
            ys = np.array([s.y for s in stroke])
            dx, dy = xs[-1] - xs[0], ys[-1] - ys[0]
            if abs(dx) < 0.02 and abs(dy) < 0.02:
                continue  # too little net motion to be a real stroke
            direction = "down" if dy > 0 else "up"
            if abs(dx) > abs(dy):
                direction = "right" if dx > 0 else "left"
            mask_counts: dict[int, int] = {}
            for s in stroke:
                mask_counts[s.mask] = mask_counts.get(s.mask, 0) + 1
            mask = max(mask_counts, key=mask_counts.get)
            speed = float(np.mean([s.speed for s in stroke[1:]]))
            # Path embedding: resample + normalize the stroke, then DROP
            # the raw samples (only the compact embedding is kept).
            points = [np.array([s.x, s.y]) for s in stroke]
            sizes = [s.palm_size for s in stroke]
            emb = path_embedding(points, sizes)
            variations.append(
                MotionRecording(
                    samples=stroke,
                    direction=direction,
                    finger_mask=mask,
                    speed=speed,
                    valid=bool(direction) and speed > 0 and emb.size > 0,
                    embedding=emb,
                )
            )

        self._samples = []
        self._mask_counts = {}
        return variations
