"""Hand landmark extraction via MediaPipe HandLandmarker.

Wraps the MediaPipe Tasks API into a minimal, replaceable interface:

    extractor = HandLandmarker()          # loads model, connects pipeline
    result = extractor.process(frame)     # numpy BGR frame -> HandData | None
    extractor.close()

HandData carries 21 normalized landmarks (x, y in [0,1], z relative to
wrist) plus handedness — exactly what the gesture engine consumes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import mediapipe as mp
import numpy as np
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

from axis.landmarks.models import ensure_hand_landmarker

# MediaPipe palm landmark indices
WRIST = 0
THUMB_TIP = 4
INDEX_TIP = 8
MIDDLE_TIP = 12
RING_TIP = 16
PINKY_TIP = 20

INDEX_MCP = 5
MIDDLE_MCP = 9


@dataclass
class Hand:
    """Normalized landmarks for one detected hand."""

    landmarks: np.ndarray  # shape (21, 3) — x, y (normalized), z (relative to wrist)
    handedness: str        # "Left" | "Right"
    score: float
    frame_h: int = 0       # frame height in pixels (for size gating)
    frame_w: int = 0       # frame width in pixels


@dataclass
class HandData:
    """All hands detected in one frame."""

    hands: List[Hand]
    frame_w: int
    frame_h: int

    @property
    def primary(self) -> Optional[Hand]:
        """Highest-confidence hand (best proxy for "the user's hand")."""
        if not self.hands:
            return None
        return max(self.hands, key=lambda h: h.score)


class HandLandmarker:
    def __init__(
        self,
        model_path: Optional[str] = None,
        num_hands: int = 2,
        smoothing: float = 0.5,
    ):
        """Hand landmark extractor with per-hand EMA smoothing.

        `smoothing` is the EMA alpha: higher = more smoothing (jitter
        reduction, more lag). 0.5 is a good balance for 30fps video.
        """
        model_path = model_path or str(ensure_hand_landmarker())
        # Lower confidence thresholds: hands are re-detected more readily
        # when tracking is lost (e.g. hand leaves and returns, different
        # hand sizes, weaker lighting). The cost is marginally more palm
        # detections per frame, which is worth the robustness.
        self._options = vision.HandLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=model_path),
            running_mode=vision.RunningMode.VIDEO,
            num_hands=num_hands,
            min_hand_detection_confidence=0.3,
            min_hand_presence_confidence=0.3,
            min_tracking_confidence=0.3,
        )
        self._landmarker = vision.HandLandmarker.create_from_options(
            self._options
        )
        self._timestamp_ms = 0
        self._smoothing = smoothing
        self._smoothed: dict[int, np.ndarray] = {}  # hand index -> EMA landmarks

    def _smooth(self, idx: int, arr: np.ndarray) -> np.ndarray:
        """EMA-smooth a hand's landmarks to kill frame-to-frame jitter."""
        if self._smoothing <= 0:
            return arr
        if idx in self._smoothed:
            prev = self._smoothed[idx]
            arr = prev * self._smoothing + arr * (1 - self._smoothing)
        self._smoothed[idx] = arr.copy()
        return arr

    def process(self, frame: np.ndarray) -> Optional[HandData]:
        """Run detection on one BGR frame. Returns HandData or None."""
        if frame is None or frame.size == 0:
            return None
        h, w = frame.shape[:2]
        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB, data=frame
        )
        self._timestamp_ms += 33
        result = self._landmarker.detect_for_video(
            mp_image, self._timestamp_ms
        )
        hands: List[Hand] = []
        for idx, lm in enumerate(result.hand_landmarks):
            arr = np.array(
                [[p.x, p.y, p.z] for p in lm], dtype=np.float32
            )  # (21,3) normalized
            arr = self._smooth(idx, arr)
            handedness = "Left"  # MediaPipe's "left" = user's left when mirrored
            score = 0.0
            if idx < len(result.handedness) and result.handedness[idx]:
                hd = result.handedness[idx][0]
                handedness = hd.category_name
                score = float(hd.score)
            hands.append(Hand(landmarks=arr, handedness=handedness, score=score, frame_h=h, frame_w=w))
        if not hands:
            return None
        return HandData(hands=hands, frame_w=w, frame_h=h)

    def close(self) -> None:
        self._landmarker.close()


__all__ = [
    "Hand",
    "HandData",
    "HandLandmarker",
    "WRIST",
    "THUMB_TIP",
    "INDEX_TIP",
    "MIDDLE_TIP",
    "RING_TIP",
    "PINKY_TIP",
    "INDEX_MCP",
    "MIDDLE_MCP",
]
