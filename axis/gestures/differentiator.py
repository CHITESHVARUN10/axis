"""Motion differentiator — decide *which* dynamic gesture a hand is doing.

The core idea (from the plan): split dynamic gestures by **palm translation
vs. finger articulation** *first*, then disambiguate within a branch.

  - Zoom in/out   = the palm stays put while the thumb-index spread ratio
                    changes (spread slope above threshold, displacement
                    below threshold). Checked first so an articulating
                    stationary hand is never treated as static.
  - Scroll/swipe  = the palm translates across the camera plane with a
                    scroll-compatible hand shape (index+middle extended).
  - Static poses  = no meaningful path length at all (pinch/fist/open palm).

Learned motion embeddings only *refine* a branch — they never decide between
scroll and zoom. That kills the scroll-vs-zoom confusion: a translating palm
is never zoom, an articulating stationary palm is never scroll.

All quantities are scale-invariant (normalized by palm size).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np

from axis.gestures.definitions import (
    FINGER_INDEX,
    FINGER_MIDDLE,
    FINGER_PINKY,
    MotionRegistry,
    MotionTemplate,
    finger_pattern,
    is_scroll_shape,
    palm_size,
)
from axis.gestures.embedding import embedding_distance, path_embedding
from axis.landmarks import Hand

# Palm-center landmarks: wrist + the four MCPs (stable under rotation).
PALM_CENTER_IDX = [0, 5, 9, 13, 17]
# Thumb tip / index tip for the spread ratio.
THUMB_TIP = 4
INDEX_TIP = 8

# Scroll requires the two-finger pose: index + middle extended, pinky
# folded (thumb and ring ignored — both read noisy on real hands).
# A full open palm (pinky OUT) NEVER scrolls — it moves the cursor.
SCROLL_MIN_FINGERS = FINGER_INDEX | FINGER_MIDDLE


@dataclass
class DifferentiatorConfig:
    """Thresholds for the differentiator. All in palm-size-normalized units."""

    window_frames: int = 8
    path_min: float = 0.10        # below this: static branch
    scroll_disp_min: float = 0.12  # net displacement to declare scroll
    zoom_slope_min: float = 0.02  # normalized spread-ratio change / frame
    margin: float = 0.10           # best vs second-best learned score gap
    min_score: float = 0.35        # absolute score floor for learned classes
    hysteresis: float = 0.05       # score must beat the committed class by this
    embedding_tolerance: float = 0.5  # max embedding distance to accept a template

    @classmethod
    def from_dict(cls, d: dict) -> "DifferentiatorConfig":
        known = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in d.items() if k in known})


@dataclass
class MotionWindow:
    """Sliding window of per-frame features for one hand."""

    centers: List[np.ndarray] = field(default_factory=list)   # palm center xy
    ratios: List[float] = field(default_factory=list)         # spread ratio
    masks: List[int] = field(default_factory=list)            # finger pattern
    sizes: List[float] = field(default_factory=list)          # palm size


def _spread_slope(ratios: List[float]) -> float:
    """Least-squares slope of the spread ratio over time, normalized by mean.

    Closed-form slope (no np.polyfit — fewer FLOPs/allocations on the
    per-frame call). Returns 0.0 for too few samples or a degenerate mean.
    Positive = fingers spreading (zoom in), negative = pinching (zoom out).
    """
    n = len(ratios)
    if n < 3:
        return 0.0
    mean_r = sum(ratios) / n
    if mean_r <= 1e-6:
        return 0.0
    mean_t = (n - 1) / 2.0  # t = 0..n-1
    num = den = 0.0
    for t, r in enumerate(ratios):
        dt = t - mean_t
        num += dt * (r - mean_r)
        den += dt * dt
    if den <= 1e-12:
        return 0.0
    return float((num / den) / mean_r)


class MotionDifferentiator:
    """Classify the primary hand's motion into a dynamic gesture branch."""

    def __init__(self, config: Optional[DifferentiatorConfig] = None):
        self.config = config or DifferentiatorConfig()
        self._window = MotionWindow()
        self._last_committed: Tuple[str, str, float] = ("none", "", 0.0)

    # --- window upkeep --------------------------------------------------

    def update(self, hand: Optional[Hand]) -> None:
        """Append one frame of features for the primary hand (or None)."""
        if hand is None:
            self.reset()
            return
        lm = hand.landmarks
        size = palm_size(lm)
        if size <= 1e-6:
            self.reset()
            return
        center = np.mean(lm[PALM_CENTER_IDX, :2], axis=0)
        ratio = float(np.linalg.norm(lm[INDEX_TIP, :2] - lm[THUMB_TIP, :2]) / size)
        w = self._window
        w.centers.append(center)
        w.ratios.append(ratio)
        w.masks.append(finger_pattern(lm))
        w.sizes.append(size)
        n = self.config.window_frames
        if len(w.centers) > n:
            del w.centers[0], w.ratios[0], w.masks[0], w.sizes[0]

    def reset(self) -> None:
        self._window = MotionWindow()
        self._last_committed = ("none", "", 0.0)

    # --- features -------------------------------------------------------

    def features(self) -> dict:
        """Compute the window's aggregate features (scale-invariant)."""
        w = self._window
        empty = {"D": np.zeros(2), "L": 0.0, "speed": 0.0,
                 "spread_slope": 0.0, "direction": "", "sizes": [], "centers": []}
        if len(w.centers) < 2:
            return empty
        c = np.asarray(w.centers)
        D = c[-1] - c[0]
        L = float(np.sum(np.linalg.norm(np.diff(c, axis=0), axis=1)))
        speed = L / (len(w.centers) - 1)
        # Normalize displacement/path by mean palm size over the window.
        mean_size = float(np.mean([s for s in w.sizes if s > 1e-6])) or 1.0
        Dn = D / mean_size
        Ln = L / mean_size
        direction = ""
        if np.hypot(*Dn) >= self.config.scroll_disp_min:
            if abs(Dn[1]) >= abs(Dn[0]):
                direction = "down" if Dn[1] > 0 else "up"
            else:
                direction = "right" if Dn[0] > 0 else "left"
        return {
            "D": Dn,
            "L": Ln,
            "speed": speed / mean_size if mean_size else 0.0,
            "spread_slope": _spread_slope(w.ratios),
            "direction": direction,
            "sizes": list(w.sizes),
            "centers": list(w.centers),
            "mask": w.masks[-1] if w.masks else 0,
        }

    # --- classification -------------------------------------------------

    def classify(
        self, motions: MotionRegistry
    ) -> Tuple[str, str, float, str]:
        """Return (kind, direction, confidence, label) for the current window.

        kind: "none" | "zoom" | "scroll" | "static". Label is the learned
        template's name when one refined the result, else "".
        """
        w = self._window
        if len(w.centers) < 2:
            return "none", "", 0.0, ""
        f = self.features()

        # 1) Zoom branch FIRST: fingers articulating while the palm stays
        #    put. This must run before the static branch — a zoom keeps the
        #    palm stationary, so path length is tiny and would otherwise be
        #    swallowed by "static".
        if (
            abs(f["spread_slope"]) >= self.config.zoom_slope_min
            and np.hypot(*f["D"]) < self.config.scroll_disp_min
        ):
            direction = "in" if f["spread_slope"] > 0 else "out"
            score = min(1.0, 0.5 + abs(f["spread_slope"]) * 5.0)
            kind, direction, score = self._hysteresis("zoom", direction, score)
            return kind, direction, score, ""

        # 2) Static branch: no meaningful path length at all.
        if f["L"] < self.config.path_min:
            return self._hysteresis("static", "", 0.0) + ("",)

        # 3) Scroll branch: two-finger palm translating. The shape gate
        #    (index + middle out, pinky folded) keeps a moving open palm
        #    on the cursor — it NEVER scrolls.
        w = self._window
        scroll_shape = False
        if w.masks:
            scroll_shape = (w.masks[-1] & SCROLL_MIN_FINGERS) == SCROLL_MIN_FINGERS and not (
                w.masks[-1] & FINGER_PINKY
            )
        if scroll_shape and np.hypot(*f["D"]) >= self.config.scroll_disp_min:
            kind, direction, score, label = self._scroll_classify(f, motions)
            kind, direction, score = self._hysteresis(kind, direction, score)
            return kind, direction, score, label

        # 4) Ambiguous transition (moving but below latch, or wrong shape).
        return self._hysteresis("none", "", 0.0) + ("",)

    # --- internals ------------------------------------------------------

    def _scroll_classify(
        self, f: dict, motions: MotionRegistry
    ) -> Tuple[str, str, float, str]:
        """Refine a translating palm with learned templates, else geometric.

        Direction is authoritative: only templates whose direction matches
        the geometric direction are eligible. The learned label + embedding
        shape then confirm the class; if the best template doesn't win by a
        clear margin, fall back to the geometric answer.
        """
        geo_dir = f["direction"]
        geo_score = min(1.0, 0.5 + f["speed"] * 2.0)

        emb = None
        if len(f["centers"]) >= 2:
            emb = path_embedding(f["centers"], f["sizes"])
        mask = f["mask"]

        candidates: List[Tuple[MotionTemplate, float]] = []
        for t in motions.templates.values():
            if t.direction not in ("up", "down", "left", "right"):
                continue
            if t.direction != geo_dir:
                continue  # geometric direction is authoritative
            emb_score = 0.0
            if emb is not None and emb.size and t.embedding is not None and t.embedding.size:
                d = embedding_distance(emb, t.embedding)
                emb_score = 1.0 - min(d, 1.0)
            else:
                emb_score = 0.5  # neutral when no embedding info
            finger_score = 0.0
            if t.finger_mask:
                finger_score = 1.0 if (mask & t.finger_mask) == t.finger_mask else 0.0
            else:
                finger_score = 0.5
            speed_score = 0.0
            tol = max(t.speed * t.speed_tolerance, 0.02)
            dev = abs(f["speed"] - t.speed)
            if dev <= tol:
                speed_score = 1.0
            elif dev <= tol * 2:
                speed_score = 0.5
            score = (
                0.4 * emb_score
                + 0.3 * 1.0  # direction matches (pre-filtered)
                + 0.15 * finger_score
                + 0.15 * speed_score
            )
            candidates.append((t, score))

        if candidates:
            candidates.sort(key=lambda c: c[1], reverse=True)
            best_t, best_score = candidates[0]
            second = candidates[1][1] if len(candidates) > 1 else 0.0
            if (
                best_score >= self.config.min_score
                and (best_score - second) >= self.config.margin
            ):
                return "scroll", best_t.direction, best_score, best_t.name
        return "scroll", geo_dir, geo_score, ""

    def _hysteresis(
        self, kind: str, direction: str, score: float
    ) -> Tuple[str, str, float]:
        """Prevent scroll↔zoom flapping and scroll direction flips.

        Dynamic-class flips (scroll↔zoom) need the new class to beat the
        committed one by `hysteresis`. A committed scroll DIRECTION is
        additionally locked: mid-stroke wobble reporting the opposite axis
        (e.g. up during a down stroke) holds the latched direction until
        the motion stops (via "none"/"static"). Entering/leaving a dynamic
        gesture is never blocked, so a held gesture still clears as soon
        as the motion stops.
        """
        last_kind, last_dir, last_score = self._last_committed
        if (
            kind in ("scroll", "zoom")
            and last_kind in ("scroll", "zoom")
            and kind != last_kind
            and score < last_score + self.config.hysteresis
        ):
            return last_kind, last_dir, last_score
        if (
            kind == "scroll"
            and last_kind == "scroll"
            and direction
            and last_dir
            and direction != last_dir
        ):
            return last_kind, last_dir, last_score
        self._last_committed = (kind, direction, score)
        return kind, direction, score


__all__ = [
    "DifferentiatorConfig",
    "MotionDifferentiator",
    "MotionWindow",
    "_spread_slope",
]
