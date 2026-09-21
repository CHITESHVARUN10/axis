"""Gesture definitions and registration for Axis.

Gestures are defined by *distance ratios* between landmark pairs. Ratios
are scale-invariant (hand size and camera distance don't matter) and are
compared with a tolerance. Every registered gesture stores:

    {
      "id": "pinch",
      "name": "Pinch",
      "signature": {
        "pairs": [[8, 4], [12, 4]],      # landmark index pairs
        "ratios": [0.25, 0.40],          # expected distance / palm size
        "tolerance": 0.10
      }
    }

The palm size is the distance between the wrist (0) and middle-finger MCP
(9). Ratios for a held hand pose are stable across people and cameras.

Recognizing a pose: compute the same ratios for a live hand; if every
ratio is within tolerance of the stored signature, the gesture matches.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from axis.landmarks import INDEX_MCP, MIDDLE_MCP, WRIST

# Landmark index pairs that define a gesture's shape.
Pair = Tuple[int, int]

# Canonical landmark pairs used by the built-in gestures.
PINCH_PAIRS: List[Pair] = [
    (8, 4),   # index tip to thumb tip
    (12, 4),  # middle tip to thumb tip
]
OPEN_PALM_PAIRS: List[Pair] = [
    (8, 4),   # index tip to thumb tip (spread = large)
    (12, 4),  # middle tip to thumb tip
    (16, 4),  # ring tip to thumb tip
    (20, 4),  # pinky tip to thumb tip
]
SCROLL_PAIRS: List[Pair] = [
    (8, 4),   # index tip to thumb tip
    (12, 4),  # middle tip to thumb tip
    (16, 4),  # ring tip to thumb tip
]

DEFAULT_TOLERANCE = 0.10


@dataclass
class GestureDefinition:
    """A named, recognizable hand shape."""

    id: str
    name: str
    pairs: List[Pair]
    ratios: List[float]
    tolerance: float = DEFAULT_TOLERANCE

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "name": self.name,
            "signature": {
                "pairs": [list(p) for p in self.pairs],
                "ratios": [float(r) for r in self.ratios],
                "tolerance": self.tolerance,
            },
        }

    @classmethod
    def from_dict(cls, d: dict) -> "GestureDefinition":
        sig = d["signature"]
        return cls(
            id=d["id"],
            name=d["name"],
            pairs=[tuple(p) for p in sig["pairs"]],
            ratios=[float(r) for r in sig["ratios"]],
            tolerance=float(sig.get("tolerance", DEFAULT_TOLERANCE)),
        )


def palm_size(landmarks: np.ndarray) -> float:
    """Scale-invariant size unit: wrist (0) to middle MCP (9), in pixels."""
    return float(np.linalg.norm(landmarks[WRIST][:2] - landmarks[MIDDLE_MCP][:2]))


def pair_distances(landmarks: np.ndarray, pairs: List[Pair]) -> np.ndarray:
    """Euclidean (x,y) distance for each landmark pair, in pixels."""
    pts = landmarks[:, :2]
    return np.linalg.norm(pts[[p[0] for p in pairs]] - pts[[p[1] for p in pairs]], axis=1)


def normalized_signature(landmarks: np.ndarray, pairs: List[Pair]) -> List[float]:
    """Ratio of each pair distance to palm size. Empty if palm not visible."""
    size = palm_size(landmarks)
    if size <= 1e-6:
        return []
    return [float(d / size) for d in pair_distances(landmarks, pairs)]


def matches(landmarks: np.ndarray, gesture: GestureDefinition) -> bool:
    """True if the live hand shape matches the gesture's stored signature."""
    live = normalized_signature(landmarks, gesture.pairs)
    if not live:
        return False
    return all(
        abs(live[i] - gesture.ratios[i]) <= gesture.tolerance
        for i in range(len(gesture.ratios))
    )


def distance_to(landmarks: np.ndarray, gesture: GestureDefinition) -> float:
    """Mean ratio deviation from a gesture — lower is closer. Inf if invalid."""
    live = normalized_signature(landmarks, gesture.pairs)
    if not live:
        return float("inf")
    return float(np.mean([abs(live[i] - gesture.ratios[i]) for i in range(len(gesture.ratios))]))


# --- registry -----------------------------------------------------------

@dataclass
class GestureRegistry:
    """Loaded gesture definitions + runtime additions."""

    definitions: Dict[str, GestureDefinition] = field(default_factory=dict)

    @classmethod
    def load(cls, config_path: Path) -> "GestureRegistry":
        with open(config_path) as f:
            data = json.load(f)
        defs = {}
        for g in data.get("gestures", []):
            d = GestureDefinition.from_dict(g)
            defs[d.id] = d
        return cls(definitions=defs)

    def add(self, gesture: GestureDefinition) -> None:
        self.definitions[gesture.id] = gesture

    def save(self, config_path: Path) -> None:
        payload = {
            "version": 1,
            "schema": "axis.gestures.v1",
            "gestures": [g.to_dict() for g in self.definitions.values()],
        }
        config_path.parent.mkdir(parents=True, exist_ok=True)
        with open(config_path, "w") as f:
            json.dump(payload, f, indent=2)
            f.write("\n")

    def closest(self, landmarks: np.ndarray, max_deviation: float) -> Optional[GestureDefinition]:
        """Best-matching gesture within max_deviation ratio error, or None."""
        best, best_dist = None, float("inf")
        for g in self.definitions.values():
            d = distance_to(landmarks, g)
            if d < best_dist:
                best, best_dist = g, d
        if best is None or best_dist > max_deviation:
            return None
        return best


# --- motion templates ----------------------------------------------------

# Finger-pattern bitmask: which fingertips are extended vs. folded.
FINGER_THUMB = 1
FINGER_INDEX = 2
FINGER_MIDDLE = 4
FINGER_RING = 8
FINGER_PINKY = 16

# Move pose = full open palm (all four fingers + thumb out). Scroll pose =
# index + middle extended with pinky folded and thumb tucked (a "two-finger"
# hand). Splitting by shape means translating an open palm moves the cursor
# and NEVER scrolls — scroll needs the distinct two-finger pose.
MOVE_FINGERS = FINGER_THUMB | FINGER_INDEX | FINGER_MIDDLE | FINGER_RING | FINGER_PINKY
SCROLL_FINGERS = FINGER_INDEX | FINGER_MIDDLE


def finger_pattern(landmarks: np.ndarray) -> int:
    """Classify which fingers are extended, as a bitmask.

    A finger is "extended" if its tip is further from the palm center
    than its MCP (roughly straight vs. curled). Scale-invariant.
    """
    size = palm_size(landmarks)
    if size <= 1e-6:
        return 0
    palm = np.mean(landmarks[[0, 5, 9, 13, 17], :2], axis=0)
    tips = [4, 8, 12, 16, 20]      # thumb, index, middle, ring, pinky
    mcps = [2, 5, 9, 13, 17]       # thumb IP, index/middle/ring/pinky MCP
    bits = [FINGER_THUMB, FINGER_INDEX, FINGER_MIDDLE, FINGER_RING, FINGER_PINKY]
    pattern = 0
    for tip_i, mcp_i, bit in zip(tips, mcps, bits):
        tip_d = np.linalg.norm(landmarks[tip_i, :2] - palm)
        mcp_d = np.linalg.norm(landmarks[mcp_i, :2] - palm)
        # Thumb uses a lower threshold (shorter, angles differently).
        threshold = 0.75 if bit == FINGER_THUMB else 1.0
        if tip_d > mcp_d * threshold:
            pattern |= bit
    return pattern


def finger_pattern_name(pattern: int) -> str:
    names = []
    if pattern & FINGER_THUMB:
        names.append("thumb")
    if pattern & FINGER_INDEX:
        names.append("index")
    if pattern & FINGER_MIDDLE:
        names.append("middle")
    if pattern & FINGER_RING:
        names.append("ring")
    if pattern & FINGER_PINKY:
        names.append("pinky")
    return "+".join(names) if names else "fist"


def is_move_shape(landmarks: np.ndarray) -> bool:
    """True for the MOVE pose: full open palm, all fingers extended.

    Strict: ring + pinky must be out too, so a two-finger scroll hand
    never counts as move (and vice versa).
    """
    pattern = finger_pattern(landmarks)
    return (pattern & MOVE_FINGERS) == MOVE_FINGERS


def is_scroll_shape(landmarks: np.ndarray) -> bool:
    """True for the SCROLL pose: index + middle out, pinky folded.

    The two-finger hand (like a trackpad two-finger scroll): index and
    middle extended, pinky folded. Ring is ignored (its MCP sits so close
    to the palm center that the reading is noisy) and the thumb is ignored
    too (its 0.75 threshold flips on real hands and vetoed most scrolls).
    A full open palm (pinky OUT) is NOT a scroll shape — it moves the
    cursor instead.
    """
    pattern = finger_pattern(landmarks)
    if (pattern & SCROLL_FINGERS) != SCROLL_FINGERS:
        return False
    if pattern & FINGER_PINKY:
        return False
    return True


@dataclass
class MotionTemplate:
    """A learned motion gesture (e.g. scroll down, scroll up).

    Captures the essential, scale-invariant signature of a recorded
    motion: the dominant direction, the finger pattern used, the average
    speed, and a *path embedding* (fixed-length resampled shape). The raw
    recording is discarded after embedding — only this compact signature
    is persisted.
    """

    id: str
    name: str            # e.g. "Scroll Down", "Scroll Up"
    direction: str       # "up" | "down" | "left" | "right" | ""
    finger_mask: int     # finger pattern bitmask
    speed: float         # mean |velocity| in palm-sizes/frame
    speed_tolerance: float = 0.5
    finger_allow_extra: bool = True  # extra extended fingers still match
    embedding: np.ndarray = None     # fixed-length path embedding (2*(N-1),)
    embedding_tolerance: float = 0.5  # max embedding distance to accept a match

    def to_dict(self) -> dict:
        emb = None
        if self.embedding is not None and self.embedding.size:
            emb = [float(x) for x in self.embedding]
        return {
            "id": self.id,
            "name": self.name,
            "type": "motion",
            "direction": self.direction,
            "finger_mask": self.finger_mask,
            "speed": float(self.speed),
            "speed_tolerance": self.speed_tolerance,
            "finger_allow_extra": self.finger_allow_extra,
            "embedding": emb,
            "embedding_tolerance": self.embedding_tolerance,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "MotionTemplate":
        emb = d.get("embedding")
        if emb:
            emb = np.array(emb, dtype=np.float64)
        else:
            emb = None
        return cls(
            id=d["id"],
            name=d["name"],
            direction=d.get("direction", ""),
            finger_mask=int(d.get("finger_mask", 0)),
            speed=float(d.get("speed", 0.0)),
            speed_tolerance=float(d.get("speed_tolerance", 0.5)),
            finger_allow_extra=bool(d.get("finger_allow_extra", True)),
            embedding=emb,
            embedding_tolerance=float(d.get("embedding_tolerance", 0.5)),
        )

    def matches(self, direction: str, finger_mask: int, speed: float, embedding: np.ndarray = None) -> bool:
        """True if a live motion fits this template."""
        if direction != self.direction:
            return False
        if self.finger_mask == 0:
            pass  # no finger constraint
        elif self.finger_allow_extra:
            # Template fingers must all be extended; extra is fine.
            if (finger_mask & self.finger_mask) != self.finger_mask:
                return False
        else:
            if finger_mask != self.finger_mask:
                return False
        if abs(speed - self.speed) > self.speed * self.speed_tolerance + 0.02:
            return False
        # Embedding check (soft tie-breaker): when the live path embedding
        # is available and the template has one, require reasonable shape
        # similarity. Skip if either side lacks an embedding.
        if embedding is not None and self.embedding is not None and self.embedding.size:
            from axis.gestures.embedding import embedding_distance

            if embedding_distance(embedding, self.embedding) > self.embedding_tolerance:
                return False
        return True


class MotionRegistry:
    """Loaded + runtime motion templates."""

    def __init__(self, templates: Optional[Dict[str, MotionTemplate]] = None):
        self.templates: Dict[str, MotionTemplate] = templates or {}

    @classmethod
    def load(cls, path: Path) -> "MotionRegistry":
        if not path.exists():
            return cls()
        with open(path) as f:
            data = json.load(f)
        templates = {}
        for t in data.get("motions", []):
            mt = MotionTemplate.from_dict(t)
            templates[mt.id] = mt
        return cls(templates=templates)

    def add(self, template: MotionTemplate) -> None:
        self.templates[template.id] = template

    def save(self, path: Path) -> None:
        payload = {
            "version": 1,
            "schema": "axis.motions.v1",
            "motions": [t.to_dict() for t in self.templates.values()],
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(payload, f, indent=2)
            f.write("\n")

    def rank(self, direction: str, finger_mask: int, speed: float, embedding: np.ndarray = None) -> List[Tuple[MotionTemplate, float]]:
        """All templates that pass the scalar gates, sorted best-first.

        Each entry is (template, score) where score is a similarity in
        [0, 1] (1 = best): 0.6 base for passing the gates, plus embedding
        agreement (when both sides have an embedding), minus a speed-
        deviation penalty.
        """
        from axis.gestures.embedding import embedding_distance

        scored: List[Tuple[MotionTemplate, float]] = []
        for t in self.templates.values():
            if not t.matches(direction, finger_mask, speed, embedding):
                continue
            score = 0.6
            if (
                embedding is not None
                and embedding.size
                and t.embedding is not None
                and t.embedding.size
            ):
                d = embedding_distance(embedding, t.embedding)
                if d > t.embedding_tolerance:
                    continue  # matches() already gates; keep double safety
                score += 0.4 * (1.0 - min(d, 1.0))
            if t.speed > 0:
                tol = max(t.speed * t.speed_tolerance, 0.02)
                dev = abs(speed - t.speed) / tol
                score -= 0.2 * min(dev, 1.0)
            scored.append((t, max(0.0, min(1.0, score))))
        scored.sort(key=lambda c: c[1], reverse=True)
        return scored

    def find(self, direction: str, finger_mask: int, speed: float, embedding: np.ndarray = None) -> Optional[MotionTemplate]:
        """Best-matching template for a live motion, or None.

        Among templates that pass the scalar gates (direction/fingers/
        speed), the one with the closest embedding (when both have one)
        wins. Falls back to insertion order if no embedding info.
        """
        ranked = self.rank(direction, finger_mask, speed, embedding)
        return ranked[0][0] if ranked else None
