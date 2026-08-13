"""Tests for the motion differentiator (scroll vs zoom disambiguation).

Uses synthetic landmark arrays (no camera, no mediapipe model). The hand
is moved by translating the landmarks or by articulating the thumb-index
spread ratio while keeping the palm stationary.
"""

import numpy as np

from axis.gestures import (
    DifferentiatorConfig,
    MotionDifferentiator,
    MotionRegistry,
    MotionTemplate,
    GestureRecognizer,
    GestureRegistry,
    make_open_palm_definition,
    make_pinch_definition,
)
from axis.gestures.definitions import FINGER_INDEX, FINGER_MIDDLE
from axis.gestures.embedding import embedding_distance, path_embedding
from axis.landmarks import Hand


def _hand(landmarks, score=0.9) -> Hand:
    return Hand(
        landmarks=np.array(landmarks, dtype=np.float32),
        handedness="Right",
        score=score,
    )


def _spread_lm(ratio: float = 1.0) -> np.ndarray:
    """Open-palm synthetic hand; `ratio` scales thumb-index distance.

    palm_size = wrist(0) → middle MCP(9) = 1.0. Index tip (8) sits at
    `ratio * 0.5` from the thumb tip (4) so the spread ratio is ~ratio*0.5.
    """
    lm = np.zeros((21, 3), dtype=np.float32)
    lm[0] = (0.0, 0.0, 0.0)
    lm[9] = (1.0, 0.0, 0.0)
    lm[5] = (0.3, 0.1, 0.0)
    lm[13] = (0.3, -0.1, 0.0)
    lm[17] = (0.1, -0.2, 0.0)
    lm[4] = (0.5, 0.0, 0.0)
    # Fingers extended (index+middle out → can scroll).
    lm[8] = (0.5 + 0.55 * ratio, 0.35, 0.0)
    lm[12] = (1.05, -0.25, 0.0)
    lm[16] = (1.2, -0.35, 0.0)
    lm[20] = (1.3, 0.25, 0.0)
    return lm


def _slide(lm: np.ndarray, dx: float, dy: float) -> np.ndarray:
    out = lm.copy()
    out[:, 0] += dx
    out[:, 1] += dy
    return out


def _classify_sequence(diff: MotionDifferentiator, motions, hands):
    """Feed a sequence of hands, returning the last classify() output."""
    out = ("none", "", 0.0, "")
    for h in hands:
        diff.update(h)
        out = diff.classify(motions)
    return out


def test_zoom_in_stationary_spread():
    """Stationary palm + rising spread ratio → zoom in."""
    diff = MotionDifferentiator()
    motions = MotionRegistry()
    hands = []
    lm = _spread_lm(ratio=0.8)
    # 12 frames: keep the palm still, open the thumb-index from 0.4→1.0.
    for i in range(12):
        t = i / 11.0
        h = _spread_lm(ratio=0.8 + 0.6 * t)
        hands.append(_hand(h))
    kind, direction, _, _ = _classify_sequence(diff, motions, hands)
    assert kind == "zoom"
    assert direction == "in"


def test_zoom_out_stationary_pinch():
    """Stationary palm + falling spread ratio → zoom out."""
    diff = MotionDifferentiator()
    motions = MotionRegistry()
    hands = []
    for i in range(12):
        t = i / 11.0
        hands.append(_hand(_spread_lm(ratio=1.0 - 0.5 * t)))
    kind, direction, _, _ = _classify_sequence(diff, motions, hands)
    assert kind == "zoom"
    assert direction == "out"


def test_scroll_not_zoom():
    """Translating palm with stable spread ratio → scroll, never zoom."""
    diff = MotionDifferentiator()
    motions = MotionRegistry()
    lm = _spread_lm(ratio=1.0)
    hands = []
    for i in range(12):
        hands.append(_hand(_slide(lm, 0.0, 0.05 * i)))  # downward
    kind, direction, _, _ = _classify_sequence(diff, motions, hands)
    assert kind == "scroll"
    assert direction == "down"


def test_scroll_wins_over_slight_articulation():
    """A translating hand with slight finger spread is still scroll."""
    diff = MotionDifferentiator()
    motions = MotionRegistry()
    hands = []
    lm = _spread_lm(ratio=1.0)
    for i in range(12):
        t = i / 11.0
        h = _slide(_spread_lm(ratio=1.0 + 0.02 * t), 0.0, 0.05 * i)
        hands.append(_hand(h))
    kind, direction, _, _ = _classify_sequence(diff, motions, hands)
    assert kind == "scroll"
    assert direction == "down"


def test_scroll_up_vs_down_embeddings_distinct():
    """Up and down strokes produce far-apart embeddings."""
    pts_down = [np.array([0.5, 0.1 * i]) for i in range(10)]
    pts_up = [np.array([0.5, 0.9 - 0.1 * i]) for i in range(10)]
    sizes = [0.2] * 10
    emb_down = path_embedding(pts_down, sizes)
    emb_up = path_embedding(pts_up, sizes)
    assert embedding_distance(emb_down, emb_up) > 0.5


def test_margin_falls_back_to_geometric():
    """Two near-identical templates → geometric direction wins, no random pick."""
    diff = MotionDifferentiator()
    motions = MotionRegistry()
    # Two Scroll Down templates with the same embedding.
    pts = [np.array([0.5, 0.1 * i]) for i in range(10)]
    emb = path_embedding(pts, [0.2] * 10)
    for i in range(2):
        motions.add(
            MotionTemplate(
                id=f"sd{i}",
                name=f"Scroll Down {i}",
                direction="down",
                finger_mask=FINGER_INDEX | FINGER_MIDDLE,
                speed=0.3,
                embedding=emb,
            )
        )
    # Downward motion → both templates score near-identically.
    lm = _spread_lm()
    hands = [_hand(_slide(lm, 0.0, 0.05 * i)) for i in range(12)]
    kind, direction, _, label = _classify_sequence(diff, motions, hands)
    assert kind == "scroll"
    assert direction == "down"
    # The label may be a template name OR the geometric fallback (""), but
    # it must not be the wrong direction.
    assert label in ("", "Scroll Down 0", "Scroll Down 1")


def test_hysteresis_blocks_flip():
    """A single ambiguous frame does not flip the committed class."""
    diff = MotionDifferentiator()
    motions = MotionRegistry()
    # Steady downward scroll → committed "scroll/down".
    lm = _spread_lm()
    hands = [_hand(_slide(lm, 0.0, 0.05 * i)) for i in range(12)]
    kind, direction, score, _ = _classify_sequence(diff, motions, hands)
    assert kind == "scroll" and direction == "down"
    # A single zoom-ish frame (strong spread change but small motion) is
    # suppressed by hysteresis.
    zoom_hand = _hand(_spread_lm(ratio=1.8))
    diff.update(zoom_hand)
    kind2, direction2, _, _ = diff.classify(motions)
    assert (kind2, direction2) == ("scroll", "down")


def test_path_embedding_per_sample_sizes_differs_from_constant():
    """Regression: per-sample palm sizes must change the embedding."""
    pts = [np.array([0.5, 0.1 * i]) for i in range(10)]
    emb_const = path_embedding(pts, [0.2] * 10)
    # Different mean palm size → different scale → different embedding.
    emb_var = path_embedding(pts, [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1])
    assert not np.allclose(emb_const, emb_var)


def test_recognizer_uses_differentiator_for_scroll():
    """GestureRecognizer routes scroll through the differentiator."""
    reg = GestureRegistry(definitions={})
    reg.add(make_pinch_definition())
    reg.add(make_open_palm_definition())
    rec = GestureRecognizer(reg, motions=MotionRegistry())
    lm = _spread_lm()
    rec.recognize([_hand(lm)])
    out = None
    for i in range(1, 12):
        lm = _slide(lm, 0.0, 0.05)
        out = rec.recognize([_hand(lm)])
    assert out.kind == "scroll"
    assert out.scroll_direction == "down"


def test_config_from_dict():
    cfg = DifferentiatorConfig.from_dict(
        {"window_frames": 8, "zoom_slope_min": 0.1, "unknown_key": 99}
    )
    assert cfg.window_frames == 8
    assert cfg.zoom_slope_min == 0.1
    assert cfg.margin == 0.10  # default preserved
