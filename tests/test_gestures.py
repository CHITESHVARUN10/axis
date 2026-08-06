"""Smoke tests for the Phase 1 gesture engine.

Uses synthetic landmark arrays (no camera, no mediapipe model) to validate
the recognizer's classification math. `recognize()` takes a list of hands
(0, 1, or 2) — matching the two-hand tracking.
"""

import numpy as np

from axis.gestures import (
    GestureRecognizer,
    GestureRegistry,
    MotionRecorder,
    MotionRegistry,
    MotionTemplate,
    can_scroll,
    is_fist_shape,
    is_pinch_shape,
    is_spread,
    make_open_palm_definition,
    make_pinch_definition,
)
from axis.gestures.definitions import FINGER_INDEX, FINGER_MIDDLE
from axis.landmarks import Hand


def _hand(landmarks, score=0.9, frame_h=480) -> Hand:
    return Hand(
        landmarks=np.array(landmarks, dtype=np.float32),
        handedness="Right",
        score=score,
        frame_h=frame_h,
    )


def _synthetic(pinch: bool = False, fist: bool = False, spread: bool = True):
    """Build a 21-landmark array. palm_size = distance wrist(0) → middle MCP(9) = 1.0."""
    lm = np.zeros((21, 3), dtype=np.float32)
    lm[0] = (0.0, 0.0, 0.0)
    lm[9] = (1.0, 0.0, 0.0)
    lm[5] = (0.3, 0.1, 0.0)
    lm[13] = (0.3, -0.1, 0.0)
    lm[17] = (0.1, -0.2, 0.0)
    lm[4] = (0.5, 0.0, 0.0)
    if pinch:
        lm[8] = (0.62, 0.0, 0.0)   # index tip near thumb tip (pinched)
        lm[12] = (0.64, 0.0, 0.0)  # middle tip near thumb tip
        lm[16] = (1.0, 0.0, 0.0)   # ring extended
        lm[20] = (1.2, 0.0, 0.0)   # pinky extended
    elif fist:
        lm[8] = (0.34, -0.04, 0.0)
        lm[12] = (0.34, -0.04, 0.0)
        lm[16] = (0.34, -0.04, 0.0)
        lm[20] = (0.34, -0.04, 0.0)
    elif spread:
        lm[8] = (1.0, 0.35, 0.0)    # index extended
        lm[12] = (1.05, -0.25, 0.0) # middle extended
        lm[16] = (1.2, -0.35, 0.0)  # ring extended
        lm[20] = (1.3, 0.25, 0.0)   # pinky extended
    return lm


def test_pinch_shape():
    assert is_pinch_shape(_synthetic(pinch=True))
    assert not is_pinch_shape(_synthetic(spread=True))
    assert not is_pinch_shape(_synthetic(fist=True))


def test_fist_shape():
    assert is_fist_shape(_synthetic(fist=True))
    assert not is_fist_shape(_synthetic(spread=True))
    assert not is_fist_shape(_synthetic(pinch=True))


def test_spread_shape():
    assert is_spread(_synthetic(spread=True))
    assert not is_spread(_synthetic(pinch=True))
    assert not is_spread(_synthetic(fist=True))


def test_can_scroll():
    assert can_scroll(_synthetic(spread=True))     # open palm scrolls
    assert not can_scroll(_synthetic(fist=True))   # fist can't scroll


def _recognizer(motions=None) -> GestureRecognizer:
    reg = GestureRegistry(definitions={})
    reg.add(make_pinch_definition())
    reg.add(make_open_palm_definition())
    return GestureRecognizer(reg, motions=motions)


def _hold(rec: GestureRecognizer, hand: Hand, frames: int = 4):
    out = None
    for _ in range(frames):
        out = rec.recognize([hand])
    return out


def test_recognize_pinch():
    r = _hold(_recognizer(), _hand(_synthetic(pinch=True)))
    assert r.kind == "pinch"
    assert r.label == "You are trying to click"


def test_recognize_open_palm():
    r = _hold(_recognizer(), _hand(_synthetic(spread=True)))
    assert r.kind == "open_palm"
    assert r.label == "Open palm"


def test_recognize_no_hand():
    r = _recognizer().recognize([])
    assert r.kind == "none"


def test_recognize_fist_is_known():
    rec = _recognizer()
    first = _hold(rec, _hand(_synthetic(fist=True)))
    assert first.kind == "fist"
    assert first.registration_candidate is None


def test_scroll_detection():
    rec = _recognizer()
    lm = _synthetic(spread=True)
    rec.recognize([_hand(lm)])
    out = None
    for i in range(1, 9):
        lm = lm + np.array([0.0, -0.05, 0.0])
        out = rec.recognize([_hand(lm)])
    assert out.kind == "scroll"
    assert out.scroll_direction in ("up", "down")


def test_moving_fist_does_not_scroll():
    rec = _recognizer()
    lm = _synthetic(fist=True)
    rec.recognize([_hand(lm)])
    out = None
    for i in range(1, 9):
        lm = lm + np.array([0.0, -0.05, 0.0])
        out = rec.recognize([_hand(lm)])
    assert out.kind != "scroll"


def test_two_hands_pinch_primary():
    rec = _recognizer()
    primary_pinch = _hand(_synthetic(pinch=True), score=0.95)
    other_spread = _hand(_synthetic(spread=True), score=0.7)
    out = None
    for _ in range(4):
        out = rec.recognize([other_spread, primary_pinch])
    assert out.kind == "pinch"


def test_motion_template_matches():
    """A learned Scroll Down template fires on matching motion."""
    motions = MotionRegistry()
    motions.add(
        MotionTemplate(
            id="scroll_down",
            name="Scroll Down",
            direction="down",
            finger_mask=FINGER_INDEX | FINGER_MIDDLE,
            speed=0.3,
        )
    )
    rec = _recognizer(motions=motions)
    lm = _synthetic(spread=True)
    rec.recognize([_hand(lm)])
    out = None
    for i in range(1, 9):
        lm = lm + np.array([0.0, 0.05, 0.0])  # down (y increasing in frame)
        out = rec.recognize([_hand(lm)])
    assert out.kind == "scroll"
    assert out.label == "Scroll Down"


def test_motion_template_wrong_direction_no_match():
    """A Scroll Down template does NOT fire on upward motion."""
    motions = MotionRegistry()
    motions.add(
        MotionTemplate(
            id="scroll_down",
            name="Scroll Down",
            direction="down",
            finger_mask=FINGER_INDEX | FINGER_MIDDLE,
            speed=0.3,
        )
    )
    rec = _recognizer(motions=motions)
    lm = _synthetic(spread=True)
    rec.recognize([_hand(lm)])
    out = None
    for i in range(1, 9):
        lm = lm + np.array([0.0, -0.05, 0.0])  # up
        out = rec.recognize([_hand(lm)])
    assert out.label != "Scroll Down"


def test_recorder_captures_direction():
    """The recorder summarizes a downward motion into a direction."""
    rec = MotionRecorder(duration_s=0.1)
    rec.start()
    lm = _synthetic(spread=True)
    for i in range(10):
        lm = lm + np.array([0.0, 0.05, 0.0])  # down
        rec.update(_hand(lm))
    recording = rec.finish()
    assert recording is not None
    assert recording.direction == "down"
    assert recording.valid


def test_stroke_latch_no_flap():
    """Direction latches: jitter around zero doesn't flip up/down."""
    rec = _recognizer()
    lm = _synthetic(spread=True)
    rec.recognize([_hand(lm)])
    out = None
    # Move down with small wobbles (0.01 up/down noise) but net downward.
    for i in range(12):
        wobble = 0.01 * (1 if i % 3 == 0 else -1)
        lm = lm + np.array([0.0, 0.04 + wobble, 0.0])
        out = rec.recognize([_hand(lm)])
    assert out.kind == "scroll"
    assert out.scroll_direction == "down"
    # A subsequent tiny upward blip must NOT flip the latched direction.
    for i in range(4):
        lm = lm + np.array([0.0, -0.01, 0.0])
        out = rec.recognize([_hand(lm)])
    assert out.scroll_direction == "down"


def test_horizontal_scroll():
    """Horizontal (left/right) motion scrolls."""
    rec = _recognizer()
    lm = _synthetic(spread=True)
    rec.recognize([_hand(lm)])
    out = None
    for i in range(1, 9):
        lm = lm + np.array([0.05, 0.0, 0.0])  # right
        out = rec.recognize([_hand(lm)])
    assert out.kind == "scroll"
    assert out.scroll_direction == "right"


def test_zoom_pinch_out():
    """Pinching fingers together (thumb-index distance shrinking) = zoom out."""
    rec = _recognizer()
    # Start spread (ratio ~0.61), then close fingers toward pinch (ratio ~0.12).
    lm = _synthetic(spread=True)
    rec.recognize([_hand(lm)])
    saw_zoom = False
    for i in range(1, 12):
        t = i / 10.0
        lm[8] = (1.0 - (1.0 - 0.62) * t, 0.35 * (1 - t), 0.0)
        lm[12] = (1.05 - (1.05 - 0.64) * t, -0.25 * (1 - t), 0.0)
        out = rec.recognize([_hand(lm)])
        if out.kind == "zoom" and out.zoom_direction == "out":
            saw_zoom = True
    assert saw_zoom


def test_zoom_spread_in():
    """Spreading fingers apart (thumb-index distance growing) = zoom in."""
    rec = _recognizer()
    # Start pinched (ratio ~0.12), then open fingers toward spread.
    lm = _synthetic(pinch=True)
    rec.recognize([_hand(lm)])
    saw_zoom = False
    for i in range(1, 16):
        t = i / 10.0  # overshoot so ratio crosses 0.60
        lm[8] = (0.62 + (1.0 - 0.62) * t, 0.35 * t, 0.0)
        lm[12] = (0.64 + (1.05 - 0.64) * t, -0.25 * t, 0.0)
        out = rec.recognize([_hand(lm)])
        if out.kind == "zoom" and out.zoom_direction == "in":
            saw_zoom = True
    assert saw_zoom


def test_recorder_variations():
    """A 20s window with two downward strokes yields two variations."""
    rec = MotionRecorder(duration_s=0.2)
    rec.start()
    lm = _synthetic(spread=True)
    # Stroke 1: down.
    for i in range(5):
        lm = lm + np.array([0.0, 0.05, 0.0])
        rec.update(_hand(lm))
    # Pause (stationary) to separate strokes.
    for i in range(3):
        rec.update(_hand(lm))
    # Stroke 2: down again.
    for i in range(5):
        lm = lm + np.array([0.0, 0.05, 0.0])
        rec.update(_hand(lm))
    variations = rec.finish_variations()
    assert len(variations) == 2
    assert all(v.direction == "down" for v in variations)
    assert all(v.valid for v in variations)
    # Embeddings: fixed length, non-empty.
    assert all(v.embedding is not None and v.embedding.size > 0 for v in variations)
    # Raw samples are dropped after embedding (recorder reset).
    assert len(rec._samples) == 0


def test_embedding_similar_down():
    """Two downward strokes at different speeds/angles → similar embeddings."""
    from axis.gestures.embedding import embedding_distance, path_embedding

    # Stroke A: straight down.
    pts_a = [np.array([0.5, 0.1 * i]) for i in range(10)]
    sizes_a = [0.2] * 10
    # Stroke B: same path, slightly different angle + faster spacing.
    pts_b = [np.array([0.55 + 0.02 * i, 0.12 * i]) for i in range(10)]
    sizes_b = [0.2] * 10
    emb_a = path_embedding(pts_a, sizes_a)
    emb_b = path_embedding(pts_b, sizes_b)
    assert emb_a.size == emb_b.size
    d = embedding_distance(emb_a, emb_b)
    assert d < 0.3, f"similar down strokes should be close, got {d:.3f}"


def test_embedding_differs_up_vs_down():
    """Down vs. up embeddings are far apart (the scroll-direction fix)."""
    from axis.gestures.embedding import embedding_distance, path_embedding

    pts_down = [np.array([0.5, 0.1 * i]) for i in range(10)]
    pts_up = [np.array([0.5, 0.9 - 0.1 * i]) for i in range(10)]
    sizes = [0.2] * 10
    emb_down = path_embedding(pts_down, sizes)
    emb_up = path_embedding(pts_up, sizes)
    d = embedding_distance(emb_down, emb_up)
    assert d > 0.5, f"up and down should be distinct, got {d:.3f}"


def test_navbar_click_selects_gesture():
    """Clicking the navbar gesture button cycles the selection."""
    from axis.ui import DebugView

    view = DebugView()
    # Fake layout: simulate a 640-wide frame so buttons get positioned.
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    view._layout_buttons(640)
    assert view.selected_gesture == "Scroll Down"
    # Click the gesture selector (first button) → opens dropdown.
    b = view._buttons[0]
    cmd = view.handle_click(b.x + 5, b.y + 5)
    assert cmd == "open_dropdown"
    assert view.dropdown_open is True
    # Click a dropdown item (e.g. "Scroll Up", the 2nd item).
    item_h = 26
    y_item = b.y + b.h + 2 + item_h + 5  # inside the 2nd item
    cmd = view.handle_click(b.x + 5, y_item)
    assert cmd == "select_gesture:Scroll Up"
    assert view.selected_gesture == "Scroll Up"
    assert view.dropdown_open is False
    # Click TRAIN.
    b2 = view._buttons[1]
    cmd = view.handle_click(b2.x + 5, b2.y + 5)
    assert cmd == "toggle_train"
    assert view.training is True
    # Click STOP (label changed when training).
    cmd = view.handle_click(b2.x + 5, b2.y + 5)
    assert cmd == "toggle_train"
    assert view.training is False


def test_motion_template_roundtrip_embedding():
    """A template with an embedding survives to_dict/from_dict."""
    from axis.gestures.embedding import path_embedding

    pts = [np.array([0.5, 0.1 * i]) for i in range(10)]
    emb = path_embedding(pts, [0.2] * 10)
    tpl = MotionTemplate(
        id="sd_0",
        name="Scroll Down",
        direction="down",
        finger_mask=FINGER_INDEX | FINGER_MIDDLE,
        speed=0.3,
        embedding=emb,
    )
    d = tpl.to_dict()
    tpl2 = MotionTemplate.from_dict(d)
    assert tpl2.embedding is not None
    assert np.allclose(tpl2.embedding, emb)


if __name__ == "__main__":
    import pytest
    raise SystemExit(pytest.main([__file__, "-v"]))
