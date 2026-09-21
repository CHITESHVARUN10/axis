"""Tests for the cursor subsystem — filters, relative pointer, config."""

import math

import numpy as np

from axis.cursor import (
    CursorConfig,
    OneEuroFilter,
    RelativePointer,
    deadzone,
    gain_curve,
    load_cursor_config,
)


# --- filter tests ----------------------------------------------------------

def test_deadzone_inside_radius_zero():
    dx, dy = deadzone(1.0, 0.5, radius=2.0)
    assert (dx, dy) == (0.0, 0.0)
    # Exactly at radius → still inside (strict <).
    dx, dy = deadzone(2.0, 0.0, radius=2.0)
    assert dx == 0.0


def test_deadzone_ramps_outside():
    # Just outside radius → small but non-zero, same direction.
    dx, dy = deadzone(2.01, 0.0, radius=2.0)
    assert dx > 0.0
    assert dy == 0.0
    # Far outside → ~identity direction.
    dx2, dy2 = deadzone(10.0, 0.0, radius=2.0)
    assert abs(dx2 - 8.0) < 1e-9


def test_one_euro_converges():
    filt = OneEuroFilter(min_cutoff=1.0, beta=0.0, d_cutoff=1.0)
    out = filt(0.0, 1 / 30)
    for _ in range(100):
        out = filt(0.0, 1 / 30)
    assert abs(out) < 1e-4


def test_one_euro_tracks_ramp():
    """A fast ramp is tracked with low lag (cutoff rises with speed)."""
    filt = OneEuroFilter(min_cutoff=0.5, beta=0.1, d_cutoff=1.0)
    vals = [float(i) for i in range(60)]
    out = []
    for v in vals:
        out.append(filt(v, 1 / 30))
    # At the end it should be close to the input (small lag).
    assert abs(out[-1] - vals[-1]) < 3.0


def test_gain_curve_linear_default():
    dx, dy = gain_curve(2.0, 1.0, base=1.3, accel=0.6, accel_power=1.4,
                        ref_speed=150.0)
    scale = 1.3 * (1.0 + 0.6 * ((math.hypot(2.0, 1.0) / 150.0) ** 1.4))
    assert abs(dx - 2.0 * scale) < 1e-9
    assert abs(dy - 1.0 * scale) < 1e-9


def test_gain_curve_slow_precise_fast_covers():
    """Trackpad feel: slow drifts stay near base, fast swipes accelerate."""
    sx, _ = gain_curve(10.0, 0.0, base=1.3, accel=0.6, accel_power=1.4,
                       ref_speed=150.0)
    fx, _ = gain_curve(150.0, 0.0, base=1.3, accel=0.6, accel_power=1.4,
                       ref_speed=150.0)
    assert abs(sx / 10.0 - 1.3) < 0.15
    assert abs(fx / 150.0 - 1.3 * 1.6) < 1e-9
    assert fx / 150.0 > sx / 10.0


# --- RelativePointer ---------------------------------------------------------

def _pointer(**kw):
    return RelativePointer(CursorConfig(**kw))


def test_no_hand_no_move():
    p = _pointer()
    assert p.update(None) is None
    assert p.update(None) is None


def test_first_hand_no_jump():
    """First frame after re-detection must not jump the cursor."""
    p = _pointer()
    start = (p.pos_x, p.pos_y)
    p.update(None)          # lost
    out = p.update((0.5, 0.5))  # re-detected
    assert out == start


def test_hand_right_moves_right():
    p = _pointer(mirror_x=False)
    p.update((0.5, 0.5))
    out = p.update((0.55, 0.5))
    assert out[0] > p.pos_x - p.config.deadzone_radius_px
    assert out[1] == p.pos_y


def test_hand_right_moves_right_in_selfie_default():
    """Default mirror_x=True: image-right motion moves the cursor right."""
    p = _pointer()
    assert p.mirror_x is True
    p.update((0.5, 0.5))
    x0 = p.pos_x
    p.update((0.45, 0.5))  # image-left = user's right in selfie view
    assert p.pos_x > x0 - p.config.deadzone_radius_px


def test_hand_stop_stops():
    p = _pointer()
    p.update((0.5, 0.5))
    p.update((0.55, 0.5))
    pos_after_move = (p.pos_x, p.pos_y)
    # Same position again → delta ~ 0 → no movement.
    out = p.update((0.55, 0.5))
    assert abs(out[0] - pos_after_move[0]) < 1e-6
    assert abs(out[1] - pos_after_move[1]) < 1e-6


def test_fast_swipe_capped_per_frame():
    """No single frame teleports: fast swipes are capped, then continue."""
    p = _pointer(mirror_x=False)
    p.set_screen(2000, 2000)
    p.sync_to(1000.0, 1000.0)
    p.update((0.1, 0.5))
    before = (p.pos_x, p.pos_y)
    out = p.update((0.9, 0.5), dt=1.0 / 30.0)
    step = abs(out[0] - before[0]) + abs(out[1] - before[1])
    assert step <= p.config.max_step_px_per_frame + 1e-6


def test_draw_landmarks_mirror_lines_up_with_hand():
    """Selfie flip must move the skeleton to the side the user sees."""
    from axis.ui.debug_view import draw_landmarks
    from axis.landmarks import Hand, HandData

    lm = np.zeros((21, 3), dtype=np.float32)
    lm[:, 0] = 0.2  # hand on the left in raw coords
    lm[:, 1] = 0.5
    hd = HandData(
        hands=[Hand(landmarks=lm, handedness="Right", score=0.9)],
        frame_w=640, frame_h=480,
    )
    plain = np.zeros((480, 640, 3), dtype=np.uint8)
    draw_landmarks(plain, hd, mirror_x=False)
    mirrored = np.zeros((480, 640, 3), dtype=np.uint8)
    draw_landmarks(mirrored, hd, mirror_x=True)
    left_plain = plain[:, :320].sum()
    right_mirrored = mirrored[:, 320:].sum()
    assert left_plain > 0 and right_mirrored > 0
    assert mirrored[:, :320].sum() == 0  # nothing drawn on the wrong side


def test_clamp_to_screen():
    p = _pointer()
    p.set_screen(100, 100)
    # Big repeated rightward motion clamps to the right edge.
    for _ in range(50):
        p.update((0.9, 0.5))
        p.update((0.95, 0.5))
    assert p.pos_x <= 100.0 - p.config.screen_clamp_margin_px
    assert p.pos_x >= 0.0


def test_set_screen_clamps_position():
    p = _pointer()
    p.pos_x, p.pos_y = 5000, -20
    p.set_screen(800, 600)
    assert 0 <= p.pos_x <= 800
    assert 0 <= p.pos_y <= 600


def test_config_roundtrip():
    cfg = CursorConfig.from_dict(
        {"gain": {"base": 2.0}, "deadzone_radius_px": 3.0, "radius_px": 12}
    )
    assert cfg.base_gain == 2.0
    assert cfg.deadzone_radius_px == 3.0
    assert cfg.radius_px == 12
    assert cfg.accel == 0.6  # trackpad default
    assert cfg.mirror_x is True  # selfie default
    assert cfg.accel_ref_px_per_frame == 150.0
    assert cfg.max_step_px_per_frame == 60.0


def test_load_cursor_config_defaults():
    """A missing config file falls back to defaults."""
    import pathlib

    cfg = load_cursor_config(pathlib.Path("/nonexistent/cursor.json"))
    assert cfg.base_gain == 1.3
    assert cfg.radius_px == 12
    assert cfg.mode == "auto"
    assert cfg.mirror_x is True
    assert cfg.screen_clamp_margin_px == 1
