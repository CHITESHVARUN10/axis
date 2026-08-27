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
    dx, dy = gain_curve(2.0, 1.0, base=1.5, accel=0.0)
    assert abs(dx - 3.0) < 1e-9
    assert abs(dy - 1.5) < 1e-9


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
    p = _pointer()
    p.update((0.5, 0.5))
    out = p.update((0.55, 0.5))
    assert out[0] > p.pos_x - p.config.deadzone_radius_px
    assert out[1] == p.pos_y


def test_hand_stop_stops():
    p = _pointer()
    p.update((0.5, 0.5))
    p.update((0.55, 0.5))
    pos_after_move = (p.pos_x, p.pos_y)
    # Same position again → delta ~ 0 → no movement.
    out = p.update((0.55, 0.5))
    assert abs(out[0] - pos_after_move[0]) < 1e-6
    assert abs(out[1] - pos_after_move[1]) < 1e-6


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
    assert cfg.accel == 0.0  # default


def test_load_cursor_config_defaults():
    """A missing config file falls back to defaults."""
    import pathlib

    cfg = load_cursor_config(pathlib.Path("/nonexistent/cursor.json"))
    assert cfg.base_gain == 1.5
    assert cfg.radius_px == 12
    assert cfg.mode == "auto"
