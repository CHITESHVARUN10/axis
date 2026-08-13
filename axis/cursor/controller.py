"""Relative pointer — maps hand motion to a screen-space cursor position.

Trackpad-style: only *relative* hand motion moves the cursor. The hand
position in the camera frame is never mapped absolutely to the screen —
the user re-centers the hand like lifting a finger off a trackpad.

Pipeline per frame (with a tracked hand):

    delta    = c_t - c_prev                    (normalized camera coords)
    delta_px = delta * (screen_w, screen_h)
    delta_px = gain_curve(delta_px)            (acceleration curve)
    delta_px = deadzone(delta_px, radius)      (jitter suppression)
    delta_px = one_euro(delta_px, dt)          (adaptive smoothing)
    pos     += delta_px
    pos      = clamp(pos, screen_bounds)

When no hand is tracked: nothing moves, and `c_prev` resets so the first
frame after re-detection produces zero delta (no cursor jump).
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

from axis.cursor.config import CursorConfig
from axis.cursor.filter import OneEuroFilter, deadzone, gain_curve


class RelativePointer:
    """Trackpad-style relative cursor integrator."""

    def __init__(self, config: Optional[CursorConfig] = None):
        self.config = config or CursorConfig()
        self.screen_w: float = 1440.0
        self.screen_h: float = 900.0
        self.pos_x: float = self.screen_w / 2.0
        self.pos_y: float = self.screen_h / 2.0
        self._c_prev: Optional[Tuple[float, float]] = None
        self._fx = OneEuroFilter(self.config.min_cutoff, self.config.beta, self.config.d_cutoff)
        self._fy = OneEuroFilter(self.config.min_cutoff, self.config.beta, self.config.d_cutoff)

    def set_screen(self, w: float, h: float) -> None:
        """Set the screen size in points; clamp the current position."""
        self.screen_w, self.screen_h = float(w), float(h)
        self.pos_x = min(max(self.pos_x, 0.0), self.screen_w)
        self.pos_y = min(max(self.pos_y, 0.0), self.screen_h)

    def reset(self) -> None:
        self._c_prev = None
        self._fx.reset()
        self._fy.reset()

    def update(self, c: Optional[Tuple[float, float]], dt: float = 1.0 / 30.0) -> Optional[Tuple[float, float]]:
        """Feed one frame's palm center (normalized [0,1] camera coords).

        Returns the new cursor position in screen points (top-left origin),
        or None when there is no hand (nothing moved).
        """
        if c is None:
            self.reset()
            return None
        if self._c_prev is None:
            self._c_prev = c
            return self.pos_x, self.pos_y

        dx = c[0] - self._c_prev[0]
        dy = c[1] - self._c_prev[1]
        self._c_prev = c

        # A truly stationary hand (delta below deadzone) must not move the
        # cursor — the One Euro filter's internal lag would otherwise keep
        # feeding residual motion after the hand stops.
        raw = math.hypot(dx * self.screen_w, dy * self.screen_h)
        if raw < self.config.deadzone_radius_px:
            self._fx.reset()
            self._fy.reset()
            return self.pos_x, self.pos_y

        dx *= self.screen_w
        dy *= self.screen_h

        dx, dy = gain_curve(
            dx, dy,
            base=self.config.base_gain,
            accel=self.config.accel,
            accel_power=self.config.accel_power,
        )
        dx, dy = deadzone(dx, dy, self.config.deadzone_radius_px)
        dx = self._fx(dx, dt)
        dy = self._fy(dy, dt)

        self.pos_x += dx
        self.pos_y += dy

        m = self.config.screen_clamp_margin_px
        self.pos_x = min(max(self.pos_x, m), self.screen_w - m)
        self.pos_y = min(max(self.pos_y, m), self.screen_h - m)
        return self.pos_x, self.pos_y


__all__ = ["RelativePointer"]
