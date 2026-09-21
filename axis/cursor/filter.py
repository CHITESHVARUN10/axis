"""Pure-math cursor filters — One Euro filter, dead zone, gain curve.

All functions are stateless (except the One Euro filter, which keeps the
internal state needed for adaptive smoothing). No OS calls here.
"""

from __future__ import annotations

import math
from typing import Optional, Tuple


class OneEuroFilter:
    """Adaptive low-pass filter for noisy 1-D signals.

    At low speeds the cutoff frequency is low (strong smoothing, kills
    jitter); at high speeds it rises (less lag). This is the classic
    "one euro" filter (Casiez et al., 2012).
    """

    def __init__(self, min_cutoff: float = 1.2, beta: float = 0.03, d_cutoff: float = 1.0):
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff
        self._x_prev: Optional[float] = None
        self._dx_prev: float = 0.0
        self._t_prev: Optional[float] = None

    def reset(self) -> None:
        self._x_prev = None
        self._dx_prev = 0.0
        self._t_prev = None

    def _alpha(self, cutoff: float, dt: float) -> float:
        tau = 1.0 / (2.0 * math.pi * cutoff)
        return 1.0 / (1.0 + tau / dt)

    def __call__(self, x: float, dt: float) -> float:
        if self._x_prev is None:
            self._x_prev = x
            self._dx_prev = 0.0
            self._t_prev = dt
            return x
        dt = max(dt, 1e-6)
        dx = (x - self._x_prev) / dt
        alpha_d = self._alpha(self.d_cutoff, dt)
        edx = alpha_d * dx + (1.0 - alpha_d) * self._dx_prev
        cutoff = self.min_cutoff + self.beta * abs(edx)
        alpha = self._alpha(cutoff, dt)
        out = alpha * x + (1.0 - alpha) * self._x_prev
        self._x_prev = out
        self._dx_prev = edx
        self._t_prev = dt
        return out


def deadzone(dx: float, dy: float, radius: float) -> Tuple[float, float]:
    """Circular dead zone: zero inside the radius, smooth ramp outside."""
    m = math.hypot(dx, dy)
    if m == 0.0 or m < radius:
        return 0.0, 0.0
    s = (m - radius) / m
    return dx * s, dy * s


def gain_curve(
    dx: float, dy: float, base: float = 1.5, accel: float = 0.0, accel_power: float = 1.0,
    ref_speed: float = 1.0,
) -> Tuple[float, float]:
    """Scale a delta by a base gain with an optional acceleration curve.

    `speed` is measured in the same units as dx/dy (px/frame); `ref_speed`
    is the speed at which the accel term equals `accel` (so a 10px/frame
    hand drift with ref_speed=200 stays near base gain, while a 200px/frame
    swipe gets the full boost). Normalizes the old raw-power curve, whose
    accel term exploded on real pixel magnitudes.
    """
    speed = math.hypot(dx, dy)
    scale = base * (1.0 + accel * ((speed / ref_speed) ** accel_power))
    return dx * scale, dy * scale


__all__ = ["OneEuroFilter", "deadzone", "gain_curve"]
