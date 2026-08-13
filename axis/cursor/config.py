"""Cursor configuration — dataclass + loader from axis/config/cursor.json."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class CursorConfig:
    """Tunable parameters for the virtual cursor."""

    base_gain: float = 1.5
    accel: float = 0.0
    accel_power: float = 1.0
    deadzone_radius_px: float = 2.0
    min_cutoff: float = 1.2
    beta: float = 0.03
    d_cutoff: float = 1.0
    radius_px: int = 8
    screen_clamp_margin_px: int = 4

    @classmethod
    def from_dict(cls, d: dict) -> "CursorConfig":
        gain = d.get("gain", {})
        smoothing = d.get("smoothing", {})
        return cls(
            base_gain=float(gain.get("base", 1.5)),
            accel=float(gain.get("accel", 0.0)),
            accel_power=float(gain.get("accel_power", 1.0)),
            deadzone_radius_px=float(d.get("deadzone_radius_px", 2.0)),
            min_cutoff=float(smoothing.get("min_cutoff", 1.2)),
            beta=float(smoothing.get("beta", 0.03)),
            d_cutoff=float(smoothing.get("d_cutoff", 1.0)),
            radius_px=int(d.get("radius_px", 8)),
            screen_clamp_margin_px=int(d.get("screen_clamp_margin_px", 4)),
        )


def load_cursor_config(path: Optional[Path] = None) -> CursorConfig:
    """Load cursor.json; fall back to defaults if missing/unreadable."""
    if path is None:
        path = Path(__file__).resolve().parent.parent / "config" / "cursor.json"
    try:
        with open(path) as f:
            data = json.load(f)
        return CursorConfig.from_dict(data)
    except (FileNotFoundError, json.JSONDecodeError, KeyError):
        return CursorConfig()


__all__ = ["CursorConfig", "load_cursor_config"]
