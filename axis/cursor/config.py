"""Cursor configuration — dataclass + loader from axis/config/cursor.json."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class ControlConfig:
    """Tuning for `control` mode — real system-cursor control (PRD §6.2/6.3).

    All timeouts in ms, distances in screen px unless noted.
    """

    clutch_hand_absent_ms: float = 300.0
    click_debounce_frames: int = 3
    click_freeze_ms: float = 100.0
    double_click_ms: float = 500.0
    double_click_max_px: float = 10.0
    drag_hold_ms: float = 300.0
    drag_start_px: float = 5.0
    drop_hand_absent_ms: float = 500.0
    precision_gain: float = 0.25
    scroll_px_per_line: float = 120.0

    @classmethod
    def from_dict(cls, d: dict) -> "ControlConfig":
        return cls(
            clutch_hand_absent_ms=float(d.get("clutch_hand_absent_ms", 300.0)),
            click_debounce_frames=int(d.get("click_debounce_frames", 3)),
            click_freeze_ms=float(d.get("click_freeze_ms", 100.0)),
            double_click_ms=float(d.get("double_click_ms", 500.0)),
            double_click_max_px=float(d.get("double_click_max_px", 10.0)),
            drag_hold_ms=float(d.get("drag_hold_ms", 300.0)),
            drag_start_px=float(d.get("drag_start_px", 5.0)),
            drop_hand_absent_ms=float(d.get("drop_hand_absent_ms", 500.0)),
            precision_gain=float(d.get("precision_gain", 0.25)),
            scroll_px_per_line=float(d.get("scroll_px_per_line", 120.0)),
        )


@dataclass
class CursorConfig:
    """Tunable parameters for the virtual cursor."""

    mode: str = "auto"           # auto | avatar | real | control
    base_gain: float = 2.5
    accel: float = 0.0
    accel_power: float = 1.0
    deadzone_radius_px: float = 2.0
    min_cutoff: float = 1.2
    beta: float = 0.03
    d_cutoff: float = 1.0
    radius_px: int = 12
    screen_clamp_margin_px: int = 4
    mirror_x: bool = False
    control: ControlConfig = field(default_factory=ControlConfig)

    @classmethod
    def from_dict(cls, d: dict) -> "CursorConfig":
        gain = d.get("gain", {})
        smoothing = d.get("smoothing", {})
        return cls(
            mode=str(d.get("mode", "auto")),
            base_gain=float(gain.get("base", 2.5)),
            accel=float(gain.get("accel", 0.0)),
            accel_power=float(gain.get("accel_power", 1.0)),
            deadzone_radius_px=float(d.get("deadzone_radius_px", 2.0)),
            min_cutoff=float(smoothing.get("min_cutoff", 1.2)),
            beta=float(smoothing.get("beta", 0.03)),
            d_cutoff=float(smoothing.get("d_cutoff", 1.0)),
            radius_px=int(d.get("radius_px", 12)),
            screen_clamp_margin_px=int(d.get("screen_clamp_margin_px", 4)),
            mirror_x=bool(d.get("mirror_x", False)),
            control=ControlConfig.from_dict(d.get("control", {})),
        )


def resolve_cursor_mode(mode: str, trusted: bool) -> str:
    """Resolve the effective cursor mode (pure, unit-testable).

    - auto    → "control" if Accessibility is trusted, else "avatar"
    - control → control when trusted, degrades to "avatar" otherwise
    - real    → real when trusted, degrades to "avatar" otherwise
    - avatar  → always avatar (no permission needed)
    """
    mode = (mode or "auto").lower()
    if mode == "auto":
        return "control" if trusted else "avatar"
    if mode == "control":
        return "control" if trusted else "avatar"
    if mode == "real":
        return "real" if trusted else "avatar"
    return "avatar"


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


__all__ = ["ControlConfig", "CursorConfig", "load_cursor_config", "resolve_cursor_mode"]
