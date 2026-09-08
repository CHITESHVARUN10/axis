"""Cursor subsystem — trackpad-style relative pointer + control mode.

Relative, trackpad-like pointer mapping (hand motion → screen-space
position) plus a click-through AppKit overlay. `control` mode
(`axis/cursor/control.py`) gates the real system cursor behind gestures:
move / click / drag / scroll with clutch.
"""

from axis.cursor.config import (
    ControlConfig,
    CursorConfig,
    load_cursor_config,
    resolve_cursor_mode,
)
from axis.cursor.control import ControlDecision, ControlState
from axis.cursor.controller import RelativePointer
from axis.cursor.filter import OneEuroFilter, deadzone, gain_curve
from axis.cursor.overlay import VirtualCursor

__all__ = [
    "ControlConfig",
    "ControlDecision",
    "ControlState",
    "CursorConfig",
    "load_cursor_config",
    "resolve_cursor_mode",
    "RelativePointer",
    "OneEuroFilter",
    "deadzone",
    "gain_curve",
    "VirtualCursor",
]
