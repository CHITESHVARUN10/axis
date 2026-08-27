"""Cursor subsystem — trackpad-style virtual cursor.

Relative, trackpad-like pointer mapping (hand motion → screen-space
position) plus a click-through AppKit overlay dot. No OS cursor control,
no click gestures — the virtual cursor only tracks.
"""

from axis.cursor.config import CursorConfig, load_cursor_config, resolve_cursor_mode
from axis.cursor.controller import RelativePointer
from axis.cursor.filter import OneEuroFilter, deadzone, gain_curve
from axis.cursor.overlay import VirtualCursor

__all__ = [
    "CursorConfig",
    "load_cursor_config",
    "resolve_cursor_mode",
    "RelativePointer",
    "OneEuroFilter",
    "deadzone",
    "gain_curve",
    "VirtualCursor",
]
