"""Real cursor control — move & click the actual macOS pointer via Quartz.

This is the "final phase" layer: instead of (or in addition to) the visual
avatar arrow, the hand drives the REAL system cursor so apps see hover
states, clicks, etc. Requires Accessibility trust (AXIsProcessTrusted)
since macOS 10.14; every call is wrapped so a missing permission degrades
gracefully instead of crashing the camera loop.

Coordinate note (researched, verified): Quartz's global display space
(CGEvent / CGWarpMouseCursorPosition / CGEventGetLocation) has its origin
at the TOP-LEFT of the main display with +Y downward — the same convention
as OpenCV image coords and this project's RelativePointer output. So
`map_to_global(x, y) = (x, y)` — identity, no flip. Only AppKit windows
(NSScreen frames) use a bottom-left origin.
"""

from __future__ import annotations

import sys
from typing import Optional, Tuple


class RealCursorController:
    """Thin, dependency-free wrapper around Quartz cursor functions."""

    def __init__(self):
        self._pos: Tuple[float, float] = (0.0, 0.0)

    # --- availability ---------------------------------------------------

    @staticmethod
    def map_to_global(x: float, y: float) -> Tuple[float, float]:
        """Map a screen point to the Quartz global display space.

        Identity: both are top-left origin, +Y down. Kept as a named
        function so the coordinate contract is explicit and testable.
        """
        return float(x), float(y)

    def available(self) -> bool:
        """True if the OS will accept posted cursor events."""
        if sys.platform != "darwin":
            return False
        try:
            from axis.macos.accessibility import is_accessibility_trusted

            return bool(is_accessibility_trusted())
        except Exception:
            return False

    # --- actions --------------------------------------------------------

    def move_to(self, x: float, y: float) -> bool:
        """Move the real cursor to (x, y) (top-left screen points)."""
        gx, gy = self.map_to_global(x, y)
        self._pos = (gx, gy)
        try:
            import Quartz

            Quartz.CGWarpMouseCursorPosition((gx, gy))
            ev = Quartz.CGEventCreateMouseEvent(
                None, Quartz.kCGEventMouseMoved, (gx, gy), Quartz.kCGMouseButtonLeft
            )
            Quartz.CGEventPost(Quartz.kCGHIDEventTap, ev)
            return True
        except Exception:
            return False

    def click(self, button: str = "left") -> bool:
        """Post a click (down+up) at the current position."""
        if button not in ("left", "right"):
            button = "left"
        try:
            import Quartz

            if button == "right":
                kinds = (Quartz.kCGEventRightMouseDown, Quartz.kCGEventRightMouseUp)
                btn = Quartz.kCGMouseButtonRight
            else:
                kinds = (Quartz.kCGEventLeftMouseDown, Quartz.kCGEventLeftMouseUp)
                btn = Quartz.kCGMouseButtonLeft
            for kind in kinds:
                ev = Quartz.CGEventCreateMouseEvent(None, kind, self._pos, btn)
                Quartz.CGEventPost(Quartz.kCGHIDEventTap, ev)
            return True
        except Exception:
            return False

    def scroll(self, dy: float) -> bool:
        """Post a vertical scroll-wheel event (positive = scroll up)."""
        try:
            import Quartz

            ev = Quartz.CGEventCreateScrollWheelEvent(
                None, Quartz.kCGScrollEventUnitLine, 1, int(dy)
            )
            Quartz.CGEventPost(Quartz.kCGHIDEventTap, ev)
            return True
        except Exception:
            return False

    def position(self) -> Tuple[float, float]:
        """Last known cursor position (top-left screen points)."""
        return self._pos

    def stop(self) -> None:
        """No-op; kept for API symmetry with the overlay."""
        return None


__all__ = ["RealCursorController"]