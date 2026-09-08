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

    def get_position(self) -> Tuple[float, float]:
        """Read the live OS cursor position (top-left screen points).

        Syncs the cached `_pos` so a RelativePointer can re-anchor to
        where the mouse/trackpad left the cursor (kills warp jumps).
        Falls back to the cached position when Quartz is unavailable.
        """
        try:
            import Quartz

            ev = Quartz.CGEventCreate(None)
            if ev is None:
                return self._pos
            pt = Quartz.CGEventGetLocation(ev)
            try:
                x, y = float(pt.x), float(pt.y)
            except AttributeError:
                x, y = float(pt[0]), float(pt[1])
            self._pos = (x, y)
            return self._pos
        except Exception:
            return self._pos

    @staticmethod
    def _button_kinds(button: str):
        """Return (down, up, dragged, btn) Quartz constants for a button."""
        import Quartz

        if button == "right":
            return (
                Quartz.kCGEventRightMouseDown,
                Quartz.kCGEventRightMouseUp,
                Quartz.kCGEventRightMouseDragged,
                Quartz.kCGMouseButtonRight,
            )
        return (
            Quartz.kCGEventLeftMouseDown,
            Quartz.kCGEventLeftMouseUp,
            Quartz.kCGEventLeftMouseDragged,
            Quartz.kCGMouseButtonLeft,
        )

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

            down, up, _, btn = self._button_kinds(button)
            for kind in (down, up):
                ev = Quartz.CGEventCreateMouseEvent(None, kind, self._pos, btn)
                Quartz.CGEventPost(Quartz.kCGHIDEventTap, ev)
            return True
        except Exception:
            return False

    def mouse_down(self, button: str = "left") -> bool:
        """Post a mouse-down at the current position (drag start)."""
        if button not in ("left", "right"):
            button = "left"
        try:
            import Quartz

            down, _, _, btn = self._button_kinds(button)
            ev = Quartz.CGEventCreateMouseEvent(None, down, self._pos, btn)
            Quartz.CGEventPost(Quartz.kCGHIDEventTap, ev)
            return True
        except Exception:
            return False

    def mouse_up(self, button: str = "left") -> bool:
        """Post a mouse-up at the current position (drag end)."""
        if button not in ("left", "right"):
            button = "left"
        try:
            import Quartz

            _, up, _, btn = self._button_kinds(button)
            ev = Quartz.CGEventCreateMouseEvent(None, up, self._pos, btn)
            Quartz.CGEventPost(Quartz.kCGHIDEventTap, ev)
            return True
        except Exception:
            return False

    def dragged_to(self, x: float, y: float, button: str = "left") -> bool:
        """Move to (x, y) with the button held (drag step)."""
        gx, gy = self.map_to_global(x, y)
        self._pos = (gx, gy)
        if button not in ("left", "right"):
            button = "left"
        try:
            import Quartz

            _, _, dragged, btn = self._button_kinds(button)
            Quartz.CGWarpMouseCursorPosition((gx, gy))
            ev = Quartz.CGEventCreateMouseEvent(None, dragged, (gx, gy), btn)
            Quartz.CGEventPost(Quartz.kCGHIDEventTap, ev)
            return True
        except Exception:
            return False

    def double_click(self, button: str = "left") -> bool:
        """Post two click pairs at the current position."""
        return bool(self.click(button)) and bool(self.click(button))

    def scroll(self, dy: float, dx: float = 0.0) -> bool:
        """Post a scroll-wheel event (positive dy = scroll up).

        Supports 2-axis scroll; falls back to the 1-axis call when the
        installed Quartz only accepts a single wheel argument.
        """
        try:
            import Quartz

            try:
                ev = Quartz.CGEventCreateScrollWheelEvent(
                    None, Quartz.kCGScrollEventUnitLine, 2, int(dy), int(dx)
                )
            except Exception:
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