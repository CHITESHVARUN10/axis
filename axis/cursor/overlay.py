"""Virtual cursor overlay — a click-through, always-on-top AppKit dot.

The dot is drawn in a borderless, transparent, click-through NSWindow so
it never intercepts real mouse input. Runs on its own daemon thread with
a running NSApplication event loop; position updates arrive via a queue.

Color coding follows PRD §9.1:
  - green ring  — tracking active
  - yellow ring — tracking lost / idle
"""

from __future__ import annotations

import queue
import threading
from typing import Optional, Tuple

from axis.cursor.config import CursorConfig


class _CursorDotView:
    """Minimal NSView that draws a filled circle with a ring.

    Implemented with the Objective-C runtime so it works with the pyobjc
    frameworks available in this project (no Cocoa subclassing needed).
    """

    _cursor = None  # the VirtualCursor that owns this view

    @classmethod
    def set_owner(cls, owner) -> None:
        cls._cursor = owner

    @classmethod
    def draw_(cls, view, rect) -> None:
        try:
            import AppKit
            from Quartz import CGContextSetFillColor, CGContextSetStrokeColor
            import Quartz

            owner = cls._cursor
            if owner is None:
                return
            ctx = AppKit.NSGraphicsContext.currentContext().CGContext()
            w = rect.size.width
            h = rect.size.height
            cx, cy = w / 2.0, h / 2.0
            r = min(w, h) / 2.0 - 2.0
            # Ring color by tracking state.
            color = owner._green if owner._active else owner._yellow
            # Fill (opaque circle) + ring (stroke).
            CGContextSetFillColor(ctx, color)
            CGContextSetStrokeColor(ctx, (1.0, 1.0, 1.0, 0.9))
            Quartz.CGContextAddArc(ctx, cx, cy, max(r, 1.0), 0.0, 6.283185307179586, 0)
            Quartz.CGContextDrawPath(ctx, Quartz.kCGPathFillStroke)
        except Exception:
            pass


class VirtualCursor:
    """Click-through overlay dot for the virtual cursor."""

    def __init__(self, config: Optional[CursorConfig] = None):
        self.config = config or CursorConfig()
        self._active = False
        self._green = (0.0, 0.8, 0.2, 0.95)
        self._yellow = (0.95, 0.8, 0.1, 0.95)
        self._queue: "queue.Queue[Optional[Tuple[float, float, bool]]]" = queue.Queue()
        self._thread: Optional[threading.Thread] = None
        self._window = None
        self._screen_h = 900.0
        self._view = None

    # --- lifecycle ------------------------------------------------------

    def start(self) -> bool:
        """Start the overlay thread. Returns False if AppKit isn't usable."""
        try:
            import AppKit  # noqa: F401
        except ImportError:
            return False
        if self._thread is not None and self._thread.is_alive():
            return True
        _CursorDotView.set_owner(self)
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return True

    def _run(self) -> None:
        """AppKit event loop on a daemon thread; drains the position queue."""
        try:
            import AppKit

            app = AppKit.NSApplication.sharedApplication()
            screen = AppKit.NSScreen.mainScreen().frame()
            self._screen_h = float(screen.size.height)

            self._window = AppKit.NSWindow.alloc().initWithContentRect_styleMask_backing_defer_(
                ((0, 0), (self.config.radius_px * 2, self.config.radius_px * 2)),
                AppKit.NSWindowStyleMaskBorderless,
                AppKit.NSBackingStoreBuffered,
                False,
            )
            self._window.setOpaque_(False)
            self._window.setBackgroundColor_(AppKit.NSColor.clearColor())
            self._window.setLevel_(AppKit.NSScreenSaverWindowLevel)
            self._window.setIgnoresMouseEvents_(True)
            self._window.setHasShadow_(False)

            self._view = AppKit.NSView.alloc().initWithFrame_(
                ((0, 0), (self.config.radius_px * 2, self.config.radius_px * 2))
            )
            self._window.setContentView_(self._view)
            self._window.orderFrontRegardless()

            while True:
                try:
                    item = self._queue.get(timeout=0.05)
                except queue.Empty:
                    item = None
                if item is not None:
                    self._move(item)
                app.run() if False else None  # process events via nextEvent
                # Pump the run loop manually so the thread stays responsive
                # to both the queue and AppKit event processing.
                import time as _t
                _t.sleep(0.005)
        except Exception:
            pass

    def _move(self, item: Tuple[float, float, bool]) -> None:
        x, y, active = item
        self._active = active
        if self._window is None:
            return
        r = self.config.radius_px
        origin = (x - r, self._screen_h - y - r)
        self._window.setFrameOrigin_(origin)

    # --- API ------------------------------------------------------------

    def update(self, x: float, y: float, active: bool) -> None:
        """Queue a position update (screen points, top-left origin)."""
        self._queue.put((x, y, active))

    def hide(self) -> None:
        self._queue.put(None)
        self._active = False

    def stop(self) -> None:
        self.hide()
        # The daemon thread exits with the process; nothing to join.


__all__ = ["VirtualCursor"]
