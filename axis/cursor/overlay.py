"""Virtual cursor overlay — a click-through, always-on-top AppKit arrow.

Draws a crisp macOS-style arrow cursor (like the real mouse cursor) in a
borderless, transparent, click-through NSWindow so it never intercepts
real mouse input.

IMPORTANT (threading): AppKit forbids creating an NSWindow on anything but
the main thread (NSInternalInconsistencyException). The overlay is therefore
MAIN-THREAD DRIVEN: `start()` builds the window, the camera loop calls
`update()`/`hide()` directly each frame, and `pump()` services the AppKit
run loop so drawing actually lands. No daemon thread, no queue.

Two visual styles:
  - "arrow" (default) — the full arrow, the visible "second cursor".
  - "ring"           — a small tracking ring only, used as a marker when
                       the real macOS cursor itself is being moved.

Color coding follows PRD §9.1:
  - green ring  — tracking active
  - yellow ring — tracking lost / idle
"""

from __future__ import annotations

from typing import Optional, Tuple

from axis.cursor.config import CursorConfig

# --- AppKit view subclass (module-level, guarded for non-macOS) ----------

try:
    import AppKit  # noqa: F401
    import objc

    _HAS_APPKIT = True
except ImportError:  # pragma: no cover - non-macOS
    _HAS_APPKIT = False


if _HAS_APPKIT:

    class _ArrowCursorView(AppKit.NSView):
        """NSView that draws the macOS arrow (or a tracking ring)."""

        owner = None  # VirtualCursor

        def initWithFrame_(self, frame):
            self = objc.super(_ArrowCursorView, self).initWithFrame_(frame)
            if self is not None:
                self.owner = None
            return self

        def isFlipped(self) -> bool:
            # Top-left origin (+Y down) so arrow coordinates match the
            # screen/cursor space directly.
            return True

        def drawRect_(self, rect):
            owner = self.owner
            if owner is None:
                return
            try:
                import Quartz

                ctx = AppKit.NSGraphicsContext.currentContext().CGContext()
                w = float(self.bounds().size.width)
                h = float(self.bounds().size.height)

                if owner._style == "ring":
                    self._draw_ring(ctx, w, h, owner)
                else:
                    self._draw_arrow(ctx, w, h, owner, Quartz)
            except Exception:
                pass

        def _draw_ring(self, ctx, w, h, owner) -> None:
            import Quartz

            cx, cy = w / 2.0, h / 2.0
            color = owner.ring_color()
            Quartz.CGContextSetRGBStrokeColor(ctx, *color)
            Quartz.CGContextSetLineWidth(ctx, 2.5)
            Quartz.CGContextAddArc(ctx, cx, cy, min(w, h) / 2.0 - 2.0, 0.0, 6.283185307179586, 0)
            try:
                Quartz.CGContextStrokePath(ctx)
            except Exception:
                pass

        def _draw_arrow(self, ctx, w, h, owner, Quartz) -> None:
            """Draw the macOS pointer arrow (white fill + black outline)."""
            s = owner._scale  # arrow scale factor (radius_px / 12)

            # Classic macOS arrow silhouette in a 24x24 box with the tip at
            # the TOP-LEFT and the body extending down-right (screen space,
            # +Y down) — matching the real arrow cursor orientation.
            pts = [
                (0.0, 0.0), (23.0, 11.0), (16.0, 12.0), (18.0, 22.0),
                (14.0, 23.0), (12.0, 14.0), (5.0, 19.0),
            ]
            pad = 2.0
            poly = [(pad + px * s, pad + py * s) for px, py in pts]

            path = Quartz.CGPathCreateMutable()
            for i, (x, y) in enumerate(poly):
                if i == 0:
                    Quartz.CGPathMoveToPoint(path, None, x, y)
                else:
                    Quartz.CGPathAddLineToPoint(path, None, x, y)
            Quartz.CGPathCloseSubpath(path)

            Quartz.CGContextAddPath(ctx, path)
            Quartz.CGContextSetRGBFillColor(ctx, 1.0, 1.0, 1.0, 0.95)
            Quartz.CGContextSetRGBStrokeColor(ctx, 0.0, 0.0, 0.0, 0.9)
            Quartz.CGContextSetLineWidth(ctx, 1.5)
            Quartz.CGContextDrawPath(ctx, Quartz.kCGPathFillStroke)

            # Tracking ring below the arrow tip.
            cx, cy = poly[0][0] + 4.0, poly[0][1] + 4.0
            color = owner.ring_color()
            Quartz.CGContextSetRGBStrokeColor(ctx, *color)
            Quartz.CGContextSetLineWidth(ctx, 2.0)
            Quartz.CGContextAddArc(ctx, cx, cy, 3.0, 0.0, 6.283185307179586, 0)
            Quartz.CGContextStrokePath(ctx)


class VirtualCursor:
    """Click-through overlay arrow/ring. Main-thread driven (no daemon)."""

    WINDOW = 28

    def __init__(self, config: Optional[CursorConfig] = None):
        self.config = config or CursorConfig()
        self._active = False
        self._style = "arrow"
        self._scale = self.config.radius_px / 12.0
        self._green = (0.0, 0.8, 0.2, 0.95)
        self._yellow = (0.95, 0.8, 0.1, 0.95)
        self._blue = (0.2, 0.5, 1.0, 0.95)
        self._purple = (0.7, 0.3, 0.9, 0.95)
        self._tint = None
        self._window = None
        self._view = None
        self._screen_h = 900.0

    # --- lifecycle (call from the main thread) --------------------------

    def start(self) -> bool:
        """Create the overlay window. Returns False if AppKit is unusable.

        Must be called from the main thread (AppKit requirement).
        """
        if not _HAS_APPKIT:
            return False
        if self._window is not None:
            return True
        try:
            app = AppKit.NSApplication.sharedApplication()
            screen = AppKit.NSScreen.mainScreen().frame()
            self._screen_h = float(screen.size.height)

            w = self.WINDOW
            self._window = AppKit.NSWindow.alloc().initWithContentRect_styleMask_backing_defer_(
                ((0, 0), (w, w)),
                AppKit.NSWindowStyleMaskBorderless,
                AppKit.NSBackingStoreBuffered,
                False,
            )
            self._window.setOpaque_(False)
            self._window.setBackgroundColor_(AppKit.NSColor.clearColor())
            self._window.setLevel_(AppKit.NSScreenSaverWindowLevel)
            self._window.setIgnoresMouseEvents_(True)
            self._window.setHasShadow_(False)

            self._view = _ArrowCursorView.alloc().initWithFrame_(((0, 0), (w, w)))
            self._view.owner = self
            self._window.setContentView_(self._view)
            self._window.orderFrontRegardless()
            return True
        except Exception:
            self._window = None
            return False

    def update(self, x: float, y: float, active: bool) -> None:
        """Position the arrow tip at (x, y) — top-left screen points."""
        self._active = active
        if self._window is None:
            return
        if self._view is not None:
            self._view.setNeedsDisplay_(True)
        # Window is 28x28; tip drawn 2pt from its top-left (flipped view).
        # AppKit frame origin is the window's bottom-left:
        #   screen_y_from_top(origin) = screen_h - origin.y - 26
        # so screen_x = origin.x + 2 → origin.x = x - 2, and
        # screen_y = origin.y + 26 flipped → origin.y = screen_h - y - 26.
        origin = (x - 2.0, self._screen_h - y - 26.0)
        self._window.setFrameOrigin_(origin)

    def pump(self, seconds: float = 0.005) -> None:
        """Service the AppKit run loop so drawing lands; call each frame."""
        if self._window is None:
            return
        try:
            rl = AppKit.NSRunLoop.currentRunLoop()
            rl.runMode_beforeDate_(
                AppKit.NSDefaultRunLoopMode,
                AppKit.NSDate.dateWithTimeIntervalSinceNow_(seconds),
            )
        except Exception:
            pass

    def set_style(self, style: str) -> None:
        """'arrow' (default) or 'ring' (marker when the real cursor moves)."""
        self._style = "ring" if style == "ring" else "arrow"
        if self._view is not None:
            self._view.setNeedsDisplay_(True)

    def set_state(self, state: str) -> None:
        """Tint the ring for a control-mode state (PRD §9.1).

        tracking → green, clutched/idle → yellow, dragging → blue,
        scrolling → purple. Unknown states clear the override.
        """
        tints = {
            "tracking": self._green,
            "clutched": self._yellow,
            "idle": self._yellow,
            "dragging": self._blue,
            "scrolling": self._purple,
        }
        self._tint = tints.get(state)
        self._active = state in ("tracking", "dragging", "scrolling")
        if self._view is not None:
            self._view.setNeedsDisplay_(True)

    def ring_color(self):
        """Active ring color (tint override wins; testable without AppKit)."""
        if self._tint is not None:
            return self._tint
        return self._green if self._active else self._yellow

    def show(self) -> None:
        if self._window is not None:
            self._window.orderFrontRegardless()

    def hide(self) -> None:
        self._active = False
        if self._window is not None:
            self._window.orderOut_(None)

    def stop(self) -> None:
        self.hide()


__all__ = ["VirtualCursor"]