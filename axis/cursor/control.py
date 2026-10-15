"""Control mode — gesture-gated real system-cursor state machine (PRD §6.2/6.3/8.2).

Pure logic, no Quartz: `ControlState.update(kind, hand_present, ...)` maps
one frame's recognition result to an explicit control state + action. The
caller (main.py) owns the pointer/OS calls:

    Tracking  → caller runs pointer.update + real_cursor.move_to
    Scrolling → caller runs real_cursor.scroll (no cursor move)
    Dragging  → caller runs pointer.update + real_cursor.dragged_to
    Clutched  → caller freezes (pointer.reset, no OS calls)

States: idle → tracking ⇄ clutched ⇄ dragging, tracking → scrolling.
Movement is gated — pinch/scroll/zoom/fist never move the cursor.

Click semantics (all pinch-driven):
    Left click   — pinch, released before right_click_hold_ms
    Double click — two releases inside double_click_ms at the same spot
    Right click  — hold the pinch WITHOUT moving past drag_start_px for
                   right_click_hold_ms (sub-threshold jitter is allowed);
                   consumed, so the release fires nothing
    Drag         — pinch + move ≥ drag_start_px at ≥ drag_hold_ms

Scroll is pixel-proportional: palm travel this frame along the latched
axis (fed in as `palm_center` and tracked HERE so per-frame pointer
resets can't zero it — the old bug) × scroll_gain_px → float px; the
real cursor post rounds with a residue carry. Small strokes scroll a
little, fast strokes scroll a lot, no truncation cliffs.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Optional, Tuple

from axis.cursor.config import ControlConfig


@dataclass
class ControlDecision:
    """One frame's decision: state + at most one action."""

    state: str  # idle | tracking | clutched | dragging | scrolling
    action: str  # none | move | click | double_click | right_click | drag_start | drag_end | scroll
    scroll_dx: float = 0.0
    scroll_dy: float = 0.0


class ControlState:
    """Gesture-gated control state with debounce/latch timers."""

    def __init__(self, config: Optional[ControlConfig] = None,
                 clock=time.monotonic):
        self.config = config or ControlConfig()
        self._clock = clock
        self.state: str = "idle"
        # Pinch lifecycle.
        self._pinch_frames: int = 0
        self._pinch_anchor: Optional[Tuple[float, float]] = None
        self._pinch_start_t: float = 0.0
        self._pinch_consumed: bool = False  # right click consumed this pinch
        self._last_hand_t: float = 0.0
        self._last_click_t: float = -1e9
        self._last_click_pos: Optional[Tuple[float, float]] = None
        self._button_held: bool = False
        # Scroll: palm delta tracked here so pointer resets can't zero it.
        self._last_palm: Optional[Tuple[float, float]] = None
        self._scroll_grace_until: float = 0.0

    @property
    def button_held(self) -> bool:
        return self._button_held

    def update(self, kind: str, hand_present: bool,
               scroll_direction: str = "",
               cursor_pos: Optional[Tuple[float, float]] = None,
               palm_center: Optional[Tuple[float, float]] = None) -> ControlDecision:
        now = self._clock()
        cfg = self.config

        if hand_present:
            self._last_hand_t = now
        elif self._button_held:
            if (now - self._last_hand_t) * 1000.0 >= cfg.drop_hand_absent_ms:
                self._drop_drag()
                return ControlDecision("clutched", "drag_end")
            return ControlDecision("dragging", "none")
        elif self.state not in ("idle",):
            if (now - self._last_hand_t) * 1000.0 >= cfg.clutch_hand_absent_ms:
                self.state = "clutched"
                self._reset_pinch()
                self._last_palm = None
                return ControlDecision("clutched", "none")
        if not hand_present:
            # A pinch interrupted by losing the hand must not fire on the
            # later "release"; also reset the scroll palm anchor.
            self._reset_pinch()
            self._last_palm = None
            if self.state == "idle":
                return ControlDecision("idle", "none")
            return ControlDecision(self.state, "none")

        if kind == "fist":
            self._reset_pinch()
            self._last_palm = None
            if self._button_held:
                self._drop_drag()
                return ControlDecision("clutched", "drag_end")
            self.state = "clutched"
            return ControlDecision("clutched", "none")

        if kind == "scroll":
            return self._update_scroll(now, scroll_direction, palm_center)
        if kind == "pinch":
            return self._update_pinch(now, cursor_pos)
        return self._finish_pinch(now, kind, cursor_pos)

    # --- pinch lifecycle -------------------------------------------------

    def _drop_drag(self) -> None:
        self._button_held = False
        self.state = "clutched"
        self._reset_pinch()
        self._last_palm = None

    def _reset_pinch(self) -> None:
        self._pinch_frames = 0
        self._pinch_anchor = None
        self._pinch_start_t = 0.0
        self._pinch_consumed = False

    def _update_pinch(self, now, cursor_pos) -> ControlDecision:
        cfg = self.config
        if self.state == "scrolling":
            self.state = "tracking"
        # Pinch carries no palm motion: drop any scroll anchor so the next
        # stroke starts fresh instead of jumping from a stale position.
        self._last_palm = None
        if self._pinch_frames == 0:
            self._pinch_anchor = cursor_pos
            self._pinch_start_t = now
            self._pinch_consumed = False
        self._pinch_frames += 1
        moved = 0.0
        if cursor_pos is not None and self._pinch_anchor is not None:
            moved = abs(cursor_pos[0] - self._pinch_anchor[0]) + abs(
                cursor_pos[1] - self._pinch_anchor[1])
        held_ms = (now - self._pinch_start_t) * 1000.0
        if self._button_held:
            return ControlDecision("dragging", "move")
        if held_ms >= cfg.drag_hold_ms and moved >= cfg.drag_start_px:
            self._button_held = True
            self.state = "dragging"
            return ControlDecision("dragging", "drag_start")
        if (held_ms >= cfg.right_click_hold_ms and moved < cfg.drag_start_px
                and not self._pinch_consumed):
            # Held still long enough for a context menu → right click; the
            # pinch is consumed so the later release fires nothing.
            self._pinch_consumed = True
            self.state = "tracking"
            self._last_click_t = -1e9
            return ControlDecision("tracking", "right_click")
        if self._pinch_consumed:
            # Right click already fired; wait for the release.
            self.state = "tracking"
            return ControlDecision("tracking", "none")
        self.state = "tracking"
        return ControlDecision("tracking", "none")

    def _finish_pinch(self, now, kind, cursor_pos) -> ControlDecision:
        """Pinch ended (another kind). Click on release if due; scroll grace
        holds the cursor still right after a stroke."""
        cfg = self.config
        click_action = ""
        if (
            self._pinch_frames >= max(cfg.click_debounce_frames, 1)
            and not self._button_held
            and not self._pinch_consumed
        ):
            held_ms = (now - self._pinch_start_t) * 1000.0
            if held_ms < cfg.right_click_hold_ms:
                click_action = "click"
                if (self._last_click_t > 0
                        and (now - self._last_click_t) * 1000.0 <= cfg.double_click_ms
                        # Same-spot check only when both positions exist;
                        # without them the time window alone decides.
                        and (cursor_pos is None or self._last_click_pos is None
                             or abs(cursor_pos[0] - self._last_click_pos[0])
                             + abs(cursor_pos[1] - self._last_click_pos[1])
                             <= cfg.double_click_max_px)):
                    click_action = "double_click"
        self._reset_pinch()
        self._last_palm = None
        if self._button_held:
            self._button_held = False
            self.state = "tracking"
            # A drag release is not a click: leave _last_click_t alone so a
            # quick click right after a drag can't become a phantom double.
            return ControlDecision("tracking", "drag_end")
        if click_action:
            self.state = "tracking"
            self._last_click_t = now
            self._last_click_pos = cursor_pos
            return ControlDecision("tracking", click_action)
        scroll_grace = self.state == "scrolling" and now < self._scroll_grace_until
        if scroll_grace:
            # Stroke settling: hold scroll state, no cursor move.
            return ControlDecision("scrolling", "none")
        prev = self.state
        self.state = "tracking"
        if prev in ("idle", "clutched"):
            return ControlDecision("tracking", "move", 0.0, 0.0)
        return ControlDecision("tracking", "move")

    # --- scroll ---------------------------------------------------------

    def _update_scroll(self, now, scroll_direction,
                       palm_center) -> ControlDecision:
        cfg = self.config
        self._reset_pinch()
        if self._button_held:
            self._drop_drag()
            self.state = "scrolling"
            self._scroll_grace_until = now + 0.35
            return ControlDecision("scrolling", "drag_end")
        self.state = "scrolling"
        self._scroll_grace_until = now + 0.35
        prev_palm = self._last_palm
        self._last_palm = palm_center
        if not scroll_direction:
            # Stroke settling frames: hold scrolling, emit nothing — no
            # cursor move, no wheel tick, no direction flip risk.
            return ControlDecision("scrolling", "none")
        dx = dy = 0.0
        if palm_center is not None and prev_palm is not None:
            if scroll_direction in ("left", "right"):
                # Hand moves left → dx negative (scroll left), right → positive.
                signed = (palm_center[0] - prev_palm[0]) * cfg.scroll_gain_px
            else:
                # Camera Y grows downward: hand DOWN = page DOWN → dy
                # negative (positive dy in CGEvent terms scrolls up).
                signed = -(palm_center[1] - prev_palm[1]) * cfg.scroll_gain_px
            cap = cfg.scroll_max_px
            if cap > 0 and abs(signed) > cap:
                signed = cap if signed > 0 else -cap
            if scroll_direction in ("up", "down"):
                dy = signed
            else:
                dx = signed
        if abs(dx) + abs(dy) < 0.05:
            return ControlDecision("scrolling", "none")
        return ControlDecision("scrolling", "scroll", dx, dy)


__all__ = ["ControlDecision", "ControlState"]
