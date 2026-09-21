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
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Optional

from axis.cursor.config import ControlConfig


@dataclass
class ControlDecision:
    """One frame's decision: state + at most one action."""

    state: str  # idle | tracking | clutched | dragging | scrolling
    action: str  # none | move | click | double_click | drag_start | drag_end | scroll
    scroll_dx: float = 0.0
    scroll_dy: float = 0.0


class ControlState:
    """Gesture-gated control state with debounce/latch timers."""

    def __init__(self, config: Optional[ControlConfig] = None,
                 clock=time.monotonic):
        self.config = config or ControlConfig()
        self._clock = clock
        self.state: str = "idle"
        self._pinch_frames: int = 0
        self._pinch_anchor: Optional[tuple] = None
        self._pinch_start_t: float = 0.0
        self._last_hand_t: float = 0.0
        self._last_click_t: float = -1e9
        self._last_click_pos: Optional[tuple] = None
        self._click_pending_double: bool = False
        self._button_held: bool = False
        self._last_scroll_t: float = 0.0
        self._scroll_grace_until: float = 0.0
        self._last_decision_was_scroll: bool = False
        self._scroll_dir: str = ""

    @property
    def button_held(self) -> bool:
        return self._button_held

    def update(self, kind: str, hand_present: bool,
               scroll_direction: str = "",
               cursor_pos: Optional[tuple] = None,
               palm_delta_px: float = 0.0) -> ControlDecision:
        now = self._clock()
        cfg = self.config

        if hand_present:
            self._last_hand_t = now
        elif self._button_held:
            if (now - self._last_hand_t) * 1000.0 >= cfg.drop_hand_absent_ms:
                self._button_held = False
                self.state = "clutched"
                self._pinch_frames = 0
                return ControlDecision("clutched", "drag_end")
            return ControlDecision("dragging", "none")
        elif self.state not in ("idle",):
            if (now - self._last_hand_t) * 1000.0 >= cfg.clutch_hand_absent_ms:
                self.state = "clutched"
                self._pinch_frames = 0
                return ControlDecision("clutched", "none")

        if not hand_present:
            if self.state == "idle":
                return ControlDecision("idle", "none")
            return ControlDecision(self.state, "none")

        if kind == "fist":
            self._pinch_frames = 0
            if self._button_held:
                self._button_held = False
                self.state = "clutched"
                return ControlDecision("clutched", "drag_end")
            self.state = "clutched"
            return ControlDecision("clutched", "none")

        if kind == "scroll":
            self._pinch_frames = 0
            if self._button_held:
                self._button_held = False
                self.state = "scrolling"
                self._last_scroll_t = now
                self._scroll_grace_until = now + 0.35
                self._last_decision_was_scroll = False
                self._scroll_dir = ""
                return ControlDecision("scrolling", "drag_end")
            self.state = "scrolling"
            self._last_scroll_t = now
            self._scroll_grace_until = now + 0.35
            self._last_decision_was_scroll = True
            # Direction lock: once a stroke latches a direction, hold it
            # until the hand stops (empty direction = stroke settling).
            # A held two-finger stroke therefore keeps scrolling the same
            # way every frame instead of flipping up/down on wobble. A new
            # stroke after a full stop starts fresh (see _scroll_dir reset).
            if not self._scroll_dir and scroll_direction:
                self._scroll_dir = scroll_direction
            direction = self._scroll_dir or scroll_direction
            if not direction:
                # Stroke settling (no latched direction yet): hold scrolling,
                # emit nothing — no cursor move, no wheel tick, no flip.
                return ControlDecision("scrolling", "none")
            step = cfg.scroll_px_per_line / 120.0
            # Velocity-proportional: a sustained stroke repeats the wheel
            # tick each frame (continuous scroll); scale by palm travel so
            # a fast stroke scrolls further than a slow one.
            ticks = 1.0 + min(max(palm_delta_px, 0.0) / 40.0, 2.0)
            step *= ticks
            dx = dy = 0.0
            if direction == "up":
                dy = step
            elif direction == "down":
                dy = -step
            elif direction == "left":
                dx = -step
            elif direction == "right":
                dx = step
            return ControlDecision("scrolling", "scroll", dx, dy)

        if kind == "pinch":
            if self.state == "dragging":
                return ControlDecision("dragging", "move")
            self._pinch_frames += 1
            if self._pinch_frames == 1:
                self._pinch_anchor = cursor_pos
                self._pinch_start_t = now
            moved = 0.0
            if cursor_pos is not None and self._pinch_anchor is not None:
                moved = abs(cursor_pos[0] - self._pinch_anchor[0]) + abs(
                    cursor_pos[1] - self._pinch_anchor[1])
            held_ms = (now - self._pinch_start_t) * 1000.0
            if held_ms >= cfg.drag_hold_ms and moved >= cfg.drag_start_px:
                self._button_held = True
                self.state = "dragging"
                return ControlDecision("dragging", "drag_start")
            if self._pinch_frames == cfg.click_debounce_frames:
                action = "click"
                if (self._last_click_t > 0
                        and (now - self._last_click_t) * 1000.0 <= cfg.double_click_ms
                        and cursor_pos is not None and self._last_click_pos is not None
                        and abs(cursor_pos[0] - self._last_click_pos[0])
                        + abs(cursor_pos[1] - self._last_click_pos[1]) <= cfg.double_click_max_px):
                    action = "double_click"
                    self._last_click_t = -1e9
                    self._click_pending_double = False
                else:
                    self._click_pending_double = True
                self._last_click_t = now
                self._last_click_pos = cursor_pos
                self.state = "tracking"
                return ControlDecision("tracking", action)
            self.state = "tracking"
            return ControlDecision("tracking", "none")

        self._pinch_frames = 0
        if self._button_held:
            self._button_held = False
            self.state = "tracking"
            return ControlDecision("tracking", "drag_end")
        # Scroll grace: right after a scroll stroke ends, the recognizer
        # reports a frame or two of open_palm/none while the hand settles.
        # Hold the scrolling state (no cursor move) instead of jumping.
        if now < self._scroll_grace_until and kind in ("open_palm", "none"):
            self.state = "scrolling"
            self._last_decision_was_scroll = False
            return ControlDecision("scrolling", "none")
        self._last_decision_was_scroll = False
        self._scroll_dir = ""
        if self.state in ("idle", "clutched"):
            self.state = "tracking"
            return ControlDecision("tracking", "move", 0.0, 0.0)
        self.state = "tracking"
        return ControlDecision("tracking", "move")


__all__ = ["ControlDecision", "ControlState"]
