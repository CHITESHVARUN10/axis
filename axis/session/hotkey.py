"""Session management — inactivity timeout + optional global hotkey.

The hotkey uses a Quartz event tap (no extra deps beyond the pyobjc
frameworks already pinned). The event tap is only active when the hotkey
feature is enabled.
"""

from __future__ import annotations

import time
from typing import Callable, Optional


class SessionGate:
    """Tracks hand activity and decides whether Gesture Mode is live.

    The camera keeps streaming (so re-activation is instant), but the
    cursor overlay hides and recognition is considered "paused" after
    `timeout_s` of no tracked hand.
    """

    def __init__(self, timeout_s: float = 120.0):
        self.timeout_s = timeout_s
        self._last_activity = time.monotonic()
        self._forced_off = False

    def note_hand(self) -> None:
        self._last_activity = time.monotonic()

    @property
    def active(self) -> bool:
        if self._forced_off:
            return False
        return (time.monotonic() - self._last_activity) < self.timeout_s

    def force_off(self) -> None:
        self._forced_off = True

    def force_on(self) -> None:
        self._forced_off = False
        self._last_activity = time.monotonic()


class HotkeyListener:
    """Optional global hotkey to toggle Gesture Mode (Quartz event tap)."""

    def __init__(self, keycode: int = 97):  # 97 = F9 (kVK_ANSI_F9-ish)
        self.keycode = keycode
        self.on_toggle: Optional[Callable[[], None]] = None
        self._tap = None

    def start(self) -> bool:
        """Install the event tap. Returns False if unavailable (non-macOS)."""
        if sys.platform != "darwin":
            return False
        try:
            import Quartz
        except ImportError:
            return False
        try:

            def _cb(proxy, _type, event, _refcon):
                if Quartz.CGEventGetIntegerValueField(
                    event, Quartz.kCGKeyboardEventKeycode
                ) == self.keycode:
                    if self.on_toggle is not None:
                        self.on_toggle()
                return event

            self._tap = Quartz.CGEventTapCreate(
                Quartz.kCGHIDEventTap,
                Quartz.kCGHeadInsertEventTap,
                Quartz.kCGEventTapOptionDefault,
                (1 << Quartz.kCGEventKeyDown),
                _cb,
                None,
            )
            if self._tap is None:
                return False
            self._source = Quartz.CFMachPortCreateRunLoopSource(
                None, self._tap, 0
            )
            Quartz.CFRunLoopAddSource(
                Quartz.CFRunLoopGetCurrent(), self._source, Quartz.kCFRunLoopCommonModes
            )
            return True
        except Exception:
            return False

    def stop(self) -> None:
        if self._tap is not None:
            try:
                import Quartz

                Quartz.CGEventTapEnable(self._tap, False)
            except Exception:
                pass


import sys  # noqa: E402

__all__ = ["HotkeyListener", "SessionGate"]
