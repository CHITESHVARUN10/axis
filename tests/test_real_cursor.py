"""Tests for the real-cursor controller and cursor-mode resolution.

The Quartz cursor functions are mocked — no real cursor movement happens
during tests.
"""

from unittest import mock

from axis.cursor.config import CursorConfig, resolve_cursor_mode
from axis.macos.real_cursor import RealCursorController

# --- coordinate mapping (the classic bug trap) ---------------------------

def test_map_to_global_is_identity():
    """Quartz global display space is top-left origin, +Y down — matching
    our RelativePointer output. No y-flip allowed (regression guard)."""
    assert RealCursorController.map_to_global(120.0, 340.0) == (120.0, 340.0)
    assert RealCursorController.map_to_global(0.0, 0.0) == (0.0, 0.0)
    assert RealCursorController.map_to_global(1920.0, 1080.0) == (1920.0, 1080.0)


# --- mode resolution ----------------------------------------------------

def test_resolve_auto_trusted_real():
    assert resolve_cursor_mode("auto", trusted=True) == "control"


def test_resolve_control_trusted_stays_control():
    assert resolve_cursor_mode("control", trusted=True) == "control"


def test_resolve_control_untrusted_degrades_to_avatar():
    assert resolve_cursor_mode("control", trusted=False) == "avatar"


def test_resolve_auto_untrusted_avatar():
    assert resolve_cursor_mode("auto", trusted=False) == "avatar"


def test_resolve_real_untrusted_degrades_to_avatar():
    assert resolve_cursor_mode("real", trusted=False) == "avatar"


def test_resolve_real_trusted_real():
    assert resolve_cursor_mode("real", trusted=True) == "real"


def test_resolve_avatar_always_avatar():
    assert resolve_cursor_mode("avatar", trusted=True) == "avatar"
    assert resolve_cursor_mode("avatar", trusted=False) == "avatar"


def test_resolve_default_auto():
    assert resolve_cursor_mode(None, trusted=True) == "control"
    assert resolve_cursor_mode("", trusted=False) == "avatar"


# --- RealCursorController (Quartz mocked) --------------------------------

@mock.patch("Quartz.CGWarpMouseCursorPosition")
@mock.patch("Quartz.CGEventCreateMouseEvent")
@mock.patch("Quartz.CGEventPost")
def test_move_to_warps_and_posts_mousemoved(mock_post, mock_create, mock_warp):
    import Quartz

    ctrl = RealCursorController()
    assert ctrl.move_to(640, 480) is True
    mock_warp.assert_called_once_with((640.0, 480.0))
    mock_create.assert_called_once()
    assert mock_create.call_args.args[1] == Quartz.kCGEventMouseMoved
    assert mock_create.call_args.args[2] == (640.0, 480.0)
    assert mock_post.call_count == 1


@mock.patch("Quartz.CGEventCreateMouseEvent")
@mock.patch("Quartz.CGEventPost")
def test_click_left_posts_down_and_up(mock_post, mock_create):
    import Quartz

    ctrl = RealCursorController()
    ctrl._pos = (100.0, 200.0)
    assert ctrl.click("left") is True
    kinds = [c.args[1] for c in mock_create.call_args_list]
    assert Quartz.kCGEventLeftMouseDown in kinds
    assert Quartz.kCGEventLeftMouseUp in kinds
    assert mock_post.call_count == 2


@mock.patch("Quartz.CGEventCreateMouseEvent")
@mock.patch("Quartz.CGEventPost")
def test_click_right_uses_right_button_events(mock_post, mock_create):
    import Quartz

    ctrl = RealCursorController()
    ctrl._pos = (50.0, 50.0)
    assert ctrl.click("right") is True
    kinds = [c.args[1] for c in mock_create.call_args_list]
    assert Quartz.kCGEventRightMouseDown in kinds
    assert Quartz.kCGEventRightMouseUp in kinds


@mock.patch("Quartz.CGEventCreateScrollWheelEvent")
@mock.patch("Quartz.CGEventPost")
def test_scroll_posts_scrollwheel(mock_post, mock_scroll):
    ctrl = RealCursorController()
    assert ctrl.scroll(3) is True
    mock_scroll.assert_called_once()
    assert mock_post.call_count == 1


@mock.patch("Quartz.CGEventPost", side_effect=Exception("not trusted"))
@mock.patch("Quartz.CGEventCreateMouseEvent")
@mock.patch("Quartz.CGWarpMouseCursorPosition")
def test_move_to_survives_quartz_failure(mock_warp, mock_create, mock_post):
    """A permission/API failure must never crash the caller."""
    ctrl = RealCursorController()
    assert ctrl.move_to(100, 100) is False


def test_cursor_config_mode_from_dict():
    cfg = CursorConfig.from_dict({"mode": "real", "radius_px": 16})
    assert cfg.mode == "real"
    assert cfg.radius_px == 16