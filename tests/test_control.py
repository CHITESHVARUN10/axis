"""Tests for control mode — gesture-gated real system-cursor control."""

from unittest import mock

from axis.cursor import (
    ControlConfig,
    ControlState,
    CursorConfig,
    RelativePointer,
    VirtualCursor,
    resolve_cursor_mode,
)
from axis.macos.real_cursor import RealCursorController
from main import _control_frame


class FakeClock:
    def __init__(self):
        self.t = 1000.0

    def __call__(self):
        return self.t

    def advance(self, s):
        self.t += s


def _state(**kw):
    cfg = ControlConfig(**kw) if kw else ControlConfig()
    clock = FakeClock()
    return ControlState(cfg, clock=clock), clock


def _rec(kind, scroll_direction=""):
    return type("R", (), {"kind": kind, "scroll_direction": scroll_direction})()


def _hand(x=0.5, y=0.5):
    import numpy as np

    lm = np.zeros((21, 3))
    for i in (0, 5, 9, 13, 17):
        lm[i, 0], lm[i, 1] = x, y
    return type("H", (), {"landmarks": lm})()


# --- mode resolution ----------------------------------------------------

def test_resolve_auto_trusted_is_control():
    assert resolve_cursor_mode("auto", trusted=True) == "control"


def test_resolve_control_untrusted_degrades_to_avatar():
    assert resolve_cursor_mode("control", trusted=False) == "avatar"


def test_resolve_control_trusted_stays_control():
    assert resolve_cursor_mode("control", trusted=True) == "control"


def test_resolve_real_still_works():
    assert resolve_cursor_mode("real", trusted=True) == "real"
    assert resolve_cursor_mode("real", trusted=False) == "avatar"


def test_control_config_from_cursor_json():
    cfg = CursorConfig.from_dict(
        {"mode": "control", "control": {"click_debounce_frames": 5}}
    )
    assert cfg.mode == "control"
    assert cfg.control.click_debounce_frames == 5
    assert cfg.control.precision_gain == 0.25


# --- tracking / clutch --------------------------------------------------

def test_open_palm_tracks_and_moves():
    st, _ = _state()
    d = st.update("open_palm", True)
    assert (d.state, d.action) == ("tracking", "move")


def test_fist_clutches_no_move():
    st, _ = _state()
    st.update("open_palm", True)
    d = st.update("fist", True)
    assert d.state == "clutched"


def test_hand_absent_clutches_after_timeout():
    st, clock = _state(clutch_hand_absent_ms=300.0)
    st.update("open_palm", True)
    clock.advance(0.5)
    d = st.update("none", False)
    assert d.state == "clutched"


def test_hand_absent_briefly_stays_tracking():
    st, clock = _state(clutch_hand_absent_ms=300.0)
    st.update("open_palm", True)
    clock.advance(0.1)
    d = st.update("none", False)
    assert d.state == "tracking"


# --- click debounce -----------------------------------------------------

def test_pinch_clicks_after_debounce_not_before():
    st, _ = _state(click_debounce_frames=3)
    st.update("open_palm", True)
    assert st.update("pinch", True, cursor_pos=(100, 100)).action == "none"
    assert st.update("pinch", True, cursor_pos=(100, 100)).action == "none"
    d = st.update("pinch", True, cursor_pos=(100, 100))
    assert d.action == "click"


def test_double_pinch_close_together_is_double_click():
    st, clock = _state(click_debounce_frames=1, double_click_ms=500.0)
    st.update("open_palm", True)
    assert st.update("pinch", True, cursor_pos=(50, 50)).action == "click"
    st.update("open_palm", True)
    clock.advance(0.2)
    d = st.update("pinch", True, cursor_pos=(52, 50))
    assert d.action == "double_click"


# --- drag latch ---------------------------------------------------------

def test_pinch_hold_and_move_starts_drag():
    st, clock = _state(drag_hold_ms=300.0, drag_start_px=5.0)
    st.update("open_palm", True)
    st.update("pinch", True, cursor_pos=(100, 100))
    clock.advance(0.4)
    d = st.update("pinch", True, cursor_pos=(120, 100))
    assert (d.state, d.action) == ("dragging", "drag_start")
    assert st.button_held is True


def test_drag_releases_on_open_palm():
    st, clock = _state(drag_hold_ms=0.0, drag_start_px=1.0)
    st.update("open_palm", True)
    st.update("pinch", True, cursor_pos=(0, 0))
    st.update("pinch", True, cursor_pos=(50, 50))
    assert st.button_held is True
    d = st.update("open_palm", True)
    assert d.action == "drag_end"
    assert st.button_held is False


def test_drag_auto_drops_when_hand_lost():
    st, clock = _state(drag_hold_ms=0.0, drag_start_px=1.0,
                       drop_hand_absent_ms=500.0)
    st.update("open_palm", True)
    st.update("pinch", True, cursor_pos=(0, 0))
    st.update("pinch", True, cursor_pos=(50, 50))
    assert st.button_held is True
    clock.advance(0.6)
    d = st.update("none", False)
    assert d.action == "drag_end"


# --- scroll -------------------------------------------------------------

def test_scroll_maps_direction_to_wheel():
    st, _ = _state()
    d = st.update("scroll", True, scroll_direction="down")
    assert d.state == "scrolling"
    assert d.action == "scroll"
    assert d.scroll_dy < 0
    # Continuous scroll: an empty-direction frame mid-stroke keeps
    # scrolling the latched direction (no flip, no stall).
    d = st.update("scroll", True, scroll_direction="")
    assert d.state == "scrolling"
    assert d.action == "scroll"
    assert d.scroll_dy < 0


def test_scroll_without_direction_posts_nothing():
    st, _ = _state()
    d = st.update("scroll", True, scroll_direction="")
    assert d.state == "scrolling"
    assert d.action == "none"


def test_scroll_velocity_scales_wheel():
    st, _ = _state()
    slow = st.update("scroll", True, scroll_direction="down", palm_delta_px=0.0)
    st2, _ = _state()
    fast = st2.update("scroll", True, scroll_direction="down", palm_delta_px=80.0)
    assert fast.action == "scroll" and slow.action == "scroll"
    assert abs(fast.scroll_dy) > abs(slow.scroll_dy)


def test_scroll_grace_holds_cursor_between_strokes():
    """After a scroll stroke, settling frames hold (no cursor jump)."""
    st, clock = _state()
    st.update("scroll", True, scroll_direction="down")
    d = st.update("open_palm", True)
    assert d.state == "scrolling"
    assert d.action == "none"
    clock.advance(0.5)
    d = st.update("open_palm", True)
    assert (d.state, d.action) == ("tracking", "move")


def test_pointer_mirror_default_on():
    """Selfie view is the default: mirror_x on unless opted out."""
    assert CursorConfig().mirror_x is True
    assert CursorConfig.from_dict({}).mirror_x is True
    assert CursorConfig.from_dict({"mirror_x": False}).mirror_x is False
    p = RelativePointer(CursorConfig())
    assert p.mirror_x is True


def test_pointer_mirror_x_flips_horizontal():
    p = RelativePointer(CursorConfig(), mirror_x=True)
    p.update((0.5, 0.5))
    x0 = p.pos_x
    p.update((0.55, 0.5))
    assert p.pos_x < x0
    q = RelativePointer(CursorConfig(), mirror_x=False)
    q.update((0.5, 0.5))
    qx0 = q.pos_x
    q.update((0.55, 0.5))
    assert q.pos_x >= qx0 - q.config.deadzone_radius_px


def test_cursor_reaches_screen_edge_for_dock():
    """Clamp margin 1px: repeated same-direction strokes reach the edge."""
    p = RelativePointer(CursorConfig(), mirror_x=False)
    p.set_screen(200, 200)
    p.sync_to(100.0, 100.0)
    for _ in range(12):
        # One trackpad-style stroke: engage, swipe, lift (reset).
        p.update((0.3, 0.3))
        p.update((0.55, 0.55))
        p.reset()
    assert p.pos_x >= 200.0 - 1.0 - 1e-6
    assert p.pos_y >= 200.0 - 1.0 - 1e-6


# --- pointer additions --------------------------------------------------

def test_pointer_sync_to_reanchors_without_jump():
    p = RelativePointer(CursorConfig())
    p.sync_to(300.0, 400.0)
    assert (p.pos_x, p.pos_y) == (300.0, 400.0)
    out = p.update((0.5, 0.5))
    assert out == (300.0, 400.0)


def test_pointer_gain_scale_slows_precision():
    p1 = RelativePointer(CursorConfig(base_gain=1.5), mirror_x=False)
    p2 = RelativePointer(CursorConfig(base_gain=1.5), mirror_x=False)
    p2.set_gain_scale(0.25)
    for p in (p1, p2):
        p.set_screen(1000, 1000)
        p.sync_to(500.0, 500.0)
        p.update((0.5, 0.5))
    o1 = p1.update((0.51, 0.5))
    o2 = p2.update((0.51, 0.5))
    assert (o1[0] - 500.0) > (o2[0] - 500.0) > 0


def test_pointer_freeze_suppresses_motion():
    p = RelativePointer(CursorConfig(deadzone_radius_px=0.0))
    p.update((0.5, 0.5))
    p.freeze(100.0, now=1000.0)
    out = p.update((0.7, 0.5), now=1000.05)
    assert out == (p.pos_x, p.pos_y)
    out = p.update((0.7, 0.5), now=1000.2)
    assert out[0] > p.pos_x - 200


# --- real cursor additions (Quartz mocked) ------------------------------

@mock.patch("Quartz.CGEventCreateMouseEvent")
@mock.patch("Quartz.CGEventPost")
def test_mouse_down_up_dragged(mock_post, mock_create):
    import Quartz

    ctrl = RealCursorController()
    ctrl._pos = (10.0, 20.0)
    assert ctrl.mouse_down() is True
    assert mock_create.call_args.args[1] == Quartz.kCGEventLeftMouseDown
    assert ctrl.mouse_up() is True
    assert mock_create.call_args.args[1] == Quartz.kCGEventLeftMouseUp


@mock.patch("Quartz.CGWarpMouseCursorPosition")
@mock.patch("Quartz.CGEventCreateMouseEvent")
@mock.patch("Quartz.CGEventPost")
def test_dragged_to_warps_and_posts_dragged(mock_post, mock_create, mock_warp):
    import Quartz

    ctrl = RealCursorController()
    assert ctrl.dragged_to(300, 400) is True
    mock_warp.assert_called_once_with((300.0, 400.0))
    assert mock_create.call_args.args[1] == Quartz.kCGEventLeftMouseDragged


@mock.patch("Quartz.CGEventCreate")
@mock.patch("Quartz.CGEventGetLocation")
def test_get_position_reads_live_os_cursor(mock_loc, mock_create):
    mock_loc.return_value = (900.0, 100.0)
    ctrl = RealCursorController()
    assert ctrl.get_position() == (900.0, 100.0)
    assert ctrl.position() == (900.0, 100.0)


@mock.patch("Quartz.CGEventCreateMouseEvent")
@mock.patch("Quartz.CGEventPost")
def test_double_click_posts_two_pairs(mock_post, mock_create):
    ctrl = RealCursorController()
    assert ctrl.double_click() is True
    assert mock_post.call_count == 4


# --- overlay state tints ------------------------------------------------

def test_overlay_state_tints():
    ov = VirtualCursor()
    ov.set_state("tracking")
    assert ov.ring_color() == ov._green
    ov.set_state("dragging")
    assert ov.ring_color() == ov._blue
    ov.set_state("scrolling")
    assert ov.ring_color() == ov._purple
    ov.set_state("clutched")
    assert ov.ring_color() == ov._yellow


# --- _control_frame wiring (Quartz-free fakes) --------------------------

class FakePointer:
    def __init__(self):
        self.pos_x, self.pos_y = 700.0, 500.0
        self.screen_w, self.screen_h = 1440.0, 900.0
        self._c_prev = None
        self.moves = 0
        self.resets = 0

    def update(self, c, dt=1.0 / 30.0):
        self.moves += 1
        return (self.pos_x, self.pos_y)

    def reset(self):
        self.resets += 1

    def sync_to(self, x, y):
        self.pos_x, self.pos_y = x, y
        self.reset()

    def freeze(self, ms):
        pass


class FakeReal:
    def __init__(self):
        self.calls = []

    def get_position(self):
        return (700.0, 500.0)

    def move_to(self, x, y):
        self.calls.append("move")

    def click(self, b="left"):
        self.calls.append("click")

    def double_click(self, b="left"):
        self.calls.append("double")

    def mouse_down(self):
        self.calls.append("down")

    def mouse_up(self):
        self.calls.append("up")

    def dragged_to(self, x, y):
        self.calls.append("drag")

    def scroll(self, dy, dx=0.0):
        self.calls.append("scroll")


class FakeOverlay:
    def __init__(self):
        self.states = []

    def set_state(self, s):
        self.states.append(s)

    def update(self, x, y, active):
        pass

    def hide(self):
        pass


def test_control_frame_clicks_on_debounced_pinch():
    st, _ = _state(click_debounce_frames=1)
    ptr, real, ov = FakePointer(), FakeReal(), FakeOverlay()
    cfg = CursorConfig()
    _control_frame(st, ptr, real, ov, _rec("open_palm"), _hand(), cfg)
    _control_frame(st, ptr, real, ov, _rec("pinch"), _hand(), cfg)
    assert "click" in real.calls
    assert "move" in real.calls


def test_control_frame_clutch_freezes_cursor():
    st, _ = _state()
    ptr, real, ov = FakePointer(), FakeReal(), FakeOverlay()
    cfg = CursorConfig()
    _control_frame(st, ptr, real, ov, _rec("fist"), _hand(), cfg)
    assert real.calls == []
    assert ptr.resets >= 1


def test_control_frame_scroll_posts_wheel_not_move():
    st, _ = _state()
    ptr, real, ov = FakePointer(), FakeReal(), FakeOverlay()
    cfg = CursorConfig()
    _control_frame(st, ptr, real, ov, _rec("scroll", "down"), _hand(), cfg)
    assert real.calls == ["scroll"]
