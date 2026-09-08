"""Axis — privacy-first gesture interaction layer for macOS.

Phase 1 entry point. Runs a dev harness:
  - requests macOS camera permission
  - opens the camera, runs MediaPipe hand landmarking
  - recognizes pinch + scroll gestures (built-in rules + learned templates)
  - shows the live feed with overlays when --debug is set
  - when a motion isn't recognized: records it (~8s), shows a trail,
    then offers to classify it (Scroll Down / Scroll Up / custom name)

No real OS actions yet — detection and feedback only.
"""

from __future__ import annotations

import argparse
import queue
import sys
import threading
from typing import Optional

import cv2
import numpy as np

from axis.camera import Camera, CameraError
from axis.cursor import CursorConfig, RelativePointer, VirtualCursor, load_cursor_config
from axis.gestures import (
    GestureRecognizer,
    GestureRegistry,
    MotionRecorder,
    MotionRegistry,
    Recognition,
    load_user_gestures,
    load_user_motions,
    make_open_palm_definition,
    make_pinch_definition,
    save_user_gesture,
    save_user_motion,
)
from axis.landmarks import HandData, HandLandmarker
from axis.ui import DebugView

# --------------------------------------------------------------------------
# Registration / classification prompt. Runs on a background thread so the
# camera keeps streaming while the user types.
# --------------------------------------------------------------------------

RECORD_DURATION_S = 20.0


def _prompt_name(prompt: str) -> Optional[str]:
    """Blocking input() on a background thread; returns name or None."""
    try:
        return input(prompt).strip() or None
    except (EOFError, KeyboardInterrupt):
        return None


def _registration_worker(candidate: Recognition) -> None:
    """Prompt for a name and register a static gesture."""
    name = _prompt_name(
        "New gesture detected — would you like to register it? "
        "Enter a name (or press Enter to skip): "
    )
    if name:
        gesture = candidate.registration_candidate
        gesture.id = _slugify(name) or "custom"
        gesture.name = name
        save_user_gesture(gesture)
        print(f"\nRegistered gesture '{name}' — it will be recognized in future sessions.")


def _continue_worker(view) -> None:
    """Every ~60s of training, ask whether to keep recording.

    Blocks on input() in a background thread; the camera keeps streaming.
    If the user declines (or the window closes), training stops.
    """
    try:
        ans = input(
            "\nOne minute of training captured. Continue recording this "
            "gesture? [y/N]: "
        ).strip().lower()
    except (EOFError, KeyboardInterrupt):
        ans = "n"
    if ans not in ("y", "yes"):
        view.training = False
        view.record_count = 0
        view._toast("Training stopped — DONE")
    view._continue_done = True


def _slugify(name: str) -> str:
    """Sanitize a name into an id."""
    return "".join(c for c in name.lower() if c.isalnum() or c == "_")[:32]


# --------------------------------------------------------------------------
# Main pipeline
# --------------------------------------------------------------------------


def build_registry() -> GestureRegistry:
    """Built-in gestures + user gestures from ~/.axis/gestures.json."""
    registry = GestureRegistry(definitions={})
    for g in (make_pinch_definition(), make_open_palm_definition()):
        registry.add(g)
    user = load_user_gestures()
    for g in user.definitions.values():
        registry.add(g)
    return registry


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="axis",
        description="Privacy-first gesture interaction layer for macOS",
    )
    parser.add_argument("--version", action="version", version="axis 0.1.0")
    parser.add_argument(
        "--debug",
        dest="debug",
        action="store_true",
        help="show live camera feed with detection overlays (default)",
    )
    parser.add_argument(
        "--no-debug",
        dest="debug",
        action="store_false",
        help="run headless — no camera feed window",
    )
    parser.set_defaults(debug=True)
    parser.add_argument(
        "--camera",
        default=0,
        help="camera device: index (0 = default) or name, e.g. 'iPhone Camera'",
    )
    parser.add_argument(
        "--list-cameras",
        action="store_true",
        help="list available camera devices and exit",
    )
    parser.add_argument(
        "--config",
        default=None,
        help="path to a custom config directory (default: ~/.axis)",
    )
    parser.add_argument(
        "--no-cursor",
        dest="cursor",
        action="store_false",
        default=True,
        help="disable the virtual-cursor overlay (default: on)",
    )
    parser.add_argument(
        "--cursor-mode",
        choices=["auto", "avatar", "real", "control"],
        default=None,
        help="cursor mode: auto (best available), avatar (on-screen arrow, "
        "click-through), real (moves the actual macOS cursor; needs "
        "Accessibility permission), control (gesture-gated real cursor: "
        "move + click + drag + scroll; needs Accessibility permission)",
    )
    parser.add_argument(
        "--clicks",
        dest="clicks",
        action="store_true",
        default=False,
        help="enable pinch-to-click (posts real clicks at the cursor; "
        "requires real cursor mode + Accessibility permission; "
        "control mode always clicks)",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=120.0,
        help="inactivity timeout in seconds before the session pauses (default: 120)",
    )
    parser.add_argument(
        "--hotkey",
        default="",
        help="global hotkey (keycode) to toggle Gesture Mode, e.g. '97' for F9 "
        "(default: none)",
    )
    return parser.parse_args(argv)


def _palm_center(primary) -> tuple:
    """Normalized palm center (wrist + 4 MCPs) for the relative pointer."""
    lm = primary.landmarks
    return (
        float(np.mean(lm[[0, 5, 9, 13, 17], :2], axis=0)[0]),
        float(np.mean(lm[[0, 5, 9, 13, 17], :2], axis=0)[1]),
    )


def _control_frame(control, pointer, real_cursor, overlay, rec, primary,
                   cursor_config):
    """Run one control-mode frame: gate movement, route OS actions.

    Returns the cursor position for the debug-view mirror dot.
    """
    hand_present = primary is not None
    decision = control.update(
        rec.kind if hand_present else "none",
        hand_present,
        scroll_direction=getattr(rec, "scroll_direction", ""),
        cursor_pos=(pointer.pos_x, pointer.pos_y),
    )
    if overlay is not None:
        overlay.set_state(decision.state)

    if decision.action == "scroll":
        real_cursor.scroll(decision.scroll_dy, decision.scroll_dx)
        pointer.reset()
        if overlay is not None:
            overlay.update(pointer.pos_x, pointer.pos_y, True)
        return (pointer.pos_x, pointer.pos_y)
    if decision.action == "drag_end":
        real_cursor.mouse_up()
        pointer.reset()
        return (pointer.pos_x, pointer.pos_y)
    if decision.state == "scrolling":
        # Scroll grace frames (hand settling between strokes): hold the
        # cursor still AND re-anchor the pointer to the live OS position,
        # so exiting scroll never jumps from a stale position.
        try:
            pointer.sync_to(*real_cursor.get_position())
        except Exception:
            pointer.reset()
        if overlay is not None:
            overlay.update(pointer.pos_x, pointer.pos_y, True)
        return (pointer.pos_x, pointer.pos_y)
    if decision.state in ("tracking", "dragging") and hand_present:
        c = _palm_center(primary)
        cursor_pos = pointer.update(c)
        if cursor_pos is None:
            return (pointer.pos_x, pointer.pos_y)
        if decision.action == "drag_start":
            real_cursor.move_to(*cursor_pos)
            real_cursor.mouse_down()
        elif decision.state == "dragging":
            real_cursor.dragged_to(*cursor_pos)
        else:
            real_cursor.move_to(*cursor_pos)
        if overlay is not None:
            overlay.update(cursor_pos[0], cursor_pos[1], True)
        if decision.action == "click":
            real_cursor.click("left")
            pointer.freeze(cursor_config.control.click_freeze_ms)
        elif decision.action == "double_click":
            real_cursor.double_click("left")
            pointer.freeze(cursor_config.control.click_freeze_ms)
        return cursor_pos
    pointer.reset()
    if overlay is not None:
        overlay.hide()
    return (pointer.pos_x, pointer.pos_y)


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)

    if args.list_cameras:
        try:
            from axis.camera import list_cameras

            for name in list_cameras():
                print(name)
        except RuntimeError as e:
            print(f"Error: {e}")
            return 1
        return 0

    # --- camera ---------------------------------------------------------
    # "0" → integer index; other strings → resolve name to index via
    # AVFoundation (so "iPhone Camera" works regardless of device order).
    device: str | int = args.camera
    if isinstance(device, str) and device.isdigit():
        device = int(device)
    elif isinstance(device, str):
        from axis.camera import resolve_camera

        resolved = resolve_camera(device)
        if resolved is None:
            print(f"Error: no camera named '{device}'. Use --list-cameras.")
            return 1
        device = resolved
        print(f"Using camera '{args.camera}' (index {device}).")
    camera = Camera(device=device)
    try:
        camera.ensure_permission()
    except CameraError as e:
        print(f"Error: {e}")
        return 1
    camera.open()

    # --- landmarks ------------------------------------------------------
    extractor = HandLandmarker()

    # --- gesture engine -------------------------------------------------
    registry = build_registry()
    motions = load_user_motions()
    recognizer = GestureRecognizer(registry, motions=motions)

    view = DebugView() if args.debug else None
    click_queue: "queue.Queue[tuple[int, int]]" = queue.Queue()

    def _on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            click_queue.put((x, y))

    # Use a resize-free window so image rect == window content rect and
    # mouse coords map 1:1 without Retina/aspect compensation.
    if view is not None:
        cv2.namedWindow(view.window_name, cv2.WINDOW_AUTOSIZE)
        cv2.imshow(view.window_name, np.zeros((10, 10, 3), dtype=np.uint8))
        cv2.setMouseCallback(view.window_name, _on_mouse)

    recorder = MotionRecorder(duration_s=RECORD_DURATION_S)
    prompt_thread: Optional[threading.Thread] = None
    train_gesture_name: Optional[str] = None  # gesture selected in the navbar
    continue_prompt_active = False
    was_pinch = False

    # --- virtual cursor -------------------------------------------------
    from axis.cursor import (
        ControlState,
        RelativePointer,
        VirtualCursor,
        load_cursor_config,
        resolve_cursor_mode,
    )
    from axis.macos.accessibility import ensure_accessibility, print_accessibility_guidance
    from axis.macos.real_cursor import RealCursorController

    cursor_config = load_cursor_config()
    requested_mode = args.cursor_mode or cursor_config.mode
    trusted = ensure_accessibility()
    cursor_mode = resolve_cursor_mode(requested_mode, trusted)
    if requested_mode in ("real", "auto", "control") and not trusted:
        print_accessibility_guidance()
    pointer = RelativePointer(cursor_config, mirror_x=cursor_config.mirror_x)
    try:
        from AppKit import NSScreen

        screens = NSScreen.screens()
        xs = [float(s.frame().origin.x) for s in screens]
        ys = [float(s.frame().origin.y) for s in screens]
        ws = [float(s.frame().size.width) for s in screens]
        hs = [float(s.frame().size.height) for s in screens]
        pointer.set_screen(max(x + w for x, w in zip(xs, ws)),
                           max(y + h for y, h in zip(ys, hs)))
    except Exception:
        try:
            from AppKit import NSScreen

            s = NSScreen.mainScreen().frame()
            pointer.set_screen(s.size.width, s.size.height)
        except Exception:
            pass  # keep pointer defaults
    overlay = VirtualCursor(cursor_config)
    overlay_on = args.cursor and overlay.start()
    real_cursor: Optional[RealCursorController] = None
    control: Optional[ControlState] = None
    if args.cursor and cursor_mode == "control":
        real_cursor = RealCursorController()
        try:
            pointer.sync_to(*real_cursor.get_position())
        except Exception:
            pass
        control = ControlState(cursor_config.control)
        overlay.set_style("ring")  # marker over the real cursor
        print("Cursor mode: control (hand drives the real macOS cursor)")
    elif args.cursor and cursor_mode == "real":
        real_cursor = RealCursorController()
        overlay.set_style("ring")  # marker over the real cursor
        print("Cursor mode: real (moves the actual macOS cursor)")
    elif overlay_on:
        print("Cursor mode: avatar (click-through on-screen arrow)")
    else:
        print("Cursor overlay: disabled")

    # --- session gate (inactivity timeout) ------------------------------
    from axis.session.hotkey import HotkeyListener, SessionGate

    gate = SessionGate(timeout_s=args.timeout)
    hotkey: Optional[HotkeyListener] = None
    if args.hotkey:
        hotkey = HotkeyListener(keycode=int(args.hotkey))
        hotkey.on_toggle = lambda: (
            gate.force_on() if not gate.active else gate.force_off()
        )
        if hotkey.start():
            print(f"Global hotkey enabled (keycode {args.hotkey}).")
        else:
            hotkey = None
            print("Could not install the global hotkey event tap.")

    print("Axis running. Press Ctrl+C to stop.")
    print(f"Hand tracking: {'on' if args.debug else 'headless'}")

    try:
        while True:
            frame = camera.read()
            if frame is None:
                continue

            hand_data = extractor.process(frame)
            hands = hand_data.hands if hand_data is not None else []
            primary = hand_data.primary if hand_data is not None else None
            rec = recognizer.recognize(hands)

            # Session gate: note hand activity; the virtual cursor hides
            # when the session has been inactive for the timeout.
            if primary is not None:
                gate.note_hand()
            session_active = gate.active
            cursor_pos = None
            if overlay_on or real_cursor is not None:
                if session_active and primary is not None:
                    if control is not None and real_cursor is not None:
                        cursor_pos = _control_frame(
                            control, pointer, real_cursor, overlay,
                            rec, primary, cursor_config,
                        )
                    else:
                        # Palm center (normalized) → relative pointer.
                        lm = primary.landmarks
                        c = (
                            float(np.mean(lm[[0, 5, 9, 13, 17], :2], axis=0)[0]),
                            float(np.mean(lm[[0, 5, 9, 13, 17], :2], axis=0)[1]),
                        )
                        cursor_pos = pointer.update(c)
                        if real_cursor is not None:
                            real_cursor.move_to(*cursor_pos)
                        if overlay_on:
                            overlay.update(cursor_pos[0], cursor_pos[1], True)
                else:
                    pointer.reset()
                    if control is not None:
                        control.update("none", False)
                        if overlay_on:
                            overlay.set_state("clutched")
                    if overlay_on:
                        overlay.hide()
            if overlay_on:
                overlay.pump()

            # Optional pinch-to-click (real mode only, edge-triggered;
            # control mode handles its own debounced clicks above).
            if args.clicks and real_cursor is not None and control is None:
                is_pinch = rec.kind == "pinch"
                if is_pinch and not was_pinch:
                    real_cursor.click("left")
                was_pinch = is_pinch

            # Headless progress line.
            if not args.debug and (prompt_thread is None or not prompt_thread.is_alive()):
                state = (
                    f"{rec.label}"
                    if primary is not None
                    else "No hand in view — move your hand into frame"
                )
                print(f"\r[{state}]" + " " * 20, end="", flush=True)

            # Process navbar clicks (dropdown / TRAIN-STOP / close).
            while not click_queue.empty():
                cx, cy = click_queue.get_nowait()
                if view is not None:
                    cx, cy = view.map_coords(cx, cy)
                cmd = view.handle_click(cx, cy) if view is not None else None
                if cmd == "close":
                    raise KeyboardInterrupt
                if cmd == "toggle_train":
                    if view.training:
                        # TRAIN pressed: start a recording window NOW.
                        train_gesture_name = view.selected_gesture
                        recorder.start()
                        print(f"\n[Recording '{view.selected_gesture}' — window 1]")
                    else:
                        # STOP pressed: end the training loop.
                        recorder.cancel()
                        continue_prompt_active = False
                        print("\n[Training stopped]")
                if cmd and cmd.startswith("select_gesture:"):
                    train_gesture_name = cmd.split(":", 1)[1]

            # --- state machine: train loop / classify / normal ----------
            if view is not None and view.training and recorder.active:
                # A 20s recording window in progress.
                recorder.update(primary)
                if recorder.done:
                    # Save this window's variations, then either continue
                    # to the next window or stop.
                    variations = recorder.finish_variations()
                    view.record_count += 1
                    if variations:
                        for i, v in enumerate(variations):
                            gid = f"{_slugify(train_gesture_name or 'custom')}_{view.record_count}_{i}"
                            template = v.to_template(
                                name=train_gesture_name or "Custom", gesture_id=gid
                            )
                            motions.add(template)
                            save_user_motion(template)
                        print(
                            f"\n[Window {view.record_count}: saved {len(variations)} "
                            f"variation(s) for '{train_gesture_name or 'Custom'}']"
                        )
                    else:
                        print(
                            f"\n[Window {view.record_count}: no valid motion captured]"
                        )
                    # Every 3 windows (~60s), ask to continue.
                    if view.record_count % 3 == 0 and not continue_prompt_active:
                        continue_prompt_active = True
                        view._continue_done = False
                        t = threading.Thread(
                            target=_continue_worker,
                            args=(view,), daemon=True,
                        )
                        t.start()
                    else:
                        recorder.start()  # next window
            elif continue_prompt_active:
                # Waiting on the "continue?" input thread.
                if getattr(view, "_continue_done", False):
                    continue_prompt_active = False
                    if view.training:
                        recorder.start()  # user said yes — next window
                    # else: worker already stopped training
                pass
            elif view is not None and view.training and not recorder.active:
                # Train armed but not recording (after STOP or before start).
                pass
            elif rec.kind == "unknown" and rec.registration_candidate is not None:
                # Static unknown pose → old-style name prompt (threaded).
                if prompt_thread is None or not prompt_thread.is_alive():
                    prompt_thread = threading.Thread(
                        target=_registration_worker, args=(rec,), daemon=True
                    )
                    prompt_thread.start()

            # --- render -------------------------------------------------
            if view is not None:
                view.show(
                    frame,
                    rec,
                    hand_data,
                    recorder=recorder if recorder.active else None,
                    cursor_pos=cursor_pos,
                )

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if view is not None:
                # Keyboard fallback for gesture selection (the mouse may not
                # map perfectly on Retina displays):
                #   [  previous gesture   ]  next gesture
                #   1..7  jump to gesture by number (1 = Scroll Down ...)
                if key == ord("["):
                    view.cycle_gesture(-1)
                    train_gesture_name = view.selected_gesture
                elif key == ord("]"):
                    view.cycle_gesture(1)
                    train_gesture_name = view.selected_gesture
                elif ord("1") <= key <= ord("9"):
                    idx = key - ord("1")
                    if view.select_gesture_by_index(idx) is not None:
                        train_gesture_name = view.selected_gesture
            if recorder.active and key == 27:  # ESC cancels recording
                recorder.cancel()
                print("\nRecording cancelled.")
    except KeyboardInterrupt:
        pass
    finally:
        if overlay_on:
            overlay.stop()
        if real_cursor is not None:
            if control is not None and control.button_held:
                try:
                    real_cursor.mouse_up()
                except Exception:
                    pass
            real_cursor.stop()
        if hotkey is not None:
            hotkey.stop()
        if view is not None:
            view.close()
        extractor.close()
        camera.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
