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
import sys
import threading
from typing import Optional

import cv2
import numpy as np

from axis.camera import Camera, CameraError
from axis.gestures import (
    GestureRecognizer,
    GestureRegistry,
    MotionRecorder,
    MotionRegistry,
    Recognition,
    finger_pattern_name,
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

MENU_OPTIONS = ["Scroll Down", "Scroll Up", "Scroll Left", "Scroll Right", "Zoom", "Name it myself", "Discard"]


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


def _classify_worker(
    rec: MotionRecorder, motions: MotionRegistry, registry: GestureRegistry
) -> None:
    """After recording, offer to save captured variations as named templates.

    The 20s recording window captures multiple strokes (variations) — e.g.
    scroll-down performed at different angles or with different finger
    styles. Each variation is stored as its own template so all of them are
    recognized later.
    """
    variations = rec.finish_variations()
    if not variations:
        print("\nNo valid motion captured — discarding.")
        return

    print(f"\nCaptured {len(variations)} motion variation(s):")
    for i, v in enumerate(variations):
        print(
            f"  {i + 1}. direction={v.direction}, fingers="
            f"{finger_pattern_name(v.finger_mask)}, speed={v.speed:.2f}"
        )

    print("\nClassify these gestures:")
    for i, opt in enumerate(MENU_OPTIONS):
        print(f"  {i + 1}. {opt}")
    try:
        choice = input("Choose (1-7): ").strip()
    except (EOFError, KeyboardInterrupt):
        choice = ""

    # Map the choice to a base name for all variations.
    name = None
    if choice == "1":
        name = "Scroll Down"
    elif choice == "2":
        name = "Scroll Up"
    elif choice == "3":
        name = "Scroll Left"
    elif choice == "4":
        name = "Scroll Right"
    elif choice == "5":
        name = "Zoom"
    elif choice == "6":
        name = _prompt_name("Name these gestures: ")
    # choice 7 / invalid → discard

    if name:
        for i, v in enumerate(variations):
            gid = f"{_slugify(name)}_{i}"
            template = v.to_template(name=name, gesture_id=gid)
            motions.add(template)
            save_user_motion(template)
        print(
            f"Saved {len(variations)} motion template(s) for '{name}' — "
            "recognized in future sessions."
        )


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
    return parser.parse_args(argv)


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
    recorder = MotionRecorder(duration_s=RECORD_DURATION_S)
    menu_options: Optional[list[str]] = None
    menu_title = ""
    prompt_thread: Optional[threading.Thread] = None
    confirmation_thread: Optional[threading.Thread] = None
    # Gate: only trigger the record flow when we've seen a real, sustained
    # "unknown" motion — not noise, not a held static pose.
    unknown_motion_frames = 0

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

            # Headless progress line.
            if not args.debug and (prompt_thread is None or not prompt_thread.is_alive()):
                state = (
                    f"{rec.label}"
                    if primary is not None
                    else "No hand in view — move your hand into frame"
                )
                print(f"\r[{state}]" + " " * 20, end="", flush=True)

            # --- state machine: normal → confirm → record → classify ----
            if recorder.active:
                # Recording in progress: feed frames to the recorder.
                recorder.update(primary)
                if recorder.done:
                    recorder.cancel()  # stop; classify thread takes over
                    menu_options = MENU_OPTIONS
                    menu_title = "What did you mean by that gesture?"
                    confirmation_thread = threading.Thread(
                        target=_classify_worker,
                        args=(recorder, motions, registry),
                        daemon=True,
                    )
                    confirmation_thread.start()
            elif confirmation_thread is not None and confirmation_thread.is_alive():
                # Classification prompt active — camera keeps streaming.
                pass
            elif menu_options is not None and (
                confirmation_thread is None or not confirmation_thread.is_alive()
            ):
                # Classification finished; clear the menu.
                menu_options = None
                menu_title = ""
                confirmation_thread = None
            elif rec.kind in ("none", "unknown") and rec.registration_candidate is None:
                # Unrecognized *motion*: the recognizer returns "none" while
                # the hand is moving in a way no template matches. Start the
                # confirm → record flow after a few frames of sustained
                # unknown motion (avoid noise).
                if primary is not None:
                    unknown_motion_frames += 1
                    if unknown_motion_frames >= 5 and not recorder.active:
                        unknown_motion_frames = 0
                        recorder.start()
                        print("\n[Recording unrecognized motion — perform it a few times]")
            else:
                unknown_motion_frames = 0
                # Static unknown pose → old-style name prompt (threaded).
                if rec.kind == "unknown" and rec.registration_candidate is not None:
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
                    menu_options=menu_options,
                    menu_title=menu_title,
                )

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if recorder.active and key == 27:  # ESC cancels recording
                recorder.cancel()
                print("\nRecording cancelled.")
    except KeyboardInterrupt:
        pass
    finally:
        if view is not None:
            view.close()
        extractor.close()
        camera.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
