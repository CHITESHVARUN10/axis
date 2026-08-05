# Axis

Open-source, privacy-first gesture interaction layer for macOS. Control your computer with hand gestures via webcam — fully on-device, no cloud. See [PRD.md](PRD.md) for the full product spec.

**Status:** Phase 1 (dev harness) — camera access, hand tracking, pinch + scroll recognition, and a **learning flow** that records unrecognized motions and turns them into recognized gestures. No real OS actions yet.

**License:** [MIT](LICENSE)

## Requirements

- macOS
- Python 3.11 (required — MediaPipe has no wheels for 3.12+; use Homebrew: `brew install python@3.11`)

## Setup

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

The hand landmarker model (~8 MB) downloads automatically on first run to `~/.axis/models/`.

## Run

```bash
python main.py            # debug view ON by default
python main.py --no-debug # headless — recognition runs, no window
python main.py --camera "iPhone Camera"   # use the iPhone as the camera
python main.py --list-cameras             # show available camera devices
```

- The debug view shows the live feed, hand bounding boxes (both hands), landmarks, and status overlay.
- Camera is captured at **640×480** with EMA landmark smoothing — reduced jitter and lag.
- Press **`q`** or **Ctrl+C** to quit; **ESC** cancels a recording.
- Camera permission: macOS may attribute access to your terminal app. If denied, enable Camera in System Settings → Privacy & Security → Camera for your terminal, then restart.

### Using your iPhone as the camera

macOS Continuity Camera exposes your iPhone as a virtual camera (the same one FaceTime/Photo Booth use). Axis can use it:

1. Make sure the iPhone is awake, unlocked, and **Continuity Camera is enabled** (iPhone Settings → General → AirPlay & Continuity → Continuity Camera; on the Mac, the device appears automatically when the iPhone is near).
2. `python main.py --list-cameras` — you'll see e.g. `Chitesh's iphone Camera`.
3. `python main.py --camera "iphone"` — the name is matched case-insensitively (a substring is fine).

The camera index may change if devices reorder, which is why name-based selection is supported.

### Gestures (Phase 1)

| Gesture | Detected by | Overlay shows |
|---|---|---|
| Pinch (index+middle to thumb) | static rule | "You are trying to click" (green) |
| Scroll (hand moving, any direction) | stroke latch | "You are trying to scroll" + arrow (green) |
| Zoom in (spread fingers) | continuous thumb-index distance | "Zooming in" + diverging arrows |
| Zoom out (pinch fingers) | continuous thumb-index distance | "Zooming out" + converging arrows |
| Open palm (held still) | static rule | "Open palm" (yellow) |
| Fist (closed hand) | static rule | "Fist" (yellow) |
| Learned motion (Scroll Down/Up/Left/Right) | recorded template | template name (green) |
| Unknown motion | — | **recording flow** (below) |

**Scroll direction is stroke-latched**: the app accumulates displacement and only declares a direction after a threshold is crossed, then holds it until the hand stops. This eliminates the up/down (and left/right) flapping — a downward stroke with a small upward wobble stays "down".

**Zoom is continuous**: thumb-index distance over time. Pinch fingers together = zoom out, spread apart = zoom in. It fires once per crossing (a held pinch is a click, not a zoom).

### Two-hand tracking

- Both hands tracked simultaneously (MediaPipe `num_hands=2`).
- Static gestures classify the **primary** (highest-confidence) hand.
- Scroll fires on either hand, but only an **extended** hand (index+middle out) — a moving fist/pinch never scrolls. This covers full-palm, two-finger, three-finger, and angled scrolls.
- Lowered detection/tracking confidence thresholds (0.3) so hands re-acquire faster after leaving/returning the frame, and smaller/different hands (e.g. your sister's) are detected more readily.

### Learning flow (when a motion isn't recognized)

The heart of the dev harness. When you move your hand in a way no template matches:

1. The app shows **"Recording — perform the gesture a few times"** with a progress bar and a live motion trail.
2. It records **20 seconds** (ESC to cancel).
3. A menu appears: **1. Scroll Down, 2. Scroll Up, 3. Scroll Left, 4. Scroll Right, 5. Zoom, 6. Name it myself, 7. Discard**.
4. The recording is **split into strokes** — each time you perform the gesture (with a pause between), it becomes its own variation. Every variation (direction + finger pattern + speed) is saved as a separate template to `~/.axis/motions.json`.
5. All variations are recognized in future sessions.

**This is how scroll up vs. down gets trained**: during one 20s window, do "scroll down" a few times at different angles/finger styles (2 fingers, 3 fingers, full palm, tilted). Each becomes a template — and because direction is latched, a downward stroke never flips to "up".

### Registering a static gesture

Hold a deliberate, unknown pose (not pinch/fist/open-palm) for ~0.3s. The app prompts for a name, saving to `~/.axis/gestures.json`.

## Architecture

Modular pipeline — each stage independently replaceable:

```
Camera → Frame Capture → Hand Detection → Landmark Extraction
       → Motion Engine → Gesture Engine → Action Engine → macOS Controller
```

| Stage | Package | Phase 1 status |
|---|---|---|
| Capture | `axis/camera/` | ✅ permission + capture (640×480) |
| Hand detection | `axis/landmarks/` | ✅ MediaPipe HandLandmarker + EMA smoothing |
| Motion engine | `axis/gestures/recorder.py` | ✅ motion capture → templates |
| Gesture engine | `axis/gestures/` | ✅ pinch/scroll/open-palm/fist + motion templates |
| Action engine | `axis/pipeline/action_engine/` | placeholder |
| macOS controller | `axis/pipeline/macos_controller/` | placeholder (Quartz installed, not wired) |

Config:

- `axis/config/gestures.json` — built-in gesture definitions (landmark pair-distance ratios)
- `~/.axis/gestures.json` — user-registered static gestures
- `~/.axis/motions.json` — learned motion templates (scroll down/up, custom)
- `axis/config/actions.json` — Action Registry bindings (Phase 2)

## Dependencies

- `mediapipe==0.10.35` — hand landmark detection. Pinned to 0.10.x: **1.0.0 has a broken `mediapipe.tasks.python` namespace**.
- `opencv-python==4.11.0.86` + `opencv-contrib-python==4.11.0.86` — camera capture, frame processing, debug view.
- `numpy==2.2.6` — mediapipe 0.10.x needs `numpy<2.3`.
- `pyobjc-framework-Quartz` — macOS cursor/click/scroll APIs (Phase 3, not wired).
- `pyobjc-framework-AVFoundation` — camera permission check.

## Tests

```bash
python -m pytest tests/ -v
```

Unit tests cover the gesture engine's classification math with synthetic landmark arrays (no camera needed), including motion-template matching and the recorder's direction capture.

## Known Phase-1 limitations

- Recognition rules are intentionally simple (ratio thresholds + templates, not learned models) — expect tuning as real-hand data comes in.
- Static gestures require 3 consecutive frames (debounce) to fire.
- Scroll uses stroke-latched direction (needs ~0.25 palm-sizes of displacement to declare); micro-movements won't trigger it.
- The classify menu is terminal-based (number input); the camera keeps streaming.
- Camera permission for a CLI is attributed to the host terminal app by macOS; the AVFoundation prompt may no-op under a bare python process. Detection relies on the host having Camera access.

## License

MIT — see [LICENSE](LICENSE).
