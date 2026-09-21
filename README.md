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
python main.py --no-cursor                # disable the cursor overlay + real cursor
python main.py --cursor-mode avatar       # arrow-only, click-through (no permission)
python main.py --cursor-mode real         # move the actual macOS cursor (Accessibility)
python main.py --cursor-mode control      # hand drives the REAL system cursor: move + click + drag + scroll
python main.py --clicks                   # pinch → left-click at the cursor (real mode)
python main.py --timeout 60               # inactivity timeout in seconds
python main.py --hotkey 97                # global hotkey (keycode) toggles Gesture Mode
```

- The debug view shows the live feed, hand bounding boxes (both hands), landmarks, and status overlay.
- Camera is captured at **640×480** with EMA landmark smoothing — reduced jitter and lag.
- Press **`q`** or **Ctrl+C** to quit; **ESC** cancels a recording; **`t`** steps through the in-app gesture tutorial (or click the **?** button in the navbar).
- Camera permission: macOS may attribute access to your terminal app. If denied, enable Camera in System Settings → Privacy & Security → Camera for your terminal, then restart.

### Virtual cursor (trackpad-style)

Axis shows a **second cursor** on the whole screen, controlled by hand motion, trackpad-style:

```bash
python main.py                     # auto mode: control if Accessibility granted, else avatar arrow
python main.py --cursor-mode avatar # arrow-only, click-through, no permission needed
python main.py --cursor-mode real   # moves the ACTUAL macOS cursor (needs Accessibility)
python main.py --cursor-mode control # gesture-gated REAL system cursor: move + click + drag + scroll
python main.py --clicks             # pinch → left-click at the cursor (real mode; control always clicks)
```

- **Avatar cursor** (default without Accessibility): a macOS-style **arrow** (with a green ring when tracking, yellow when idle) roams the screen in a borderless, always-on-top, **click-through** window. It coexists with your real mouse cursor and stays visible even when the Axis camera window is minimized.
- **Real cursor mode** (`--cursor-mode real` or `auto` with Accessibility granted): the hand moves the **actual macOS pointer** via Quartz — apps see hover states, and the optional `--clicks` flag turns a pinch into a left-click at the cursor. A small green ring marks the hand cursor.
- **Control mode** (`--cursor-mode control`, the new `auto` default with Accessibility): your hand drives the **REAL system cursor** — the same one the mouse moves. **Open palm** (all fingers out) moves — hover works in every app, and moving it NEVER scrolls. **Two fingers** (index + middle out, pinky folded) + move = scroll, held strokes keep scrolling the latched direction (up stays up). Pinch clicks (debounced, double-pinch = double-click), pinch-hold + move drags, fist (or lost hand) clutches/freezes. Re-engaging re-anchors to the live OS position, so no cursor jump. Ring tint: green = tracking, yellow = clutched, blue = dragging, purple = scrolling.
- **Relative** movement: small hand motions move the cursor; no absolute hand→screen mapping, no large arm sweeps; hand stops → cursor stops (dead zone + One Euro filter). **Speed follows your hand**: slow drifts stay near base gain (pixel-precise), fast swipes accelerate up to a per-frame cap (no teleporting). Lift and re-center like a trackpad finger to keep going; the cursor reaches the screen edges (Dock/menu bar).
- **Accessibility**: enable it in System Settings → Privacy & Security → Accessibility, add your terminal app, then relaunch. Without it the avatar arrow is used and guidance is printed once.
- Tuning lives in `axis/config/cursor.json` (mode, gain, acceleration, dead zone, smoothing, arrow size, plus the `control` block: clutch/click/drag/scroll thresholds).
- Gesture→action bindings live in `axis/config/actions.json` (pinch → click-left, pinch-hold-move → drag, scroll → scroll-wheel, fist → clutch).
- The cursor position is mirrored as a small dot on the debug-view frame for diagnostics.

### Gesture differentiation

Scroll and zoom are disambiguated structurally — the engine first decides **palm translation vs. finger articulation**:

- A **translating palm** is scroll (direction from net displacement).
- A **stationary palm with changing thumb-index spread** is zoom in/out.
- Learned embeddings refine the scroll direction (up/down/left/right) but never decide scroll vs. zoom.

Thresholds live in `axis/config/differentiator.json` (path minimum, scroll displacement, zoom slope, margin, hysteresis).

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
| Scroll (**two fingers**: index+middle out, pinky folded, then move) | stroke latch + direction lock | "You are trying to scroll" + arrow (green) |
| Move (**open palm**, all fingers out, then move) | tracking | cursor moves, never scrolls |
| Zoom in (spread fingers) | continuous thumb-index distance | "Zooming in" + diverging arrows |
| Zoom out (pinch fingers) | continuous thumb-index distance | "Zooming out" + converging arrows |
| Open palm (held still) | static rule | "Open palm" (yellow) |
| Fist (closed hand) | static rule | "Fist" (yellow) |
| Learned motion (Scroll Down/Up/Left/Right) | recorded template | template name (green) |
| Unknown motion | — | **recording flow** (below) |

**Scroll direction is stroke-latched and locked**: the app accumulates displacement and only declares a direction after a threshold is crossed, then holds it until the hand fully stops — up stays up, down stays down. A held two-finger stroke keeps scrolling (faster strokes scroll further). Learned embeddings only refine the label; they can never flip the direction or veto a scroll, and they ignore where the stroke started.

**Zoom is continuous**: thumb-index distance over time. Pinch fingers together = zoom out, spread apart = zoom in. It fires once per crossing (a held pinch is a click, not a zoom).

### Two-hand tracking

- Both hands tracked simultaneously (MediaPipe `num_hands=2`).
- Static gestures classify the **primary** (highest-confidence) hand.
- Scroll fires on a **two-finger** hand only (index+middle out, pinky folded; thumb/ring ignored as noisy) — a moving open palm moves the cursor and never scrolls; a moving fist/pinch never scrolls. This covers two-finger up/down/left/right strokes.
- The preview is a **selfie view** (mirrored like Photo Booth) and the cursor follows what you see: move your hand right → cursor goes right. If it ever feels inverted, set `"mirror_x": false` in `axis/config/cursor.json`.
- Lowered detection/tracking confidence thresholds (0.3) so hands re-acquire faster after leaving/returning the frame, and smaller/different hands (e.g. your sister's) are detected more readily.

### Learning flow (navbar-driven training)

The debug view has a **navbar** with clickable buttons:

- **Gesture selector** (first button, e.g. `Scroll Down [+]`) — click to open a **dropdown** listing all gestures: `Scroll Down`, `Scroll Up`, `Scroll Left`, `Scroll Right`, `Zoom In`, `Zoom Out`, `Custom`. Click an item to select it. Works exactly like an HTML `<select>` — tap to open, tap to choose.
- **TRAIN / STOP** — starts / stops a training session.
- **REC n** — shows how many 20-second windows have been recorded in this session.
- **HAND: YES/NO** — live indicator that the hand is being tracked.
- **?** — in-app gesture tutorial (or press **`t`**). Walks through move → click → scroll → drag → clutch → Dock; steps auto-advance when the gesture is detected, `t` steps manually.
- **X** — close the app.

**To train a gesture:**

1. Click the gesture selector → pick your target (e.g. **Scroll Down**) from the dropdown.
2. Click **TRAIN** — a 20-second recording window starts **immediately** (nothing is automatic; the button becomes **STOP**).
3. Perform the gesture during the window, varying angle and finger style (2 fingers, 3 fingers, full palm, tilted).
4. When the 20s window ends, the recording is **split into strokes** — each repetition becomes a variation, converted to a **path embedding** (fixed-length, scale-invariant) and saved to `~/.axis/motions.json` under the selected gesture. The raw recording is discarded.
5. The next 20-second window starts automatically. This repeats **every 20 seconds until you press STOP**.
6. **Every 3 windows (≈60 seconds)** the app asks in the terminal: `Continue recording this gesture? [y/N]`. Answer `y` to keep going, `n` (or Enter) to stop.

Because direction is latched and the embedding is shape-aware, a "Scroll Down" template won't match an upward stroke — the up/down confusion is structurally eliminated.

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
| Action engine | `axis/pipeline/action_engine/` | placeholder; control-mode bindings in `axis/config/actions.json` |
| macOS controller | `axis/pipeline/macos_controller/` | partial (Quartz move/click/drag/scroll via `axis/macos/real_cursor.py`) |

Config:

- `axis/config/gestures.json` — built-in gesture definitions (landmark pair-distance ratios)
- `~/.axis/gestures.json` — user-registered static gestures
- `~/.axis/motions.json` — learned motion templates (scroll down/up, custom)
- `axis/config/actions.json` — Action Registry bindings (pinch → click-left, pinch-hold-move → drag, scroll → scroll-wheel, fist → clutch)

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
