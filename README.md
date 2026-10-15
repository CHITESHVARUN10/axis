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
python main.py --cursor-mode control      # hand drives the REAL system cursor: move + click + right-click + drag + scroll
python main.py --clicks                   # pinch → left-click at the cursor (real mode)
python main.py --hands 2                  # track both hands (default 1 — faster)
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
- **Control mode** (`--cursor-mode control`, the new `auto` default with Accessibility): your hand drives the **REAL system cursor** — the same one the mouse moves. **Open palm** (all fingers out) moves — hover works in every app, and moving it NEVER scrolls. **Two fingers** (index + middle out, pinky folded) + move = proportional pixel scroll in the locked direction. Pinch = click, pinch-hold still ≈0.6s = right click, pinch + move = drag, fist (or lost hand) clutches/freezes. Re-engaging re-anchors to the live OS position, so no cursor jump. Ring tint: green = tracking, yellow = clutched, blue = dragging, purple = scrolling. Every click event pulses a toast in the debug view (Left click / Double click / Right click / Drag), so you always see what the system identified.
- **Precision slowdown**: as your fingers approach a pinch the cursor gain drops to ~25%, so clicks land exactly where you aim; the speed returns the instant the hand opens again.
- **Relative** movement: small hand motions move the cursor; no absolute hand→screen mapping, no large arm sweeps; hand stops → cursor stops (dead zone + One Euro filter). **Speed follows your hand**: slow drifts stay near base gain (pixel-precise), fast swipes accelerate up to a per-frame cap (no teleporting). Lift and re-center like a trackpad finger to keep going; the cursor reaches the screen edges (Dock/menu bar).
- **Accessibility**: enable it in System Settings → Privacy & Security → Accessibility, add your terminal app, then relaunch. Without it the avatar arrow is used and guidance is printed once.
- Tuning lives in `axis/config/cursor.json` (mode, gain, acceleration, dead zone, smoothing, arrow size, plus the `control` block: clutch/click/drag/scroll thresholds).
- Gesture→action bindings live in `axis/config/actions.json` (pinch → click-left, pinch-hold-move → drag, scroll → scroll-wheel, fist → clutch).
- The cursor position is mirrored as a small dot on the debug-view frame for diagnostics.

### Gesture differentiation

Scroll and zoom are disambiguated structurally — the engine first decides **palm translation vs. finger articulation**:

- A **translating two-finger palm** is scroll (direction from net displacement, then locked).
- A **stationary palm with changing thumb-index spread** is zoom in/out.
- Static poses (pinch / fist / open palm) come from landmark ratio rules; move is just "open palm + translating".
- Learned embeddings refine the scroll direction label but never decide scroll vs. zoom, never flip a direction, and can never veto a scroll.

Thresholds live in `axis/config/differentiator.json` (path minimum, scroll displacement, zoom slope, margin, hysteresis).

### Performance & memory notes

- MediaPipe tracks **1 hand by default** (`--hands 2` re-enables two-hand recording) — roughly half the inference time and per-frame allocations.
- Frames are converted BGR→RGB into a **reused buffer** before MediaPipe (channel-correct input; no per-frame allocation), and the preview flip writes into a reused buffer too.
- The scroll wheel posts integer pixel events with a float residue carry; Quartz is imported once per process, not per call.
- If the camera stream stalls (iPhone locks or disconnects), Axis tries to reopen every 5s with a printed notice instead of looping silently.

### Using your iPhone as the camera

macOS Continuity Camera exposes your iPhone as a virtual camera (the same one FaceTime/Photo Booth use). Axis can use it:

1. Make sure the iPhone is awake, unlocked, and **Continuity Camera is enabled** (iPhone Settings → General → AirPlay & Continuity → Continuity Camera; on the Mac, the device appears automatically when the iPhone is near).
2. `python main.py --list-cameras` — you'll see e.g. `Chitesh's iphone Camera`.
3. `python main.py --camera "iphone"` — the name is matched case-insensitively (a substring is fine).

The camera index may change if devices reorder, which is why name-based selection is supported.

### Gestures (Phase 1)

| Action | How to do it | Overlay shows |
|---|---|---|
| **Move cursor** | Open palm (all fingers out), drift slowly — approaching a pinch slows the cursor for precision | tracking ring (green) |
| **Left click** | Pinch (index+middle to thumb) and release quickly | "Left click" toast |
| **Double click** | Two pinch-releases within 0.5s at the same spot | "Double click" toast |
| **Right click** | Hold the pinch **still** for ~0.6s (sub-threshold jitter is fine) | "Right click" toast |
| **Drag** | Pinch, keep holding and move ≥ few px at 0.3s — release to drop | "Drag" toast + blue ring |
| **Scroll** | Two fingers (index+middle out, pinky folded) then stroke; travel-proportional, direction locked, both axes | "You are trying to scroll" + arrow (green) |
| **Clutch / freeze** | Fist (or hand out of frame) | "Fist" (yellow ring) |
| **Zoom in/out** | Spread / close the fingers while the palm stays put | "Zooming in/out" + arrows |
| Learned motion | Recorded template | template name (green) |
| Unknown pose | Held ~0.3s | **recording flow** (below) |

**Scroll is pixel-proportional**: the wheel amount per frame equals palm travel × gain (tunable `control.scroll_gain_px`), accumulated as float and rounded with a carry — small strokes scroll a little, fast strokes a lot, and no amount is lost to integer truncation. The direction latches on the stroke and only resets after the hand fully stops, so up stays up.

**Zoom is continuous**: thumb-index distance over time. Pinch fingers together = zoom out, spread apart = zoom in. It fires once per crossing (a held pinch is a click, not a zoom).

### Two-hand tracking

- MediaPipe tracks **1 hand by default** (`--hands 2` for simultaneous two-hand recording — templates still train per hand).
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
