# Product Requirements Document: Axis

**An Open-Source Gesture Interaction Layer for macOS**

| | |
|---|---|
| **Status** | Draft — Phase 1 planning |
| **Owner** | Product Owner (TBD assignment) |
| **Version** | 0.1 |
| **Last Updated** | August 2026 |

> **Naming note:** The source vision document titles the project "Axis" but refers to it as "Access" throughout the body text. This PRD uses **Axis** as the working name — confirm before external-facing use (README, App Store listing, marketing) since the two aren't interchangeable in search/branding.

---

## 1. Executive Summary

Axis is a native macOS application that lets users control their computer using hand gestures, captured via webcam and processed entirely on-device. It is explicitly **not** a mouse replacement — it's an additional input modality that coexists with the mouse and trackpad, activated on demand via hotkey, and shut off automatically when not in use.

The product differentiates itself on three axes: **privacy** (zero cloud processing, camera off by default), **user control** (no AI-driven intent prediction — only gesture recognition), and **native feel** (relative cursor movement, dead zones, smoothing — designed to feel like a trackpad, not a novelty).

---

## 2. Problem Statement

Existing gesture-control tools for desktop OSes tend to fall into one of two failure modes:

1. **Novelty-first products** that try to replace the mouse entirely, are exhausting to use for sustained sessions (absolute positioning requiring large arm movements), and fail on precision tasks.
2. **Cloud-dependent or heavyweight tools** that raise privacy concerns (continuous camera streaming, remote processing) or consume excessive CPU/battery, making them impractical for daily use.

There is no widely-adopted, open-source, privacy-first gesture layer for macOS that treats gestures as a *complement* to existing input methods rather than a replacement — and that gives users full transparency and control over when the camera is active and what actions gestures trigger.

---

## 3. Vision & Goals

### 3.1 Vision Statement
Axis should evolve from "software that moves a mouse with your hand" into a **gesture interaction platform for macOS** — an open, extensible foundation as native and reliable as the keyboard, mouse, or trackpad, built entirely under the user's control.

### 3.2 Goals
- Enable accurate, low-latency gesture control of cursor movement, clicking, dragging, scrolling, window management, Dock interaction, and navigation.
- Keep all processing on-device with no data leaving the machine.
- Make gestures feel like natural extensions of real-world actions (grab → drag, swipe → back/forward, pinch → click).
- Support per-user personalization/calibration without requiring daily recalibration.
- Ship a modular architecture that allows individual components (hand tracking, gesture engine, action engine) to be swapped or extended independently.
- Build toward a plugin system and user-definable gesture-to-action bindings (shortcuts, shell commands, AppleScript, Python scripts).

### 3.3 Non-Goals (v1)
- Replacing the mouse/trackpad as primary pointing device.
- Any form of AI-driven intent prediction ("the user probably wants Chrome").
- Cloud sync, telemetry, or remote processing of any kind.
- Full two-handed interaction design (two-hand *tracking* is supported technically, but two-hand *gesture UX* is out of scope for v1).
- Context-aware, per-app gesture behavior (e.g., different swipe behavior in Safari vs. Finder) — explicitly deferred to Phase 3+.

---

## 4. Target Users

- Power users and developers who are comfortable installing accessibility-permission-gated macOS utilities and want a faster/more ergonomic way to perform repetitive actions (window management, navigation, media control).
- Open-source contributors interested in an extensible gesture/action framework they can build plugins for.
- Users with an interest in accessibility-adjacent tooling, though Axis is **not** initially positioned or validated as an assistive-technology product (see Open Questions).

*(Persona detail — usage frequency, accessibility needs, hardware setup — is not yet defined and should be validated before Phase 2 UX work begins.)*

---

## 5. Core Design Principles

These are non-negotiable constraints that every feature must satisfy:

| Principle | Implication |
|---|---|
| Mouse is never replaced | Gestures are additive; no feature should require abandoning the mouse |
| User remains in complete control | No intent prediction, no autonomous actions — AI only recognizes gestures |
| Privacy first | Camera inactive by default; all computation local; no cloud APIs |
| Lightweight | Minimal RAM/CPU footprint; camera runs only in short, timeout-bound sessions |
| Modular | Every pipeline stage (capture → detection → recognition → action) is independently replaceable |
| Configurable over hardcoded | Gesture → action bindings live in configuration, not code |
| Native feel | Relative cursor movement, dead zones, smoothing — modeled on trackpad ergonomics |

---

## 6. Functional Requirements

### 6.1 Activation & Session Model
- User presses a global hotkey → camera activates → Gesture Mode begins.
- After a configurable inactivity timeout (default: 2 minutes), the camera automatically deactivates.
- Camera must never run continuously in the background outside an active session.

### 6.2 Cursor Operations
- **Relative** cursor movement (trackpad-style, not absolute mapping) — small hand movements produce natural cursor movement without requiring large arm motion.
- Dead-zone filtering to prevent jitter from micro-movements.
- Smoothing (Exponential Moving Average and/or One Euro Filter at launch; Kalman Filter as a future enhancement).
- Cursor acceleration curve, tunable.
- **Precision Mode**: a gesture temporarily reduces cursor sensitivity for fine-grained clicking, then returns to normal speed on release.

### 6.3 Mouse Operations
- Left click, right click, double click
- Drag and drop (grab → drag → release → drop mapping)
- Scroll
- Hover state recognition

### 6.4 Window Management
- Minimize / restore (push-away / pull-toward gestures)
- Close
- Switch between windows
- Move and resize windows

### 6.5 Dock Interaction
- Navigate through Dock items
- Highlight an application
- Launch the selected application
- Switch between running applications

### 6.6 Navigation
- Back / forward (swipe left / swipe right)
- Page and document navigation
- Browser and Finder navigation

### 6.7 System Actions (future scope, not committed for v1)
Mission Control, Launchpad, Notification Center, Spotlight, Desktop switching, Control Center, Volume, Brightness, Media controls.

### 6.8 User-Defined Actions (Phase 3)
Users can bind gestures to: launching applications, Apple Shortcuts, shell commands, Python scripts, AppleScript, and other automation workflows — via the Action Registry (see §8.3).

---

## 7. Gesture System

### 7.1 Gesture Philosophy
Gestures should mimic real-world actions rather than requiring memorization of arbitrary movements:

| Gesture | Action |
|---|---|
| Grab → Drag → Release | Drag and drop |
| Swipe left / right | Back / forward |
| Push away | Minimize |
| Pull toward user | Restore |
| Pinch | Click |

### 7.2 Gesture Categories
- **Static gestures** (v1 focus): open palm, closed fist, pinch, finger combinations — recognized via rule-based logic.
- **Dynamic gestures** (Phase 3): swipe, circle, throw, slap, rotation — require learned models; explicitly flagged as advanced/complex and deferred.

### 7.3 Personalization & Calibration
- One-time onboarding flow captures hand size, motion range, preferred wrist angle, pinch distance, gesture timing, and motion characteristics while the user performs gestures naturally (not "recording" gestures in a technical sense).
- Calibration should not be required daily — one initial pass should generally suffice.
- This is **personalization**, not prediction: the model adapts to how a given user naturally performs known gestures; it does not infer new intent.

### 7.4 Recognition Approach
- Simple/static gestures: deterministic, rule-based recognition.
- Complex/dynamic gestures: learned models, improved via personalization data.
- AI scope is strictly limited to: hand landmark detection, gesture recognition, and personalized gesture adaptation. AI must never predict applications, predict future actions, or take autonomous decisions.

---

## 8. Technical Architecture

### 8.1 Pipeline
```
Camera → Frame Capture → Hand Detection → Landmark Extraction
       → Motion Engine → Gesture Engine → Action Engine → macOS Controller
```
Each stage is independently replaceable to avoid locking the architecture to any single technology choice.

### 8.2 Interaction States
The system operates as a state machine to simplify gesture interpretation:
`Idle → Tracking → Selecting → Dragging → Scrolling → Window Control → Navigation`

### 8.3 Action Registry
Gestures never directly trigger hardcoded behavior. Instead: **Gesture → Action → Action Handler**, with the mapping defined in configuration. This is what allows a gesture like Pinch to mean "left click" today and be rebound to "Launch Raycast" or "Execute Apple Shortcut" later without code changes.

### 8.4 Plugin System (Phase 3)
Future extensibility point allowing third parties to register new gestures, new actions, automation workflows, shortcut integrations, AppleScript, shell commands, and Python scripts.

### 8.5 Tech Stack (proposed direction)
- **Language:** Python
- **Hand tracking:** MediaPipe
- **Cursor control:** Native macOS APIs
- **OS target:** macOS
- **Philosophy:** prototype quickly, keep modular so components can be replaced as better tooling emerges.

*(Note: Python + MediaPipe is a reasonable prototyping choice, but sustained low CPU/RAM usage and macOS Accessibility API integration should be validated early — see Risks.)*

---

## 9. UX Requirements

### 9.1 Visual Feedback
State should be communicated subtly — no sounds, no unnecessary notifications:
- **Green** — gesture recognized
- **Yellow** — tracking active
- **Red** — gesture not recognized

Feedback mechanisms may include cursor color changes, highlight borders around selected objects, and hover indication.

### 9.2 Onboarding
Calibration is framed as "let's learn how you naturally interact" rather than technical gesture recording — e.g., "move the cursor," "click naturally," "scroll naturally," "drag naturally."

---

## 10. Privacy & Security Requirements

- Camera must be inactive by default and only activate during an explicit, hotkey-triggered session.
- Sessions auto-terminate after inactivity timeout.
- No frame data, landmark data, or gesture data may be transmitted off-device.
- No cloud APIs, no telemetry, no remote model inference.
- macOS Accessibility permissions must be requested transparently with clear justification at first launch.

---

## 11. Roadmap

### Phase 1 — Core Interaction
Camera input · MediaPipe integration · relative cursor movement · left/right click · drag and drop · scroll · hotkey activation · Accessibility permissions · cursor smoothing · dead zone.

### Phase 2 — Productivity
Gesture customization · personalized calibration · precision mode · Dock navigation · window management · app launching · configurable shortcuts · improved onboarding · visual feedback refinements.

### Phase 3 — Advanced Features
Dynamic gesture recognition · plugin system · Apple Shortcuts integration · user-defined gesture embeddings · context-aware actions · advanced automation · community plugins.

---

## 12. Success Metrics *(proposed — needs validation)*

These are suggested starting metrics, not yet committed targets:
- Gesture recognition accuracy rate (target TBD, e.g. >95% for static gestures under calibrated conditions)
- False-positive gesture rate during normal (non-gesture) mouse/keyboard use
- CPU/RAM footprint while Gesture Mode is active vs. idle
- Time-to-recalibrate / onboarding completion rate
- Retention: % of users still using Axis weekly after 30 days

---

## 13. Open Questions & Risks

- **Naming:** "Axis" (title) vs. "Access" (body text) — needs a single confirmed name before any public-facing work.
- **Performance:** Python + MediaPipe running continuous frame processing may be CPU/battery-heavier than the "lightweight" principle demands — worth an early prototyping spike to validate against real constraints, with a native (Swift/Vision framework) fallback path considered if needed.
- **Accessibility positioning:** the doc doesn't position this as an assistive-technology product, but the feature set (hands-free control) will likely attract that audience — worth deciding intentionally rather than by default.
- **False positives:** everyday hand movement (talking, gesturing while on a call, eating) near the camera during an active session could trigger unintended actions — needs explicit false-positive mitigation strategy beyond the dead zone.
- **Context-aware actions** (Phase 3) meaningfully increases scope/complexity — worth re-scoping once Phase 1/2 usage data exists.

---

## 14. Out of Scope for v1

- Dynamic gesture recognition
- Plugin system / third-party integrations
- Context-aware, per-app gesture behavior
- Two-handed gesture UX
- System-level actions beyond core mouse/window/navigation (Mission Control, Spotlight, etc.)