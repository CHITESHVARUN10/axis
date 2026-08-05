"""Capture stage — camera access, frame capture, permission handling.

Interface (implemented in Phase 1):
    start() -> None
    read() -> numpy.ndarray | None      # BGR frame, or None if unavailable
    stop() -> None
    on_frames(callback) -> None         # push callback, if streaming preferred

Replaceable: swap webcam backend without touching other stages.
"""
