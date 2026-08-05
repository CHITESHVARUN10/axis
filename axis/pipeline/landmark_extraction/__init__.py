"""Landmark extraction stage — 21 MediaPipe-style hand landmarks per hand.

Interface (implemented in Phase 1):
    extract(detection, frame) -> HandLandmarks | None
    HandLandmarks: 21 (x, y, z) points + handedness

Replaceable: change the landmark model without touching gesture logic.
"""
