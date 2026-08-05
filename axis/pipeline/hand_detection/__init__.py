"""Hand detection stage — locate hand(s)/palm in a frame.

Interface (implemented in Phase 1):
    detect(frame: numpy.ndarray) -> list[Detection]
    Detection: bbox, confidence, handedness

Replaceable: swap MediaPipe for a different detector without touching
landmark extraction or downstream stages.
"""
