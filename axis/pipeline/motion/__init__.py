"""Motion engine — temporal smoothing, dead zones, velocity.

Interface (implemented in Phase 2+):
    update(landmarks) -> MotionState
    MotionState: position, velocity, confidence, frame_timestamp

Also home of the interaction state machine (Idle → Tracking → …).
"""
