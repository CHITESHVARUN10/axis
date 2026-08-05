"""Action engine — resolve recognized gestures to actions via the Action Registry.

Interface (implemented in Phase 2+):
    resolve(gesture_id) -> Action | None
    Action: handler id + parameters

Mappings live in config (axis/config/actions), never in code.
"""
