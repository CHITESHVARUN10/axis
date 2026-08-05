"""User gesture registry store — persisted to the user config dir.

Newly registered gestures are appended to ~/.axis/gestures.json so they
survive across sessions. The file format is the same as the built-in
gestures.json (axis.gestures.v1), so user gestures are indistinguishable
from built-ins downstream.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, List, Optional

from axis.gestures.definitions import (
    GestureDefinition,
    GestureRegistry,
    MotionRegistry,
    MotionTemplate,
)


def user_config_dir() -> Path:
    """Return the per-user config directory (default ~/.axis)."""
    d = Path(os.environ.get("AXIS_HOME", Path.home() / ".axis"))
    d.mkdir(parents=True, exist_ok=True)
    return d


def user_gestures_path() -> Path:
    return user_config_dir() / "gestures.json"


def user_motions_path() -> Path:
    return user_config_dir() / "motions.json"


def load_user_gestures() -> GestureRegistry:
    """Load user gestures; return an empty registry if none exist yet."""
    path = user_gestures_path()
    if not path.exists():
        return GestureRegistry(definitions={})
    try:
        return GestureRegistry.load(path)
    except (json.JSONDecodeError, KeyError, OSError):
        # Corrupt file — don't crash the app over a config file.
        return GestureRegistry(definitions={})


def save_user_gesture(gesture: GestureDefinition) -> None:
    """Append a newly registered gesture to the user registry."""
    path = user_gestures_path()
    registry = load_user_gestures()
    # Avoid duplicate ids.
    defs = {g.id: g for g in registry.definitions.values()}
    if gesture.id in defs:
        gesture.id = gesture.id + "_" + str(len(defs))
    defs[gesture.id] = gesture
    registry.definitions = defs
    registry.save(path)


def delete_user_gesture(gesture_id: str) -> None:
    """Remove a user gesture by id. No-op if not present."""
    path = user_gestures_path()
    registry = load_user_gestures()
    if gesture_id in registry.definitions:
        del registry.definitions[gesture_id]
        registry.save(path)


# --- motion templates ----------------------------------------------------

def load_user_motions() -> MotionRegistry:
    """Load user motion templates; empty registry if none exist."""
    path = user_motions_path()
    if not path.exists():
        return MotionRegistry()
    try:
        return MotionRegistry.load(path)
    except (json.JSONDecodeError, KeyError, OSError):
        return MotionRegistry()


def save_user_motion(template: MotionTemplate) -> None:
    """Persist a learned motion template."""
    path = user_motions_path()
    registry = load_user_motions()
    registry.add(template)
    registry.save(path)
