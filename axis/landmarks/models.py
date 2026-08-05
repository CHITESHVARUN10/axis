"""MediaPipe model asset management.

Downloads the hand landmarker model once and caches it under
~/.axis/models/. All model assets live under this directory so the app
stays self-contained and never re-downloads on every run.
"""

from __future__ import annotations

import os
import shutil
import urllib.request
from pathlib import Path

HAND_LANDMARKER_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
)
HAND_LANDMARKER_FILENAME = "hand_landmarker.task"


def models_dir() -> Path:
    """Return the per-user model cache directory, creating it if needed."""
    base = Path(
        os.environ.get("AXIS_HOME", Path.home() / ".axis")
    )
    d = base / "models"
    d.mkdir(parents=True, exist_ok=True)
    return d


def ensure_hand_landmarker(directory: Path | None = None) -> Path:
    """Ensure the hand landmarker model file exists; download if missing.

    Returns the path to the .task file. Raises RuntimeError on download
    failure with a message that points the user at manual setup.
    """
    d = models_dir() if directory is None else Path(directory)
    d.mkdir(parents=True, exist_ok=True)
    target = d / HAND_LANDMARKER_FILENAME
    if target.exists():
        return target

    tmp = target.with_suffix(".task.part")
    try:
        print(f"Downloading hand landmarker model ({HAND_LANDMARKER_URL})…")
        with urllib.request.urlopen(HAND_LANDMARKER_URL, timeout=60) as resp, \
                open(tmp, "wb") as out:
            shutil.copyfileobj(resp, out)
        os.replace(tmp, target)
    except Exception as e:
        if tmp.exists():
            tmp.unlink(missing_ok=True)
        raise RuntimeError(
            f"Could not download the hand landmarker model to {target}: {e}. "
            f"Download it manually from {HAND_LANDMARKER_URL} and place it "
            f"at {target}."
        ) from e
    return target
