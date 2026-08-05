"""Camera access — open the built-in webcam via OpenCV and yield frames.

Phase 1 is a dev harness: the camera opens on launch when run with --debug,
and every captured frame is pushed to registered subscribers.
"""

from __future__ import annotations

import threading
from typing import Callable, Optional

import cv2
import numpy as np

from axis.camera.permissions import (
    CameraPermission,
    check_camera_permission,
    request_camera_permission,
)

FrameCallback = Callable[[np.ndarray], None]


class CameraError(RuntimeError):
    """Raised when the camera cannot be opened or a frame cannot be read."""


class Camera:
    def __init__(
        self,
        device: str | int = 0,
        width: int = 640,
        height: int = 480,
    ):
        """Camera access via OpenCV's AVFoundation backend.

        `device` is either an integer index (0 = default) or a device name
        string (e.g. "iPhone Camera" — the Continuity Camera virtual device).
        """
        self._device: str | int = device
        self._width = width
        self._height = height
        self._cap: Optional[cv2.VideoCapture] = None
        self._subscribers: list[FrameCallback] = []

    # --- lifecycle ------------------------------------------------------

    def ensure_permission(self) -> None:
        """Prepare camera access; raise CameraError if definitively denied.

        A bare CLI has no Info.plist usage description, so the AVFoundation
        request may no-op — the real test is camera.open(). We only hard-fail
        on an explicit DENIED/RESTRICTED status, and otherwise let open()
        determine availability (with a clear error if it fails).
        """
        status = check_camera_permission()
        if status in (CameraPermission.DENIED, CameraPermission.RESTRICTED):
            raise CameraError(
                "Camera permission denied. Enable camera access for your "
                "terminal app in System Settings → Privacy & Security → "
                "Camera, then restart."
            )
        if status == CameraPermission.NOT_DETERMINED:
            status = request_camera_permission()  # may no-op under a CLI
            if status in (CameraPermission.DENIED, CameraPermission.RESTRICTED):
                raise CameraError(
                    "Camera permission denied. Enable camera access for your "
                    "terminal app in System Settings → Privacy & Security → "
                    "Camera, then restart."
                )

    def open(self) -> None:
        """Open the camera device. Call after ensure_permission()."""
        if self._cap is not None:
            return
        self._cap = cv2.VideoCapture(self._device)
        if self._cap.isOpened():
            # Lower resolution → less per-frame compute, less tracking lag.
            self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self._width)
            self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self._height)
        if not self._cap.isOpened():
            self._cap = None
            raise CameraError(
                f"Could not open camera device {self._device!r}. Make "
                "sure no other app is using it, or list available devices "
                "with --list-cameras."
            )

    def read(self) -> Optional[np.ndarray]:
        """Return the next frame (BGR), or None if the camera has failed."""
        if self._cap is None:
            return None
        ok, frame = self._cap.read()
        if not ok or frame is None:
            return None
        return frame

    def close(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    def subscribe(self, callback: FrameCallback) -> None:
        self._subscribers.append(callback)

    # --- convenience ----------------------------------------------------

    def pump(self) -> None:
        """Read one frame and deliver to subscribers. No-op if none."""
        frame = self.read()
        if frame is None:
            return
        for cb in self._subscribers:
            cb(frame)
