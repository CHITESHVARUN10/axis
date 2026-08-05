"""macOS camera permission via AVFoundation.

TCC reality for a CLI dev harness: a bare python process has no
Info.plist usage-description, so `requestAccessForMediaType:` silently
no-ops on modern macOS — the prompt is attributed to the host app
(Terminal/iTerm/IDE) instead. The camera therefore works if the host app
already has Camera permission, and fails cleanly if not.

Strategy:
  - query status (authorized / denied / restricted / not-determined)
  - try the AVFoundation request for at most a few seconds (helps when
    the host app *can* prompt, e.g. under some launch contexts)
  - never block indefinitely: camera.open() is the real test
"""

from __future__ import annotations

import threading
from enum import Enum

# AVAuthorizationStatus
_NOT_DETERMINED = 0
_RESTRICTED = 1
_DENIED = 2
_AUTHORIZED = 3

_REQUEST_TIMEOUT_S = 5.0


class CameraPermission(Enum):
    NOT_DETERMINED = "not_determined"
    DENIED = "denied"
    RESTRICTED = "restricted"
    AUTHORIZED = "authorized"


def _avfoundation():
    try:
        from AVFoundation import AVCaptureDevice, AVMediaTypeVideo
    except ImportError as e:  # pragma: no cover - non-macOS
        raise RuntimeError(
            "PyObjC AVFoundation not available — camera permission check "
            "requires macOS."
        ) from e
    return AVCaptureDevice, AVMediaTypeVideo


def check_camera_permission() -> CameraPermission:
    """Query current camera permission without prompting."""
    AVCaptureDevice, AVMediaTypeVideo = _avfoundation()
    status = AVCaptureDevice.authorizationStatusForMediaType_(AVMediaTypeVideo)
    if status == _AUTHORIZED:
        return CameraPermission.AUTHORIZED
    if status == _DENIED:
        return CameraPermission.DENIED
    if status == _RESTRICTED:
        return CameraPermission.RESTRICTED
    return CameraPermission.NOT_DETERMINED


def request_camera_permission(timeout: float = _REQUEST_TIMEOUT_S) -> CameraPermission:
    """Request camera access, bounded by timeout.

    Returns the resulting status. Under a bare CLI the request usually
    no-ops (no Info.plist usage description) and we return NOT_DETERMINED
    after the timeout — camera.open() is the real access test.
    """
    AVCaptureDevice, AVMediaTypeVideo = _avfoundation()

    done = threading.Event()
    result: list[CameraPermission] = []

    def handler(granted):
        result.append(
            CameraPermission.AUTHORIZED if granted else CameraPermission.DENIED
        )
        done.set()

    AVCaptureDevice.requestAccessForMediaType_completionHandler_(
        AVMediaTypeVideo, handler
    )
    done.wait(timeout)
    if result:
        return result[0]
    return CameraPermission.NOT_DETERMINED


__all__ = [
    "CameraPermission",
    "check_camera_permission",
    "request_camera_permission",
]
