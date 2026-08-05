"""Camera device discovery via AVFoundation.

Lists available camera devices by localized name (e.g. "FaceTime HD
Camera", "iPhone Camera" — the Continuity Camera virtual device). OpenCV's
AVFoundation backend can open devices by index; name→index resolution is
done here so `--camera "iPhone Camera"` works reliably regardless of
device ordering.
"""

from __future__ import annotations

from typing import List, Optional


def _av_devices():
    from AVFoundation import AVCaptureDevice, AVMediaTypeVideo

    return AVCaptureDevice.devicesWithMediaType_(AVMediaTypeVideo) or []


def list_cameras() -> List[str]:
    """Return localized names of all video capture devices."""
    names = []
    for d in _av_devices():
        try:
            names.append(str(d.localizedName()))
        except Exception:
            continue
    return names


def resolve_camera(name: str) -> Optional[int]:
    """Resolve a device name to its AVFoundation enumeration index.

    Returns None if no device matches (case-insensitive substring match).
    OpenCV and AVFoundation enumerate devices in the same order, so the
    index works for cv2.VideoCapture(index).
    """
    wanted = name.lower()
    for i, d in enumerate(_av_devices()):
        try:
            if wanted in str(d.localizedName()).lower():
                return i
        except Exception:
            continue
    return None


__all__ = ["list_cameras", "resolve_camera"]
