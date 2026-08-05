"""Camera subsystem: permission handling + frame capture + device discovery."""

from axis.camera.camera import Camera, CameraError, FrameCallback
from axis.camera.devices import list_cameras, resolve_camera
from axis.camera.permissions import CameraPermission

__all__ = ["Camera", "CameraError", "FrameCallback", "CameraPermission", "list_cameras", "resolve_camera"]
