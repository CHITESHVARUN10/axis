"""Tests for the camera layer + extractor input handling (cv2 mocked)."""

from unittest import mock

import numpy as np
import pytest

from axis.camera import Camera, CameraError


def _fake_cap(opened):
    cap = mock.Mock()
    cap.isOpened.return_value = opened
    cap.read.return_value = (opened, None)
    cap.set.return_value = True
    cap.release = mock.Mock()
    return cap


def test_reopen_reopens_after_stall():
    """A stalled stream must be recoverable: close() then open()."""
    with mock.patch("cv2.VideoCapture") as vc:
        caps = [_fake_cap(True), _fake_cap(True)]
        vc.side_effect = caps
        cam = Camera(device=0)
        cam.open()
        first = caps[0]
        cam.reopen()
        first.release.assert_called_once()
        assert cam._cap is caps[1]


def test_reopen_raises_camera_error_when_unavailable():
    with mock.patch("cv2.VideoCapture") as vc:
        vc.return_value = _fake_cap(False)
        cam = Camera(device=0)
        with pytest.raises(CameraError):
            cam.reopen()


def test_read_returns_none_on_failed_frame():
    with mock.patch("cv2.VideoCapture") as vc:
        cap = _fake_cap(True)
        cap.read.return_value = (False, None)
        vc.return_value = cap
        cam = Camera(device=0)
        cam.open()
        assert cam.read() is None


def _stub_landmarker(num_hands=1):
    """A HandLandmarker with the model load + MediaPipe stubbed out."""
    with mock.patch("axis.landmarks.models.ensure_hand_landmarker",
                    return_value="x.model"), \
         mock.patch("mediapipe.tasks.python.vision.HandLandmarker"
                    ".create_from_options") as create:
        create.return_value = mock.Mock()
        hl = __import__(
            "axis.landmarks.extractor", fromlist=["HandLandmarker"]
        ).HandLandmarker(num_hands=num_hands)
    return hl, create


def test_extractor_defaults_to_single_hand():
    """Default num_hands=1 (≈2x faster); two hands remain opt-in."""
    hl, create = _stub_landmarker()
    assert create.call_args.args[0].num_hands == 1
    hl2, create2 = _stub_landmarker(num_hands=2)
    assert create2.call_args.args[0].num_hands == 2


def test_process_converts_bgr_to_rgb():
    """Frames must be BGR→RGB converted before MediaPipe — the old code
    fed BGR bytes as SRGB, silently swapping the color channels."""
    hl, _ = _stub_landmarker()
    frame = np.zeros((4, 4, 3), dtype=np.uint8)
    frame[:, :, 2] = 255  # BGR blue → becomes RGB red (255,0,0)

    captured = {}

    class _FakeImage:
        def __init__(self, image_format=None, data=None):
            captured["data"] = np.asarray(data).copy()

        def numpy_view(self):
            return captured["data"]

    result = mock.Mock(hand_landmarks=[], handedness=[])
    hl._landmarker = mock.Mock()
    hl._landmarker.detect_for_video.return_value = result

    with mock.patch("axis.landmarks.extractor.mp.Image", _FakeImage):
        out = hl.process(frame)

    assert out is None  # stub result has no hands
    assert captured["data"][0, 0].tolist() == [255, 0, 0]


def test_process_rgb_buffer_reused():
    """The staging buffer is allocated once, not per frame (memory)."""
    hl, _ = _stub_landmarker()
    frame = np.zeros((4, 4, 3), dtype=np.uint8)
    stub = mock.Mock(hand_landmarks=[], handedness=[])
    hl._landmarker = mock.Mock()
    hl._landmarker.detect_for_video.return_value = stub
    with mock.patch("axis.landmarks.extractor.mp.Image"):
        hl.process(frame)
        buf1 = hl._rgb
        hl.process(frame)
        assert hl._rgb is buf1
