"""Eyes — camera capture and frame encoding for TBS Eyes.

Wraps OpenCV so the rest of the app never touches cv2 directly:
- Camera: open a device, grab frames, release cleanly
- CameraFeed: persistent background capture thread; newest frame always available
- frame_to_jpeg_b64: downscale + JPEG-encode + base64 a frame for the Claude API
- list_cameras: probe which device indexes actually work
"""

import base64
import os
import platform
import threading

# Silence OpenCV's warning spam when probing camera indexes that don't exist.
# Must be set before cv2 is imported.
os.environ.setdefault("OPENCV_LOG_LEVEL", "ERROR")

import cv2
import numpy as np


class CameraError(Exception):
    """Raised when a camera can't be opened or a frame can't be read."""


class Camera:
    """A single camera device. Use as a context manager or call release() yourself."""

    def __init__(self, index: int = 0):
        self.index = index
        # CAP_DSHOW makes camera open near-instant on Windows (default backend
        # can take 10+ seconds). Other platforms use the default backend.
        if platform.system() == "Windows":
            self.cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
        else:
            self.cap = cv2.VideoCapture(index)
        if not self.cap.isOpened():
            raise CameraError(
                f"Could not open camera {index}. "
                "Check it's plugged in and not in use by another app."
            )
        # Ask for 1080p; the driver falls back to the closest resolution the
        # camera actually supports.
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

    def capture_frame(self) -> np.ndarray:
        """Grab one fresh frame (BGR ndarray).

        Reads a few frames first to flush the driver's buffer — otherwise you
        get a stale image from several seconds ago.
        """
        for _ in range(3):
            self.cap.grab()
        ok, frame = self.cap.read()
        if not ok or frame is None:
            raise CameraError(f"Camera {self.index} opened but returned no frame.")
        return frame

    def release(self) -> None:
        self.cap.release()

    def __enter__(self) -> "Camera":
        return self

    def __exit__(self, *exc) -> None:
        self.release()


class CameraFeed:
    """Owns one cv2.VideoCapture (via Camera) on a background thread.

    Grabs frames continuously (~15 fps); the newest frame is always available
    from get_frame() under a lock. Reopens the device if it drops.
    """

    def __init__(self, index: int, on_status=None):
        self.index = index
        self.on_status = on_status or (lambda msg: None)
        self._lock = threading.Lock()
        self._frame = None
        self._stop = threading.Event()
        self._thread = None

    def start(self) -> None:
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, daemon=True, name=f"camera-{self.index}")
        self._thread.start()

    def _run(self) -> None:
        cam = None
        failures = 0
        while not self._stop.is_set():
            if cam is None:
                try:
                    cam = Camera(self.index)
                    self.on_status(f"Camera {self.index} connected")
                    failures = 0
                except CameraError as e:
                    self.on_status(f"Camera error: {e}")
                    if self._stop.wait(2.0):
                        break
                    continue
            ok, frame = cam.cap.read()
            if ok and frame is not None:
                with self._lock:
                    self._frame = frame
                failures = 0
                self._stop.wait(1 / 15)
            else:
                failures += 1
                if failures >= 15:  # ~ a second of dead reads -> reopen
                    self.on_status(f"Camera {self.index} lost — reconnecting…")
                    cam.release()
                    cam = None
                    with self._lock:
                        self._frame = None
                self._stop.wait(0.1)
        if cam is not None:
            cam.release()

    def get_frame(self):
        """Return a copy of the newest frame (BGR ndarray), or None."""
        with self._lock:
            return None if self._frame is None else self._frame.copy()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=3)
            self._thread = None


def frame_to_jpeg_b64(frame: np.ndarray, quality: int = 85, max_dim: int = 1024) -> str:
    """Downscale a frame so its longest edge <= max_dim, JPEG-encode it,
    and return a base64 string ready for the Claude API's image block."""
    h, w = frame.shape[:2]
    longest = max(h, w)
    if longest > max_dim:
        scale = max_dim / longest
        frame = cv2.resize(frame, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)

    ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise CameraError("JPEG encoding failed.")
    return base64.standard_b64encode(buf.tobytes()).decode("utf-8")


def save_frame(frame: np.ndarray, path) -> bool:
    """Write a frame to disk as JPEG (full resolution, no downscale).
    Returns False when the write failed (e.g. write-locked flash drive) —
    cv2.imwrite reports failure via its return value, not an exception."""
    try:
        return bool(cv2.imwrite(str(path), frame, [cv2.IMWRITE_JPEG_QUALITY, 92]))
    except cv2.error:
        return False


def list_cameras(max_index: int = 5) -> list[int]:
    """Probe device indexes 0..max_index and return the ones that produce frames."""
    found = []
    for i in range(max_index + 1):
        if platform.system() == "Windows":
            cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
        else:
            cap = cv2.VideoCapture(i)
        if cap.isOpened():
            ok, _ = cap.read()
            if ok:
                found.append(i)
        cap.release()
    return found
