"""Camera input: opening a webcam (or a video file for demos/tests) and a
background worker that feeds frames to the tracking engine.

The worker owns the VideoCapture object; nothing else touches it. It
reconnects automatically after a camera is unplugged, never holds the
engine lock while waiting for the camera, and always releases the device
when stopped.
"""
from __future__ import annotations

import sys
import threading
import time
from pathlib import Path
from typing import Callable

import cv2
import numpy as np


def parse_source(value: str | int) -> int | str:
    """'0' -> webcam index 0; anything else is treated as a file path / URL."""
    if isinstance(value, int):
        return value
    text = str(value).strip()
    return int(text) if text.isdigit() else text


def open_capture(source: int | str, width: int | None = 960, height: int | None = 540) -> cv2.VideoCapture:
    if isinstance(source, int):
        # DirectShow opens most Windows webcams quickly and reliably.
        cap = cv2.VideoCapture(source, cv2.CAP_DSHOW) if sys.platform == 'win32' else cv2.VideoCapture(source)
        if cap.isOpened():
            if width and height:
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)   # prefer fresh frames over queued ones
    else:
        cap = cv2.VideoCapture(source)
    return cap


def describe_source(source: int | str) -> str:
    return f'camera {source}' if isinstance(source, int) else Path(str(source)).name


class CameraWorker:
    """Runs capture + processing in a background thread.

    state: 'stopped' | 'starting' | 'running' | 'reconnecting' | 'error'
    """

    def __init__(self, process: Callable[[np.ndarray, float], np.ndarray | None],
                 source: int | str = 0, max_fps: float = 15.0,
                 resolution: tuple[int, int] | None = (960, 540),
                 on_stop: Callable[[], None] | None = None):
        self._process = process
        self.source = source
        self.max_fps = max_fps
        self.resolution = resolution
        self._on_stop = on_stop
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.state = 'stopped'
        self.error: str | None = None
        self.jpeg: bytes | None = None
        self.frame_size: tuple[int, int] | None = None
        self.fps = 0.0
        self.frames = 0

    # ---- control -----------------------------------------------------------
    def start(self, source: int | str | None = None) -> None:
        with self._lock:
            if self._thread and self._thread.is_alive():
                if source is None or source == self.source:
                    return
                self._stop_locked()
            if source is not None:
                self.source = source
            self._stop.clear()
            self.state, self.error = 'starting', None
            self._thread = threading.Thread(target=self._run, daemon=True, name='ctrl-f-camera')
            self._thread.start()

    def stop(self) -> None:
        with self._lock:
            self._stop_locked()

    def _stop_locked(self) -> None:
        self._stop.set()
        thread = self._thread
        if thread and thread.is_alive() and thread is not threading.current_thread():
            thread.join(timeout=5)
        self._thread = None
        self.state = 'stopped'
        self.jpeg = None
        self.fps = 0.0

    @property
    def running(self) -> bool:
        return self.state == 'running'

    def status(self) -> dict:
        return {'state': self.state, 'error': self.error, 'source': describe_source(self.source),
                'fps': round(self.fps, 1), 'width': self.frame_size[0] if self.frame_size else None,
                'height': self.frame_size[1] if self.frame_size else None}

    # ---- loop --------------------------------------------------------------
    def _run(self) -> None:
        is_file = not isinstance(self.source, int)
        attempts = 0
        while not self._stop.is_set():
            cap = open_capture(self.source, *(self.resolution or (None, None)))
            if not cap.isOpened():
                cap.release()
                attempts += 1
                self.state = 'error' if attempts >= 3 else 'reconnecting'
                self.error = (f'Cannot open {describe_source(self.source)}. Close other apps using the camera, '
                              'check camera privacy permissions, or choose another camera number.')
                if self._stop.wait(2.0):
                    break
                continue
            attempts = 0
            try:
                self._pump(cap, is_file)
            finally:
                cap.release()
                self._notify_stopped()
            if not self._stop.is_set():
                self.state = 'reconnecting'
                self.error = 'The camera stopped sending frames. Trying to reconnect…'
                self._stop.wait(1.5)
        self.state = 'stopped'

    def _pump(self, cap: cv2.VideoCapture, is_file: bool) -> None:
        failures = 0
        errors = 0
        interval = 1.0 / self.max_fps if self.max_fps else 0
        last_tick = time.monotonic()
        while not self._stop.is_set():
            started = time.monotonic()
            ok, frame = cap.read()
            if not ok or frame is None:
                if is_file:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)   # loop demo videos
                    failures += 1
                    if failures > 3:
                        return
                    continue
                failures += 1
                if failures >= 25:
                    return
                time.sleep(.04)
                continue
            failures = 0
            try:
                annotated = self._process(frame, time.time())
                errors = 0
            except Exception as exc:  # keep running; one bad frame must not kill tracking
                errors += 1
                self.error = f'Frame processing error: {exc}'
                if errors >= 50:
                    self.state = 'error'
                    return
                continue
            if annotated is not None:
                ok, buffer = cv2.imencode('.jpg', annotated, [cv2.IMWRITE_JPEG_QUALITY, 80])
                if ok:
                    self.jpeg = buffer.tobytes()
            h, w = frame.shape[:2]
            self.frame_size = (w, h)
            self.state, self.error = 'running', None
            self.frames += 1
            now = time.monotonic()
            dt = now - last_tick
            last_tick = now
            if dt > 0:
                self.fps = .9 * self.fps + .1 * (1 / dt) if self.fps else 1 / dt
            spare = interval - (time.monotonic() - started)
            if spare > 0:
                self._stop.wait(spare)

    def _notify_stopped(self) -> None:
        if self._on_stop:
            try:
                self._on_stop()
            except Exception:
                pass
