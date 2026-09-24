"""Synthetic camera frames and a fake serial port for camera-free tests."""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for extra in (ROOT, ROOT / 'tools'):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))

W, H = 960, 540
DICT = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)


def desk(seed: int = 1) -> np.ndarray:
    """A textured, static desk with enough detail for the camera-position check."""
    import benchmark_recognition as bench
    bench_rng = np.random.default_rng(seed)
    return bench.make_desk(bench_rng)


def plain(value: int = 215) -> np.ndarray:
    return np.full((H, W, 3), value, np.uint8)


def with_tag(frame: np.ndarray, marker_id: int, x: int, y: int, size: int = 110) -> np.ndarray:
    frame = frame.copy()
    quiet = size // 5
    card = np.full((size + 2 * quiet, size + 2 * quiet, 3), 255, np.uint8)
    card[quiet:quiet + size, quiet:quiet + size] = cv2.cvtColor(cv2.aruco.generateImageMarker(DICT, marker_id, size),
                                                                cv2.COLOR_GRAY2BGR)
    frame[y:y + card.shape[0], x:x + card.shape[1]] = card
    return frame


def textured_object(seed: int = 19, size=(210, 160)) -> np.ndarray:
    import benchmark_recognition as bench
    rng = np.random.default_rng(seed)
    obj = bench.make_object(rng)
    return cv2.resize(obj, size, interpolation=cv2.INTER_AREA)


def place(frame: np.ndarray, obj: np.ndarray, cx: float, cy: float, angle: float = 0.0,
          scale: float = 1.0) -> np.ndarray:
    frame = frame.copy()
    h, w = obj.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, scale)
    M[:, 2] += (cx - w / 2, cy - h / 2)
    warped = cv2.warpAffine(obj, M, (frame.shape[1], frame.shape[0]))
    mask = cv2.warpAffine(np.full((h, w), 255, np.uint8), M, (frame.shape[1], frame.shape[0]))
    frame[mask > 128] = warped[mask > 128]
    return frame


def shift(frame: np.ndarray, dx: int, dy: int) -> np.ndarray:
    """Simulate the camera being bumped: the whole picture moves."""
    M = np.float32([[1, 0, dx], [0, 1, dy]])
    return cv2.warpAffine(frame, M, (frame.shape[1], frame.shape[0]), borderMode=cv2.BORDER_REFLECT)


class FakeSerial:
    """Emulates the v2 firmware protocol (or a silent v1 board)."""

    def __init__(self, firmware: bool = True, fail_writes: bool = False):
        self.firmware = firmware
        self.fail_writes = fail_writes
        self.is_open = True
        self.sent: list[str] = []
        self._replies: list[bytes] = [b'READY CTRLF-SPOTLIGHT 2\n'] if firmware else []
        self.dtr = self.rts = None

    def write(self, data: bytes) -> int:
        if self.fail_writes:
            raise OSError('device disconnected')
        for line in data.decode().splitlines():
            self.sent.append(line)
            if not self.firmware:
                continue
            if line == '?':
                self._replies.append(b'PONG CTRLF-SPOTLIGHT 2\n')
            elif line.startswith('P,'):
                pan, tilt = (max(30, min(150, int(v))) for v in line[2:].split(','))
                self._replies.append(f'OK P,{pan},{tilt}\n'.encode())
            elif line in ('L,1', 'L,0', 'H'):
                self._replies.append(f'OK {line}\n'.encode())
            else:
                self._replies.append(b'ERR unknown command\n')
        return len(data)

    def readline(self) -> bytes:
        return self._replies.pop(0) if self._replies else b''

    def flush(self):
        pass

    def reset_input_buffer(self):
        self._replies.clear()

    def close(self):
        self.is_open = False
