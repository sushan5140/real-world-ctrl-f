"""Optional ESP32 pan/tilt LED controller. No serial device is required for software use."""
from __future__ import annotations
import json
from pathlib import Path

DEFAULT_CORNERS = {'tl': [135, 55], 'tr': [45, 55],
                   'bl': [135, 125], 'br': [45, 125]}


def angles(x: float, y: float, corners: dict) -> tuple[int, int]:
    """Map fixed-camera normalized desk point to user-calibrated servo angles."""
    if not 0 <= x <= 1 or not 0 <= y <= 1:
        raise ValueError('Camera coordinates must be between 0 and 1.')
    result = []
    for axis in (0, 1):
        a = corners['tl'][axis] * (1 - x) + corners['tr'][axis] * x
        b = corners['bl'][axis] * (1 - x) + corners['br'][axis] * x
        result.append(round(max(0, min(180, a * (1 - y) + b * y))))
    return result[0], result[1]


class Spotlight:
    def __init__(self, port: str | None, calibration: Path):
        self.calibration = calibration
        self.corners = DEFAULT_CORNERS
        if calibration.exists():
            data = json.loads(calibration.read_text(encoding='utf-8'))
            candidate = data['corners']
            if set(candidate) != {'tl', 'tr', 'bl', 'br'} or any(
                len(v) != 2 or any(not isinstance(a, (int, float)) or not 0 <= a <= 180 for a in v)
                for v in candidate.values()
            ):
                raise ValueError('Invalid spotlight calibration corners.')
            self.corners = candidate
        self.serial = None
        if port:
            try:
                import serial
            except ImportError as exc:
                raise RuntimeError('Install pyserial to use --serial-port.') from exc
            self.serial = serial.Serial(port, 115200, timeout=1, write_timeout=1)

    @property
    def connected(self) -> bool:
        return self.serial is not None and self.serial.is_open

    def point(self, x: float, y: float) -> tuple[int, int]:
        pan, tilt = angles(x, y, self.corners)
        if self.connected:
            self.serial.write(f'P,{pan},{tilt}\n'.encode('ascii'))
        return pan, tilt

    def light(self, enabled: bool) -> None:
        if self.connected:
            self.serial.write(f'L,{1 if enabled else 0}\n'.encode('ascii'))

    def close(self) -> None:
        if self.connected:
            self.light(False)
            self.serial.close()
