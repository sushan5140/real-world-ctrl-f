"""Optional ESP32 pan/tilt LED spotlight (serial, 115200 baud).

Nothing here is required for the camera app. Protocol (firmware v2):
    ?            -> PONG CTRLF-SPOTLIGHT 2
    P,<pan>,<tilt> -> OK P,<pan>,<tilt>   (angles actually applied, after limits)
    L,1 / L,0    -> OK L,1 / OK L,0
    H            -> OK H                 (home 90/90, light off)
The firmware prints "READY CTRLF-SPOTLIGHT 2" after boot. The original
firmware (v1) never replies; it still works, but without confirmations.

Calibration maps a normalised camera point (x, y in 0..1) to servo angles by
bilinear interpolation between four measured corner poses (tl, tr, bl, br).
This is a good approximation for a flat desk seen by a fixed camera with the
lamp mounted near the camera; it is not exact for tall objects or a lamp far
from the camera. User calibration is saved to data/spotlight.json; the
repository's spotlight.json only holds placeholder defaults.
"""
from __future__ import annotations

import json
import os
import threading
import time
from pathlib import Path
from typing import Any, Callable

DEFAULT_CORNERS = {'tl': [135, 55], 'tr': [45, 55],
                   'bl': [135, 125], 'br': [45, 125]}
DEFAULT_LIMITS = {'pan': [30, 150], 'tilt': [30, 150]}   # must match firmware MIN_/MAX_
CORNER_NAMES = ('tl', 'tr', 'bl', 'br')
BAUD = 115200


class SpotlightError(RuntimeError):
    pass


def _valid_pair(value) -> bool:
    return (isinstance(value, (list, tuple)) and len(value) == 2 and
            all(isinstance(a, (int, float)) and not isinstance(a, bool) and 0 <= a <= 180 for a in value))


def angles(x: float, y: float, corners: dict, limits: dict | None = None) -> tuple[int, int]:
    """Map a fixed-camera normalised point to calibrated (pan, tilt) angles."""
    if not (0 <= x <= 1 and 0 <= y <= 1):
        raise ValueError('Camera coordinates must be between 0 and 1.')
    result = []
    for axis, name in ((0, 'pan'), (1, 'tilt')):
        top = corners['tl'][axis] * (1 - x) + corners['tr'][axis] * x
        bottom = corners['bl'][axis] * (1 - x) + corners['br'][axis] * x
        low, high = (limits or {}).get(name, (0, 180))
        result.append(int(round(max(low, min(high, top * (1 - y) + bottom * y)))))
    return result[0], result[1]


class Calibration:
    def __init__(self, default_path: Path, user_path: Path | None = None):
        self.default_path = default_path
        self.user_path = user_path
        self.corners = {k: list(v) for k, v in DEFAULT_CORNERS.items()}
        self.limits = {k: list(v) for k, v in DEFAULT_LIMITS.items()}
        self.placeholder = True
        self.source = 'built-in defaults'
        for path in (default_path, user_path):
            if path and path.is_file():
                self._load(path)

    def _load(self, path: Path) -> None:
        try:
            data = json.loads(path.read_text(encoding='utf-8'))
        except (OSError, ValueError) as exc:
            raise ValueError(f'Cannot read spotlight calibration {path}: {exc}') from exc
        corners = data.get('corners') if isinstance(data, dict) else None
        if not isinstance(corners, dict) or set(corners) != set(CORNER_NAMES) or \
                not all(_valid_pair(v) for v in corners.values()):
            raise ValueError(f'Invalid spotlight calibration corners in {path}.')
        limits = data.get('limits', DEFAULT_LIMITS)
        if not isinstance(limits, dict) or set(limits) != {'pan', 'tilt'} or \
                not all(_valid_pair(v) and v[0] < v[1] for v in limits.values()):
            raise ValueError(f'Invalid spotlight limits in {path}.')
        self.corners = {k: [float(a) for a in v] for k, v in corners.items()}
        self.limits = {k: [int(a) for a in v] for k, v in limits.items()}
        self.placeholder = bool(data.get('placeholder', False))
        self.source = str(path.name)

    def save(self) -> None:
        if not self.user_path:
            raise ValueError('No calibration file configured.')
        self.user_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.user_path.with_name(self.user_path.name + '.tmp')
        tmp.write_text(json.dumps({'corners': self.corners, 'limits': self.limits,
                                   'placeholder': self.placeholder}, indent=2), encoding='utf-8')
        os.replace(tmp, self.user_path)
        self.source = self.user_path.name

    def set_corner(self, name: str, pan: int, tilt: int) -> None:
        if name not in CORNER_NAMES:
            raise ValueError('Corner must be tl, tr, bl or br.')
        self.corners[name] = [int(pan), int(tilt)]
        self.placeholder = False

    def clamp(self, pan: float, tilt: float) -> tuple[int, int]:
        p0, p1 = self.limits['pan']
        t0, t1 = self.limits['tilt']
        return int(round(max(p0, min(p1, pan)))), int(round(max(t0, min(t1, tilt))))

    def map(self, x: float, y: float) -> tuple[int, int]:
        return angles(x, y, self.corners, self.limits)


def list_ports() -> list[dict[str, str]]:
    try:
        from serial.tools import list_ports as lp
    except ImportError:
        return []
    return [{'device': p.device, 'description': p.description or ''} for p in lp.comports()]


def _default_serial_factory(port: str):
    import serial
    link = serial.Serial()
    link.port = port
    link.baudrate = BAUD
    link.timeout = .1
    link.write_timeout = 1
    # Opening a USB serial port normally pulses DTR/RTS, which *resets* most
    # ESP32 boards; commands sent during the ~1 s boot are lost. Keep them low.
    link.dtr = False
    link.rts = False
    link.open()
    return link


class Spotlight:
    def __init__(self, port: str | None = None, calibration: Path | None = None,
                 user_calibration: Path | None = None,
                 serial_factory: Callable[[str], Any] | None = None):
        self.calibration = Calibration(calibration or Path(__file__).with_name('spotlight.json'), user_calibration)
        self._factory = serial_factory or _default_serial_factory
        self._io = threading.Lock()
        self.serial = None
        self.port: str | None = None
        self.firmware: str | None = None     # None = no reply (v1 firmware or not a spotlight)
        self.error: str | None = None
        self.pan: int | None = None
        self.tilt: int | None = None
        self.light_on = False
        if port:
            try:
                self.connect(port)
            except SpotlightError as exc:
                self.error = str(exc)

    # compat: the original class exposed .corners
    @property
    def corners(self) -> dict:
        return self.calibration.corners

    @property
    def connected(self) -> bool:
        return self.serial is not None and bool(getattr(self.serial, 'is_open', False))

    def status(self) -> dict[str, Any]:
        return {'connected': self.connected, 'port': self.port, 'firmware': self.firmware,
                'confirmed': self.firmware is not None, 'error': self.error,
                'pan': self.pan, 'tilt': self.tilt, 'light': self.light_on,
                'calibrated': not self.calibration.placeholder, 'calibration': self.calibration.corners,
                'limits': self.calibration.limits}

    # ---- connection ----------------------------------------------------------
    def connect(self, port: str) -> None:
        self.disconnect()
        try:
            link = self._factory(port)
        except ImportError as exc:
            raise SpotlightError('Install pyserial to use the spotlight (pip install pyserial).') from exc
        except Exception as exc:
            self.error = f'Cannot open {port}: {exc}. Close Arduino Serial Monitor and check the port.'
            raise SpotlightError(self.error) from exc
        with self._io:
            self.serial, self.port, self.error = link, port, None
            self.firmware = self._handshake()
        if self.firmware is None:
            self.error = ('Connected, but the board did not answer. Commands will still be sent. '
                          'Flash firmware v2 for confirmed movements.')
        self.home()

    def _handshake(self) -> str | None:
        deadline = time.monotonic() + 2.5
        self._write_raw('?')
        while time.monotonic() < deadline:
            line = self._readline()
            if line.startswith(('PONG', 'READY')):
                return line.split(' ', 1)[1] if ' ' in line else line
            if not line and time.monotonic() > deadline - 1.2:
                self._write_raw('?')            # board may have rebooted on open
        return None

    def disconnect(self) -> None:
        with self._io:
            if self.serial is not None:
                try:
                    if self.serial.is_open:
                        self._write_raw('L,0')
                    self.serial.close()
                except Exception:
                    pass
            self.serial, self.port, self.firmware = None, None, None
            self.light_on = False

    close = disconnect

    # ---- low level -------------------------------------------------------------
    def _write_raw(self, command: str) -> None:
        self.serial.write((command + '\n').encode('ascii'))
        self.serial.flush()

    def _readline(self) -> str:
        raw = self.serial.readline()
        return raw.decode('ascii', 'replace').strip() if raw else ''

    def _command(self, command: str) -> str | None:
        if not self.connected:
            raise SpotlightError('Spotlight not connected.')
        with self._io:
            try:
                self.serial.reset_input_buffer()
                self._write_raw(command)
                if self.firmware is None:
                    return None
                deadline = time.monotonic() + .8
                while time.monotonic() < deadline:
                    line = self._readline()
                    if line.startswith('OK'):
                        return line
                    if line.startswith('ERR'):
                        raise SpotlightError(f'Spotlight rejected "{command}": {line[3:].strip()}')
                raise SpotlightError('The spotlight did not confirm the command (check USB cable and power).')
            except SpotlightError:
                raise
            except Exception as exc:
                # Unplugged / driver error: drop the link so the UI shows it.
                try:
                    self.serial.close()
                except Exception:
                    pass
                self.serial = None
                self.error = f'Spotlight connection lost: {exc}'
                raise SpotlightError(self.error) from exc

    # ---- actions -----------------------------------------------------------------
    def move(self, pan: float, tilt: float) -> tuple[int, int]:
        pan_i, tilt_i = self.calibration.clamp(pan, tilt)
        reply = self._command(f'P,{pan_i},{tilt_i}')
        if reply and reply.startswith('OK P,'):
            try:
                pan_i, tilt_i = (int(v) for v in reply[5:].split(','))
            except ValueError:
                pass
        self.pan, self.tilt = pan_i, tilt_i
        return pan_i, tilt_i

    def point(self, x: float, y: float) -> tuple[int, int]:
        pan, tilt = self.calibration.map(x, y)
        if self.connected:
            return self.move(pan, tilt)
        return pan, tilt

    def light(self, enabled: bool) -> None:
        if not self.connected:
            return
        self._command(f'L,{1 if enabled else 0}')
        self.light_on = bool(enabled)

    def home(self) -> None:
        if not self.connected:
            return
        try:
            self._command('H' if self.firmware else 'L,0')
        except SpotlightError:
            return
        self.light_on = False
        if self.firmware:
            self.pan, self.tilt = 90, 90

    def save_corner(self, name: str) -> dict:
        if self.pan is None or self.tilt is None:
            raise SpotlightError('Move the spotlight first, then save the corner.')
        self.calibration.set_corner(name, self.pan, self.tilt)
        self.calibration.save()
        return self.calibration.corners
