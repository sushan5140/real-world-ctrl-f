"""Local browser studio: webcam + visual memory + optional ESP32 spotlight.

Run: python server.py --camera 0 --port 8765 [--serial-port COM3]
The HTTP server binds to 127.0.0.1 only. No cloud camera uploads.
"""
from __future__ import annotations

import argparse
import json
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse, StreamingResponse
from pydantic import BaseModel, Field

from spotlight import Spotlight
from tracker import (Observation, detect_markers, get_last_seen, init_db,
                     load_objects, match_object, save_observation)
from vision import create_template, find_templates, load_templates, save_template

ROOT = Path(__file__).resolve().parent
DATA = ROOT / 'data'
BUILTIN = ROOT / 'objects.json'
CATALOG = DATA / 'catalog.json'
TEMPLATES = DATA / 'templates'
SNAPSHOTS = DATA / 'snapshots'


class Rect(BaseModel):
    x: float = Field(ge=0, le=1)
    y: float = Field(ge=0, le=1)
    w: float = Field(gt=0, le=1)
    h: float = Field(gt=0, le=1)


class EnrollRequest(BaseModel):
    name: str = Field(min_length=2, max_length=45)
    aliases: list[str] = Field(default_factory=list, max_length=8)
    rect: Rect


class Query(BaseModel):
    query: str = Field(min_length=1, max_length=100)


class Toggle(BaseModel):
    enabled: bool


class Engine:
    def __init__(self, camera: int = 0, serial_port: str | None = None):
        DATA.mkdir(exist_ok=True)
        SNAPSHOTS.mkdir(exist_ok=True)
        self.camera = camera
        self.conn = init_db(DATA / 'locations.sqlite3')
        self.lock = threading.RLock()
        self.builtin = load_objects(BUILTIN)
        self.custom: dict[int, dict[str, Any]] = {}
        if CATALOG.is_file():
            try:
                loaded = json.loads(CATALOG.read_text(encoding='utf-8'))
                self.custom = {int(k): v for k, v in loaded.items() if int(k) >= 50}
            except (ValueError, TypeError, OSError, AttributeError) as exc:
                raise RuntimeError(f'Invalid local catalog {CATALOG}: {exc}') from exc
        self.templates = load_templates(TEMPLATES, set(self.custom))
        # A missing / damaged image cannot silently become a functional enrolled item.
        self.custom = {k: v for k, v in self.custom.items() if k in self.templates}
        self.objects = {**self.builtin, **self.custom}
        self.spotlight = Spotlight(serial_port, ROOT / 'spotlight.json')
        self.selected: int | None = None
        self.observations: dict[int, Observation] = {}
        self.last_frames: dict[int, tuple[np.ndarray, tuple[int, int, int, int]]] = {}
        self.last_saved: dict[int, float] = {}
        self.raw: np.ndarray | None = None
        self.jpeg: bytes | None = None
        self.error: str | None = None
        self.stop_event = threading.Event()
        self.thread: threading.Thread | None = None

    def start(self):
        if self.thread and self.thread.is_alive():
            return
        self.stop_event.clear()
        self.thread = threading.Thread(target=self._camera_loop, daemon=True, name='ctrl-f-camera')
        self.thread.start()

    def close(self):
        self.stop_event.set()
        if self.thread:
            self.thread.join(timeout=4)
        with self.lock:
            self.spotlight.close()
            self.conn.close()

    def _camera_loop(self):
        import sys
        camera = cv2.VideoCapture(self.camera, cv2.CAP_DSHOW) if sys.platform == 'win32' else cv2.VideoCapture(self.camera)
        if not camera.isOpened():
            with self.lock:
                self.error = f'Cannot open camera {self.camera}. Check permissions or try --camera 1.'
            camera.release()
            return
        camera.set(cv2.CAP_PROP_FRAME_WIDTH, 960)
        camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 540)
        failures = 0
        try:
            while not self.stop_event.is_set():
                ok, frame = camera.read()
                if not ok:
                    failures += 1
                    if failures >= 20:
                        with self.lock:
                            self.error = 'Camera stopped sending frames. Reconnect it and restart the app.'
                        break
                    time.sleep(.1)
                    continue
                failures = 0
                self.process(frame)
                time.sleep(.055)
        except Exception as exc:
            with self.lock:
                self.error = f'Camera loop stopped: {exc}'
        finally:
            camera.release()

    def process(self, frame: np.ndarray) -> None:
        """Process one BGR frame. Public to permit deterministic camera-free tests."""
        with self.lock:
            h, w = frame.shape[:2]
            observed = {i: o for i, o in detect_markers(frame).items() if i in self.builtin}
            for i, result in find_templates(frame, self.templates).items():
                observed[i] = Observation(i, result.x, result.y, result.box)
            now = time.monotonic()
            for i, obs in observed.items():
                self.last_frames[i] = (frame.copy(), obs.box)
                if now - self.last_saved.get(i, 0) >= .8:
                    save_observation(self.conn, obs, w, h)
                    self.last_saved[i] = now
            for i in set(self.observations) - set(observed):
                old = self.observations[i]
                previous = self.last_frames.pop(i, None)
                snapshot = None
                if previous:
                    source, (x0, y0, x1, y1) = previous
                    margin = 30
                    cropped = source[max(0, y0-margin):min(h, y1+margin),
                                     max(0, x0-margin):min(w, x1+margin)]
                    if cropped.size:
                        snapshot = SNAPSHOTS / f'{i}.jpg'
                        if not cv2.imwrite(str(snapshot), cropped):
                            snapshot = None
                save_observation(self.conn, old, w, h, str(snapshot) if snapshot else None)
            self.observations = observed
            self.raw = frame.copy()
            canvas = frame.copy()
            for i, obs in observed.items():
                x0, y0, x1, y1 = obs.box
                active = i == self.selected
                color = (82, 140, 225) if active else (105, 179, 122)
                cv2.rectangle(canvas, (x0, y0), (x1, y1), color, 3 if active else 2)
                cv2.putText(canvas, self.objects[i]['name'], (x0, max(20, y0 - 9)),
                            cv2.FONT_HERSHEY_SIMPLEX, .65, color, 2, cv2.LINE_AA)
            if self.selected is not None:
                record = get_last_seen(self.conn, self.selected)
                if record:
                    x, y = round(record['x'] * w), round(record['y'] * h)
                    visible = self.selected in observed
                    color = (105, 179, 122) if visible else (82, 140, 225)
                    cv2.circle(canvas, (x, y), 26, color, 3)
                    cv2.circle(canvas, (x, y), 4, color, -1)
                    label = 'HERE NOW' if visible else 'LAST SEEN - NOT CONFIRMED HERE'
                    cv2.putText(canvas, label, (max(8, x - 120), max(24, y - 36)),
                                cv2.FONT_HERSHEY_SIMPLEX, .53, color, 2, cv2.LINE_AA)
            ok, buffer = cv2.imencode('.jpg', canvas, [cv2.IMWRITE_JPEG_QUALITY, 82])
            if ok:
                self.jpeg = buffer.tobytes()
            self.error = None

    def status(self) -> dict[str, Any]:
        with self.lock:
            items = []
            for obj_id, info in self.objects.items():
                rec = get_last_seen(self.conn, obj_id)
                seen = obj_id in self.observations
                items.append({
                    'id': obj_id, 'name': info['name'], 'kind': 'tag' if obj_id < 50 else 'enrolled',
                    'visible': seen, 'last_seen': datetime.fromtimestamp(rec['timestamp']).isoformat() if rec else None,
                    'age_seconds': max(0, round(time.time() - rec['timestamp'])) if rec else None,
                    'x': rec['x'] if rec else None, 'y': rec['y'] if rec else None,
                    'snapshot': f'/api/snapshot/{obj_id}' if rec and rec['snapshot_path'] and Path(rec['snapshot_path']).is_file() else None,
                })
            return {'ready': self.raw is not None, 'error': self.error, 'selected': self.selected,
                    'spotlight': self.spotlight.connected, 'items': items}

    def find(self, query: str) -> int:
        with self.lock:
            obj_id = match_object(query, self.objects)
            if obj_id is None:
                raise ValueError('No matching object. Enroll it first, or use a registered marker.')
            self.selected = obj_id
            return obj_id

    def select(self, obj_id: int) -> None:
        with self.lock:
            if obj_id not in self.objects:
                raise ValueError('Unknown object.')
            self.selected = obj_id

    def enroll(self, req: EnrollRequest) -> int:
        with self.lock:
            if self.raw is None:
                raise ValueError('No camera frame yet. Check camera access.')
            name = req.name.strip()
            if any(info['name'].casefold() == name.casefold() for info in self.objects.values()):
                raise ValueError('Name already registered. Choose another name.')
            rect = req.rect
            if rect.x + rect.w > 1.000001 or rect.y + rect.h > 1.000001:
                raise ValueError('Selection extends beyond the camera image.')
            h, w = self.raw.shape[:2]
            x0, y0 = round(rect.x * w), round(rect.y * h)
            x1, y1 = round((rect.x + rect.w) * w), round((rect.y + rect.h) * h)
            obj_id = max([49, *self.custom]) + 1
            template = create_template(obj_id, self.raw[y0:y1, x0:x1])
            path = TEMPLATES / f'{obj_id}.png'
            save_template(path, template)
            aliases = [a.strip() for a in req.aliases if a.strip()][:8]
            updated = {**self.custom, obj_id: {'name': name, 'aliases': aliases}}
            try:
                CATALOG.parent.mkdir(exist_ok=True)
                tmp = CATALOG.with_suffix('.tmp')
                tmp.write_text(json.dumps(updated, indent=2), encoding='utf-8')
                tmp.replace(CATALOG)
            except Exception:
                path.unlink(missing_ok=True)
                raise
            self.custom = updated
            self.templates[obj_id] = template
            self.objects = {**self.builtin, **self.custom}
            self.selected = obj_id
            return obj_id

    def forget(self, obj_id: int):
        with self.lock:
            if obj_id not in self.objects:
                raise ValueError('Unknown object.')
            self.conn.execute('DELETE FROM last_seen WHERE marker_id=?', (obj_id,))
            self.conn.commit()
            (SNAPSHOTS / f'{obj_id}.jpg').unlink(missing_ok=True)
            if obj_id in self.custom:
                revised = {k: v for k, v in self.custom.items() if k != obj_id}
                tmp = CATALOG.with_suffix('.tmp')
                tmp.write_text(json.dumps(revised, indent=2), encoding='utf-8')
                tmp.replace(CATALOG)
                (TEMPLATES / f'{obj_id}.png').unlink(missing_ok=True)
                self.custom = revised
                self.templates.pop(obj_id, None)
                self.objects = {**self.builtin, **self.custom}
            self.observations.pop(obj_id, None)
            self.last_frames.pop(obj_id, None)
            self.last_saved.pop(obj_id, None)
            if self.selected == obj_id:
                self.selected = None

    def point(self, obj_id: int) -> tuple[int, int]:
        with self.lock:
            rec = get_last_seen(self.conn, obj_id)
            if obj_id not in self.objects or rec is None:
                raise ValueError('This object has no remembered position.')
            if not self.spotlight.connected:
                raise ValueError('Spotlight not connected. Start with --serial-port COM3.')
            result = self.spotlight.point(rec['x'], rec['y'])
            self.spotlight.light(True)
            return result


def create_app(engine: Engine, start_camera: bool = True) -> FastAPI:
    app = FastAPI(title='Real-World Ctrl+F (local-only)', docs_url=None, redoc_url=None)

    if start_camera:
        @app.on_event('startup')
        def startup():
            engine.start()

        @app.on_event('shutdown')
        def shutdown():
            engine.close()

    @app.get('/')
    def home():
        return HTMLResponse((ROOT / 'web' / 'index.html').read_text(encoding='utf-8'))

    @app.get('/style.css')
    def css():
        return FileResponse(ROOT / 'web' / 'style.css', media_type='text/css')

    @app.get('/client.js')
    def js():
        return FileResponse(ROOT / 'web' / 'client.js', media_type='application/javascript')

    @app.get('/api/status')
    def status():
        return engine.status()

    @app.get('/stream')
    def stream():
        def frames():
            while not engine.stop_event.is_set():
                with engine.lock:
                    jpeg = engine.jpeg
                    error = engine.error
                if jpeg:
                    yield b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + jpeg + b'\r\n'
                elif error:
                    break
                time.sleep(.12)
        return StreamingResponse(frames(), media_type='multipart/x-mixed-replace; boundary=frame',
                                 headers={'Cache-Control': 'no-store'})

    @app.post('/api/find')
    def find(query: Query):
        try:
            return {'id': engine.find(query.query)}
        except ValueError as exc:
            raise HTTPException(404, str(exc)) from exc

    @app.post('/api/select/{obj_id}')
    def select(obj_id: int):
        try:
            engine.select(obj_id)
            return {'id': obj_id}
        except ValueError as exc:
            raise HTTPException(404, str(exc)) from exc

    @app.post('/api/enroll')
    def enroll(request: EnrollRequest):
        try:
            return {'id': engine.enroll(request)}
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc

    @app.delete('/api/object/{obj_id}')
    def forget(obj_id: int):
        try:
            engine.forget(obj_id)
            return {'deleted': obj_id}
        except ValueError as exc:
            raise HTTPException(404, str(exc)) from exc

    @app.get('/api/snapshot/{obj_id}')
    def snapshot(obj_id: int):
        with engine.lock:
            rec = get_last_seen(engine.conn, obj_id)
            path = SNAPSHOTS / f'{obj_id}.jpg'
            if obj_id not in engine.objects or rec is None or not rec['snapshot_path'] or not path.is_file():
                raise HTTPException(404, 'No last-seen snapshot yet.')
            return FileResponse(path, media_type='image/jpeg', headers={'Cache-Control': 'no-store'})

    @app.post('/api/point/{obj_id}')
    def point(obj_id: int):
        try:
            pan, tilt = engine.point(obj_id)
            return {'pan': pan, 'tilt': tilt}
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc

    @app.post('/api/light')
    def light(req: Toggle):
        if not engine.spotlight.connected:
            raise HTTPException(400, 'Spotlight not connected.')
        try:
            with engine.lock:
                engine.spotlight.light(req.enabled)
            return {'enabled': req.enabled}
        except Exception as exc:
            raise HTTPException(503, 'Spotlight communication failed.') from exc

    return app


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--camera', type=int, default=0)
    parser.add_argument('--port', type=int, default=8765)
    parser.add_argument('--serial-port', default=None, help='Optional ESP32 COM port, e.g. COM3')
    args = parser.parse_args()
    if not 1024 <= args.port <= 65535:
        parser.error('--port must be between 1024 and 65535')
    engine = Engine(args.camera, args.serial_port)
    import uvicorn
    print(f'Opening local studio at http://127.0.0.1:{args.port}')
    uvicorn.run(create_app(engine), host='127.0.0.1', port=args.port, access_log=False)
