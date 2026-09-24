"""Local browser studio: webcam + visual memory + optional ESP32 spotlight.

Run:   python server.py [--camera 0] [--port 8765] [--serial-port COM3]
Open:  http://127.0.0.1:8765

The server binds to 127.0.0.1 only and never uploads frames. Requests from
other websites are refused (Host/Origin checks + a required request header),
so a web page you visit cannot drive your spotlight or read your camera.
"""
from __future__ import annotations

import argparse
import asyncio
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse, Response, StreamingResponse
from pydantic import BaseModel, Field

from camera import CameraWorker, parse_source
from engine import ROOT, TrackingEngine
from spotlight import CORNER_NAMES, Spotlight, SpotlightError, list_ports

WEB = ROOT / 'web'
ALLOWED_HOSTS = {'127.0.0.1', 'localhost', '[::1]'}
SAFE_METHODS = {'GET', 'HEAD', 'OPTIONS'}
CSP = ("default-src 'self'; img-src 'self' blob: data:; style-src 'self'; script-src 'self'; "
       "connect-src 'self'; font-src 'self'; object-src 'none'; base-uri 'none'; frame-ancestors 'none'; "
       "form-action 'self'")


class Rect(BaseModel):
    x: float = Field(ge=0, le=1)
    y: float = Field(ge=0, le=1)
    w: float = Field(gt=0, le=1)
    h: float = Field(gt=0, le=1)


class EnrollRequest(BaseModel):
    token: str = Field(min_length=8, max_length=64)
    name: str = Field(min_length=2, max_length=45)
    aliases: list[str] = Field(default_factory=list, max_length=8)
    rect: Rect


class ViewRequest(BaseModel):
    token: str = Field(min_length=8, max_length=64)
    rect: Rect


class TagRequest(BaseModel):
    marker_id: int = Field(ge=0, le=49)
    name: str = Field(min_length=2, max_length=45)
    aliases: list[str] = Field(default_factory=list, max_length=8)


class Query(BaseModel):
    query: str = Field(min_length=1, max_length=100)


class Toggle(BaseModel):
    enabled: bool


class CameraRequest(BaseModel):
    source: str | None = Field(default=None, max_length=300)


class PortRequest(BaseModel):
    port: str = Field(min_length=1, max_length=100)


class MoveRequest(BaseModel):
    pan: float = Field(ge=0, le=180)
    tilt: float = Field(ge=0, le=180)


class AimRequest(BaseModel):
    x: float = Field(ge=0, le=1)
    y: float = Field(ge=0, le=1)


class CornerRequest(BaseModel):
    corner: str


class Studio:
    """Wires the shared engine to a camera worker and the optional spotlight."""

    def __init__(self, engine: TrackingEngine, camera_source: int | str = 0,
                 spotlight: Spotlight | None = None, max_fps: float = 15.0,
                 resolution: tuple[int, int] | None = (960, 540)):
        self.engine = engine
        self.spotlight = spotlight or Spotlight(None, ROOT / 'spotlight.json', engine.data_dir / 'spotlight.json')
        self.camera = CameraWorker(self._process, camera_source, max_fps, resolution, on_stop=engine.flush)
        self.closed = False

    def _process(self, frame, now):
        return self.engine.annotate(frame, self.engine.process(frame, now))

    def status(self) -> dict[str, Any]:
        data = self.engine.status()
        data['camera'] = self.camera.status()
        data['spotlight'] = self.spotlight.status()
        return data

    def close(self) -> None:
        self.closed = True
        self.camera.stop()
        self.spotlight.disconnect()
        self.engine.close()


def host_name(header: str) -> str:
    """'127.0.0.1:8765' -> '127.0.0.1', '[::1]:8765' -> '[::1]'."""
    header = header.strip().lower()
    if header.startswith('['):
        return header.split(']', 1)[0] + ']'
    return header.rsplit(':', 1)[0] if ':' in header else header


def _error(status: int, exc: Exception) -> HTTPException:
    message = exc.args[0] if exc.args and isinstance(exc.args[0], str) else str(exc)
    return HTTPException(status, message)


def create_app(studio: Studio, start_camera: bool = True) -> FastAPI:
    engine = studio.engine

    @asynccontextmanager
    async def lifespan(_app):
        if start_camera:
            studio.camera.start()
        yield
        await asyncio.to_thread(studio.close)

    app = FastAPI(title='Real-World Ctrl+F (local-only)', docs_url=None, redoc_url=None,
                  openapi_url=None, lifespan=lifespan)

    @app.middleware('http')
    async def local_only(request: Request, call_next):
        if host_name(request.headers.get('host', '')) not in ALLOWED_HOSTS:
            # Blocks DNS-rebinding: a hostile site resolving its name to 127.0.0.1.
            return JSONResponse({'detail': 'Unknown host.'}, status_code=400)
        if request.method not in SAFE_METHODS:
            origin = request.headers.get('origin')
            if origin and origin.split('://', 1)[-1] != request.headers.get('host'):
                return JSONResponse({'detail': 'Cross-site request refused.'}, status_code=403)
            if request.headers.get('x-ctrlf') != '1':
                # A custom header cannot be sent cross-site without a CORS
                # preflight, which this server never approves.
                return JSONResponse({'detail': 'Missing X-CtrlF header.'}, status_code=403)
        response = await call_next(request)
        response.headers['X-Content-Type-Options'] = 'nosniff'
        response.headers['X-Frame-Options'] = 'DENY'
        response.headers['Referrer-Policy'] = 'no-referrer'
        response.headers['Cross-Origin-Resource-Policy'] = 'same-origin'
        response.headers['Content-Security-Policy'] = CSP
        response.headers['Permissions-Policy'] = 'camera=(), geolocation=(), microphone=(self)'
        response.headers.setdefault('Cache-Control', 'no-store')
        return response

    # ---- static -------------------------------------------------------------
    def static(name: str, media: str):
        path = WEB / name
        return FileResponse(path, media_type=media, headers={'Cache-Control': 'no-cache'})

    @app.get('/')
    def home():
        return static('index.html', 'text/html; charset=utf-8')

    @app.get('/style.css')
    def css():
        return static('style.css', 'text/css; charset=utf-8')

    @app.get('/client.js')
    def js():
        return static('client.js', 'application/javascript; charset=utf-8')

    @app.get('/favicon.svg')
    def favicon():
        return static('favicon.svg', 'image/svg+xml')

    # ---- live data ----------------------------------------------------------
    @app.get('/api/status')
    def status():
        return studio.status()

    @app.get('/stream')
    async def stream(request: Request):
        async def frames():
            last = None
            while not studio.closed:
                if await request.is_disconnected():
                    break
                jpeg = studio.camera.jpeg
                if jpeg is not None and jpeg is not last:
                    last = jpeg
                    yield (b'--frame\r\nContent-Type: image/jpeg\r\nContent-Length: ' +
                           str(len(jpeg)).encode() + b'\r\n\r\n' + jpeg + b'\r\n')
                await asyncio.sleep(.04)
        return StreamingResponse(frames(), media_type='multipart/x-mixed-replace; boundary=frame',
                                 headers={'Cache-Control': 'no-store'})

    # ---- camera -------------------------------------------------------------
    @app.post('/api/camera/start')
    def camera_start(req: CameraRequest | None = None):
        source = parse_source(req.source) if req and req.source not in (None, '') else None
        studio.camera.start(source)
        return studio.camera.status()

    @app.post('/api/camera/stop')
    def camera_stop():
        studio.camera.stop()
        engine.flush()
        return studio.camera.status()

    # ---- search / selection -------------------------------------------------
    @app.post('/api/find')
    def find(query: Query):
        try:
            return engine.search(query.query)
        except LookupError as exc:
            raise _error(404, exc) from exc

    @app.post('/api/select/{obj_id}')
    def select(obj_id: int):
        try:
            engine.select(obj_id)
        except KeyError as exc:
            raise _error(404, exc) from exc
        return {'id': obj_id}

    @app.delete('/api/select')
    def unselect():
        engine.select(None)
        return {'id': None}

    # ---- library --------------------------------------------------------------
    @app.post('/api/enroll/still')
    def enroll_still():
        try:
            return engine.capture_still()
        except ValueError as exc:
            raise _error(409, exc) from exc

    @app.get('/api/enroll/still/{token}.jpg')
    def enroll_still_image(token: str):
        try:
            return Response(engine.still_jpeg(token), media_type='image/jpeg')
        except ValueError as exc:
            raise _error(404, exc) from exc

    @app.post('/api/enroll')
    def enroll(req: EnrollRequest):
        try:
            return engine.enroll_visual(req.token, req.rect.model_dump(), req.name, req.aliases)
        except ValueError as exc:
            raise _error(400, exc) from exc

    @app.post('/api/object/{obj_id}/views')
    def add_view(obj_id: int, req: ViewRequest):
        try:
            return engine.add_view(obj_id, req.token, req.rect.model_dump())
        except KeyError as exc:
            raise _error(404, exc) from exc
        except ValueError as exc:
            raise _error(400, exc) from exc

    @app.post('/api/tags')
    def register_tag(req: TagRequest):
        try:
            return engine.register_tag(req.marker_id, req.name, req.aliases)
        except ValueError as exc:
            raise _error(400, exc) from exc

    @app.delete('/api/object/{obj_id}')
    def forget(obj_id: int):
        try:
            return engine.forget(obj_id)
        except KeyError as exc:
            raise _error(404, exc) from exc

    @app.delete('/api/memory')
    def clear_memory():
        engine.clear_memory()
        return {'cleared': True}

    @app.get('/api/snapshot/{obj_id}')
    def snapshot(obj_id: int):
        record = engine.memory.get(obj_id)
        name = record.get('snapshot_path') if record else None
        path = engine.snapshots_dir / name if name else None
        if obj_id not in engine.catalog.objects or path is None or not path.is_file():
            raise HTTPException(404, 'No last-seen picture yet.')
        return FileResponse(path, media_type='image/jpeg', headers={'Cache-Control': 'no-store'})

    # ---- spotlight ------------------------------------------------------------
    spot = studio.spotlight

    def spot_call(fn, *args):
        try:
            return fn(*args)
        except SpotlightError as exc:
            raise _error(503 if spot.connected or 'lost' in str(exc) else 400, exc) from exc
        except ValueError as exc:
            raise _error(400, exc) from exc

    @app.get('/api/spotlight/ports')
    def spotlight_ports():
        return {'ports': list_ports()}

    @app.post('/api/spotlight/connect')
    def spotlight_connect(req: PortRequest):
        spot_call(spot.connect, req.port)
        return spot.status()

    @app.post('/api/spotlight/disconnect')
    def spotlight_disconnect():
        spot.disconnect()
        return spot.status()

    @app.post('/api/point/{obj_id}')
    def point(obj_id: int):
        try:
            target = engine.target(obj_id)
        except KeyError as exc:
            raise _error(404, exc) from exc
        except ValueError as exc:
            raise _error(400, exc) from exc
        if not spot.connected:
            raise HTTPException(400, 'Spotlight not connected. Connect it in the Spotlight panel.')
        pan, tilt = spot_call(spot.point, target['x'], target['y'])
        spot_call(spot.light, True)
        return {'pan': pan, 'tilt': tilt, 'state': target['state'],
                'calibrated': not spot.calibration.placeholder}

    @app.post('/api/spotlight/aim')
    def aim(req: AimRequest):
        if not spot.connected:
            raise HTTPException(400, 'Spotlight not connected.')
        pan, tilt = spot_call(spot.point, req.x, req.y)
        return {'pan': pan, 'tilt': tilt}

    @app.post('/api/spotlight/move')
    def move(req: MoveRequest):
        if not spot.connected:
            raise HTTPException(400, 'Spotlight not connected.')
        pan, tilt = spot_call(spot.move, req.pan, req.tilt)
        return {'pan': pan, 'tilt': tilt}

    @app.post('/api/spotlight/corner')
    def save_corner(req: CornerRequest):
        if req.corner not in CORNER_NAMES:
            raise HTTPException(400, 'Corner must be tl, tr, bl or br.')
        return {'corners': spot_call(spot.save_corner, req.corner)}

    @app.post('/api/light')
    def light(req: Toggle):
        if not spot.connected:
            raise HTTPException(400, 'Spotlight not connected.')
        spot_call(spot.light, req.enabled)
        return {'enabled': req.enabled}

    @app.post('/api/spotlight/home')
    def home_spotlight():
        if not spot.connected:
            raise HTTPException(400, 'Spotlight not connected.')
        spot_call(spot.home)
        return spot.status()

    return app


def parse_resolution(text: str) -> tuple[int, int] | None:
    if text.lower() in ('', 'auto', 'native'):
        return None
    width, height = (int(v) for v in text.lower().split('x'))
    return width, height


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--camera', default='0', help='webcam number (0, 1, ...) or a video file for demos')
    parser.add_argument('--port', type=int, default=8765)
    parser.add_argument('--serial-port', default=None, help='optional ESP32 port, e.g. COM3 or /dev/ttyUSB0')
    parser.add_argument('--data-dir', default=str(ROOT / 'data'), help='where memory and templates are stored')
    parser.add_argument('--resolution', default='960x540', help='requested camera size, e.g. 1280x720 or auto')
    parser.add_argument('--fps', type=float, default=15, help='maximum frames processed per second')
    parser.add_argument('--no-autostart', action='store_true', help='wait for "Start camera" in the browser')
    args = parser.parse_args()
    if not 1024 <= args.port <= 65535:
        parser.error('--port must be between 1024 and 65535')
    try:
        resolution = parse_resolution(args.resolution)
    except ValueError:
        parser.error('--resolution must look like 1280x720')
    data_dir = Path(args.data_dir)
    try:
        engine = TrackingEngine(data_dir)
    except RuntimeError as exc:
        raise SystemExit(f'Error: {exc}') from exc
    spotlight = Spotlight(args.serial_port, ROOT / 'spotlight.json', data_dir / 'spotlight.json')
    if spotlight.error:
        print(f'Spotlight: {spotlight.error}')
    studio = Studio(engine, parse_source(args.camera), spotlight, args.fps, resolution)
    import uvicorn
    print(f'Real-World Ctrl+F is running at http://127.0.0.1:{args.port}  (Ctrl+C to stop)')
    uvicorn.run(create_app(studio, start_camera=not args.no_autostart), host='127.0.0.1', port=args.port,
                access_log=False, timeout_graceful_shutdown=2)


if __name__ == '__main__':
    main()
