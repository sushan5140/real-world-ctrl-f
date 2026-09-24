"""The shared tracking engine used by both the browser studio (server.py) and
the desktop window (app.py).

Responsibilities (and nothing UI-specific):
  * detect printed tags and enrolled objects in a frame
  * turn noisy per-frame detections into stable "tracks" (an object must be
    seen in a few consecutive frames before it counts as visible, and must be
    missing for a short while before it counts as gone)
  * write last-seen records and last-seen crops (visual memory)
  * watch for camera movement (scene.py) and restarts
  * classify every object as VISIBLE / LAST_SEEN / UNRELIABLE / NOT_OBSERVED
  * enrolment, tag registration, forgetting, search

Frames are processed outside the state lock so API calls stay responsive.
"""
from __future__ import annotations

import os
import sys
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np

import vision
from catalog import Catalog, CatalogObject
from memory import Memory
from scene import SceneMonitor
from tracker import detect_markers

ROOT = Path(__file__).resolve().parent

VISIBLE, LAST_SEEN, UNRELIABLE, NOT_OBSERVED = 'visible', 'last_seen', 'unreliable', 'not_observed'
MAX_VIEWS = 4


@dataclass
class EngineConfig:
    confirm_tag_frames: int = 2         # consecutive detections before a tag counts as visible
    confirm_visual_frames: int = 3      # markerless matches are noisier -> need more
    lost_after: float = .8              # seconds without detection before "gone"
    lost_min_misses: int = 3            # ... and at least this many missed frames
    save_interval: float = 1.0          # seconds between memory writes while visible
    stale_after: float = 8 * 3600       # older sightings are flagged as stale
    edge_margin: float = .025           # fraction of the frame treated as "edge"
    weak_confidence: float = .55
    unregistered_ttl: float = 2.5


@dataclass
class Detection:
    object_id: int
    x: float
    y: float
    box: tuple[int, int, int, int]
    confidence: float
    kind: str                            # 'tag' | 'visual'
    outline: tuple[tuple[float, float], ...] = ()
    copies: int = 1


@dataclass
class Track:
    first_seen: float
    last_seen: float
    detection: Detection
    frame_size: tuple[int, int]
    hits: int = 1
    total_hits: int = 1
    misses: int = 0
    confirmed: bool = False
    confidence_sum: float = 0.0
    crop: np.ndarray | None = None
    last_saved: float = 0.0
    snapshot_written: bool = False


@dataclass
class FrameResult:
    time: float
    size: tuple[int, int]
    visible: dict[int, Detection] = field(default_factory=dict)
    unregistered: dict[int, Detection] = field(default_factory=dict)


class DataFolderLock:
    """Prevents two app instances from writing the same data folder (and
    fighting over the same camera). Released automatically if the process dies."""

    def __init__(self, folder: Path):
        folder.mkdir(parents=True, exist_ok=True)
        self.handle = open(folder / '.lock', 'a+')
        try:
            if sys.platform == 'win32':
                import msvcrt
                self.handle.seek(0)
                msvcrt.locking(self.handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(self.handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            self.handle.close()
            raise RuntimeError(f'Another Real-World Ctrl+F window is already using {folder}. '
                               'Close it first (only one app can use the camera and memory at a time).') from exc

    def release(self) -> None:
        try:
            if sys.platform == 'win32':
                import msvcrt
                self.handle.seek(0)
                msvcrt.locking(self.handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl
                fcntl.flock(self.handle.fileno(), fcntl.LOCK_UN)
        except OSError:
            pass
        self.handle.close()


class TrackingEngine:
    def __init__(self, data_dir: Path = ROOT / 'data', builtin: Path = ROOT / 'objects.json',
                 config: EngineConfig | None = None, lock_folder: bool = True):
        self.config = config or EngineConfig()
        self.data_dir = Path(data_dir)
        self.templates_dir = self.data_dir / 'templates'
        self.snapshots_dir = self.data_dir / 'snapshots'
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.snapshots_dir.mkdir(exist_ok=True)
        self._folder_lock = DataFolderLock(self.data_dir) if lock_folder else None
        try:
            self.lock = threading.RLock()
            self._process_lock = threading.Lock()
            self.catalog = Catalog(builtin, self.data_dir / 'catalog.json')
            self.memory = Memory(self.data_dir / 'locations.sqlite3')
            self.scene = SceneMonitor(self.data_dir / 'scene_reference.npz')
            self.matcher = vision.Matcher()
            self.session = uuid.uuid4().hex[:12]
            self.epoch = int(self.memory.get_meta('camera_epoch', '1'))
            if self.scene.reference is None and any(r.get('epoch') == self.epoch for r in self.memory.all().values()):
                # Records exist but the camera pose they belong to is unknown.
                self._new_epoch()
            self.templates: dict[int, vision.Template] = {}
            self._load_templates()
            self.tracks: dict[int, Track] = {}
            self.unregistered: dict[int, tuple[Detection, float]] = {}
            self.selected: int | None = None
            self.camera_running = False
            self.last_frame: np.ndarray | None = None
            self.last_frame_time = 0.0
            self._still: tuple[str, np.ndarray, float] | None = None
        except Exception:
            if self._folder_lock:
                self._folder_lock.release()
            raise

    # ------------------------------------------------------------------ setup
    def _load_templates(self) -> None:
        for obj_id, obj in self.catalog.visual().items():
            views, errors = [], []
            for filename in obj.views:
                try:
                    view = vision.load_view(self.templates_dir, filename)
                except (OSError, ValueError) as exc:
                    errors.append(str(exc))
                    continue
                if not (self.templates_dir / filename.replace('.png', '.mask.png')).is_file():
                    try:  # persist the derived mask for templates from older versions
                        vision.save_view(self.templates_dir, filename, view)
                    except OSError:
                        pass
                views.append(view)
            if views:
                self.templates[obj_id] = vision.Template(obj_id, views)
            else:
                # Keep the entry visible with an explanation instead of silently
                # dropping it (and later reusing its ID).
                obj.problem = ('Cannot be recognised: ' + (errors[0] if errors else 'no template image.') +
                               ' Remove it and enrol it again with a tight box around the object.')

    def _new_epoch(self) -> None:
        self.epoch += 1
        self.memory.set_meta('camera_epoch', str(self.epoch))

    def close(self) -> None:
        with self.lock:
            self.flush()
            self.memory.close()
            if self._folder_lock:
                self._folder_lock.release()
                self._folder_lock = None

    # -------------------------------------------------------------- detection
    def detect(self, frame: np.ndarray, templates: dict[int, vision.Template],
               tag_ids: set[int]) -> tuple[dict[int, Detection], dict[int, Detection]]:
        found: dict[int, Detection] = {}
        unregistered: dict[int, Detection] = {}
        for marker_id, obs in detect_markers(frame).items():
            det = Detection(marker_id, obs.x, obs.y, obs.box, 1.0, 'tag', obs.corners, obs.copies)
            (found if marker_id in tag_ids else unregistered)[marker_id] = det
        if templates:
            for obj_id, match in self.matcher.find(frame, templates).items():
                found[obj_id] = Detection(obj_id, match.x, match.y, match.box, match.confidence,
                                          'visual', match.quad)
        return found, unregistered

    def process(self, frame: np.ndarray, now: float | None = None) -> FrameResult:
        """Process one BGR frame (public so tests can use synthetic frames)."""
        now = time.time() if now is None else now
        with self._process_lock:
            with self.lock:
                templates = dict(self.templates)
                tag_ids = self.catalog.tag_ids()
            found, unregistered = self.detect(frame, templates, tag_ids)
            with self.lock:
                moving = [t.detection.box for t in self.tracks.values()]
            moving += [d.box for d in (*found.values(), *unregistered.values())]
            scene_event = self.scene.update(frame, now, moving)
            with self.lock:
                return self._update(frame, now, found, unregistered, scene_event)

    def _update(self, frame, now, found, unregistered, scene_event) -> FrameResult:
        h, w = frame.shape[:2]
        cfg = self.config
        self.camera_running = True
        self.last_frame, self.last_frame_time = frame, now
        if scene_event == 'moved':
            self._new_epoch()
        pending: dict[int, dict[str, Any]] = {}
        for obj_id, det in found.items():
            if obj_id not in self.catalog.objects:
                continue
            track = self.tracks.get(obj_id)
            if track is None or track.frame_size != (w, h):
                track = self.tracks[obj_id] = Track(now, now, det, (w, h), hits=0, total_hits=0)
            track.hits += 1
            track.total_hits += 1
            track.misses = 0
            track.last_seen = now
            track.detection = det
            track.confidence_sum += det.confidence
            track.crop = self._crop(frame, det.box)
            needed = cfg.confirm_tag_frames if det.kind == 'tag' else cfg.confirm_visual_frames
            newly_confirmed = not track.confirmed and track.hits >= needed
            if newly_confirmed:
                track.confirmed = True
            if track.confirmed and (newly_confirmed or now - track.last_saved >= cfg.save_interval):
                pending[obj_id] = self._record(track)
                if not track.snapshot_written:
                    pending[obj_id]['snapshot_path'] = self._write_snapshot(obj_id, track.crop)
                    track.snapshot_written = True
                track.last_saved = now
        for obj_id in list(self.tracks):
            if obj_id in found:
                continue
            track = self.tracks[obj_id]
            track.misses += 1
            track.hits = 0
            if not track.confirmed:
                if track.misses >= 2:
                    del self.tracks[obj_id]
                continue
            if now - track.last_seen >= cfg.lost_after and track.misses >= cfg.lost_min_misses:
                pending[obj_id] = self._final_record(obj_id, track)
                del self.tracks[obj_id]
        self.memory.record_many(pending)
        for marker_id, det in unregistered.items():
            self.unregistered[marker_id] = (det, now)
        for marker_id in [k for k, (_, t) in self.unregistered.items() if now - t > cfg.unregistered_ttl]:
            del self.unregistered[marker_id]
        result = FrameResult(now, (w, h))
        result.visible = {i: t.detection for i, t in self.tracks.items() if t.confirmed and t.misses == 0}
        result.unregistered = unregistered
        return result

    @staticmethod
    def _crop(frame: np.ndarray, box, margin: int = 30) -> np.ndarray | None:
        h, w = frame.shape[:2]
        x0, y0, x1, y1 = box
        crop = frame[max(0, y0 - margin):min(h, y1 + margin), max(0, x0 - margin):min(w, x1 + margin)]
        return crop.copy() if crop.size else None

    def _record(self, track: Track) -> dict[str, Any]:
        det = track.detection
        w, h = track.frame_size
        m = self.config.edge_margin
        x0, y0, x1, y1 = det.box
        at_edge = x0 <= m * w or y0 <= m * h or x1 >= (1 - m) * w - 1 or y1 >= (1 - m) * h - 1
        return {
            'timestamp': track.last_seen, 'first_seen': track.first_seen, 'x': det.x, 'y': det.y,
            'box': [round(x0 / w, 4), round(y0 / h, 4), round(x1 / w, 4), round(y1 / h, 4)],
            'width': w, 'height': h, 'epoch': self.epoch, 'session': self.session,
            'at_edge': int(at_edge), 'confidence': round(track.confidence_sum / max(1, track.total_hits), 3),
            'hits': track.total_hits, 'copies': det.copies, 'kind': det.kind, 'snapshot_path': None,
        }

    def _final_record(self, obj_id: int, track: Track) -> dict[str, Any]:
        record = self._record(track)
        record['snapshot_path'] = self._write_snapshot(obj_id, track.crop)
        return record

    def _write_snapshot(self, obj_id: int, crop: np.ndarray | None) -> str | None:
        if crop is None:
            return None
        name = f'{obj_id}.jpg'
        path = self.snapshots_dir / name
        tmp = self.snapshots_dir / f'{obj_id}.tmp.jpg'
        try:
            if cv2.imwrite(str(tmp), crop, [cv2.IMWRITE_JPEG_QUALITY, 85]):
                os.replace(tmp, path)
                return name
        except (OSError, cv2.error):
            pass
        tmp.unlink(missing_ok=True)
        return None

    def flush(self) -> None:
        """Camera stopped / app closing: everything visible becomes 'last seen'."""
        with self.lock:
            pending = {i: self._final_record(i, t) for i, t in self.tracks.items() if t.confirmed}
            self.memory.record_many(pending)
            self.tracks.clear()
            self.unregistered.clear()
            self.camera_running = False

    # ----------------------------------------------------------------- status
    def object_state(self, obj: CatalogObject, now: float | None = None) -> dict[str, Any]:
        now = time.time() if now is None else now
        with self.lock:
            track = self.tracks.get(obj.id)
            record = self.memory.get(obj.id)
            live = track.detection if (track and track.confirmed and self.camera_running) else None
            reasons, notes = [], []
            if live:
                state = VISIBLE
            elif record is None:
                state = NOT_OBSERVED
            else:
                if record.get('epoch', 0) == 0:
                    reasons.append(('legacy', 'Recorded by an older version that could not check whether '
                                              'the camera moved afterwards.'))
                elif record['epoch'] != self.epoch:
                    reasons.append(('camera_moved', 'The camera has moved since this sighting, so the marked '
                                                    'spot may not match the picture any more.'))
                if record.get('at_edge'):
                    reasons.append(('left_view', 'It was last seen at the edge of the picture, so it was '
                                                 'probably carried out of view.'))
                age = now - record['timestamp']
                if age > self.config.stale_after:
                    reasons.append(('stale', f'This sighting is more than {round(self.config.stale_after / 3600)}'
                                             ' hours old.'))
                if record.get('kind') == 'visual' and ((record.get('confidence') or 0) < self.config.weak_confidence
                                                       or (record.get('hits') or 0) < 5):
                    reasons.append(('weak', 'It was only briefly or weakly recognised.'))
                if (record.get('copies') or 1) > 1:
                    reasons.append(('duplicate_tag', 'Several copies of this tag were visible at once.'))
                if record.get('session') and record['session'] != self.session:
                    notes.append(('restarted', 'The app was restarted after this sighting, so it was not '
                                               'watching the whole time.'))
                if not self.camera_running:
                    notes.append(('camera_off', 'The camera is off, so nothing is being watched right now.'))
                elif record.get('epoch') == self.epoch and not self.scene.verified:
                    notes.append(('checking', 'Checking that the camera is still in the same position…'))
                state = UNRELIABLE if reasons else LAST_SEEN
            item = {
                'id': obj.id, 'name': obj.name, 'aliases': obj.aliases, 'kind': obj.kind,
                'builtin': obj.builtin, 'views': len(obj.views), 'problem': obj.problem,
                'state': state, 'visible': state == VISIBLE,
                'reasons': [{'code': c, 'text': t} for c, t in reasons],
                'notes': [{'code': c, 'text': t} for c, t in notes],
                'last_seen': record['timestamp'] if record else None,
                'first_seen': record.get('first_seen') if record else None,
                'age_seconds': max(0, round(now - record['timestamp'])) if record else None,
                'x': record['x'] if record else None, 'y': record['y'] if record else None,
                'box': record.get('box') if record else None,
                'confidence': record.get('confidence') if record else None,
                'snapshot': (f'/api/snapshot/{obj.id}?v={int(record["timestamp"])}'
                             if record and record.get('snapshot_path') and
                             (self.snapshots_dir / record['snapshot_path']).is_file() else None),
                'live': None,
            }
            if live:
                w, h = track.frame_size
                x0, y0, x1, y1 = live.box
                item['live'] = {'x': live.x, 'y': live.y, 'confidence': live.confidence,
                                'box': [x0 / w, y0 / h, x1 / w, y1 / h]}
                item['age_seconds'] = 0
            return item

    def status(self) -> dict[str, Any]:
        now = time.time()
        with self.lock:
            items = [self.object_state(obj, now) for obj in self.catalog.objects.values()]
            order = {VISIBLE: 0, LAST_SEEN: 1, UNRELIABLE: 2, NOT_OBSERVED: 3}
            items.sort(key=lambda i: (order[i['state']], i['kind'] != 'tag', i['name'].casefold()))
            frame_size = self.last_frame.shape[1::-1] if self.last_frame is not None else None
            unregistered = []
            for marker_id, (det, seen) in sorted(self.unregistered.items()):
                w, h = frame_size or (1, 1)
                unregistered.append({'marker_id': marker_id, 'x': det.x, 'y': det.y,
                                     'box': [det.box[0] / w, det.box[1] / h, det.box[2] / w, det.box[3] / h]})
            return {
                'selected': self.selected, 'items': items, 'unregistered_tags': unregistered,
                'scene': {'state': self.scene.state, 'verified': self.scene.verified, 'epoch': self.epoch,
                          'shift': round(self.scene.shift, 4)},
                'stale_after_hours': round(self.config.stale_after / 3600, 1),
            }

    # ----------------------------------------------------------------- search
    def search(self, query: str) -> dict[str, Any]:
        with self.lock:
            ranked = self.catalog.search(query)
            if not ranked:
                raise LookupError('No registered object matches that search. Try another name, or register it first.')
            best_score = ranked[0][0]
            # Near-equal scores (only the small length tie-breaker differs) are a real tie.
            top = [obj_id for score, obj_id in ranked if score >= best_score - 1.5]
            if len(top) > 1:
                return {'id': None, 'ambiguous': True,
                        'candidates': [{'id': i, 'name': self.catalog.objects[i].name} for i in top]}
            self.selected = top[0]
            return {'id': top[0], 'ambiguous': False, 'candidates': []}

    def select(self, obj_id: int | None) -> None:
        with self.lock:
            if obj_id is not None and obj_id not in self.catalog.objects:
                raise KeyError('Unknown object.')
            self.selected = obj_id

    def target(self, obj_id: int) -> dict[str, Any]:
        """Where the spotlight should aim: live position if visible, else memory."""
        with self.lock:
            obj = self.catalog.get(obj_id)
            if obj is None:
                raise KeyError('Unknown object.')
            state = self.object_state(obj)
            if state['live']:
                return {'x': state['live']['x'], 'y': state['live']['y'], 'state': state['state']}
            if state['x'] is None:
                raise ValueError(f'{obj.name} has never been seen, so there is nowhere to point.')
            return {'x': state['x'], 'y': state['y'], 'state': state['state']}

    # ------------------------------------------------------------ enrolment
    def capture_still(self) -> dict[str, Any]:
        """Freeze the current frame so the user can draw on a still image."""
        with self.lock:
            if self.last_frame is None or not self.camera_running:
                raise ValueError('No live camera picture yet. Start the camera first.')
            token = uuid.uuid4().hex
            self._still = (token, self.last_frame.copy(), time.time())
            h, w = self.last_frame.shape[:2]
            return {'token': token, 'width': w, 'height': h}

    def still_jpeg(self, token: str) -> bytes:
        frame = self._get_still(token)
        ok, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
        if not ok:
            raise ValueError('Could not encode the still frame.')
        return buffer.tobytes()

    def _get_still(self, token: str) -> np.ndarray:
        with self.lock:
            if not self._still or self._still[0] != token or time.time() - self._still[2] > 900:
                raise ValueError('That still picture has expired. Start enrolment again.')
            return self._still[1]

    def _crop_rect(self, frame: np.ndarray, rect: dict[str, float]) -> np.ndarray:
        x, y, rw, rh = (float(rect[k]) for k in ('x', 'y', 'w', 'h'))
        if not (0 <= x <= 1 and 0 <= y <= 1 and 0 < rw <= 1 and 0 < rh <= 1):
            raise ValueError('The selection must lie inside the camera picture.')
        if x + rw > 1.000001 or y + rh > 1.000001:
            raise ValueError('The selection extends beyond the camera picture.')
        h, w = frame.shape[:2]
        x0, y0 = round(x * w), round(y * h)
        x1, y1 = round((x + rw) * w), round((y + rh) * h)
        return frame[y0:y1, x0:x1]

    def enroll_visual(self, token: str, rect: dict[str, float], name: str, aliases=()) -> dict[str, Any]:
        frame = self._get_still(token)
        view = vision.build_view(self._crop_rect(frame, rect))   # slow part, outside the lock
        with self.lock:
            obj = self.catalog.add_visual(name, aliases)
            filename = vision.view_filename(obj.id, 0)
            try:
                vision.save_view(self.templates_dir, filename, view)
                obj.views = [filename]
                self.catalog.save()
            except Exception:
                self.catalog.objects.pop(obj.id, None)
                vision.delete_view_files(self.templates_dir, filename)
                raise
            self.memory.delete(obj.id)
            (self.snapshots_dir / f'{obj.id}.jpg').unlink(missing_ok=True)
            self.templates[obj.id] = vision.Template(obj.id, [view])
            self.selected = obj.id
            return {'id': obj.id, 'name': obj.name, 'quality': view.quality,
                    'features': len(view.keypoints), 'mask': view.mask_method}

    def add_view(self, obj_id: int, token: str, rect: dict[str, float]) -> dict[str, Any]:
        frame = self._get_still(token)
        view = vision.build_view(self._crop_rect(frame, rect))
        with self.lock:
            obj = self.catalog.get(obj_id)
            if obj is None or obj.kind != 'visual':
                raise KeyError('Only enrolled (markerless) objects can get extra views.')
            if len(obj.views) >= MAX_VIEWS:
                raise ValueError(f'An object can have at most {MAX_VIEWS} views.')
            used = set(obj.views)
            index = next(i for i in range(1, 100) if vision.view_filename(obj_id, i) not in used)
            filename = vision.view_filename(obj_id, index)
            vision.save_view(self.templates_dir, filename, view)
            obj.views = [*obj.views, filename]
            try:
                self.catalog.save()
            except Exception:
                obj.views = obj.views[:-1]
                vision.delete_view_files(self.templates_dir, filename)
                raise
            template = self.templates.get(obj_id) or vision.Template(obj_id, [])
            self.templates[obj_id] = vision.Template(obj_id, [*template.views, view])
            obj.problem = None          # a fresh, usable view repairs a broken entry
            return {'id': obj_id, 'views': len(obj.views), 'quality': view.quality,
                    'features': len(view.keypoints)}

    def register_tag(self, marker_id: int, name: str, aliases=()) -> dict[str, Any]:
        with self.lock:
            obj = self.catalog.add_tag(marker_id, name, aliases)
            try:
                self.catalog.save()
            except Exception:
                self.catalog.objects.pop(marker_id, None)
                raise
            self.memory.delete(marker_id)
            (self.snapshots_dir / f'{marker_id}.jpg').unlink(missing_ok=True)
            self.unregistered.pop(marker_id, None)
            self.selected = marker_id
            return {'id': marker_id, 'name': obj.name}

    def forget(self, obj_id: int) -> dict[str, Any]:
        """Clear an object's history; user-created objects are removed entirely."""
        with self.lock:
            obj = self.catalog.get(obj_id)
            if obj is None:
                raise KeyError('Unknown object.')
            self.memory.delete(obj_id)
            (self.snapshots_dir / f'{obj_id}.jpg').unlink(missing_ok=True)
            self.tracks.pop(obj_id, None)
            removed = False
            if not obj.builtin:
                self.catalog.remove(obj_id)
                try:
                    self.catalog.save()
                except Exception:
                    self.catalog.objects[obj_id] = obj
                    raise
                for filename in obj.views:
                    vision.delete_view_files(self.templates_dir, filename)
                self.templates.pop(obj_id, None)
                removed = True
            if self.selected == obj_id:
                self.selected = None
            return {'id': obj_id, 'removed': removed}

    def clear_memory(self) -> None:
        """Delete every last-seen record and crop (the object library stays)."""
        with self.lock:
            self.memory.clear()
            for path in self.snapshots_dir.glob('*.jpg'):
                path.unlink(missing_ok=True)
            self.tracks.clear()

    # ------------------------------------------------------------- drawing
    def annotate(self, frame: np.ndarray, result: FrameResult) -> np.ndarray:
        """Boxes and names drawn onto the frame (kept in sync with the video)."""
        canvas = frame.copy()
        with self.lock:
            names = {i: o.name for i, o in self.catalog.objects.items()}
            selected = self.selected
        for obj_id, det in result.visible.items():
            active = obj_id == selected
            color = (84, 125, 231) if active else (104, 176, 118)
            if len(det.outline) == 4:
                pts = np.int32(det.outline).reshape(-1, 1, 2)
                cv2.polylines(canvas, [pts], True, color, 3 if active else 2, cv2.LINE_AA)
            else:
                x0, y0, x1, y1 = det.box
                cv2.rectangle(canvas, (x0, y0), (x1, y1), color, 3 if active else 2, cv2.LINE_AA)
            label = names.get(obj_id, f'#{obj_id}')
            _label(canvas, label, det.box[0], det.box[1], color)
        for marker_id, det in result.unregistered.items():
            x0, y0, x1, y1 = det.box
            cv2.rectangle(canvas, (x0, y0), (x1, y1), (170, 170, 170), 1, cv2.LINE_AA)
            _label(canvas, f'Tag #{marker_id} - not registered', x0, y0, (120, 120, 120))
        return canvas


def _label(canvas: np.ndarray, text: str, x: int, y: int, color) -> None:
    text = text.encode('ascii', 'replace').decode('ascii')   # OpenCV fonts are ASCII only
    font, scale, thick = cv2.FONT_HERSHEY_SIMPLEX, .55, 1
    (tw, th), base = cv2.getTextSize(text, font, scale, thick)
    top = max(0, y - th - base - 8)
    cv2.rectangle(canvas, (x, top), (x + tw + 10, top + th + base + 8), color, -1)
    cv2.putText(canvas, text, (x + 5, top + th + 4), font, scale, (255, 255, 255), thick, cv2.LINE_AA)
