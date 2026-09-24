"""Printed-tag (ArUco) detection, plus small compatibility helpers.

Tags use the DICT_4X4_50 dictionary (IDs 0..49). A tag identifies the
*card*, so it only tracks an object when the card is attached to it.

The helpers at the bottom (load_objects, init_db, save_observation,
get_last_seen, match_object) keep the original v0.1/v0.2 function names
working; new code uses catalog.py / memory.py / engine.py directly.
"""
from __future__ import annotations

import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

import catalog as _catalog


@dataclass(frozen=True)
class Observation:
    marker_id: int
    x: float
    y: float
    box: tuple[int, int, int, int]
    copies: int = 1                      # >1: the same tag is visible several times
    corners: tuple[tuple[float, float], ...] = ()


_DICTIONARY = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
_DETECTOR = None


def _detector():
    global _DETECTOR
    if _DETECTOR is None and hasattr(cv2.aruco, 'ArucoDetector'):
        params = cv2.aruco.DetectorParameters()
        params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
        _DETECTOR = cv2.aruco.ArucoDetector(_DICTIONARY, params)
    return _DETECTOR


def detect_markers(frame: np.ndarray) -> dict[int, Observation]:
    """Detect all DICT_4X4_50 tags. If the same ID appears several times, the
    largest instance is reported and `copies` says how many were seen."""
    h, w = frame.shape[:2]
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
    detector = _detector()
    if detector is not None:
        corners, ids, _ = detector.detectMarkers(gray)
    else:  # OpenCV < 4.7
        corners, ids, _ = cv2.aruco.detectMarkers(gray, _DICTIONARY)
    result: dict[int, Observation] = {}
    if ids is None:
        return result
    areas: dict[int, float] = {}
    counts: dict[int, int] = {}
    for marker_id, corner_set in zip(ids.flatten(), corners):
        marker_id = int(marker_id)
        points = corner_set.reshape(4, 2)
        area = abs(float(cv2.contourArea(points.astype(np.float32))))
        counts[marker_id] = counts.get(marker_id, 0) + 1
        if area <= areas.get(marker_id, -1):
            continue
        areas[marker_id] = area
        xmin, ymin = np.min(points, axis=0)
        xmax, ymax = np.max(points, axis=0)
        cx, cy = points.mean(axis=0)
        result[marker_id] = Observation(
            marker_id, float(cx / w), float(cy / h),
            (int(xmin), int(ymin), int(xmax), int(ymax)),
            corners=tuple((float(px), float(py)) for px, py in points))
    return {i: Observation(o.marker_id, o.x, o.y, o.box, counts[i], o.corners) for i, o in result.items()}


# ---------------------------------------------------------------------------
# Compatibility helpers (original API)
# ---------------------------------------------------------------------------
def load_objects(path: Path) -> dict[int, dict[str, Any]]:
    return {i: {'name': o.name, 'aliases': list(o.aliases)} for i, o in _catalog.load_builtin(path).items()}


def match_object(query: str, objects: dict[int, dict[str, Any]]) -> int | None:
    """Best matching object id for a query, or None. See catalog.search."""
    entries = {i: _catalog.CatalogObject(i, info['name'], list(info.get('aliases', [])))
               for i, info in objects.items()}
    ranked = _catalog.search(query, entries)
    return ranked[0][1] if ranked else None


def init_db(path: Path) -> sqlite3.Connection:
    from memory import Memory
    return Memory(path).conn


def save_observation(conn: sqlite3.Connection, obs: Observation, width: int, height: int,
                     snapshot_path: str | None = None) -> None:
    conn.execute("""INSERT INTO last_seen (marker_id, timestamp, x, y, width, height, snapshot_path)
      VALUES (?, ?, ?, ?, ?, ?, ?)
      ON CONFLICT(marker_id) DO UPDATE SET
      timestamp=excluded.timestamp, x=excluded.x, y=excluded.y,
      width=excluded.width, height=excluded.height,
      snapshot_path=COALESCE(excluded.snapshot_path, last_seen.snapshot_path)""",
                 (obs.marker_id, time.time(), obs.x, obs.y, width, height, snapshot_path))
    conn.commit()


def get_last_seen(conn: sqlite3.Connection, marker_id: int) -> dict[str, Any] | None:
    row = conn.execute('SELECT timestamp, x, y, width, height, snapshot_path FROM last_seen WHERE marker_id=?',
                       (marker_id,)).fetchone()
    if row is None:
        return None
    return dict(zip(('timestamp', 'x', 'y', 'width', 'height', 'snapshot_path'), row))
