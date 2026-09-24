"""Local visual-object tracking with persistent last-seen records.

Version 0.1 tracks printed ArUco tags attached to objects, not arbitrary
untagged objects.
"""
from __future__ import annotations

import json
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np


@dataclass(frozen=True)
class Observation:
    marker_id: int
    x: float
    y: float
    box: tuple[int, int, int, int]


def load_objects(path: Path) -> dict[int, dict[str, Any]]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    objects: dict[int, dict[str, Any]] = {}
    for key, item in raw.items():
        marker = int(key)
        if not 0 <= marker <= 49:
            raise ValueError("Marker IDs must be between 0 and 49 for DICT_4X4_50")
        if not isinstance(item.get("name"), str) or not item["name"].strip():
            raise ValueError(f"Object {marker} must have a name")
        item.setdefault("aliases", [])
        objects[marker] = item
    return objects


def detect_markers(frame: np.ndarray) -> dict[int, Observation]:
    h, w = frame.shape[:2]
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    if hasattr(cv2.aruco, "ArucoDetector"):
        corners, ids, _ = cv2.aruco.ArucoDetector(dictionary).detectMarkers(gray)
    else:
        corners, ids, _ = cv2.aruco.detectMarkers(gray, dictionary)
    result: dict[int, Observation] = {}
    if ids is None:
        return result
    for marker_id, corner_set in zip(ids.flatten(), corners):
        points = corner_set.reshape(4, 2)
        xmin, ymin = np.min(points, axis=0)
        xmax, ymax = np.max(points, axis=0)
        cx, cy = points.mean(axis=0)
        result[int(marker_id)] = Observation(
            int(marker_id), float(cx / w), float(cy / h),
            (int(xmin), int(ymin), int(xmax), int(ymax)),
        )
    return result


def init_db(path: Path) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    # The browser studio handles writes/reads from different threads under
    # Engine.lock; the minimal OpenCV V0.1 app remains single-threaded.
    conn = sqlite3.connect(str(path), check_same_thread=False)
    conn.execute("""CREATE TABLE IF NOT EXISTS last_seen (
        marker_id INTEGER PRIMARY KEY,
        timestamp REAL NOT NULL,
        x REAL NOT NULL,
        y REAL NOT NULL,
        width INTEGER NOT NULL,
        height INTEGER NOT NULL,
        snapshot_path TEXT
    )""")
    conn.commit()
    return conn


def save_observation(conn: sqlite3.Connection, obs: Observation,
                     width: int, height: int, snapshot_path: str | None = None) -> None:
    conn.execute("""INSERT INTO last_seen
      (marker_id, timestamp, x, y, width, height, snapshot_path)
      VALUES (?, ?, ?, ?, ?, ?, ?)
      ON CONFLICT(marker_id) DO UPDATE SET
      timestamp=excluded.timestamp, x=excluded.x, y=excluded.y,
      width=excluded.width, height=excluded.height,
      snapshot_path=COALESCE(excluded.snapshot_path, last_seen.snapshot_path)
    """, (obs.marker_id, time.time(), obs.x, obs.y,
           width, height, snapshot_path))
    conn.commit()


def get_last_seen(conn: sqlite3.Connection, marker_id: int) -> dict[str, Any] | None:
    row = conn.execute("""SELECT timestamp, x, y, width, height, snapshot_path
                          FROM last_seen WHERE marker_id=?""", (marker_id,)).fetchone()
    if row is None:
        return None
    return dict(zip(("timestamp", "x", "y", "width", "height", "snapshot_path"), row))


def match_object(query: str, objects: dict[int, dict[str, Any]]) -> int | None:
    """Simple, auditable local name/alias match; no internet or AI required."""
    query = query.lower().strip()
    if not query:
        return None
    matches = []
    for obj_id, info in objects.items():
        names = [info["name"], *info.get("aliases", [])]
        if any(name.lower() in query for name in names):
            matches.append((max(len(name) for name in names if name.lower() in query), obj_id))
    return max(matches)[1] if matches else None
