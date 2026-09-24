"""Persistent visual memory (SQLite): where each object was last seen.

One row per object (the latest sighting). Rows are updated in place, so
continuous observation never piles up duplicate records. Every row stores
enough context to judge how trustworthy it is later:

  timestamp     when the object was last actually detected (frame time)
  first_seen    when the current/last continuous sighting started
  x, y          normalised centre (0..1) in the camera image
  box           normalised bounding box [x0, y0, x1, y1]
  width/height  camera resolution at the time
  epoch         camera-position epoch (changes when the camera is moved);
                0 = recorded by an older version without this check
  session       app run that recorded it (restart detection)
  at_edge       last seen touching the image border (probably carried out)
  confidence    detection confidence (1.0 for printed tags)
  hits          frames the object was detected in during that sighting
  copies        >1 when several identical tags were visible at once
  snapshot_path file name of the last-seen crop inside data/snapshots/
"""
from __future__ import annotations

import json
import sqlite3
import threading
from pathlib import Path
from typing import Any

SCHEMA_VERSION = 2

_COLUMNS = {
    # name: SQL type/default for columns added after version 1
    'first_seen': 'REAL',
    'box': 'TEXT',
    'epoch': 'INTEGER NOT NULL DEFAULT 0',
    'session': 'TEXT',
    'at_edge': 'INTEGER NOT NULL DEFAULT 0',
    'confidence': 'REAL',
    'hits': 'INTEGER NOT NULL DEFAULT 1',
    'copies': 'INTEGER NOT NULL DEFAULT 1',
    'kind': 'TEXT',
}
FIELDS = ('timestamp', 'x', 'y', 'width', 'height', 'snapshot_path', *_COLUMNS)


class Memory:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self._lock = threading.RLock()
        self.conn = sqlite3.connect(str(path), check_same_thread=False, timeout=5)
        self.conn.execute('PRAGMA journal_mode=WAL')
        self.conn.execute('PRAGMA synchronous=NORMAL')
        self._migrate()
        self._cache: dict[int, dict[str, Any]] = {}
        for row in self.conn.execute(f'SELECT marker_id, {", ".join(FIELDS)} FROM last_seen'):
            self._cache[int(row[0])] = self._row_to_dict(row[1:])

    def _migrate(self) -> None:
        with self.conn:
            self.conn.execute("""CREATE TABLE IF NOT EXISTS last_seen (
                marker_id INTEGER PRIMARY KEY,
                timestamp REAL NOT NULL,
                x REAL NOT NULL,
                y REAL NOT NULL,
                width INTEGER NOT NULL,
                height INTEGER NOT NULL,
                snapshot_path TEXT)""")
            self.conn.execute('CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT NOT NULL)')
            existing = {r[1] for r in self.conn.execute('PRAGMA table_info(last_seen)')}
            for name, decl in _COLUMNS.items():
                if name not in existing:
                    self.conn.execute(f'ALTER TABLE last_seen ADD COLUMN {name} {decl}')
            self.conn.execute("INSERT OR REPLACE INTO meta VALUES ('schema_version', ?)", (str(SCHEMA_VERSION),))

    @staticmethod
    def _row_to_dict(values) -> dict[str, Any]:
        record = dict(zip(FIELDS, values))
        if record.get('box'):
            try:
                record['box'] = json.loads(record['box'])
            except ValueError:
                record['box'] = None
        # Version 1 stored absolute paths, which break when the project folder
        # is moved or renamed. Only the file name is meaningful.
        if record.get('snapshot_path'):
            record['snapshot_path'] = Path(str(record['snapshot_path']).replace('\\', '/')).name
        return record

    # ---- meta ---------------------------------------------------------------
    def get_meta(self, key: str, default: str | None = None) -> str | None:
        with self._lock:
            row = self.conn.execute('SELECT value FROM meta WHERE key=?', (key,)).fetchone()
            return row[0] if row else default

    def set_meta(self, key: str, value: str) -> None:
        with self._lock, self.conn:
            self.conn.execute('INSERT OR REPLACE INTO meta VALUES (?, ?)', (key, value))

    # ---- records ------------------------------------------------------------
    def get(self, obj_id: int) -> dict[str, Any] | None:
        with self._lock:
            record = self._cache.get(obj_id)
            return dict(record) if record else None

    def all(self) -> dict[int, dict[str, Any]]:
        with self._lock:
            return {k: dict(v) for k, v in self._cache.items()}

    def record_many(self, records: dict[int, dict[str, Any]]) -> None:
        """Upsert several sightings in one transaction."""
        if not records:
            return
        with self._lock:
            rows = []
            for obj_id, rec in records.items():
                merged = {**(self._cache.get(obj_id) or {}), **rec}
                if merged.get('snapshot_path') is None and obj_id in self._cache:
                    merged['snapshot_path'] = self._cache[obj_id].get('snapshot_path')
                rows.append((obj_id, *[json.dumps(merged.get(f)) if f == 'box' and merged.get(f) is not None
                                       else merged.get(f) for f in FIELDS]))
                self._cache[obj_id] = merged
            placeholders = ', '.join('?' * (len(FIELDS) + 1))
            with self.conn:
                self.conn.executemany(
                    f'INSERT OR REPLACE INTO last_seen (marker_id, {", ".join(FIELDS)}) VALUES ({placeholders})', rows)

    def record(self, obj_id: int, rec: dict[str, Any]) -> None:
        self.record_many({obj_id: rec})

    def delete(self, obj_id: int) -> None:
        with self._lock, self.conn:
            self.conn.execute('DELETE FROM last_seen WHERE marker_id=?', (obj_id,))
            self._cache.pop(obj_id, None)

    def clear(self) -> None:
        with self._lock, self.conn:
            self.conn.execute('DELETE FROM last_seen')
            self._cache.clear()

    def close(self) -> None:
        with self._lock:
            try:
                self.conn.close()
            except sqlite3.Error:
                pass
