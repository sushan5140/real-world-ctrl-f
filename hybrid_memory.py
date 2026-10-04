"""Semantic spatial memory for Grok-assisted laptop/mobile scans.

This deliberately lives beside the deterministic tracking engine rather than
pretending semantic recognition and instance tracking are the same problem.
"""
from __future__ import annotations

import json
import re
import sqlite3
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from grok_vision import SemanticObject, SemanticScan

SCHEMA_VERSION = 1


def _norm(text: str) -> str:
    return " ".join(re.findall(r"[\w]+", str(text).casefold()))


@dataclass(frozen=True)
class SemanticHit:
    object_id: int
    name: str
    category: str
    description: str
    aliases: tuple[str, ...]
    signature: str
    last_seen: float
    source: str
    zone: str
    location_hint: str
    confidence: float
    evidence_path: str | None
    scene: str
    model: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.object_id,
            "name": self.name,
            "category": self.category,
            "description": self.description,
            "aliases": list(self.aliases),
            "signature": self.signature,
            "last_seen": self.last_seen,
            "age_seconds": max(0, round(time.time() - self.last_seen)),
            "source": self.source,
            "zone": self.zone,
            "location_hint": self.location_hint,
            "confidence": self.confidence,
            "evidence": self.evidence_path,
            "scene": self.scene,
            "model": self.model,
        }


class HybridMemory:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self._lock = threading.RLock()
        self.conn = sqlite3.connect(str(path), check_same_thread=False, timeout=5)
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA synchronous=NORMAL")
        self._migrate()

    def _migrate(self) -> None:
        with self.conn:
            self.conn.execute("""CREATE TABLE IF NOT EXISTS semantic_objects (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                signature TEXT NOT NULL UNIQUE,
                name TEXT NOT NULL,
                category TEXT NOT NULL,
                description TEXT NOT NULL,
                aliases TEXT NOT NULL DEFAULT '[]',
                created REAL NOT NULL,
                updated REAL NOT NULL
            )""")
            self.conn.execute("""CREATE TABLE IF NOT EXISTS semantic_observations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                object_id INTEGER NOT NULL REFERENCES semantic_objects(id) ON DELETE CASCADE,
                timestamp REAL NOT NULL,
                source TEXT NOT NULL,
                zone TEXT NOT NULL,
                location_hint TEXT NOT NULL,
                confidence REAL NOT NULL,
                evidence_path TEXT,
                scene TEXT NOT NULL,
                model TEXT NOT NULL
            )""")
            self.conn.execute("CREATE INDEX IF NOT EXISTS semantic_obs_object_time ON semantic_observations(object_id, timestamp DESC)")
            self.conn.execute("""CREATE TABLE IF NOT EXISTS hybrid_meta (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            )""")
            self.conn.execute("INSERT OR REPLACE INTO hybrid_meta VALUES ('schema_version', ?)", (str(SCHEMA_VERSION),))

    def ingest(self, scan: SemanticScan, source: str, zone: str,
               evidence_names: dict[int, str] | None = None,
               when: float | None = None) -> list[SemanticHit]:
        source = " ".join(str(source).split())[:40] or "scan"
        zone = " ".join(str(zone).split())[:80] or "unspecified"
        when = float(when or time.time())
        evidence_names = evidence_names or {}
        hits: list[SemanticHit] = []
        with self._lock, self.conn:
            for index, obj in enumerate(scan.objects):
                signature = self._signature(obj)
                row = self.conn.execute(
                    "SELECT id, name, category, description, aliases FROM semantic_objects WHERE signature=?",
                    (signature,),
                ).fetchone()
                if row:
                    object_id = int(row[0])
                    aliases = self._merge_aliases(json.loads(row[4] or "[]"), obj.aliases, row[1], obj.name)
                    name = obj.name if obj.confidence >= .55 else row[1]
                    category = obj.category if obj.confidence >= .55 else row[2]
                    description = obj.description if obj.confidence >= .55 else row[3]
                    self.conn.execute("""UPDATE semantic_objects SET name=?, category=?, description=?,
                        aliases=?, updated=? WHERE id=?""",
                        (name, category, description, json.dumps(aliases), when, object_id))
                else:
                    aliases = self._merge_aliases([], obj.aliases, obj.name)
                    cur = self.conn.execute("""INSERT INTO semantic_objects
                        (signature, name, category, description, aliases, created, updated)
                        VALUES (?, ?, ?, ?, ?, ?, ?)""",
                        (signature, obj.name, obj.category, obj.description, json.dumps(aliases), when, when))
                    object_id = int(cur.lastrowid)
                    name, category, description = obj.name, obj.category, obj.description
                evidence = evidence_names.get(index)
                self.conn.execute("""INSERT INTO semantic_observations
                    (object_id, timestamp, source, zone, location_hint, confidence, evidence_path, scene, model)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (object_id, when, source, zone, obj.location_hint, obj.confidence,
                     evidence, scan.scene, scan.model))
                hits.append(SemanticHit(
                    object_id, name, category, description, tuple(aliases), signature,
                    when, source, zone, obj.location_hint, obj.confidence, evidence,
                    scan.scene, scan.model,
                ))
        return hits

    @staticmethod
    def _signature(obj: SemanticObject) -> str:
        signature = _norm(obj.instance_signature).replace(" ", "_")
        category = _norm(obj.category).replace(" ", "_")
        name = _norm(obj.name).replace(" ", "_")
        return (f"{category}:{signature or name}")[:180]

    @staticmethod
    def _merge_aliases(existing: Iterable[str], new: Iterable[str], *names: str) -> list[str]:
        out, seen = [], set()
        for value in [*existing, *new, *names]:
            value = " ".join(str(value).split())[:60]
            key = value.casefold()
            if len(value) >= 2 and key not in seen:
                seen.add(key)
                out.append(value)
        return out[:12]

    def latest(self, limit: int = 100) -> list[dict[str, Any]]:
        limit = max(1, min(500, int(limit)))
        with self._lock:
            rows = self.conn.execute("""SELECT o.id, o.name, o.category, o.description, o.aliases, o.signature,
                    x.timestamp, x.source, x.zone, x.location_hint, x.confidence, x.evidence_path, x.scene, x.model
                FROM semantic_objects o
                JOIN semantic_observations x ON x.id = (
                    SELECT id FROM semantic_observations WHERE object_id=o.id ORDER BY timestamp DESC, id DESC LIMIT 1
                )
                ORDER BY x.timestamp DESC LIMIT ?""", (limit,)).fetchall()
        return [self._row_hit(row).to_dict() for row in rows]

    def search(self, query: str, limit: int = 12) -> list[dict[str, Any]]:
        tokens = [t for t in _norm(query).split() if len(t) > 1]
        if not tokens:
            return []
        candidates = self.latest(500)
        scored = []
        for item in candidates:
            hay = _norm(" ".join([item["name"], item["category"], item["description"], *item["aliases"]]))
            words = set(hay.split())
            exact = sum(1 for t in tokens if t in words)
            partial = sum(1 for t in tokens if len(t) >= 4 and any(w.startswith(t) or t.startswith(w) for w in words))
            score = exact * 10 + partial * 2
            if score:
                scored.append((score, item["last_seen"], item))
        scored.sort(key=lambda x: (-x[0], -x[1]))
        return [item for _, _, item in scored[:max(1, min(50, limit))]]

    def history(self, object_id: int, limit: int = 30) -> list[dict[str, Any]]:
        with self._lock:
            obj = self.conn.execute(
                "SELECT id, name, category, description, aliases, signature FROM semantic_objects WHERE id=?",
                (int(object_id),),
            ).fetchone()
            if not obj:
                return []
            rows = self.conn.execute("""SELECT timestamp, source, zone, location_hint, confidence,
                    evidence_path, scene, model FROM semantic_observations
                WHERE object_id=? ORDER BY timestamp DESC, id DESC LIMIT ?""",
                (int(object_id), max(1, min(200, int(limit))))).fetchall()
        aliases = tuple(json.loads(obj[4] or "[]"))
        return [{
            "id": obj[0], "name": obj[1], "category": obj[2], "description": obj[3],
            "aliases": list(aliases), "signature": obj[5], "last_seen": row[0],
            "source": row[1], "zone": row[2], "location_hint": row[3],
            "confidence": row[4], "evidence": row[5], "scene": row[6], "model": row[7],
        } for row in rows]

    def clear(self) -> None:
        with self._lock, self.conn:
            self.conn.execute("DELETE FROM semantic_observations")
            self.conn.execute("DELETE FROM semantic_objects")

    def close(self) -> None:
        with self._lock:
            try:
                self.conn.close()
            except sqlite3.Error:
                pass

    @staticmethod
    def _row_hit(row) -> SemanticHit:
        return SemanticHit(
            int(row[0]), row[1], row[2], row[3], tuple(json.loads(row[4] or "[]")), row[5],
            float(row[6]), row[7], row[8], row[9], float(row[10]), row[11], row[12], row[13],
        )
