"""Hybrid semantic discovery runtime.

Laptop camera -> optional sampled stills -> Grok -> semantic memory
Mobile room sweep -> selected stills -> Grok -> same semantic memory

The deterministic engine remains the source of truth for precise desk
coordinates. Semantic memory is a separate, explicitly less-certain layer.
"""
from __future__ import annotations

import hashlib
import queue
import secrets
import threading
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from grok_vision import GrokError, GrokVision, SemanticScan
from hybrid_memory import HybridMemory


class HybridRuntime:
    def __init__(self, data_dir: Path, grok: GrokVision | None = None,
                 auto_interval: float = 30.0):
        self.data_dir = Path(data_dir)
        self.root = self.data_dir / "hybrid"
        self.root.mkdir(parents=True, exist_ok=True)
        self.evidence_dir = self.root / "evidence"
        self.evidence_dir.mkdir(exist_ok=True)
        self.memory = HybridMemory(self.root / "semantic.sqlite3")
        self.grok = grok or GrokVision()
        self.auto_interval = max(10.0, float(auto_interval))
        self.auto_enabled = False
        self.mobile_enabled = False
        self.mobile_token = secrets.token_urlsafe(12)
        self._queue: queue.Queue[tuple[bytes, str, str, bool]] = queue.Queue(maxsize=1)
        self._thread = threading.Thread(target=self._worker, daemon=True, name="ctrlf-grok")
        self._stop = threading.Event()
        self._started = False
        self._last_submit = 0.0
        self._last_signature: bytes | None = None
        self.last_error: str | None = None
        self.last_scan_at: float | None = None
        self.last_scan_count = 0
        self._lock = threading.RLock()

    def start(self) -> None:
        if self._started:
            return
        self._started = True
        self._thread.start()

    def close(self) -> None:
        self._stop.set()
        if self._started:
            self._thread.join(timeout=2.0)
        self.memory.close()

    def status(self) -> dict[str, Any]:
        with self._lock:
            return {
                "configured": self.grok.configured,
                "provider": getattr(self.grok, "provider", "unknown"),
                "model": self.grok.model,
                "auto_enabled": self.auto_enabled,
                "mobile_enabled": self.mobile_enabled,
                "mobile_token": self.mobile_token if self.mobile_enabled else None,
                "last_error": self.last_error,
                "last_scan_at": self.last_scan_at,
                "last_scan_count": self.last_scan_count,
                "queued": not self._queue.empty(),
            }

    def set_auto(self, enabled: bool) -> None:
        if enabled and not self.grok.configured:
            raise GrokError("Set XAI_API_KEY before enabling passive Grok discovery.")
        self.auto_enabled = bool(enabled)

    def set_mobile(self, enabled: bool) -> str | None:
        if enabled and not self.grok.configured:
            raise GrokError("Set XAI_API_KEY before enabling mobile room scans.")
        self.mobile_enabled = bool(enabled)
        if enabled:
            self.mobile_token = secrets.token_urlsafe(12)
            return self.mobile_token
        return None

    def maybe_submit_frame(self, frame: np.ndarray, now: float | None = None) -> bool:
        if not self.auto_enabled or not self.grok.configured:
            return False
        now = time.time() if now is None else float(now)
        if now - self._last_submit < self.auto_interval or not self._scene_changed(frame):
            return False
        ok, encoded = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 82])
        if not ok:
            return False
        try:
            self._queue.put_nowait((encoded.tobytes(), "laptop", "desk", False))
        except queue.Full:
            return False
        self._last_submit = now
        return True

    def submit_mobile(self, jpeg: bytes, zone: str, keep_evidence: bool = False) -> SemanticScan:
        return self.scan_now(jpeg, "mobile", zone, keep_evidence)

    def scan_now(self, jpeg: bytes, source: str, zone: str,
                 keep_evidence: bool = False) -> SemanticScan:
        scan = self.grok.analyze_jpeg(jpeg, context=f"source={source}; zone={zone}")
        evidence = self._write_evidence(jpeg, scan) if keep_evidence else {}
        self.memory.ingest(scan, source, zone, evidence)
        with self._lock:
            self.last_error = None
            self.last_scan_at = time.time()
            self.last_scan_count = len(scan.objects)
        return scan

    def _worker(self) -> None:
        while not self._stop.is_set():
            try:
                jpeg, source, zone, keep = self._queue.get(timeout=.25)
            except queue.Empty:
                continue
            try:
                self.scan_now(jpeg, source, zone, keep)
            except Exception as exc:
                with self._lock:
                    self.last_error = str(exc)[:500]
            finally:
                self._queue.task_done()

    def _scene_changed(self, frame: np.ndarray) -> bool:
        tiny = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (32, 18), interpolation=cv2.INTER_AREA)
        signature = tiny.tobytes()
        if self._last_signature is None:
            self._last_signature = signature
            return True
        previous = np.frombuffer(self._last_signature, np.uint8).reshape(18, 32)
        diff = float(np.mean(cv2.absdiff(tiny, previous)))
        if diff < 4.0:
            return False
        self._last_signature = signature
        return True

    def _write_evidence(self, jpeg: bytes, scan: SemanticScan) -> dict[int, str]:
        frame = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
        if frame is None:
            return {}
        h, w = frame.shape[:2]
        names: dict[int, str] = {}
        stamp = int(time.time() * 1000)
        for index, obj in enumerate(scan.objects):
            if obj.bbox is None:
                continue
            x, y, bw, bh = obj.bbox
            x0, y0 = max(0, int(x * w)), max(0, int(y * h))
            x1, y1 = min(w, int((x + bw) * w)), min(h, int((y + bh) * h))
            if x1 - x0 < 10 or y1 - y0 < 10:
                continue
            crop = frame[y0:y1, x0:x1]
            digest = hashlib.sha1(f"{stamp}:{index}:{obj.instance_signature}".encode()).hexdigest()[:12]
            name = f"{stamp}_{digest}.jpg"
            if cv2.imwrite(str(self.evidence_dir / name), crop, [cv2.IMWRITE_JPEG_QUALITY, 82]):
                names[index] = name
        return names
