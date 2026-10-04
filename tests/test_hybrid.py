"""Camera-free tests for the hybrid semantic memory layer."""
import json
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from grok_vision import SemanticObject, SemanticScan, _extract_output_text, _validate_scan
from hybrid import HybridRuntime
from hybrid_memory import HybridMemory


def fake_scan(name="Black mouse", signature="black_mouse_red_wheel"):
    return SemanticScan(
        scene="desk",
        objects=(SemanticObject(
            name=name,
            category="computer accessory",
            description="black wireless mouse with a red scroll wheel",
            instance_signature=signature,
            aliases=("mouse", "wireless mouse"),
            location_hint="right side of desk",
            confidence=.91,
            bbox=(.2, .3, .25, .2),
        ),),
        model="fake-grok",
    )


class FakeGrok:
    configured = True
    model = "fake-grok"

    def analyze_jpeg(self, jpeg, context=""):
        if not jpeg:
            raise ValueError("empty")
        return fake_scan()


class HybridTests(unittest.TestCase):
    def test_response_output_text_and_validation(self):
        text = json.dumps({
            "scene": "desk",
            "objects": [{
                "name": "Phone", "category": "electronics", "description": "black phone",
                "instance_signature": "black_phone_clear_case", "aliases": ["mobile"],
                "location_hint": "beside notebook", "confidence": .8,
                "bbox": [.1, .2, .3, .4],
            }],
        })
        self.assertEqual(_extract_output_text({"output": [{"content": [{"text": text}]}]}), text)
        scan = _validate_scan(json.loads(text), "grok-test")
        self.assertEqual(scan.objects[0].name, "Phone")
        self.assertEqual(scan.objects[0].bbox, (.1, .2, .3, .4))

    def test_semantic_memory_merges_conservative_signature(self):
        with tempfile.TemporaryDirectory() as tmp:
            memory = HybridMemory(Path(tmp) / "semantic.sqlite3")
            first = memory.ingest(fake_scan(), "laptop", "desk", when=1000)
            second = memory.ingest(fake_scan(name="Computer mouse"), "mobile", "bedroom", when=2000)
            self.assertEqual(first[0].object_id, second[0].object_id)
            latest = memory.latest()
            self.assertEqual(len(latest), 1)
            self.assertEqual(latest[0]["zone"], "bedroom")
            self.assertEqual(len(memory.history(first[0].object_id)), 2)
            self.assertEqual(memory.search("mouse")[0]["id"], first[0].object_id)
            memory.close()

    def test_different_visible_signatures_do_not_merge(self):
        with tempfile.TemporaryDirectory() as tmp:
            memory = HybridMemory(Path(tmp) / "semantic.sqlite3")
            a = memory.ingest(fake_scan(signature="black_mouse_red_wheel"), "laptop", "desk")
            b = memory.ingest(fake_scan(signature="white_mouse_blue_wheel"), "mobile", "room")
            self.assertNotEqual(a[0].object_id, b[0].object_id)
            self.assertEqual(len(memory.latest()), 2)
            memory.close()

    def test_runtime_scan_and_optional_crop_evidence(self):
        with tempfile.TemporaryDirectory() as tmp:
            runtime = HybridRuntime(Path(tmp), grok=FakeGrok())
            frame = np.full((200, 300, 3), 210, np.uint8)
            cv2.rectangle(frame, (60, 60), (140, 120), (20, 20, 20), -1)
            ok, encoded = cv2.imencode(".jpg", frame)
            self.assertTrue(ok)
            scan = runtime.scan_now(encoded.tobytes(), "mobile", "bedside", keep_evidence=True)
            self.assertEqual(len(scan.objects), 1)
            item = runtime.memory.latest()[0]
            self.assertEqual(item["source"], "mobile")
            self.assertEqual(item["zone"], "bedside")
            self.assertTrue(item["evidence"])
            self.assertTrue((runtime.evidence_dir / item["evidence"]).is_file())
            runtime.close()

    def test_passive_sampling_is_opt_in_and_rate_limited(self):
        with tempfile.TemporaryDirectory() as tmp:
            runtime = HybridRuntime(Path(tmp), grok=FakeGrok(), auto_interval=10)
            frame = np.zeros((120, 160, 3), np.uint8)
            self.assertFalse(runtime.maybe_submit_frame(frame, now=100))
            runtime.set_auto(True)
            self.assertTrue(runtime.maybe_submit_frame(frame, now=100))
            changed = frame.copy()
            changed[:, :80] = 255
            self.assertFalse(runtime.maybe_submit_frame(changed, now=105))
            runtime.close()


if __name__ == "__main__":
    unittest.main()
