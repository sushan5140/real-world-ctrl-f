"""The shared tracking engine: confirmation, disappearance, reappearance,
visual-memory states, camera movement, restarts and persistence."""
import json
import sqlite3
import tempfile
import time
import unittest
from pathlib import Path

import cv2
import numpy as np

from helpers import ROOT, desk, place, shift, textured_object, with_tag
from engine import LAST_SEEN, NOT_OBSERVED, UNRELIABLE, VISIBLE, TrackingEngine


class EngineTestCase(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.data = Path(self.folder.name)
        self.engine = TrackingEngine(self.data, ROOT / 'objects.json')
        self.desk = desk(2)
        self.t = time.time() - 60

    def tearDown(self):
        try:
            self.engine.close()
        except Exception:
            pass
        self.folder.cleanup()

    def feed(self, frame, seconds=.1, count=1):
        result = None
        for _ in range(count):
            self.t += seconds
            result = self.engine.process(frame, self.t)
        return result

    def item(self, obj_id):
        return next(i for i in self.engine.status()['items'] if i['id'] == obj_id)


class TrackingTests(EngineTestCase):
    def test_single_frame_is_not_enough(self):
        tagged = with_tag(self.desk, 0, 400, 200)
        self.feed(tagged)
        self.assertEqual(self.item(0)['state'], NOT_OBSERVED)
        self.feed(self.desk, count=3)               # vanished again: never recorded
        self.assertIsNone(self.engine.memory.get(0))
        self.feed(tagged, count=2)
        self.assertEqual(self.item(0)['state'], VISIBLE)

    def test_brief_dropout_does_not_count_as_gone(self):
        tagged = with_tag(self.desk, 0, 400, 200)
        self.feed(tagged, count=3)
        self.feed(self.desk, count=2)               # e.g. motion blur for 0.2 s
        self.assertEqual(self.item(0)['state'], VISIBLE)
        self.feed(tagged)
        self.assertEqual(self.item(0)['state'], VISIBLE)

    def test_disappear_then_last_seen_with_snapshot(self):
        tagged = with_tag(self.desk, 0, 400, 200)
        self.feed(tagged, count=4)
        seen_at = self.t
        self.feed(self.desk, seconds=.3, count=5)
        item = self.item(0)
        self.assertEqual(item['state'], LAST_SEEN)
        self.assertAlmostEqual(item['last_seen'], seen_at, delta=.01)   # detection time, not loss time
        self.assertAlmostEqual(item['x'], (400 + 22 + 55) / 960, delta=.02)
        self.assertIsNotNone(item['snapshot'])
        self.assertTrue((self.data / 'snapshots' / '0.jpg').is_file())
        self.assertEqual(item['reasons'], [])

    def test_reappearance_updates_position_without_duplicates(self):
        self.feed(with_tag(self.desk, 1, 100, 100), count=3)
        self.feed(self.desk, seconds=.3, count=5)
        self.feed(with_tag(self.desk, 1, 700, 300), count=3)
        self.feed(self.desk, seconds=.3, count=5)
        rows = self.engine.memory.conn.execute('SELECT COUNT(*) FROM last_seen WHERE marker_id=1').fetchone()[0]
        self.assertEqual(rows, 1)
        self.assertGreater(self.item(1)['x'], .7)

    def test_leaving_at_the_edge_is_unreliable(self):
        self.feed(with_tag(self.desk, 2, 0, 200), count=3)
        self.feed(self.desk, seconds=.3, count=5)
        item = self.item(2)
        self.assertEqual(item['state'], UNRELIABLE)
        self.assertIn('left_view', [r['code'] for r in item['reasons']])

    def test_stale_sighting_is_unreliable(self):
        self.t = time.time() - 9 * 3600
        self.feed(with_tag(self.desk, 0, 400, 200), count=3)
        self.feed(self.desk, seconds=.3, count=5)
        self.assertIn('stale', [r['code'] for r in self.item(0)['reasons']])

    def test_camera_move_marks_old_positions_unreliable(self):
        self.feed(with_tag(self.desk, 0, 400, 200), seconds=.5, count=4)
        self.feed(self.desk, seconds=.5, count=3)
        self.assertEqual(self.item(0)['state'], LAST_SEEN)
        epoch = self.engine.epoch
        moved = shift(self.desk, 70, 25)            # the camera was bumped
        self.feed(moved, seconds=1.1, count=5)
        self.assertEqual(self.engine.epoch, epoch + 1)
        item = self.item(0)
        self.assertEqual(item['state'], UNRELIABLE)
        self.assertIn('camera_moved', [r['code'] for r in item['reasons']])

    def test_moving_object_is_not_mistaken_for_camera_motion(self):
        obj = textured_object(8, (300, 220))
        epoch = self.engine.epoch
        for step in range(8):
            self.feed(place(self.desk, obj, 300 + step * 40, 300, step * 5), seconds=1.1)
        self.assertEqual(self.engine.epoch, epoch)

    def test_camera_stop_flushes_visible_objects(self):
        self.feed(with_tag(self.desk, 0, 400, 200), count=3)
        self.engine.flush()
        item = self.item(0)
        self.assertEqual(item['state'], LAST_SEEN)
        self.assertIn('camera_off', [n['code'] for n in item['notes']])

    def test_unregistered_tag_can_be_registered(self):
        self.feed(with_tag(self.desk, 7, 500, 300), count=2)
        self.assertEqual([t['marker_id'] for t in self.engine.status()['unregistered_tags']], [7])
        self.engine.register_tag(7, 'Wallet', ['purse'])
        self.feed(with_tag(self.desk, 7, 500, 300), count=2)
        self.assertEqual(self.item(7)['state'], VISIBLE)
        self.assertEqual(self.engine.search('where is my purse')['id'], 7)


class EnrolmentTests(EngineTestCase):
    def setUp(self):
        super().setUp()
        self.obj = textured_object(19)
        self.scene = place(self.desk, self.obj, 300, 280)
        self.rect = {'x': (300 - 130) / 960, 'y': (280 - 100) / 540, 'w': 260 / 960, 'h': 200 / 540}

    def enrol(self, name='Journal', aliases=('my book',)):
        self.feed(self.scene)
        token = self.engine.capture_still()['token']
        return self.engine.enroll_visual(token, self.rect, name, list(aliases))

    def test_enrol_track_and_forget(self):
        result = self.enrol()
        obj_id = result['id']
        self.assertGreaterEqual(obj_id, 50)
        self.feed(place(self.desk, self.obj, 650, 320, 30), count=4)
        self.assertEqual(self.item(obj_id)['state'], VISIBLE)
        self.assertEqual(self.engine.search('find my book')['id'], obj_id)
        self.assertTrue(self.engine.forget(obj_id)['removed'])
        self.assertFalse((self.data / 'templates' / f'{obj_id}.png').exists())
        self.assertNotIn(obj_id, self.engine.catalog.objects)

    def test_ids_are_never_reused(self):
        first = self.enrol('Journal')['id']
        self.engine.forget(first)
        second = self.enrol('Other journal', ())['id']
        self.assertNotEqual(first, second)

    def test_still_frame_is_used_not_the_live_frame(self):
        self.feed(self.scene)
        token = self.engine.capture_still()['token']
        self.feed(self.desk)                       # object removed after the picture was frozen
        self.assertIn('id', self.engine.enroll_visual(token, self.rect, 'Journal'))

    def test_invalid_enrolments(self):
        self.feed(self.desk)
        token = self.engine.capture_still()['token']
        with self.assertRaises(ValueError):
            self.engine.enroll_visual(token, {'x': .9, 'y': .1, 'w': .3, 'h': .3}, 'Out of frame')
        with self.assertRaises(ValueError):
            self.engine.enroll_visual('expired-token', self.rect, 'Journal')
        blank = {'x': .01, 'y': .9, 'w': .08, 'h': .09}
        with self.assertRaises(ValueError):
            self.engine.enroll_visual(token, blank, 'Nothing')

    def test_additional_view(self):
        obj_id = self.enrol()['id']
        back = textured_object(42)
        self.feed(place(self.desk, back, 300, 280))
        token = self.engine.capture_still()['token']
        self.assertEqual(self.engine.add_view(obj_id, token, self.rect)['views'], 2)
        self.feed(place(self.desk, back, 640, 300, -20), count=4)
        self.assertEqual(self.item(obj_id)['state'], VISIBLE)


class PersistenceTests(unittest.TestCase):
    def test_restart_keeps_memory_and_notes_the_gap(self):
        with tempfile.TemporaryDirectory() as folder:
            data = Path(folder)
            scene = desk(4)
            engine = TrackingEngine(data)
            t = time.time() - 30
            for i in range(6):
                engine.process(with_tag(scene, 1, 300, 200), t + i * .6)
            for i in range(4):
                engine.process(scene, t + 4 + i * .6)
            engine.close()
            engine = TrackingEngine(data)
            try:
                engine.process(scene, time.time())
                item = next(i for i in engine.status()['items'] if i['id'] == 1)
                self.assertEqual(item['state'], LAST_SEEN)            # same camera pose verified
                self.assertIn('restarted', [n['code'] for n in item['notes']])
            finally:
                engine.close()

    def test_second_instance_is_refused(self):
        with tempfile.TemporaryDirectory() as folder:
            engine = TrackingEngine(Path(folder))
            try:
                with self.assertRaises(RuntimeError):
                    TrackingEngine(Path(folder))
            finally:
                engine.close()
            TrackingEngine(Path(folder)).close()          # lock released on close

    def test_upgrade_from_original_version(self):
        """Data written by the original v0.2 app: absolute snapshot path, a
        catalog entry whose template is missing, and no camera-pose info."""
        with tempfile.TemporaryDirectory() as folder:
            data = Path(folder)
            (data / 'snapshots').mkdir()
            cv2.imwrite(str(data / 'snapshots' / '0.jpg'), np.zeros((20, 20, 3), np.uint8))
            conn = sqlite3.connect(str(data / 'locations.sqlite3'))
            conn.execute('''CREATE TABLE last_seen (marker_id INTEGER PRIMARY KEY, timestamp REAL NOT NULL,
                x REAL NOT NULL, y REAL NOT NULL, width INTEGER NOT NULL, height INTEGER NOT NULL, snapshot_path TEXT)''')
            conn.execute('INSERT INTO last_seen VALUES (0, ?, .4, .5, 640, 480, ?)',
                         (time.time() - 100, 'C:\\Users\\someone\\old-folder\\data\\snapshots\\0.jpg'))
            conn.commit()
            conn.close()
            (data / 'catalog.json').write_text(json.dumps({'50': {'name': 'my phone', 'aliases': []}}))
            engine = TrackingEngine(data)
            try:
                items = {i['id']: i for i in engine.status()['items']}
                self.assertEqual(items[0]['state'], UNRELIABLE)
                self.assertIn('legacy', [r['code'] for r in items[0]['reasons']])
                self.assertEqual(items[0]['snapshot'].split('?')[0], '/api/snapshot/0')  # path fixed
                self.assertIn(50, items)                                   # kept, not silently dropped
                self.assertIn('Cannot be recognised', items[50]['problem'])
                self.assertEqual(engine.catalog.next_id, 51)              # ID 50 is not reused
            finally:
                engine.close()


if __name__ == '__main__':
    unittest.main()
