"""HTTP API, request security, and the camera worker end to end (using a
synthetic video file instead of a webcam)."""
import tempfile
import time
import unittest
from pathlib import Path

import cv2

from fastapi.testclient import TestClient

from helpers import ROOT, FakeSerial, desk, place, textured_object, with_tag
import server
from engine import TrackingEngine
from spotlight import Spotlight

BASE = 'http://127.0.0.1:8765'
HEADERS = {'X-CtrlF': '1'}


class ApiTestCase(unittest.TestCase):
    start_camera = False
    camera_source = 0

    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.data = Path(self.folder.name)
        self.engine = TrackingEngine(self.data, ROOT / 'objects.json')
        self.link = FakeSerial()
        self.spot = Spotlight(None, ROOT / 'spotlight.json', self.data / 'spotlight.json',
                              serial_factory=lambda port: self.link)
        self.studio = server.Studio(self.engine, self.camera_source, self.spot, max_fps=30)
        self.client = TestClient(server.create_app(self.studio, start_camera=self.start_camera),
                                 base_url=BASE, headers=HEADERS)
        self.client.__enter__()
        self.desk = desk(5)
        self.t = time.time() - 30

    def tearDown(self):
        self.client.__exit__(None, None, None)
        self.folder.cleanup()

    def feed(self, frame, count=1, seconds=.1):
        for _ in range(count):
            self.t += seconds
            self.engine.process(frame, self.t)

    def status(self):
        return self.client.get('/api/status').json()

    def item(self, obj_id):
        return next(i for i in self.status()['items'] if i['id'] == obj_id)


class ApiTests(ApiTestCase):
    def test_pages_and_status(self):
        for path, kind in (('/', 'text/html'), ('/client.js', 'javascript'), ('/style.css', 'text/css'),
                           ('/favicon.svg', 'svg')):
            response = self.client.get(path)
            self.assertEqual(response.status_code, 200, path)
            self.assertIn(kind, response.headers['content-type'])
        data = self.status()
        self.assertEqual(data['camera']['state'], 'stopped')
        self.assertFalse(data['spotlight']['connected'])
        self.assertEqual({i['state'] for i in data['items']}, {'not_observed'})

    def test_observe_occlude_search_snapshot_forget(self):
        tagged = with_tag(self.desk, 0, 300, 150)
        self.feed(tagged, count=3)
        found = self.client.post('/api/find', json={'query': 'where is my USB drive'}).json()
        self.assertEqual(found, {'id': 0, 'ambiguous': False, 'candidates': []})
        self.assertEqual(self.item(0)['state'], 'visible')
        self.assertIsNotNone(self.item(0)['live'])
        self.feed(self.desk, count=5, seconds=.3)
        gone = self.item(0)
        self.assertEqual(gone['state'], 'last_seen')
        self.assertEqual(self.client.get(gone['snapshot']).status_code, 200)
        self.assertEqual(self.client.delete('/api/object/0').json(), {'id': 0, 'removed': False})
        self.assertEqual(self.item(0)['state'], 'not_observed')
        self.assertEqual(self.client.get('/api/snapshot/0').status_code, 404)

    def test_search_errors_and_ambiguity(self):
        self.assertEqual(self.client.post('/api/find', json={'query': 'planet'}).status_code, 404)
        self.assertEqual(self.client.post('/api/find', json={'query': ''}).status_code, 422)
        self.client.post('/api/tags', json={'marker_id': 5, 'name': 'Blue keys'})
        self.client.post('/api/tags', json={'marker_id': 6, 'name': 'Red keys'})
        result = self.client.post('/api/find', json={'query': 'keys blue red'}).json()
        self.assertTrue(result['ambiguous'])
        self.assertEqual({c['id'] for c in result['candidates']}, {5, 6})

    def test_enrolment_flow(self):
        obj = textured_object(19)
        self.assertEqual(self.client.post('/api/enroll/still').status_code, 409)   # no camera picture yet
        self.feed(place(self.desk, obj, 300, 280))
        still = self.client.post('/api/enroll/still').json()
        self.assertEqual((still['width'], still['height']), (960, 540))
        image = self.client.get(f"/api/enroll/still/{still['token']}.jpg")
        self.assertEqual(image.headers['content-type'], 'image/jpeg')
        rect = {'x': 170 / 960, 'y': 180 / 540, 'w': 260 / 960, 'h': 200 / 540}
        response = self.client.post('/api/enroll', json={'token': still['token'], 'name': 'Journal',
                                                         'aliases': ['my book'], 'rect': rect})
        self.assertEqual(response.status_code, 200, response.text)
        obj_id = response.json()['id']
        duplicate = self.client.post('/api/enroll', json={'token': still['token'], 'name': 'journal', 'rect': rect})
        self.assertEqual(duplicate.status_code, 400)
        outside = self.client.post('/api/enroll', json={'token': still['token'], 'name': 'Box',
                                                        'rect': {'x': .9, 'y': .3, 'w': .3, 'h': .3}})
        self.assertEqual(outside.status_code, 400)
        self.feed(place(self.desk, obj, 640, 300, 20), count=4)
        self.assertEqual(self.item(obj_id)['state'], 'visible')
        view = self.client.post(f'/api/object/{obj_id}/views', json={'token': still['token'], 'rect': rect})
        self.assertEqual(view.json()['views'], 2)
        self.assertEqual(self.client.post('/api/object/0/views', json={'token': still['token'], 'rect': rect}).status_code, 404)
        self.assertTrue(self.client.delete(f'/api/object/{obj_id}').json()['removed'])
        self.assertEqual(self.client.delete(f'/api/object/{obj_id}').status_code, 404)

    def test_tag_registration_validation(self):
        self.assertEqual(self.client.post('/api/tags', json={'marker_id': 60, 'name': 'Bad'}).status_code, 422)
        self.assertEqual(self.client.post('/api/tags', json={'marker_id': 1, 'name': 'Wallet'}).status_code, 400)
        self.assertEqual(self.client.post('/api/tags', json={'marker_id': 9, 'name': 'Wallet'}).status_code, 200)

    def test_clear_memory(self):
        self.feed(with_tag(self.desk, 1, 300, 150), count=3)
        self.feed(self.desk, count=5, seconds=.3)
        self.assertEqual(self.client.delete('/api/memory').status_code, 200)
        self.assertEqual(self.item(1)['state'], 'not_observed')
        self.assertEqual(list((self.data / 'snapshots').glob('*.jpg')), [])

    def test_spotlight_endpoints(self):
        self.assertEqual(self.client.post('/api/point/0').status_code, 400)       # never seen
        self.feed(with_tag(self.desk, 0, 300, 150), count=3)
        self.assertEqual(self.client.post('/api/point/0').status_code, 400)       # not connected
        self.assertEqual(self.client.post('/api/light', json={'enabled': True}).status_code, 400)
        connected = self.client.post('/api/spotlight/connect', json={'port': 'COM3'}).json()
        self.assertTrue(connected['confirmed'])
        pointed = self.client.post('/api/point/0').json()
        self.assertEqual(pointed['state'], 'visible')
        self.assertFalse(pointed['calibrated'])
        self.assertIn('L,1', self.link.sent)
        self.assertEqual(self.client.post('/api/spotlight/move', json={'pan': 100, 'tilt': 80}).json(),
                         {'pan': 100, 'tilt': 80})
        self.assertEqual(self.client.post('/api/spotlight/aim', json={'x': .5, 'y': .5}).json(), {'pan': 90, 'tilt': 90})
        self.client.post('/api/spotlight/move', json={'pan': 120, 'tilt': 60})
        corners = self.client.post('/api/spotlight/corner', json={'corner': 'tl'}).json()['corners']
        self.assertEqual(corners['tl'], [120, 60])
        self.assertEqual(self.client.post('/api/spotlight/corner', json={'corner': 'middle'}).status_code, 400)
        self.link.fail_writes = True
        self.assertEqual(self.client.post('/api/light', json={'enabled': False}).status_code, 503)
        self.assertFalse(self.status()['spotlight']['connected'])
        self.assertEqual(self.client.post('/api/spotlight/disconnect').status_code, 200)
        self.assertIn('ports', self.client.get('/api/spotlight/ports').json())


class SecurityTests(ApiTestCase):
    def test_foreign_host_rejected(self):
        # DNS rebinding: evil.example resolving to 127.0.0.1
        self.assertEqual(self.client.get('/api/status', headers={'host': 'evil.example:8765'}).status_code, 400)
        self.assertEqual(self.client.get('/stream', headers={'host': 'evil.example'}).status_code, 400)

    def test_state_changes_need_header_and_same_origin(self):
        plain = TestClient(self.client.app, base_url=BASE)
        self.assertEqual(plain.post('/api/select/0').status_code, 403)          # simple cross-site form/fetch
        self.assertEqual(plain.delete('/api/memory').status_code, 403)
        self.assertEqual(self.client.post('/api/select/0', headers={'origin': 'https://evil.example'}).status_code, 403)
        self.assertEqual(self.client.post('/api/select/0', headers={'origin': BASE}).status_code, 200)

    def test_security_headers(self):
        headers = self.client.get('/').headers
        self.assertIn("frame-ancestors 'none'", headers['content-security-policy'])
        self.assertEqual(headers['cross-origin-resource-policy'], 'same-origin')
        self.assertEqual(headers['x-content-type-options'], 'nosniff')

    def test_host_parsing(self):
        self.assertEqual(server.host_name('127.0.0.1:8765'), '127.0.0.1')
        self.assertEqual(server.host_name('[::1]:8765'), '[::1]')
        self.assertEqual(server.host_name('LOCALHOST'), 'localhost')


class CameraWorkerTests(ApiTestCase):
    """Runs the real background camera thread on a synthetic video."""

    def setUp(self):
        self.video_dir = tempfile.TemporaryDirectory()
        path = Path(self.video_dir.name) / 'desk.avi'
        writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*'MJPG'), 15, (960, 540))
        scene = desk(6)
        for i in range(45):
            writer.write(with_tag(scene, 1, 200 + i * 4, 200) if i < 30 else scene)
        writer.release()
        self.camera_source = str(path)
        super().setUp()

    def tearDown(self):
        super().tearDown()
        self.video_dir.cleanup()

    def wait_for(self, predicate, timeout=20):
        deadline = time.time() + timeout
        while time.time() < deadline:
            if predicate():
                return True
            time.sleep(.1)
        return False

    def test_start_track_stop(self):
        self.assertEqual(self.client.post('/api/camera/start', json={}).status_code, 200)
        self.assertTrue(self.wait_for(lambda: self.status()['camera']['state'] == 'running'))
        self.assertTrue(self.wait_for(lambda: self.item(1)['state'] == 'visible'))
        self.assertIsNotNone(self.studio.camera.jpeg)
        self.assertEqual(self.status()['camera']['width'], 960)
        self.client.post('/api/camera/stop')
        data = self.status()
        self.assertEqual(data['camera']['state'], 'stopped')
        self.assertIn(self.item(1)['state'], ('last_seen', 'unreliable'))

    def test_missing_camera_reports_error(self):
        self.client.post('/api/camera/start', json={'source': str(Path(self.video_dir.name) / 'nope.avi')})
        self.assertTrue(self.wait_for(lambda: self.status()['camera']['state'] in ('reconnecting', 'error')))
        self.assertIn('Cannot open', self.status()['camera']['error'])
        self.client.post('/api/camera/stop')


if __name__ == '__main__':
    unittest.main()
