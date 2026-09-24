"""Web API, visual memory, and optional-hardware calculations without a webcam."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np
from fastapi.testclient import TestClient

import server
from spotlight import DEFAULT_CORNERS, angles
from vision import create_template, find_templates


def marker_frame():
    image = np.full((600, 800, 3), 238, dtype=np.uint8)
    tag = cv2.aruco.generateImageMarker(cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50), 0, 155)
    image[145:300, 180:335] = cv2.cvtColor(tag, cv2.COLOR_GRAY2BGR)
    return image


def distinctive_frame():
    rng = np.random.default_rng(19)
    canvas = np.full((600, 800, 3), 235, dtype=np.uint8)
    patch = rng.integers(20, 235, (210, 210, 3), dtype=np.uint8)
    for i in range(12):
        cv2.putText(patch, str(i), (i * 13 % 170, 22 + i * 14),
                    cv2.FONT_HERSHEY_SIMPLEX, .5, (255, 255, 255), 1)
    canvas[180:390, 260:470] = patch
    return canvas


class StudioTests(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.base = Path(self.folder.name)
        patches = [patch.object(server, 'DATA', self.base),
                   patch.object(server, 'CATALOG', self.base / 'catalog.json'),
                   patch.object(server, 'TEMPLATES', self.base / 'templates'),
                   patch.object(server, 'SNAPSHOTS', self.base / 'snapshots')]
        for p in patches:
            p.start()
            self.addCleanup(p.stop)
        self.engine = server.Engine()
        self.client = TestClient(server.create_app(self.engine, start_camera=False))
        self.client.__enter__()

    def tearDown(self):
        self.client.__exit__(None, None, None)
        self.engine.close()
        self.folder.cleanup()

    def test_tag_observe_occlude_search_snapshot_and_forget(self):
        self.engine.process(marker_frame())
        r = self.client.post('/api/find', json={'query': 'where is my USB drive'})
        self.assertEqual(r.status_code, 200)
        present = self.client.get('/api/status').json()
        self.assertTrue(present['items'][0]['visible'])
        self.engine.process(np.full((600, 800, 3), 238, np.uint8))
        disappeared = self.client.get('/api/status').json()['items'][0]
        self.assertFalse(disappeared['visible'])
        self.assertIsNotNone(disappeared['last_seen'])
        self.assertEqual(self.client.get('/api/snapshot/0').status_code, 200)
        self.assertEqual(self.client.post('/api/point/0').status_code, 400)
        self.assertEqual(self.client.delete('/api/object/0').status_code, 200)
        self.assertIsNone(self.client.get('/api/status').json()['items'][0]['last_seen'])
        self.assertEqual(self.client.get('/api/snapshot/0').status_code, 404)

    def test_enroll_local_object_and_persistence(self):
        image = distinctive_frame()
        self.engine.process(image)
        body = {'name': 'Journal', 'aliases': ['my book'],
                'rect': {'x': 260/800, 'y': 180/600, 'w': 210/800, 'h': 210/600}}
        r = self.client.post('/api/enroll', json=body)
        self.assertEqual(r.status_code, 200, r.text)
        obj_id = r.json()['id']
        self.engine.process(image)
        self.assertTrue(next(i for i in self.client.get('/api/status').json()['items'] if i['id'] == obj_id)['visible'])
        self.assertEqual(self.client.post('/api/find', json={'query': 'find my book'}).json()['id'], obj_id)
        self.assertTrue((self.base / 'templates' / f'{obj_id}.png').is_file())
        self.assertEqual(self.client.delete(f'/api/object/{obj_id}').status_code, 200)
        self.assertFalse((self.base / 'templates' / f'{obj_id}.png').exists())

    def test_validation(self):
        self.assertEqual(self.client.post('/api/find', json={'query': 'planet'}).status_code, 404)
        self.assertEqual(self.client.post('/api/enroll', json={
            'name': 'bad', 'rect': {'x': .9, 'y': .3, 'w': .3, 'h': .3}}).status_code, 400)
        self.assertEqual(self.client.get('/api/snapshot/100').status_code, 404)
        self.assertEqual(self.client.get('/').status_code, 200)
        self.assertEqual(self.client.get('/client.js').status_code, 200)

    def test_spotlight_calibration(self):
        self.assertEqual(angles(0, 0, DEFAULT_CORNERS), (135, 55))
        self.assertEqual(angles(1, 1, DEFAULT_CORNERS), (45, 125))
        self.assertEqual(angles(.5, .5, DEFAULT_CORNERS), (90, 90))
        with self.assertRaises(ValueError):
            angles(-.1, .5, DEFAULT_CORNERS)

    def test_blank_object_enrollment_rejected(self):
        with self.assertRaises(ValueError):
            create_template(50, np.full((100, 100, 3), 240, np.uint8))


if __name__ == '__main__':
    unittest.main()
