import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from tracker import (Observation, detect_markers, get_last_seen, init_db,
                     load_objects, match_object, save_observation)

ROOT = Path(__file__).resolve().parents[1]


class TrackerTests(unittest.TestCase):
    def test_marker_detection_in_synthetic_frame(self):
        frame = np.full((600, 800, 3), 255, np.uint8)
        dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        marker = cv2.aruco.generateImageMarker(dictionary, 0, 180)
        frame[160:340, 250:430] = cv2.cvtColor(marker, cv2.COLOR_GRAY2BGR)
        obs = detect_markers(frame)
        self.assertIn(0, obs)
        self.assertAlmostEqual(obs[0].x, 340/800, delta=.02)
        self.assertAlmostEqual(obs[0].y, 250/600, delta=.02)

    def test_database_and_search(self):
        objects = load_objects(ROOT / "objects.json")
        self.assertEqual(match_object("where is my pendrive", objects), 0)
        self.assertEqual(match_object("find my house keys", objects), 1)
        self.assertIsNone(match_object("find my car", objects))
        with tempfile.TemporaryDirectory() as folder:
            conn = init_db(Path(folder) / "memory.sqlite3")
            self.assertIsNone(get_last_seen(conn, 0))
            save_observation(conn, Observation(0, .3, .7, (1,2,3,4)), 800, 600)
            value = get_last_seen(conn, 0)
            self.assertAlmostEqual(value['x'], .3)
            self.assertEqual(value['width'], 800)
            conn.close()


if __name__ == '__main__':
    unittest.main()
