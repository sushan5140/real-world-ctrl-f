"""Printed-tag detection, name search and the original compatibility API."""
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from helpers import ROOT, plain, with_tag
from catalog import Catalog, CatalogObject, search
from tracker import (Observation, detect_markers, get_last_seen, init_db,
                     load_objects, match_object, save_observation)


class MarkerTests(unittest.TestCase):
    def test_marker_detection_in_synthetic_frame(self):
        frame = np.full((600, 800, 3), 255, np.uint8)
        dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        marker = cv2.aruco.generateImageMarker(dictionary, 0, 180)
        frame[160:340, 250:430] = cv2.cvtColor(marker, cv2.COLOR_GRAY2BGR)
        obs = detect_markers(frame)
        self.assertIn(0, obs)
        self.assertAlmostEqual(obs[0].x, 340 / 800, delta=.02)
        self.assertAlmostEqual(obs[0].y, 250 / 600, delta=.02)
        self.assertEqual(len(obs[0].corners), 4)

    def test_duplicate_tags_are_counted_and_largest_is_used(self):
        frame = with_tag(with_tag(plain(), 1, 60, 60, 90), 1, 500, 200, 150)
        obs = detect_markers(frame)
        self.assertEqual(obs[1].copies, 2)
        self.assertGreater(obs[1].x, .5)          # the larger copy on the right

    def test_no_markers(self):
        self.assertEqual(detect_markers(plain()), {})


class SearchTests(unittest.TestCase):
    def setUp(self):
        self.objects = load_objects(ROOT / 'objects.json')
        self.objects[50] = {'name': 'my phone', 'aliases': []}

    def test_original_queries_still_work(self):
        self.assertEqual(match_object('where is my pendrive', self.objects), 0)
        self.assertEqual(match_object('find my house keys', self.objects), 1)
        self.assertIsNone(match_object('find my car', self.objects))

    def test_word_matching_instead_of_substrings(self):
        # The original substring search returned Keys for "keyboard" and
        # could not find an object called "my phone" by the word "phone".
        self.assertIsNone(match_object('where is my keyboard', self.objects))
        self.assertEqual(match_object('phone', self.objects), 50)
        self.assertEqual(match_object("where's the phones?", self.objects), 50)

    def test_plurals_and_typos(self):
        self.assertEqual(match_object('key', self.objects), 1)
        self.assertEqual(match_object('earphnes', self.objects), 2)
        self.assertEqual(match_object('my EARBUDS?!', self.objects), 2)

    def test_more_specific_phrase_wins(self):
        objects = {1: CatalogObject(1, 'Keys'), 7: CatalogObject(7, 'Car keys')}
        self.assertEqual(search('where are my car keys', objects)[0][1], 7)
        self.assertEqual(search('keys', objects)[0][1], 1)

    def test_catalog_rejects_duplicate_names_and_aliases(self):
        with tempfile.TemporaryDirectory() as folder:
            catalog = Catalog(ROOT / 'objects.json', Path(folder) / 'catalog.json')
            with self.assertRaises(ValueError):
                catalog.add_visual('USB drive')
            with self.assertRaises(ValueError):
                catalog.add_visual('Notebook', ['pendrive'])
            with self.assertRaises(ValueError):
                catalog.add_tag(0, 'Wallet')           # tag 0 is built in
            catalog.add_tag(9, 'Wallet', ['purse'])
            catalog.save()
            self.assertEqual(Catalog(ROOT / 'objects.json', Path(folder) / 'catalog.json').get(9).name, 'Wallet')


class CompatibilityTests(unittest.TestCase):
    def test_database_and_search(self):
        with tempfile.TemporaryDirectory() as folder:
            conn = init_db(Path(folder) / 'memory.sqlite3')
            self.assertIsNone(get_last_seen(conn, 0))
            save_observation(conn, Observation(0, .3, .7, (1, 2, 3, 4)), 800, 600)
            value = get_last_seen(conn, 0)
            self.assertAlmostEqual(value['x'], .3)
            self.assertEqual(value['width'], 800)
            conn.close()


if __name__ == '__main__':
    unittest.main()
