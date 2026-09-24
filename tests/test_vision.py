"""Markerless recognition: enrolment quality, geometric checks, background
false positives, multiple views and storage."""
import tempfile
import unittest
from pathlib import Path

import numpy as np

from helpers import desk, place, plain, textured_object
import vision


def enrol(frame, cx, cy, w, h, margin=.18, object_id=50):
    x0, y0 = int(cx - w / 2 - w * margin), int(cy - h / 2 - h * margin)
    x1, y1 = int(cx + w / 2 + w * margin), int(cy + h / 2 + h * margin)
    return vision.create_template(object_id, frame[y0:y1, x0:x1])


class VisionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.desk = desk(3)
        cls.obj = textured_object(19)
        cls.enrol_frame = place(cls.desk, cls.obj, 300, 280)
        cls.template = enrol(cls.enrol_frame, 300, 280, 210, 160)
        cls.matcher = vision.Matcher()

    def test_blank_object_enrolment_rejected_with_advice(self):
        with self.assertRaises(ValueError) as ctx:
            vision.create_template(50, np.full((120, 120, 3), 240, np.uint8))
        self.assertIn('printed tag', str(ctx.exception))

    def test_too_small_selection_rejected(self):
        with self.assertRaises(ValueError):
            vision.create_template(50, np.zeros((20, 200, 3), np.uint8))

    def test_found_after_moving_rotating_and_scaling(self):
        for cx, cy, angle, scale in ((650, 300, 35, 1.0), (500, 350, -120, .8), (420, 250, 180, 1.15)):
            frame = place(self.desk, self.obj, cx, cy, angle, scale)
            found = self.matcher.find(frame, {50: self.template})
            self.assertIn(50, found, (cx, cy, angle, scale))
            self.assertAlmostEqual(found[50].x * frame.shape[1], cx, delta=20)
            self.assertAlmostEqual(found[50].y * frame.shape[0], cy, delta=20)
            self.assertGreaterEqual(found[50].confidence, vision.MIN_CONFIDENCE)

    def test_outline_follows_object_not_loose_box(self):
        frame = place(self.desk, self.obj, 600, 300)
        x0, y0, x1, y1 = self.matcher.find(frame, {50: self.template})[50].box
        self.assertLess(x1 - x0, 210 * 1.2)        # enrolment box was 1.36x wider
        self.assertLess(y1 - y0, 160 * 1.2)

    def test_no_false_positive_when_object_removed(self):
        # The enrolment rectangle contained desk background. The original
        # matcher kept "finding" the object there after it was taken away.
        self.assertEqual(self.matcher.find(self.desk, {50: self.template}), {})
        self.assertEqual(self.matcher.find(plain(), {50: self.template}), {})

    def test_loose_box_background_regression(self):
        # Same scenario for the original matcher and the new one: enrol with a
        # generous box on a textured desk, then take the object away.
        import benchmark_recognition as legacy
        empty = desk(1)
        frame = place(empty, self.obj, 300, 280)
        crop = frame[int(280 - 80 - 40):int(280 + 80 + 40), int(300 - 105 - 52):int(300 + 105 + 52)]
        self.assertTrue(legacy.legacy_find(empty, {50: legacy.legacy_template(crop)}),
                        'scenario should reproduce the original false positive')
        template = vision.create_template(50, crop)
        self.assertEqual(self.matcher.find(empty, {50: template}), {})
        self.assertIn(50, self.matcher.find(place(empty, self.obj, 640, 330, 25), {50: template}))

    def test_other_objects_are_not_confused(self):
        others = self.desk.copy()
        for seed, (cx, cy) in zip((5, 6, 7), ((250, 150), (600, 380), (800, 200))):
            others = place(others, textured_object(seed), cx, cy)
        self.assertEqual(self.matcher.find(others, {50: self.template}), {})

    def test_lighting_change(self):
        frame = place(self.desk, self.obj, 600, 300)
        darker = np.clip(frame.astype(np.float32) * .65 + 10, 0, 255).astype(np.uint8)
        self.assertIn(50, self.matcher.find(darker, {50: self.template}))

    def test_extra_view_finds_other_side(self):
        back = textured_object(42)
        frame = place(self.desk, back, 600, 300)
        self.assertEqual(self.matcher.find(frame, {50: self.template}), {})
        back_view = vision.build_view(place(self.desk, back, 300, 280)[190:370, 180:420])
        both = vision.Template(50, [*self.template.views, back_view])
        found = self.matcher.find(frame, {50: both})
        self.assertIn(50, found)
        self.assertEqual(found[50].view, 1)

    def test_two_objects_one_place_keeps_stronger_claim(self):
        same = vision.Template(51, list(self.template.views))
        frame = place(self.desk, self.obj, 600, 300)
        found = self.matcher.find(frame, {50: self.template, 51: same})
        self.assertEqual(len(found), 1)

    def test_views_round_trip_with_mask(self):
        with tempfile.TemporaryDirectory() as folder:
            vision.save_view(Path(folder), '50.png', self.template.views[0])
            self.assertTrue((Path(folder) / '50.mask.png').is_file())
            loaded = vision.load_view(Path(folder), '50.png')
            self.assertEqual(len(loaded.keypoints), len(self.template.views[0].keypoints))
            # Templates saved by the original version have no mask file.
            (Path(folder) / '50.mask.png').unlink()
            legacy = vision.load_view(Path(folder), '50.png')
            self.assertGreaterEqual(len(legacy.keypoints), vision.MIN_TEMPLATE_FEATURES)
            with self.assertRaises(OSError):
                vision.load_view(Path(folder), '99.png')


if __name__ == '__main__':
    unittest.main()
