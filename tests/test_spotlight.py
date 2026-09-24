"""Spotlight calibration, serial protocol (with a fake port) and the real
firmware source compiled on the host with stand-in Arduino headers."""
import json
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

from helpers import ROOT, FakeSerial
from spotlight import DEFAULT_CORNERS, Calibration, Spotlight, SpotlightError, angles


class CalibrationTests(unittest.TestCase):
    def test_corner_mapping(self):
        self.assertEqual(angles(0, 0, DEFAULT_CORNERS), (135, 55))
        self.assertEqual(angles(1, 1, DEFAULT_CORNERS), (45, 125))
        self.assertEqual(angles(.5, .5, DEFAULT_CORNERS), (90, 90))
        with self.assertRaises(ValueError):
            angles(-.1, .5, DEFAULT_CORNERS)

    def test_limits_match_firmware(self):
        corners = {'tl': [175, 5], 'tr': [5, 5], 'bl': [175, 175], 'br': [5, 175]}
        self.assertEqual(angles(0, 0, corners, {'pan': [30, 150], 'tilt': [30, 150]}), (150, 30))

    def test_repository_defaults_are_marked_placeholder(self):
        calibration = Calibration(ROOT / 'spotlight.json')
        self.assertTrue(calibration.placeholder)

    def test_invalid_files_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            bad = Path(folder) / 'bad.json'
            for content in ({'corners': {'tl': [1, 2]}},
                            {'corners': {**DEFAULT_CORNERS, 'tl': [True, 50]}},
                            {'corners': DEFAULT_CORNERS, 'limits': {'pan': [150, 30], 'tilt': [30, 150]}}):
                bad.write_text(json.dumps(content))
                with self.assertRaises(ValueError):
                    Calibration(bad)
            bad.write_text('{not json')
            with self.assertRaises(ValueError):
                Calibration(bad)

    def test_saving_a_corner_writes_user_file(self):
        with tempfile.TemporaryDirectory() as folder:
            user = Path(folder) / 'spotlight.json'
            spot = Spotlight('COM9', ROOT / 'spotlight.json', user, serial_factory=lambda port: FakeSerial())
            spot.move(120, 60)
            spot.save_corner('tl')
            saved = json.loads(user.read_text())
            self.assertEqual(saved['corners']['tl'], [120, 60])
            self.assertFalse(saved['placeholder'])
            reloaded = Calibration(ROOT / 'spotlight.json', user)
            self.assertEqual(reloaded.corners['tl'], [120, 60])


class SerialTests(unittest.TestCase):
    def make(self, **kwargs):
        link = FakeSerial(**kwargs)
        folder = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, folder, True)
        spot = Spotlight('COM7', ROOT / 'spotlight.json', Path(folder) / 's.json', serial_factory=lambda port: link)
        return spot, link

    def test_handshake_point_and_light(self):
        spot, link = self.make()
        self.assertTrue(spot.connected)
        self.assertEqual(spot.firmware, 'CTRLF-SPOTLIGHT 2')
        self.assertIn('H', link.sent)                      # safe, known pose after connecting
        self.assertEqual(spot.point(0, 0), (135, 55))
        self.assertIn('P,135,55', link.sent)
        spot.light(True)
        self.assertTrue(spot.light_on)
        spot.close()
        self.assertEqual(link.sent[-1], 'L,0')           # light off on shutdown
        self.assertFalse(spot.connected)

    def test_angles_confirmed_by_board(self):
        spot, _ = self.make()
        self.assertEqual(spot.move(10, 175), (30, 150))

    def test_silent_v1_firmware_still_works(self):
        spot, link = self.make(firmware=False)
        self.assertTrue(spot.connected)
        self.assertIsNone(spot.firmware)
        self.assertIn('did not answer', spot.error)
        spot.point(.5, .5)
        self.assertIn('P,90,90', link.sent)

    def test_unplugged_board_reports_and_disconnects(self):
        spot, link = self.make()
        link.fail_writes = True
        with self.assertRaises(SpotlightError):
            spot.point(.5, .5)
        self.assertFalse(spot.connected)
        self.assertIn('lost', spot.status()['error'])

    def test_bad_port(self):
        def refuse(port):
            raise OSError('could not open port')
        spot = Spotlight(None, ROOT / 'spotlight.json', serial_factory=refuse)
        with self.assertRaises(SpotlightError):
            spot.connect('COM99')
        self.assertFalse(spot.connected)
        # Startup with a bad --serial-port must not crash the app.
        self.assertIsNotNone(Spotlight('COM99', ROOT / 'spotlight.json', serial_factory=refuse).error)


@unittest.skipUnless(shutil.which('g++'), 'g++ not installed')
class FirmwareHostTests(unittest.TestCase):
    """Compiles firmware/spotlight/spotlight.ino for the PC with stand-in
    Arduino headers and checks the serial protocol. This validates the
    command parser and safety logic, not the ESP32 hardware."""

    @classmethod
    def setUpClass(cls):
        cls.folder = tempfile.mkdtemp()
        cls.binary = Path(cls.folder) / 'firmware_host'
        host = ROOT / 'tests' / 'firmware_host'
        subprocess.run(['g++', '-std=c++17', '-Wall', '-Wextra', '-Werror', f'-I{host}', '-o', str(cls.binary),
                        str(host / 'main.cpp')], check=True, capture_output=True)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.folder, ignore_errors=True)

    def run_commands(self, *lines):
        result = subprocess.run([str(self.binary)], input='\n'.join(lines) + '\n', capture_output=True,
                                text=True, check=True, timeout=30)
        return result.stdout.strip().splitlines()

    def test_protocol(self):
        out = self.run_commands('?', 'P,100,40', 'L,1', 'L,0', 'H')
        self.assertEqual(out[:6], ['READY CTRLF-SPOTLIGHT 2', 'PONG CTRLF-SPOTLIGHT 2', 'OK P,100,40',
                                   'OK L,1', 'OK L,0', 'OK H'])
        self.assertEqual(out[-1], '#state 90 90 0')

    def test_limits_and_smooth_arrival(self):
        out = self.run_commands('P,5,179')
        self.assertIn('OK P,30,150', out)
        self.assertEqual(out[-1], '#state 30 150 0')

    def test_rejects_malformed_commands(self):
        out = self.run_commands('P,abc,10', 'P,10', 'P,200,10', 'FOO', 'P,' + '9' * 60)
        self.assertEqual(out[1:6], ['ERR angles must be whole numbers 0-180', 'ERR expected P,<pan>,<tilt>',
                                    'ERR angles must be whole numbers 0-180', 'ERR unknown command',
                                    'ERR line too long'])
        self.assertEqual(out[-1], '#state 90 90 0')     # nothing moved

    def test_led_turns_itself_off(self):
        out = self.run_commands('L,1', '#wait 125000')
        self.assertIn('EVT LED_TIMEOUT', out)
        self.assertTrue(out[-1].endswith(' 0'))


if __name__ == '__main__':
    unittest.main()
