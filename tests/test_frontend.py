"""Front-end / back-end integration.

Static checks always run: every element the script uses exists, every API
path it calls is served, and the page obeys its own Content-Security-Policy
(no inline scripts/styles, no external resources).

A real-browser test runs when Playwright + Chromium are installed
(pip install playwright; python -m playwright install chromium). It serves the
studio on a synthetic video and drives it like a user.
"""
import os
import re
import socket
import tempfile
import threading
import time
import unittest
from pathlib import Path

import cv2

from helpers import ROOT, desk, place, textured_object, with_tag
import server
from engine import TrackingEngine
from spotlight import Spotlight

WEB = ROOT / 'web'
HTML = (WEB / 'index.html').read_text(encoding='utf-8')
JS = (WEB / 'client.js').read_text(encoding='utf-8')
CSS = (WEB / 'style.css').read_text(encoding='utf-8')


class StaticIntegrationTests(unittest.TestCase):
    def test_every_referenced_element_exists(self):
        ids = set(re.findall(r'id="([^"]+)"', HTML))
        used = set(re.findall(r"""\$\(['"]#([\w-]+)['"]\)""", JS))
        self.assertTrue(used)
        self.assertEqual(used - ids, set())
        for symbol in re.findall(r"#i-([\w-]+)", JS + HTML):
            self.assertIn(f'id="i-{symbol}"', HTML, symbol)

    def test_every_api_call_has_a_route(self):
        app = server.create_app(server.Studio(TrackingEngine(Path(tempfile.mkdtemp()), lock_folder=False)),
                                start_camera=False)
        routes = [(re.sub(r'\{[^}]+\}', '[^/]+', r.path), r.methods) for r in app.routes if hasattr(r, 'methods')]
        calls = re.findall(r"api\(\s*[`'\"]([^`'\"]+)[`'\"](?:\s*,\s*'(\w+)')?", JS)
        self.assertGreater(len(calls), 15)
        for path, method in calls:
            path = re.sub(r'\$\{[^}]+\}', 'X', path)
            method = method or 'GET'
            self.assertTrue(any(re.fullmatch(p, path) and method in m for p, m in routes), f'{method} {path}')

    def test_page_obeys_content_security_policy(self):
        self.assertNotRegex(HTML, r'\sstyle="')
        self.assertNotRegex(HTML, r'\son\w+="')
        self.assertEqual(re.findall(r'<script(?![^>]*\bsrc=)[^>]*>', HTML), [])
        self.assertNotIn('innerHTML', JS)            # DOM is built safely from text
        for text in (HTML, JS, CSS):
            external = [u for u in re.findall(r'https?://[^\s"\')]+', text) if 'www.w3.org' not in u]
            self.assertEqual(external, [])

    def test_state_names_match_backend(self):
        for state in ('visible', 'last_seen', 'unreliable', 'not_observed'):
            self.assertIn(state, JS)
            self.assertIn(f'.state-badge.{state}', CSS)


def _playwright_ready() -> bool:
    try:
        import playwright.sync_api  # noqa: F401
    except ImportError:
        return False
    return True


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(('127.0.0.1', 0))
        return s.getsockname()[1]


@unittest.skipUnless(_playwright_ready(), 'Playwright not installed (optional browser test)')
class BrowserTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import uvicorn
        from playwright.sync_api import sync_playwright
        cls.folder = tempfile.TemporaryDirectory()
        base = Path(cls.folder.name)
        video = base / 'desk.avi'
        writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*'MJPG'), 15, (960, 540))
        scene = place(desk(7), textured_object(19), 300, 300)
        for i in range(60):
            frame = with_tag(scene, 1, 650, 150)
            writer.write(frame if i < 45 else scene)       # keys disappear near the end of each loop
        writer.release()
        engine = TrackingEngine(base / 'data')
        cls.studio = server.Studio(engine, str(video), Spotlight(None, ROOT / 'spotlight.json'), max_fps=15)
        cls.port = _free_port()
        config = uvicorn.Config(server.create_app(cls.studio), host='127.0.0.1', port=cls.port, log_level='error')
        cls.server = uvicorn.Server(config)
        cls.thread = threading.Thread(target=cls.server.run, daemon=True)
        cls.thread.start()
        deadline = time.time() + 20
        while not cls.server.started and time.time() < deadline:
            time.sleep(.1)
        cls.pw = sync_playwright().start()
        executable = os.environ.get('CTRLF_CHROMIUM') or next(
            (str(p) for p in Path('/opt/pw-browsers').glob('chromium-*/chrome-linux/chrome')), None)
        try:
            cls.browser = cls.pw.chromium.launch(executable_path=executable) if executable else cls.pw.chromium.launch()
        except Exception as exc:
            cls.pw.stop()
            cls.server.should_exit = True
            raise unittest.SkipTest(f'Chromium not available: {exc}')

    @classmethod
    def tearDownClass(cls):
        cls.browser.close()
        cls.pw.stop()
        cls.server.should_exit = True
        cls.thread.join(timeout=10)
        cls.folder.cleanup()

    def open(self, width=1366, height=860):
        page = self.browser.new_page(viewport={'width': width, 'height': height})
        errors = []
        page.on('pageerror', lambda exc: errors.append(str(exc)))
        page.on('console', lambda msg: errors.append(msg.text) if msg.type == 'error' else None)
        page.goto(f'http://127.0.0.1:{self.port}/')
        page.wait_for_function("() => document.querySelector('#camera-status').textContent.startsWith('Watching')",
                               timeout=20000)
        return page, errors

    def test_search_enrol_and_forget(self):
        page, errors = self.open()
        page.wait_for_function(
            "() => [...document.querySelectorAll('.object small')].some(e => e.textContent === 'In view now')",
                               timeout=20000)
        page.keyboard.press('Control+f')
        page.keyboard.type('where are my keys?')
        page.keyboard.press('Enter')
        page.wait_for_selector('#detail:not([hidden])')
        self.assertEqual(page.text_content('#detail-name'), 'Keys')
        # Enrol the notebook by drawing on the frozen picture.
        page.click('#add')
        page.click('button.choice[value=draw]')
        page.wait_for_selector('#still:not([hidden])')
        page.wait_for_function("() => document.querySelector('#still').naturalWidth > 0")
        box = page.locator('#media').bounding_box()
        page.mouse.move(box['x'] + box['width'] * (170 / 960), box['y'] + box['height'] * (190 / 540))
        page.mouse.down()
        page.mouse.move(box['x'] + box['width'] * (430 / 960), box['y'] + box['height'] * (410 / 540), steps=8)
        page.mouse.up()
        page.wait_for_selector('#enroll-dialog[open]')
        page.fill('#object-name', 'Sketchbook')
        page.click('#enroll-save')
        page.wait_for_function("() => !document.querySelector('#enroll-dialog').open", timeout=15000)
        page.wait_for_function("() => document.querySelector('#detail-name').textContent === 'Sketchbook'")
        page.wait_for_function("() => document.querySelector('#detail-state').textContent === 'IN VIEW NOW'", timeout=20000)
        page.click('#forget')
        page.click('#confirm-ok')
        page.wait_for_function("() => ![...document.querySelectorAll('.object b')].some(e => e.textContent === 'Sketchbook')")
        self.assertEqual(errors, [])
        page.close()

    def test_mobile_layout_has_no_horizontal_scroll(self):
        page, errors = self.open(390, 844)
        self.assertLessEqual(page.evaluate('document.documentElement.scrollWidth'), 390)
        self.assertEqual(errors, [])
        page.close()


if __name__ == '__main__':
    unittest.main()
