"""Real-World Ctrl+F — minimal desktop window (no browser needed).

Uses exactly the same tracking engine, object library and memory as the
browser studio (server.py), so objects enrolled in the browser are found
here too. Run only one of the two at a time.

Keys:  F = type a search   1-9 = quick select   C = clear   Q / Esc = quit
"""
from __future__ import annotations

import argparse
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from camera import open_capture, parse_source
from engine import LAST_SEEN, ROOT, UNRELIABLE, VISIBLE, TrackingEngine

WINDOW = 'REAL-WORLD CTRL+F'
WHITE = (247, 246, 241)
INK = (48, 55, 53)
CORAL = (84, 125, 231)
GREEN = (104, 176, 118)
AMBER = (60, 160, 230)
GREY = (150, 150, 150)
STATE_LABEL = {VISIBLE: 'IN VIEW NOW', LAST_SEEN: 'LAST SEEN', UNRELIABLE: 'UNCERTAIN',
               'not_observed': 'NOT SEEN YET'}
STATE_COLOR = {VISIBLE: GREEN, LAST_SEEN: CORAL, UNRELIABLE: AMBER, 'not_observed': GREY}


def text(img, msg, x, y, scale=.62, color=INK, thick=1):
    msg = str(msg).encode('ascii', 'replace').decode('ascii')
    cv2.putText(img, msg, (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick, cv2.LINE_AA)


def make_display(frame, items, selected, typing, query, status_line):
    h, w = frame.shape[:2]
    scale = min(1.0, 980 / w)
    fw, fh = round(w * scale), round(h * scale)
    display = cv2.resize(frame, (fw, fh), interpolation=cv2.INTER_AREA)
    sidebar = 340
    panel = np.full((fh, sidebar, 3), WHITE, dtype=np.uint8)
    text(panel, 'REAL-WORLD', 22, 42, .85, INK, 2)
    text(panel, 'CTRL + F', 22, 79, .96, INK, 2)
    cv2.line(panel, (20, 98), (sidebar - 20, 98), (211, 211, 205), 1)
    text(panel, 'F = FIND OBJECT', 20, 129, .58, INK, 2)
    text(panel, '1-9 = QUICK SELECT', 20, 156, .46, INK)
    text(panel, 'C = CLEAR     Q = QUIT', 20, 179, .46, INK)
    cv2.line(panel, (20, 195), (sidebar - 20, 195), (211, 211, 205), 1)
    for n, item in enumerate(items):
        yy = 232 + n * 52
        if yy > fh - 105:
            break
        active = item['id'] == selected
        if active:
            cv2.rectangle(panel, (10, yy - 24), (sidebar - 10, yy + 20), (224, 231, 235), -1)
        text(panel, f"{n + 1}. {item['name']}", 23, yy, .6, INK, 2 if active else 1)
        text(panel, STATE_LABEL[item['state']], 26, yy + 16, .38, STATE_COLOR[item['state']], 1)
    if typing:
        cv2.rectangle(panel, (12, fh - 89), (sidebar - 12, fh - 33), (255, 255, 255), -1)
        cv2.rectangle(panel, (12, fh - 89), (sidebar - 12, fh - 33), INK, 1)
        text(panel, 'Search:', 22, fh - 95, .44)
        text(panel, query[-31:] + '_', 22, fh - 53, .52)
    else:
        text(panel, status_line[:46], 20, fh - 30, .42)
    item = next((i for i in items if i['id'] == selected), None)
    if item:
        state = item['state']
        pos = item['live'] or ({'x': item['x'], 'y': item['y']} if item['x'] is not None else None)
        cv2.rectangle(display, (0, max(fh - 70, 0)), (fw, fh), (42, 49, 48), -1)
        if pos:
            cx, cy = round(pos['x'] * fw), round(pos['y'] * fh)
            color = STATE_COLOR[state]
            pulse = 18 + round(4 * abs(np.sin(time.time() * 3)))
            cv2.circle(display, (cx, cy), pulse, color, 3, cv2.LINE_AA)
            cv2.circle(display, (cx, cy), 5, color, -1, cv2.LINE_AA)
        if state == VISIBLE:
            message = f"{item['name']}: IN VIEW NOW"
        elif state == 'not_observed':
            message = f"{item['name']}: NOT SEEN YET - SHOW IT TO THE CAMERA"
        else:
            when = datetime.fromtimestamp(item['last_seen']).strftime('%H:%M:%S')
            prefix = 'LAST SEEN HERE' if state == LAST_SEEN else 'UNCERTAIN - LAST SEEN HERE'
            message = f"{item['name']}: {prefix} AT {when} (MAY HAVE MOVED)"
        text(display, message, 14, fh - 30, .58, WHITE, 2)
    return np.concatenate([display, panel], axis=1)


def run(camera, mirrored: bool, data_dir: Path) -> None:
    engine = TrackingEngine(data_dir)
    cap = open_capture(camera)
    if not cap.isOpened():
        engine.close()
        raise SystemExit('Could not open the camera. Try --camera 1 or allow camera permissions.')
    selected, query, typing = None, '', False
    status_line = 'Show a printed tag or an enrolled object'
    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                status_line = 'Camera frame unavailable'
                break
            if mirrored:
                frame = cv2.flip(frame, 1)
            result = engine.process(frame)
            engine.selected = selected
            canvas = engine.annotate(frame, result)
            items = engine.status()['items']
            cv2.imshow(WINDOW, make_display(canvas, items, selected, typing, query, status_line))
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord('q')) and not typing:
                break
            if typing:
                if key in (13, 10):
                    try:
                        found = engine.search(query)
                        if found['ambiguous']:
                            names = ' / '.join(c['name'] for c in found['candidates'])
                            status_line = f'Which one? {names}'
                        else:
                            selected = found['id']
                            status_line = f"Finding {engine.catalog.objects[selected].name}"
                    except LookupError:
                        status_line = 'No match - try another name'
                    typing = False
                elif key == 27:
                    typing = False
                elif key in (8, 127):
                    query = query[:-1]
                elif 32 <= key <= 126 and len(query) < 80:
                    query += chr(key)
            elif key == ord('f'):
                query, typing = '', True
            elif key == ord('c'):
                selected, status_line = None, 'Selection cleared'
            elif ord('1') <= key <= ord('9'):
                number = key - ord('1')
                if number < len(items):
                    selected = items[number]['id']
                    status_line = f"Finding {items[number]['name']}"
            if cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
                break
    finally:
        cap.release()
        engine.close()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--camera', default='0', help='webcam number, usually 0, or a video file')
    parser.add_argument('--mirror', action='store_true', help='flip the picture like a selfie camera '
                        '(note: positions are remembered in the flipped picture)')
    parser.add_argument('--data-dir', default=str(ROOT / 'data'))
    args = parser.parse_args()
    try:
        run(parse_source(args.camera), args.mirror, Path(args.data_dir))
    except RuntimeError as exc:
        raise SystemExit(f'Error: {exc}') from exc
