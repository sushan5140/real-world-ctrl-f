"""Real-World Ctrl+F — webcam proof of concept (all processing local)."""
from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
import time

import cv2
import numpy as np

from tracker import (Observation, detect_markers, get_last_seen, init_db,
                     load_objects, match_object, save_observation)

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
WINDOW = "REAL-WORLD CTRL+F"
# BGR color palette: warm white, slate, coral (not neon).
WHITE = (247, 246, 241)
INK = (48, 55, 53)
CORAL = (115, 143, 232)
GREEN = (112, 136, 80)


def text(img, msg, x, y, scale=0.62, color=INK, thick=1):
    cv2.putText(img, msg, (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX,
                scale, color, thick, cv2.LINE_AA)


def make_display(frame, observations, objects, selected, record, typing, query, status):
    h, w = frame.shape[:2]
    scale = min(1.0, 980 / w)
    fw, fh = round(w * scale), round(h * scale)
    display = cv2.resize(frame, (fw, fh), interpolation=cv2.INTER_AREA)
    sidebar = 336
    panel = np.full((fh, sidebar, 3), WHITE, dtype=np.uint8)
    text(panel, "REAL-WORLD", 22, 42, 0.85, INK, 2)
    text(panel, "CTRL + F", 22, 79, 0.96, INK, 2)
    cv2.line(panel, (20, 98), (sidebar - 20, 98), (211, 211, 205), 1)
    text(panel, "F = FIND OBJECT", 20, 129, 0.58, INK, 2)
    text(panel, "1 / 2 / 3 = QUICK SEARCH", 20, 156, 0.46, INK)
    text(panel, "C = CLEAR     Q = QUIT", 20, 179, 0.46, INK)
    cv2.line(panel, (20, 195), (sidebar - 20, 195), (211, 211, 205), 1)
    for n, (id_, info) in enumerate(objects.items()):
        yy = 232 + n * 58
        if yy > fh - 105:
            break
        active = id_ == selected
        if active:
            cv2.rectangle(panel, (10, yy - 26), (sidebar - 10, yy + 19), (224, 231, 235), -1)
        text(panel, f"{n+1}. {info['name']}", 23, yy, 0.62, INK, 2 if active else 1)
        seen = "IN VIEW" if id_ in observations else "LAST SEEN" if id_ == selected and record else ""
        if seen:
            text(panel, seen, 24, yy + 17, 0.38, GREEN if seen == "IN VIEW" else CORAL)
    if typing:
        cv2.rectangle(panel, (12, fh - 89), (sidebar - 12, fh - 33), (255, 255, 255), -1)
        cv2.rectangle(panel, (12, fh - 89), (sidebar - 12, fh - 33), INK, 1)
        text(panel, "Search:", 22, fh - 95, 0.44)
        text(panel, query[-31:] + "_", 22, fh - 53, 0.52)
    else:
        text(panel, status[:45], 20, fh - 30, 0.42)
    for id_, obs in observations.items():
        if id_ not in objects:
            continue
        x1, y1, x2, y2 = (round(p * scale) for p in obs.box)
        color = CORAL if id_ == selected else GREEN
        cv2.rectangle(display, (x1, y1), (x2, y2), color, 2 if id_ != selected else 4)
        text(display, objects[id_]["name"], x1, max(y1 - 9, 23), 0.65, color, 2)
    if selected is not None and record:
        px, py = (record['x'] * fw, record['y'] * fh)
        cx, cy = round(px), round(py)
        is_visible = selected in observations
        # Last-known position remains useful even after the tag is occluded.
        pulse = 18 + round(4 * abs(np.sin(time.time() * 3)))
        cv2.circle(display, (cx, cy), pulse, CORAL if not is_visible else GREEN, 3)
        cv2.circle(display, (cx, cy), 5, CORAL if not is_visible else GREEN, -1)
        cv2.rectangle(display, (0, max(fh - 70, 0)), (fw, fh), (42, 49, 48), -1)
        name = objects[selected]['name']
        when = datetime.fromtimestamp(record['timestamp']).strftime('%H:%M:%S')
        message = (f"{name}: HERE NOW" if is_visible else
                   f"{name}: LAST SEEN HERE AT {when}")
        text(display, message, 14, fh - 30, 0.62, WHITE, 2)
    elif selected is not None:
        cv2.rectangle(display, (0, max(fh - 70, 0)), (fw, fh), (42, 49, 48), -1)
        text(display, "NOT SEEN YET - SHOW ITS TAG TO CAMERA", 14, fh - 30, 0.58, WHITE, 2)
    return np.concatenate([display, panel], axis=1)


def run(camera: int, mirrored: bool):
    objects = load_objects(ROOT / "objects.json")
    DATA.mkdir(exist_ok=True)
    conn = init_db(DATA / "locations.sqlite3")
    cap = cv2.VideoCapture(camera, cv2.CAP_DSHOW) if __import__('sys').platform == 'win32' else cv2.VideoCapture(camera)
    if not cap.isOpened():
        raise RuntimeError("Could not open webcam. Try --camera 1 or allow camera permissions.")
    selected = None
    query = ""
    typing = False
    status = "Show a printed object tag"
    last_saved: dict[int, float] = {}
    was_visible: dict[int, Observation] = {}
    # A cached camera frame exists only until the tag disappears; then we
    # persist that *last unobscured* image for an optional evidence snapshot.
    last_frame: dict[int, np.ndarray] = {}
    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                status = "Camera frame unavailable"
                break
            if mirrored:
                frame = cv2.flip(frame, 1)
            h, w = frame.shape[:2]
            observations = {i: obs for i, obs in detect_markers(frame).items()
                            if i in objects}
            now = time.monotonic()
            for i, obs in observations.items():
                last_frame[i] = frame.copy()
                if now - last_saved.get(i, 0) >= 0.75:
                    save_observation(conn, obs, w, h)
                    last_saved[i] = now
            for i in set(was_visible) - set(observations):
                old = was_visible[i]
                image = last_frame.pop(i, None)
                snapshot = None
                if image is not None:
                    snapshot = DATA / f"last_seen_{i}.jpg"
                    cv2.imwrite(str(snapshot), image, [int(cv2.IMWRITE_JPEG_QUALITY), 87])
                save_observation(conn, old, w, h, str(snapshot) if snapshot else None)
            was_visible = observations
            record = get_last_seen(conn, selected) if selected is not None else None
            display = make_display(frame, observations, objects, selected,
                                   record, typing, query, status)
            cv2.imshow(WINDOW, display)
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord('q')) and not typing:
                break
            if typing:
                if key in (13, 10):
                    selected = match_object(query, objects)
                    status = (f"Finding {objects[selected]['name']}" if selected is not None
                              else "No match - edit objects.json")
                    typing = False
                elif key == 27:
                    typing = False
                elif key in (8, 127):
                    query = query[:-1]
                elif 32 <= key <= 126 and len(query) < 80:
                    query += chr(key)
            elif key == ord('f'):
                query = ""
                typing = True
            elif key == ord('c'):
                selected = None
                status = "Selection cleared"
            elif ord('1') <= key <= ord('9'):
                number = key - ord('1')
                if number < len(objects):
                    selected = list(objects)[number]
                    status = f"Finding {objects[selected]['name']}"
            if cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
                break
    finally:
        cap.release()
        conn.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--camera", type=int, default=0, help="Webcam index, usually 0")
    parser.add_argument("--mirror", action="store_true", help="Flip camera like a selfie")
    args = parser.parse_args()
    run(camera=args.camera, mirrored=args.mirror)
