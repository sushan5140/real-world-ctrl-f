"""Make a synthetic desk video for trying the studio without a webcam.

    python tools/make_demo_video.py            -> demo_desk.mp4
    python server.py --camera demo_desk.mp4 --data-dir demo_data

The video shows the built-in USB drive tag (#0) and keys tag (#1) on a
textured desk, an unregistered spare tag (#4), and a printed "notebook"
cover you can enrol by drawing a box around it. Midway, a hand-coloured
block covers the USB-drive tag so you can watch it change from
"in view" to "last seen". Synthetic video is for UI testing only; it
says nothing about how well a real webcam performs.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

W, H, FPS = 960, 540, 15
DICT = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)


def tag(marker_id: int, size: int) -> np.ndarray:
    quiet = size // 5
    card = np.full((size + 2 * quiet, size + 2 * quiet, 3), 250, np.uint8)
    card[quiet:quiet + size, quiet:quiet + size] = cv2.cvtColor(cv2.aruco.generateImageMarker(DICT, marker_id, size),
                                                                cv2.COLOR_GRAY2BGR)
    return card


def notebook() -> np.ndarray:
    rng = np.random.default_rng(4)
    cover = np.zeros((170, 230, 3), np.uint8)
    cover[:] = (150, 90, 40)
    for i in range(14):
        color = tuple(int(c) for c in rng.integers(60, 255, 3))
        cv2.circle(cover, (int(rng.integers(0, 230)), int(rng.integers(0, 170))), int(rng.integers(6, 30)), color, -1)
    cv2.putText(cover, 'FIELD', (18, 60), cv2.FONT_HERSHEY_DUPLEX, 1.4, (245, 240, 225), 3, cv2.LINE_AA)
    cv2.putText(cover, 'NOTES 2026', (20, 100), cv2.FONT_HERSHEY_SIMPLEX, .8, (245, 240, 225), 2, cv2.LINE_AA)
    cv2.rectangle(cover, (14, 120), (216, 150), (40, 45, 60), -1)
    cv2.putText(cover, 'Vol. 7  /  A5', (26, 142), cv2.FONT_HERSHEY_PLAIN, 1.3, (230, 230, 230), 1, cv2.LINE_AA)
    return cover


def desk() -> np.ndarray:
    rng = np.random.default_rng(1)
    base = np.array([168, 186, 196], np.float32)
    grain = cv2.resize(rng.normal(0, 1, (H // 3, 8)).astype(np.float32), (W, H))
    low = cv2.resize(rng.normal(0, 1, (H // 30, W // 30)).astype(np.float32), (W, H), interpolation=cv2.INTER_CUBIC)
    img = np.clip(base + (grain * 7 + low * 10)[..., None], 0, 255).astype(np.uint8)
    cv2.rectangle(img, (620, 330), (900, 500), (225, 228, 230), -1)          # a sheet of paper
    for i in range(6):
        cv2.putText(img, 'meeting notes - buy cables - call lab'[i * 5:i * 5 + 24], (640, 360 + i * 22),
                    cv2.FONT_HERSHEY_PLAIN, 1.0, (90, 90, 110), 1, cv2.LINE_AA)
    # A keyboard along the top edge and sticky notes: static detail that lets
    # the app verify the camera has not moved.
    for r in range(3):
        for c in range(22):
            x, y = 260 + c * 26, 18 + r * 26
            cv2.rectangle(img, (x, y), (x + 22, y + 22), (58, 60, 64), -1)
            cv2.putText(img, 'QWERTYUIOPASDFGHJKLZXCVBNM'[(r * 22 + c) % 26], (x + 6, y + 16),
                        cv2.FONT_HERSHEY_PLAIN, .9, (225, 225, 225), 1)
    for (x, y, color, word) in ((40, 330, (120, 220, 250), 'TODO'), (40, 430, (180, 230, 160), 'LAB 3'),
                                (880, 120, (200, 170, 250), 'IDEA')):
        cv2.rectangle(img, (x, y), (x + 70, y + 70), color, -1)
        cv2.putText(img, word, (x + 8, y + 40), cv2.FONT_HERSHEY_SIMPLEX, .6, (60, 60, 60), 2, cv2.LINE_AA)
    return img


def paste(frame, image, x, y, angle=0.0):
    h, w = image.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    M[:, 2] += (x - w / 2, y - h / 2)
    warped = cv2.warpAffine(image, M, (W, H))
    mask = cv2.warpAffine(np.full((h, w), 255, np.uint8), M, (W, H))
    frame[mask > 128] = warped[mask > 128]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--out', default='demo_desk.mp4')
    parser.add_argument('--seconds', type=float, default=16)
    args = parser.parse_args()
    background = desk()
    usb, keys, spare, book = tag(0, 64), tag(1, 64), tag(4, 56), notebook()
    writer = cv2.VideoWriter(args.out, cv2.VideoWriter_fourcc(*'mp4v'), FPS, (W, H))
    if not writer.isOpened():
        raise SystemExit('Could not create the video file.')
    rng = np.random.default_rng(3)
    total = int(args.seconds * FPS)
    for i in range(total):
        t = i / total
        frame = background.copy()
        paste(frame, book, 330 + 40 * np.sin(t * 2 * np.pi), 300, 8 * np.sin(t * 4 * np.pi))
        paste(frame, usb, 150, 150)
        paste(frame, keys, 760 - 120 * t, 170, 20 * t)
        paste(frame, spare, 820, 420)
        if .45 < t < .8:            # a hand covers the USB drive tag
            cv2.ellipse(frame, (160, 150), (95, 70), 0, 0, 360, (120, 150, 205), -1)
        noise = rng.normal(0, 2.5, frame.shape)
        writer.write(np.clip(frame + noise, 0, 255).astype(np.uint8))
    writer.release()
    print(f'Wrote {args.out} ({total} frames). Try: python server.py --camera {args.out} --data-dir demo_data')


if __name__ == '__main__':
    main()
