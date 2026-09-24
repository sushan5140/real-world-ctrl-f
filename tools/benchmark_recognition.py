"""Synthetic benchmark for markerless (enrolled-object) recognition.

Why: claims about recognition quality should be measured, not assumed.
This script builds reproducible synthetic desk scenes with printed-looking
flat objects, enrols them the way a user would (a *loose* rectangle drawn on
a fixed-camera frame, which includes some desk background), then measures:

  recall        object present and found at the right place with the right id
  false alarms  an enrolled object reported although it is NOT in the frame
                (the scene still contains the desk area it was enrolled on)
  wrong place   reported, but far from where it really is
  ms/frame      recognition time on this computer

It compares the original matcher shipped in v0.2 ("legacy") with the
current vision.py pipeline. Synthetic scenes are optimistic compared with a
real webcam (flat objects, no reflections), so treat results as *relative*.

Run:  python tools/benchmark_recognition.py [--trials 60] [--seed 7]
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import vision  # noqa: E402

W, H = 960, 540
FONTS = [cv2.FONT_HERSHEY_SIMPLEX, cv2.FONT_HERSHEY_DUPLEX, cv2.FONT_HERSHEY_COMPLEX,
         cv2.FONT_HERSHEY_TRIPLEX, cv2.FONT_HERSHEY_PLAIN]
WORDS = ['NOTES', 'COFFEE', 'ATLAS', 'VOL 2', 'LAB', 'MAP', 'TEA', 'FIELD', 'KIT', 'NOVA',
         'SKETCH', '2026', 'ORBIT', 'DESK', 'PAPER', 'MINT', 'ARC', 'DELTA', 'Q7', 'JOURNAL']


# ---------------------------------------------------------------------------
# Legacy matcher (verbatim logic of the original vision.find_templates)
# ---------------------------------------------------------------------------
def legacy_orb():
    return cv2.ORB_create(nfeatures=1200, scaleFactor=1.2, nlevels=8,
                          edgeThreshold=12, patchSize=31, fastThreshold=12)


def legacy_template(crop):
    kp, desc = legacy_orb().detectAndCompute(cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY), None)
    if desc is None or len(kp) < 16:
        raise ValueError('few features')
    return crop.copy(), kp, desc


def legacy_find(frame, templates):
    height, width = frame.shape[:2]
    kp_frame, desc_frame = legacy_orb().detectAndCompute(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), None)
    if desc_frame is None or len(kp_frame) < 16:
        return {}
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
    found = {}
    for obj_id, (image, kps, descs) in templates.items():
        pairs = matcher.knnMatch(descs, desc_frame, k=2)
        good = [a for a, b in (p for p in pairs if len(p) == 2) if a.distance < 0.72 * b.distance]
        if len(good) < 10:
            continue
        src = np.float32([kps[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst = np.float32([kp_frame[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
        Hm, mask = cv2.findHomography(src, dst, cv2.RANSAC, 3.0)
        if Hm is None or mask is None or int(mask.sum()) < 10:
            continue
        th, tw = image.shape[:2]
        corners = np.float32([[0, 0], [tw - 1, 0], [tw - 1, th - 1], [0, th - 1]]).reshape(-1, 1, 2)
        warped = cv2.perspectiveTransform(corners, Hm).reshape(4, 2)
        if not np.isfinite(warped).all():
            continue
        area = abs(cv2.contourArea(warped.astype(np.float32)))
        if area < 300 or area > 0.75 * width * height or not cv2.isContourConvex(warped.astype(np.float32)):
            continue
        x0, y0 = warped.min(axis=0)
        x1, y1 = warped.max(axis=0)
        if x0 < -0.1 * width or y0 < -0.1 * height or x1 > 1.1 * width or y1 > 1.1 * height:
            continue
        cx, cy = warped.mean(axis=0)
        found[obj_id] = (cx / width, cy / height)
    return found


# ---------------------------------------------------------------------------
# Scene synthesis
# ---------------------------------------------------------------------------
def make_object(rng) -> np.ndarray:
    w, h = int(rng.integers(170, 260)), int(rng.integers(120, 190))
    c0, c1 = rng.integers(40, 230, 3), rng.integers(40, 230, 3)
    t = np.linspace(0, 1, w)[None, :, None]
    img = (c0 * (1 - t) + c1 * t).repeat(h, axis=0).astype(np.uint8)
    for _ in range(int(rng.integers(3, 7))):
        color = tuple(int(v) for v in rng.integers(0, 255, 3))
        kind = rng.integers(0, 3)
        if kind == 0:
            cv2.circle(img, (int(rng.integers(0, w)), int(rng.integers(0, h))), int(rng.integers(8, 40)), color, -1)
        elif kind == 1:
            p = rng.integers(0, [w, h, w, h])
            cv2.rectangle(img, (int(p[0]), int(p[1])), (int(p[2]), int(p[3])), color, int(rng.integers(-1, 4)) or -1)
        else:
            pts = rng.integers(0, [w, h], (int(rng.integers(3, 6)), 2)).astype(np.int32)
            cv2.polylines(img, [pts], True, color, 2)
    for _ in range(int(rng.integers(2, 5))):
        word = ' '.join(rng.choice(WORDS, int(rng.integers(1, 3))))
        color = tuple(int(v) for v in rng.integers(0, 255, 3))
        cv2.putText(img, word, (int(rng.integers(0, w // 2)), int(rng.integers(20, h))),
                    FONTS[int(rng.integers(0, len(FONTS)))], float(rng.uniform(.5, 1.1)), color,
                    int(rng.integers(1, 3)), cv2.LINE_AA)
    cv2.rectangle(img, (0, 0), (w - 1, h - 1), (30, 30, 30), 2)
    return img


def make_desk(rng) -> np.ndarray:
    base = np.array(rng.integers(120, 200, 3), np.float32)
    noise = cv2.resize(rng.normal(0, 1, (H // 24, W // 24)).astype(np.float32), (W, H), interpolation=cv2.INTER_CUBIC)
    grain = cv2.resize(rng.normal(0, 1, (H // 2, 6)).astype(np.float32), (W, H), interpolation=cv2.INTER_LINEAR)
    desk = base[None, None, :] + (noise * 14 + grain * 8)[..., None]
    desk = np.clip(desk, 0, 255).astype(np.uint8)
    # Clutter that is NOT enrolled: papers with text, a keyboard-like grid.
    for _ in range(4):
        clutter = make_object(rng)
        ch, cw = clutter.shape[:2]
        x, y = int(rng.integers(0, W - cw)), int(rng.integers(0, H - ch))
        desk[y:y + ch, x:x + cw] = clutter
    kx, ky = int(rng.integers(0, W - 300)), int(rng.integers(0, H - 110))
    for r in range(4):
        for c in range(12):
            cv2.rectangle(desk, (kx + c * 25, ky + r * 26), (kx + c * 25 + 21, ky + r * 26 + 22), (60, 60, 60), -1)
            cv2.putText(desk, chr(65 + (r * 12 + c) % 26), (kx + c * 25 + 5, ky + r * 26 + 16),
                        cv2.FONT_HERSHEY_PLAIN, .9, (230, 230, 230), 1)
    return desk


def place(scene: np.ndarray, obj: np.ndarray, quad: np.ndarray) -> None:
    h, w = obj.shape[:2]
    src = np.float32([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]])
    M = cv2.getPerspectiveTransform(src, quad.astype(np.float32))
    warped = cv2.warpPerspective(obj, M, (W, H))
    mask = cv2.warpPerspective(np.full((h, w), 255, np.uint8), M, (W, H))
    scene[mask > 128] = warped[mask > 128]


def random_quad(rng, obj, scale_range=(.55, 1.2), persp=.12, avoid=()):
    h, w = obj.shape[:2]
    for _ in range(200):
        s = rng.uniform(*scale_range)
        angle = rng.uniform(-np.pi, np.pi)
        cx, cy = rng.uniform(.15 * W, .85 * W), rng.uniform(.2 * H, .8 * H)
        rect = np.float32([[-w / 2, -h / 2], [w / 2, -h / 2], [w / 2, h / 2], [-w / 2, h / 2]]) * s
        rect += rng.uniform(-persp, persp, (4, 2)) * np.float32([w, h]) * s
        rot = np.float32([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
        quad = rect @ rot.T + np.float32([cx, cy])
        if quad.min() < 0 or quad[:, 0].max() >= W or quad[:, 1].max() >= H:
            continue
        box = (*quad.min(axis=0), *quad.max(axis=0))
        if any(not (box[2] < b[0] - 10 or box[0] > b[2] + 10 or box[3] < b[1] - 10 or box[1] > b[3] + 10) for b in avoid):
            continue
        return quad
    return None


def photometric(rng, frame: np.ndarray) -> np.ndarray:
    img = frame.astype(np.float32)
    img = img * rng.uniform(.7, 1.3) + rng.uniform(-40, 40)
    img = np.clip(img, 0, 255)
    gamma = rng.uniform(.7, 1.4)
    img = 255 * (img / 255) ** gamma
    sigma = rng.uniform(0, 1.2)
    if sigma > .3:
        img = cv2.GaussianBlur(img, (0, 0), sigma)
    img += rng.normal(0, rng.uniform(0, 6), img.shape)
    img = np.clip(img, 0, 255).astype(np.uint8)
    ok, enc = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, 75])
    return cv2.imdecode(enc, cv2.IMREAD_COLOR)


def occlude(rng, frame, quad, max_fraction=.35):
    x0, y0 = quad.min(axis=0)
    x1, y1 = quad.max(axis=0)
    fraction = rng.uniform(0, max_fraction)
    ow, oh = (x1 - x0) * np.sqrt(fraction), (y1 - y0) * np.sqrt(fraction)
    ox, oy = rng.uniform(x0, x1 - ow), rng.uniform(y0, y1 - oh)
    cv2.rectangle(frame, (int(ox), int(oy)), (int(ox + ow), int(oy + oh)), (95, 125, 185), -1)


# ---------------------------------------------------------------------------
def run(trials: int, seed: int, objects_per_scene: int = 3):
    rng = np.random.default_rng(seed)
    configs = {
        'legacy (v0.2)': 'legacy',
        'new, without object mask': 'nomask',
        'new (mask + checks)': 'new',
    }
    stats = {name: dict(pos=0, hit=0, wrong_place=0, neg=0, false_alarm=0, ms=0.0, frames=0) for name in configs}
    scenes = max(1, trials // 10)
    for _scene in range(scenes):
        desk = make_desk(rng)
        objects = [make_object(rng) for _ in range(objects_per_scene)]
        # Enrolment: each object lies on the desk; the user drags a loose box.
        enrol_frame = desk.copy()
        boxes = []
        for obj in objects:
            q = random_quad(rng, obj, (.8, 1.0), .03, boxes)
            if q is None:
                break
            boxes.append((*q.min(axis=0), *q.max(axis=0)))
            place(enrol_frame, obj, q)
        if len(boxes) < len(objects):
            continue  # could not lay the objects out without overlap; next scene
        enrol_frame = photometric(np.random.default_rng(seed + 99), enrol_frame)
        templates = {name: {} for name in configs}
        for obj_id, box in enumerate(boxes):
            x0, y0, x1, y1 = box
            mx, my = (x1 - x0) * .15, (y1 - y0) * .15
            crop = enrol_frame[int(max(0, y0 - my)):int(min(H, y1 + my)), int(max(0, x0 - mx)):int(min(W, x1 + mx))]
            templates['legacy (v0.2)'][obj_id] = legacy_template(crop)
            templates['new, without object mask'][obj_id] = vision.create_template(obj_id, crop, auto_mask=False)
            try:
                templates['new (mask + checks)'][obj_id] = vision.create_template(obj_id, crop)
            except ValueError:
                pass
        matcher = vision.Matcher()
        for _ in range(10):
            frame = desk.copy()
            present = {}
            placed = []
            for obj_id, obj in enumerate(objects):
                if rng.random() < .55:
                    q = random_quad(rng, obj, avoid=placed)
                    if q is None:
                        continue
                    place(frame, obj, q)
                    placed.append((*q.min(axis=0), *q.max(axis=0)))
                    present[obj_id] = q
            for q in present.values():
                if rng.random() < .5:
                    occlude(rng, frame, q)
            frame = photometric(rng, frame)
            for name, kind in configs.items():
                start = time.perf_counter()
                if kind == 'legacy':
                    found = legacy_find(frame, templates[name])
                else:
                    found = {k: (m.x, m.y) for k, m in matcher.find(frame, templates[name]).items()}
                s = stats[name]
                s['ms'] += (time.perf_counter() - start) * 1000
                s['frames'] += 1
                for obj_id in range(len(objects)):
                    if obj_id in present:
                        s['pos'] += 1
                        q = present[obj_id]
                        if obj_id in found:
                            cx, cy = q.mean(axis=0)
                            diag = np.linalg.norm(q.max(axis=0) - q.min(axis=0))
                            fx, fy = found[obj_id]
                            if np.hypot(fx * W - cx, fy * H - cy) <= .25 * diag:
                                s['hit'] += 1
                            else:
                                s['wrong_place'] += 1
                    else:
                        s['neg'] += 1
                        if obj_id in found:
                            s['false_alarm'] += 1
    return stats


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--trials', type=int, default=60, help='number of frames (scenes x 10)')
    parser.add_argument('--seed', type=int, default=7)
    args = parser.parse_args()
    stats = run(args.trials, args.seed)
    print(f"{'matcher':28} {'recall':>8} {'wrong place':>12} {'false alarms':>13} {'ms/frame':>9}")
    for name, s in stats.items():
        recall = s['hit'] / max(1, s['pos'])
        wrong = s['wrong_place'] / max(1, s['pos'])
        fa = s['false_alarm'] / max(1, s['neg'])
        print(f"{name:28} {recall:8.1%} {wrong:12.1%} {fa:13.1%} {s['ms'] / max(1, s['frames']):9.1f}")
    print(f"(positives per matcher: {stats['legacy (v0.2)']['pos']}, negatives: {stats['legacy (v0.2)']['neg']})")


if __name__ == '__main__':
    main()
