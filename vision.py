"""Offline visual enrollment for reasonably textured physical objects.

This is feature-template matching, NOT open-world semantic recognition.
A position is reported only after a geometrically consistent homography.
"""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import cv2
import numpy as np

MIN_FEATURES = 16
MIN_INLIERS = 10


@dataclass
class Template:
    object_id: int
    image: np.ndarray
    keypoints: list
    descriptors: np.ndarray


@dataclass(frozen=True)
class Match:
    object_id: int
    x: float
    y: float
    box: tuple[int, int, int, int]
    inliers: int


def orb():
    return cv2.ORB_create(nfeatures=1200, scaleFactor=1.2, nlevels=8,
                          edgeThreshold=12, patchSize=31, fastThreshold=12)


def create_template(object_id: int, crop: np.ndarray) -> Template:
    if crop is None or crop.ndim != 3 or min(crop.shape[:2]) < 40:
        raise ValueError('Select a region at least 40 by 40 pixels.')
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    kp, desc = orb().detectAndCompute(gray, None)
    if desc is None or len(kp) < MIN_FEATURES:
        raise ValueError('Not enough distinctive details. Try better light, a closer view or a textured object.')
    return Template(object_id, crop.copy(), kp, desc)


def find_templates(frame: np.ndarray, templates: dict[int, Template]) -> dict[int, Match]:
    if not templates:
        return {}
    height, width = frame.shape[:2]
    kp_frame, desc_frame = orb().detectAndCompute(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), None)
    if desc_frame is None or len(kp_frame) < MIN_FEATURES:
        return {}
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
    found: dict[int, Match] = {}
    for obj_id, template in templates.items():
        pairs = matcher.knnMatch(template.descriptors, desc_frame, k=2)
        good = [a for a, b in pairs if a.distance < 0.72 * b.distance]
        if len(good) < MIN_INLIERS:
            continue
        src = np.float32([template.keypoints[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst = np.float32([kp_frame[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
        H, mask = cv2.findHomography(src, dst, cv2.RANSAC, 3.0)
        if H is None or mask is None or int(mask.sum()) < MIN_INLIERS:
            continue
        th, tw = template.image.shape[:2]
        corners = np.float32([[0, 0], [tw - 1, 0], [tw - 1, th - 1], [0, th - 1]]).reshape(-1, 1, 2)
        warped = cv2.perspectiveTransform(corners, H).reshape(4, 2)
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
        found[obj_id] = Match(obj_id, float(np.clip(cx / width, 0, 1)),
                              float(np.clip(cy / height, 0, 1)),
                              (int(max(0, x0)), int(max(0, y0)),
                               int(min(width - 1, x1)), int(min(height - 1, y1))),
                              int(mask.sum()))
    return found


def save_template(path: Path, template: Template) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), template.image):
        raise OSError('Failed to save template image.')


def load_templates(folder: Path, ids: set[int]) -> dict[int, Template]:
    loaded = {}
    for obj_id in ids:
        path = folder / f'{obj_id}.png'
        image = cv2.imread(str(path)) if path.is_file() else None
        if image is None:
            continue
        try:
            loaded[obj_id] = create_template(obj_id, image)
        except ValueError:
            continue
    return loaded
