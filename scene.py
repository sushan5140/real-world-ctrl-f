"""Camera-position watchdog.

Remembered positions are image coordinates. They only mean something while
the camera stays where it was. This module compares the live view with a
reference of the static scene about once per second:

  * stable     the view still lines up with the reference
  * moved      the view shifted/rotated for several checks in a row, or the
               whole scene changed -> a new "camera epoch" starts and every
               older location is shown as unreliable
  * uncertain  too little matching detail to tell (dark, covered lens,
               blank desk); nothing is concluded

The reference is persisted (keypoint positions + ORB descriptors only, not an
image), so a restart can tell whether the camera is still in the same pose.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

WORK_WIDTH = 480
CHECK_INTERVAL = 1.0
MOVED_SHIFT = .04        # fraction of image width
STABLE_SHIFT = .015
MOVED_CONFIRMATIONS = 3
LOST_CONFIRMATIONS = 12  # checks with a detailed view that matches nothing
MIN_INLIERS = 25
MIN_SPREAD = 6           # of 16 grid cells that must agree on a camera move
REFRESH_SECONDS = 60
REFRESH_MAX_SHIFT = .006  # only re-anchor when the view is essentially identical


@dataclass
class _Reference:
    points: np.ndarray
    descriptors: np.ndarray
    size: tuple[int, int]         # original frame (w, h)


class SceneMonitor:
    def __init__(self, reference_path: Path | None = None):
        self.path = reference_path
        self.orb = cv2.ORB_create(nfeatures=1200, fastThreshold=12)
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
        self.reference: _Reference | None = self._load()
        self.state = 'checking' if self.reference else 'new'
        self.verified = False          # at least one stable check since start
        self.last_check = 0.0
        self.last_refresh = 0.0
        self.shift = 0.0
        self._moved_count = 0
        self._lost_count = 0

    # ---- persistence -------------------------------------------------------
    def _load(self) -> _Reference | None:
        if not self.path or not self.path.is_file():
            return None
        try:
            data = np.load(self.path)
            return _Reference(data['points'].astype(np.float32), data['descriptors'].astype(np.uint8),
                              (int(data['size'][0]), int(data['size'][1])))
        except (OSError, ValueError, KeyError):
            return None

    def _save(self) -> None:
        if not self.path or self.reference is None:
            return
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.path.with_name(self.path.stem + '.tmp.npz')
            np.savez_compressed(tmp, points=self.reference.points, descriptors=self.reference.descriptors,
                                size=np.array(self.reference.size))
            tmp.replace(self.path)
        except OSError:
            pass

    def forget(self) -> None:
        self.reference = None
        self.state = 'new'
        if self.path:
            self.path.unlink(missing_ok=True)

    # ---- analysis ----------------------------------------------------------
    def _features(self, frame: np.ndarray):
        h, w = frame.shape[:2]
        scale = WORK_WIDTH / w
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
        small = cv2.resize(gray, (WORK_WIDTH, max(1, round(h * scale))), interpolation=cv2.INTER_AREA)
        keypoints, descriptors = self.orb.detectAndCompute(small, None)
        points = np.float32([k.pt for k in keypoints]) / scale if keypoints else np.zeros((0, 2), np.float32)
        return points, descriptors

    def _set_reference(self, frame, points, descriptors) -> None:
        h, w = frame.shape[:2]
        self.reference = _Reference(points, descriptors, (w, h))
        self._save()

    def update(self, frame: np.ndarray, now: float, exclude=()) -> str | None:
        """Feed a frame. Returns 'moved' when a new camera epoch should start,
        'verified' the first time the pose is confirmed, otherwise None.

        `exclude` holds pixel boxes of objects currently being tracked: their
        own movement must not be mistaken for the camera moving."""
        if now - self.last_check < CHECK_INTERVAL:
            return None
        self.last_check = now
        points, descriptors = self._features(frame)
        if exclude and len(points):
            keep = np.ones(len(points), bool)
            for x0, y0, x1, y1 in exclude:
                px, py = (x1 - x0) * .15 + 8, (y1 - y0) * .15 + 8
                keep &= ~((points[:, 0] >= x0 - px) & (points[:, 0] <= x1 + px) &
                          (points[:, 1] >= y0 - py) & (points[:, 1] <= y1 + py))
            points = points[keep]
            descriptors = descriptors[keep] if descriptors is not None else None
        detailed = descriptors is not None and len(points) >= 150
        h, w = frame.shape[:2]
        if self.reference is None:
            if detailed:
                self._set_reference(frame, points, descriptors)
                self.last_refresh = now
                self.state, self.verified = 'stable', True
                return 'verified'
            self.state = 'uncertain'
            return None
        if self.reference.size != (w, h):
            # Different camera or resolution: old coordinates do not transfer.
            if detailed:
                self._set_reference(frame, points, descriptors)
            self.state, self.verified, self.shift = 'moved', True, 1.0
            return 'moved'
        transform, inliers, spread = None, 0, 0
        if descriptors is not None and len(descriptors) >= 10:
            pairs = self.matcher.knnMatch(self.reference.descriptors, descriptors, k=2)
            good = [p[0] for p in pairs if len(p) == 2 and p[0].distance < .75 * p[1].distance]
            if len(good) >= MIN_INLIERS:
                src = self.reference.points[[m.queryIdx for m in good]]
                dst = points[[m.trainIdx for m in good]]
                # First ask the cheap, direct question: do enough features,
                # spread over the picture, sit exactly where they were? If
                # so the camera has not moved, whatever else moves on the desk.
                still = np.linalg.norm(dst - src, axis=1) <= 3.0 * w / WORK_WIDTH
                if int(still.sum()) >= MIN_INLIERS and _spread(dst[still], w, h) >= 4:
                    return self._stable(frame, points, descriptors, now, 0.0)
                transform, mask = cv2.estimateAffinePartial2D(src, dst, method=cv2.RANSAC,
                                                              ransacReprojThreshold=3.0 * w / WORK_WIDTH)
                inliers = int(mask.sum()) if mask is not None else 0
                if inliers:
                    # A camera move shifts the *whole* picture. Agreement that
                    # comes from one region only (e.g. a book being slid across
                    # the desk) is not evidence about the camera.
                    spread = _spread(dst[mask.ravel().astype(bool)], w, h)
        if transform is None or inliers < MIN_INLIERS:
            self._moved_count = 0
            self._lost_count = self._lost_count + 1 if detailed else 0
            if self._lost_count >= LOST_CONFIRMATIONS:
                # A detailed view that shares nothing with the reference for
                # a while: the camera now looks at a different scene.
                self._set_reference(frame, points, descriptors)
                self._lost_count = 0
                self.state, self.verified, self.shift = 'moved', True, 1.0
                return 'moved'
            self.state = 'uncertain'
            return None
        self._lost_count = 0
        corners = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
        moved_corners = corners @ transform[:, :2].T + transform[:, 2]
        self.shift = float(np.linalg.norm(moved_corners - corners, axis=1).max() / w)
        if self.shift >= MOVED_SHIFT and spread < MIN_SPREAD:
            self._moved_count = 0
            self.state = 'uncertain'
            return None
        if self.shift >= MOVED_SHIFT:
            self._moved_count += 1
            if self._moved_count >= MOVED_CONFIRMATIONS:
                self._set_reference(frame, points, descriptors)
                self._moved_count = 0
                self.state, self.verified = 'moved', True
                return 'moved'
            self.state = 'checking'
            return None
        self._moved_count = 0
        if self.shift <= STABLE_SHIFT:
            return self._stable(frame, points, descriptors, now, self.shift)
        self.state = 'checking'
        return None

    def _stable(self, frame, points, descriptors, now: float, shift: float) -> str | None:
        self._moved_count = self._lost_count = 0
        self.shift = shift
        first = not self.verified
        self.state, self.verified = 'stable', True
        if now - self.last_refresh >= REFRESH_SECONDS and shift <= REFRESH_MAX_SHIFT:
            # Follow slow changes (lighting, objects on the desk) while the
            # camera is demonstrably still in the same place.
            self._set_reference(frame, points, descriptors)
            self.last_refresh = now
        return 'verified' if first else None


def _spread(points: np.ndarray, w: int, h: int) -> int:
    """How many cells of a 4x4 grid over the picture contain these points."""
    if not len(points):
        return 0
    return len(set(zip(np.minimum((points[:, 0] / w * 4).astype(int), 3),
                       np.minimum((points[:, 1] / h * 4).astype(int), 3))))
