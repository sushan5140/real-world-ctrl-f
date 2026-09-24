"""Markerless recognition of *specific enrolled objects* (instance matching).

This is local feature matching (ORB + geometric verification), NOT open-world
or category-level recognition. It can find "this particular notebook cover"
that was enrolled; it cannot find "any phone". It works best for flat-ish,
textured, matte surfaces and poorly for plain, shiny, transparent, thin or
strongly 3-D objects.

Pipeline for one frame:
  1. ORB features are extracted once for the whole frame.
  2. Each enrolled view is matched: Lowe ratio test, absolute distance cap,
     one-to-one matches.
  3. Geometric verification: MAGSAC/RANSAC homography, plausible
     quadrilateral (convex, sane angles, scale and aspect), inliers spread
     over the object's surface and present in the *centre* of the enrolment
     box. If the strongest consistent group fails (typically the desk
     background captured around the object), the next group is tried.
  4. A confidence score (0..1) is returned; overlapping detections of
     different objects are resolved in favour of the stronger one.

Single-frame results are not trusted on their own: engine.py confirms a
detection over consecutive frames and tolerates short dropouts.

Evaluated alternatives (see tools/benchmark_recognition.py and the README):
CLAHE contrast normalisation, SIFT, gridded ORB and region-of-interest
tracking gave no measurable gain on the benchmark, so they are not used.

Templates can have several views (e.g. front and back, or two distances).
Each view has a mask that keeps features on the object and drops features
on the surrounding background that was inside the user's rectangle.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

# ---- tunables (validated with tools/benchmark_recognition.py) -------------
MIN_TEMPLATE_FEATURES = 25      # features on the object itself needed to enrol
GOOD_TEMPLATE_FEATURES = 80     # above this we call the view "good"
MIN_INLIERS = 12                # geometric inliers needed for a detection
MIN_CONFIDENCE = 0.40           # detections below this are ignored
RATIO = 0.75                    # Lowe ratio test
MAX_HAMMING = 72                # absolute descriptor-distance ceiling
GRID = 4                        # coverage grid (GRID x GRID cells per view)
CENTRAL_MARGIN = .2             # central region = box shrunk by 20% per side
CENTRAL_MIN_HITS = 5
CENTRAL_FRACTION = .5
FRAME_FEATURES = 5000
VIEW_FEATURES = 1000
MIN_CROP_PX = 40

_HOMOGRAPHY_METHOD = getattr(cv2, 'USAC_MAGSAC', cv2.RANSAC)


def _orb(n_features: int) -> cv2.ORB:
    return cv2.ORB_create(nfeatures=n_features, scaleFactor=1.2, nlevels=8,
                          edgeThreshold=15, patchSize=31, fastThreshold=10)


def normalise_gray(image: np.ndarray) -> np.ndarray:
    # CLAHE contrast normalisation was evaluated and *reduced* recall on the
    # benchmark (it amplifies webcam noise / JPEG artefacts), so plain grey is used.
    return image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


# ---------------------------------------------------------------------------
# Templates
# ---------------------------------------------------------------------------
@dataclass
class View:
    image: np.ndarray                 # BGR crop as enrolled
    mask: np.ndarray                  # uint8, 255 where features may be used
    keypoints: list
    descriptors: np.ndarray
    points: np.ndarray                # (N, 2) float32 keypoint coordinates
    cells: np.ndarray                 # (N,) coverage-grid cell of each keypoint
    occupied_cells: int
    mask_method: str = 'none'
    central: np.ndarray | None = None  # (N,) bool: keypoint in the central region
    outline: np.ndarray | None = None  # (4, 2) object corners inside the crop (from the mask)

    @property
    def size(self) -> tuple[int, int]:
        h, w = self.image.shape[:2]
        return w, h

    @property
    def quality(self) -> str:
        return 'good' if len(self.keypoints) >= GOOD_TEMPLATE_FEATURES else 'fair'


@dataclass
class Template:
    object_id: int
    views: list[View] = field(default_factory=list)

    # Backward-compatible accessors used by older code / tests.
    @property
    def image(self) -> np.ndarray:
        return self.views[0].image

    @property
    def keypoints(self) -> list:
        return self.views[0].keypoints

    @property
    def descriptors(self) -> np.ndarray:
        return self.views[0].descriptors


@dataclass(frozen=True)
class Match:
    object_id: int
    x: float
    y: float
    box: tuple[int, int, int, int]
    inliers: int
    confidence: float = 1.0
    quad: tuple[tuple[float, float], ...] = ()
    view: int = 0


def object_mask(crop: np.ndarray) -> tuple[np.ndarray, str]:
    """Estimate which pixels inside the user's rectangle belong to the object.

    GrabCut separates the object from the background inside the rectangle.
    The mask is then eroded a little, because features on the object's
    silhouette also describe whatever was behind it at enrolment time.
    If GrabCut gives an implausible answer we fall back to dropping a thin
    border, which is where background usually sits in a hand-drawn box.
    """
    h, w = crop.shape[:2]
    fallback = np.zeros((h, w), np.uint8)
    my, mx = max(2, int(h * .07)), max(2, int(w * .07))
    fallback[my:h - my, mx:w - mx] = 255
    if min(h, w) < 60:
        return fallback, 'border'
    try:
        gc_mask = np.zeros((h, w), np.uint8)
        margin = max(2, int(min(h, w) * .04))
        rect = (margin, margin, w - 2 * margin, h - 2 * margin)
        bgd, fgd = np.zeros((1, 65), np.float64), np.zeros((1, 65), np.float64)
        cv2.grabCut(crop, gc_mask, rect, bgd, fgd, 4, cv2.GC_INIT_WITH_RECT)
        fg = np.where((gc_mask == cv2.GC_FGD) | (gc_mask == cv2.GC_PR_FGD), 255, 0).astype(np.uint8)
    except cv2.error:
        return fallback, 'border'
    fraction = float(fg.mean()) / 255
    if not .18 <= fraction <= .97:
        return fallback, 'border'
    # Keep the largest connected blob; stray background islands are dropped.
    count, labels, stats, _ = cv2.connectedComponentsWithStats(fg, connectivity=8)
    if count > 2:
        largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        fg = np.where(labels == largest, 255, 0).astype(np.uint8)
    k = max(3, int(min(h, w) * .03) | 1)
    fg = cv2.erode(fg, np.ones((k, k), np.uint8))
    return fg, 'grabcut'


def build_view(crop: np.ndarray, mask: np.ndarray | None = None,
               auto_mask: bool = True) -> View:
    """Extract features for one enrolled view. Raises ValueError if unusable."""
    if crop is None or crop.ndim != 3 or min(crop.shape[:2]) < MIN_CROP_PX:
        raise ValueError(f'Select a region at least {MIN_CROP_PX} by {MIN_CROP_PX} pixels.')
    method = 'provided'
    if mask is None:
        if auto_mask:
            mask, method = object_mask(crop)
        else:
            mask, method = np.full(crop.shape[:2], 255, np.uint8), 'none'
    gray = normalise_gray(crop)
    keypoints, descriptors = _orb(VIEW_FEATURES).detectAndCompute(gray, mask)
    if descriptors is None or len(keypoints) < MIN_TEMPLATE_FEATURES:
        found = 0 if descriptors is None else len(keypoints)
        raise ValueError(
            f'Not enough surface detail on the object ({found} features, need '
            f'{MIN_TEMPLATE_FEATURES}). Draw the box tightly around the object, improve '
            'the lighting or move the camera closer. Plain, shiny or transparent objects '
            'are better tracked with a printed tag.')
    points = np.float32([kp.pt for kp in keypoints])
    h, w = crop.shape[:2]
    cx = np.minimum((points[:, 0] / w * GRID).astype(int), GRID - 1)
    cy = np.minimum((points[:, 1] / h * GRID).astype(int), GRID - 1)
    cells = cy * GRID + cx
    central = ((points[:, 0] > w * CENTRAL_MARGIN) & (points[:, 0] < w * (1 - CENTRAL_MARGIN)) &
               (points[:, 1] > h * CENTRAL_MARGIN) & (points[:, 1] < h * (1 - CENTRAL_MARGIN)))
    # The object's own extent inside the (usually loose) box, so detections
    # outline the object rather than the whole enrolment rectangle.
    ys, xs = np.nonzero(mask)
    grow = max(2, int(min(h, w) * .03))
    if len(xs):
        ox0, oy0 = max(0, xs.min() - grow), max(0, ys.min() - grow)
        ox1, oy1 = min(w - 1, xs.max() + grow), min(h - 1, ys.max() + grow)
    else:
        ox0, oy0, ox1, oy1 = 0, 0, w - 1, h - 1
    outline = np.float32([[ox0, oy0], [ox1, oy0], [ox1, oy1], [ox0, oy1]])
    return View(crop.copy(), mask, keypoints, descriptors, points, cells,
                int(len(np.unique(cells))), method, central, outline)


def create_template(object_id: int, crop: np.ndarray, auto_mask: bool = True) -> Template:
    return Template(object_id, [build_view(crop, auto_mask=auto_mask)])


# ---------------------------------------------------------------------------
# Matching
# ---------------------------------------------------------------------------
def _plausible_quad(quad: np.ndarray, view_size: tuple[int, int],
                    frame_w: int, frame_h: int) -> bool:
    if not np.isfinite(quad).all():
        return False
    contour = quad.astype(np.float32)
    if not cv2.isContourConvex(contour):
        return False
    area = abs(cv2.contourArea(contour))
    tw, th = view_size
    if area < 300 or area > .8 * frame_w * frame_h:
        return False
    scale = math.sqrt(area / max(1.0, tw * th))
    if not .12 <= scale <= 8:
        return False
    x0, y0 = quad.min(axis=0)
    x1, y1 = quad.max(axis=0)
    if x0 < -.25 * frame_w or y0 < -.25 * frame_h or x1 > 1.25 * frame_w or y1 > 1.25 * frame_h:
        return False
    sides = [np.linalg.norm(quad[(i + 1) % 4] - quad[i]) for i in range(4)]
    if min(sides) < 4:
        return False
    # Opposite sides of a (perspective-warped) rectangle stay comparable.
    if max(sides[0], sides[2]) / min(sides[0], sides[2]) > 3 or \
       max(sides[1], sides[3]) / min(sides[1], sides[3]) > 3:
        return False
    # Aspect ratio may not change wildly relative to the template.
    template_aspect = tw / max(1, th)
    observed_aspect = ((sides[0] + sides[2]) / 2) / max(1e-6, (sides[1] + sides[3]) / 2)
    ratio = observed_aspect / template_aspect
    if not 1 / 3.5 <= ratio <= 3.5:
        return False
    for i in range(4):
        a = quad[i - 1] - quad[i]
        b = quad[(i + 1) % 4] - quad[i]
        cos = float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))
        if not -.87 <= cos <= .87:        # corner angle within ~30..150 degrees
            return False
    return True


def _match_view(view: View, kp_pts: np.ndarray, descriptors: np.ndarray,
                matcher: cv2.BFMatcher, frame_w: int, frame_h: int):
    """Return (inliers, confidence, quad in frame coordinates) or None."""
    if descriptors is None or len(descriptors) < 2:
        return None
    pairs = matcher.knnMatch(view.descriptors, descriptors, k=2)
    best_for_train: dict[int, cv2.DMatch] = {}
    for pair in pairs:
        if len(pair) < 2:
            continue
        m, n = pair
        if m.distance > MAX_HAMMING or m.distance >= RATIO * n.distance:
            continue
        prev = best_for_train.get(m.trainIdx)
        if prev is None or m.distance < prev.distance:
            best_for_train[m.trainIdx] = m
    good = list(best_for_train.values())
    # Several consistent groups can exist (e.g. the object where it is now AND
    # the desk background that was inside the enrolment box). If the strongest
    # group fails verification, remove it and try the next one.
    for _attempt in range(3):
        if len(good) < MIN_INLIERS:
            return None
        result, inlier_mask = _verify(view, good, kp_pts, frame_w, frame_h)
        if result is not None:
            return result
        if inlier_mask is None:
            return None
        good = [m for m, used in zip(good, inlier_mask) if not used]
    return None


def _verify(view: View, good: list, kp_pts: np.ndarray, frame_w: int, frame_h: int):
    """Fit and check one homography. Returns (result or None, inlier mask or None)."""
    src = view.points[[m.queryIdx for m in good]].reshape(-1, 1, 2)
    dst = kp_pts[[m.trainIdx for m in good]].reshape(-1, 1, 2)
    H, mask = cv2.findHomography(src, dst, _HOMOGRAPHY_METHOD, 4.0,
                                 maxIters=2000, confidence=.995)
    if H is None or mask is None:
        return None, None
    inlier_mask = mask.ravel().astype(bool)
    inliers = int(inlier_mask.sum())
    if inliers < MIN_INLIERS:
        return None, None
    if inliers < .2 * len(good):
        return None, inlier_mask
    tw, th = view.size
    corners = np.float32([[0, 0], [tw - 1, 0], [tw - 1, th - 1], [0, th - 1]]).reshape(-1, 1, 2)
    quad = cv2.perspectiveTransform(corners, H).reshape(4, 2)
    if not _plausible_quad(quad, view.size, frame_w, frame_h):
        return None, inlier_mask
    inlier_idx = np.array([m.queryIdx for m, keep in zip(good, inlier_mask) if keep])
    # The user centres the box on the object, so a real detection must include
    # the template's centre. Matches only on the rim of the box are usually
    # the desk/background that was captured around the object.
    if view.central is not None and view.central.any():
        expected = float(view.central.mean())
        central_hits = int(view.central[inlier_idx].sum())
        if central_hits < CENTRAL_MIN_HITS or central_hits / inliers < CENTRAL_FRACTION * expected:
            return None, inlier_mask
    hit_cells = len(np.unique(view.cells[inlier_idx]))
    needed_cells = max(3, int(round(.3 * view.occupied_cells)))
    if hit_cells < min(needed_cells, view.occupied_cells):
        return None, inlier_mask
    strength = 1 - math.exp(-inliers / 18)
    spread = min(1.0, hit_cells / max(4, .5 * view.occupied_cells))
    confidence = round(float(min(.99, strength * (.55 + .45 * spread))), 3)
    if view.outline is not None:
        quad = cv2.perspectiveTransform(view.outline.reshape(-1, 1, 2), H).reshape(4, 2)
    return (inliers, confidence, quad), inlier_mask


def _iou(a, b) -> float:
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


class Matcher:
    """Create once and reuse; holds the ORB detector and descriptor matcher."""

    def __init__(self, frame_features: int = FRAME_FEATURES):
        self.frame_orb = _orb(frame_features)
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING)

    def find(self, frame: np.ndarray, templates: dict[int, Template]) -> dict[int, Match]:
        if not templates:
            return {}
        height, width = frame.shape[:2]
        keypoints, descriptors = self.frame_orb.detectAndCompute(normalise_gray(frame), None)
        if descriptors is None or len(keypoints) < MIN_INLIERS:
            return {}
        points = np.float32([k.pt for k in keypoints])
        results: dict[int, Match] = {}
        for obj_id, template in templates.items():
            best = None
            for index, view in enumerate(template.views):
                found = _match_view(view, points, descriptors, self.matcher, width, height)
                if found and (best is None or found[1] > best[1]):
                    best = (*found, index)
            if best is None or best[1] < MIN_CONFIDENCE:
                continue
            inliers, confidence, quad, view_index = best
            x0, y0 = quad.min(axis=0)
            x1, y1 = quad.max(axis=0)
            box = (int(max(0, x0)), int(max(0, y0)), int(min(width - 1, x1)), int(min(height - 1, y1)))
            if box[2] - box[0] < 4 or box[3] - box[1] < 4:
                continue
            cx, cy = quad.mean(axis=0)
            results[obj_id] = Match(obj_id, float(np.clip(cx / width, 0, 1)), float(np.clip(cy / height, 0, 1)),
                                    box, inliers, confidence,
                                    tuple((float(px), float(py)) for px, py in quad), view_index)
        # Identity conflicts: two enrolled objects claimed at the same place.
        # Keep the stronger claim; the weaker one is not reported this frame.
        kept: dict[int, Match] = {}
        for match in sorted(results.values(), key=lambda m: m.confidence, reverse=True):
            if all(_iou(match.box, other.box) < .5 for other in kept.values()):
                kept[match.object_id] = match
        return kept


_default_matcher: Matcher | None = None


def find_templates(frame: np.ndarray, templates: dict[int, Template]) -> dict[int, Match]:
    """Convenience wrapper kept for compatibility with the original API."""
    global _default_matcher
    if _default_matcher is None:
        _default_matcher = Matcher()
    return _default_matcher.find(frame, templates)


# ---------------------------------------------------------------------------
# Storage: data/templates/<id>.png, <id>_<n>.png (+ matching .mask.png)
# ---------------------------------------------------------------------------
def view_filename(object_id: int, index: int) -> str:
    return f'{object_id}.png' if index == 0 else f'{object_id}_{index}.png'


def save_view(folder: Path, filename: str, view: View) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    image_path = folder / filename
    mask_path = folder / filename.replace('.png', '.mask.png')
    if not cv2.imwrite(str(image_path), view.image) or not cv2.imwrite(str(mask_path), view.mask):
        image_path.unlink(missing_ok=True)
        mask_path.unlink(missing_ok=True)
        raise OSError('Failed to save the object template.')


def delete_view_files(folder: Path, filename: str) -> None:
    (folder / filename).unlink(missing_ok=True)
    (folder / filename.replace('.png', '.mask.png')).unlink(missing_ok=True)


def save_template(path: Path, template: Template) -> None:
    """Compatibility helper: save the first view at an explicit path."""
    save_view(path.parent, path.name, template.views[0])


def load_view(folder: Path, filename: str) -> View:
    """Load one stored view. Raises ValueError/OSError when it is unusable."""
    image_path = folder / filename
    image = cv2.imread(str(image_path)) if image_path.is_file() else None
    if image is None:
        raise OSError(f'Template image {filename} is missing or unreadable.')
    mask_path = folder / filename.replace('.png', '.mask.png')
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE) if mask_path.is_file() else None
    if mask is not None and mask.shape != image.shape[:2]:
        mask = None
    # Templates enrolled by the original version have no mask; derive one so
    # they get the same background protection as new enrolments.
    return build_view(image, mask=mask, auto_mask=True)


def load_templates(folder: Path, ids) -> dict[int, Template]:
    """Compatibility loader: one view per id stored as <id>.png."""
    loaded = {}
    for obj_id in ids:
        try:
            loaded[obj_id] = Template(obj_id, [load_view(folder, view_filename(obj_id, 0))])
        except (OSError, ValueError):
            continue
    return loaded
