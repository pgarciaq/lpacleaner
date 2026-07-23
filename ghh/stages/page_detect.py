"""Stage 4: Page detection.

Locates the page quadrilateral in each photograph.  The quad corners
are stored in the metadata sidecar; the **full image passes through
unchanged**.  Stage 5 (perspective correction) will use the corners to
map the quad to a rectangle, which is the natural crop point.

Cropping is deferred because downstream stages (dewarp, deskew) need
full page context including edges.  Cropping early would clip content
that cannot be recovered.

Detection uses a cascading fallback chain:

1. Otsu threshold (light page on dark background)
2. Inverted Otsu (dark page on light background)
3. Canny edge detection + contour finding
4. Adaptive threshold (handles uneven lighting)
5. Full-image fallback (entire image treated as the page)

After finding the largest contour, it is refined to a 4-point quad via
``approxPolyDP`` with escalating epsilon, or ``minAreaRect`` as a last
resort.  The quad is ordered TL → TR → BR → BL via ``order_corners``
and then expanded outward by ``page_detect_expand_frac`` to compensate
for Otsu contours that sit inside the actual page boundary (due to
edge shadows and coloured elements like red titles).

Page type is classified as "music", "text", or "other" based on the
presence of staff-line-like horizontal structures in the detected page
region.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import cv2
import numpy as np

from ghh.config import Config
from ghh.pipeline import BaseStage
from ghh.utils.geometry import order_corners
from ghh.utils.line_detect import detect_staff_lines

logger = logging.getLogger(__name__)

_STAFF_LINE_THRESHOLD = 4


# ---------------------------------------------------------------------------
# Page border detection
# ---------------------------------------------------------------------------

@dataclass
class PageBorders:
    """Detected page border line positions."""
    left_x: float | None = None
    right_x: float | None = None
    top_y: float | None = None
    bottom_y: float | None = None
    confidence: float = 0.0


_BORDER_MIN_SPAN_FRAC = 0.30
_BORDER_CLUSTER_GAP_PX = 20
_BORDER_EDGE_MARGIN_FRAC = 0.35

# Only clip when the quad extends more than this fraction of the image
# dimension past the border -- small overshoots are parchment margin.
_BORDER_CLIP_THRESHOLD_FRAC = 0.04
# When clipping, preserve this fraction of image dimension as margin
# beyond the border line so the parchment margin isn't lost.
_BORDER_PRESERVE_MARGIN_FRAC = 0.02


def _detect_page_borders(
    img: np.ndarray,
    cfg: Config,
) -> PageBorders:
    """Detect red page border lines via HSV segmentation + Hough lines.

    Returns a ``PageBorders`` with the x-positions of the leftmost and
    rightmost vertical border lines, and optionally the y-positions of
    horizontal borders.  ``confidence`` reflects how many line segments
    support the detection.
    """
    h, w = img.shape[:2]
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Red spans both ends of the hue circle (OpenCV H: 0-180)
    mask_lo = cv2.inRange(hsv, (0, 40, 50), (15, 255, 255))
    mask_hi = cv2.inRange(hsv, (165, 40, 50), (180, 255, 255))
    red_mask = cv2.bitwise_or(mask_lo, mask_hi)

    # Close small gaps where neumes/text cross the border lines
    k_close = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 15))
    red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_CLOSE, k_close)

    # Thin the mask so Hough detects single lines
    k_erode = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 1))
    red_mask = cv2.erode(red_mask, k_erode, iterations=1)

    min_line_len = int(h * 0.08)
    lines = cv2.HoughLinesP(
        red_mask, rho=1, theta=np.pi / 180,
        threshold=40, minLineLength=min_line_len, maxLineGap=30,
    )
    if lines is None:
        return PageBorders()

    segments = lines.reshape(-1, 4)

    vert_segs: list[tuple[float, float, float, float]] = []
    horiz_segs: list[tuple[float, float, float, float]] = []

    for x1, y1, x2, y2 in segments:
        dx, dy = float(x2 - x1), float(y2 - y1)
        length = np.hypot(dx, dy)
        if length < 1:
            continue
        angle = abs(np.degrees(np.arctan2(dx, dy)))
        if angle < 15 or angle > 165:
            vert_segs.append((float(x1 + x2) / 2, length,
                              min(float(y1), float(y2)),
                              max(float(y1), float(y2))))
        elif 75 < angle < 105:
            horiz_segs.append((float(y1 + y2) / 2, length,
                               min(float(x1), float(x2)),
                               max(float(x1), float(x2))))

    left_x = _cluster_border_lines(
        vert_segs, w, h, side="left",
    )
    right_x = _cluster_border_lines(
        vert_segs, w, h, side="right",
    )
    top_y = _cluster_border_lines(
        horiz_segs, h, w, side="left",
    )
    bottom_y = _cluster_border_lines(
        horiz_segs, h, w, side="right",
    )

    n_found = sum(1 for v in (left_x, right_x, top_y, bottom_y) if v is not None)
    has_pair = (left_x is not None and right_x is not None)
    confidence = 0.0
    if has_pair:
        confidence = 0.8
        if top_y is not None or bottom_y is not None:
            confidence = 0.9
        if top_y is not None and bottom_y is not None:
            confidence = 1.0
    elif n_found == 1:
        confidence = 0.3

    return PageBorders(
        left_x=left_x, right_x=right_x,
        top_y=top_y, bottom_y=bottom_y,
        confidence=confidence,
    )


def _cluster_border_lines(
    segments: list[tuple[float, float, float, float]],
    span: int,
    cross_span: int,
    side: str,
) -> float | None:
    """Cluster line segments by position and return the best border.

    *segments* is a list of ``(position, length, cross_min, cross_max)``
    tuples.  *span* is the image dimension along which position varies
    (width for vertical lines, height for horizontal lines).
    *cross_span* is the other dimension (used for minimum coverage).
    *side* is ``"left"`` (pick cluster closest to 0) or ``"right"``
    (pick cluster closest to *span*).

    A cluster must satisfy two conditions to qualify:
    - total segment length >= *cross_span* * ``_BORDER_MIN_SPAN_FRAC``
    - coverage (cross_max - cross_min) >= *cross_span* * ``_BORDER_MIN_SPAN_FRAC``
      to reject compact blobs (e.g. red initials)

    Returns the weighted-mean position of the best cluster, or None.
    """
    if not segments:
        return None

    edge_margin = span * _BORDER_EDGE_MARGIN_FRAC
    min_total = cross_span * _BORDER_MIN_SPAN_FRAC
    min_coverage = cross_span * _BORDER_MIN_SPAN_FRAC

    sorted_segs = sorted(segments, key=lambda s: s[0])

    clusters: list[list[tuple[float, float, float, float]]] = []
    current: list[tuple[float, float, float, float]] = [sorted_segs[0]]

    for seg in sorted_segs[1:]:
        if seg[0] - current[-1][0] <= _BORDER_CLUSTER_GAP_PX:
            current.append(seg)
        else:
            clusters.append(current)
            current = [seg]
    clusters.append(current)

    candidates: list[tuple[float, float]] = []
    for cluster in clusters:
        total_len = sum(s[1] for s in cluster)
        if total_len < min_total:
            continue
        cross_min = min(s[2] for s in cluster)
        cross_max = max(s[3] for s in cluster)
        coverage = cross_max - cross_min
        if coverage < min_coverage:
            continue
        weighted_pos = (
            sum(s[0] * s[1] for s in cluster) / total_len
        )
        if side == "left" and weighted_pos > edge_margin:
            continue
        if side == "right" and weighted_pos < span - edge_margin:
            continue
        candidates.append((weighted_pos, total_len))

    if not candidates:
        return None

    if side == "left":
        best = min(candidates, key=lambda c: c[0])
    else:
        best = max(candidates, key=lambda c: c[0])

    return best[0]


def _refine_quad_with_borders(
    quad: np.ndarray,
    borders: PageBorders,
    img_h: int,
    img_w: int,
    confidence_threshold: float,
) -> tuple[np.ndarray, bool]:
    """Clip the quad edges inward to the detected page border lines.

    Only adjusts the **left and right** (vertical border) edges, and
    only when the quad extends **significantly** past the border
    (more than ``_BORDER_CLIP_THRESHOLD_FRAC`` of image width).

    Small overshoots are normal parchment margin and are preserved.
    When clipping *is* needed, ``_BORDER_PRESERVE_MARGIN_FRAC`` of the
    image width is kept beyond the border so the parchment margin
    isn't completely lost.

    Horizontal borders (top/bottom) are intentionally not clipped
    because page titles, page numbers, and margin annotations sit
    outside the red border lines that define the music area.

    Returns ``(refined_quad, applied)`` where *applied* is True if any
    edge was actually adjusted.
    """
    if borders.confidence < confidence_threshold:
        return quad, False

    refined = quad.copy()
    applied = False

    clip_threshold = img_w * _BORDER_CLIP_THRESHOLD_FRAC
    preserve_margin = img_w * _BORDER_PRESERVE_MARGIN_FRAC

    # TL=0, TR=1, BR=2, BL=3 (after order_corners)
    if borders.left_x is not None:
        lx = borders.left_x - preserve_margin
        for idx in (0, 3):
            overshoot = borders.left_x - refined[idx][0]
            if overshoot > clip_threshold:
                refined[idx][0] = lx
                applied = True

    if borders.right_x is not None:
        rx = borders.right_x + preserve_margin
        for idx in (1, 2):
            overshoot = refined[idx][0] - borders.right_x
            if overshoot > clip_threshold:
                refined[idx][0] = rx
                applied = True

    refined[:, 0] = np.clip(refined[:, 0], 0, img_w - 1)
    refined[:, 1] = np.clip(refined[:, 1], 0, img_h - 1)

    return refined, applied


class PageDetectStage(BaseStage):
    name = "page_detect"
    number = 4
    checkpoint_name = "04_page_detected"
    error_class = "skippable"
    config_keys = (
        "minimize_diskspace",
        "page_detect_border_refinement",
        "page_detect_border_confidence_threshold",
        "page_detect_expand_frac",
        "page_detect_method",
        "page_detect_morph_kernel",
        "page_detect_min_area_frac",
        "page_detect_epsilon",
    )

    def run(self, input_dir, output_dir, cfg, state, progress_callback=None,
            max_workers=1):
        if cfg.minimize_diskspace:
            self.writes_image = False
        return super().run(
            input_dir, output_dir, cfg, state,
            progress_callback=progress_callback,
            max_workers=max_workers,
        )

    def process_image(
        self,
        img: np.ndarray,
        metadata: dict,
        cfg: Config,
    ) -> tuple[np.ndarray, dict]:
        h, w = img.shape[:2]
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img

        quad, method = _find_page_quad(gray, cfg)

        if quad is None:
            quad = _full_image_quad(h, w)
            method = "full_image"

        quad = order_corners(quad)

        # Border-based refinement: clip quad to detected page borders
        border_meta = None
        if cfg.page_detect_border_refinement and img.ndim == 3:
            borders = _detect_page_borders(img, cfg)
            quad, border_applied = _refine_quad_with_borders(
                quad, borders, h, w,
                cfg.page_detect_border_confidence_threshold,
            )
            border_meta = {
                "left_x": borders.left_x,
                "right_x": borders.right_x,
                "top_y": borders.top_y,
                "bottom_y": borders.bottom_y,
                "confidence": round(borders.confidence, 3),
                "applied": border_applied,
            }
            if border_applied:
                logger.info(
                    "Border refinement applied (confidence=%.2f, "
                    "left=%s, right=%s, top=%s, bottom=%s)",
                    borders.confidence,
                    f"{borders.left_x:.0f}" if borders.left_x else "n/a",
                    f"{borders.right_x:.0f}" if borders.right_x else "n/a",
                    f"{borders.top_y:.0f}" if borders.top_y else "n/a",
                    f"{borders.bottom_y:.0f}" if borders.bottom_y else "n/a",
                )

        quad = _expand_quad(quad, h, w, cfg.page_detect_expand_frac)

        page_type = _classify_page_type(img, quad, cfg)

        meta: dict = {
            "stage": "page_detect",
            "method": method,
            "page_type": page_type,
            "quad_corners": quad.tolist(),
        }
        if border_meta is not None:
            meta["border_refinement"] = border_meta

        logger.info(
            "Page detected: method=%s type=%s quad_area=%.0f%%",
            method,
            page_type,
            _quad_area_frac(quad, h, w) * 100,
        )
        return img, meta


# ---------------------------------------------------------------------------
# Detection cascade
# ---------------------------------------------------------------------------

def _find_page_quad(
    gray: np.ndarray,
    cfg: Config,
) -> tuple[np.ndarray | None, str]:
    """Try each detection method in order, return (quad, method_name)."""
    if cfg.page_detect_method != "auto":
        dispatch = {
            "otsu": lambda: _try_otsu(gray, cfg, invert=False),
            "otsu_inverted": lambda: _try_otsu(gray, cfg, invert=True),
            "canny": lambda: _try_canny(gray, cfg),
            "adaptive": lambda: _try_adaptive(gray, cfg),
        }
        fn = dispatch.get(cfg.page_detect_method)
        if fn is not None:
            quad = fn()
            if quad is not None:
                return quad, cfg.page_detect_method
        return None, "none"

    for name, fn in [
        ("otsu", lambda: _try_otsu(gray, cfg, invert=False)),
        ("otsu_inverted", lambda: _try_otsu(gray, cfg, invert=True)),
        ("canny", lambda: _try_canny(gray, cfg)),
        ("adaptive", lambda: _try_adaptive(gray, cfg)),
    ]:
        quad = fn()
        if quad is not None:
            return quad, name

    return None, "none"


def _try_otsu(
    gray: np.ndarray,
    cfg: Config,
    *,
    invert: bool = False,
) -> np.ndarray | None:
    """Otsu threshold → morphological close → largest contour → quad."""
    flag = cv2.THRESH_BINARY_INV if invert else cv2.THRESH_BINARY
    _, binary = cv2.threshold(gray, 0, 255, flag + cv2.THRESH_OTSU)

    k = cfg.page_detect_morph_kernel
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k, k))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

    return _largest_quad(binary, gray.shape, cfg)


def _try_canny(gray: np.ndarray, cfg: Config) -> np.ndarray | None:
    """Canny edge detection → dilate → largest contour → quad."""
    edges = cv2.Canny(gray, 30, 100)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    dilated = cv2.dilate(edges, kernel, iterations=3)
    closed = cv2.morphologyEx(dilated, cv2.MORPH_CLOSE, kernel, iterations=2)
    return _largest_quad(closed, gray.shape, cfg)


def _try_adaptive(gray: np.ndarray, cfg: Config) -> np.ndarray | None:
    """Adaptive threshold → morphological close → largest contour → quad."""
    binary = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 51, 10,
    )
    binary = cv2.bitwise_not(binary)
    k = cfg.page_detect_morph_kernel
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k, k))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    return _largest_quad(binary, gray.shape, cfg)


def _full_image_quad(h: int, w: int) -> np.ndarray:
    """Return a quad covering the entire image (last resort).

    Uses ``[w, h]`` (not ``w-1, h-1``) so that edge-to-edge spans
    equal the image dimensions, preserving size through Stage 5.
    """
    return np.array(
        [[0, 0], [w, 0], [w, h], [0, h]],
        dtype=np.float32,
    )


# ---------------------------------------------------------------------------
# Contour → quad refinement
# ---------------------------------------------------------------------------

def _largest_quad(
    binary: np.ndarray,
    img_shape: tuple[int, ...],
    cfg: Config,
) -> np.ndarray | None:
    """Find the largest valid contour and refine it to a 4-point quad."""
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    largest = max(contours, key=cv2.contourArea)
    img_area = img_shape[0] * img_shape[1]
    contour_area = cv2.contourArea(largest)

    if contour_area < img_area * cfg.page_detect_min_area_frac:
        return None
    if contour_area > img_area * 0.999:
        return None

    return _refine_to_quad(largest, cfg)


def _refine_to_quad(contour: np.ndarray, cfg: Config) -> np.ndarray:
    """Convert a contour to a 4-point quad.

    Tries ``approxPolyDP`` with escalating epsilon values.  Falls back
    to ``minAreaRect`` if no epsilon yields exactly 4 vertices.
    """
    peri = cv2.arcLength(contour, True)

    for eps_mult in (cfg.page_detect_epsilon, 0.03, 0.04, 0.05, 0.06, 0.08):
        approx = cv2.approxPolyDP(contour, eps_mult * peri, True)
        if len(approx) == 4:
            return approx.reshape(4, 2).astype(np.float32)

    hull = cv2.convexHull(contour)
    peri_hull = cv2.arcLength(hull, True)
    for eps_mult in (0.02, 0.03, 0.04, 0.05, 0.06, 0.08):
        approx = cv2.approxPolyDP(hull, eps_mult * peri_hull, True)
        if len(approx) == 4:
            return approx.reshape(4, 2).astype(np.float32)

    rect = cv2.minAreaRect(contour)
    box = cv2.boxPoints(rect)
    return box.astype(np.float32)


# ---------------------------------------------------------------------------
# Crop and classification helpers
# ---------------------------------------------------------------------------

_BOUNDARY_MARGIN = 5


def _expand_quad(
    quad: np.ndarray,
    img_h: int,
    img_w: int,
    frac: float,
) -> np.ndarray:
    """Push each quad corner outward from the centroid by *frac* of the
    average edge length, then clamp to the image bounds.

    Corners already on the image boundary are left in place -- expanding
    them would only push the *opposite* corners further out, making the
    trapezoid more extreme without recovering any real page content.

    This compensates for Otsu contours that sit slightly inside the
    actual page boundary due to edge shadows and coloured elements.
    """
    if frac <= 0:
        return quad

    centroid = quad.mean(axis=0)
    edge_lengths = [
        np.linalg.norm(quad[1] - quad[0]),
        np.linalg.norm(quad[2] - quad[1]),
        np.linalg.norm(quad[3] - quad[2]),
        np.linalg.norm(quad[0] - quad[3]),
    ]
    avg_edge = np.mean(edge_lengths)
    expand_px = frac * avg_edge

    expanded = np.empty_like(quad)
    for i in range(4):
        x, y = quad[i]
        on_boundary = (
            x <= _BOUNDARY_MARGIN
            or x >= img_w - 1 - _BOUNDARY_MARGIN
            or y <= _BOUNDARY_MARGIN
            or y >= img_h - 1 - _BOUNDARY_MARGIN
        )
        if on_boundary:
            expanded[i] = quad[i]
        else:
            direction = quad[i] - centroid
            length = np.linalg.norm(direction)
            if length > 0:
                unit = direction / length
                expanded[i] = quad[i] + unit * expand_px
            else:
                expanded[i] = quad[i]

    expanded[:, 0] = np.clip(expanded[:, 0], 0, img_w - 1)
    expanded[:, 1] = np.clip(expanded[:, 1], 0, img_h - 1)

    return expanded


def _classify_page_type(
    img: np.ndarray,
    quad: np.ndarray,
    cfg: Config,
) -> str:
    """Classify the detected page as music, text, blank, or other.

    Uses ink-color-aware staff line detection (not generic edge detection)
    so that black text lines are not confused with coloured staff lines.
    Internally crops to the quad bounding box for efficiency.
    """
    h, w = img.shape[:2]
    x0 = int(max(0, np.floor(quad[:, 0].min())))
    y0 = int(max(0, np.floor(quad[:, 1].min())))
    x1 = int(min(w, np.ceil(quad[:, 0].max())))
    y1 = int(min(h, np.ceil(quad[:, 1].max())))
    roi = np.ascontiguousarray(img[y0:y1, x0:x1])

    staff = detect_staff_lines(roi, cfg)
    if len(staff) >= _STAFF_LINE_THRESHOLD:
        return "music"

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY) if roi.ndim == 3 else roi
    mean_val = float(np.mean(gray))
    std_val = float(np.std(gray))

    if std_val < 15 and mean_val > 200:
        return "blank"
    if std_val < 20:
        return "other"

    return "text"


def _quad_area_frac(quad: np.ndarray, h: int, w: int) -> float:
    """Fraction of image area covered by the quad."""
    area = cv2.contourArea(quad.astype(np.float32))
    return area / (h * w) if h * w > 0 else 0.0
