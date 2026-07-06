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

import cv2
import numpy as np

from ghh.config import Config
from ghh.pipeline import BaseStage
from ghh.utils.geometry import order_corners
from ghh.utils.line_detect import detect_staff_lines

logger = logging.getLogger(__name__)

_STAFF_LINE_THRESHOLD = 4


class PageDetectStage(BaseStage):
    name = "page_detect"
    number = 4
    checkpoint_name = "04_page_detected"
    error_class = "skippable"

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
        quad = _expand_quad(quad, h, w, cfg.page_detect_expand_frac)

        page_type = _classify_page_type(img, quad, cfg)

        meta = {
            "stage": "page_detect",
            "method": method,
            "page_type": page_type,
            "quad_corners": quad.tolist(),
        }

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

def _expand_quad(
    quad: np.ndarray,
    img_h: int,
    img_w: int,
    frac: float,
) -> np.ndarray:
    """Push each quad corner outward from the centroid by *frac* of the
    average edge length, then clamp to the image bounds.

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
