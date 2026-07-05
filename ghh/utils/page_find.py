"""Simplified page quad detection for the analyze command.

Uses Otsu thresholding + largest contour to find the page region in a
photograph. This is intentionally simpler than the full Stage 4 fallback
chain -- it only needs to be accurate enough for calibration measurements.
"""

from __future__ import annotations

import cv2
import numpy as np


def find_page_quad(img: np.ndarray) -> np.ndarray | None:
    """Detect the page quadrilateral in an image.

    Tries normal Otsu first (light page on dark background), then
    inverted Otsu (dark page on light background).

    Returns
    -------
    (4, 2) float32 array of corner points ordered TL, TR, BR, BL,
    or None if no page boundary is detectable.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    for invert in (False, True):
        quad = _try_otsu(gray, invert=invert)
        if quad is not None:
            return quad

    return None


def crop_to_page(
    img: np.ndarray,
    quad: np.ndarray | None,
) -> np.ndarray:
    """Crop image to the page region defined by *quad*.

    If *quad* is None, falls back to the central 80% of the image.
    """
    if quad is None:
        h, w = img.shape[:2]
        margin_y = h // 10
        margin_x = w // 10
        return np.ascontiguousarray(img[margin_y:h - margin_y, margin_x:w - margin_x])

    xs = quad[:, 0]
    ys = quad[:, 1]
    x0 = max(0, int(np.floor(xs.min())))
    y0 = max(0, int(np.floor(ys.min())))
    x1 = min(img.shape[1], int(np.ceil(xs.max())))
    y1 = min(img.shape[0], int(np.ceil(ys.max())))

    return np.ascontiguousarray(img[y0:y1, x0:x1])


def _try_otsu(gray: np.ndarray, invert: bool = False) -> np.ndarray | None:
    """Attempt Otsu threshold and find largest quadrilateral contour."""
    if invert:
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    else:
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=3)

    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    largest = max(contours, key=cv2.contourArea)

    img_area = gray.shape[0] * gray.shape[1]
    contour_area = cv2.contourArea(largest)
    if contour_area < img_area * 0.1:
        return None
    if contour_area > img_area * 0.95:
        return None

    peri = cv2.arcLength(largest, True)
    approx = cv2.approxPolyDP(largest, 0.02 * peri, True)

    if len(approx) == 4:
        return _order_quad(approx.reshape(4, 2).astype(np.float32))

    rect = cv2.minAreaRect(largest)
    box = cv2.boxPoints(rect)
    return _order_quad(box.astype(np.float32))


def _order_quad(pts: np.ndarray) -> np.ndarray:
    """Order 4 points as TL, TR, BR, BL."""
    s = pts.sum(axis=1)
    d = np.diff(pts, axis=1).ravel()

    ordered = np.zeros((4, 2), dtype=np.float32)
    ordered[0] = pts[np.argmin(s)]   # TL: smallest sum
    ordered[2] = pts[np.argmax(s)]   # BR: largest sum
    ordered[1] = pts[np.argmin(d)]   # TR: smallest difference
    ordered[3] = pts[np.argmax(d)]   # BL: largest difference
    return ordered
