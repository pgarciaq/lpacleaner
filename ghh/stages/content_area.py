"""Stage 6: Content area detection.

Detects the border frame that surrounds content on most pages, providing
a tighter crop than the page edge, and masks residual adjacent-page edges.

Algorithm
---------
1. **Border frame detection** (when ``has_border_frame`` is True):

   a. Isolate ink mask via ``detect_ink_mask``
   b. Morphological close (5x5) to join nearby fragments
   c. HoughLinesP to find long horizontal and vertical ink-colored lines
   d. Filter: horizontals near top/bottom, verticals near left/right
   e. Intersect the 4 border lines to find the content rectangle
   f. Fallback if ``has_border_frame`` is False or < 4 lines found:
      ink-density bounding box with padding, or fixed inset

2. **Mask adjacent-page edges**:

   Fill pixels outside the content rectangle with the estimated
   background color using Gaussian feathering to avoid hard edges.

3. **Crop and add uniform margins**:

   Crop to the content rectangle, then add padding (default 2% of
   width) filled with the background color.

4. Store content rectangle coordinates in sidecar metadata.

If Stage 6 is skipped (``skip_content_area``) or the page is classified
as "blank" (from Stage 4/5 metadata), the image passes through unchanged.
"""

from __future__ import annotations

import logging

import cv2
import numpy as np

from ghh.config import Config
from ghh.pipeline import BaseStage
from ghh.utils.line_detect import detect_ink_mask

logger = logging.getLogger(__name__)

_MIN_CONTENT_FRAC = 0.10


class ContentAreaStage(BaseStage):
    name = "content_area"
    number = 6
    checkpoint_name = "06_content"
    error_class = "skippable"
    config_keys = (
        "has_border_frame",
        "content_detect_inset_fallback",
        "content_feather_sigma",
        "content_margin_padding",
    )

    def process_image(
        self,
        img: np.ndarray,
        metadata: dict,
        cfg: Config,
    ) -> tuple[np.ndarray, dict]:
        if metadata.get("page_type") == "blank":
            logger.info("Blank page, passing through unchanged")
            return img, _passthrough_meta(metadata)

        h, w = img.shape[:2]

        rect, method = _detect_content_rect(img, cfg)

        if rect is None:
            rect = _inset_fallback(h, w, cfg.content_detect_inset_fallback)
            method = "inset_fallback"

        x, y, rw, rh = rect
        if rw * rh < h * w * _MIN_CONTENT_FRAC:
            logger.warning(
                "Content rect too small (%.1f%% of image), using inset fallback",
                100.0 * rw * rh / (h * w),
            )
            rect = _inset_fallback(h, w, cfg.content_detect_inset_fallback)
            method = "inset_fallback"
            x, y, rw, rh = rect

        bg_color = _estimate_background(img, rect)
        masked = _feather_outside(img, rect, bg_color, cfg.content_feather_sigma)

        cropped = masked[y : y + rh, x : x + rw]

        padded, pad_px = _add_margins(cropped, bg_color, cfg.content_margin_padding)

        meta = {
            "stage": "content_area",
            "method": method,
            "content_rect": [x, y, rw, rh],
            "margin_px": pad_px,
            "background_color": [int(c) for c in bg_color],
        }
        if "page_type" in metadata:
            meta["page_type"] = metadata["page_type"]

        logger.info(
            "Content area: method=%s rect=[%d,%d,%d,%d] margin=%dpx",
            method, x, y, rw, rh, pad_px,
        )
        return padded, meta


def _passthrough_meta(metadata: dict) -> dict:
    meta = {"stage": "content_area", "method": "passthrough"}
    if "page_type" in metadata:
        meta["page_type"] = metadata["page_type"]
    return meta


# ---------------------------------------------------------------------------
# Border frame detection via Hough lines
# ---------------------------------------------------------------------------

def _detect_content_rect(
    img: np.ndarray,
    cfg: Config,
) -> tuple[tuple[int, int, int, int] | None, str]:
    """Try to detect the border frame rectangle.

    Returns ((x, y, w, h), method) or (None, "") if detection fails.
    """
    if not cfg.has_border_frame:
        return _ink_density_rect(img, cfg)

    h, w = img.shape[:2]

    ink_mask = detect_ink_mask(img, cfg)
    if np.count_nonzero(ink_mask) == 0:
        return None, ""

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    closed = cv2.morphologyEx(ink_mask, cv2.MORPH_CLOSE, kernel)

    rect = _hough_border_rect(closed, h, w)
    if rect is not None:
        return rect, "hough_border"

    return _ink_density_rect(img, cfg)


def _hough_border_rect(
    mask: np.ndarray,
    img_h: int,
    img_w: int,
) -> tuple[int, int, int, int] | None:
    """Find 4 border lines via HoughLinesP and intersect them."""
    min_length = int(min(img_h, img_w) * 0.25)

    lines = cv2.HoughLinesP(
        mask,
        rho=1,
        theta=np.pi / 180,
        threshold=80,
        minLineLength=min_length,
        maxLineGap=30,
    )

    if lines is None:
        return None

    horizontals = []
    verticals = []
    margin_h = int(img_h * 0.35)
    margin_w = int(img_w * 0.35)

    for line in lines:
        x1, y1, x2, y2 = line.ravel()
        dx, dy = x2 - x1, y2 - y1
        length = np.sqrt(dx * dx + dy * dy)
        if length < min_length:
            continue

        angle = abs(np.degrees(np.arctan2(dy, dx)))

        if angle < 10 or angle > 170:
            y_mid = (y1 + y2) / 2
            horizontals.append((y_mid, x1, x2, length))
        elif 80 < angle < 100:
            x_mid = (x1 + x2) / 2
            verticals.append((x_mid, y1, y2, length))

    top_lines = [ln for ln in horizontals if ln[0] < margin_h]
    bottom_lines = [ln for ln in horizontals if ln[0] > img_h - margin_h]
    left_lines = [ln for ln in verticals if ln[0] < margin_w]
    right_lines = [ln for ln in verticals if ln[0] > img_w - margin_w]

    if not (top_lines and bottom_lines and left_lines and right_lines):
        return None

    top_y = int(min(ln[0] for ln in top_lines))
    bottom_y = int(max(ln[0] for ln in bottom_lines))
    left_x = int(min(ln[0] for ln in left_lines))
    right_x = int(max(ln[0] for ln in right_lines))

    if right_x <= left_x or bottom_y <= top_y:
        return None

    return (left_x, top_y, right_x - left_x, bottom_y - top_y)


def _ink_density_rect(
    img: np.ndarray,
    cfg: Config,
) -> tuple[tuple[int, int, int, int] | None, str]:
    """Fallback: bounding box of ink-dense regions with padding."""
    ink_mask = detect_ink_mask(img, cfg)
    if np.count_nonzero(ink_mask) == 0:
        return None, ""

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
    dilated = cv2.dilate(ink_mask, kernel)

    coords = cv2.findNonZero(dilated)
    if coords is None:
        return None, ""

    x, y, w, h = cv2.boundingRect(coords)
    img_h, img_w = img.shape[:2]

    pad = int(min(img_h, img_w) * 0.02)
    x = max(0, x - pad)
    y = max(0, y - pad)
    w = min(img_w - x, w + 2 * pad)
    h = min(img_h - y, h + 2 * pad)

    return (x, y, w, h), "ink_density"


def _inset_fallback(
    img_h: int, img_w: int, frac: float,
) -> tuple[int, int, int, int]:
    """Fixed-percentage inset from all edges."""
    dx = int(img_w * frac)
    dy = int(img_h * frac)
    return (dx, dy, img_w - 2 * dx, img_h - 2 * dy)


# ---------------------------------------------------------------------------
# Background estimation and masking
# ---------------------------------------------------------------------------

def _estimate_background(
    img: np.ndarray,
    content_rect: tuple[int, int, int, int],
) -> tuple[int, ...]:
    """Estimate background color from pixels outside the content rect."""
    h, w = img.shape[:2]
    x, y, rw, rh = content_rect

    mask = np.ones((h, w), dtype=bool)
    mask[y : y + rh, x : x + rw] = False

    if not np.any(mask):
        border = max(1, int(min(h, w) * 0.05))
        strips = [img[:border, :], img[-border:, :]]
        if h > 2 * border:
            strips.extend([img[border:-border, :border], img[border:-border, -border:]])
        if img.ndim == 3:
            pixels = np.concatenate([s.reshape(-1, img.shape[2]) for s in strips])
            return tuple(int(np.median(pixels[:, c])) for c in range(img.shape[2]))
        pixels = np.concatenate([s.ravel() for s in strips])
        return (int(np.median(pixels)),)

    outside = img[mask]
    if img.ndim == 3:
        return tuple(int(np.median(outside[:, c])) for c in range(img.shape[2]))
    return (int(np.median(outside)),)


def _feather_outside(
    img: np.ndarray,
    rect: tuple[int, int, int, int],
    bg_color: tuple[int, ...],
    sigma: int,
) -> np.ndarray:
    """Replace pixels outside rect with bg_color, feathered at the edges."""
    h, w = img.shape[:2]
    x, y, rw, rh = rect

    alpha = np.zeros((h, w), dtype=np.float32)
    alpha[y : y + rh, x : x + rw] = 1.0

    if sigma > 0:
        ksize = sigma * 6 + 1
        if ksize % 2 == 0:
            ksize += 1
        alpha = cv2.GaussianBlur(alpha, (ksize, ksize), sigma)

    bg = np.full_like(img, bg_color, dtype=np.uint8)

    if img.ndim == 3:
        alpha_3 = alpha[:, :, np.newaxis]
        result = (img.astype(np.float32) * alpha_3 +
                  bg.astype(np.float32) * (1.0 - alpha_3))
    else:
        result = img.astype(np.float32) * alpha + bg.astype(np.float32) * (1.0 - alpha)

    return np.clip(result, 0, 255).astype(np.uint8)


def _add_margins(
    img: np.ndarray,
    bg_color: tuple[int, ...],
    padding_frac: float,
) -> tuple[np.ndarray, int]:
    """Add uniform margins around the image filled with bg_color."""
    h, w = img.shape[:2]
    pad = max(1, int(w * padding_frac))

    if img.ndim == 3:
        padded = np.full(
            (h + 2 * pad, w + 2 * pad, img.shape[2]),
            bg_color, dtype=np.uint8,
        )
    else:
        padded = np.full((h + 2 * pad, w + 2 * pad), bg_color[0], dtype=np.uint8)

    padded[pad : pad + h, pad : pad + w] = img
    return padded, pad
