"""Flash hotspot removal (R1) and finger detection/masking (R8).

Used by Stage 0 (pre-processing). Runs before stitching so that
fingers and hotspots do not contaminate feature matching.
"""

from __future__ import annotations

import logging

import cv2
import numpy as np

from ghh.config import Config

logger = logging.getLogger(__name__)

# Default thresholds (overridable via Config in future)
_CLIP_THRESHOLD = 250
_HOTSPOT_MIN_AREA_FRAC = 0.005
_HOTSPOT_INPAINT_RADIUS = 5
_FINGER_MIN_AREA_FRAC = 0.01
_FINGER_INPAINT_RADIUS = 10


def remove_hotspots(
    img: np.ndarray,
    cfg: Config,
) -> tuple[np.ndarray, dict]:
    """R1: Detect and remove flash hotspots / specular highlights via inpainting.

    Algorithm:
        1. Detect clipped regions: mask = (B > 250) & (G > 250) & (R > 250)
        2. Dilate mask with 5x5 kernel (catch hotspot edges)
        3. If hotspot area > 0.5% of image: flag as flash-affected
        4. Inpaint: cv2.inpaint(img, mask, inpaintRadius=5, flags=INPAINT_TELEA)
        5. Record hotspot locations in metadata

    Returns:
        (result_image, metadata_dict)
    """
    clip = _CLIP_THRESHOLD
    b, g, r = cv2.split(img)
    mask = ((b > clip) & (g > clip) & (r > clip)).astype(np.uint8) * 255

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    mask = cv2.dilate(mask, kernel)

    area_frac = np.count_nonzero(mask) / mask.size
    meta: dict = {}

    if area_frac > _HOTSPOT_MIN_AREA_FRAC:
        logger.info("Hotspot detected: %.2f%% of image clipped", area_frac * 100)
        # Scale inpaint radius to the largest hotspot blob so the center gets filled
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        max_dim = 0
        for i in range(1, num_labels):
            max_dim = max(max_dim, stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT])
        inpaint_radius = max(_HOTSPOT_INPAINT_RADIUS, max_dim // 2)
        result = cv2.inpaint(img, mask, inpaint_radius, cv2.INPAINT_TELEA)
        meta["hotspot_detected"] = True
        meta["hotspot_area_frac"] = float(area_frac)
    else:
        result = img.copy()
        meta["hotspot_detected"] = False

    return result, meta


def detect_fingers(
    img: np.ndarray,
    cfg: Config,
) -> np.ndarray:
    """R8: Detect skin-colored regions touching the image border.

    Algorithm:
        1. Convert to YCrCb color space
        2. Skin mask: (133 < Cr < 173) & (77 < Cb < 127)
        3. Subtract pixels matching the book's ink color (avoids red ink
           being mistaken for skin)
        4. Filter: border-touching, area 1-15%, aspect ratio < 4:1

    Returns:
        Binary uint8 mask (255 = finger, 0 = not finger).
    """
    ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    cr = ycrcb[:, :, 1]
    cb = ycrcb[:, :, 2]

    skin_mask = ((cr > 133) & (cr < 173) & (cb > 77) & (cb < 127)).astype(np.uint8) * 255

    # Exclude pixels that match the book's ink color -- red/brown ink
    # falls in the YCrCb skin range and must not be treated as skin.
    # Ink is highly saturated; skin is more pastel (lower saturation,
    # higher value). Requiring saturation > 80 avoids masking actual skin.
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hue = hsv[:, :, 0].astype(np.int16)
    sat = hsv[:, :, 1]
    ink_hue = cfg.staff_color_hue
    ink_range = cfg.staff_color_range
    hue_diff = np.minimum(
        np.abs(hue - ink_hue),
        180 - np.abs(hue - ink_hue),
    )
    ink_pixels = (hue_diff < ink_range) & (sat > 120)
    skin_mask[ink_pixels] = 0

    # Morphological cleanup
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_CLOSE, kernel)
    skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_OPEN, kernel)

    # Keep only components that look like fingers:
    # - Touch the image border
    # - Area between 1% and 15% of image (fingers are localized, not page-wide)
    # - Bounding box aspect ratio < 4:1 (fingers are blob-shaped, not thin
    #   horizontal lines like staff lines or thin vertical bars)
    h, w = skin_mask.shape
    img_area = h * w
    min_area = int(img_area * _FINGER_MIN_AREA_FRAC)
    max_area = int(img_area * 0.15)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(skin_mask, connectivity=8)
    result = np.zeros_like(skin_mask)

    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if area < min_area or area > max_area:
            continue

        x = stats[i, cv2.CC_STAT_LEFT]
        y = stats[i, cv2.CC_STAT_TOP]
        cw = stats[i, cv2.CC_STAT_WIDTH]
        ch = stats[i, cv2.CC_STAT_HEIGHT]

        aspect = max(cw, ch) / max(min(cw, ch), 1)
        if aspect > 4.0:
            continue

        touches_border = (x == 0 or y == 0 or x + cw >= w or y + ch >= h)
        if not touches_border:
            continue

        # A real finger is concentrated at an image edge. If the centroid
        # is deep in the image interior, it's parchment or background, not
        # skin. Require centroid within the outer 20% border zone.
        cx = x + cw / 2
        cy = y + ch / 2
        margin_x = w * 0.2
        margin_y = h * 0.2
        in_border_zone = (
            cx < margin_x or cx > w - margin_x
            or cy < margin_y or cy > h - margin_y
        )
        if in_border_zone:
            result[labels == i] = 255

    return result


def remove_fingers(
    img: np.ndarray,
    mask: np.ndarray,
    cfg: Config,
) -> np.ndarray:
    """R8: Inpaint detected finger regions.

    Args:
        img: Input BGR image.
        mask: Binary mask from detect_fingers() (255 = finger).
        cfg: Pipeline configuration.

    Returns:
        Image with finger regions inpainted.
    """
    if np.count_nonzero(mask) == 0:
        return img.copy()

    return cv2.inpaint(img, mask, _FINGER_INPAINT_RADIUS, cv2.INPAINT_TELEA)
