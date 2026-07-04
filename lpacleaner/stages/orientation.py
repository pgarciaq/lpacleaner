"""Stage 2: Orientation normalization.

Two-phase content-based orientation:
1. **Axis detection**: horizontal line counting determines whether to
   rotate 0° or 90° so staff lines run left-to-right.
2. **Polarity detection**: the vertical centroid of non-staff-line red ink
   (title text, initials) determines right-side-up vs upside-down.
   In chant books red titles appear at the top of the page.

Falls back to portrait enforcement for non-music pages (covers, blanks).
EXIF is intentionally not relied upon.

Also computes a Laplacian focus QA score per image.
Mandatory stage -- never skipped.
"""

from __future__ import annotations

import logging
from pathlib import Path

import cv2
import numpy as np

from lpacleaner.config import Config
from lpacleaner.pipeline import BaseStage
from lpacleaner.utils.line_detect import count_horizontal_lines

logger = logging.getLogger(__name__)

_FOCUS_THRESHOLD_DEFAULT = 100.0
_HORIZONTAL_LINE_MIN_COUNT = 5


class OrientationStage(BaseStage):
    name = "orientation"
    number = 2
    checkpoint_name = "02_oriented"
    error_class = "critical"

    def process_image(
        self,
        img: np.ndarray,
        metadata: dict,
        cfg: Config,
    ) -> tuple[np.ndarray, dict]:
        meta: dict = {"stage": "orientation"}

        img, rotation, method = _orient_by_content(img, cfg)

        meta["rotation_applied"] = rotation
        meta["orientation_method"] = method

        focus = _compute_focus_score(img)
        threshold = _FOCUS_THRESHOLD_DEFAULT
        meta["focus_score"] = focus
        meta["focus_threshold"] = threshold
        meta["is_blurry"] = focus < threshold

        if meta["is_blurry"]:
            logger.warning("Image is blurry (focus_score=%.1f < %.1f)", focus, threshold)

        return img, meta


def _orient_by_content(
    img: np.ndarray,
    cfg: Config,
) -> tuple[np.ndarray, int, str]:
    """Orient image so staff lines are horizontal and right-side-up.

    Phase 1 -- axis: count horizontal line segments at 0° and 90° CCW,
    pick whichever has more (with a 2:1 confidence ratio).

    Phase 2 -- polarity: after making staff lines horizontal, detect
    non-staff-line red ink (titles, initials). If its vertical centroid
    is in the lower half, the image is upside-down → rotate 180°.

    Returns (oriented_image, total_degrees_applied, method_name).
    """
    h, w = img.shape[:2]
    max_dim = max(h, w)
    if max_dim > 1200:
        scale = 1200.0 / max_dim
        small = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    else:
        small = img

    h_lines_0 = count_horizontal_lines(small)
    h_lines_90 = count_horizontal_lines(
        cv2.rotate(small, cv2.ROTATE_90_COUNTERCLOCKWISE)
    )

    logger.debug(
        "Orientation axis: h_lines(0°)=%d, h_lines(90°)=%d",
        h_lines_0, h_lines_90,
    )

    max_lines = max(h_lines_0, h_lines_90)
    min_lines = max(min(h_lines_0, h_lines_90), 1)
    ratio = max_lines / min_lines

    if max_lines >= _HORIZONTAL_LINE_MIN_COUNT and ratio >= 2.0:
        if h_lines_90 > h_lines_0:
            img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
            base_rotation = 90
        else:
            base_rotation = 0

        # Phase 2: check if upside-down using red title position
        flipped, did_flip = _correct_polarity(img, cfg)
        total = (base_rotation + (180 if did_flip else 0)) % 360
        method = "staff_lines" + ("+title_flip" if did_flip else "")
        return flipped, total, method

    # Fallback: no confident staff line signal (cover, blank, text page).
    if w > h:
        logger.info("No staff lines detected; falling back to portrait enforcement")
        img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
        base_rotation = 90
    else:
        base_rotation = 0

    # Still try polarity on the fallback orientation
    flipped, did_flip = _correct_polarity(img, cfg)
    total = (base_rotation + (180 if did_flip else 0)) % 360
    method = "portrait_fallback" + ("+title_flip" if did_flip else "")
    return flipped, total, method


def _correct_polarity(img: np.ndarray, cfg: Config) -> tuple[np.ndarray, bool]:
    """Check if the image is upside-down using the red title position.

    Chant book pages have red title text at the top. After removing
    horizontal staff lines from the red ink mask, the remaining red
    (titles, initials, rubrication) should have its vertical centroid
    in the upper half. If it's in the lower half, rotate 180°.

    Returns (image, did_flip).
    """
    oh, ow = img.shape[:2]

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hue = hsv[:, :, 0].astype(np.int16)
    sat = hsv[:, :, 1]

    ink_hue = cfg.staff_color_hue
    ink_range = cfg.staff_color_range
    hue_diff = np.minimum(
        np.abs(hue - ink_hue),
        180 - np.abs(hue - ink_hue),
    )
    red_mask = ((hue_diff < ink_range) & (sat > 120)).astype(np.uint8) * 255

    # Remove horizontal staff line structures so only titles/initials remain
    horiz_kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT, (max(ow // 20, 30), 1)
    )
    staff_only = cv2.morphologyEx(red_mask, cv2.MORPH_OPEN, horiz_kernel)
    non_staff_red = cv2.subtract(red_mask, staff_only)

    total_red = np.count_nonzero(non_staff_red)
    if total_red < 100:
        logger.debug("Polarity: insufficient non-staff red pixels (%d)", total_red)
        return img, False

    centroid_y = float(np.mean(np.where(non_staff_red > 0)[0])) / oh

    logger.debug(
        "Polarity: centroid_y=%.3f, non_staff_red=%d", centroid_y, total_red,
    )

    if centroid_y > 0.5:
        logger.info(
            "Red title centroid in lower half (%.3f) → rotating 180°", centroid_y
        )
        return cv2.rotate(img, cv2.ROTATE_180), True

    return img, False


def _compute_focus_score(img: np.ndarray) -> float:
    """Compute Laplacian variance on the central 80% of the image.

    Avoids edges where blur is expected from depth of field.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    h, w = gray.shape[:2]
    margin_y = h // 10
    margin_x = w // 10
    center = gray[margin_y:h - margin_y, margin_x:w - margin_x]
    laplacian = cv2.Laplacian(center, cv2.CV_64F)
    return float(laplacian.var())
