"""Stage 2: Orientation normalization.

Two-phase content-based orientation:
1. **Axis detection**: horizontal line counting determines whether to
   rotate 0° or 90° so staff lines run left-to-right.  A staff-area
   validation rejects textured surfaces (e.g. rusty book covers) that
   generate many false horizontal lines.
2. **Polarity detection**: compares title-eligible red ink in the top
   vs bottom edges of the page.  Title-eligible rows are those with
   significant red coverage and very little dark/black text in the
   central area (titles are pure red lines; body rubrics are always
   mixed with dark text).  Only the outer 15% edges are compared,
   ignoring body content in the middle 70%.
3. **Spine fallback**: when no red title signal is available (covers,
   blanks), compares left/right edge saturation-to-brightness ratios.
   The more-saturated, darker edge is assumed to be the spine and is
   placed on the left (standard Western book orientation).

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
_STAFF_AREA_MAX = 0.05


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
        # Validate that these are real staff lines, not texture.
        # On textured surfaces (rusty covers), large areas of red
        # survive the horizontal morphological opening because the
        # patches are wide, not thin like actual staff lines.
        if _has_real_staff_lines(img, cfg):
            if h_lines_90 > h_lines_0:
                img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
                base_rotation = 90
            else:
                base_rotation = 0

            flipped, did_flip = _correct_polarity(img, cfg)
            total = (base_rotation + (180 if did_flip else 0)) % 360
            method = "staff_lines" + ("+polarity_flip" if did_flip else "")
            return flipped, total, method
        else:
            logger.info(
                "Horizontal lines detected but staff-area too large "
                "(textured surface) → falling back to portrait"
            )

    # Fallback: no confident staff line signal (cover, blank, text page).
    if w > h:
        logger.info("No staff lines detected; falling back to portrait enforcement")
        img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
        base_rotation = 90
    else:
        base_rotation = 0

    flipped, did_flip = _correct_polarity(img, cfg)
    total = (base_rotation + (180 if did_flip else 0)) % 360
    method = "portrait_fallback" + ("+polarity_flip" if did_flip else "")
    return flipped, total, method


def _correct_polarity(img: np.ndarray, cfg: Config) -> tuple[np.ndarray, bool]:
    """Check if the image is upside-down using weighted edge title detection.

    Chant book pages have a red title line near the top of the page.
    Titles are close to the page edge with very little blank space.
    Body rubrics and section headers appear deeper inside the page.

    Algorithm:
    1. Build a non-staff red mask and a dark ink mask.
    2. For each row, classify it as "title-eligible" when it has
       significant red coverage (>3%) and little dark text (<2.5%)
       in the central 80% of the row.
    3. Keep only red pixels on title-eligible rows.
    4. Compare weighted scores in the top 10% vs bottom 10% of the
       image.  Pixels closer to the physical edge receive higher
       weight (linear decay), favoring actual titles over section
       headers deeper in the page body.
    5. If the bottom edge scores higher → upside-down → rotate 180°.

    Returns (image, did_flip).
    """
    oh, ow = img.shape[:2]

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hue = hsv[:, :, 0].astype(np.int16)
    sat = hsv[:, :, 1]
    val = hsv[:, :, 2]

    ink_hue = cfg.staff_color_hue
    ink_range = cfg.staff_color_range
    hue_diff = np.minimum(
        np.abs(hue - ink_hue),
        180 - np.abs(hue - ink_hue),
    )
    red_mask = ((hue_diff < ink_range) & (sat > 120)).astype(np.uint8) * 255

    horiz_kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT, (max(ow // 20, 30), 1)
    )
    staff_only = cv2.morphologyEx(red_mask, cv2.MORPH_OPEN, horiz_kernel)
    non_staff_red = cv2.subtract(red_mask, staff_only)

    dark_mask = (val < 80).astype(np.uint8)

    margin = int(ow * 0.10)
    central_width = max(ow - 2 * margin, 1)
    central_dark_per_row = (
        np.sum(dark_mask[:, margin : ow - margin], axis=1) / central_width
    )

    red_per_row = np.sum(non_staff_red > 0, axis=1) / ow

    exclude = (red_per_row < 0.03) | (central_dark_per_row > 0.025)
    title_red = non_staff_red.copy()
    title_red[exclude] = 0

    edge_frac = 0.10
    top_cut = int(oh * edge_frac)
    bot_cut = int(oh * (1.0 - edge_frac))

    top_weights = np.linspace(1.0, 0.0, top_cut).reshape(-1, 1)
    bot_weights = np.linspace(0.0, 1.0, oh - bot_cut).reshape(-1, 1)

    top_score = float(np.sum((title_red[:top_cut] > 0) * top_weights))
    bot_score = float(np.sum((title_red[bot_cut:] > 0) * bot_weights))
    total_score = top_score + bot_score

    logger.debug(
        "Polarity: title_score top=%.0f bot=%.0f (edge=%.0f%%)",
        top_score,
        bot_score,
        edge_frac * 100,
    )

    if total_score < 50:
        logger.debug("Polarity: insufficient title signal (%.0f)", total_score)
        return _detect_spine_polarity(img)

    if bot_score > top_score:
        logger.info(
            "Title signal at bottom edge (%.0f) > top (%.0f) → rotating 180°",
            bot_score,
            top_score,
        )
        return cv2.rotate(img, cv2.ROTATE_180), True

    return img, False


def _has_real_staff_lines(img: np.ndarray, cfg: Config) -> bool:
    """Check whether detected horizontal lines are real staff lines.

    Textured surfaces (e.g. rusty book covers) generate many false
    horizontal lines in HoughLinesP.  Real staff lines are *thin*
    horizontal red structures that occupy a tiny fraction of the image
    area.  On a textured surface the red that survives a horizontal
    morphological opening is *wide* (patches, not lines) and covers
    a much larger area fraction.

    Returns True if the staff-like red area is small enough to be
    credible staff lines (< ``_STAFF_AREA_MAX``).
    """
    oh, ow = img.shape[:2]

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hue = hsv[:, :, 0].astype(np.int16)
    sat = hsv[:, :, 1]

    hue_diff = np.minimum(
        np.abs(hue - cfg.staff_color_hue),
        180 - np.abs(hue - cfg.staff_color_hue),
    )
    red_mask = ((hue_diff < cfg.staff_color_range) & (sat > 120)).astype(np.uint8) * 255

    horiz_kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT, (max(ow // 20, 30), 1)
    )
    staff_only = cv2.morphologyEx(red_mask, cv2.MORPH_OPEN, horiz_kernel)

    staff_area = np.count_nonzero(staff_only) / (oh * ow)
    logger.debug("Staff-area ratio: %.4f (threshold %.4f)", staff_area, _STAFF_AREA_MAX)
    return bool(staff_area <= _STAFF_AREA_MAX)


def _detect_spine_polarity(img: np.ndarray) -> tuple[np.ndarray, bool]:
    """Fallback polarity detection using spine location.

    When no red title signal is available (covers, blanks), try to
    find the book spine by comparing the saturation-to-brightness
    ratio of the left and right edges.  The spine is typically the
    most worn / oxidized edge, appearing darker and more saturated.

    Western book convention: spine on the left when viewing the
    front cover.  If the spine appears to be on the right, rotate
    180° to correct.

    Requires a clear asymmetry (ratio difference > 10%) to act;
    otherwise returns the image unchanged.

    Returns (image, did_flip).
    """
    oh, ow = img.shape[:2]
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    margin = int(ow * 0.15)
    if margin < 10:
        return img, False

    left_hsv = hsv[:, :margin]
    right_hsv = hsv[:, ow - margin :]

    def _sv_ratio(band: np.ndarray) -> float:
        s = float(np.mean(band[:, :, 1]))
        v = max(float(np.mean(band[:, :, 2])), 1.0)
        return s / v

    left_sv = _sv_ratio(left_hsv)
    right_sv = _sv_ratio(right_hsv)

    logger.debug(
        "Spine detection: left S/V=%.3f, right S/V=%.3f",
        left_sv,
        right_sv,
    )

    # Require at least 10% relative difference to be confident
    max_sv = max(left_sv, right_sv)
    if max_sv < 0.01:
        return img, False
    diff_pct = abs(right_sv - left_sv) / max_sv
    if diff_pct < 0.10:
        logger.debug("Spine S/V difference too small (%.0f%%), no flip", diff_pct * 100)
        return img, False

    if right_sv > left_sv:
        logger.info(
            "Spine on right (S/V=%.3f) > left (%.3f) → rotating 180°",
            right_sv,
            left_sv,
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
