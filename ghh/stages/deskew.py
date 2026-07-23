"""Stage 8: Deskew.

Corrects the small residual skew angle (typically 0-3 degrees) left
after Stage 2's coarse orientation correction.

Two detection methods:

1. **Staff-line angle** (music pages): median angle of HoughLinesP
   segments from ``detect_dominant_angle()``.  Fast and accurate when
   staff lines are present.

2. **Projection profile** (text-only / blank pages): coarse-to-fine
   search over candidate angles.  The image is binarized, downscaled
   to 25%, and rotated at each candidate angle; the angle that
   maximises the variance of horizontal row sums wins.

After rotation, ``trim_to_content()`` cleans up the background-filled
corners introduced by the affine transform.
"""

from __future__ import annotations

import logging

import cv2
import numpy as np

from ghh.config import Config
from ghh.pipeline import BaseStage
from ghh.utils.image_utils import estimate_background, trim_to_content
from ghh.utils.line_detect import detect_dominant_angle

logger = logging.getLogger(__name__)


class DeskewStage(BaseStage):
    name = "deskew"
    number = 8
    checkpoint_name = "08_deskewed"
    error_class = "skippable"

    def process_image(
        self,
        img: np.ndarray,
        metadata: dict,
        cfg: Config,
    ) -> tuple[np.ndarray, dict]:
        small = _downscale_for_detection(img)
        angle = detect_dominant_angle(small, cfg)
        method = "staff_lines"

        if angle == 0.0:
            angle = _projection_profile_angle(img, cfg)
            method = "projection_profile"

        if abs(angle) > cfg.deskew_max_angle:
            logger.warning(
                "Skew angle %.2f exceeds max %.1f, clamping "
                "-- check Stage 2 orientation",
                angle,
                cfg.deskew_max_angle,
            )
            angle = max(-cfg.deskew_max_angle, min(cfg.deskew_max_angle, angle))

        if abs(angle) < cfg.deskew_skip_threshold:
            logger.debug("Skew angle %.3f below threshold, skipping rotation", angle)
            result = img
            method = "skipped"
        else:
            bg_color = estimate_background(img)
            result = _rotate(img, -angle, bg_color)

        result, _trim_x, _trim_y = trim_to_content(result)

        meta = {
            "stage": "deskew",
            "method": method,
            "skew_angle": round(angle, 4),
        }
        if "page_type" in metadata:
            meta["page_type"] = metadata["page_type"]

        return result, meta


_DETECT_MAX_DIM = 1500


def _downscale_for_detection(img: np.ndarray) -> np.ndarray:
    """Downscale image so its longest side is at most _DETECT_MAX_DIM.

    Used to speed up staff-line detection (HoughLinesP) which does
    not need full-resolution accuracy for angle estimation.
    """
    h, w = img.shape[:2]
    longest = max(h, w)
    if longest <= _DETECT_MAX_DIM:
        return img
    scale = _DETECT_MAX_DIM / longest
    return cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)


def _rotate(
    img: np.ndarray,
    angle_deg: float,
    bg_color: tuple[int, ...],
) -> np.ndarray:
    """Rotate image by *angle_deg* around its center, filling with bg_color."""
    h, w = img.shape[:2]
    cx, cy = w / 2, h / 2
    M = cv2.getRotationMatrix2D((cx, cy), angle_deg, 1.0)
    return cv2.warpAffine(
        img, M, (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=bg_color,
    )


def _projection_profile_angle(img: np.ndarray, cfg: Config) -> float:
    """Estimate skew angle via projection profile (coarse-to-fine).

    Used for text-only pages where no staff lines are detected.
    The image is binarized and downscaled to 25% for speed.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    scale = 0.25
    small = cv2.resize(binary, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)

    max_angle = cfg.deskew_max_angle
    step = cfg.deskew_angle_step

    coarse_step = 1.0
    coarse_angles = np.arange(-max_angle, max_angle + coarse_step / 2, coarse_step)
    best_angle = _best_profile_angle(small, coarse_angles)

    fine_angles = np.arange(
        best_angle - coarse_step,
        best_angle + coarse_step + step / 2,
        step,
    )
    fine_angles = fine_angles[
        (fine_angles >= -max_angle) & (fine_angles <= max_angle)
    ]
    best_angle = _best_profile_angle(small, fine_angles)

    # best_angle is the rotation that straightens the image (correction angle);
    # negate to return the skew angle (how much the image is actually skewed).
    return -float(best_angle)


def _best_profile_angle(binary: np.ndarray, angles: np.ndarray) -> float:
    """Return the angle from *angles* that maximises row-sum variance."""
    h, w = binary.shape[:2]
    cx, cy = w / 2, h / 2
    best_var = -1.0
    best_angle = 0.0

    for angle in angles:
        M = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)
        rotated = cv2.warpAffine(
            binary, M, (w, h),
            flags=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )
        row_sums = np.sum(rotated, axis=1, dtype=np.float64)
        var = np.var(row_sums)
        if var > best_var:
            best_var = var
            best_angle = angle

    return float(best_angle)
