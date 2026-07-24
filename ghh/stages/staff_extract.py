"""Stage 7: Staff Extract (Score branch only).

Isolates music staff regions from mixed-content pages by detecting
horizontal staff line clusters and cropping to their bounding box.
Pages without detected staves pass through unchanged.

Detection uses a two-path strategy:

1. **Color path** (primary): ``detect_ink_mask()`` isolates ink pixels
   by HSV color, then morphological opening keeps only horizontal
   line structures.  Works for any ink color (red, brown, sepia, …).
2. **Grayscale fallback**: Otsu binarization when the color mask is
   empty (e.g. black ink on white paper where HSV thresholds miss).
"""

from __future__ import annotations

import logging

import cv2
import numpy as np

from ghh.config import Config
from ghh.pipeline import BaseStage
from ghh.utils.line_detect import detect_ink_mask

logger = logging.getLogger(__name__)


class StaffExtractStage(BaseStage):
    name = "staff_extract"
    number = 7
    checkpoint_name = "07_staff_extract"
    error_class = "skippable"
    config_keys = (
        "staff_color_hue",
        "staff_color_range",
        "staff_saturation_min",
        "staff_value_min",
    )
    symlink_unchanged = True

    def process_image(
        self,
        img: np.ndarray,
        metadata: dict,
        cfg: Config,
    ) -> tuple[np.ndarray, dict]:
        h, w = img.shape[:2]

        morph, method = _detect_horizontal_lines(img, h, w, cfg)

        contours, _ = cv2.findContours(
            morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE,
        )

        if not contours:
            metadata["staff_extract_action"] = "passthrough"
            metadata["staff_extract_reason"] = "no_horizontal_lines"
            metadata["staff_extract_method"] = method
            return img, metadata

        min_width = int(w * 0.3)
        staff_contours = [
            cv2.boundingRect(c) for c in contours
            if cv2.boundingRect(c)[2] >= min_width
        ]

        if not staff_contours:
            metadata["staff_extract_action"] = "passthrough"
            metadata["staff_extract_reason"] = "no_wide_lines"
            metadata["staff_extract_method"] = method
            return img, metadata

        min_y = min(y for _, y, _, _ in staff_contours)
        max_y = max(y + ch for _, y, _, ch in staff_contours)

        margin = int(h * 0.03)
        y_start = max(0, min_y - margin)
        y_end = min(h, max_y + margin)

        if (y_end - y_start) < h * 0.5:
            metadata["staff_extract_action"] = "passthrough"
            metadata["staff_extract_reason"] = "staff_region_too_small"
            metadata["staff_extract_coverage"] = round((y_end - y_start) / h, 3)
            metadata["staff_extract_method"] = method
            return img, metadata

        cropped = img[y_start:y_end, 0:w]
        metadata["staff_extract_action"] = "cropped"
        metadata["staff_extract_method"] = method
        metadata["staff_extract_y_start"] = y_start
        metadata["staff_extract_y_end"] = y_end
        metadata["staff_extract_lines_found"] = len(staff_contours)
        metadata["staff_extract_coverage"] = round((y_end - y_start) / h, 3)

        return cropped, metadata

    def is_unchanged(self, metadata: dict) -> bool:
        return metadata.get("staff_extract_action") == "passthrough"


def _detect_horizontal_lines(
    img: np.ndarray,
    h: int,
    w: int,
    cfg: Config,
) -> tuple[np.ndarray, str]:
    """Return (binary_mask, method) with horizontal line structures.

    Tries the color-aware ink mask first; falls back to grayscale Otsu
    if the color path yields no contours.
    """
    horiz_kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT, (max(w // 8, 50), 1),
    )

    if img.ndim == 3:
        ink_mask = detect_ink_mask(img, cfg)
        if np.count_nonzero(ink_mask) > 0:
            morph = cv2.morphologyEx(ink_mask, cv2.MORPH_OPEN, horiz_kernel)
            if np.count_nonzero(morph) > 0:
                return morph, "color"

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    binary = cv2.threshold(
        gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU,
    )[1]
    morph = cv2.morphologyEx(binary, cv2.MORPH_OPEN, horiz_kernel)
    return morph, "grayscale"
