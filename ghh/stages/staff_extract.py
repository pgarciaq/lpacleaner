"""Stage 7: Staff Extract (Score branch only).

Isolates music staff regions from mixed-content pages by detecting
horizontal staff line clusters and cropping to their bounding box.
Pages without detected staves pass through unchanged.
"""

from __future__ import annotations

import logging

import cv2
import numpy as np

from ghh.config import Config
from ghh.pipeline import BaseStage

logger = logging.getLogger(__name__)


class StaffExtractStage(BaseStage):
    name = "staff_extract"
    number = 7
    checkpoint_name = "07_staff_extract"
    error_class = "skippable"
    config_keys = ()
    symlink_unchanged = True

    def process_image(
        self,
        img: np.ndarray,
        metadata: dict,
        cfg: Config,
    ) -> tuple[np.ndarray, dict]:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
        h, w = gray.shape[:2]

        # Detect horizontal lines using morphological operations
        horiz_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (max(w // 8, 50), 1))
        morph = cv2.morphologyEx(
            cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1],
            cv2.MORPH_OPEN,
            horiz_kernel,
        )

        contours, _ = cv2.findContours(morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            metadata["staff_extract_action"] = "passthrough"
            metadata["staff_extract_reason"] = "no_horizontal_lines"
            return img, metadata

        # Filter to contours that span a reasonable width (at least 30% of image)
        min_width = int(w * 0.3)
        staff_contours = []
        for c in contours:
            x, y, cw, ch = cv2.boundingRect(c)
            if cw >= min_width:
                staff_contours.append((x, y, cw, ch))

        if not staff_contours:
            metadata["staff_extract_action"] = "passthrough"
            metadata["staff_extract_reason"] = "no_wide_lines"
            return img, metadata

        # Compute bounding box of all staff regions with margin
        min_y = min(y for _, y, _, _ in staff_contours)
        max_y = max(y + ch for _, y, _, ch in staff_contours)

        margin = int(h * 0.03)  # 3% vertical margin
        y_start = max(0, min_y - margin)
        y_end = min(h, max_y + margin)

        # Only crop if we're actually removing significant content
        if (y_end - y_start) < h * 0.5:
            metadata["staff_extract_action"] = "passthrough"
            metadata["staff_extract_reason"] = "staff_region_too_small"
            metadata["staff_extract_coverage"] = round((y_end - y_start) / h, 3)
            return img, metadata

        cropped = img[y_start:y_end, 0:w]
        metadata["staff_extract_action"] = "cropped"
        metadata["staff_extract_y_start"] = y_start
        metadata["staff_extract_y_end"] = y_end
        metadata["staff_extract_lines_found"] = len(staff_contours)
        metadata["staff_extract_coverage"] = round((y_end - y_start) / h, 3)

        return cropped, metadata

    def is_unchanged(self, metadata: dict) -> bool:
        return metadata.get("staff_extract_action") == "passthrough"
