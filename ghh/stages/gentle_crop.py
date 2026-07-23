"""Stage 5: Gentle Crop.

Computes the axis-aligned bounding box of the quad corners detected by
Stage 4, expands it by a configurable margin fraction, clamps to image
bounds, and crops.  No perspective warp is applied -- that is deferred
to Stage 9 (after Deskew) where geometric corrections benefit from a
straighter image.

The transformed ``quad_corners`` (shifted by the crop offset) are
written to the output metadata so downstream stages can still use
them for perspective correction, ROI masking, etc.
"""

from __future__ import annotations

import logging

import numpy as np

from ghh.config import Config
from ghh.pipeline import BaseStage

logger = logging.getLogger(__name__)


class GentleCropStage(BaseStage):
    name = "gentle_crop"
    number = 5
    checkpoint_name = "05_gentle_crop"
    error_class = "skippable"
    config_keys = ("gentle_crop_margin_frac",)

    def process_image(
        self,
        img: np.ndarray,
        metadata: dict,
        cfg: Config,
    ) -> tuple[np.ndarray, dict]:
        quad = _load_quad(metadata)

        if quad is None:
            logger.warning("No quad corners in metadata; passing through unchanged")
            meta: dict = {"stage": "gentle_crop", "method": "passthrough"}
            if "page_type" in metadata:
                meta["page_type"] = metadata["page_type"]
            return img, meta

        h, w = img.shape[:2]

        x_min, y_min = quad.min(axis=0)
        x_max, y_max = quad.max(axis=0)

        bbox_w = x_max - x_min
        bbox_h = y_max - y_min
        margin_frac = cfg.gentle_crop_margin_frac
        margin_x = int(bbox_w * margin_frac)
        margin_y = int(bbox_h * margin_frac)

        x0 = max(0, int(x_min) - margin_x)
        y0 = max(0, int(y_min) - margin_y)
        x1 = min(w, int(x_max) + margin_x)
        y1 = min(h, int(y_max) + margin_y)

        if x1 <= x0 or y1 <= y0:
            logger.warning("Degenerate crop box; passing through unchanged")
            meta = {"stage": "gentle_crop", "method": "passthrough"}
            if "page_type" in metadata:
                meta["page_type"] = metadata["page_type"]
            return img, meta

        cropped = img[y0:y1, x0:x1]

        new_quad = quad - np.array([x0, y0], dtype=np.float32)

        out_meta: dict = {
            "stage": "gentle_crop",
            "method": "bbox_crop",
            "crop_box": [x0, y0, x1, y1],
            "margin_frac": margin_frac,
            "quad_corners": new_quad.tolist(),
        }
        if "page_type" in metadata:
            out_meta["page_type"] = metadata["page_type"]

        logger.info(
            "Gentle crop: %dx%d -> %dx%d (margin=%.1f%%)",
            w, h, x1 - x0, y1 - y0, margin_frac * 100,
        )
        return cropped, out_meta


def _load_quad(metadata: dict) -> np.ndarray | None:
    """Extract quad corners from Stage 4 metadata."""
    corners = metadata.get("quad_corners")
    if corners is None:
        return None
    arr = np.array(corners, dtype=np.float32)
    if arr.shape != (4, 2):
        logger.warning("Invalid quad_corners shape: %s", arr.shape)
        return None
    return arr
