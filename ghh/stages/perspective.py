"""Stage 5: Perspective correction.

Uses the quad corners detected by Stage 4 to map the page to a
rectangle via ``cv2.warpPerspective``.  This is where the actual
crop happens -- Stage 4 only detects the quad and passes the full
image through.

Target rectangle dimensions use the *maximum* of opposite edge pairs
so that no content is lost from the longer edge::

    width  = max(dist(TL,TR), dist(BL,BR))
    height = max(dist(TL,BL), dist(TR,BR))

Out-of-bounds pixels are filled with the estimated page background
color (median of border pixels), not black.  This prevents downstream
stages (enhance, normalize) from being confused by black corners.

If no quad corners are found in the incoming metadata (e.g. Stage 5
run in isolation without Stage 4), the image passes through unchanged.
"""

from __future__ import annotations

import logging

import cv2
import numpy as np

from ghh.config import Config
from ghh.pipeline import BaseStage
from ghh.utils.geometry import (
    compute_target_size,
    get_perspective_transform,
    order_corners,
)
from ghh.utils.image_utils import estimate_background

logger = logging.getLogger(__name__)

_MIN_SIDE_PX = 10


class PerspectiveStage(BaseStage):
    name = "perspective"
    number = 5
    checkpoint_name = "05_perspective"
    error_class = "skippable"

    def process_image(
        self,
        img: np.ndarray,
        metadata: dict,
        cfg: Config,
    ) -> tuple[np.ndarray, dict]:
        quad = _load_quad(metadata)

        if quad is None:
            logger.warning("No quad corners in metadata; passing through unchanged")
            pt = {"stage": "perspective", "method": "passthrough"}
            if "page_type" in metadata:
                pt["page_type"] = metadata["page_type"]
            return img, pt

        quad = order_corners(quad)
        width, height = compute_target_size(quad)

        if width < _MIN_SIDE_PX or height < _MIN_SIDE_PX:
            logger.warning(
                "Degenerate quad (w=%d h=%d); passing through unchanged",
                width, height,
            )
            pt = {"stage": "perspective", "method": "passthrough"}
            if "page_type" in metadata:
                pt["page_type"] = metadata["page_type"]
            return img, pt

        dst = np.array(
            [[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]],
            dtype=np.float32,
        )

        M = get_perspective_transform(quad, dst)
        bg_color = estimate_background(img)

        rectified = cv2.warpPerspective(
            img, M, (width, height),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=bg_color,
        )

        meta = {
            "stage": "perspective",
            "method": "warpPerspective",
            "src_quad": quad.tolist(),
            "dst_size": [width, height],
            "background_color": list(bg_color),
        }
        if "page_type" in metadata:
            meta["page_type"] = metadata["page_type"]

        logger.info(
            "Perspective correction: %dx%d -> %dx%d",
            img.shape[1], img.shape[0], width, height,
        )
        return rectified, meta


def _load_quad(metadata: dict) -> np.ndarray | None:
    """Extract quad corners from Stage 4 metadata sidecar."""
    corners = metadata.get("quad_corners")
    if corners is None:
        return None
    arr = np.array(corners, dtype=np.float32)
    if arr.shape != (4, 2):
        logger.warning("Invalid quad_corners shape: %s", arr.shape)
        return None
    return arr
