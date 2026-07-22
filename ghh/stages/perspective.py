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
_BOUNDARY_MARGIN = 5


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
        h, w = img.shape[:2]

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

        boundary_count = _count_boundary_corners(quad, h, w)
        skew = _quad_skew_degrees(quad)
        crop = _crop_ratio(quad, h, w)

        skip_reason = _should_skip_warp(
            boundary_count, skew, crop,
            cfg.perspective_max_skew_deg,
            cfg.perspective_max_crop_frac,
        )

        if skip_reason:
            logger.info(
                "Skipping perspective: %s "
                "(boundary_corners=%d, skew=%.1f°, crop=%.1f%%)",
                skip_reason, boundary_count, skew, crop * 100,
            )
            meta = {
                "stage": "perspective",
                "method": "passthrough_unreliable",
                "skip_reason": skip_reason,
                "boundary_corners": boundary_count,
                "skew_degrees": round(skew, 2),
                "crop_fraction": round(crop, 3),
                "src_quad": quad.tolist(),
            }
            if "page_type" in metadata:
                meta["page_type"] = metadata["page_type"]
            return img, meta

        # Apply padding to prevent edge content clipping
        pad_x = int(width * cfg.perspective_output_padding_frac)
        pad_y = int(height * cfg.perspective_output_padding_frac)
        dst = np.array(
            [[pad_x, pad_y], [width - 1 - pad_x, pad_y],
             [width - 1 - pad_x, height - 1 - pad_y], [pad_x, height - 1 - pad_y]],
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
            "boundary_corners": boundary_count,
            "skew_degrees": round(skew, 2),
            "crop_fraction": round(crop, 3),
        }
        if "page_type" in metadata:
            meta["page_type"] = metadata["page_type"]

        logger.info(
            "Perspective correction: %dx%d -> %dx%d "
            "(boundary=%d, skew=%.1f°, crop=%.1f%%)",
            img.shape[1], img.shape[0], width, height,
            boundary_count, skew, crop * 100,
        )
        return rectified, meta


# ---------------------------------------------------------------------------
# Quad validation helpers
# ---------------------------------------------------------------------------

def _count_boundary_corners(
    quad: np.ndarray, img_h: int, img_w: int,
) -> int:
    """Count how many quad corners sit on or near the image boundary."""
    count = 0
    for x, y in quad:
        if (x <= _BOUNDARY_MARGIN or x >= img_w - 1 - _BOUNDARY_MARGIN
                or y <= _BOUNDARY_MARGIN or y >= img_h - 1 - _BOUNDARY_MARGIN):
            count += 1
    return count


def _quad_skew_degrees(quad: np.ndarray) -> float:
    """Max absolute angle of top/bottom edges relative to horizontal."""
    tl, tr, br, bl = quad
    top_angle = abs(np.degrees(np.arctan2(tr[1] - tl[1], tr[0] - tl[0])))
    bottom_angle = abs(np.degrees(np.arctan2(br[1] - bl[1], br[0] - bl[0])))
    return max(top_angle, bottom_angle)


def _crop_ratio(quad: np.ndarray, img_h: int, img_w: int) -> float:
    """Fraction of image area that falls outside the quad."""
    quad_area = cv2.contourArea(quad.astype(np.float32))
    img_area = img_h * img_w
    if img_area == 0:
        return 0.0
    return max(0.0, 1.0 - quad_area / img_area)


def _should_skip_warp(
    boundary_corners: int,
    skew_deg: float,
    crop_frac: float,
    max_skew: float,
    max_crop: float,
) -> str | None:
    """Return a skip reason string, or None if the warp should proceed.

    Having 3-4 boundary corners is normal when the page fills the photo
    frame.  The real signal for an unreliable quad is excessive skew
    (bad edge detection) or excessive crop (quad much smaller than the
    image, suggesting a false detection).
    """
    if skew_deg > max_skew:
        return "excessive_skew"
    if crop_frac > max_crop:
        return "excessive_crop"
    return None


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
