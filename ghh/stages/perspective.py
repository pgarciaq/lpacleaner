"""Stage 9: Perspective correction.

Uses the quad corners (propagated from Stage 4 through Gentle Crop
and Deskew) to map the page to a rectangle via ``cv2.warpPerspective``.

Target rectangle dimensions use the *maximum* of opposite edge pairs
so that no content is lost from the longer edge::

    width  = max(dist(TL,TR), dist(BL,BR))
    height = max(dist(TL,BL), dist(TR,BR))

Out-of-bounds pixels are filled with the estimated page background
color (median of border pixels), not black.

Three safety checks prevent the warp from making things worse:

1. **Unreliable quad**: excessive skew or crop ratio → passthrough
2. **Near-rectangular quad**: all interior angles within threshold of
   90° → passthrough (warp would only introduce noise)
3. **Tilt introduction**: if the homography would introduce more tilt
   than the original quad had → passthrough

If no quad corners are found in the incoming metadata, the image
passes through unchanged (symlinked when possible).
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
    number = 9
    checkpoint_name = "09_perspective"
    error_class = "skippable"
    config_keys = (
        "perspective_max_skew_deg",
        "perspective_max_crop_frac",
        "perspective_near_rect_threshold_deg",
        "perspective_max_introduced_tilt_deg",
    )
    symlink_unchanged = True

    def is_unchanged(self, metadata: dict) -> bool:
        return "passthrough" in metadata.get("method", "")

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
        max_angle_dev = _max_angle_deviation(quad)

        # --- Check 1: unreliable quad (excessive skew or crop) ---
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
            return img, _passthrough_meta(
                quad, metadata, "passthrough_unreliable",
                skip_reason=skip_reason,
                boundary_corners=boundary_count,
                skew_degrees=round(skew, 2),
                crop_fraction=round(crop, 3),
                max_angle_deviation=round(max_angle_dev, 2),
            )

        # --- Check 2: quad is already near-rectangular ---
        near_rect_thresh = cfg.perspective_near_rect_threshold_deg
        if max_angle_dev < near_rect_thresh:
            logger.info(
                "Skipping perspective: near_rectangular "
                "(max_angle_dev=%.2f° < threshold=%.1f°)",
                max_angle_dev, near_rect_thresh,
            )
            return img, _passthrough_meta(
                quad, metadata, "passthrough_near_rectangular",
                skip_reason="near_rectangular",
                boundary_corners=boundary_count,
                skew_degrees=round(skew, 2),
                crop_fraction=round(crop, 3),
                max_angle_deviation=round(max_angle_dev, 2),
            )

        # --- Map quad to tight rectangle ---
        dst = np.array(
            [[0, 0], [width - 1, 0],
             [width - 1, height - 1], [0, height - 1]],
            dtype=np.float32,
        )

        M = get_perspective_transform(quad, dst)

        # --- Check 3: reject if homography introduces excessive tilt ---
        introduced_tilt = _homography_tilt_degrees(M)
        max_tilt = cfg.perspective_max_introduced_tilt_deg
        original_tilt = _quad_tilt_degrees(quad)
        net_tilt = abs(introduced_tilt) - abs(original_tilt)
        if net_tilt > max_tilt:
            logger.info(
                "Skipping perspective: introduced_tilt "
                "(homography=%.2f°, original=%.2f°, net=%.2f° > max=%.1f°)",
                introduced_tilt, original_tilt, net_tilt, max_tilt,
            )
            return img, _passthrough_meta(
                quad, metadata, "passthrough_tilt_introduced",
                skip_reason="introduced_tilt",
                boundary_corners=boundary_count,
                skew_degrees=round(skew, 2),
                crop_fraction=round(crop, 3),
                max_angle_deviation=round(max_angle_dev, 2),
                homography_tilt=round(introduced_tilt, 2),
                original_tilt=round(original_tilt, 2),
            )

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
            "max_angle_deviation": round(max_angle_dev, 2),
            "homography_tilt": round(introduced_tilt, 2),
        }
        if "page_type" in metadata:
            meta["page_type"] = metadata["page_type"]

        logger.info(
            "Perspective correction: %dx%d -> %dx%d "
            "(boundary=%d, skew=%.1f°, angle_dev=%.1f°, tilt=%.2f°)",
            w, h, width, height,
            boundary_count, skew, max_angle_dev, introduced_tilt,
        )
        return rectified, meta


# ---------------------------------------------------------------------------
# Metadata helpers
# ---------------------------------------------------------------------------

def _passthrough_meta(
    quad: np.ndarray,
    metadata: dict,
    method: str,
    **extra: object,
) -> dict:
    """Build a passthrough metadata dict with common fields."""
    meta: dict = {
        "stage": "perspective",
        "method": method,
        "src_quad": quad.tolist(),
        **extra,
    }
    if "page_type" in metadata:
        meta["page_type"] = metadata["page_type"]
    return meta


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


def _quad_tilt_degrees(quad: np.ndarray) -> float:
    """Average tilt of the quad's horizontal edges (signed).

    Positive = clockwise tilt. Used to compare with homography tilt.
    """
    tl, tr, br, bl = quad
    top = np.degrees(np.arctan2(tr[1] - tl[1], tr[0] - tl[0]))
    bottom = np.degrees(np.arctan2(br[1] - bl[1], br[0] - bl[0]))
    return (top + bottom) / 2.0


def _max_angle_deviation(quad: np.ndarray) -> float:
    """Max deviation of any interior angle from 90 degrees.

    A perfect rectangle returns 0. Used to detect near-rectangular quads
    where perspective correction would add noise without benefit.
    """
    pts = quad.astype(np.float64)
    max_dev = 0.0
    for i in range(4):
        p0 = pts[i]
        p1 = pts[(i + 1) % 4]
        p2 = pts[(i - 1) % 4]
        v1 = p1 - p0
        v2 = p2 - p0
        dot = np.dot(v1, v2)
        mag = np.linalg.norm(v1) * np.linalg.norm(v2)
        if mag < 1e-6:
            continue
        cos_angle = np.clip(dot / mag, -1.0, 1.0)
        angle = np.degrees(np.arccos(cos_angle))
        max_dev = max(max_dev, abs(angle - 90.0))
    return max_dev


def _homography_tilt_degrees(M: np.ndarray) -> float:
    """Extract the rotation component from a 3x3 homography matrix.

    Returns the approximate tilt in degrees (positive = clockwise).
    Uses the upper-left 2x2 sub-matrix as an affine approximation.
    """
    if M[2, 2] != 0:
        M_norm = M / M[2, 2]
    else:
        M_norm = M
    return np.degrees(np.arctan2(M_norm[1, 0], M_norm[0, 0]))


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
