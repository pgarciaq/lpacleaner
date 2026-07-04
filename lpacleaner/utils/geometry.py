"""Quad corner ordering, homography computation, and distance helpers.

Used by Stage 4 (page detection) and Stage 5 (perspective correction).
"""

from __future__ import annotations

import cv2
import numpy as np


def order_corners(pts: np.ndarray) -> np.ndarray:
    """Order 4 corner points as [top-left, top-right, bottom-right, bottom-left].

    Uses sum and difference of coordinates:
    - Top-left has smallest sum (x+y)
    - Bottom-right has largest sum (x+y)
    - Top-right has smallest difference (y-x)
    - Bottom-left has largest difference (y-x)

    Args:
        pts: Array of shape (4, 2) with corner coordinates.

    Returns:
        Array of shape (4, 2) with corners in TL, TR, BR, BL order, dtype float32.

    Raises:
        ValueError: If input does not have exactly 4 points.
    """
    pts = np.asarray(pts, dtype=np.float32)
    if pts.shape[0] != 4:
        raise ValueError(f"Expected 4 points, got {pts.shape[0]}")

    ordered = np.zeros((4, 2), dtype=np.float32)

    s = pts.sum(axis=1)
    d = np.diff(pts, axis=1).ravel()

    ordered[0] = pts[np.argmin(s)]   # TL: smallest x+y
    ordered[2] = pts[np.argmax(s)]   # BR: largest x+y
    ordered[1] = pts[np.argmin(d)]   # TR: smallest y-x
    ordered[3] = pts[np.argmax(d)]   # BL: largest y-x

    return ordered


def compute_target_size(corners: np.ndarray) -> tuple[int, int]:
    """Compute the target rectangle dimensions from 4 ordered corners.

    Takes the average of top/bottom edge lengths as width and
    left/right edge lengths as height.

    Args:
        corners: Array of shape (4, 2) in TL, TR, BR, BL order.

    Returns:
        (width, height) as integers.
    """
    tl, tr, br, bl = corners

    top_w = np.linalg.norm(tr - tl)
    bottom_w = np.linalg.norm(br - bl)
    width = int(round((top_w + bottom_w) / 2))

    left_h = np.linalg.norm(bl - tl)
    right_h = np.linalg.norm(br - tr)
    height = int(round((left_h + right_h) / 2))

    return width, height


def get_perspective_transform(
    src: np.ndarray,
    dst: np.ndarray,
) -> np.ndarray:
    """Compute the 3x3 perspective transform matrix from src to dst points.

    Thin wrapper around cv2.getPerspectiveTransform for consistent typing.

    Args:
        src: Source points, shape (4, 2), float32.
        dst: Destination points, shape (4, 2), float32.

    Returns:
        3x3 transformation matrix (float64).
    """
    src = np.asarray(src, dtype=np.float32)
    dst = np.asarray(dst, dtype=np.float32)
    return cv2.getPerspectiveTransform(src, dst)
