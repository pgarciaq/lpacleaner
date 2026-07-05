"""Image grouping, stitching, non-content detection, and retake dedup.

Used by Stage 1. Groups consecutive images by ORB feature overlap,
stitches partial photos into composites, detects book covers, and
removes near-duplicate retakes (keeping the sharper image).
"""

from __future__ import annotations

import logging
from collections import defaultdict

import cv2
import numpy as np

from ghh.config import Config

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Focus metric
# ---------------------------------------------------------------------------

def compute_focus(img: np.ndarray) -> float:
    """Compute focus quality via Laplacian variance (higher = sharper)."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


# ---------------------------------------------------------------------------
# Non-content detection
# ---------------------------------------------------------------------------

_DARK_THRESHOLD = 80
_DARK_PIXEL_FRAC = 0.80


def is_non_content(img: np.ndarray) -> bool:
    """Detect non-content images (book covers, spine shots, equipment).

    An image is classified as non-content if >80% of its pixels are
    dark (all channels below threshold). This catches dark leather/metal
    book covers like LPA-1's IMG_0231.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    dark_frac = float(np.count_nonzero(gray < _DARK_THRESHOLD)) / gray.size
    return dark_frac > _DARK_PIXEL_FRAC


# ---------------------------------------------------------------------------
# ORB matching helpers
# ---------------------------------------------------------------------------

def _match_pair(
    img_a: np.ndarray,
    img_b: np.ndarray,
    cfg: Config,
) -> tuple[int, float, float]:
    """Match two images using ORB features.

    Returns:
        (good_match_count, inlier_ratio, overlap_frac)
        If matching fails at any step, returns (0, 0.0, 0.0).
    """
    orb = cv2.ORB_create(nfeatures=2000)

    gray_a = cv2.cvtColor(img_a, cv2.COLOR_BGR2GRAY) if img_a.ndim == 3 else img_a
    gray_b = cv2.cvtColor(img_b, cv2.COLOR_BGR2GRAY) if img_b.ndim == 3 else img_b

    kp_a, des_a = orb.detectAndCompute(gray_a, None)
    kp_b, des_b = orb.detectAndCompute(gray_b, None)

    if des_a is None or des_b is None or len(des_a) < 2 or len(des_b) < 2:
        return 0, 0.0, 0.0

    bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    try:
        raw_matches = bf.knnMatch(des_a, des_b, k=2)
    except cv2.error:
        return 0, 0.0, 0.0

    # Lowe's ratio test
    good = []
    for m_pair in raw_matches:
        if len(m_pair) == 2:
            m, n = m_pair
            if m.distance < cfg.stitch_ratio_threshold * n.distance:
                good.append(m)

    good_count = len(good)
    if good_count < cfg.stitch_min_matches:
        return good_count, 0.0, 0.0

    pts_a = np.float32([kp_a[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    pts_b = np.float32([kp_b[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

    H, mask = cv2.findHomography(pts_a, pts_b, cv2.RANSAC, 5.0)
    if H is None or mask is None:
        return good_count, 0.0, 0.0

    inlier_ratio = float(mask.sum()) / len(mask)

    # Estimate overlap area
    h_a, w_a = img_a.shape[:2]
    corners_a = np.float32([[0, 0], [w_a, 0], [w_a, h_a], [0, h_a]]).reshape(-1, 1, 2)
    warped_corners = cv2.perspectiveTransform(corners_a, H)

    h_b, w_b = img_b.shape[:2]
    rect_b = np.array([0, 0, w_b, h_b], dtype=np.float32)

    wc = warped_corners.reshape(-1, 2)
    x_min = max(float(wc[:, 0].min()), 0)
    x_max = min(float(wc[:, 0].max()), w_b)
    y_min = max(float(wc[:, 1].min()), 0)
    y_max = min(float(wc[:, 1].max()), h_b)

    if x_max <= x_min or y_max <= y_min:
        return good_count, inlier_ratio, 0.0

    overlap_area = (x_max - x_min) * (y_max - y_min)
    img_b_area = h_b * w_b
    overlap_frac = overlap_area / img_b_area

    return good_count, inlier_ratio, overlap_frac


# ---------------------------------------------------------------------------
# Group detection
# ---------------------------------------------------------------------------

def detect_groups(
    images: dict[str, np.ndarray],
    cfg: Config,
) -> list[list[str]]:
    """Detect which images are overlapping partials of the same page.

    Uses ORB feature matching on consecutive image pairs, then builds
    transitive groups. Respects manual overrides from Config.

    Args:
        images: dict mapping image stem names to BGR arrays, in filename order.
        cfg: Pipeline configuration with stitch parameters and overrides.

    Returns:
        List of groups, where each group is a sorted list of image stems.
    """
    names = sorted(images.keys())

    # If manual stitch_groups are provided, use them as the primary grouping
    if cfg.stitch_groups:
        manual_stems = set()
        groups: list[list[str]] = []
        for group in cfg.stitch_groups:
            stems = sorted(n.rsplit(".", 1)[0] if "." in n else n for n in group)
            stems = [s for s in stems if s in images]
            if stems:
                groups.append(stems)
                manual_stems.update(stems)

        for name in names:
            if name not in manual_stems:
                groups.append([name])

        return groups

    # no_stitch images are forced into singleton groups
    no_stitch = set()
    if cfg.no_stitch_images:
        no_stitch = {
            n.rsplit(".", 1)[0] if "." in n else n for n in cfg.no_stitch_images
        }

    # Build adjacency via ORB matching on consecutive pairs
    adjacency: dict[str, set[str]] = defaultdict(set)

    for i in range(len(names) - 1):
        name_a, name_b = names[i], names[i + 1]

        if name_a in no_stitch or name_b in no_stitch:
            continue

        good_count, inlier_ratio, overlap_frac = _match_pair(
            images[name_a], images[name_b], cfg,
        )

        if (
            good_count >= cfg.stitch_min_matches
            and inlier_ratio >= cfg.stitch_inlier_ratio
            and overlap_frac >= cfg.stitch_min_overlap_frac
        ):
            adjacency[name_a].add(name_b)
            adjacency[name_b].add(name_a)
            logger.info(
                "Match %s ↔ %s: %d matches, %.1f%% inliers, %.1f%% overlap",
                name_a, name_b, good_count,
                inlier_ratio * 100, overlap_frac * 100,
            )

    # Build transitive groups via BFS
    visited: set[str] = set()
    groups = []

    for name in names:
        if name in visited:
            continue
        if name not in adjacency:
            groups.append([name])
            visited.add(name)
            continue

        # BFS from this node
        group: list[str] = []
        queue = [name]
        while queue:
            current = queue.pop(0)
            if current in visited:
                continue
            visited.add(current)
            group.append(current)
            for neighbor in adjacency[current]:
                if neighbor not in visited:
                    queue.append(neighbor)

        groups.append(sorted(group))

    return groups


# ---------------------------------------------------------------------------
# Retake deduplication
# ---------------------------------------------------------------------------

def deduplicate_retakes(
    group_images: dict[str, np.ndarray],
    cfg: Config,
) -> tuple[dict[str, np.ndarray], dict[str, str]]:
    """Remove near-duplicate retakes from a group, keeping the sharper one.

    Two images are considered retakes (not complementary partials) when
    their overlap exceeds retake_overlap_threshold (default 90%).

    Returns:
        (kept_images, discarded) where discarded maps stem → reason.
    """
    if len(group_images) < 2:
        return dict(group_images), {}

    names = sorted(group_images.keys())
    discarded: dict[str, str] = {}
    keep_set: set[str] = set(names)

    for i in range(len(names)):
        if names[i] not in keep_set:
            continue
        for j in range(i + 1, len(names)):
            if names[j] not in keep_set:
                continue

            _, _, overlap_frac = _match_pair(
                group_images[names[i]], group_images[names[j]], cfg,
            )

            if overlap_frac >= cfg.retake_overlap_threshold:
                focus_i = compute_focus(group_images[names[i]])
                focus_j = compute_focus(group_images[names[j]])

                if focus_i >= focus_j:
                    keep_set.discard(names[j])
                    discarded[names[j]] = (
                        f"retake of {names[i]} "
                        f"(overlap={overlap_frac:.1%}, focus={focus_j:.1f} vs {focus_i:.1f})"
                    )
                else:
                    keep_set.discard(names[i])
                    discarded[names[i]] = (
                        f"retake of {names[j]} "
                        f"(overlap={overlap_frac:.1%}, focus={focus_i:.1f} vs {focus_j:.1f})"
                    )
                    break

    kept = {n: group_images[n] for n in names if n in keep_set}
    return kept, discarded


# ---------------------------------------------------------------------------
# Stitching fallback chain
# ---------------------------------------------------------------------------

def stitch_images(
    images: dict[str, np.ndarray],
    cfg: Config,
) -> tuple[np.ndarray, str, bool]:
    """Stitch a group of images using the fallback chain.

    Fallback order:
    1. cv2.Stitcher PANORAMA mode
    2. cv2.Stitcher SCANS mode
    3. Manual homography stitching
    4. Best single image (fallback)

    Returns:
        (result_image, method_name, success)
        success=False means stitching failed and best_single was used.
    """
    names = sorted(images.keys())

    if len(names) == 1:
        return images[names[0]].copy(), "single", True

    img_list = [images[n] for n in names]

    # Try PANORAMA mode
    result = _try_cv_stitcher(img_list, cv2.Stitcher_PANORAMA)
    if result is not None:
        logger.info("Stitched %d images with PANORAMA mode", len(names))
        return result, "panorama", True

    # Try SCANS mode
    result = _try_cv_stitcher(img_list, cv2.Stitcher_SCANS)
    if result is not None:
        logger.info("Stitched %d images with SCANS mode", len(names))
        return result, "scans", True

    # Try manual homography
    result = _try_manual_homography(img_list, cfg)
    if result is not None:
        logger.info("Stitched %d images with manual homography", len(names))
        return result, "homography", True

    # Fallback: best single image
    logger.warning(
        "All stitching methods failed for group %s, using best single image",
        names,
    )
    best_name = max(names, key=lambda n: compute_focus(images[n]))
    return images[best_name].copy(), "best_single", False


def _try_cv_stitcher(
    images: list[np.ndarray],
    mode: int,
) -> np.ndarray | None:
    """Attempt stitching with OpenCV's Stitcher. Returns None on failure."""
    try:
        stitcher = cv2.Stitcher.create(mode)
        status, result = stitcher.stitch(images)
        if status == cv2.Stitcher_OK:
            return result
        return None
    except cv2.error:
        return None


def _try_manual_homography(
    images: list[np.ndarray],
    cfg: Config,
) -> np.ndarray | None:
    """Attempt pairwise homography stitching with feathered blending.

    Sequentially warps each image into the coordinate frame of the first.
    """
    if len(images) < 2:
        return None

    orb = cv2.ORB_create(nfeatures=2000)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING)

    base = images[0]

    for i in range(1, len(images)):
        gray_base = cv2.cvtColor(base, cv2.COLOR_BGR2GRAY) if base.ndim == 3 else base
        gray_next = cv2.cvtColor(images[i], cv2.COLOR_BGR2GRAY) if images[i].ndim == 3 else images[i]

        kp_b, des_b = orb.detectAndCompute(gray_base, None)
        kp_n, des_n = orb.detectAndCompute(gray_next, None)

        if des_b is None or des_n is None or len(des_b) < 10 or len(des_n) < 10:
            return None

        try:
            raw_matches = bf.knnMatch(des_n, des_b, k=2)
        except cv2.error:
            return None

        good = []
        for m_pair in raw_matches:
            if len(m_pair) == 2:
                m, n = m_pair
                if m.distance < cfg.stitch_ratio_threshold * n.distance:
                    good.append(m)

        if len(good) < cfg.stitch_min_matches:
            return None

        pts_n = np.float32([kp_n[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        pts_b = np.float32([kp_b[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

        H, mask = cv2.findHomography(pts_n, pts_b, cv2.RANSAC, 5.0)
        if H is None:
            return None

        h_b, w_b = base.shape[:2]
        h_n, w_n = images[i].shape[:2]

        # Compute output canvas size
        corners_n = np.float32([[0, 0], [w_n, 0], [w_n, h_n], [0, h_n]]).reshape(-1, 1, 2)
        warped_corners = cv2.perspectiveTransform(corners_n, H)
        all_corners = np.vstack([
            np.float32([[0, 0], [w_b, 0], [w_b, h_b], [0, h_b]]),
            warped_corners.reshape(-1, 2),
        ])

        x_min, y_min = all_corners.min(axis=0).astype(int)
        x_max, y_max = all_corners.max(axis=0).astype(int)

        # Translation to keep everything in positive coordinates
        translation = np.array([
            [1, 0, -x_min],
            [0, 1, -y_min],
            [0, 0, 1],
        ], dtype=np.float64)

        out_w = x_max - x_min
        out_h = y_max - y_min

        if out_w > 10000 or out_h > 10000:
            return None

        warped_next = cv2.warpPerspective(images[i], translation @ H, (out_w, out_h))

        # Place base image on canvas
        canvas = np.zeros((out_h, out_w, 3), dtype=np.uint8)
        bx, by = -x_min, -y_min
        canvas[by:by + h_b, bx:bx + w_b] = base

        # Simple blend: where both have content, average; otherwise take non-zero
        mask_canvas = (canvas.sum(axis=2) > 0).astype(np.float32)
        mask_warped = (warped_next.sum(axis=2) > 0).astype(np.float32)

        both = (mask_canvas * mask_warped)[:, :, np.newaxis]
        only_canvas = (mask_canvas * (1 - mask_warped))[:, :, np.newaxis]
        only_warped = ((1 - mask_canvas) * mask_warped)[:, :, np.newaxis]

        base = (
            (both * (canvas.astype(np.float32) * 0.5 + warped_next.astype(np.float32) * 0.5))
            + only_canvas * canvas.astype(np.float32)
            + only_warped * warped_next.astype(np.float32)
        ).astype(np.uint8)

    return base
