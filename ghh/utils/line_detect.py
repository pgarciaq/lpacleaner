"""Generic ink detection, staff line detection, and foxing filter (R9).

Used by Stages 2 (orientation), 6 (content area), 7 (deskew), and 8 (dewarp).
All thresholds come from Config (populated by the analyze command or defaults).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import cv2
import numpy as np

from ghh.config import Config

logger = logging.getLogger(__name__)

_MIN_INK_RATIO = 0.0005  # below this, HSV mask is considered empty -> fallback


@dataclass
class StaffLine:
    """A single detected staff line with fitted polynomial."""

    y_center: float
    points: np.ndarray
    polynomial_coeffs: np.ndarray
    angle: float


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def detect_ink_mask(img: np.ndarray, cfg: Config) -> np.ndarray:
    """Isolate staff-line ink pixels using auto-detected color thresholds.

    Primary path: HSV range from cfg (hue center +/- range, min sat, min value).
    Fallback: channel-difference method if HSV produces too few pixels.

    Accepts BGR or grayscale input. Returns a binary uint8 mask (0 or 255).
    """
    if img.ndim == 2:
        return np.zeros(img.shape[:2], dtype=np.uint8)

    mask = _hsv_ink_mask(img, cfg)

    ink_ratio = np.count_nonzero(mask) / mask.size
    if ink_ratio < _MIN_INK_RATIO:
        logger.debug("HSV ink mask too sparse (%.5f), falling back to channel-difference", ink_ratio)
        mask = _channel_diff_ink_mask(img, cfg)

    return mask


def detect_ink_mask_geometric(img: np.ndarray, cfg: Config) -> np.ndarray:
    """Ink mask filtered for line-like geometry (R9 foxing discrimination).

    1. detect_ink_mask() for color isolation
    2. Morphological opening with 1xN horizontal kernel (removes round spots)
    3. Connected-component filtering: keep only components with aspect_ratio > 5:1

    Removes foxing spots, rust blobs, and other non-linear ink-colored artifacts
    while preserving staff lines and border frames.
    """
    color_mask = detect_ink_mask(img, cfg)
    if np.count_nonzero(color_mask) == 0:
        return color_mask

    # Morphological opening with wide horizontal kernel to keep only line-like structures
    kernel_width = max(30, img.shape[1] // 30)
    h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_width, 1))
    opened = cv2.morphologyEx(color_mask, cv2.MORPH_OPEN, h_kernel)

    # Connected-component aspect ratio filtering
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(opened, connectivity=8)
    filtered = np.zeros_like(opened)

    for i in range(1, num_labels):
        w = stats[i, cv2.CC_STAT_WIDTH]
        h = stats[i, cv2.CC_STAT_HEIGHT]
        if h == 0:
            continue
        aspect = w / max(h, 1)
        if aspect >= 5.0:
            filtered[labels == i] = 255

    return filtered


def detect_staff_lines(img: np.ndarray, cfg: Config) -> list[StaffLine]:
    """Detect staff lines using geometric ink mask and Hough transform.

    Returns a list of StaffLine objects sorted by y_center (top to bottom).
    Empty list if no lines detected.
    """
    mask = detect_ink_mask_geometric(img, cfg)
    if np.count_nonzero(mask) == 0:
        return []

    h, w = mask.shape[:2]

    # Close small gaps in the mask
    close_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (20, 1))
    mask_closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, close_kernel)

    min_length = int(w * 0.2)
    lines = cv2.HoughLinesP(
        mask_closed,
        rho=1,
        theta=np.pi / 180,
        threshold=50,
        minLineLength=min_length,
        maxLineGap=20,
    )

    if lines is None:
        return []

    # Filter to near-horizontal lines and collect segments
    segments = []
    for line in lines:
        coords = line.ravel()
        x1, y1, x2, y2 = int(coords[0]), int(coords[1]), int(coords[2]), int(coords[3])
        dx = x2 - x1
        dy = y2 - y1
        length = np.sqrt(dx * dx + dy * dy)
        if length < 10:
            continue
        angle_deg = np.degrees(np.arctan2(dy, dx))
        if abs(angle_deg) < 15:
            y_mid = (y1 + y2) / 2
            segments.append((x1, y1, x2, y2, y_mid, angle_deg))

    if not segments:
        return []

    # Cluster segments by y-coordinate
    segments.sort(key=lambda s: s[4])
    clusters: list[list[tuple]] = []
    cluster_eps = 15

    for seg in segments:
        y_mid = seg[4]
        if clusters and abs(y_mid - np.mean([s[4] for s in clusters[-1]])) < cluster_eps:
            clusters[-1].append(seg)
        else:
            clusters.append([seg])

    # Build StaffLine objects from clusters
    staff_lines: list[StaffLine] = []
    for cluster in clusters:
        points = []
        angles = []
        for x1, y1, x2, y2, y_mid, angle in cluster:
            points.extend([(x1, y1), (x2, y2)])
            angles.append(angle)

        pts = np.array(points)
        if len(pts) < 2:
            continue

        xs = pts[:, 0].astype(np.float64)
        ys = pts[:, 1].astype(np.float64)

        # Adaptive degree polynomial fitting.
        # Start with degree-1 (linear, handles skew). Go higher only if
        # residuals indicate real curvature (spine warping). This avoids
        # ill-conditioned fits on tightly clustered segments.
        coeffs = _adaptive_polyfit(xs, ys, max_degree=min(3, len(pts) - 1))

        y_center = float(np.mean(ys))
        mean_angle = float(np.median(angles))

        staff_lines.append(StaffLine(
            y_center=y_center,
            points=pts,
            polynomial_coeffs=coeffs,
            angle=mean_angle,
        ))

    staff_lines.sort(key=lambda sl: sl.y_center)
    return staff_lines


def detect_dominant_angle(
    img: np.ndarray,
    cfg: Config,
    quad_corners: np.ndarray | None = None,
) -> float:
    """Median angle of detected staff lines in degrees.

    When *quad_corners* (4x2 float32 array) is provided, pixels outside
    the quad are zeroed before staff-line detection, preventing background
    artifacts from corrupting the angle estimate.

    Returns 0.0 if no staff lines are found.
    """
    if quad_corners is not None:
        img = _apply_quad_mask(img, quad_corners)
    lines = detect_staff_lines(img, cfg)
    if not lines:
        return 0.0
    return float(np.median([l.angle for l in lines]))


def detect_illustration_regions(img: np.ndarray, cfg: Config) -> np.ndarray:
    """R4: Mask multi-colored regions (high local hue variance).

    Used to exclude illustrations from line detection.
    Returns a binary uint8 mask where 255 = illustration region.
    """
    if img.ndim == 2:
        return np.zeros(img.shape[:2], dtype=np.uint8)

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hue = hsv[:, :, 0].astype(np.float32)

    # Compute local hue variance using a sliding window
    block_size = 32
    h, w = hue.shape
    mask = np.zeros((h, w), dtype=np.uint8)

    for row in range(0, h - block_size, block_size // 2):
        for col in range(0, w - block_size, block_size // 2):
            block = hue[row:row + block_size, col:col + block_size]
            # Only consider saturated, non-background pixels
            sat_block = hsv[row:row + block_size, col:col + block_size, 1]
            val_block = hsv[row:row + block_size, col:col + block_size, 2]
            fg_mask = (sat_block > 50) & (val_block > 80) & (val_block < 240)

            if np.count_nonzero(fg_mask) < block_size * block_size * 0.15:
                continue

            fg_hues = block[fg_mask]
            if len(fg_hues) < 5:
                continue

            # Circular variance for hue (hue wraps at 180)
            hue_range = _circular_range(fg_hues, period=180)
            if hue_range > 40:
                mask[row:row + block_size, col:col + block_size] = 255

    # Dilate to connect nearby illustration blocks and fill gaps
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (block_size, block_size))
    mask = cv2.dilate(mask, kernel)

    return mask


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _apply_quad_mask(img: np.ndarray, quad: np.ndarray) -> np.ndarray:
    """Zero out pixels outside the quad region.

    Works on both BGR and grayscale images. Returns a copy so the
    original is not modified.
    """
    out = img.copy()
    h, w = out.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    pts = quad.astype(np.int32).reshape((-1, 1, 2))
    cv2.fillConvexPoly(mask, pts, 255)
    if out.ndim == 3:
        out[mask == 0] = 0
    else:
        out[mask == 0] = 0
    return out


_CURVATURE_THRESHOLD = 1.5  # px RMS residual: below this, lower degree is sufficient


def _adaptive_polyfit(xs: np.ndarray, ys: np.ndarray, max_degree: int = 3) -> np.ndarray:
    """Fit a polynomial with adaptive degree selection.

    Starts at degree-1 (linear). Increases degree only if the RMS residual
    exceeds the curvature threshold, indicating real page warping rather
    than noise. This prevents ill-conditioned fits on tightly clustered
    Hough segments (which trigger numpy RankWarning).
    """
    import warnings

    if max_degree < 1 or len(xs) < 2:
        return np.array([float(np.mean(ys))])

    # numpy 2.0+ moved RankWarning to np.exceptions
    _rank_warning = getattr(np, "RankWarning", None)
    if _rank_warning is None:
        _rank_warning = getattr(np.exceptions, "RankWarning", Warning)

    best_coeffs = np.array([float(np.mean(ys))])

    for deg in range(1, max_degree + 1):
        if len(xs) <= deg:
            break
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", _rank_warning)
            try:
                coeffs = np.polyfit(xs, ys, deg)
            except (np.linalg.LinAlgError, ValueError):
                break

        residuals = ys - np.polyval(coeffs, xs)
        rms = float(np.sqrt(np.mean(residuals**2)))
        best_coeffs = coeffs

        if rms < _CURVATURE_THRESHOLD:
            break

    return best_coeffs


def _hsv_ink_mask(img: np.ndarray, cfg: Config) -> np.ndarray:
    """Primary ink detection via HSV thresholds."""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    hue_center = cfg.staff_color_hue
    hue_range = cfg.staff_color_range
    sat_min = cfg.staff_saturation_min
    val_min = cfg.staff_value_min

    lower_hue = hue_center - hue_range
    upper_hue = hue_center + hue_range

    if lower_hue < 0:
        # Hue wraps around 0 (e.g., red spans 170-10)
        mask1 = cv2.inRange(hsv, (0, sat_min, val_min), (upper_hue, 255, 255))
        mask2 = cv2.inRange(hsv, (180 + lower_hue, sat_min, val_min), (180, 255, 255))
        mask = cv2.bitwise_or(mask1, mask2)
    elif upper_hue > 180:
        mask1 = cv2.inRange(hsv, (lower_hue, sat_min, val_min), (180, 255, 255))
        mask2 = cv2.inRange(hsv, (0, sat_min, val_min), (upper_hue - 180, 255, 255))
        mask = cv2.bitwise_or(mask1, mask2)
    else:
        mask = cv2.inRange(hsv, (lower_hue, sat_min, val_min), (upper_hue, 255, 255))

    return mask


def _channel_diff_ink_mask(img: np.ndarray, cfg: Config) -> np.ndarray:
    """Fallback ink detection via channel differences.

    For red ink: R-G > threshold AND R-B > threshold.
    Generalizes to other dominant channels based on hue.
    """
    b, g, r = cv2.split(img)
    b = b.astype(np.int16)
    g = g.astype(np.int16)
    r = r.astype(np.int16)

    hue = cfg.staff_color_hue
    rg_thresh = cfg.channel_diff_rg
    rb_thresh = cfg.channel_diff_rb

    if hue <= 15 or hue >= 165:
        # Red-dominant ink
        mask = ((r - g) > rg_thresh) & ((r - b) > rb_thresh)
    elif 15 < hue <= 45:
        # Orange/brown ink
        mask = ((r - b) > rb_thresh) & ((g - b) > rg_thresh // 2)
    elif 45 < hue <= 75:
        # Yellow/green ink
        mask = ((g - r) > rg_thresh) & ((g - b) > rb_thresh)
    elif 75 < hue <= 105:
        # Green ink
        mask = ((g - r) > rg_thresh) & ((g - b) > rb_thresh)
    elif 105 < hue <= 135:
        # Blue/cyan ink
        mask = ((b - r) > rb_thresh) & ((b - g) > rg_thresh)
    else:
        # Purple/magenta
        mask = ((r - g) > rg_thresh) & ((b - g) > rb_thresh)

    return (mask.astype(np.uint8) * 255)


def _circular_range(values: np.ndarray, period: int = 180) -> float:
    """Compute the range of values on a circular scale (handles wrap-around)."""
    if len(values) == 0:
        return 0.0
    sorted_vals = np.sort(values % period)
    if len(sorted_vals) < 2:
        return 0.0
    gaps = np.diff(sorted_vals)
    wrap_gap = period - sorted_vals[-1] + sorted_vals[0]
    max_gap = max(np.max(gaps), wrap_gap)
    return float(period - max_gap)


def count_horizontal_lines(img: np.ndarray) -> int:
    """Count near-horizontal line segments via Canny + HoughLinesP.

    Used by analyze (coarse orientation) and Stage 2 (per-image orientation).
    A line is "horizontal" if its angle is within 15° of the horizontal axis.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img

    edges = cv2.Canny(gray, 50, 150)

    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180,
        threshold=40,
        minLineLength=max(30, img.shape[1] // 10),
        maxLineGap=15,
    )

    if lines is None:
        return 0

    count = 0
    for line in lines:
        seg = line.ravel()
        if len(seg) < 4:
            continue
        x1, y1, x2, y2 = int(seg[0]), int(seg[1]), int(seg[2]), int(seg[3])
        dx = x2 - x1
        dy = y2 - y1
        if dx == 0:
            continue
        angle_deg = abs(np.degrees(np.arctan2(dy, dx)))
        if angle_deg < 15 or angle_deg > 165:
            count += 1

    return count
