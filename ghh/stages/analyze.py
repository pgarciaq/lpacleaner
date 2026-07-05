"""Analyze command: auto-detect book characteristics and generate book.toml.

Scans sample images from a book directory, detects ink color, layout,
photography conditions, and physical condition, then writes a book.toml
configuration file. Runs automatically as part of ``ghh run`` if
no book.toml exists.
"""

from __future__ import annotations

import logging
import sys
from collections import Counter
from pathlib import Path

if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib  # type: ignore[import-not-found]

import cv2
import numpy as np

from ghh.config import Config
from ghh.utils.image_io import load_image
from ghh.utils.line_detect import count_horizontal_lines
from ghh.utils.page_find import crop_to_page, find_page_quad
from ghh.utils.stats import adaptive_sample_count, robust_median

logger = logging.getLogger(__name__)

_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def run_analyze(
    input_dir: Path,
    output_dir: Path,
    samples: int | None = None,
) -> Path:
    """Run the full analyze pipeline and write book.toml.

    Returns the path to the generated book.toml.
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    image_paths = _list_images(input_dir)
    if not image_paths:
        logger.warning("No images found in %s, writing defaults", input_dir)
        return _write_defaults(output_dir)

    n_samples = samples or adaptive_sample_count(len(image_paths))
    sample_paths = _select_evenly_spaced(image_paths, n_samples)

    images = []
    for p in sample_paths:
        img, _ = load_image(p)
        if img is not None:
            images.append(img)

    if len(images) < 3:
        logger.warning("Fewer than 3 valid samples, using defaults")
        return _write_defaults(output_dir)

    orientation_subset = images[:min(5, len(images))]
    coarse_offset = _detect_coarse_orientation(orientation_subset)

    if coarse_offset != 0:
        images = [_apply_rotation(img, coarse_offset) for img in images]

    cropped_pages = []
    for img in images:
        quad = find_page_quad(img)
        page = crop_to_page(img, quad)
        cropped_pages.append(page)

    ink_result = _discover_ink_color(cropped_pages)
    layout_result = _analyze_layout(cropped_pages)
    photo_result = _analyze_photography(images)
    photo_result["coarse_rotation_offset"] = coarse_offset
    condition_result = _analyze_condition(cropped_pages)

    toml_path = _write_book_toml(
        output_dir,
        ink=ink_result,
        layout=layout_result,
        photography=photo_result,
        condition=condition_result,
    )

    logger.info("Wrote %s", toml_path)
    return toml_path


# ---------------------------------------------------------------------------
# Ink color discovery
# ---------------------------------------------------------------------------

def _discover_ink_color(pages: list[np.ndarray]) -> dict:
    """Detect dominant ink color across cropped pages."""
    hue_values = []
    sat_values = []
    val_values = []

    for page in pages:
        hsv = cv2.cvtColor(page, cv2.COLOR_BGR2HSV)
        h, s, v = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]

        mask = (v > 60) & (v < 200) & (s > 20)
        if mask.sum() < 100:
            continue

        hue_values.extend(h[mask].tolist())
        sat_values.extend(s[mask].tolist())
        val_values.extend(v[mask].tolist())

    if not hue_values:
        return _default_ink()

    hue_hist = np.bincount(np.array(hue_values, dtype=np.uint8), minlength=180)

    kernel = np.ones(5) / 5
    smoothed = np.convolve(np.tile(hue_hist, 3), kernel, mode="same")
    smoothed = smoothed[180:360]

    dominant_hue = int(np.argmax(smoothed))

    sat_med = robust_median(sat_values) or 40.0
    val_med = robust_median(val_values) or 80.0

    b_means, g_means, r_means = [], [], []
    for page in pages:
        hsv = cv2.cvtColor(page, cv2.COLOR_BGR2HSV)
        h = hsv[:, :, 0]
        hue_diff = np.minimum(np.abs(h.astype(int) - dominant_hue),
                              180 - np.abs(h.astype(int) - dominant_hue))
        ink_mask = hue_diff < 20
        if ink_mask.sum() < 50:
            continue
        b_means.append(float(page[:, :, 0][ink_mask].mean()))
        g_means.append(float(page[:, :, 1][ink_mask].mean()))
        r_means.append(float(page[:, :, 2][ink_mask].mean()))

    rg_diff = abs((robust_median(r_means) or 100) - (robust_median(g_means) or 100))
    rb_diff = abs((robust_median(r_means) or 100) - (robust_median(b_means) or 100))

    return {
        "staff_color_hue": dominant_hue,
        "staff_color_range": 15,
        "staff_saturation_min": max(20, int(sat_med * 0.5)),
        "staff_value_min": max(40, int(val_med * 0.5)),
        "channel_diff_rg": max(15, int(rg_diff * 0.7)),
        "channel_diff_rb": max(15, int(rb_diff * 0.7)),
    }


def _default_ink() -> dict:
    return {
        "staff_color_hue": 5,
        "staff_color_range": 15,
        "staff_saturation_min": 40,
        "staff_value_min": 80,
        "channel_diff_rg": 30,
        "channel_diff_rb": 30,
    }


# ---------------------------------------------------------------------------
# Coarse orientation detection
# ---------------------------------------------------------------------------

def _detect_coarse_orientation(images: list[np.ndarray]) -> int:
    """Try 4 cardinal rotations, return the offset with most horizontal lines."""
    rotations = {
        0: None,
        90: cv2.ROTATE_90_COUNTERCLOCKWISE,
        180: cv2.ROTATE_180,
        270: cv2.ROTATE_90_CLOCKWISE,
    }

    votes = []

    for img in images:
        best_angle = 0
        best_count = 0

        small = _downscale(img, max_width=1000)

        for angle, rot_flag in rotations.items():
            if rot_flag is not None:
                rotated = cv2.rotate(small, rot_flag)
            else:
                rotated = small

            count = count_horizontal_lines(rotated)
            if count > best_count:
                best_count = count
                best_angle = angle

        votes.append(best_angle)

    if not votes:
        return 0

    counter = Counter(votes)
    winner, _ = counter.most_common(1)[0]
    return winner


# ---------------------------------------------------------------------------
# Layout analysis
# ---------------------------------------------------------------------------

def _analyze_layout(pages: list[np.ndarray]) -> dict:
    """Detect layout features from cropped pages."""
    from ghh.utils.line_detect import detect_staff_lines

    cfg = Config(input_dir=Path("/tmp"))

    staff_counts = []
    aspect_ratios = []
    border_votes = []

    for page in pages:
        h, w = page.shape[:2]
        aspect_ratios.append(w / h if h > 0 else 1.0)

        lines = detect_staff_lines(page, cfg)
        staff_counts.append(len(lines))

        border_votes.append(_has_border_frame(page))

    median_staff = robust_median([float(c) for c in staff_counts])
    median_aspect = robust_median(aspect_ratios)

    return {
        "has_border_frame": sum(border_votes) > len(border_votes) / 2,
        "border_ink_matches_staff": True,
        "page_number_position": "top-right",
        "expected_staff_lines_per_page": int(median_staff) if median_staff else 16,
        "has_illustrations": False,
        "illustration_frequency": "none",
        "median_aspect_ratio": round(median_aspect, 2) if median_aspect else 1.33,
    }


def _has_border_frame(page: np.ndarray) -> bool:
    """Detect if a page has a rectangular border frame."""
    gray = cv2.cvtColor(page, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)

    h, w = edges.shape
    border_width = max(5, min(h, w) // 20)

    top = edges[:border_width, :].sum()
    bottom = edges[-border_width:, :].sum()
    left = edges[:, :border_width].sum()
    right = edges[:, -border_width:].sum()

    total_border = top + bottom + left + right
    total_edges = edges.sum()

    if total_edges == 0:
        return False

    return (total_border / total_edges) > 0.15


# ---------------------------------------------------------------------------
# Photography condition analysis
# ---------------------------------------------------------------------------

def _analyze_photography(images: list[np.ndarray]) -> dict:
    """Detect photography conditions from full (uncropped) images."""
    hotspot_count = 0
    finger_count = 0
    cast_deviations = []

    for img in images:
        if _has_hotspot(img):
            hotspot_count += 1

        if _has_finger(img):
            finger_count += 1

        cast_deviations.append(_color_cast_deviation(img))

    cast_median = robust_median(cast_deviations) or 0.0

    if cast_median < 5:
        cast_label = "none"
    elif cast_median < 15:
        cast_label = "slight_warm"
    else:
        cast_label = "strong_warm"

    return {
        "has_flash_hotspots": hotspot_count > len(images) * 0.3,
        "color_cast_detected": cast_label,
        "background_contrast": "dark_on_light",
        "shadow_severity": "none",
        "lens_distortion_k1": 0.0,
        "lens_distortion_k2": 0.0,
        "fingers_detected": finger_count > len(images) * 0.3,
    }


def _has_hotspot(img: np.ndarray) -> bool:
    """Check for clipped-white regions (flash)."""
    b, g, r = img[:, :, 0], img[:, :, 1], img[:, :, 2]
    mask = (b > 250) & (g > 250) & (r > 250)
    area_frac = mask.sum() / mask.size
    return area_frac > 0.005


def _has_finger(img: np.ndarray) -> bool:
    """Check for skin-colored regions at image borders."""
    from ghh.utils.preprocess import detect_fingers

    cfg = Config(input_dir=Path("/tmp"))
    mask = detect_fingers(img, cfg)
    return mask.sum() > 0


def _color_cast_deviation(img: np.ndarray) -> float:
    """Measure gray-world deviation (how far from neutral the image is)."""
    means = img.mean(axis=(0, 1))
    overall = means.mean()
    return float(np.max(np.abs(means - overall)))


# ---------------------------------------------------------------------------
# Physical condition analysis
# ---------------------------------------------------------------------------

def _analyze_condition(pages: list[np.ndarray]) -> dict:
    """Analyze physical condition of cropped pages."""
    foxing_scores = []
    stain_scores = []
    fading_scores = []

    for page in pages:
        foxing_scores.append(_foxing_score(page))
        stain_scores.append(_stain_score(page))
        fading_scores.append(_fading_score(page))

    return {
        "stain_severity": _score_to_severity(robust_median(stain_scores) or 0),
        "ink_fading": _score_to_severity(robust_median(fading_scores) or 0),
        "show_through_severity": "none",
        "foxing_severity": _score_to_severity(robust_median(foxing_scores) or 0),
        "iron_gall_halos": "none",
        "salt_deposits": "none",
    }


def _foxing_score(page: np.ndarray) -> float:
    """Score foxing presence (small reddish-brown spots)."""
    hsv = cv2.cvtColor(page, cv2.COLOR_BGR2HSV)
    h, s, v = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]

    fox_mask = ((h > 5) & (h < 25) & (s > 30) & (s < 150) &
                (v > 60) & (v < 180))

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    fox_mask = cv2.morphologyEx(fox_mask.astype(np.uint8), cv2.MORPH_OPEN, kernel)

    contours, _ = cv2.findContours(fox_mask, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)

    small_spots = 0
    for c in contours:
        area = cv2.contourArea(c)
        if 5 < area < 200:
            x, y, w, h_c = cv2.boundingRect(c)
            aspect = max(w, h_c) / max(min(w, h_c), 1)
            if aspect < 3:
                small_spots += 1

    page_area = page.shape[0] * page.shape[1]
    return small_spots / (page_area / 10000)


def _stain_score(page: np.ndarray) -> float:
    """Score large-area brightness deviations.

    Compares the image at two blur scales: large stains create
    differences between local (small-kernel) and regional (large-kernel)
    brightness. Normal page content (staff lines, text) creates high-freq
    differences that are removed by the small kernel.
    """
    gray = cv2.cvtColor(page, cv2.COLOR_BGR2GRAY)
    local = cv2.GaussianBlur(gray, (51, 51), 0)
    regional = cv2.GaussianBlur(gray, (151, 151), 0)
    diff = cv2.absdiff(local, regional)
    return float(diff.mean()) / 255.0


def _fading_score(page: np.ndarray) -> float:
    """Score ink fading by measuring saturation of colored regions."""
    hsv = cv2.cvtColor(page, cv2.COLOR_BGR2HSV)
    s = hsv[:, :, 1]
    v = hsv[:, :, 2]

    colored = s > 20
    if colored.sum() < 100:
        return 0.0

    sat_mean = float(s[colored].mean())
    return max(0, (80 - sat_mean) / 80)


def _score_to_severity(score: float) -> str:
    """Convert a 0-1+ numeric score to a severity label."""
    if score < 0.1:
        return "none"
    elif score < 0.3:
        return "mild"
    elif score < 0.6:
        return "moderate"
    else:
        return "severe"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _list_images(directory: Path) -> list[Path]:
    """List image files in directory, sorted by name."""
    paths = [p for p in sorted(directory.iterdir())
             if p.suffix.lower() in _IMAGE_EXTENSIONS]
    return paths


def _select_evenly_spaced(paths: list[Path], n: int) -> list[Path]:
    """Select n evenly-spaced items from a list."""
    if n >= len(paths):
        return list(paths)
    indices = np.linspace(0, len(paths) - 1, n, dtype=int)
    return [paths[i] for i in indices]


def _apply_rotation(img: np.ndarray, degrees: int) -> np.ndarray:
    """Apply cardinal rotation (0, 90, 180, 270)."""
    if degrees == 90:
        return cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
    elif degrees == 180:
        return cv2.rotate(img, cv2.ROTATE_180)
    elif degrees == 270:
        return cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    return img


def _downscale(img: np.ndarray, max_width: int = 1000) -> np.ndarray:
    """Downscale image if wider than max_width."""
    h, w = img.shape[:2]
    if w <= max_width:
        return img
    scale = max_width / w
    new_w = max_width
    new_h = int(h * scale)
    return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)


def _write_defaults(output_dir: Path) -> Path:
    """Write a book.toml with all default values."""
    return _write_book_toml(
        output_dir,
        ink=_default_ink(),
        layout={
            "has_border_frame": True,
            "border_ink_matches_staff": True,
            "page_number_position": "top-right",
            "expected_staff_lines_per_page": 16,
            "has_illustrations": False,
            "illustration_frequency": "none",
            "median_aspect_ratio": 1.33,
        },
        photography={
            "has_flash_hotspots": False,
            "color_cast_detected": "none",
            "background_contrast": "dark_on_light",
            "shadow_severity": "none",
            "lens_distortion_k1": 0.0,
            "lens_distortion_k2": 0.0,
            "fingers_detected": False,
            "coarse_rotation_offset": 0,
        },
        condition={
            "stain_severity": "none",
            "ink_fading": "none",
            "show_through_severity": "none",
            "foxing_severity": "none",
            "iron_gall_halos": "none",
            "salt_deposits": "none",
        },
    )


def _write_book_toml(
    output_dir: Path,
    *,
    ink: dict,
    layout: dict,
    photography: dict,
    condition: dict,
) -> Path:
    """Write all detected values to book.toml."""
    lines = []

    lines.append("[ink]")
    for k, v in ink.items():
        lines.append(f"{k} = {_toml_val(v)}")

    lines.append("")
    lines.append("[layout]")
    for k, v in layout.items():
        lines.append(f"{k} = {_toml_val(v)}")

    lines.append("")
    lines.append("[photography]")
    for k, v in photography.items():
        lines.append(f"{k} = {_toml_val(v)}")

    lines.append("")
    lines.append("[condition]")
    for k, v in condition.items():
        lines.append(f"{k} = {_toml_val(v)}")

    lines.append("")
    lines.append("[pipeline]")
    lines.append('profile = "full"')

    lines.append("")

    toml_path = output_dir / "book.toml"
    toml_path.write_text("\n".join(lines))
    return toml_path


def _toml_val(v) -> str:
    """Format a Python value as a TOML literal.

    Handles numpy scalar types (np.int64, np.float64, np.bool_) in
    addition to native Python types.
    """
    if isinstance(v, (bool, np.bool_)):
        return "true" if v else "false"
    if isinstance(v, (int, np.integer)):
        return str(int(v))
    if isinstance(v, (float, np.floating)):
        return str(float(v))
    if isinstance(v, str):
        return f'"{v}"'
    return str(v)
