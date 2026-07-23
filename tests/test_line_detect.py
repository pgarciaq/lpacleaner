"""TDD tests for ghh.utils.line_detect -- ink detection, staff lines, foxing filter."""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from ghh.config import Config
from ghh.utils.line_detect import (
    StaffLine,
    detect_dominant_angle,
    detect_illustration_regions,
    detect_ink_mask,
    detect_ink_mask_geometric,
    detect_staff_lines,
)
from tests.conftest import make_music_page, make_text_page


@pytest.fixture
def default_cfg(tmp_path) -> Config:
    """Config with default ink parameters (red staff lines, hue=5)."""
    return Config(input_dir=tmp_path)


@pytest.fixture
def brown_cfg(tmp_path) -> Config:
    """Config tuned for brown/sepia ink (hue=15, wider range)."""
    return Config(
        input_dir=tmp_path,
        staff_color_hue=15,
        staff_color_range=20,
        staff_saturation_min=30,
        staff_value_min=60,
    )


@pytest.fixture
def red_music_page() -> np.ndarray:
    """800x600 page with red staff lines (default conftest params)."""
    return make_music_page(staff_color=(0, 0, 200))


@pytest.fixture
def brown_music_page() -> np.ndarray:
    """800x600 page with brown/sepia staff lines."""
    return make_music_page(staff_color=(30, 70, 140))


# ---------------------------------------------------------------------------
# TestDetectInkMask
# ---------------------------------------------------------------------------

class TestDetectInkMask:
    """Tests for detect_ink_mask(): color-based ink isolation."""

    def test_detects_red_staff_lines(self, red_music_page, default_cfg):
        mask = detect_ink_mask(red_music_page, default_cfg)
        assert mask.dtype == np.uint8
        assert mask.shape[:2] == red_music_page.shape[:2]
        ink_ratio = np.count_nonzero(mask) / mask.size
        assert ink_ratio > 0.005, "Should detect at least some red ink pixels"

    def test_detects_brown_staff_lines(self, brown_music_page, brown_cfg):
        mask = detect_ink_mask(brown_music_page, brown_cfg)
        ink_ratio = np.count_nonzero(mask) / mask.size
        assert ink_ratio > 0.005, "Should detect at least some brown ink pixels"

    def test_ignores_background(self, red_music_page, default_cfg):
        mask = detect_ink_mask(red_music_page, default_cfg)
        # Background is beige (230, 220, 200) -- should not be in the mask
        bg_region = red_music_page[0:10, 0:10]
        mask_region = mask[0:10, 0:10]
        assert np.count_nonzero(mask_region) == 0, "Background should not be detected as ink"

    def test_fallback_to_channel_difference(self, tmp_path):
        """When HSV thresholds produce too few pixels, channel-difference
        fallback should still detect red ink."""
        img = make_music_page(staff_color=(0, 0, 200))
        # Keep hue correct (red) but set impossibly strict sat/value so HSV finds nothing
        cfg = Config(
            input_dir=tmp_path,
            staff_color_hue=5,
            staff_color_range=2,      # very narrow range
            staff_saturation_min=254,  # nearly impossible to satisfy
            staff_value_min=254,
            channel_diff_rg=25,
            channel_diff_rb=25,
        )
        mask = detect_ink_mask(img, cfg)
        ink_ratio = np.count_nonzero(mask) / mask.size
        assert ink_ratio > 0.001, "Channel-difference fallback should find some red ink"

    def test_returns_binary_mask(self, red_music_page, default_cfg):
        mask = detect_ink_mask(red_music_page, default_cfg)
        unique = set(np.unique(mask))
        assert unique <= {0, 255}, "Mask should contain only 0 and 255"

    def test_handles_grayscale_input(self, default_cfg):
        """Should handle grayscale images without crashing (return empty mask)."""
        gray = np.full((100, 100), 128, dtype=np.uint8)
        mask = detect_ink_mask(gray, default_cfg)
        assert mask.shape == (100, 100)


# ---------------------------------------------------------------------------
# TestDetectInkMaskGeometric
# ---------------------------------------------------------------------------

class TestDetectInkMaskGeometric:
    """Tests for detect_ink_mask_geometric(): R9 foxing discrimination."""

    def test_preserves_horizontal_lines(self, red_music_page, default_cfg):
        mask = detect_ink_mask_geometric(red_music_page, default_cfg)
        ink_ratio = np.count_nonzero(mask) / mask.size
        assert ink_ratio > 0.002, "Horizontal staff lines should be preserved"

    def test_filters_round_foxing_spots(self, default_cfg):
        """Round spots (simulating foxing) should be filtered out."""
        img = np.full((400, 600, 3), (230, 220, 200), dtype=np.uint8)

        # Draw round spots in staff-line color (would pass color filter)
        for cx, cy in [(100, 100), (200, 200), (300, 150)]:
            cv2.circle(img, (cx, cy), 8, (0, 0, 200), -1)

        # Draw one long horizontal line (should be kept)
        cv2.line(img, (50, 300), (550, 300), (0, 0, 200), 2)

        mask = detect_ink_mask_geometric(img, default_cfg)

        # Check that the round spots region is mostly empty
        spot_region = mask[90:110, 90:110]
        spot_pixels = np.count_nonzero(spot_region)
        assert spot_pixels < 20, f"Round spots should be filtered: found {spot_pixels} pixels"

        # Check that the horizontal line region has pixels
        line_region = mask[298:302, 100:500]
        line_pixels = np.count_nonzero(line_region)
        assert line_pixels > 50, f"Horizontal line should be preserved: found {line_pixels} pixels"

    def test_geometric_mask_is_sparser_than_color_mask(self, red_music_page, default_cfg):
        color_mask = detect_ink_mask(red_music_page, default_cfg)
        geo_mask = detect_ink_mask_geometric(red_music_page, default_cfg)
        # Geometric filtering should produce fewer pixels than raw color mask
        assert np.count_nonzero(geo_mask) <= np.count_nonzero(color_mask), (
            "Geometric mask should have fewer or equal pixels to color mask"
        )


# ---------------------------------------------------------------------------
# TestDetectStaffLines
# ---------------------------------------------------------------------------

class TestDetectStaffLines:
    """Tests for detect_staff_lines(): Hough-based staff line detection."""

    def test_finds_expected_number_of_lines(self, red_music_page, default_cfg):
        lines = detect_staff_lines(red_music_page, default_cfg)
        assert isinstance(lines, list)
        # 4 staves x 5 lines = 20 staff lines, but clustering may merge close ones
        # We expect at least some lines detected
        assert len(lines) >= 4, f"Expected at least 4 staff lines, got {len(lines)}"

    def test_returns_staff_line_objects(self, red_music_page, default_cfg):
        lines = detect_staff_lines(red_music_page, default_cfg)
        assert len(lines) > 0
        line = lines[0]
        assert isinstance(line, StaffLine)
        assert hasattr(line, "y_center")
        assert hasattr(line, "angle")
        assert hasattr(line, "polynomial_coeffs")

    def test_returns_polynomial_coefficients(self, red_music_page, default_cfg):
        lines = detect_staff_lines(red_music_page, default_cfg)
        assert len(lines) > 0
        for line in lines:
            assert line.polynomial_coeffs is not None
            assert len(line.polynomial_coeffs) > 0

    def test_returns_empty_for_text_page(self, default_cfg):
        text = make_text_page()
        lines = detect_staff_lines(text, default_cfg)
        # Text page has dark lines but they won't match the red ink color
        assert len(lines) == 0, "Text page should have no red staff lines"

    def test_staff_lines_are_sorted_by_y(self, red_music_page, default_cfg):
        lines = detect_staff_lines(red_music_page, default_cfg)
        if len(lines) > 1:
            y_centers = [l.y_center for l in lines]
            assert y_centers == sorted(y_centers), "Staff lines should be sorted top-to-bottom"

    def test_lines_have_reasonable_angles(self, red_music_page, default_cfg):
        """Staff lines on an un-skewed page should be near-horizontal."""
        lines = detect_staff_lines(red_music_page, default_cfg)
        for line in lines:
            assert abs(line.angle) < 15, f"Angle {line.angle} too steep for horizontal staff"


# ---------------------------------------------------------------------------
# TestDetectDominantAngle
# ---------------------------------------------------------------------------

class TestDetectDominantAngle:
    """Tests for detect_dominant_angle(): median angle of detected staff lines."""

    def test_returns_near_zero_for_unrotated_page(self, red_music_page, default_cfg):
        angle = detect_dominant_angle(red_music_page, default_cfg)
        assert isinstance(angle, float)
        assert abs(angle) < 2.0, f"Expected near-zero angle, got {angle}"

    def test_detects_skew(self, default_cfg):
        skewed = make_music_page(skew_deg=3.0)
        angle = detect_dominant_angle(skewed, default_cfg)
        # Should detect approximately 3 degrees of skew (sign may vary
        # depending on rotation direction convention)
        assert abs(abs(angle) - 3.0) < 2.0, f"Expected ~3 degrees magnitude, got {angle}"

    def test_returns_zero_for_text_page(self, default_cfg):
        text = make_text_page()
        angle = detect_dominant_angle(text, default_cfg)
        assert angle == 0.0, "No staff lines means angle should be 0"

    def test_quad_none_works_same_as_no_quad(self, red_music_page, default_cfg):
        angle_no_quad = detect_dominant_angle(red_music_page, default_cfg)
        angle_none = detect_dominant_angle(red_music_page, default_cfg, quad_corners=None)
        assert angle_no_quad == angle_none

    def test_quad_mask_excludes_background_lines(self, default_cfg):
        """Lines outside the quad should be ignored when quad is provided."""
        h, w = 800, 600
        img = np.full((h, w, 3), (230, 220, 200), dtype=np.uint8)

        # Staff lines inside the quad region (center of image)
        for y in range(200, 600, 30):
            cv2.line(img, (100, y), (500, y), (0, 0, 200), 2)

        # Spurious skewed lines in the corner (outside the quad)
        for y in range(50, 150, 20):
            cv2.line(img, (10, y), (200, y + 40), (0, 0, 200), 2)

        quad = np.array([[80, 180], [520, 180], [520, 620], [80, 620]], dtype=np.float32)

        angle_masked = detect_dominant_angle(img, default_cfg, quad_corners=quad)
        assert abs(angle_masked) < 3.0, (
            f"Masked detection should find near-horizontal lines, got {angle_masked}"
        )

    def test_full_image_quad_same_as_unmasked(self, red_music_page, default_cfg):
        """A quad covering the full image should produce the same result."""
        h, w = red_music_page.shape[:2]
        quad = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.float32)

        angle_unmasked = detect_dominant_angle(red_music_page, default_cfg)
        angle_full_quad = detect_dominant_angle(red_music_page, default_cfg, quad_corners=quad)
        assert abs(angle_unmasked - angle_full_quad) < 0.5


# ---------------------------------------------------------------------------
# TestDetectIllustrationRegions
# ---------------------------------------------------------------------------

class TestDetectIllustrationRegions:
    """Tests for detect_illustration_regions(): R4 multi-colored region masking."""

    def test_returns_mask_same_size(self, red_music_page, default_cfg):
        mask = detect_illustration_regions(red_music_page, default_cfg)
        assert mask.shape[:2] == red_music_page.shape[:2]
        assert mask.dtype == np.uint8

    def test_no_illustrations_on_music_page(self, red_music_page, default_cfg):
        mask = detect_illustration_regions(red_music_page, default_cfg)
        illust_ratio = np.count_nonzero(mask) / mask.size
        assert illust_ratio < 0.05, "Plain music page should have few illustration pixels"

    def test_detects_colorful_region(self, default_cfg):
        """A region with many different hues should be flagged as illustration."""
        img = np.full((400, 600, 3), (230, 220, 200), dtype=np.uint8)

        # Paint a colorful block (simulating an illustration)
        for i in range(100):
            for j in range(100):
                hue = (i * 2 + j * 3) % 180
                img[150 + i, 250 + j] = cv2.cvtColor(
                    np.array([[[hue, 200, 180]]], dtype=np.uint8),
                    cv2.COLOR_HSV2BGR
                )[0, 0]

        mask = detect_illustration_regions(img, default_cfg)
        illust_in_region = np.count_nonzero(mask[150:250, 250:350])
        assert illust_in_region > 500, "Colorful region should be flagged as illustration"
