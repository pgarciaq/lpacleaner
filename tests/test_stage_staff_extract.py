"""Tests for Stage 7: Staff Extract."""

from __future__ import annotations

import cv2
import numpy as np

from ghh.config import Config
from ghh.stages.staff_extract import StaffExtractStage, _detect_horizontal_lines

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_staff_page(
    width: int = 800,
    height: int = 600,
    ink_color: tuple[int, int, int] = (0, 0, 200),
    bg_color: tuple[int, int, int] = (230, 220, 200),
    num_lines: int = 16,
    line_thickness: int = 2,
) -> np.ndarray:
    """Synthetic page with horizontal lines spanning most of the width."""
    img = np.full((height, width, 3), bg_color, dtype=np.uint8)
    margin_y = int(height * 0.08)
    margin_x = int(width * 0.08)
    usable_h = height - 2 * margin_y
    gap = usable_h // (num_lines + 1)
    for i in range(1, num_lines + 1):
        y = margin_y + i * gap
        cv2.line(img, (margin_x, y), (width - margin_x, y),
                 ink_color, line_thickness)
    return img


def _red_cfg(tmp_path) -> Config:
    """Config tuned for red ink detection."""
    return Config(
        input_dir=tmp_path,
        staff_color_hue=5,
        staff_color_range=15,
        staff_saturation_min=40,
        staff_value_min=80,
    )


def _black_cfg(tmp_path) -> Config:
    """Config with default (black-ish) ink parameters."""
    return Config(input_dir=tmp_path)


# ---------------------------------------------------------------------------
# Stage contract
# ---------------------------------------------------------------------------

class TestStageContract:
    def test_name(self):
        assert StaffExtractStage.name == "staff_extract"

    def test_number(self):
        assert StaffExtractStage.number == 7

    def test_checkpoint_name(self):
        assert StaffExtractStage.checkpoint_name == "07_staff_extract"

    def test_registered(self):
        from ghh.stages import STAGE_BY_NAME, STAGE_BY_NUMBER
        assert "staff_extract" in STAGE_BY_NAME
        assert 7 in STAGE_BY_NUMBER

    def test_config_keys_include_color_params(self):
        expected = {
            "staff_color_hue",
            "staff_color_range",
            "staff_saturation_min",
            "staff_value_min",
        }
        assert expected.issubset(set(StaffExtractStage.config_keys))


# ---------------------------------------------------------------------------
# Color detection path
# ---------------------------------------------------------------------------

class TestColorDetection:
    def test_detects_red_lines(self, tmp_path):
        """Red staff lines on beige parchment are found via the color path."""
        img = _make_staff_page(ink_color=(0, 0, 200))
        cfg = _red_cfg(tmp_path)
        stage = StaffExtractStage()
        result, meta = stage.process_image(img, {}, cfg)

        assert meta["staff_extract_action"] == "cropped"
        assert meta["staff_extract_method"] == "color"
        assert meta["staff_extract_lines_found"] > 0
        assert result.shape[0] < img.shape[0]

    def test_detect_horizontal_lines_color(self, tmp_path):
        """Low-level helper returns color method for colored ink."""
        img = _make_staff_page(ink_color=(0, 0, 200))
        cfg = _red_cfg(tmp_path)
        h, w = img.shape[:2]
        morph, method = _detect_horizontal_lines(img, h, w, cfg)
        assert method == "color"
        assert np.count_nonzero(morph) > 0


# ---------------------------------------------------------------------------
# Grayscale fallback
# ---------------------------------------------------------------------------

class TestGrayscaleFallback:
    def test_detects_black_lines(self, tmp_path):
        """Black staff lines on white paper are found via the grayscale fallback."""
        img = _make_staff_page(
            ink_color=(30, 30, 30),
            bg_color=(250, 250, 250),
        )
        cfg = _black_cfg(tmp_path)
        stage = StaffExtractStage()
        result, meta = stage.process_image(img, {}, cfg)

        assert meta["staff_extract_action"] == "cropped"
        assert meta["staff_extract_method"] == "grayscale"
        assert result.shape[0] < img.shape[0]

    def test_detect_horizontal_lines_grayscale(self, tmp_path):
        """Low-level helper falls back to grayscale when color mask is empty."""
        img = _make_staff_page(
            ink_color=(30, 30, 30),
            bg_color=(250, 250, 250),
        )
        cfg = _black_cfg(tmp_path)
        h, w = img.shape[:2]
        morph, method = _detect_horizontal_lines(img, h, w, cfg)
        assert method == "grayscale"
        assert np.count_nonzero(morph) > 0


# ---------------------------------------------------------------------------
# Passthrough behaviour
# ---------------------------------------------------------------------------

class TestPassthrough:
    def test_blank_page_passes_through(self, tmp_path):
        """Page with no content passes through unchanged."""
        img = np.full((600, 800, 3), (230, 220, 200), dtype=np.uint8)
        cfg = _red_cfg(tmp_path)
        stage = StaffExtractStage()
        result, meta = stage.process_image(img, {}, cfg)

        assert meta["staff_extract_action"] == "passthrough"
        assert np.array_equal(result, img)

    def test_vertical_lines_pass_through(self, tmp_path):
        """Vertical lines (not staff) do not trigger cropping."""
        img = np.full((600, 800, 3), (230, 220, 200), dtype=np.uint8)
        for x in range(100, 700, 60):
            cv2.line(img, (x, 50), (x, 550), (0, 0, 200), 2)
        cfg = _red_cfg(tmp_path)
        stage = StaffExtractStage()
        result, meta = stage.process_image(img, {}, cfg)

        assert meta["staff_extract_action"] == "passthrough"

    def test_short_lines_pass_through(self, tmp_path):
        """Lines shorter than 30% of width are filtered out."""
        img = np.full((600, 800, 3), (230, 220, 200), dtype=np.uint8)
        for y in range(100, 500, 30):
            cv2.line(img, (300, y), (450, y), (0, 0, 200), 2)
        cfg = _red_cfg(tmp_path)
        stage = StaffExtractStage()
        result, meta = stage.process_image(img, {}, cfg)

        assert meta["staff_extract_action"] == "passthrough"


# ---------------------------------------------------------------------------
# Metadata fields
# ---------------------------------------------------------------------------

class TestMetadata:
    def test_cropped_metadata_fields(self, tmp_path):
        img = _make_staff_page(ink_color=(0, 0, 200))
        cfg = _red_cfg(tmp_path)
        stage = StaffExtractStage()
        _, meta = stage.process_image(img, {}, cfg)

        assert "staff_extract_action" in meta
        assert "staff_extract_method" in meta
        assert "staff_extract_y_start" in meta
        assert "staff_extract_y_end" in meta
        assert "staff_extract_lines_found" in meta
        assert "staff_extract_coverage" in meta
        assert 0 < meta["staff_extract_coverage"] <= 1.0

    def test_passthrough_metadata_fields(self, tmp_path):
        img = np.full((600, 800, 3), (230, 220, 200), dtype=np.uint8)
        cfg = _red_cfg(tmp_path)
        stage = StaffExtractStage()
        _, meta = stage.process_image(img, {}, cfg)

        assert meta["staff_extract_action"] == "passthrough"
        assert "staff_extract_method" in meta
        assert "staff_extract_reason" in meta


# ---------------------------------------------------------------------------
# is_unchanged
# ---------------------------------------------------------------------------

class TestIsUnchanged:
    def test_passthrough_is_unchanged(self):
        stage = StaffExtractStage()
        assert stage.is_unchanged({"staff_extract_action": "passthrough"})

    def test_cropped_is_not_unchanged(self):
        stage = StaffExtractStage()
        assert not stage.is_unchanged({"staff_extract_action": "cropped"})
