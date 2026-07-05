"""TDD tests for stages/analyze.py -- auto-detect book characteristics.

Tests the analyze command's ability to:
- Sample images adaptively
- Detect coarse orientation
- Discover ink color
- Analyze layout, photography conditions, physical condition
- Write valid book.toml
- Round-trip through Config.from_toml()
"""

from __future__ import annotations

import sys
from pathlib import Path

if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib  # type: ignore[import-not-found]

import cv2
import numpy as np
import pytest

from tests.conftest import make_music_page, make_page_on_background, make_text_page


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_book_dir(tmp_path: Path, n_images: int = 20, **page_kwargs) -> Path:
    """Create a directory with N synthetic book images on dark background."""
    book_dir = tmp_path / "book_photos"
    book_dir.mkdir()
    for i in range(n_images):
        page = make_music_page(width=400, height=300, **page_kwargs)
        photo = make_page_on_background(page, bg=(35, 28, 22), border=40)
        cv2.imwrite(str(book_dir / f"IMG_{i:04d}.jpg"), photo,
                    [cv2.IMWRITE_JPEG_QUALITY, 90])
    return book_dir


def _make_rotated_book_dir(tmp_path: Path, n_images: int = 10, rotation: int = 90) -> Path:
    """Create images that need coarse rotation correction."""
    book_dir = tmp_path / "rotated_book"
    book_dir.mkdir()
    for i in range(n_images):
        page = make_music_page(width=400, height=300)
        photo = make_page_on_background(page, bg=(35, 28, 22), border=40)
        if rotation == 90:
            photo = cv2.rotate(photo, cv2.ROTATE_90_CLOCKWISE)
        elif rotation == 180:
            photo = cv2.rotate(photo, cv2.ROTATE_180)
        elif rotation == 270:
            photo = cv2.rotate(photo, cv2.ROTATE_90_COUNTERCLOCKWISE)
        cv2.imwrite(str(book_dir / f"IMG_{i:04d}.jpg"), photo,
                    [cv2.IMWRITE_JPEG_QUALITY, 90])
    return book_dir


def _make_book_with_brown_ink(tmp_path: Path, n_images: int = 15) -> Path:
    """Create a book directory with brown staff lines."""
    book_dir = tmp_path / "brown_book"
    book_dir.mkdir()
    brown_bgr = (30, 50, 100)
    for i in range(n_images):
        page = make_music_page(width=400, height=300, staff_color=brown_bgr)
        photo = make_page_on_background(page, bg=(35, 28, 22), border=40)
        cv2.imwrite(str(book_dir / f"IMG_{i:04d}.jpg"), photo,
                    [cv2.IMWRITE_JPEG_QUALITY, 90])
    return book_dir


# ---------------------------------------------------------------------------
# TestAnalyzeIntegration
# ---------------------------------------------------------------------------

class TestAnalyzeIntegration:
    """End-to-end tests for the analyze command."""

    def test_produces_book_toml(self, tmp_path):
        from ghh.stages.analyze import run_analyze

        book_dir = _make_book_dir(tmp_path)
        output_dir = tmp_path / "output"
        output_dir.mkdir()

        run_analyze(book_dir, output_dir)

        toml_path = output_dir / "book.toml"
        assert toml_path.exists()

    def test_book_toml_has_required_sections(self, tmp_path):
        from ghh.stages.analyze import run_analyze

        book_dir = _make_book_dir(tmp_path)
        output_dir = tmp_path / "output"
        output_dir.mkdir()

        run_analyze(book_dir, output_dir)

        with open(output_dir / "book.toml", "rb") as f:
            data = tomllib.load(f)

        assert "ink" in data
        assert "layout" in data
        assert "photography" in data
        assert "condition" in data
        assert "pipeline" in data

    def test_book_toml_round_trips_through_config(self, tmp_path):
        from ghh.config import Config
        from ghh.stages.analyze import run_analyze

        book_dir = _make_book_dir(tmp_path)
        output_dir = tmp_path / "output"
        output_dir.mkdir()

        run_analyze(book_dir, output_dir)

        cfg = Config.from_toml(
            input_dir=book_dir,
            toml_path=output_dir / "book.toml",
        )
        assert isinstance(cfg, Config)
        assert cfg.staff_color_hue >= 0
        assert cfg.profile == "full"

    def test_graceful_with_few_images(self, tmp_path):
        """With <3 valid samples, should fall back to defaults."""
        from ghh.stages.analyze import run_analyze

        book_dir = tmp_path / "tiny_book"
        book_dir.mkdir()
        page = make_music_page(width=200, height=150)
        cv2.imwrite(str(book_dir / "IMG_0001.jpg"), page)

        output_dir = tmp_path / "output"
        output_dir.mkdir()

        run_analyze(book_dir, output_dir)

        assert (output_dir / "book.toml").exists()


# ---------------------------------------------------------------------------
# TestInkDiscovery
# ---------------------------------------------------------------------------

class TestInkDiscovery:
    """Test ink color detection from sampled pages."""

    def test_detects_red_staff_ink(self, tmp_path):
        from ghh.stages.analyze import _discover_ink_color

        pages = []
        for i in range(5):
            page = make_music_page(width=400, height=300, staff_color=(0, 0, 200))
            pages.append(page)

        result = _discover_ink_color(pages)

        assert result["staff_color_hue"] < 20 or result["staff_color_hue"] > 160

    def test_detects_brown_staff_ink(self, tmp_path):
        from ghh.stages.analyze import _discover_ink_color

        brown_bgr = (30, 50, 100)
        pages = []
        for i in range(5):
            page = make_music_page(width=400, height=300, staff_color=brown_bgr)
            pages.append(page)

        result = _discover_ink_color(pages)

        assert 5 < result["staff_color_hue"] < 30

    def test_returns_expected_keys(self, tmp_path):
        from ghh.stages.analyze import _discover_ink_color

        pages = [make_music_page(width=400, height=300) for _ in range(5)]
        result = _discover_ink_color(pages)

        assert "staff_color_hue" in result
        assert "staff_color_range" in result
        assert "staff_saturation_min" in result
        assert "staff_value_min" in result
        assert "channel_diff_rg" in result
        assert "channel_diff_rb" in result


# ---------------------------------------------------------------------------
# TestCoarseOrientation
# ---------------------------------------------------------------------------

class TestCoarseOrientation:
    """Test coarse (90°) orientation detection."""

    def test_upright_returns_zero(self, tmp_path):
        from ghh.stages.analyze import _detect_coarse_orientation

        pages = []
        for i in range(3):
            page = make_music_page(width=400, height=300)
            photo = make_page_on_background(page, bg=(35, 28, 22), border=40)
            pages.append(photo)

        offset = _detect_coarse_orientation(pages)
        assert offset == 0

    def test_detects_90_degree_rotation(self, tmp_path):
        from ghh.stages.analyze import _detect_coarse_orientation

        pages = []
        for i in range(3):
            page = make_music_page(width=400, height=300)
            photo = make_page_on_background(page, bg=(35, 28, 22), border=40)
            rotated = cv2.rotate(photo, cv2.ROTATE_90_CLOCKWISE)
            pages.append(rotated)

        offset = _detect_coarse_orientation(pages)
        assert offset in (90, 270)

    def test_handles_no_staff_lines_gracefully(self, tmp_path):
        from ghh.stages.analyze import _detect_coarse_orientation

        pages = [np.full((300, 400, 3), (200, 200, 200), dtype=np.uint8)
                 for _ in range(3)]

        offset = _detect_coarse_orientation(pages)
        assert offset == 0


# ---------------------------------------------------------------------------
# TestLayoutAnalysis
# ---------------------------------------------------------------------------

class TestLayoutAnalysis:
    """Test layout feature detection."""

    def test_counts_staff_lines(self, tmp_path):
        from ghh.stages.analyze import _analyze_layout

        pages = [make_music_page(width=400, height=300, num_staves=4)
                 for _ in range(5)]

        result = _analyze_layout(pages)

        assert "expected_staff_lines_per_page" in result
        assert result["expected_staff_lines_per_page"] > 0

    def test_detects_border_frame(self, tmp_path):
        from ghh.stages.analyze import _analyze_layout

        pages = [make_music_page(width=400, height=300)
                 for _ in range(5)]

        result = _analyze_layout(pages)

        assert "has_border_frame" in result

    def test_computes_aspect_ratio(self, tmp_path):
        from ghh.stages.analyze import _analyze_layout

        pages = [make_music_page(width=400, height=300)
                 for _ in range(5)]

        result = _analyze_layout(pages)

        assert "median_aspect_ratio" in result
        assert result["median_aspect_ratio"] > 0


# ---------------------------------------------------------------------------
# TestPhotographyCondition
# ---------------------------------------------------------------------------

class TestPhotographyCondition:
    """Test photography condition detection."""

    def test_detects_hotspots(self, tmp_path):
        from ghh.stages.analyze import _analyze_photography
        from tests.conftest import add_hotspot

        pages = []
        for i in range(5):
            page = make_music_page(width=400, height=300)
            if i < 3:
                page = add_hotspot(page, center=(200, 150), radius=30)
            pages.append(page)

        result = _analyze_photography(pages)

        assert "has_flash_hotspots" in result
        assert result["has_flash_hotspots"] is True

    def test_clean_images_no_hotspots(self, tmp_path):
        from ghh.stages.analyze import _analyze_photography

        pages = [make_music_page(width=400, height=300) for _ in range(5)]

        result = _analyze_photography(pages)

        assert result["has_flash_hotspots"] is False

    def test_returns_expected_keys(self, tmp_path):
        from ghh.stages.analyze import _analyze_photography

        pages = [make_music_page(width=400, height=300) for _ in range(5)]

        result = _analyze_photography(pages)

        assert "has_flash_hotspots" in result
        assert "color_cast_detected" in result
        assert "background_contrast" in result
        assert "fingers_detected" in result


# ---------------------------------------------------------------------------
# TestPhysicalCondition
# ---------------------------------------------------------------------------

class TestPhysicalCondition:
    """Test physical condition analysis."""

    def test_pristine_page_reports_none(self, tmp_path):
        from ghh.stages.analyze import _analyze_condition

        pages = [make_music_page(width=400, height=300) for _ in range(5)]

        result = _analyze_condition(pages)

        assert result["foxing_severity"] == "none"
        assert result["stain_severity"] == "none"
        assert result["salt_deposits"] == "none"

    def test_returns_expected_keys(self, tmp_path):
        from ghh.stages.analyze import _analyze_condition

        pages = [make_music_page(width=400, height=300) for _ in range(5)]

        result = _analyze_condition(pages)

        for key in ("stain_severity", "ink_fading", "show_through_severity",
                    "foxing_severity", "iron_gall_halos", "salt_deposits"):
            assert key in result
            assert result[key] in ("none", "mild", "slight", "moderate", "severe")
