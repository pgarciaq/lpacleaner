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

from tests.conftest import make_music_page, make_page_on_background

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
# TestLinePixelDiscovery
# ---------------------------------------------------------------------------

class TestLinePixelDiscovery:
    """Test color-agnostic line pixel detection for ink discovery."""

    def test_finds_staff_lines(self):
        from ghh.stages.analyze import _discover_line_pixels

        page = make_music_page(width=600, height=400, num_staves=4)
        mask = _discover_line_pixels(page)

        assert np.count_nonzero(mask) > 100

    def test_blank_page_returns_sparse_mask(self):
        from ghh.stages.analyze import _discover_line_pixels

        blank = np.full((400, 600, 3), (220, 215, 200), dtype=np.uint8)
        mask = _discover_line_pixels(blank)

        assert np.count_nonzero(mask) < 200

    def test_works_on_grayscale(self):
        from ghh.stages.analyze import _discover_line_pixels

        gray = np.full((400, 600), 200, dtype=np.uint8)
        cv2.line(gray, (0, 100), (600, 100), 50, 2)
        cv2.line(gray, (0, 200), (600, 200), 50, 2)
        cv2.line(gray, (0, 300), (600, 300), 50, 2)
        mask = _discover_line_pixels(gray)

        assert np.count_nonzero(mask) > 0


# ---------------------------------------------------------------------------
# TestLayoutAnalysis
# ---------------------------------------------------------------------------

class TestLayoutAnalysis:
    """Test layout feature detection."""

    def test_counts_staff_lines(self, tmp_path):
        from ghh.stages.analyze import _analyze_layout, _default_ink

        pages = [make_music_page(width=400, height=300, num_staves=4)
                 for _ in range(5)]

        result = _analyze_layout(pages, _default_ink())

        assert "expected_staff_lines_per_page" in result
        assert result["expected_staff_lines_per_page"] > 0

    def test_detects_border_frame(self, tmp_path):
        from ghh.stages.analyze import _analyze_layout, _default_ink

        pages = [make_music_page(width=400, height=300)
                 for _ in range(5)]

        result = _analyze_layout(pages, _default_ink())

        assert "has_border_frame" in result

    def test_detects_colored_border_frame(self, tmp_path):
        """Border frames drawn in colored ink (red) are detected."""
        from ghh.stages.analyze import _analyze_layout, _default_ink

        pages = [make_music_page(
            width=400, height=300,
            staff_color=(0, 0, 200),
        ) for _ in range(5)]

        result = _analyze_layout(pages, _default_ink())
        assert result["has_border_frame"] is True

    def test_colored_border_frame_via_has_border_frame(self, tmp_path):
        """Direct test of _has_border_frame with colored ink config."""
        from ghh.config import Config
        from ghh.stages.analyze import _has_border_frame

        img = make_music_page(
            width=400, height=300,
            staff_color=(0, 0, 200),
        )
        cfg = Config(
            input_dir=tmp_path,
            staff_color_hue=5,
            staff_color_range=15,
            staff_saturation_min=40,
            staff_value_min=80,
        )
        assert _has_border_frame(img, cfg) is True

    def test_no_border_returns_false(self, tmp_path):
        """Page without a border frame returns False."""
        from ghh.config import Config
        from ghh.stages.analyze import _has_border_frame

        img = np.full((300, 400, 3), (230, 220, 200), dtype=np.uint8)
        cfg = Config(
            input_dir=tmp_path,
            staff_color_hue=5,
            staff_color_range=15,
            staff_saturation_min=40,
            staff_value_min=80,
        )
        assert _has_border_frame(img, cfg) is False

    def test_computes_aspect_ratio(self, tmp_path):
        from ghh.stages.analyze import _analyze_layout, _default_ink

        pages = [make_music_page(width=400, height=300)
                 for _ in range(5)]

        result = _analyze_layout(pages, _default_ink())

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
        from ghh.stages.analyze import _analyze_condition, _default_ink

        pages = [make_music_page(width=400, height=300) for _ in range(5)]

        result = _analyze_condition(pages, _default_ink())

        assert result["foxing_severity"] == "none"
        assert result["stain_severity"] == "none"
        assert result["salt_deposits"] == "none"

    def test_returns_expected_keys(self, tmp_path):
        from ghh.stages.analyze import _analyze_condition, _default_ink

        pages = [make_music_page(width=400, height=300) for _ in range(5)]

        result = _analyze_condition(pages, _default_ink())

        for key in ("stain_severity", "ink_fading", "show_through_severity",
                    "foxing_severity", "iron_gall_halos", "salt_deposits"):
            assert key in result
            assert result[key] in ("none", "mild", "slight", "moderate", "severe")
