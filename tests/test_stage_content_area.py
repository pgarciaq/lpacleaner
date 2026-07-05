"""Tests for Stage 6: Content Area Detection."""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from lpacleaner.config import Config
from lpacleaner.pipeline import PipelineState
from tests.conftest import make_music_page, make_text_page


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _setup_stage_input(tmp_path: Path, images: dict[str, np.ndarray]) -> Path:
    """Write images into an input directory and return its path."""
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    for name, img in images.items():
        cv2.imwrite(str(input_dir / name), img)
    return input_dir


def _make_bordered_page(
    width: int = 800,
    height: int = 600,
    border_color: tuple[int, int, int] = (0, 0, 200),
    bg_color: tuple[int, int, int] = (230, 220, 200),
    border_inset: float = 0.08,
    border_thickness: int = 3,
) -> np.ndarray:
    """Create a page with a visible border frame in the ink color.

    The border is drawn at ``border_inset`` fraction from each edge.
    Content (staff lines) is drawn inside the border.
    """
    img = np.full((height, width, 3), bg_color, dtype=np.uint8)

    mx = int(width * border_inset)
    my = int(height * border_inset)

    cv2.rectangle(img, (mx, my), (width - mx, height - my),
                  border_color, border_thickness)

    inner_mx = mx + 15
    inner_my = my + 15
    usable_h = height - 2 * inner_my
    num_lines = 16
    line_gap = usable_h // (num_lines + 1)
    for i in range(1, num_lines + 1):
        y = inner_my + i * line_gap
        cv2.line(img, (inner_mx, y), (width - inner_mx, y),
                 border_color, 2)

    return img


def _make_page_no_border(
    width: int = 800,
    height: int = 600,
    bg_color: tuple[int, int, int] = (230, 220, 200),
    ink_color: tuple[int, int, int] = (0, 0, 200),
) -> np.ndarray:
    """Create a page with content but no border frame."""
    img = np.full((height, width, 3), bg_color, dtype=np.uint8)

    margin = int(width * 0.12)
    usable_h = height - 2 * margin
    num_lines = 16
    gap = usable_h // (num_lines + 1)
    for i in range(1, num_lines + 1):
        y = margin + i * gap
        cv2.line(img, (margin, y), (width - margin, y), ink_color, 2)

    return img


# ---------------------------------------------------------------------------
# TestContentAreaStageContract
# ---------------------------------------------------------------------------

class TestContentAreaStageContract:
    """Verify BaseStage interface compliance."""

    def test_has_correct_name(self):
        from lpacleaner.stages.content_area import ContentAreaStage
        assert ContentAreaStage.name == "content_area"

    def test_has_correct_number(self):
        from lpacleaner.stages.content_area import ContentAreaStage
        assert ContentAreaStage.number == 6

    def test_has_correct_checkpoint_name(self):
        from lpacleaner.stages.content_area import ContentAreaStage
        assert ContentAreaStage.checkpoint_name == "06_content"

    def test_is_base_stage_subclass(self):
        from lpacleaner.pipeline import BaseStage
        from lpacleaner.stages.content_area import ContentAreaStage
        assert issubclass(ContentAreaStage, BaseStage)

    def test_error_class_is_skippable(self):
        from lpacleaner.stages.content_area import ContentAreaStage
        assert ContentAreaStage.error_class == "skippable"

    def test_registered_in_stage_registry(self):
        from lpacleaner.stages import STAGE_BY_NUMBER
        assert 6 in STAGE_BY_NUMBER
        assert STAGE_BY_NUMBER[6].name == "content_area"


# ---------------------------------------------------------------------------
# TestBorderDetection
# ---------------------------------------------------------------------------

class TestBorderDetection:
    """Test border frame detection via Hough lines."""

    def test_detects_border_frame(self):
        from lpacleaner.stages.content_area import ContentAreaStage

        stage = ContentAreaStage()
        img = _make_bordered_page()
        cfg = Config(input_dir=Path("/tmp"))

        _, meta = stage.process_image(img, {}, cfg)

        assert meta["method"] in ("hough_border", "ink_density")
        assert "content_rect" in meta
        x, y, w, h = meta["content_rect"]
        assert w > 0 and h > 0
        assert w < img.shape[1]
        assert h < img.shape[0]

    def test_border_rect_is_inside_image(self):
        from lpacleaner.stages.content_area import ContentAreaStage

        stage = ContentAreaStage()
        img = _make_bordered_page(width=1000, height=800)
        cfg = Config(input_dir=Path("/tmp"))

        _, meta = stage.process_image(img, {}, cfg)
        x, y, w, h = meta["content_rect"]
        assert x >= 0
        assert y >= 0
        assert x + w <= img.shape[1]
        assert y + h <= img.shape[0]

    def test_output_is_smaller_than_input(self):
        from lpacleaner.stages.content_area import ContentAreaStage

        stage = ContentAreaStage()
        img = _make_bordered_page()
        cfg = Config(input_dir=Path("/tmp"))

        result, meta = stage.process_image(img, {}, cfg)
        rh, rw = result.shape[:2]
        ih, iw = img.shape[:2]
        assert rw < iw, "Cropped output should be narrower"
        assert rh < ih, "Cropped output should be shorter"


# ---------------------------------------------------------------------------
# TestFallbackDetection
# ---------------------------------------------------------------------------

class TestFallbackDetection:
    """Test fallback methods when no border frame is found."""

    def test_no_border_uses_ink_density(self):
        from lpacleaner.stages.content_area import ContentAreaStage

        stage = ContentAreaStage()
        img = _make_page_no_border()
        cfg = Config(input_dir=Path("/tmp"), has_border_frame=False)

        _, meta = stage.process_image(img, {}, cfg)
        assert meta["method"] in ("ink_density", "inset_fallback")

    def test_uniform_image_uses_inset_fallback(self):
        from lpacleaner.stages.content_area import ContentAreaStage

        stage = ContentAreaStage()
        img = np.full((600, 800, 3), (230, 220, 200), dtype=np.uint8)
        cfg = Config(input_dir=Path("/tmp"))

        _, meta = stage.process_image(img, {}, cfg)
        assert meta["method"] == "inset_fallback"

    def test_inset_fallback_uses_config_fraction(self):
        from lpacleaner.stages.content_area import _inset_fallback

        x, y, w, h = _inset_fallback(1000, 800, 0.10)
        assert x == 80
        assert y == 100
        assert w == 640
        assert h == 800


# ---------------------------------------------------------------------------
# TestBlankPagePassthrough
# ---------------------------------------------------------------------------

class TestBlankPagePassthrough:
    """Blank pages should pass through unchanged."""

    def test_blank_page_passthrough(self):
        from lpacleaner.stages.content_area import ContentAreaStage

        stage = ContentAreaStage()
        img = np.full((600, 800, 3), 200, dtype=np.uint8)
        cfg = Config(input_dir=Path("/tmp"))

        result, meta = stage.process_image(img, {"page_type": "blank"}, cfg)

        assert meta["method"] == "passthrough"
        assert result.shape == img.shape
        np.testing.assert_array_equal(result, img)

    def test_blank_page_forwards_page_type(self):
        from lpacleaner.stages.content_area import ContentAreaStage

        stage = ContentAreaStage()
        img = np.full((600, 800, 3), 200, dtype=np.uint8)
        cfg = Config(input_dir=Path("/tmp"))

        _, meta = stage.process_image(img, {"page_type": "blank"}, cfg)
        assert meta["page_type"] == "blank"


# ---------------------------------------------------------------------------
# TestFeathering
# ---------------------------------------------------------------------------

class TestFeathering:
    """Test background masking with Gaussian feathering."""

    def test_feathering_replaces_outside_pixels(self):
        from lpacleaner.stages.content_area import _feather_outside

        img = np.full((100, 100, 3), (50, 50, 50), dtype=np.uint8)
        img[20:80, 20:80] = (200, 200, 200)

        result = _feather_outside(img, (20, 20, 60, 60), (128, 128, 128), sigma=0)

        corner = result[0, 0]
        assert all(c == 128 for c in corner), f"Corner should be bg color, got {corner}"

        center = result[50, 50]
        assert all(c == 200 for c in center), f"Center should be preserved, got {center}"

    def test_feathering_with_sigma_creates_gradient(self):
        from lpacleaner.stages.content_area import _feather_outside

        img = np.full((200, 200, 3), (100, 100, 100), dtype=np.uint8)
        bg = (200, 200, 200)

        result = _feather_outside(img, (50, 50, 100, 100), bg, sigma=10)

        edge = result[50, 50]
        corner = result[0, 0]
        assert corner[0] > edge[0], "Corner should be closer to bg color"


# ---------------------------------------------------------------------------
# TestMargins
# ---------------------------------------------------------------------------

class TestMargins:
    """Test margin padding."""

    def test_adds_margins(self):
        from lpacleaner.stages.content_area import _add_margins

        img = np.full((100, 200, 3), (150, 150, 150), dtype=np.uint8)
        padded, pad_px = _add_margins(img, (200, 200, 200), 0.05)

        assert pad_px == 10  # 200 * 0.05
        assert padded.shape[0] == 120  # 100 + 2*10
        assert padded.shape[1] == 220  # 200 + 2*10
        assert all(padded[0, 0, c] == 200 for c in range(3))

    def test_content_preserved_in_center(self):
        from lpacleaner.stages.content_area import _add_margins

        img = np.full((100, 200, 3), (42, 42, 42), dtype=np.uint8)
        padded, pad_px = _add_margins(img, (200, 200, 200), 0.05)

        center = padded[pad_px : pad_px + 100, pad_px : pad_px + 200]
        np.testing.assert_array_equal(center, img)


# ---------------------------------------------------------------------------
# TestMetadata
# ---------------------------------------------------------------------------

class TestMetadata:
    """Test metadata output."""

    def test_metadata_has_required_fields(self):
        from lpacleaner.stages.content_area import ContentAreaStage

        stage = ContentAreaStage()
        img = _make_bordered_page()
        cfg = Config(input_dir=Path("/tmp"))

        _, meta = stage.process_image(img, {}, cfg)

        assert meta["stage"] == "content_area"
        assert "method" in meta
        assert "content_rect" in meta
        assert "margin_px" in meta
        assert "background_color" in meta

    def test_forwards_page_type(self):
        from lpacleaner.stages.content_area import ContentAreaStage

        stage = ContentAreaStage()
        img = _make_bordered_page()
        cfg = Config(input_dir=Path("/tmp"))

        _, meta = stage.process_image(img, {"page_type": "music"}, cfg)
        assert meta["page_type"] == "music"

    def test_content_rect_is_four_ints(self):
        from lpacleaner.stages.content_area import ContentAreaStage

        stage = ContentAreaStage()
        img = _make_bordered_page()
        cfg = Config(input_dir=Path("/tmp"))

        _, meta = stage.process_image(img, {}, cfg)
        rect = meta["content_rect"]
        assert len(rect) == 4
        assert all(isinstance(v, int) for v in rect)


# ---------------------------------------------------------------------------
# TestContentAreaStageRun
# ---------------------------------------------------------------------------

class TestContentAreaStageRun:
    """Integration tests via BaseStage.run()."""

    def test_produces_checkpoint_directory(self, tmp_path):
        from lpacleaner.stages.content_area import ContentAreaStage

        img = _make_bordered_page()
        input_dir = _setup_stage_input(tmp_path, {"IMG_0001.png": img})
        cfg = Config(input_dir=input_dir, output_dir=tmp_path)
        state = PipelineState(tmp_path)
        stage = ContentAreaStage()

        stage.run(input_dir, tmp_path, cfg, state)

        checkpoint = tmp_path / "06_content"
        assert checkpoint.is_dir()
        assert (checkpoint / "IMG_0001.png").exists()

    def test_processes_multiple_images(self, tmp_path):
        from lpacleaner.stages.content_area import ContentAreaStage

        img1 = _make_bordered_page()
        img2 = _make_page_no_border()
        input_dir = _setup_stage_input(tmp_path, {
            "IMG_0001.png": img1,
            "IMG_0002.png": img2,
        })
        cfg = Config(input_dir=input_dir, output_dir=tmp_path)
        state = PipelineState(tmp_path)
        stage = ContentAreaStage()

        result = stage.run(input_dir, tmp_path, cfg, state)

        assert result.processed == 2
        assert (tmp_path / "06_content" / "IMG_0001.png").exists()
        assert (tmp_path / "06_content" / "IMG_0002.png").exists()

    def test_writes_metadata_sidecar(self, tmp_path):
        from lpacleaner.stages.content_area import ContentAreaStage

        img = _make_bordered_page()
        input_dir = _setup_stage_input(tmp_path, {"IMG_0001.png": img})
        cfg = Config(input_dir=input_dir, output_dir=tmp_path)
        state = PipelineState(tmp_path)
        stage = ContentAreaStage()

        stage.run(input_dir, tmp_path, cfg, state)

        sidecar = tmp_path / "06_content" / "IMG_0001.json"
        assert sidecar.exists()
        meta = json.loads(sidecar.read_text())
        assert meta["stage"] == "content_area"
        assert "content_rect" in meta

    def test_resume_skips_completed(self, tmp_path):
        from lpacleaner.stages.content_area import ContentAreaStage

        img = _make_bordered_page()
        input_dir = _setup_stage_input(tmp_path, {
            "IMG_0001.png": img,
            "IMG_0002.png": img,
        })
        cfg = Config(input_dir=input_dir, output_dir=tmp_path)
        state = PipelineState(tmp_path)
        stage = ContentAreaStage()

        stage.run(input_dir, tmp_path, cfg, state)
        result2 = stage.run(input_dir, tmp_path, cfg, state)

        assert result2.skipped == 2
        assert result2.processed == 0

    def test_returns_stage_result(self, tmp_path):
        from lpacleaner.stages.content_area import ContentAreaStage

        img = _make_bordered_page()
        input_dir = _setup_stage_input(tmp_path, {"IMG_0001.png": img})
        cfg = Config(input_dir=input_dir, output_dir=tmp_path)
        state = PipelineState(tmp_path)
        stage = ContentAreaStage()

        result = stage.run(input_dir, tmp_path, cfg, state)

        assert result.stage_name == "content_area"
        assert result.processed == 1
        assert result.failed == 0

    def test_reads_sidecar_from_previous_stage(self, tmp_path):
        """Stage 6 should read page_type from Stage 5 sidecar."""
        from lpacleaner.stages.content_area import ContentAreaStage

        img = _make_bordered_page()
        input_dir = _setup_stage_input(tmp_path, {"IMG_0001.png": img})

        sidecar = input_dir / "IMG_0001.json"
        sidecar.write_text(json.dumps({
            "stage": "perspective",
            "page_type": "music",
        }))

        cfg = Config(input_dir=input_dir, output_dir=tmp_path)
        state = PipelineState(tmp_path)
        stage = ContentAreaStage()

        stage.run(input_dir, tmp_path, cfg, state)

        out_sidecar = tmp_path / "06_content" / "IMG_0001.json"
        meta = json.loads(out_sidecar.read_text())
        assert meta["page_type"] == "music"


# ---------------------------------------------------------------------------
# TestConfig
# ---------------------------------------------------------------------------

class TestContentAreaConfig:
    """Test config defaults and TOML loading."""

    def test_default_config_values(self):
        cfg = Config(input_dir=Path("/tmp"))
        assert cfg.content_detect_inset_fallback == 0.05
        assert cfg.content_margin_padding == 0.02
        assert cfg.content_feather_sigma == 20

    def test_config_from_toml(self, tmp_path):
        toml_file = tmp_path / "book.toml"
        toml_file.write_text("""
[content_area]
inset_fallback = 0.08
margin_padding = 0.03
feather_sigma = 30
""")
        cfg = Config.from_toml(input_dir=tmp_path, toml_path=toml_file)
        assert cfg.content_detect_inset_fallback == 0.08
        assert cfg.content_margin_padding == 0.03
        assert cfg.content_feather_sigma == 30
