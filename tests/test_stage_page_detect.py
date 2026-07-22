"""Tests for Stage 4 (PageDetectStage): page detection and cropping.

Covers the BaseStage contract, quad detection via the fallback chain,
quad refinement, page type classification, and integration tests.
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from ghh.config import Config
from ghh.pipeline import BaseStage, PipelineState, StageResult

from tests.conftest import (
    make_music_page,
    make_page_on_background,
    make_text_page,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _save_test_image(path: Path, img: np.ndarray) -> None:
    cv2.imwrite(str(path), img)


def _setup_stage_input(tmp_path: Path, images: dict[str, np.ndarray]) -> Path:
    input_dir = tmp_path / "02_oriented"
    input_dir.mkdir()
    for name, img in images.items():
        _save_test_image(input_dir / name, img)
    return input_dir


def _make_blank_page(
    width: int = 800,
    height: int = 600,
    bg_color: tuple[int, int, int] = (240, 235, 225),
) -> np.ndarray:
    """Uniform bright page with very low variance."""
    return np.full((height, width, 3), bg_color, dtype=np.uint8)


# ---------------------------------------------------------------------------
# TestPageDetectStageContract
# ---------------------------------------------------------------------------

class TestPageDetectStageContract:
    """Verify that PageDetectStage satisfies the BaseStage contract."""

    def test_has_correct_name(self):
        from ghh.stages.page_detect import PageDetectStage

        assert PageDetectStage().name == "page_detect"

    def test_has_correct_number(self):
        from ghh.stages.page_detect import PageDetectStage

        assert PageDetectStage().number == 4

    def test_has_correct_checkpoint_name(self):
        from ghh.stages.page_detect import PageDetectStage

        assert PageDetectStage().checkpoint_name == "04_page_detected"

    def test_is_base_stage_subclass(self):
        from ghh.stages.page_detect import PageDetectStage

        assert issubclass(PageDetectStage, BaseStage)

    def test_error_class_is_skippable(self):
        from ghh.stages.page_detect import PageDetectStage

        assert PageDetectStage().error_class == "skippable"

    def test_is_mandatory_stage(self):
        cfg = Config(input_dir=Path("/tmp"))
        assert cfg.should_skip_stage("page_detect") is False

    def test_registered_in_stage_registry(self):
        from ghh.stages import STAGE_BY_NUMBER, STAGE_BY_NAME

        assert 4 in STAGE_BY_NUMBER
        assert "page_detect" in STAGE_BY_NAME


# ---------------------------------------------------------------------------
# TestPageDetection
# ---------------------------------------------------------------------------

class TestPageDetection:
    """Test quad detection on synthetic images."""

    def test_detects_page_on_dark_background(self):
        """A light page on a dark background should be found by Otsu."""
        from ghh.stages.page_detect import PageDetectStage

        stage = PageDetectStage()
        page = make_music_page(width=600, height=400)
        photo = make_page_on_background(page, border=100)
        cfg = Config(input_dir=Path("/tmp"))

        result, meta = stage.process_image(photo, {}, cfg)

        assert meta["method"] in ("otsu", "otsu_inverted", "canny", "adaptive")
        assert meta["page_type"] == "music"
        assert "quad_corners" in meta
        corners = np.array(meta["quad_corners"])
        assert corners.shape == (4, 2)

    def test_detects_text_page_on_dark_background(self):
        """A text page should be detected and classified as text."""
        from ghh.stages.page_detect import PageDetectStage

        stage = PageDetectStage()
        page = make_text_page(width=600, height=400)
        photo = make_page_on_background(page, border=100)
        cfg = Config(input_dir=Path("/tmp"))

        result, meta = stage.process_image(photo, {}, cfg)

        assert meta["method"] != "full_image"
        assert meta["page_type"] in ("text", "other")

    def test_full_image_passes_through(self):
        """The full image should pass through unchanged (no crop)."""
        from ghh.stages.page_detect import PageDetectStage

        stage = PageDetectStage()
        page = make_music_page(width=600, height=400)
        photo = make_page_on_background(page, border=120)
        cfg = Config(input_dir=Path("/tmp"))

        result, meta = stage.process_image(photo, {}, cfg)

        assert result.shape == photo.shape

    def test_quad_corners_are_ordered(self):
        """Corners should be in TL, TR, BR, BL order."""
        from ghh.stages.page_detect import PageDetectStage

        stage = PageDetectStage()
        page = make_music_page(width=600, height=400)
        photo = make_page_on_background(page, border=100)
        cfg = Config(input_dir=Path("/tmp"))

        _, meta = stage.process_image(photo, {}, cfg)
        corners = np.array(meta["quad_corners"])

        tl, tr, br, bl = corners
        assert tl[0] < tr[0], "TL should be left of TR"
        assert tl[1] < bl[1], "TL should be above BL"
        assert tr[1] < br[1], "TR should be above BR"

    def test_quad_corners_within_image_bounds(self):
        """Quad corners should be within the image bounds."""
        from ghh.stages.page_detect import PageDetectStage

        stage = PageDetectStage()
        page = make_music_page(width=600, height=400)
        photo = make_page_on_background(page, border=100)
        cfg = Config(input_dir=Path("/tmp"))

        result, meta = stage.process_image(photo, {}, cfg)
        corners = np.array(meta["quad_corners"])
        rh, rw = result.shape[:2]

        assert np.all(corners[:, 0] >= 0)
        assert np.all(corners[:, 1] >= 0)
        assert np.all(corners[:, 0] <= rw)
        assert np.all(corners[:, 1] <= rh)

    def test_uniform_image_still_produces_quad(self):
        """A uniform image should still produce a valid quad (possibly full_image)."""
        from ghh.stages.page_detect import PageDetectStage

        stage = PageDetectStage()
        uniform = np.full((400, 600, 3), (128, 128, 128), dtype=np.uint8)
        cfg = Config(input_dir=Path("/tmp"))

        result, meta = stage.process_image(uniform, {}, cfg)

        assert "quad_corners" in meta
        corners = np.array(meta["quad_corners"])
        assert corners.shape == (4, 2)
        assert result.shape == uniform.shape

    def test_quad_covers_page_region(self):
        """The detected quad should cover the page area, not just a sliver."""
        from ghh.stages.page_detect import PageDetectStage

        stage = PageDetectStage()
        page = make_music_page(width=600, height=400)
        photo = make_page_on_background(page, border=100)
        cfg = Config(input_dir=Path("/tmp"))

        _, meta = stage.process_image(photo, {}, cfg)
        corners = np.array(meta["quad_corners"])
        quad_area = cv2.contourArea(corners.astype(np.float32))
        img_area = photo.shape[0] * photo.shape[1]

        assert quad_area > img_area * 0.3

    def test_respects_forced_method(self):
        """When page_detect_method is set, only that method is tried."""
        from ghh.stages.page_detect import PageDetectStage

        stage = PageDetectStage()
        page = make_music_page(width=600, height=400)
        photo = make_page_on_background(page, border=100)
        cfg = Config(input_dir=Path("/tmp"), page_detect_method="otsu")

        _, meta = stage.process_image(photo, {}, cfg)

        assert meta["method"] in ("otsu", "full_image")


# ---------------------------------------------------------------------------
# TestQuadRefinement
# ---------------------------------------------------------------------------

class TestQuadExpansion:
    """Test quad expansion logic."""

    def test_expand_pushes_corners_outward(self):
        from ghh.stages.page_detect import _expand_quad

        quad = np.array(
            [[100, 100], [500, 100], [500, 400], [100, 400]], dtype=np.float32,
        )
        expanded = _expand_quad(quad, 600, 800, 0.05)

        assert expanded[0][0] < quad[0][0], "TL should move left"
        assert expanded[0][1] < quad[0][1], "TL should move up"
        assert expanded[2][0] > quad[2][0], "BR should move right"
        assert expanded[2][1] > quad[2][1], "BR should move down"

    def test_expand_clamps_to_image_bounds(self):
        from ghh.stages.page_detect import _expand_quad

        quad = np.array(
            [[5, 5], [795, 5], [795, 595], [5, 595]], dtype=np.float32,
        )
        expanded = _expand_quad(quad, 600, 800, 0.10)

        assert np.all(expanded[:, 0] >= 0)
        assert np.all(expanded[:, 1] >= 0)
        assert np.all(expanded[:, 0] <= 799)
        assert np.all(expanded[:, 1] <= 599)

    def test_expand_zero_frac_is_noop(self):
        from ghh.stages.page_detect import _expand_quad

        quad = np.array(
            [[100, 100], [500, 100], [500, 400], [100, 400]], dtype=np.float32,
        )
        result = _expand_quad(quad, 600, 800, 0.0)
        np.testing.assert_array_equal(result, quad)

    def test_expand_skips_boundary_clipped_corners(self):
        """Corners already on the image boundary should not be expanded."""
        from ghh.stages.page_detect import _expand_quad

        quad = np.array(
            [[0, 0], [500, 100], [500, 400], [100, 400]], dtype=np.float32,
        )
        expanded = _expand_quad(quad, 600, 800, 0.05)

        np.testing.assert_array_equal(expanded[0], quad[0])
        assert expanded[1][0] > quad[1][0], "Interior TR should expand right"
        assert expanded[2][0] > quad[2][0], "Interior BR should expand right"
        assert expanded[3][0] < quad[3][0], "Interior BL should expand left"

    def test_expand_skips_all_boundary_corners(self):
        """When all corners are on the boundary, none should be expanded."""
        from ghh.stages.page_detect import _expand_quad

        quad = np.array(
            [[0, 0], [799, 0], [799, 599], [0, 599]], dtype=np.float32,
        )
        expanded = _expand_quad(quad, 600, 800, 0.05)

        np.testing.assert_array_equal(expanded, quad)


class TestQuadRefinement:
    """Test contour-to-quad refinement logic."""

    def test_refine_rectangle_contour(self):
        """A rectangle contour should yield a 4-point quad."""
        from ghh.stages.page_detect import _refine_to_quad

        contour = np.array([
            [[100, 100]], [[500, 100]], [[500, 400]], [[100, 400]]
        ], dtype=np.int32)
        cfg = Config(input_dir=Path("/tmp"))

        quad = _refine_to_quad(contour, cfg)
        assert quad.shape == (4, 2)

    def test_refine_complex_contour(self):
        """A many-sided contour should still produce a 4-point quad."""
        from ghh.stages.page_detect import _refine_to_quad

        angles = np.linspace(0, 2 * np.pi, 20, endpoint=False)
        pts = np.column_stack([
            300 + 200 * np.cos(angles),
            250 + 150 * np.sin(angles),
        ]).astype(np.int32)
        contour = pts.reshape(-1, 1, 2)
        cfg = Config(input_dir=Path("/tmp"))

        quad = _refine_to_quad(contour, cfg)
        assert quad.shape == (4, 2)


# ---------------------------------------------------------------------------
# TestPageClassification
# ---------------------------------------------------------------------------

class TestPageClassification:
    """Test page type classification."""

    def test_music_page_classified_as_music(self):
        """A page with staff lines should be classified as music."""
        from ghh.stages.page_detect import _classify_page_type

        page = make_music_page(width=800, height=600)
        quad = np.array(
            [[0, 0], [799, 0], [799, 599], [0, 599]], dtype=np.float32,
        )
        cfg = Config(input_dir=Path("/tmp"))

        assert _classify_page_type(page, quad, cfg) == "music"

    def test_text_page_classified_as_text(self):
        """A text-only page should be classified as text."""
        from ghh.stages.page_detect import _classify_page_type

        page = make_text_page(width=800, height=600)
        quad = np.array(
            [[0, 0], [799, 0], [799, 599], [0, 599]], dtype=np.float32,
        )
        cfg = Config(input_dir=Path("/tmp"))

        result = _classify_page_type(page, quad, cfg)
        assert result in ("text", "other")

    def test_blank_page_classified_as_blank(self):
        """A uniform bright page should be classified as blank."""
        from ghh.stages.page_detect import _classify_page_type

        page = _make_blank_page(width=800, height=600)
        quad = np.array(
            [[0, 0], [799, 0], [799, 599], [0, 599]], dtype=np.float32,
        )
        cfg = Config(input_dir=Path("/tmp"))

        assert _classify_page_type(page, quad, cfg) == "blank"


# ---------------------------------------------------------------------------
# TestPageDetectConfig
# ---------------------------------------------------------------------------

class TestPageDetectConfig:
    """Test page_detect configuration fields."""

    def test_default_config_values(self):
        cfg = Config(input_dir=Path("/tmp"))
        assert cfg.page_detect_method == "auto"
        assert cfg.page_detect_morph_kernel == 50
        assert cfg.page_detect_epsilon == 0.02
        assert cfg.page_detect_min_area_frac == 0.30
        assert cfg.page_detect_padding == 10

    def test_config_from_toml(self, tmp_path):
        toml_content = b"""
[page_detect]
method = "otsu"
morph_kernel = 30
epsilon = 0.04
min_area_frac = 0.25
padding = 20
"""
        toml_file = tmp_path / "book.toml"
        toml_file.write_bytes(toml_content)

        cfg = Config.from_toml(tmp_path, toml_path=toml_file)
        assert cfg.page_detect_method == "otsu"
        assert cfg.page_detect_morph_kernel == 30
        assert cfg.page_detect_epsilon == 0.04
        assert cfg.page_detect_min_area_frac == 0.25
        assert cfg.page_detect_padding == 20


# ---------------------------------------------------------------------------
# TestPageDetectStageRun (integration)
# ---------------------------------------------------------------------------

@pytest.mark.slow
class TestPageDetectStageRun:
    """Integration tests for PageDetectStage.run()."""

    def test_produces_checkpoint_directory(self, tmp_path):
        from ghh.stages.page_detect import PageDetectStage

        page = make_music_page(width=600, height=400)
        photo = make_page_on_background(page, border=80)
        input_dir = _setup_stage_input(tmp_path, {"IMG_0001.png": photo})
        cfg = Config(input_dir=input_dir, output_dir=tmp_path)
        state = PipelineState(tmp_path)
        stage = PageDetectStage()

        stage.run(input_dir, tmp_path, cfg, state)

        assert (tmp_path / "04_page_detected").exists()

    def test_processes_multiple_images(self, tmp_path):
        from ghh.stages.page_detect import PageDetectStage

        page = make_music_page(width=600, height=400)
        photo = make_page_on_background(page, border=80)
        input_dir = _setup_stage_input(tmp_path, {
            "IMG_0001.png": photo,
            "IMG_0002.png": photo,
            "IMG_0003.png": photo,
        })
        cfg = Config(input_dir=input_dir, output_dir=tmp_path)
        state = PipelineState(tmp_path)
        stage = PageDetectStage()

        result = stage.run(input_dir, tmp_path, cfg, state)

        assert result.processed == 3
        assert result.failed == 0
        out_files = list((tmp_path / "04_page_detected").glob("*.png"))
        assert len(out_files) == 3

    def test_writes_metadata_sidecar(self, tmp_path):
        from ghh.stages.page_detect import PageDetectStage

        page = make_music_page(width=600, height=400)
        photo = make_page_on_background(page, border=80)
        input_dir = _setup_stage_input(tmp_path, {"IMG_0001.png": photo})
        cfg = Config(input_dir=input_dir, output_dir=tmp_path)
        state = PipelineState(tmp_path)
        stage = PageDetectStage()

        stage.run(input_dir, tmp_path, cfg, state)

        sidecar = tmp_path / "04_page_detected" / "IMG_0001.json"
        assert sidecar.exists()
        meta = json.loads(sidecar.read_text())
        assert meta["stage"] == "page_detect"
        assert "quad_corners" in meta
        assert "page_type" in meta
        assert "method" in meta

    def test_resume_skips_completed(self, tmp_path):
        from ghh.stages.page_detect import PageDetectStage

        page = make_music_page(width=600, height=400)
        photo = make_page_on_background(page, border=80)
        input_dir = _setup_stage_input(tmp_path, {
            "IMG_0001.png": photo,
            "IMG_0002.png": photo,
        })
        cfg = Config(input_dir=input_dir, output_dir=tmp_path)
        state = PipelineState(tmp_path)
        stage = PageDetectStage()

        stage.run(input_dir, tmp_path, cfg, state)
        result2 = stage.run(input_dir, tmp_path, cfg, state)

        assert result2.skipped == 2
        assert result2.processed == 0

    def test_perspective_skew_detected(self, tmp_path):
        """Pages with perspective skew should still be detected."""
        from ghh.stages.page_detect import PageDetectStage

        page = make_music_page(width=600, height=400)
        photo = make_page_on_background(page, border=100, perspective_skew=0.03)
        input_dir = _setup_stage_input(tmp_path, {"IMG_0001.png": photo})
        cfg = Config(input_dir=input_dir, output_dir=tmp_path)
        state = PipelineState(tmp_path)
        stage = PageDetectStage()

        result = stage.run(input_dir, tmp_path, cfg, state)

        assert result.processed == 1
        assert result.failed == 0

        sidecar = tmp_path / "04_page_detected" / "IMG_0001.json"
        meta = json.loads(sidecar.read_text())
        assert meta["method"] != "full_image"

    def test_minimize_diskspace_creates_symlinks(self, tmp_path):
        """With minimize_diskspace, Stage 4 should symlink images, not copy."""
        from ghh.stages.page_detect import PageDetectStage

        page = make_music_page(width=600, height=400)
        photo = make_page_on_background(page, border=80)
        input_dir = _setup_stage_input(tmp_path, {"IMG_0001.png": photo})
        cfg = Config(
            input_dir=input_dir, output_dir=tmp_path,
            minimize_diskspace=True,
        )
        state = PipelineState(tmp_path)
        stage = PageDetectStage()

        result = stage.run(input_dir, tmp_path, cfg, state)

        assert result.processed == 1
        out_img = tmp_path / "04_page_detected" / "IMG_0001.png"
        assert out_img.is_symlink(), "Image should be a symlink"
        assert out_img.resolve() == (input_dir / "IMG_0001.png").resolve()
        sidecar = tmp_path / "04_page_detected" / "IMG_0001.json"
        assert sidecar.exists(), "Sidecar should still be written"


# ---------------------------------------------------------------------------
# Helpers for border detection tests
# ---------------------------------------------------------------------------

def _make_page_with_red_borders(
    width: int = 600,
    height: int = 800,
    border: int = 80,
    line_thickness: int = 3,
    red_bgr: tuple[int, int, int] = (0, 0, 200),
    parchment_bgr: tuple[int, int, int] = (210, 215, 225),
    bg_bgr: tuple[int, int, int] = (40, 30, 25),
    draw_horizontal: bool = True,
) -> np.ndarray:
    """Create a synthetic photo of a page with red border lines.

    Returns a BGR image of size (height + 2*border, width + 2*border)
    with a parchment rectangle surrounded by dark background and red
    border lines inside the parchment area.
    """
    out_h = height + 2 * border
    out_w = width + 2 * border
    canvas = np.full((out_h, out_w, 3), bg_bgr, dtype=np.uint8)

    # Parchment area
    canvas[border:border + height, border:border + width] = parchment_bgr

    # Red vertical border lines (inside the parchment, near left/right edges)
    inset = 10
    lx = border + inset
    rx = border + width - inset
    cv2.line(canvas, (lx, border), (lx, border + height), red_bgr, line_thickness)
    cv2.line(canvas, (rx, border), (rx, border + height), red_bgr, line_thickness)

    if draw_horizontal:
        ty = border + inset
        by = border + height - inset
        cv2.line(canvas, (border, ty), (border + width, ty), red_bgr, line_thickness)
        cv2.line(canvas, (border, by), (border + width, by), red_bgr, line_thickness)

    return canvas


def _make_page_with_fore_edge(
    width: int = 600,
    height: int = 800,
    border: int = 80,
    fore_edge_width: int = 60,
) -> np.ndarray:
    """Page with red borders plus a bright fore-edge strip on the right."""
    img = _make_page_with_red_borders(width, height, border)
    parchment_color = (210, 215, 225)
    # Add a bright fore-edge strip to the right of the page
    fe_x = border + width
    fe_w = min(fore_edge_width, img.shape[1] - fe_x)
    if fe_w > 0:
        img[border:border + height, fe_x:fe_x + fe_w] = parchment_color
    return img


# ---------------------------------------------------------------------------
# TestBorderDetection
# ---------------------------------------------------------------------------

class TestBorderDetection:
    """Test red page border line detection (#70)."""

    def test_detects_red_vertical_lines(self):
        """Should detect left and right vertical red borders."""
        from ghh.stages.page_detect import _detect_page_borders

        img = _make_page_with_red_borders()
        cfg = Config(input_dir=Path("/tmp"))

        borders = _detect_page_borders(img, cfg)

        assert borders.left_x is not None, "Should detect left border"
        assert borders.right_x is not None, "Should detect right border"
        assert borders.confidence >= 0.5

    def test_border_positions_are_reasonable(self):
        """Detected border positions should be near the actual red lines."""
        from ghh.stages.page_detect import _detect_page_borders

        border = 80
        width = 600
        inset = 10
        expected_left = border + inset
        expected_right = border + width - inset

        img = _make_page_with_red_borders(width=width, border=border)
        cfg = Config(input_dir=Path("/tmp"))

        borders = _detect_page_borders(img, cfg)

        assert borders.left_x is not None
        assert borders.right_x is not None
        assert abs(borders.left_x - expected_left) < 15
        assert abs(borders.right_x - expected_right) < 15

    def test_detects_horizontal_borders(self):
        """Should detect top and bottom horizontal red borders."""
        from ghh.stages.page_detect import _detect_page_borders

        img = _make_page_with_red_borders(draw_horizontal=True)
        cfg = Config(input_dir=Path("/tmp"))

        borders = _detect_page_borders(img, cfg)

        assert borders.top_y is not None or borders.bottom_y is not None, \
            "Should detect at least one horizontal border"

    def test_no_borders_returns_low_confidence(self):
        """Image without red lines should return low confidence."""
        from ghh.stages.page_detect import _detect_page_borders

        # Plain parchment page with black text lines (no red)
        page = np.full((400, 600, 3), (210, 215, 225), dtype=np.uint8)
        for y in range(50, 380, 40):
            cv2.line(page, (30, y), (570, y), (0, 0, 0), 1)
        photo = make_page_on_background(page, border=80)
        cfg = Config(input_dir=Path("/tmp"))

        borders = _detect_page_borders(photo, cfg)

        assert borders.confidence < 0.5

    def test_ignores_red_initials(self):
        """Red decorative elements (short, not near edges) should not
        be detected as page borders."""
        from ghh.stages.page_detect import _detect_page_borders

        # Image with a red rectangle in the middle (simulating an initial)
        # but no border lines
        img = np.full((800, 600, 3), (210, 215, 225), dtype=np.uint8)
        cv2.rectangle(img, (200, 300), (280, 400), (0, 0, 200), -1)
        cfg = Config(input_dir=Path("/tmp"))

        borders = _detect_page_borders(img, cfg)

        assert borders.left_x is None
        assert borders.right_x is None
        assert borders.confidence < 0.5

    def test_partial_borders_left_only(self):
        """If only the left border line is present, should detect it."""
        from ghh.stages.page_detect import _detect_page_borders

        border = 80
        width = 600
        height = 800
        img = np.full((height + 2 * border, width + 2 * border, 3),
                       (40, 30, 25), dtype=np.uint8)
        parchment = (210, 215, 225)
        img[border:border + height, border:border + width] = parchment
        # Only left border line
        lx = border + 10
        cv2.line(img, (lx, border), (lx, border + height), (0, 0, 200), 3)
        cfg = Config(input_dir=Path("/tmp"))

        borders = _detect_page_borders(img, cfg)

        assert borders.left_x is not None
        assert borders.confidence > 0

    def test_faded_red_lines_detected(self):
        """Faded red (lower saturation, lower value) should still be found."""
        from ghh.stages.page_detect import _detect_page_borders

        faded_red = (80, 80, 150)
        img = _make_page_with_red_borders(red_bgr=faded_red)
        cfg = Config(input_dir=Path("/tmp"))

        borders = _detect_page_borders(img, cfg)

        assert borders.left_x is not None or borders.right_x is not None


# ---------------------------------------------------------------------------
# TestBorderRefinement
# ---------------------------------------------------------------------------

class TestBorderRefinement:
    """Test quad refinement using detected border lines (#71)."""

    def test_clips_quad_to_borders(self):
        """Quad extending past borders should be clipped inward."""
        from ghh.stages.page_detect import PageBorders, _refine_quad_with_borders

        quad = np.array(
            [[10, 10], [700, 10], [700, 500], [10, 500]], dtype=np.float32,
        )
        borders = PageBorders(left_x=50.0, right_x=650.0, confidence=0.8)

        refined, applied = _refine_quad_with_borders(quad, borders, 600, 800, 0.5)

        assert applied
        assert refined[0][0] >= 47, "TL.x should clip to left border"
        assert refined[3][0] >= 47, "BL.x should clip to left border"
        assert refined[1][0] <= 653, "TR.x should clip to right border"
        assert refined[2][0] <= 653, "BR.x should clip to right border"

    def test_leaves_tight_quad_unchanged(self):
        """Quad already inside borders should not be changed."""
        from ghh.stages.page_detect import PageBorders, _refine_quad_with_borders

        quad = np.array(
            [[60, 30], [640, 30], [640, 480], [60, 480]], dtype=np.float32,
        )
        borders = PageBorders(left_x=50.0, right_x=650.0, confidence=0.8)

        refined, applied = _refine_quad_with_borders(quad, borders, 600, 800, 0.5)

        assert not applied
        np.testing.assert_array_equal(refined, quad)

    def test_partial_border_refinement(self):
        """Only the left border clips the left edge; right stays unchanged."""
        from ghh.stages.page_detect import PageBorders, _refine_quad_with_borders

        quad = np.array(
            [[10, 10], [700, 10], [700, 500], [10, 500]], dtype=np.float32,
        )
        borders = PageBorders(left_x=50.0, right_x=None, confidence=0.8)

        refined, applied = _refine_quad_with_borders(quad, borders, 600, 800, 0.5)

        assert applied
        assert refined[0][0] >= 47, "TL.x should clip to left border"
        assert refined[1][0] == 700, "TR.x should be unchanged (no right border)"

    def test_low_confidence_skips_refinement(self):
        """Below threshold, refinement is a no-op."""
        from ghh.stages.page_detect import PageBorders, _refine_quad_with_borders

        quad = np.array(
            [[10, 10], [700, 10], [700, 500], [10, 500]], dtype=np.float32,
        )
        borders = PageBorders(left_x=50.0, right_x=650.0, confidence=0.3)

        refined, applied = _refine_quad_with_borders(quad, borders, 600, 800, 0.5)

        assert not applied
        np.testing.assert_array_equal(refined, quad)

    def test_disabled_refinement(self):
        """Config flag disables refinement entirely."""
        from ghh.stages.page_detect import PageDetectStage

        stage = PageDetectStage()
        img = _make_page_with_red_borders()
        cfg = Config(input_dir=Path("/tmp"), page_detect_border_refinement=False)

        _, meta = stage.process_image(img, {}, cfg)

        assert "border_refinement" not in meta

    def test_refinement_metadata_present(self):
        """When refinement runs, metadata should include border_refinement."""
        from ghh.stages.page_detect import PageDetectStage

        stage = PageDetectStage()
        img = _make_page_with_red_borders()
        cfg = Config(input_dir=Path("/tmp"))

        _, meta = stage.process_image(img, {}, cfg)

        assert "border_refinement" in meta
        br = meta["border_refinement"]
        assert "confidence" in br
        assert "applied" in br
        assert "left_x" in br
        assert "right_x" in br

    def test_clips_all_four_sides(self):
        """When all four borders are detected, all sides should clip."""
        from ghh.stages.page_detect import PageBorders, _refine_quad_with_borders

        quad = np.array(
            [[5, 5], [795, 5], [795, 595], [5, 595]], dtype=np.float32,
        )
        borders = PageBorders(
            left_x=50.0, right_x=750.0,
            top_y=40.0, bottom_y=560.0,
            confidence=1.0,
        )

        refined, applied = _refine_quad_with_borders(quad, borders, 600, 800, 0.5)

        assert applied
        assert refined[0][0] >= 47, "TL.x clipped"
        assert refined[0][1] >= 37, "TL.y clipped"
        assert refined[2][0] <= 753, "BR.x clipped"
        assert refined[2][1] <= 563, "BR.y clipped"


# ---------------------------------------------------------------------------
# TestBorderRefinementIntegration (#72)
# ---------------------------------------------------------------------------

class TestBorderRefinementIntegration:
    """End-to-end integration tests for border-based quad refinement."""

    def test_fore_edge_excluded(self):
        """A page with a bright fore-edge strip should have it excluded."""
        from ghh.stages.page_detect import PageDetectStage

        stage = PageDetectStage()
        img = _make_page_with_fore_edge(
            width=600, height=800, border=80, fore_edge_width=60,
        )
        cfg = Config(input_dir=Path("/tmp"))

        _, meta = stage.process_image(img, {}, cfg)

        corners = np.array(meta["quad_corners"])
        page_right = 80 + 600

        tr_x = corners[1][0]
        br_x = corners[2][0]
        assert tr_x < page_right + 20, \
            f"TR.x={tr_x} should not extend far into fore-edge (page ends at {page_right})"
        assert br_x < page_right + 20, \
            f"BR.x={br_x} should not extend far into fore-edge"

    def test_no_regression_on_clean_page(self):
        """A page with borders on a clean dark background shouldn't degrade."""
        from ghh.stages.page_detect import PageDetectStage

        stage = PageDetectStage()
        img = _make_page_with_red_borders(width=600, height=800, border=80)
        cfg = Config(input_dir=Path("/tmp"))

        _, meta = stage.process_image(img, {}, cfg)

        corners = np.array(meta["quad_corners"])
        quad_area = cv2.contourArea(corners.astype(np.float32))
        page_area = 600 * 800

        assert quad_area > page_area * 0.5, \
            f"Quad area ({quad_area}) should cover most of the page ({page_area})"

    def test_refinement_applied_on_bordered_page(self):
        """Border refinement should be applied on a page with red borders."""
        from ghh.stages.page_detect import PageDetectStage

        stage = PageDetectStage()
        img = _make_page_with_red_borders()
        cfg = Config(input_dir=Path("/tmp"))

        _, meta = stage.process_image(img, {}, cfg)

        assert "border_refinement" in meta
        br = meta["border_refinement"]
        assert br["confidence"] >= 0.5

    def test_end_to_end_with_stage5(self):
        """Stage 4 + Stage 5 together should produce clean output."""
        from ghh.stages.page_detect import PageDetectStage
        from ghh.stages.perspective import PerspectiveStage

        page_detect = PageDetectStage()
        perspective = PerspectiveStage()

        img = _make_page_with_fore_edge(
            width=600, height=800, border=80, fore_edge_width=60,
        )
        cfg = Config(
            input_dir=Path("/tmp"),
            perspective_near_rect_threshold_deg=0.0,
            perspective_max_introduced_tilt_deg=90.0,
        )

        _, s4_meta = page_detect.process_image(img, {}, cfg)
        result, s5_meta = perspective.process_image(img, s4_meta, cfg)

        if s5_meta["method"] == "warpPerspective":
            rh, rw = result.shape[:2]
            assert rw < img.shape[1], \
                "Output should be narrower than input (fore-edge excluded)"
