"""Tests for Stage 5 (PerspectiveStage): perspective correction.

Covers the BaseStage contract, quad-to-rectangle warping, background
fill, pass-through on missing/degenerate quads, sidecar propagation
from Stage 4, and integration tests.
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
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _save_test_image(path: Path, img: np.ndarray) -> None:
    cv2.imwrite(str(path), img)


def _save_sidecar(path: Path, meta: dict) -> None:
    path.write_text(json.dumps(meta))


def _setup_stage4_output(
    tmp_path: Path,
    images: dict[str, tuple[np.ndarray, dict]],
) -> Path:
    """Create a fake Stage 4 output directory with images and sidecars."""
    input_dir = tmp_path / "04_page_detected"
    input_dir.mkdir()
    for name, (img, meta) in images.items():
        _save_test_image(input_dir / name, img)
        stem = Path(name).stem
        _save_sidecar(input_dir / f"{stem}.json", meta)
    return input_dir


def _make_skewed_page_with_quad(
    border: int = 80,
    skew: float = 0.04,
) -> tuple[np.ndarray, np.ndarray]:
    """Create a page on background with perspective skew, return (image, quad).

    The quad is the known corners of the page region within the photograph.
    """
    page = make_music_page(width=600, height=400)
    ph, pw = page.shape[:2]
    out_h = ph + 2 * border
    out_w = pw + 2 * border
    canvas = np.full((out_h, out_w, 3), (40, 30, 25), dtype=np.uint8)
    canvas[border:border + ph, border:border + pw] = page

    src = np.float32([
        [border, border],
        [border + pw, border],
        [border + pw, border + ph],
        [border, border + ph],
    ])
    s = skew
    dst = np.float32([
        [border + s * out_w, border + s * out_h],
        [border + pw - s * out_w, border],
        [border + pw, border + ph],
        [border, border + ph - s * out_h],
    ])
    M = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(
        canvas, M, (out_w, out_h),
        borderValue=(40, 30, 25),
    )
    return warped, dst


# ---------------------------------------------------------------------------
# TestPerspectiveStageContract
# ---------------------------------------------------------------------------

class TestPerspectiveStageContract:
    """Verify that PerspectiveStage satisfies the BaseStage contract."""

    def test_has_correct_name(self):
        from ghh.stages.perspective import PerspectiveStage

        assert PerspectiveStage().name == "perspective"

    def test_has_correct_number(self):
        from ghh.stages.perspective import PerspectiveStage

        assert PerspectiveStage().number == 5

    def test_has_correct_checkpoint_name(self):
        from ghh.stages.perspective import PerspectiveStage

        assert PerspectiveStage().checkpoint_name == "05_perspective"

    def test_is_base_stage_subclass(self):
        from ghh.stages.perspective import PerspectiveStage

        assert issubclass(PerspectiveStage, BaseStage)

    def test_error_class_is_skippable(self):
        from ghh.stages.perspective import PerspectiveStage

        assert PerspectiveStage().error_class == "skippable"

    def test_is_not_skippable_by_default(self):
        from ghh.stages.perspective import PerspectiveStage

        cfg = Config(input_dir=Path("/tmp"))
        assert PerspectiveStage().should_skip(cfg) is False

    def test_registered_in_stage_registry(self):
        from ghh.stages import STAGE_BY_NUMBER

        assert 5 in STAGE_BY_NUMBER
        assert STAGE_BY_NUMBER[5].name == "perspective"


# ---------------------------------------------------------------------------
# TestQuadValidation
# ---------------------------------------------------------------------------

class TestQuadValidation:
    """Test boundary detection, skew measurement, and crop ratio helpers."""

    def test_no_boundary_corners(self):
        from ghh.stages.perspective import _count_boundary_corners

        quad = np.array([[100, 100], [700, 100], [700, 500], [100, 500]], dtype=np.float32)
        assert _count_boundary_corners(quad, 600, 800) == 0

    def test_one_boundary_corner(self):
        from ghh.stages.perspective import _count_boundary_corners

        quad = np.array([[0, 100], [700, 100], [700, 500], [100, 500]], dtype=np.float32)
        assert _count_boundary_corners(quad, 600, 800) == 1

    def test_three_boundary_corners(self):
        from ghh.stages.perspective import _count_boundary_corners

        quad = np.array([[0, 0], [799, 100], [799, 500], [100, 500]], dtype=np.float32)
        assert _count_boundary_corners(quad, 600, 800) == 3

    def test_four_boundary_corners(self):
        from ghh.stages.perspective import _count_boundary_corners

        quad = np.array([[0, 0], [799, 0], [799, 599], [0, 599]], dtype=np.float32)
        assert _count_boundary_corners(quad, 600, 800) == 4

    def test_skew_horizontal_quad(self):
        from ghh.stages.perspective import _quad_skew_degrees

        quad = np.array([[0, 0], [600, 0], [600, 400], [0, 400]], dtype=np.float32)
        assert _quad_skew_degrees(quad) < 0.1

    def test_skew_tilted_quad(self):
        from ghh.stages.perspective import _quad_skew_degrees

        quad = np.array([[0, 0], [600, 50], [600, 450], [0, 400]], dtype=np.float32)
        skew = _quad_skew_degrees(quad)
        assert 4.0 < skew < 6.0

    def test_crop_ratio_full_image(self):
        from ghh.stages.perspective import _crop_ratio

        quad = np.array([[0, 0], [800, 0], [800, 600], [0, 600]], dtype=np.float32)
        assert _crop_ratio(quad, 600, 800) < 0.01

    def test_crop_ratio_half_image(self):
        from ghh.stages.perspective import _crop_ratio

        quad = np.array([[200, 150], [600, 150], [600, 450], [200, 450]], dtype=np.float32)
        ratio = _crop_ratio(quad, 600, 800)
        assert 0.4 < ratio < 0.8

    def test_should_skip_excessive_skew(self):
        from ghh.stages.perspective import _should_skip_warp

        result = _should_skip_warp(0, 6.0, 0.05, 5.0, 0.30)
        assert result == "excessive_skew"

    def test_should_skip_excessive_crop(self):
        from ghh.stages.perspective import _should_skip_warp

        result = _should_skip_warp(0, 1.0, 0.40, 5.0, 0.30)
        assert result == "excessive_crop"

    def test_should_not_skip_good_quad(self):
        from ghh.stages.perspective import _should_skip_warp

        result = _should_skip_warp(1, 1.5, 0.08, 5.0, 0.30)
        assert result is None

    def test_should_not_skip_boundary_corners_with_low_skew(self):
        from ghh.stages.perspective import _should_skip_warp

        result = _should_skip_warp(4, 3.0, 0.10, 5.0, 0.30)
        assert result is None

    def test_perspective_applies_output_padding(self):
        from ghh.stages.perspective import PerspectiveStage

        stage = PerspectiveStage()
        img = np.full((1000, 800, 3), 128, dtype=np.uint8)
        cfg = Config(input_dir=Path("/tmp"), perspective_output_padding_frac=0.02)

        quad = [[100, 100], [700, 100], [700, 900], [100, 900]]
        metadata = {"quad_corners": quad, "page_type": "music"}
        result, meta = stage.process_image(img, metadata, cfg)

        # Output should be larger than the quad-derived dimensions due to padding
        assert meta["dst_size"][0] == 600  # width
        assert meta["dst_size"][1] == 800  # height
        # Result should have padding applied (smaller effective content area)
        assert result.shape[1] == 600
        assert result.shape[0] == 800


class TestPerspectiveUnreliablePassthrough:
    """Test that unreliable quads trigger passthrough."""

    def test_high_skew_quad_passes_through(self):
        from ghh.stages.perspective import PerspectiveStage

        stage = PerspectiveStage()
        img = np.full((4000, 3000, 3), 128, dtype=np.uint8)
        cfg = Config(input_dir=Path("/tmp"))

        quad = [[100, 100], [2900, 500], [2900, 3900], [100, 3500]]
        metadata = {"quad_corners": quad, "page_type": "music"}
        result, meta = stage.process_image(img, metadata, cfg)

        np.testing.assert_array_equal(result, img)
        assert meta["method"] == "passthrough_unreliable"
        assert meta["skip_reason"] == "excessive_skew"

    def test_reliable_quad_applies_warp(self):
        from ghh.stages.perspective import PerspectiveStage

        stage = PerspectiveStage()
        img = np.full((1000, 800, 3), 128, dtype=np.uint8)
        cfg = Config(input_dir=Path("/tmp"))

        quad = [[20, 20], [780, 20], [780, 980], [20, 980]]
        metadata = {"quad_corners": quad}
        result, meta = stage.process_image(img, metadata, cfg)

        assert meta["method"] == "warpPerspective"


# ---------------------------------------------------------------------------
# TestPerspectivePassthrough
# ---------------------------------------------------------------------------

class TestPerspectivePassthrough:
    """Test pass-through when no quad or degenerate quad is provided."""

    def test_passthrough_when_no_quad(self):
        """Without quad_corners in metadata, image passes through unchanged."""
        from ghh.stages.perspective import PerspectiveStage

        stage = PerspectiveStage()
        img = make_music_page(width=600, height=400)
        cfg = Config(input_dir=Path("/tmp"))

        result, meta = stage.process_image(img, {}, cfg)

        np.testing.assert_array_equal(result, img)
        assert meta["method"] == "passthrough"

    def test_passthrough_when_quad_wrong_shape(self):
        """Malformed quad_corners should trigger pass-through."""
        from ghh.stages.perspective import PerspectiveStage

        stage = PerspectiveStage()
        img = make_music_page(width=600, height=400)
        cfg = Config(input_dir=Path("/tmp"))

        metadata = {"quad_corners": [[0, 0], [100, 0], [100, 100]]}
        result, meta = stage.process_image(img, metadata, cfg)

        np.testing.assert_array_equal(result, img)
        assert meta["method"] == "passthrough"

    def test_passthrough_when_quad_degenerate(self):
        """A tiny/degenerate quad should trigger pass-through."""
        from ghh.stages.perspective import PerspectiveStage

        stage = PerspectiveStage()
        img = make_music_page(width=600, height=400)
        cfg = Config(input_dir=Path("/tmp"))

        metadata = {"quad_corners": [[0, 0], [5, 0], [5, 5], [0, 5]]}
        result, meta = stage.process_image(img, metadata, cfg)

        np.testing.assert_array_equal(result, img)
        assert meta["method"] == "passthrough"


# ---------------------------------------------------------------------------
# TestPerspectiveCorrection
# ---------------------------------------------------------------------------

class TestPerspectiveCorrection:
    """Test actual perspective correction on known geometry."""

    def test_rectangular_quad_produces_correct_size(self):
        """A rectangular quad should produce output with matching dimensions."""
        from ghh.stages.perspective import PerspectiveStage

        stage = PerspectiveStage()
        page = make_music_page(width=600, height=400)
        photo = make_page_on_background(page, border=30)
        cfg = Config(input_dir=Path("/tmp"))

        quad = np.array([
            [30, 30], [630, 30], [630, 430], [30, 430],
        ], dtype=np.float32)
        metadata = {"quad_corners": quad.tolist()}

        result, meta = stage.process_image(photo, metadata, cfg)

        assert meta["method"] == "warpPerspective"
        assert result.shape[0] == 400
        assert result.shape[1] == 600

    def test_skewed_quad_is_rectified(self):
        """A perspective-skewed quad should be mapped to a rectangle."""
        from ghh.stages.perspective import PerspectiveStage

        stage = PerspectiveStage()
        photo, quad = _make_skewed_page_with_quad(border=30, skew=0.02)
        cfg = Config(input_dir=Path("/tmp"))

        metadata = {"quad_corners": quad.tolist()}
        result, meta = stage.process_image(photo, metadata, cfg)

        assert meta["method"] == "warpPerspective"
        rh, rw = result.shape[:2]
        assert rw > 0 and rh > 0
        aspect = rw / rh
        assert 1.2 < aspect < 1.8, f"Expected ~1.5 aspect ratio, got {aspect}"

    def test_output_uses_max_edge_lengths(self):
        """Output dimensions should use max (not average) of opposite edges."""
        from ghh.stages.perspective import PerspectiveStage

        stage = PerspectiveStage()
        img = np.full((600, 800, 3), 128, dtype=np.uint8)
        cfg = Config(input_dir=Path("/tmp"))

        quad = np.array([
            [20, 20], [780, 20], [760, 580], [40, 580],
        ], dtype=np.float32)
        metadata = {"quad_corners": quad.tolist()}

        result, meta = stage.process_image(img, metadata, cfg)

        assert meta["method"] == "warpPerspective"
        top_w = np.linalg.norm(quad[1] - quad[0])
        bot_w = np.linalg.norm(quad[2] - quad[3])
        expected_w = int(max(top_w, bot_w))
        assert result.shape[1] == expected_w

    def test_preserves_page_content(self):
        """Rectified output should contain page content, not just background."""
        from ghh.stages.perspective import PerspectiveStage

        stage = PerspectiveStage()
        page = make_music_page(width=600, height=400)
        photo = make_page_on_background(page, border=30)
        cfg = Config(input_dir=Path("/tmp"))

        quad = np.array([
            [30, 30], [630, 30], [630, 430], [30, 430],
        ], dtype=np.float32)
        metadata = {"quad_corners": quad.tolist()}

        result, meta = stage.process_image(photo, metadata, cfg)

        gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
        assert float(np.std(gray)) > 20, "Output should have content variance"

    def test_full_image_quad_is_near_identity(self):
        """A quad covering the full image should produce near-identical output."""
        from ghh.stages.perspective import PerspectiveStage

        stage = PerspectiveStage()
        img = make_music_page(width=600, height=400)
        cfg = Config(input_dir=Path("/tmp"))
        h, w = img.shape[:2]

        quad = np.array([
            [0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1],
        ], dtype=np.float32)
        metadata = {"quad_corners": quad.tolist()}

        result, meta = stage.process_image(img, metadata, cfg)

        rh, rw = result.shape[:2]
        assert abs(rw - w) <= 1, f"Width should be ~{w}, got {rw}"
        assert abs(rh - h) <= 1, f"Height should be ~{h}, got {rh}"

        min_h = min(rh, h) - 10
        min_w = min(rw, w) - 10
        center = result[10:min_h, 10:min_w]
        center_orig = img[10:min_h, 10:min_w]
        diff = np.mean(np.abs(center.astype(float) - center_orig.astype(float)))
        assert diff < 15.0, f"Near-identity transform should mostly preserve pixels, diff={diff}"

    def test_full_image_quad_fallback_preserves_size(self):
        """_full_image_quad uses [w,h] corners, so output matches input size."""
        from ghh.stages.page_detect import _full_image_quad
        from ghh.stages.perspective import PerspectiveStage

        stage = PerspectiveStage()
        img = make_music_page(width=600, height=400)
        cfg = Config(input_dir=Path("/tmp"))
        h, w = img.shape[:2]

        quad = _full_image_quad(h, w)
        metadata = {"quad_corners": quad.tolist()}

        result, meta = stage.process_image(img, metadata, cfg)
        assert result.shape[1] == w, f"Width should be {w}, got {result.shape[1]}"
        assert result.shape[0] == h, f"Height should be {h}, got {result.shape[0]}"


# ---------------------------------------------------------------------------
# TestBackgroundFill
# ---------------------------------------------------------------------------

class TestBackgroundFill:
    """Test that out-of-bounds pixels use background color, not black."""

    def test_background_is_not_black(self):
        """Border fill should match estimated background, not default black."""
        from ghh.stages.perspective import PerspectiveStage

        stage = PerspectiveStage()
        bg = (200, 190, 180)
        img = np.full((600, 800, 3), bg, dtype=np.uint8)
        cfg = Config(input_dir=Path("/tmp"))

        quad = np.array([
            [20, 20], [780, 10], [790, 580], [10, 570],
        ], dtype=np.float32)
        metadata = {"quad_corners": quad.tolist()}

        result, meta = stage.process_image(img, metadata, cfg)

        assert meta["method"] == "warpPerspective"
        bg_color = meta["background_color"]
        for i in range(3):
            assert abs(bg_color[i] - bg[i]) < 20, (
                f"Background channel {i}: expected ~{bg[i]}, got {bg_color[i]}"
            )

    def test_estimate_background_on_dark_image(self):
        """Background estimation on a dark border image."""
        from ghh.utils.image_utils import estimate_background

        dark = np.full((400, 600, 3), (30, 25, 20), dtype=np.uint8)
        bg = estimate_background(dark)

        assert len(bg) == 3
        for i, expected in enumerate((30, 25, 20)):
            assert abs(bg[i] - expected) < 5

    def test_estimate_background_on_grayscale(self):
        """Background estimation should work on single-channel images."""
        from ghh.utils.image_utils import estimate_background

        gray = np.full((400, 600), 180, dtype=np.uint8)
        bg = estimate_background(gray)

        assert len(bg) == 1
        assert abs(bg[0] - 180) < 5


# ---------------------------------------------------------------------------
# TestPerspectiveMetadata
# ---------------------------------------------------------------------------

class TestPerspectiveMetadata:
    """Test metadata sidecar contents."""

    def test_metadata_has_required_fields(self):
        from ghh.stages.perspective import PerspectiveStage

        stage = PerspectiveStage()
        img = np.full((600, 800, 3), 128, dtype=np.uint8)
        cfg = Config(input_dir=Path("/tmp"))

        metadata = {"quad_corners": [[20, 20], [780, 20], [780, 580], [20, 580]]}
        _, meta = stage.process_image(img, metadata, cfg)

        assert meta["stage"] == "perspective"
        assert meta["method"] == "warpPerspective"
        assert "src_quad" in meta
        assert "dst_size" in meta
        assert "background_color" in meta

    def test_forwards_page_type_from_stage4(self):
        """Stage 5 should forward page_type from Stage 4 metadata."""
        from ghh.stages.perspective import PerspectiveStage

        stage = PerspectiveStage()
        img = np.full((600, 800, 3), 128, dtype=np.uint8)
        cfg = Config(input_dir=Path("/tmp"))

        metadata = {
            "quad_corners": [[80, 80], [720, 80], [720, 520], [80, 520]],
            "page_type": "music",
        }
        _, meta = stage.process_image(img, metadata, cfg)
        assert meta["page_type"] == "music"

    def test_forwards_page_type_on_passthrough(self):
        """page_type should also be forwarded when Stage 5 passes through."""
        from ghh.stages.perspective import PerspectiveStage

        stage = PerspectiveStage()
        img = np.full((600, 800, 3), 128, dtype=np.uint8)
        cfg = Config(input_dir=Path("/tmp"))

        _, meta = stage.process_image(img, {"page_type": "text"}, cfg)
        assert meta["page_type"] == "text"

    def test_src_quad_is_ordered(self):
        """The src_quad in metadata should be in TL,TR,BR,BL order."""
        from ghh.stages.perspective import PerspectiveStage

        stage = PerspectiveStage()
        img = np.full((600, 800, 3), 128, dtype=np.uint8)
        cfg = Config(input_dir=Path("/tmp"))

        metadata = {"quad_corners": [[720, 520], [80, 520], [80, 80], [720, 80]]}
        _, meta = stage.process_image(img, metadata, cfg)

        corners = np.array(meta["src_quad"])
        assert corners[0][0] < corners[1][0], "TL.x < TR.x"
        assert corners[0][1] < corners[3][1], "TL.y < BL.y"


# ---------------------------------------------------------------------------
# TestSidecarPropagation
# ---------------------------------------------------------------------------

class TestSidecarPropagation:
    """Test that BaseStage.run() reads sidecar metadata from previous stage."""

    @pytest.mark.slow
    def test_reads_quad_from_stage4_sidecar(self, tmp_path):
        """Stage 5 run() should read quad_corners from Stage 4's JSON sidecar."""
        from ghh.stages.perspective import PerspectiveStage

        page = make_music_page(width=600, height=400)
        photo = make_page_on_background(page, border=30)

        quad = [[30, 30], [630, 30], [630, 430], [30, 430]]
        s4_meta = {"stage": "page_detect", "quad_corners": quad, "page_type": "music"}

        input_dir = _setup_stage4_output(
            tmp_path, {"IMG_0001.png": (photo, s4_meta)},
        )
        cfg = Config(input_dir=input_dir, output_dir=tmp_path)
        state = PipelineState(tmp_path)
        stage = PerspectiveStage()

        result = stage.run(input_dir, tmp_path, cfg, state)

        assert result.processed == 1
        out_img = tmp_path / "05_perspective" / "IMG_0001.png"
        assert out_img.exists()
        corrected = cv2.imread(str(out_img))
        assert corrected.shape[0] == 400
        assert corrected.shape[1] == 600

    @pytest.mark.slow
    def test_passthrough_without_sidecar(self, tmp_path):
        """Without a sidecar, Stage 5 should pass through unchanged."""
        from ghh.stages.perspective import PerspectiveStage

        img = make_music_page(width=600, height=400)
        input_dir = tmp_path / "04_page_detected"
        input_dir.mkdir()
        _save_test_image(input_dir / "IMG_0001.png", img)

        cfg = Config(input_dir=input_dir, output_dir=tmp_path)
        state = PipelineState(tmp_path)
        stage = PerspectiveStage()

        result = stage.run(input_dir, tmp_path, cfg, state)

        assert result.processed == 1
        out_img = tmp_path / "05_perspective" / "IMG_0001.png"
        corrected = cv2.imread(str(out_img))
        assert corrected.shape == img.shape


# ---------------------------------------------------------------------------
# TestPerspectiveStageRun (integration)
# ---------------------------------------------------------------------------

@pytest.mark.slow
class TestPerspectiveStageRun:
    """Integration tests for PerspectiveStage.run()."""

    def test_produces_checkpoint_directory(self, tmp_path):
        from ghh.stages.perspective import PerspectiveStage

        page = make_music_page(width=600, height=400)
        photo = make_page_on_background(page, border=30)
        quad = [[30, 30], [630, 30], [630, 430], [30, 430]]
        s4_meta = {"stage": "page_detect", "quad_corners": quad}

        input_dir = _setup_stage4_output(
            tmp_path, {"IMG_0001.png": (photo, s4_meta)},
        )
        cfg = Config(input_dir=input_dir, output_dir=tmp_path)
        state = PipelineState(tmp_path)
        stage = PerspectiveStage()

        stage.run(input_dir, tmp_path, cfg, state)

        assert (tmp_path / "05_perspective").exists()

    def test_processes_multiple_images(self, tmp_path):
        from ghh.stages.perspective import PerspectiveStage

        quad = [[30, 30], [630, 30], [630, 430], [30, 430]]
        s4_meta = {"stage": "page_detect", "quad_corners": quad}

        images = {}
        for i in range(3):
            page = make_music_page(width=600, height=400)
            photo = make_page_on_background(page, border=30)
            images[f"IMG_{i:04d}.png"] = (photo, s4_meta)

        input_dir = _setup_stage4_output(tmp_path, images)
        cfg = Config(input_dir=input_dir, output_dir=tmp_path)
        state = PipelineState(tmp_path)
        stage = PerspectiveStage()

        result = stage.run(input_dir, tmp_path, cfg, state)

        assert result.processed == 3
        for i in range(3):
            assert (tmp_path / "05_perspective" / f"IMG_{i:04d}.png").exists()

    def test_writes_metadata_sidecar(self, tmp_path):
        from ghh.stages.perspective import PerspectiveStage

        page = make_music_page(width=600, height=400)
        photo = make_page_on_background(page, border=30)
        quad = [[30, 30], [630, 30], [630, 430], [30, 430]]
        s4_meta = {"stage": "page_detect", "quad_corners": quad}

        input_dir = _setup_stage4_output(
            tmp_path, {"IMG_0001.png": (photo, s4_meta)},
        )
        cfg = Config(input_dir=input_dir, output_dir=tmp_path)
        state = PipelineState(tmp_path)
        stage = PerspectiveStage()

        stage.run(input_dir, tmp_path, cfg, state)

        sidecar = tmp_path / "05_perspective" / "IMG_0001.json"
        assert sidecar.exists()
        meta = json.loads(sidecar.read_text())
        assert meta["stage"] == "perspective"
        assert "src_quad" in meta
        assert "dst_size" in meta

    def test_resume_skips_completed(self, tmp_path):
        from ghh.stages.perspective import PerspectiveStage

        page = make_music_page(width=600, height=400)
        photo = make_page_on_background(page, border=30)
        quad = [[30, 30], [630, 30], [630, 430], [30, 430]]
        s4_meta = {"stage": "page_detect", "quad_corners": quad}

        input_dir = _setup_stage4_output(
            tmp_path, {"IMG_0001.png": (photo, s4_meta)},
        )
        cfg = Config(input_dir=input_dir, output_dir=tmp_path)
        state = PipelineState(tmp_path)
        stage = PerspectiveStage()

        r1 = stage.run(input_dir, tmp_path, cfg, state)
        assert r1.processed == 1

        r2 = stage.run(input_dir, tmp_path, cfg, state)
        assert r2.processed == 0
        assert r2.skipped == 1

    def test_returns_stage_result(self, tmp_path):
        from ghh.stages.perspective import PerspectiveStage

        page = make_music_page(width=600, height=400)
        photo = make_page_on_background(page, border=30)
        quad = [[30, 30], [630, 30], [630, 430], [30, 430]]
        s4_meta = {"stage": "page_detect", "quad_corners": quad}

        input_dir = _setup_stage4_output(
            tmp_path, {"IMG_0001.png": (photo, s4_meta)},
        )
        cfg = Config(input_dir=input_dir, output_dir=tmp_path)
        state = PipelineState(tmp_path)
        stage = PerspectiveStage()

        result = stage.run(input_dir, tmp_path, cfg, state)

        assert isinstance(result, StageResult)
        assert result.stage_name == "perspective"
        assert result.processed == 1
        assert result.failed == 0

    def test_skewed_page_is_rectified_in_run(self, tmp_path):
        from ghh.stages.perspective import PerspectiveStage

        photo, quad = _make_skewed_page_with_quad(border=30, skew=0.02)
        s4_meta = {"stage": "page_detect", "quad_corners": quad.tolist()}

        input_dir = _setup_stage4_output(
            tmp_path, {"IMG_0001.png": (photo, s4_meta)},
        )
        cfg = Config(input_dir=input_dir, output_dir=tmp_path)
        state = PipelineState(tmp_path)
        stage = PerspectiveStage()

        result = stage.run(input_dir, tmp_path, cfg, state)

        assert result.processed == 1
        out_img = tmp_path / "05_perspective" / "IMG_0001.png"
        corrected = cv2.imread(str(out_img))
        aspect = corrected.shape[1] / corrected.shape[0]
        assert 1.2 < aspect < 1.8, f"Expected ~1.5 aspect, got {aspect}"
