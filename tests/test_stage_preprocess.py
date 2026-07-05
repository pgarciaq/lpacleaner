"""TDD tests for Stage 0 (PreprocessStage): hotspot removal + finger masking.

Tests the stage as a BaseStage subclass -- verifying it integrates with the
pipeline orchestration contract while correctly delegating to the lower-level
functions in utils/preprocess.py.
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from ghh.config import Config
from ghh.pipeline import PipelineState, StageResult
from ghh.stages.preprocess import PreprocessStage


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _save_test_image(path: Path, img: np.ndarray) -> None:
    """Write an image to disk as PNG."""
    cv2.imwrite(str(path), img)


def _make_clean_page(h: int = 400, w: int = 600) -> np.ndarray:
    """Beige page with no hotspots or fingers -- should pass through unchanged."""
    return np.full((h, w, 3), (200, 220, 230), dtype=np.uint8)


def _make_page_with_hotspot(h: int = 400, w: int = 600) -> np.ndarray:
    """Beige page with a large bright-white hotspot in the center."""
    img = _make_clean_page(h, w)
    cv2.circle(img, (w // 2, h // 2), 80, (255, 255, 255), -1)
    return img


def _make_page_with_finger(h: int = 400, w: int = 600) -> np.ndarray:
    """Neutral gray page with a skin-colored region touching the right border.

    The background must NOT fall in the YCrCb skin range (Cr 133-173,
    Cb 77-127), otherwise the entire page is classified as skin and the
    finger component exceeds the 15% area cap.
    BGR=(210,210,210) → Cr=128 Cb=128 (outside skin range).

    The finger color must be IN the skin range but have HSV saturation
    ≤120 so it isn't excluded by the ink-hue filter.
    BGR=(130,160,200) → Cr=150 Cb=106 (skin ✓), S=89 (not ink ✓).
    """
    img = np.full((h, w, 3), (210, 210, 210), dtype=np.uint8)
    skin_bgr = (130, 160, 200)
    cv2.rectangle(img, (w - 60, h // 3), (w, 2 * h // 3), skin_bgr, -1)
    return img


def _setup_stage_input(tmp_path: Path, images: dict[str, np.ndarray]) -> Path:
    """Create an input directory with named images. Returns the dir path."""
    input_dir = tmp_path / "raw_input"
    input_dir.mkdir()
    for name, img in images.items():
        _save_test_image(input_dir / name, img)
    return input_dir


# ---------------------------------------------------------------------------
# TestPreprocessStage: class attributes
# ---------------------------------------------------------------------------

class TestPreprocessStageContract:
    """Verify that PreprocessStage satisfies the BaseStage contract."""

    def test_has_correct_name(self):
        stage = PreprocessStage()
        assert stage.name == "preprocess"

    def test_has_correct_number(self):
        stage = PreprocessStage()
        assert stage.number == 0

    def test_has_correct_checkpoint_name(self):
        stage = PreprocessStage()
        assert stage.checkpoint_name == "00_preprocessed"

    def test_has_correct_error_class(self):
        stage = PreprocessStage()
        assert stage.error_class == "skippable"

    def test_is_base_stage_subclass(self):
        from ghh.pipeline import BaseStage
        assert issubclass(PreprocessStage, BaseStage)


# ---------------------------------------------------------------------------
# TestPreprocessStage: process_image behavior
# ---------------------------------------------------------------------------

@pytest.mark.slow
class TestPreprocessProcessImage:
    """Test process_image() logic for various input scenarios."""

    def test_clean_page_passes_through(self):
        """A page with no hotspots or fingers should come out virtually identical."""
        stage = PreprocessStage()
        img = _make_clean_page()
        cfg = Config(input_dir=Path("/tmp"))

        result_img, meta = stage.process_image(img, {}, cfg)

        assert result_img.shape == img.shape
        assert result_img.dtype == img.dtype
        assert meta["hotspot_detected"] is False

    def test_hotspot_is_removed(self):
        """A page with a bright hotspot should have it inpainted."""
        stage = PreprocessStage()
        img = _make_page_with_hotspot()
        cfg = Config(input_dir=Path("/tmp"))

        result_img, meta = stage.process_image(img, {}, cfg)

        # The center pixel should no longer be saturated white
        cy, cx = img.shape[0] // 2, img.shape[1] // 2
        center = result_img[cy, cx]
        assert not all(c > 250 for c in center), "Hotspot center should be inpainted"
        assert meta["hotspot_detected"] is True

    def test_finger_is_removed_when_enabled(self):
        """When fingers_detected=True, skin-colored border regions are inpainted."""
        stage = PreprocessStage()
        img = _make_page_with_finger()
        cfg = Config(input_dir=Path("/tmp"), fingers_detected=True)

        result_img, meta = stage.process_image(img, {}, cfg)

        assert result_img.shape == img.shape
        assert meta.get("finger_removed") is True

    def test_finger_not_removed_when_disabled(self):
        """When fingers_detected=False (default), finger removal is skipped."""
        stage = PreprocessStage()
        img = _make_page_with_finger()
        cfg = Config(input_dir=Path("/tmp"), fingers_detected=False)

        result_img, meta = stage.process_image(img, {}, cfg)

        # The image should pass through with only hotspot processing
        assert meta.get("finger_removed", False) is False

    def test_preserves_image_dimensions(self):
        stage = PreprocessStage()
        img = _make_page_with_hotspot()
        cfg = Config(input_dir=Path("/tmp"))

        result_img, _ = stage.process_image(img, {}, cfg)
        assert result_img.shape == img.shape

    def test_metadata_includes_stage_info(self):
        """Metadata should identify which stage produced it."""
        stage = PreprocessStage()
        img = _make_clean_page()
        cfg = Config(input_dir=Path("/tmp"))

        _, meta = stage.process_image(img, {}, cfg)
        assert meta.get("stage") == "preprocess"


# ---------------------------------------------------------------------------
# TestPreprocessStage: integration with run() orchestration
# ---------------------------------------------------------------------------

@pytest.mark.slow
class TestPreprocessStageRun:
    """Test the full run() lifecycle via BaseStage.run()."""

    def test_produces_checkpoint_directory(self, tmp_path):
        input_dir = _setup_stage_input(tmp_path, {
            "IMG_0001.png": _make_clean_page(),
        })
        cfg = Config(input_dir=input_dir, output_dir=tmp_path)
        state = PipelineState(tmp_path)
        stage = PreprocessStage()

        stage.run(input_dir, tmp_path, cfg, state)

        assert (tmp_path / "00_preprocessed").exists()

    def test_processes_all_images(self, tmp_path):
        input_dir = _setup_stage_input(tmp_path, {
            "IMG_0001.png": _make_clean_page(),
            "IMG_0002.png": _make_page_with_hotspot(),
            "IMG_0003.png": _make_clean_page(),
        })
        cfg = Config(input_dir=input_dir, output_dir=tmp_path)
        state = PipelineState(tmp_path)
        stage = PreprocessStage()

        result = stage.run(input_dir, tmp_path, cfg, state)

        assert result.processed == 3
        assert result.failed == 0
        out_files = list((tmp_path / "00_preprocessed").glob("*.png"))
        assert len(out_files) == 3

    def test_writes_metadata_sidecars(self, tmp_path):
        input_dir = _setup_stage_input(tmp_path, {
            "IMG_0001.png": _make_page_with_hotspot(),
        })
        cfg = Config(input_dir=input_dir, output_dir=tmp_path)
        state = PipelineState(tmp_path)
        stage = PreprocessStage()

        stage.run(input_dir, tmp_path, cfg, state)

        sidecar = tmp_path / "00_preprocessed" / "IMG_0001.json"
        assert sidecar.exists()
        meta = json.loads(sidecar.read_text())
        assert meta["hotspot_detected"] is True
        assert meta["stage"] == "preprocess"

    def test_returns_stage_result(self, tmp_path):
        input_dir = _setup_stage_input(tmp_path, {
            "IMG_0001.png": _make_clean_page(),
        })
        cfg = Config(input_dir=input_dir, output_dir=tmp_path)
        state = PipelineState(tmp_path)
        stage = PreprocessStage()

        result = stage.run(input_dir, tmp_path, cfg, state)

        assert isinstance(result, StageResult)
        assert result.stage_name == "preprocess"

    def test_resume_skips_completed_images(self, tmp_path):
        input_dir = _setup_stage_input(tmp_path, {
            "IMG_0001.png": _make_clean_page(),
            "IMG_0002.png": _make_clean_page(),
        })
        cfg = Config(input_dir=input_dir, output_dir=tmp_path)
        state = PipelineState(tmp_path)
        stage = PreprocessStage()

        # First run
        stage.run(input_dir, tmp_path, cfg, state)

        # Second run should skip everything
        result2 = stage.run(input_dir, tmp_path, cfg, state)
        assert result2.skipped == 2
        assert result2.processed == 0

    def test_handles_jpeg_input(self, tmp_path):
        """Stage 0 is the entry point -- it must handle JPEG inputs."""
        input_dir = tmp_path / "raw_input"
        input_dir.mkdir()
        img = _make_clean_page()
        cv2.imwrite(str(input_dir / "IMG_0001.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 95])

        cfg = Config(input_dir=input_dir, output_dir=tmp_path)
        state = PipelineState(tmp_path)
        stage = PreprocessStage()

        result = stage.run(input_dir, tmp_path, cfg, state)

        assert result.processed == 1
        # Output should be PNG (lossless checkpoint)
        assert (tmp_path / "00_preprocessed" / "IMG_0001.png").exists()

    def test_mixed_hotspot_and_clean(self, tmp_path):
        """Verifies each image is processed independently."""
        input_dir = _setup_stage_input(tmp_path, {
            "IMG_0001.png": _make_clean_page(),
            "IMG_0002.png": _make_page_with_hotspot(),
        })
        cfg = Config(input_dir=input_dir, output_dir=tmp_path)
        state = PipelineState(tmp_path)
        stage = PreprocessStage()

        stage.run(input_dir, tmp_path, cfg, state)

        meta1 = json.loads((tmp_path / "00_preprocessed" / "IMG_0001.json").read_text())
        meta2 = json.loads((tmp_path / "00_preprocessed" / "IMG_0002.json").read_text())
        assert meta1["hotspot_detected"] is False
        assert meta2["hotspot_detected"] is True
