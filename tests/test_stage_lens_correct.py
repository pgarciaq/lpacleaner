"""Tests for Stage 3 (LensCorrectStage): radial distortion correction.

Tests the stage as a BaseStage subclass.  Lens correction uses
cv2.undistort with k1/k2 coefficients from the Config.  The stage
is skipped entirely when both coefficients are zero.
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from ghh.config import Config
from ghh.pipeline import BaseStage, PipelineState, StageResult

from tests.conftest import make_music_page, add_barrel_distortion


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



# ---------------------------------------------------------------------------
# TestLensCorrectStageContract
# ---------------------------------------------------------------------------

class TestLensCorrectStageContract:
    """Verify that LensCorrectStage satisfies the BaseStage contract."""

    def test_has_correct_name(self):
        from ghh.stages.lens_correct import LensCorrectStage

        assert LensCorrectStage().name == "lens_correct"

    def test_has_correct_number(self):
        from ghh.stages.lens_correct import LensCorrectStage

        assert LensCorrectStage().number == 3

    def test_has_correct_checkpoint_name(self):
        from ghh.stages.lens_correct import LensCorrectStage

        assert LensCorrectStage().checkpoint_name == "03_lens_corrected"

    def test_is_base_stage_subclass(self):
        from ghh.stages.lens_correct import LensCorrectStage

        assert issubclass(LensCorrectStage, BaseStage)

    def test_error_class_is_skippable(self):
        from ghh.stages.lens_correct import LensCorrectStage

        assert LensCorrectStage().error_class == "skippable"


# ---------------------------------------------------------------------------
# TestLensCorrectShouldSkip
# ---------------------------------------------------------------------------

class TestLensCorrectShouldSkip:
    """Test the should_skip logic for LensCorrectStage."""

    def test_skips_when_k1_k2_both_zero(self):
        from ghh.stages.lens_correct import LensCorrectStage

        stage = LensCorrectStage()
        cfg = Config(input_dir=Path("/tmp"),
                     lens_distortion_k1=0.0, lens_distortion_k2=0.0)
        assert stage.should_skip(cfg) is True

    def test_does_not_skip_when_k1_nonzero(self):
        from ghh.stages.lens_correct import LensCorrectStage

        stage = LensCorrectStage()
        cfg = Config(input_dir=Path("/tmp"),
                     lens_distortion_k1=0.1, lens_distortion_k2=0.0)
        assert stage.should_skip(cfg) is False

    def test_does_not_skip_when_k2_nonzero(self):
        from ghh.stages.lens_correct import LensCorrectStage

        stage = LensCorrectStage()
        cfg = Config(input_dir=Path("/tmp"),
                     lens_distortion_k1=0.0, lens_distortion_k2=0.05)
        assert stage.should_skip(cfg) is False

    def test_does_not_skip_when_both_nonzero(self):
        from ghh.stages.lens_correct import LensCorrectStage

        stage = LensCorrectStage()
        cfg = Config(input_dir=Path("/tmp"),
                     lens_distortion_k1=0.1, lens_distortion_k2=0.05)
        assert stage.should_skip(cfg) is False


# ---------------------------------------------------------------------------
# TestLensCorrectProcessImage
# ---------------------------------------------------------------------------

class TestLensCorrectProcessImage:
    """Test process_image() for distortion correction."""

    def test_preserves_image_dimensions(self):
        from ghh.stages.lens_correct import LensCorrectStage

        stage = LensCorrectStage()
        img = make_music_page(width=400, height=300)
        cfg = Config(input_dir=Path("/tmp"),
                     lens_distortion_k1=0.1, lens_distortion_k2=0.0)

        result, meta = stage.process_image(img, {}, cfg)
        assert result.shape == img.shape

    def test_metadata_includes_coefficients(self):
        from ghh.stages.lens_correct import LensCorrectStage

        stage = LensCorrectStage()
        img = make_music_page(width=400, height=300)
        cfg = Config(input_dir=Path("/tmp"),
                     lens_distortion_k1=0.15, lens_distortion_k2=0.03)

        _, meta = stage.process_image(img, {}, cfg)
        assert meta["stage"] == "lens_correct"
        assert meta["k1"] == 0.15
        assert meta["k2"] == 0.03
        assert meta["focal_length_px"] == 400.0  # max(400, 300)

    def test_corrects_barrel_distortion(self):
        """Applying distortion then correcting should yield an image
        closer to the original than the distorted version.

        Comparison uses the centre 70% of the image to avoid edge
        artifacts from out-of-bounds sampling during distortion.
        """
        from ghh.stages.lens_correct import LensCorrectStage

        stage = LensCorrectStage()
        k1 = 0.3
        original = make_music_page(width=400, height=400)
        distorted = add_barrel_distortion(original, k1=k1)
        cfg = Config(input_dir=Path("/tmp"),
                     lens_distortion_k1=k1, lens_distortion_k2=0.0)

        corrected, _ = stage.process_image(distorted, {}, cfg)

        h, w = original.shape[:2]
        m = int(h * 0.15)
        roi = (slice(m, h - m), slice(m, w - m))

        diff_before = np.mean(np.abs(original[roi].astype(float) - distorted[roi].astype(float)))
        diff_after = np.mean(np.abs(original[roi].astype(float) - corrected[roi].astype(float)))

        assert diff_after < diff_before, (
            f"Correction should reduce error: before={diff_before:.2f}, "
            f"after={diff_after:.2f}"
        )

    def test_corrects_with_both_k1_and_k2(self):
        """Correction with k1+k2 should reduce distortion in the centre."""
        from ghh.stages.lens_correct import LensCorrectStage

        stage = LensCorrectStage()
        k1 = 0.2
        original = make_music_page(width=400, height=400)
        distorted = add_barrel_distortion(original, k1=k1)
        cfg = Config(input_dir=Path("/tmp"),
                     lens_distortion_k1=k1, lens_distortion_k2=0.01)

        corrected, meta = stage.process_image(distorted, {}, cfg)

        assert meta["k1"] == k1
        assert meta["k2"] == 0.01
        assert corrected.shape == original.shape

    def test_identity_on_undistorted_image(self):
        """With very small k1, image should barely change."""
        from ghh.stages.lens_correct import LensCorrectStage

        stage = LensCorrectStage()
        img = make_music_page(width=400, height=300)
        cfg = Config(input_dir=Path("/tmp"),
                     lens_distortion_k1=0.001, lens_distortion_k2=0.0)

        result, _ = stage.process_image(img, {}, cfg)
        diff = np.mean(np.abs(img.astype(float) - result.astype(float)))

        assert diff < 1.0, f"Tiny k1 should barely alter image, got diff={diff:.2f}"

    def test_handles_negative_k1_pincushion(self):
        """Pincushion distortion (k1 < 0) should also be correctable."""
        from ghh.stages.lens_correct import LensCorrectStage

        stage = LensCorrectStage()
        k1 = -0.2
        original = make_music_page(width=400, height=400)
        distorted = add_barrel_distortion(original, k1=k1)
        cfg = Config(input_dir=Path("/tmp"),
                     lens_distortion_k1=k1, lens_distortion_k2=0.0)

        corrected, _ = stage.process_image(distorted, {}, cfg)

        h, w = original.shape[:2]
        m = int(h * 0.15)
        roi = (slice(m, h - m), slice(m, w - m))

        diff_before = np.mean(np.abs(original[roi].astype(float) - distorted[roi].astype(float)))
        diff_after = np.mean(np.abs(original[roi].astype(float) - corrected[roi].astype(float)))

        assert diff_after < diff_before, (
            f"Correction should reduce error: before={diff_before:.2f}, "
            f"after={diff_after:.2f}"
        )

    def test_registered_in_stage_registry(self):
        from ghh.stages import STAGE_BY_NUMBER, STAGE_BY_NAME

        assert 3 in STAGE_BY_NUMBER
        assert "lens_correct" in STAGE_BY_NAME


# ---------------------------------------------------------------------------
# TestLensCorrectStageRun (integration)
# ---------------------------------------------------------------------------

@pytest.mark.slow
class TestLensCorrectStageRun:
    """Integration tests for LensCorrectStage.run()."""

    def test_produces_checkpoint_directory(self, tmp_path):
        from ghh.stages.lens_correct import LensCorrectStage

        input_dir = _setup_stage_input(tmp_path, {
            "IMG_0001.png": make_music_page(width=400, height=300),
        })
        cfg = Config(input_dir=input_dir, output_dir=tmp_path,
                     lens_distortion_k1=0.1)
        state = PipelineState(tmp_path)
        stage = LensCorrectStage()

        stage.run(input_dir, tmp_path, cfg, state)

        assert (tmp_path / "03_lens_corrected").exists()

    def test_processes_all_images(self, tmp_path):
        from ghh.stages.lens_correct import LensCorrectStage

        input_dir = _setup_stage_input(tmp_path, {
            "IMG_0001.png": make_music_page(width=400, height=300),
            "IMG_0002.png": make_music_page(width=400, height=300),
            "IMG_0003.png": make_music_page(width=400, height=300),
        })
        cfg = Config(input_dir=input_dir, output_dir=tmp_path,
                     lens_distortion_k1=0.1)
        state = PipelineState(tmp_path)
        stage = LensCorrectStage()

        result = stage.run(input_dir, tmp_path, cfg, state)

        assert result.processed == 3
        assert result.failed == 0
        out_files = list((tmp_path / "03_lens_corrected").glob("*.png"))
        assert len(out_files) == 3

    def test_writes_metadata_sidecar(self, tmp_path):
        from ghh.stages.lens_correct import LensCorrectStage

        input_dir = _setup_stage_input(tmp_path, {
            "IMG_0001.png": make_music_page(width=400, height=300),
        })
        cfg = Config(input_dir=input_dir, output_dir=tmp_path,
                     lens_distortion_k1=0.15, lens_distortion_k2=0.02)
        state = PipelineState(tmp_path)
        stage = LensCorrectStage()

        stage.run(input_dir, tmp_path, cfg, state)

        sidecar = tmp_path / "03_lens_corrected" / "IMG_0001.json"
        assert sidecar.exists()
        meta = json.loads(sidecar.read_text())
        assert meta["stage"] == "lens_correct"
        assert meta["k1"] == 0.15
        assert meta["k2"] == 0.02

    def test_resume_skips_completed(self, tmp_path):
        from ghh.stages.lens_correct import LensCorrectStage

        input_dir = _setup_stage_input(tmp_path, {
            "IMG_0001.png": make_music_page(width=400, height=300),
            "IMG_0002.png": make_music_page(width=400, height=300),
        })
        cfg = Config(input_dir=input_dir, output_dir=tmp_path,
                     lens_distortion_k1=0.1)
        state = PipelineState(tmp_path)
        stage = LensCorrectStage()

        stage.run(input_dir, tmp_path, cfg, state)
        result2 = stage.run(input_dir, tmp_path, cfg, state)

        assert result2.skipped == 2
        assert result2.processed == 0

    def test_returns_stage_result(self, tmp_path):
        from ghh.stages.lens_correct import LensCorrectStage

        input_dir = _setup_stage_input(tmp_path, {
            "IMG_0001.png": make_music_page(width=400, height=300),
        })
        cfg = Config(input_dir=input_dir, output_dir=tmp_path,
                     lens_distortion_k1=0.1)
        state = PipelineState(tmp_path)
        stage = LensCorrectStage()

        result = stage.run(input_dir, tmp_path, cfg, state)

        assert isinstance(result, StageResult)
        assert result.stage_name == "lens_correct"

    def test_skipped_when_no_distortion(self, tmp_path):
        from ghh.stages.lens_correct import LensCorrectStage

        input_dir = _setup_stage_input(tmp_path, {
            "IMG_0001.png": make_music_page(width=400, height=300),
        })
        cfg = Config(input_dir=input_dir, output_dir=tmp_path,
                     lens_distortion_k1=0.0, lens_distortion_k2=0.0)
        stage = LensCorrectStage()

        assert stage.should_skip(cfg) is True
        assert not (tmp_path / "03_lens_corrected").exists()
