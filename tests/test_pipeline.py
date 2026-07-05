"""TDD tests for ghh.pipeline -- BaseStage contract and PipelineState."""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from ghh.config import Config
from ghh.pipeline import BaseStage, PipelineState, StageResult


# ---------------------------------------------------------------------------
# Concrete test stage (minimal implementation for testing the contract)
# ---------------------------------------------------------------------------

class InvertStage(BaseStage):
    """Test stage that inverts image colors."""

    name = "invert"
    number = 99
    checkpoint_name = "99_inverted"
    error_class = "skippable"

    def process_image(self, img, metadata, cfg):
        return cv2.bitwise_not(img), {**metadata, "inverted": True}


class FailingStage(BaseStage):
    """Test stage that always raises an error."""

    name = "failing"
    number = 98
    checkpoint_name = "98_failing"
    error_class = "skippable"

    def process_image(self, img, metadata, cfg):
        raise RuntimeError("Intentional failure for testing")


class CriticalFailStage(BaseStage):
    """Test stage with critical error class that always fails."""

    name = "critical_fail"
    number = 97
    checkpoint_name = "97_critical"
    error_class = "critical"

    def process_image(self, img, metadata, cfg):
        raise RuntimeError("Critical failure")


# ---------------------------------------------------------------------------
# TestPipelineState
# ---------------------------------------------------------------------------

class TestPipelineState:
    """Tests for PipelineState: tracking stage completion and config hashes."""

    def test_creates_empty_state(self, tmp_path):
        state = PipelineState(tmp_path)
        assert state.output_dir == tmp_path

    def test_save_and_load(self, tmp_path):
        state = PipelineState(tmp_path)
        state.set_stage_hash("stage_2", "abc123")
        state.mark_image_done("stage_2", "IMG_0001")
        state.save()

        loaded = PipelineState.load(tmp_path)
        assert loaded.get_stage_hash("stage_2") == "abc123"
        assert loaded.is_image_done("stage_2", "IMG_0001")

    def test_pipeline_json_location(self, tmp_path):
        state = PipelineState(tmp_path)
        state.save()
        assert (tmp_path / "pipeline.json").exists()

    def test_is_image_done_returns_false_for_unknown(self, tmp_path):
        state = PipelineState(tmp_path)
        assert state.is_image_done("stage_2", "IMG_0099") is False

    def test_config_hash_invalidation(self, tmp_path):
        state = PipelineState(tmp_path)
        state.set_stage_hash("stage_7", "old_hash")
        assert state.is_stage_invalidated("stage_7", "new_hash") is True
        assert state.is_stage_invalidated("stage_7", "old_hash") is False

    def test_invalidate_clears_images(self, tmp_path):
        state = PipelineState(tmp_path)
        state.mark_image_done("stage_7", "IMG_0001")
        state.mark_image_done("stage_7", "IMG_0002")
        state.invalidate_stage("stage_7")
        assert state.is_image_done("stage_7", "IMG_0001") is False
        assert state.is_image_done("stage_7", "IMG_0002") is False

    def test_config_source_tracking(self, tmp_path):
        state = PipelineState(tmp_path)
        state.config_source = "analyzed"
        state.save()

        loaded = PipelineState.load(tmp_path)
        assert loaded.config_source == "analyzed"

    def test_record_stage_result(self, tmp_path):
        state = PipelineState(tmp_path)
        result = StageResult(
            stage_name="orientation",
            processed=10,
            skipped=2,
            failed=1,
            excluded=0,
        )
        state.record_result(result)
        assert state.get_result("orientation").processed == 10

    def test_load_from_nonexistent_returns_empty(self, tmp_path):
        state = PipelineState.load(tmp_path / "nonexistent")
        assert state is not None


# ---------------------------------------------------------------------------
# TestBaseStage
# ---------------------------------------------------------------------------

class TestBaseStage:
    """Tests for BaseStage.run(): orchestration loop with checkpointing."""

    def _setup_input(self, tmp_path, num_images=3):
        """Create a fake input directory with synthetic PNGs."""
        input_dir = tmp_path / "02_oriented"
        input_dir.mkdir(parents=True)
        cfg = Config(input_dir=tmp_path, output_dir=tmp_path)

        for i in range(num_images):
            img = np.full((100, 100, 3), (100 + i * 30, 80, 60), dtype=np.uint8)
            cv2.imwrite(str(input_dir / f"IMG_{i:04d}.png"), img)

        return input_dir, cfg

    def test_produces_output_files(self, tmp_path):
        input_dir, cfg = self._setup_input(tmp_path)
        stage = InvertStage()
        state = PipelineState(tmp_path)

        result = stage.run(input_dir, tmp_path, cfg, state)

        out_dir = tmp_path / "99_inverted"
        assert out_dir.exists()
        assert len(list(out_dir.glob("*.png"))) == 3

    def test_returns_stage_result(self, tmp_path):
        input_dir, cfg = self._setup_input(tmp_path)
        stage = InvertStage()
        state = PipelineState(tmp_path)

        result = stage.run(input_dir, tmp_path, cfg, state)
        assert isinstance(result, StageResult)
        assert result.processed == 3
        assert result.failed == 0

    def test_skippable_stage_passes_through_on_failure(self, tmp_path):
        input_dir, cfg = self._setup_input(tmp_path)
        stage = FailingStage()
        state = PipelineState(tmp_path)

        result = stage.run(input_dir, tmp_path, cfg, state)

        # Skippable: all images should pass through (copied from input)
        out_dir = tmp_path / "98_failing"
        assert out_dir.exists()
        assert len(list(out_dir.glob("*.png"))) == 3
        assert result.failed == 3
        assert result.processed == 0

    def test_critical_stage_excludes_on_failure(self, tmp_path):
        input_dir, cfg = self._setup_input(tmp_path)
        stage = CriticalFailStage()
        state = PipelineState(tmp_path)

        result = stage.run(input_dir, tmp_path, cfg, state)

        # Critical: failed images should be excluded (not written)
        out_dir = tmp_path / "97_critical"
        assert out_dir.exists()
        assert len(list(out_dir.glob("*.png"))) == 0
        assert result.excluded == 3

    def test_resumes_completed_images(self, tmp_path):
        input_dir, cfg = self._setup_input(tmp_path)
        stage = InvertStage()
        state = PipelineState(tmp_path)

        # Pre-create one output (simulating a previous run)
        out_dir = tmp_path / "99_inverted"
        out_dir.mkdir(parents=True)
        first_input = cv2.imread(str(input_dir / "IMG_0000.png"))
        cv2.imwrite(str(out_dir / "IMG_0000.png"), cv2.bitwise_not(first_input))
        state.mark_image_done("99_inverted", "IMG_0000")

        result = stage.run(input_dir, tmp_path, cfg, state)
        # Should process only 2 new images, skip the pre-existing one
        assert result.processed == 2
        assert result.skipped == 1

    def test_writes_metadata_sidecars(self, tmp_path):
        input_dir, cfg = self._setup_input(tmp_path, num_images=1)
        stage = InvertStage()
        state = PipelineState(tmp_path)

        stage.run(input_dir, tmp_path, cfg, state)

        sidecar = tmp_path / "99_inverted" / "IMG_0000.json"
        assert sidecar.exists()
        meta = json.loads(sidecar.read_text())
        assert meta["inverted"] is True

    def test_should_skip_respects_config(self, tmp_path):
        cfg = Config(input_dir=tmp_path, skip_enhance=True)

        class EnhanceStage(BaseStage):
            name = "enhance"
            number = 9
            checkpoint_name = "09_enhanced"
            error_class = "skippable"
            def process_image(self, img, metadata, cfg):
                return img, metadata

        stage = EnhanceStage()
        assert stage.should_skip(cfg) is True
