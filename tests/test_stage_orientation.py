"""TDD tests for Stage 2 (OrientationStage): EXIF rotation + content orientation + focus QA.

Tests the stage as a BaseStage subclass. Orientation correction uses
EXIF tags, staff line angle detection, and coarse rotation offset from
analyze. Focus QA flags blurry images.
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pytest
from PIL import Image

from lpacleaner.config import Config
from lpacleaner.pipeline import BaseStage, PipelineState, StageResult

from tests.conftest import make_music_page, make_text_page


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _save_test_image(path: Path, img: np.ndarray) -> None:
    cv2.imwrite(str(path), img)


def _save_jpeg_with_exif(path: Path, img: np.ndarray, orientation: int = 1) -> None:
    """Save a JPEG with the given EXIF orientation tag."""
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(rgb)
    exif = pil_img.getexif()
    exif[0x0112] = orientation
    pil_img.save(path, "JPEG", quality=95, exif=exif.tobytes())


def _setup_stage_input(tmp_path: Path, images: dict[str, np.ndarray]) -> Path:
    input_dir = tmp_path / "01_stitched"
    input_dir.mkdir()
    for name, img in images.items():
        _save_test_image(input_dir / name, img)
    return input_dir


# ---------------------------------------------------------------------------
# TestOrientationStageContract
# ---------------------------------------------------------------------------

class TestOrientationStageContract:
    """Verify that OrientationStage satisfies the BaseStage contract."""

    def test_has_correct_name(self):
        from lpacleaner.stages.orientation import OrientationStage

        stage = OrientationStage()
        assert stage.name == "orientation"

    def test_has_correct_number(self):
        from lpacleaner.stages.orientation import OrientationStage

        stage = OrientationStage()
        assert stage.number == 2

    def test_has_correct_checkpoint_name(self):
        from lpacleaner.stages.orientation import OrientationStage

        stage = OrientationStage()
        assert stage.checkpoint_name == "02_oriented"

    def test_is_base_stage_subclass(self):
        from lpacleaner.stages.orientation import OrientationStage

        assert issubclass(OrientationStage, BaseStage)

    def test_should_skip_always_false(self):
        from lpacleaner.stages.orientation import OrientationStage

        stage = OrientationStage()
        cfg = Config(input_dir=Path("/tmp"))
        assert stage.should_skip(cfg) is False


# ---------------------------------------------------------------------------
# TestOrientationProcessImage
# ---------------------------------------------------------------------------

class TestOrientationProcessImage:
    """Test process_image() for various orientation scenarios."""

    def test_upright_music_page_stays_upright(self):
        """A portrait music page with horizontal staff lines stays as-is."""
        from lpacleaner.stages.orientation import OrientationStage

        stage = OrientationStage()
        img = make_music_page(width=300, height=400)
        cfg = Config(input_dir=Path("/tmp"))

        result_img, meta = stage.process_image(img, {}, cfg)

        assert result_img.shape == img.shape
        assert meta["stage"] == "orientation"
        assert "staff_lines" in meta["orientation_method"]

    def test_sideways_music_page_rotated(self):
        """A music page stored sideways (staff lines vertical) should be rotated."""
        from lpacleaner.stages.orientation import OrientationStage

        stage = OrientationStage()
        page = make_music_page(width=300, height=400)
        sideways = cv2.rotate(page, cv2.ROTATE_90_CLOCKWISE)
        cfg = Config(input_dir=Path("/tmp"))

        result_img, meta = stage.process_image(sideways, {}, cfg)

        assert result_img.shape[0] > result_img.shape[1]
        assert "staff_lines" in meta["orientation_method"]

    def test_landscape_music_page_with_horizontal_lines_stays(self):
        """A landscape image with horizontal staff lines should NOT be axis-rotated."""
        from lpacleaner.stages.orientation import OrientationStage

        stage = OrientationStage()
        img = make_music_page(width=400, height=300)
        cfg = Config(input_dir=Path("/tmp"))

        result_img, meta = stage.process_image(img, {}, cfg)

        assert "staff_lines" in meta["orientation_method"]

    def test_text_page_portrait_stays(self):
        """A portrait text page stays portrait (horizontal text lines count too)."""
        from lpacleaner.stages.orientation import OrientationStage

        stage = OrientationStage()
        img = make_text_page(width=300, height=400)
        cfg = Config(input_dir=Path("/tmp"))

        result_img, meta = stage.process_image(img, {}, cfg)

        assert result_img.shape[:2] == img.shape[:2] or result_img.shape[:2] == img.shape[:2][::-1]

    def test_blank_page_landscape_gets_portrait_fallback(self):
        """A landscape blank page (no lines at all) gets portrait fallback."""
        from lpacleaner.stages.orientation import OrientationStage

        stage = OrientationStage()
        img = np.full((300, 400, 3), (230, 220, 200), dtype=np.uint8)
        cfg = Config(input_dir=Path("/tmp"))

        result_img, meta = stage.process_image(img, {}, cfg)

        assert result_img.shape[0] > result_img.shape[1]
        assert "portrait_fallback" in meta["orientation_method"]

    def test_polarity_flips_upside_down_page(self):
        """A music page with red title at the bottom (upside-down) gets flipped."""
        from lpacleaner.stages.orientation import _correct_polarity

        img = np.full((400, 300, 3), (230, 220, 200), dtype=np.uint8)
        # Scatter small red blocks in the bottom third (simulating
        # title characters -- real text is narrow, not page-wide)
        red = (0, 0, 200)
        for x in range(50, 250, 20):
            img[330:360, x:x + 12] = red
        cfg = Config(input_dir=Path("/tmp"), staff_color_hue=0, staff_color_range=15)

        result, did_flip = _correct_polarity(img, cfg)
        assert did_flip is True

    def test_polarity_keeps_right_side_up(self):
        """A music page with red title at the top stays as-is."""
        from lpacleaner.stages.orientation import _correct_polarity

        img = np.full((400, 300, 3), (230, 220, 200), dtype=np.uint8)
        red = (0, 0, 200)
        for x in range(50, 250, 20):
            img[30:60, x:x + 12] = red
        cfg = Config(input_dir=Path("/tmp"), staff_color_hue=0, staff_color_range=15)

        result, did_flip = _correct_polarity(img, cfg)
        assert did_flip is False

    def test_computes_focus_score(self):
        from lpacleaner.stages.orientation import OrientationStage

        stage = OrientationStage()
        img = make_music_page(width=300, height=400)
        cfg = Config(input_dir=Path("/tmp"))

        _, meta = stage.process_image(img, {}, cfg)

        assert "focus_score" in meta
        assert isinstance(meta["focus_score"], float)
        assert meta["focus_score"] > 0

    def test_flags_blurry_image(self):
        from lpacleaner.stages.orientation import OrientationStage

        stage = OrientationStage()
        img = make_music_page(width=300, height=400)
        blurry = cv2.GaussianBlur(img, (31, 31), 10)
        cfg = Config(input_dir=Path("/tmp"))

        _, meta = stage.process_image(blurry, {}, cfg)

        assert meta["focus_score"] < meta.get("focus_threshold", 100.0)
        assert meta["is_blurry"] is True

    def test_sharp_image_not_flagged(self):
        from lpacleaner.stages.orientation import OrientationStage

        stage = OrientationStage()
        img = make_music_page(width=300, height=400)
        cfg = Config(input_dir=Path("/tmp"))

        _, meta = stage.process_image(img, {}, cfg)

        assert meta["is_blurry"] is False

    def test_metadata_includes_orientation_method(self):
        from lpacleaner.stages.orientation import OrientationStage

        stage = OrientationStage()
        img = make_music_page(width=300, height=400)
        cfg = Config(input_dir=Path("/tmp"))

        _, meta = stage.process_image(img, {}, cfg)

        assert "orientation_method" in meta


# ---------------------------------------------------------------------------
# TestOrientationStageRun
# ---------------------------------------------------------------------------

@pytest.mark.slow
class TestOrientationStageRun:
    """Integration tests for OrientationStage.run()."""

    def test_produces_checkpoint_directory(self, tmp_path):
        from lpacleaner.stages.orientation import OrientationStage

        input_dir = _setup_stage_input(tmp_path, {
            "IMG_0001.png": make_music_page(width=400, height=300),
        })
        cfg = Config(input_dir=input_dir, output_dir=tmp_path)
        state = PipelineState(tmp_path)
        stage = OrientationStage()

        stage.run(input_dir, tmp_path, cfg, state)

        assert (tmp_path / "02_oriented").exists()

    def test_processes_all_images(self, tmp_path):
        from lpacleaner.stages.orientation import OrientationStage

        input_dir = _setup_stage_input(tmp_path, {
            "IMG_0001.png": make_music_page(width=400, height=300),
            "IMG_0002.png": make_music_page(width=400, height=300),
            "IMG_0003.png": make_music_page(width=400, height=300),
        })
        cfg = Config(input_dir=input_dir, output_dir=tmp_path)
        state = PipelineState(tmp_path)
        stage = OrientationStage()

        result = stage.run(input_dir, tmp_path, cfg, state)

        assert result.processed == 3
        assert result.failed == 0
        out_files = list((tmp_path / "02_oriented").glob("*.png"))
        assert len(out_files) == 3

    def test_writes_metadata_with_focus_score(self, tmp_path):
        from lpacleaner.stages.orientation import OrientationStage

        input_dir = _setup_stage_input(tmp_path, {
            "IMG_0001.png": make_music_page(width=400, height=300),
        })
        cfg = Config(input_dir=input_dir, output_dir=tmp_path)
        state = PipelineState(tmp_path)
        stage = OrientationStage()

        stage.run(input_dir, tmp_path, cfg, state)

        sidecar = tmp_path / "02_oriented" / "IMG_0001.json"
        assert sidecar.exists()
        meta = json.loads(sidecar.read_text())
        assert "focus_score" in meta
        assert meta["stage"] == "orientation"

    def test_resume_skips_completed(self, tmp_path):
        from lpacleaner.stages.orientation import OrientationStage

        input_dir = _setup_stage_input(tmp_path, {
            "IMG_0001.png": make_music_page(width=400, height=300),
            "IMG_0002.png": make_music_page(width=400, height=300),
        })
        cfg = Config(input_dir=input_dir, output_dir=tmp_path)
        state = PipelineState(tmp_path)
        stage = OrientationStage()

        stage.run(input_dir, tmp_path, cfg, state)

        result2 = stage.run(input_dir, tmp_path, cfg, state)
        assert result2.skipped == 2
        assert result2.processed == 0

    def test_returns_stage_result(self, tmp_path):
        from lpacleaner.stages.orientation import OrientationStage

        input_dir = _setup_stage_input(tmp_path, {
            "IMG_0001.png": make_music_page(width=400, height=300),
        })
        cfg = Config(input_dir=input_dir, output_dir=tmp_path)
        state = PipelineState(tmp_path)
        stage = OrientationStage()

        result = stage.run(input_dir, tmp_path, cfg, state)

        assert isinstance(result, StageResult)
        assert result.stage_name == "orientation"
