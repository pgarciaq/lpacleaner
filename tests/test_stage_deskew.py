"""Tests for Stage 7 (DeskewStage): skew angle detection and correction.

Covers the BaseStage contract, staff-line angle detection, projection
profile fallback, skip threshold, max angle clamping, trim_to_content,
metadata, and integration with the run() loop.
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from lpacleaner.config import Config
from lpacleaner.pipeline import BaseStage, PipelineState
from lpacleaner.stages.deskew import DeskewStage, _projection_profile_angle

from tests.conftest import make_music_page


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _cfg(tmp_path: Path, **overrides) -> Config:
    return Config(input_dir=tmp_path, **overrides)


def _make_skewed_image(
    angle_deg: float,
    width: int = 600,
    height: int = 800,
    bg_color: tuple[int, int, int] = (230, 225, 220),
    line_color: tuple[int, int, int] = (40, 40, 40),
    n_lines: int = 20,
) -> np.ndarray:
    """Create a synthetic image with horizontal lines rotated by angle_deg."""
    img = np.full((height, width, 3), bg_color, dtype=np.uint8)
    spacing = height // (n_lines + 1)
    for i in range(1, n_lines + 1):
        y = i * spacing
        cv2.line(img, (50, y), (width - 50, y), line_color, 2)

    if abs(angle_deg) > 0.01:
        cx, cy = width / 2, height / 2
        M = cv2.getRotationMatrix2D((cx, cy), angle_deg, 1.0)
        img = cv2.warpAffine(
            img, M, (width, height),
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=bg_color,
        )
    return img


def _save_test_image(path: Path, img: np.ndarray) -> None:
    cv2.imwrite(str(path), img)


def _setup_stage_input(
    tmp_path: Path,
    images: dict[str, np.ndarray],
) -> Path:
    input_dir = tmp_path / "06_content_area"
    input_dir.mkdir()
    for name, img in images.items():
        _save_test_image(input_dir / name, img)
    return input_dir


# ---------------------------------------------------------------------------
# TestDeskewStageContract
# ---------------------------------------------------------------------------

class TestDeskewStageContract:

    def test_has_correct_name(self):
        assert DeskewStage().name == "deskew"

    def test_has_correct_number(self):
        assert DeskewStage().number == 7

    def test_has_correct_checkpoint_name(self):
        assert DeskewStage().checkpoint_name == "07_deskewed"

    def test_has_correct_error_class(self):
        assert DeskewStage().error_class == "skippable"

    def test_is_basestage_subclass(self):
        assert issubclass(DeskewStage, BaseStage)

    def test_registered_in_stage_registry(self):
        from lpacleaner.stages import STAGE_BY_NUMBER
        assert 7 in STAGE_BY_NUMBER
        assert STAGE_BY_NUMBER[7] is DeskewStage


# ---------------------------------------------------------------------------
# TestDeskewSkipThreshold
# ---------------------------------------------------------------------------

class TestDeskewSkipThreshold:

    def test_skips_when_angle_below_threshold(self, tmp_path):
        img = _make_skewed_image(0.0)
        cfg = _cfg(tmp_path, deskew_skip_threshold=2.0)
        stage = DeskewStage()
        result, meta = stage.process_image(img, {}, cfg)
        assert meta["method"] == "skipped"

    def test_unskewed_image_passes_through_similar(self, tmp_path):
        img = _make_skewed_image(0.0)
        cfg = _cfg(tmp_path)
        stage = DeskewStage()
        result, meta = stage.process_image(img, {}, cfg)
        assert result.shape[0] > 0 and result.shape[1] > 0


# ---------------------------------------------------------------------------
# TestDeskewRotation
# ---------------------------------------------------------------------------

class TestDeskewRotation:

    def test_corrects_small_skew(self, tmp_path):
        angle = 2.0
        img = _make_skewed_image(angle)
        cfg = _cfg(tmp_path)
        stage = DeskewStage()
        result, meta = stage.process_image(img, {}, cfg)
        assert abs(meta["skew_angle"]) > 0.5
        assert meta["method"] in ("staff_lines", "projection_profile")

    def test_corrects_negative_skew(self, tmp_path):
        angle = -2.0
        img = _make_skewed_image(angle)
        cfg = _cfg(tmp_path)
        stage = DeskewStage()
        result, meta = stage.process_image(img, {}, cfg)
        assert abs(meta["skew_angle"]) > 0.5
        assert meta["method"] in ("staff_lines", "projection_profile")


# ---------------------------------------------------------------------------
# TestDeskewMaxAngle
# ---------------------------------------------------------------------------

class TestDeskewMaxAngle:

    def test_clamps_to_max_angle(self, tmp_path):
        img = _make_skewed_image(0.0)
        cfg = _cfg(tmp_path, deskew_max_angle=3.0)
        stage = DeskewStage()
        result, meta = stage.process_image(img, {}, cfg)
        assert abs(meta["skew_angle"]) <= 3.0


# ---------------------------------------------------------------------------
# TestProjectionProfile
# ---------------------------------------------------------------------------

class TestProjectionProfile:

    def test_detects_angle_on_text_page(self, tmp_path):
        text_img = np.full((800, 600, 3), (230, 225, 220), dtype=np.uint8)
        for y in range(100, 700, 30):
            cv2.putText(
                text_img, "Sample text line here",
                (50, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (40, 40, 40), 1,
            )
        skew_deg = 2.0
        cx, cy = text_img.shape[1] / 2, text_img.shape[0] / 2
        M = cv2.getRotationMatrix2D((cx, cy), skew_deg, 1.0)
        skewed_text = cv2.warpAffine(
            text_img, M, (text_img.shape[1], text_img.shape[0]),
            borderMode=cv2.BORDER_CONSTANT, borderValue=(230, 225, 220),
        )

        cfg = _cfg(tmp_path)
        angle = _projection_profile_angle(skewed_text, cfg)
        assert abs(angle) > 0.5

    def test_returns_zero_for_straight_text(self, tmp_path):
        text_img = np.full((400, 400, 3), (230, 230, 230), dtype=np.uint8)
        for y in range(50, 350, 25):
            cv2.line(text_img, (30, y), (370, y), (40, 40, 40), 1)
        cfg = _cfg(tmp_path)
        angle = _projection_profile_angle(text_img, cfg)
        assert abs(angle) < 1.0


# ---------------------------------------------------------------------------
# TestDeskewMetadata
# ---------------------------------------------------------------------------

class TestDeskewMetadata:

    def test_metadata_has_required_fields(self, tmp_path):
        img = _make_skewed_image(1.0)
        cfg = _cfg(tmp_path)
        _, meta = DeskewStage().process_image(img, {}, cfg)
        assert "stage" in meta
        assert meta["stage"] == "deskew"
        assert "method" in meta
        assert "skew_angle" in meta

    def test_forwards_page_type(self, tmp_path):
        img = _make_skewed_image(0.0)
        cfg = _cfg(tmp_path)
        _, meta = DeskewStage().process_image(
            img, {"page_type": "music"}, cfg,
        )
        assert meta["page_type"] == "music"

    def test_no_page_type_when_absent(self, tmp_path):
        img = _make_skewed_image(0.0)
        cfg = _cfg(tmp_path)
        _, meta = DeskewStage().process_image(img, {}, cfg)
        assert "page_type" not in meta


# ---------------------------------------------------------------------------
# TestDeskewTrim
# ---------------------------------------------------------------------------

class TestDeskewTrim:

    def test_output_has_no_large_black_borders(self, tmp_path):
        img = _make_skewed_image(3.0)
        cfg = _cfg(tmp_path)
        result, _ = DeskewStage().process_image(img, {}, cfg)
        corners = [
            result[0, 0],
            result[0, -1],
            result[-1, 0],
            result[-1, -1],
        ]
        for corner in corners:
            assert not all(c < 10 for c in corner), "Black corner detected after trim"


# ---------------------------------------------------------------------------
# TestDeskewStageRun
# ---------------------------------------------------------------------------

class TestDeskewStageRun:

    def test_produces_checkpoint_directory(self, tmp_path):
        input_dir = _setup_stage_input(
            tmp_path, {"IMG_0001.png": _make_skewed_image(1.0)},
        )
        output_dir = tmp_path / "output"
        output_dir.mkdir()
        cfg = _cfg(tmp_path)
        state = PipelineState(output_dir)
        stage = DeskewStage()

        stage.run(input_dir, output_dir, cfg, state)

        assert (output_dir / "07_deskewed").exists()
        pngs = list((output_dir / "07_deskewed").glob("*.png"))
        assert len(pngs) == 1

    def test_processes_multiple_images(self, tmp_path):
        images = {
            f"IMG_{i:04d}.png": _make_skewed_image(i * 0.5)
            for i in range(4)
        }
        input_dir = _setup_stage_input(tmp_path, images)
        output_dir = tmp_path / "output"
        output_dir.mkdir()
        cfg = _cfg(tmp_path)
        state = PipelineState(output_dir)

        result = DeskewStage().run(input_dir, output_dir, cfg, state)
        assert result.processed + result.skipped == 4

    def test_writes_metadata_sidecar(self, tmp_path):
        input_dir = _setup_stage_input(
            tmp_path, {"IMG_0001.png": _make_skewed_image(1.5)},
        )
        output_dir = tmp_path / "output"
        output_dir.mkdir()
        cfg = _cfg(tmp_path)
        state = PipelineState(output_dir)

        DeskewStage().run(input_dir, output_dir, cfg, state)

        sidecar = output_dir / "07_deskewed" / "IMG_0001.json"
        assert sidecar.exists()
        meta = json.loads(sidecar.read_text())
        assert meta["stage"] == "deskew"
        assert "skew_angle" in meta

    def test_resume_skips_completed(self, tmp_path):
        input_dir = _setup_stage_input(
            tmp_path, {"IMG_0001.png": _make_skewed_image(1.0)},
        )
        output_dir = tmp_path / "output"
        output_dir.mkdir()
        cfg = _cfg(tmp_path)
        state = PipelineState(output_dir)
        stage = DeskewStage()

        result1 = stage.run(input_dir, output_dir, cfg, state)
        assert result1.processed == 1

        result2 = stage.run(input_dir, output_dir, cfg, state)
        assert result2.skipped == 1
        assert result2.processed == 0


# ---------------------------------------------------------------------------
# TestDeskewConfigFromTOML
# ---------------------------------------------------------------------------

class TestDeskewConfigFromTOML:

    def test_deskew_section_loaded(self, tmp_path):
        toml_file = tmp_path / "book.toml"
        toml_file.write_text(
            "[deskew]\n"
            "max_angle = 3.0\n"
            "angle_step = 0.05\n"
            "skip_threshold = 0.2\n"
        )
        cfg = Config.from_toml(input_dir=tmp_path, toml_path=toml_file)
        assert cfg.deskew_max_angle == 3.0
        assert cfg.deskew_angle_step == 0.05
        assert cfg.deskew_skip_threshold == 0.2
