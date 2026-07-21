"""Tests for Stage 14 (OmrStage): Optical Music Recognition.

All chant-omr inference is mocked -- these tests verify the stage
contract, page-type filtering, sidecar output, resume behaviour,
and the skip-when-unconfigured path.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import cv2
import numpy as np
import pytest

from ghh.config import Config
from ghh.pipeline import BaseStage, PipelineState
from ghh.stages.omr import OmrStage


def _cfg(tmp_path: Path, **overrides) -> Config:
    input_dir = tmp_path / "book"
    input_dir.mkdir(exist_ok=True)
    return Config(input_dir=input_dir, **overrides)


def _save_images_with_metadata(
    directory: Path,
    specs: list[tuple[str, str]],
) -> list[Path]:
    """Create PNGs with sidecar metadata.

    *specs* is a list of ``(stem, page_type)`` tuples.
    """
    directory.mkdir(parents=True, exist_ok=True)
    paths = []
    for stem, page_type in specs:
        img = np.full((100, 200, 3), 180, dtype=np.uint8)
        p = directory / f"{stem}.png"
        cv2.imwrite(str(p), img)
        sidecar = directory / f"{stem}.json"
        sidecar.write_text(json.dumps({"page_type": page_type}))
        paths.append(p)
    return paths


FAKE_GABC = "name:test;\n%%\n(c4) Ky(f)ri(gf)e(h)\n"


@pytest.fixture
def mock_chant_omr():
    """Patch chant-omr imports so tests don't need the real package."""
    mock_bundle = MagicMock()
    mock_bundle.manifest = {"config": {"max_seq_len": 512}}

    with (
        patch("ghh.stages.omr.CHANT_OMR_AVAILABLE", True),
        patch("ghh.stages.omr.load_openvino_models", create=True, return_value=mock_bundle) as mock_load,
        patch("ghh.stages.omr.prepare_inference_numpy_from_array", create=True, return_value=np.zeros((1, 3, 64, 128), dtype=np.float32)),
        patch("ghh.stages.omr.ov_predict_gabc_from_array", create=True, return_value=FAKE_GABC),
    ):
        yield mock_load, mock_bundle


class TestOmrStageContract:
    """Verify that OmrStage satisfies the BaseStage contract."""

    def test_has_correct_name(self):
        assert OmrStage().name == "omr"

    def test_has_correct_number(self):
        assert OmrStage().number == 13

    def test_has_correct_checkpoint_name(self):
        assert OmrStage().checkpoint_name == "13_omr"

    def test_has_correct_error_class(self):
        assert OmrStage().error_class == "skippable"

    def test_writes_image_is_false(self):
        assert OmrStage().writes_image is False

    def test_is_basestage_subclass(self):
        assert issubclass(OmrStage, BaseStage)

    def test_process_image_raises(self, tmp_path):
        with pytest.raises(NotImplementedError):
            OmrStage().process_image(
                np.zeros((10, 10, 3), dtype=np.uint8), {}, _cfg(tmp_path)
            )

    def test_registered_in_stage_registry(self):
        from ghh.stages import STAGE_BY_NUMBER
        assert 13 in STAGE_BY_NUMBER
        assert STAGE_BY_NUMBER[13] is OmrStage


class TestOmrStageSkipUnconfigured:
    """Stage skips when omr_model_dir is empty."""

    def test_skips_with_empty_model_dir(self, tmp_path):
        input_dir = tmp_path / "input"
        _save_images_with_metadata(input_dir, [("IMG_0001", "music")])
        output_dir = tmp_path / "output"
        output_dir.mkdir()

        cfg = _cfg(tmp_path, omr_model_dir="")
        state = PipelineState(output_dir)
        result = OmrStage().run(input_dir, output_dir, cfg, state)

        assert result.processed == 0
        assert result.failed == 0


class TestOmrStageNonMusic:
    """Non-music pages are symlinked with skip metadata."""

    def test_skips_non_music_pages(self, tmp_path, mock_chant_omr):
        input_dir = tmp_path / "input"
        _save_images_with_metadata(input_dir, [
            ("IMG_0001", "text"),
            ("IMG_0002", "blank"),
        ])
        output_dir = tmp_path / "output"
        output_dir.mkdir()

        model_dir = tmp_path / "models"
        model_dir.mkdir()
        cfg = _cfg(tmp_path, omr_model_dir=str(model_dir))
        state = PipelineState(output_dir)
        result = OmrStage().run(input_dir, output_dir, cfg, state)

        assert result.processed == 2
        assert result.failed == 0

        stage_dir = output_dir / "13_omr"
        for stem in ("IMG_0001", "IMG_0002"):
            sidecar = stage_dir / f"{stem}.json"
            assert sidecar.exists()
            meta = json.loads(sidecar.read_text())
            assert meta["omr_status"] == "skipped_non_music"

            png_link = stage_dir / f"{stem}.png"
            assert png_link.is_symlink()


class TestOmrStageMusicPages:
    """Music pages produce .gabc files."""

    def test_produces_gabc_for_music(self, tmp_path, mock_chant_omr):
        input_dir = tmp_path / "input"
        _save_images_with_metadata(input_dir, [
            ("IMG_0001", "music"),
            ("IMG_0002", "text"),
            ("IMG_0003", "music"),
        ])
        output_dir = tmp_path / "output"
        output_dir.mkdir()

        model_dir = tmp_path / "models"
        model_dir.mkdir()
        cfg = _cfg(tmp_path, omr_model_dir=str(model_dir))
        state = PipelineState(output_dir)
        result = OmrStage().run(input_dir, output_dir, cfg, state)

        assert result.processed == 3
        assert result.failed == 0

        stage_dir = output_dir / "13_omr"

        for stem in ("IMG_0001", "IMG_0003"):
            gabc = stage_dir / f"{stem}.gabc"
            assert gabc.exists()
            assert gabc.read_text(encoding="utf-8") == FAKE_GABC

            sidecar = stage_dir / f"{stem}.json"
            meta = json.loads(sidecar.read_text())
            assert meta["omr_status"] == "ok"
            assert meta["gabc_file"] == f"{stem}.gabc"

            assert (stage_dir / f"{stem}.png").is_symlink()

        text_sidecar = stage_dir / "IMG_0002.json"
        assert json.loads(text_sidecar.read_text())["omr_status"] == "skipped_non_music"

    def test_beam_width_passed_to_inference(self, tmp_path, mock_chant_omr):
        input_dir = tmp_path / "input"
        _save_images_with_metadata(input_dir, [("IMG_0001", "music")])
        output_dir = tmp_path / "output"
        output_dir.mkdir()

        model_dir = tmp_path / "models"
        model_dir.mkdir()
        cfg = _cfg(tmp_path, omr_model_dir=str(model_dir), omr_beam_width=3)
        state = PipelineState(output_dir)

        with patch("ghh.stages.omr.ov_predict_gabc_from_array", create=True, return_value=FAKE_GABC) as mock_predict:
            OmrStage().run(input_dir, output_dir, cfg, state)
            mock_predict.assert_called_once()
            _, kwargs = mock_predict.call_args
            assert kwargs["beam_width"] == 3


class TestOmrStageResume:
    """Resume: skips already-done images."""

    def test_skips_already_done(self, tmp_path, mock_chant_omr):
        input_dir = tmp_path / "input"
        _save_images_with_metadata(input_dir, [("IMG_0001", "music")])
        output_dir = tmp_path / "output"
        output_dir.mkdir()

        model_dir = tmp_path / "models"
        model_dir.mkdir()
        cfg = _cfg(tmp_path, omr_model_dir=str(model_dir))
        state = PipelineState(output_dir)

        result1 = OmrStage().run(input_dir, output_dir, cfg, state)
        assert result1.processed == 1

        result2 = OmrStage().run(input_dir, output_dir, cfg, state)
        assert result2.skipped == 1
        assert result2.processed == 0


class TestOmrStageProfileSkip:
    """OMR is skipped in geometry and quick profiles."""

    @pytest.mark.parametrize("profile", ["geometry", "quick"])
    def test_skipped_by_profile(self, profile):
        cfg = Config(input_dir=Path("/tmp/fake"), profile=profile)
        assert cfg.should_skip_stage("omr") is True

    def test_not_skipped_by_full_profile(self):
        cfg = Config(input_dir=Path("/tmp/fake"), profile="full")
        assert cfg.should_skip_stage("omr") is False

    def test_explicit_skip_flag(self):
        cfg = Config(input_dir=Path("/tmp/fake"), profile="full", skip_omr=True)
        assert cfg.should_skip_stage("omr") is True
