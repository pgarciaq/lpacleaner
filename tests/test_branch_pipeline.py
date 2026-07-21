"""Tests for the forked pipeline architecture.

Covers branch-aware config (for_branch), branch checkpoint resolution,
three-phase pipeline execution, StaffExtractStage, ScoreRenderStage,
two-source PDF assembly, and stage grouping constants.
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pytest
from click.testing import CliRunner

from ghh.config import Config
from ghh.pipeline import BaseStage, PipelineState

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _cfg(tmp_path: Path, **overrides) -> Config:
    return Config(input_dir=tmp_path, **overrides)


def _make_image(
    width: int = 200,
    height: int = 300,
    color: tuple[int, int, int] = (200, 180, 160),
) -> np.ndarray:
    return np.full((height, width, 3), color, dtype=np.uint8)


def _save_images(directory: Path, count: int = 3, prefix: str = "IMG_") -> list[Path]:
    directory.mkdir(parents=True, exist_ok=True)
    paths = []
    for i in range(count):
        name = f"{prefix}{i:04d}.png"
        img = _make_image(color=(200 - i * 20, 180, 160))
        p = directory / name
        cv2.imwrite(str(p), img)
        paths.append(p)
    return paths


def _make_staff_image(
    width: int = 600,
    height: int = 800,
) -> np.ndarray:
    """Create an image with horizontal staff lines (music page)."""
    img = np.full((height, width, 3), (230, 225, 220), dtype=np.uint8)
    for y in range(100, 700, 30):
        cv2.line(img, (50, y), (width - 50, y), (40, 40, 40), 2)
    return img


def _make_mixed_page(
    width: int = 600,
    height: int = 800,
) -> np.ndarray:
    """Create a page with staves in the middle and a picture at the top."""
    img = np.full((height, width, 3), (230, 225, 220), dtype=np.uint8)
    # Picture area at top (dark rectangle)
    cv2.rectangle(img, (50, 20), (width - 50, 200), (80, 60, 40), -1)
    # Staff lines in middle-bottom area
    for y in range(300, 700, 25):
        cv2.line(img, (50, y), (width - 50, y), (40, 40, 40), 2)
    return img


# ===========================================================================
# Branch-aware config tests
# ===========================================================================

class TestBranchConfig:

    def test_for_branch_returns_copy(self, tmp_path):
        cfg = _cfg(tmp_path)
        book_cfg = cfg.for_branch("book")
        assert book_cfg is not cfg
        assert book_cfg.input_dir == cfg.input_dir

    def test_for_branch_applies_overrides(self, tmp_path):
        cfg = _cfg(
            tmp_path,
            _branch_overrides={
                "book": {"deskew_max_angle": 3.0},
                "score": {"deskew_max_angle": 10.0},
            },
        )
        book = cfg.for_branch("book")
        score = cfg.for_branch("score")

        assert book.deskew_max_angle == 3.0
        assert score.deskew_max_angle == 10.0
        assert cfg.deskew_max_angle == 5.0  # original unchanged

    def test_for_branch_no_overrides(self, tmp_path):
        cfg = _cfg(tmp_path)
        book = cfg.for_branch("book")
        assert book.deskew_max_angle == cfg.deskew_max_angle

    def test_for_branch_unknown_override_warns(self, tmp_path, caplog):
        cfg = _cfg(
            tmp_path,
            _branch_overrides={"book": {"nonexistent_param": 42}},
        )
        import logging
        with caplog.at_level(logging.WARNING):
            cfg.for_branch("book")
        assert "Unknown branch override" in caplog.text

    def test_book_only_scores_only_fields(self, tmp_path):
        cfg = _cfg(tmp_path, book_only=True, scores_only=False)
        assert cfg.book_only is True
        assert cfg.scores_only is False

    def test_from_toml_branch_overrides(self, tmp_path):
        toml_file = tmp_path / "book.toml"
        toml_file.write_text(
            "[deskew]\n"
            "max_angle = 5.0\n"
            "\n"
            "[book.deskew]\n"
            "max_angle = 3.0\n"
            "\n"
            "[score.deskew]\n"
            "max_angle = 10.0\n"
        )
        cfg = Config.from_toml(input_dir=tmp_path, toml_path=toml_file)
        assert cfg.deskew_max_angle == 5.0

        book = cfg.for_branch("book")
        assert book.deskew_max_angle == 3.0

        score = cfg.for_branch("score")
        assert score.deskew_max_angle == 10.0


# ===========================================================================
# Profile skip tests
# ===========================================================================

class TestProfileSkips:

    def test_book_only_profile_skips_score_stages(self, tmp_path):
        cfg = _cfg(tmp_path, profile="book-only")
        assert cfg.should_skip_stage("content_area") is True
        assert cfg.should_skip_stage("staff_extract") is True
        assert cfg.should_skip_stage("omr") is True
        assert cfg.should_skip_stage("deskew") is False

    def test_scores_only_profile_skips_ocr(self, tmp_path):
        cfg = _cfg(tmp_path, profile="scores-only")
        assert cfg.should_skip_stage("ocr") is True
        assert cfg.should_skip_stage("content_area") is False
        assert cfg.should_skip_stage("deskew") is False


# ===========================================================================
# Stage grouping constants tests
# ===========================================================================

class TestStageGroupings:

    def test_common_stages(self):
        from ghh.stages import COMMON_STAGE_NUMBERS
        assert COMMON_STAGE_NUMBERS == [0, 1, 2, 3, 4, 5]

    def test_book_stages(self):
        from ghh.stages import BOOK_STAGE_NUMBERS
        assert 8 in BOOK_STAGE_NUMBERS

    def test_score_stages(self):
        from ghh.stages import SCORE_STAGE_NUMBERS
        assert 6 in SCORE_STAGE_NUMBERS
        assert 7 in SCORE_STAGE_NUMBERS
        assert 13 in SCORE_STAGE_NUMBERS
        assert 14 in SCORE_STAGE_NUMBERS

    def test_final_stages(self):
        from ghh.stages import FINAL_STAGE_NUMBERS
        assert 14 not in FINAL_STAGE_NUMBERS
        assert 15 in FINAL_STAGE_NUMBERS

    def test_no_stage_in_multiple_groups(self):
        from ghh.stages import COMMON_STAGE_NUMBERS, FINAL_STAGE_NUMBERS

        common = set(COMMON_STAGE_NUMBERS)
        final = set(FINAL_STAGE_NUMBERS)
        assert common.isdisjoint(final)


# ===========================================================================
# StaffExtractStage tests
# ===========================================================================

class TestStaffExtractStageContract:

    def test_has_correct_name(self):
        from ghh.stages.staff_extract import StaffExtractStage
        assert StaffExtractStage().name == "staff_extract"

    def test_has_correct_number(self):
        from ghh.stages.staff_extract import StaffExtractStage
        assert StaffExtractStage().number == 7

    def test_has_correct_checkpoint_name(self):
        from ghh.stages.staff_extract import StaffExtractStage
        assert StaffExtractStage().checkpoint_name == "07_staff_extract"

    def test_is_basestage_subclass(self):
        from ghh.stages.staff_extract import StaffExtractStage
        assert issubclass(StaffExtractStage, BaseStage)

    def test_registered_in_stage_registry(self):
        from ghh.stages import STAGE_BY_NUMBER
        assert 7 in STAGE_BY_NUMBER

    def test_symlink_unchanged(self):
        from ghh.stages.staff_extract import StaffExtractStage
        assert StaffExtractStage().symlink_unchanged is True


class TestStaffExtractProcessing:

    def test_passthrough_on_blank_page(self, tmp_path):
        from ghh.stages.staff_extract import StaffExtractStage
        img = _make_image(600, 800)
        stage = StaffExtractStage()
        result, meta = stage.process_image(img, {}, _cfg(tmp_path))
        assert meta["staff_extract_action"] == "passthrough"
        assert result.shape == img.shape

    def test_is_unchanged_for_passthrough(self, tmp_path):
        from ghh.stages.staff_extract import StaffExtractStage
        stage = StaffExtractStage()
        assert stage.is_unchanged({"staff_extract_action": "passthrough"}) is True
        assert stage.is_unchanged({"staff_extract_action": "cropped"}) is False
        assert stage.is_unchanged({}) is False

    def test_staff_page_detected(self, tmp_path):
        from ghh.stages.staff_extract import StaffExtractStage
        img = _make_staff_image()
        stage = StaffExtractStage()
        result, meta = stage.process_image(img, {}, _cfg(tmp_path))
        # Should either crop or pass through depending on coverage
        assert "staff_extract_action" in meta

    def test_run_produces_checkpoint(self, tmp_path):
        from ghh.stages.staff_extract import StaffExtractStage

        input_dir = tmp_path / "input"
        _save_images(input_dir, count=2)
        output_dir = tmp_path / "output"
        output_dir.mkdir()

        stage = StaffExtractStage()
        state = PipelineState(output_dir)
        result = stage.run(input_dir, output_dir, _cfg(tmp_path), state)

        assert (output_dir / "07_staff_extract").exists()
        assert result.processed + result.skipped == 2


# ===========================================================================
# ScoreRenderStage tests
# ===========================================================================

class TestScoreRenderStageContract:

    def test_has_correct_name(self):
        from ghh.stages.score_render import ScoreRenderStage
        assert ScoreRenderStage().name == "score_render"

    def test_has_correct_number(self):
        from ghh.stages.score_render import ScoreRenderStage
        assert ScoreRenderStage().number == 14

    def test_has_correct_checkpoint_name(self):
        from ghh.stages.score_render import ScoreRenderStage
        assert ScoreRenderStage().checkpoint_name == "14_score_render"

    def test_is_basestage_subclass(self):
        from ghh.stages.score_render import ScoreRenderStage
        assert issubclass(ScoreRenderStage, BaseStage)

    def test_registered_in_stage_registry(self):
        from ghh.stages import STAGE_BY_NUMBER
        assert 14 in STAGE_BY_NUMBER

    def test_process_image_raises(self, tmp_path):
        from ghh.stages.score_render import ScoreRenderStage
        stage = ScoreRenderStage()
        with pytest.raises(NotImplementedError):
            stage.process_image(
                np.zeros((10, 10, 3), dtype=np.uint8), {}, _cfg(tmp_path)
            )

    def test_should_skip_when_book_only(self, tmp_path):
        from ghh.stages.score_render import ScoreRenderStage
        cfg = _cfg(tmp_path, book_only=True)
        assert ScoreRenderStage().should_skip(cfg) is True

    def test_should_not_skip_by_default(self, tmp_path):
        from ghh.stages.score_render import ScoreRenderStage
        cfg = _cfg(tmp_path, book_only=False)
        assert ScoreRenderStage().should_skip(cfg) is False


class TestScoreRenderNoOmrOutput:

    def test_returns_empty_result_when_no_omr_dir(self, tmp_path):
        from ghh.stages.score_render import ScoreRenderStage

        output_dir = tmp_path / "output"
        output_dir.mkdir()
        state = PipelineState(output_dir)
        stage = ScoreRenderStage()

        result = stage.run(tmp_path, output_dir, _cfg(tmp_path), state)
        assert result.processed == 0
        assert result.failed == 0

    def test_returns_empty_result_when_no_gabc_files(self, tmp_path):
        from ghh.stages.score_render import ScoreRenderStage

        output_dir = tmp_path / "output"
        omr_dir = output_dir / "score" / "13_omr"
        omr_dir.mkdir(parents=True)
        state = PipelineState(output_dir)
        stage = ScoreRenderStage()

        result = stage.run(tmp_path, output_dir, _cfg(tmp_path), state)
        assert result.processed == 0


# ===========================================================================
# Branch checkpoint resolution tests
# ===========================================================================

class TestBranchCheckpointResolution:

    def test_finds_previous_in_branch_dir(self, tmp_path):
        from ghh.cli import _find_previous_checkpoint
        from ghh.stages import get_stages

        output_dir = tmp_path / "output"
        branch_dir = output_dir / "score"

        # Create score/06_content checkpoint (matches ContentAreaStage.checkpoint_name)
        content_dir = branch_dir / "06_content"
        _save_images(content_dir, count=1)

        stages = get_stages([6, 8])
        result = _find_previous_checkpoint(
            8, stages, branch_dir, fallback_dir=output_dir,
        )
        assert result is not None
        assert "06_content" in str(result)

    def test_falls_back_to_common_dir(self, tmp_path):
        from ghh.cli import _find_previous_checkpoint
        from ghh.stages import get_stages

        output_dir = tmp_path / "output"
        branch_dir = output_dir / "score"
        branch_dir.mkdir(parents=True)

        # Only common checkpoint exists
        perspective_dir = output_dir / "05_perspective"
        _save_images(perspective_dir, count=1)

        stages = get_stages([5, 6])
        result = _find_previous_checkpoint(
            6, stages, branch_dir, fallback_dir=output_dir,
        )
        assert result is not None
        assert "05_perspective" in str(result)

    def test_returns_none_when_no_checkpoint(self, tmp_path):
        from ghh.cli import _find_previous_checkpoint
        from ghh.stages import get_stages

        output_dir = tmp_path / "output"
        output_dir.mkdir()

        stages = get_stages([0, 8])
        result = _find_previous_checkpoint(8, stages, output_dir)
        assert result is None


# ===========================================================================
# Two-source PDF assembly tests
# ===========================================================================

class TestTwoSourcePDFAssembly:

    def test_uses_book_branch_images(self, tmp_path):
        from ghh.stages.pdf_assembly import _find_book_images

        output_dir = tmp_path / "output"
        book_dir = output_dir / "book" / "08_deskewed"
        _save_images(book_dir, count=3)

        input_dir = tmp_path / "input"
        input_dir.mkdir()

        cfg = _cfg(tmp_path)
        images = _find_book_images(input_dir, output_dir, cfg)
        assert len(images) == 3

    def test_fallback_to_input_dir(self, tmp_path):
        from ghh.stages.pdf_assembly import _find_book_images

        output_dir = tmp_path / "output"
        output_dir.mkdir()

        input_dir = tmp_path / "input"
        _save_images(input_dir, count=5)

        cfg = _cfg(tmp_path)
        images = _find_book_images(input_dir, output_dir, cfg)
        assert len(images) == 5

    def test_scores_only_returns_empty_book_images(self, tmp_path):
        from ghh.stages.pdf_assembly import _find_book_images

        output_dir = tmp_path / "output"
        output_dir.mkdir()

        input_dir = tmp_path / "input"
        _save_images(input_dir, count=5)

        cfg = _cfg(tmp_path, scores_only=True)
        images = _find_book_images(input_dir, output_dir, cfg)
        assert len(images) == 0

    def test_finds_score_annex_images(self, tmp_path):
        from ghh.stages.pdf_assembly import _find_score_annex_images

        output_dir = tmp_path / "output"
        score_render_dir = output_dir / "14_score_render"
        _save_images(score_render_dir, count=2)

        cfg = _cfg(tmp_path)
        images = _find_score_annex_images(output_dir, cfg)
        assert len(images) == 2

    def test_book_only_skips_score_annex(self, tmp_path):
        from ghh.stages.pdf_assembly import _find_score_annex_images

        output_dir = tmp_path / "output"
        score_render_dir = output_dir / "14_score_render"
        _save_images(score_render_dir, count=2)

        cfg = _cfg(tmp_path, book_only=True)
        images = _find_score_annex_images(output_dir, cfg)
        assert len(images) == 0

    def test_no_score_render_dir(self, tmp_path):
        from ghh.stages.pdf_assembly import _find_score_annex_images

        output_dir = tmp_path / "output"
        output_dir.mkdir()

        cfg = _cfg(tmp_path)
        images = _find_score_annex_images(output_dir, cfg)
        assert len(images) == 0

    def test_pdf_metadata_includes_score_annex_count(self, tmp_path):
        from ghh.stages.pdf_assembly import PDFAssemblyStage

        input_dir = tmp_path / "input"
        _save_images(input_dir, count=3)

        output_dir = tmp_path / "output"
        output_dir.mkdir()

        # Add score render images
        score_dir = output_dir / "14_score_render"
        _save_images(score_dir, count=2)

        book_dir = tmp_path / "book"
        book_dir.mkdir()
        cfg = _cfg(tmp_path)
        state = PipelineState(output_dir)
        stage = PDFAssemblyStage()

        result = stage.run(input_dir, output_dir, cfg, state)
        assert result.processed == 5  # 3 book + 2 score

        sidecar = output_dir / f"{cfg.input_dir.name}.pdf.json"
        meta = json.loads(sidecar.read_text())
        assert meta["page_count"] == 3
        assert meta["score_annex_count"] == 2

    def test_copies_gabc_sources(self, tmp_path):
        from ghh.stages.pdf_assembly import _copy_gabc_sources

        output_dir = tmp_path / "output"
        omr_dir = output_dir / "score" / "13_omr"
        omr_dir.mkdir(parents=True)

        (omr_dir / "IMG_0001.gabc").write_text("(c4) test")
        (omr_dir / "IMG_0002.gabc").write_text("(c4) test2")

        pdf_path = output_dir / "test.pdf"
        _copy_gabc_sources(output_dir, pdf_path)

        scores_dir = output_dir / "scores"
        assert scores_dir.exists()
        assert (scores_dir / "IMG_0001.gabc").exists()
        assert (scores_dir / "IMG_0002.gabc").exists()


# ===========================================================================
# Compare viewer branch awareness tests
# ===========================================================================

class TestCompareViewerBranchAwareness:

    def test_discover_book_includes_branch_dirs(self, tmp_path):
        from ghh.compare import discover_book

        output_dir = tmp_path / "output"
        # Common stage
        _save_images(output_dir / "05_perspective", count=2)
        # Book branch stage
        _save_images(output_dir / "book" / "08_deskewed", count=2)
        # Score branch stage
        _save_images(output_dir / "score" / "06_content", count=2)

        book = discover_book(output_dir)
        labels = book["stages"]
        assert any("book" in label.lower() for label in labels)
        assert any("score" in label.lower() for label in labels)

    def test_discover_book_qualifies_branch_labels(self, tmp_path):
        from ghh.compare import discover_book

        output_dir = tmp_path / "output"
        _save_images(output_dir / "05_perspective", count=1)
        _save_images(output_dir / "book" / "08_deskewed", count=1)

        book = discover_book(output_dir)
        deskew_labels = [s for s in book["stages"] if "Deskew" in s]
        assert len(deskew_labels) >= 1
        assert any("[book]" in s for s in deskew_labels)


# ===========================================================================
# CLI tests
# ===========================================================================

class TestCLIBranchFlags:

    def test_book_only_and_scores_only_are_mutually_exclusive(self):
        from ghh.cli import main
        runner = CliRunner()
        with runner.isolated_filesystem():
            Path("input").mkdir()
            result = runner.invoke(
                main,
                ["run", "input", "--book-only", "--scores-only"],
            )
            assert result.exit_code != 0
            assert "Cannot use both" in result.output or result.exit_code == 2

    def test_stages_command_includes_new_stages(self):
        from ghh.cli import main
        runner = CliRunner()
        result = runner.invoke(main, ["stages"])
        assert result.exit_code == 0
        assert "staff_extract" in result.output
        assert "score_render" in result.output
