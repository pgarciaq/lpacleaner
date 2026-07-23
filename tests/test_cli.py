"""Tests for the CLI run command and stage selection."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest
from click.testing import CliRunner

from ghh.cli import main
from ghh.stages import parse_stage_spec, get_stages, ALL_STAGE_NUMBERS
from tests.conftest import make_music_page


# ---------------------------------------------------------------------------
# parse_stage_spec
# ---------------------------------------------------------------------------

class TestParseStageSpec:
    def test_single_number(self):
        assert parse_stage_spec("0") == [0]

    def test_comma_separated(self):
        assert parse_stage_spec("0,1,2") == [0, 1, 2]

    def test_range(self):
        assert parse_stage_spec("0-2") == [0, 1, 2]

    def test_mixed_ranges_and_numbers(self):
        assert parse_stage_spec("0,2-4,6") == [0, 2, 3, 4, 6]

    def test_deduplicates(self):
        assert parse_stage_spec("0,0,1") == [0, 1]

    def test_sorted_output(self):
        assert parse_stage_spec("2,0,1") == [0, 1, 2]

    def test_spaces_tolerated(self):
        assert parse_stage_spec(" 0 , 1 , 2 ") == [0, 1, 2]

    def test_invalid_number_raises(self):
        import click

        with pytest.raises(click.BadParameter, match="Invalid stage number"):
            parse_stage_spec("abc")

    def test_invalid_range_raises(self):
        import click

        with pytest.raises(click.BadParameter, match="Invalid range"):
            parse_stage_spec("a-b")

    def test_reversed_range_raises(self):
        import click

        with pytest.raises(click.BadParameter, match="start > end"):
            parse_stage_spec("3-1")


# ---------------------------------------------------------------------------
# get_stages
# ---------------------------------------------------------------------------

class TestGetStages:
    def test_returns_all_when_none(self):
        stages = get_stages(None)
        assert len(stages) == len(ALL_STAGE_NUMBERS)
        for s, n in zip(stages, ALL_STAGE_NUMBERS):
            assert s.number == n

    def test_returns_subset(self):
        stages = get_stages([0, 2])
        assert [s.number for s in stages] == [0, 2]

    def test_sorted_regardless_of_input(self):
        stages = get_stages([2, 0, 1])
        assert [s.number for s in stages] == [0, 1, 2]

    def test_unknown_number_raises(self):
        with pytest.raises(ValueError, match="Unknown stage number"):
            get_stages([99])

    def test_returns_instances(self):
        from ghh.pipeline import BaseStage

        stages = get_stages([0])
        assert isinstance(stages[0], BaseStage)


# ---------------------------------------------------------------------------
# CLI run command
# ---------------------------------------------------------------------------

def _prepare_input(tmp_path: Path, n: int = 2) -> Path:
    """Create an input dir with synthetic music page images."""
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    for i in range(n):
        img = make_music_page(width=400, height=300)
        cv2.imwrite(str(input_dir / f"IMG_{i:04d}.jpg"), img)
    return input_dir


class TestCLIRun:
    def test_run_all_stages(self, tmp_path):
        input_dir = _prepare_input(tmp_path)
        output_dir = tmp_path / "output"
        runner = CliRunner()

        result = runner.invoke(main, [
            "run", str(input_dir), "-o", str(output_dir), "-q",
        ])

        assert result.exit_code == 0, result.output
        assert "Pipeline Summary" in result.output
        assert (output_dir / "00_preprocessed").is_dir()
        assert (output_dir / "01_stitched").is_dir()
        assert (output_dir / "02_oriented").is_dir()

    def test_run_single_stage(self, tmp_path):
        input_dir = _prepare_input(tmp_path)
        output_dir = tmp_path / "output"
        runner = CliRunner()

        result = runner.invoke(main, [
            "run", str(input_dir), "-o", str(output_dir),
            "--stages", "0", "-q",
        ])

        assert result.exit_code == 0, result.output
        assert (output_dir / "00_preprocessed").is_dir()
        assert not (output_dir / "01_stitched").exists()

    def test_run_stage_range(self, tmp_path):
        input_dir = _prepare_input(tmp_path)
        output_dir = tmp_path / "output"
        runner = CliRunner()

        result = runner.invoke(main, [
            "run", str(input_dir), "-o", str(output_dir),
            "--stages", "0-1", "-q",
        ])

        assert result.exit_code == 0, result.output
        assert (output_dir / "00_preprocessed").is_dir()
        assert (output_dir / "01_stitched").is_dir()
        assert not (output_dir / "02_oriented").exists()

    def test_run_later_stage_uses_previous_checkpoint(self, tmp_path):
        """Running stage 2 alone works if stage 1 output already exists."""
        input_dir = _prepare_input(tmp_path)
        output_dir = tmp_path / "output"
        runner = CliRunner()

        runner.invoke(main, [
            "run", str(input_dir), "-o", str(output_dir),
            "--stages", "0-1", "-q",
        ])

        result = runner.invoke(main, [
            "run", str(input_dir), "-o", str(output_dir),
            "--stages", "2", "-q",
        ])

        assert result.exit_code == 0, result.output
        assert (output_dir / "02_oriented").is_dir()

    def test_run_later_stage_without_checkpoint_skips(self, tmp_path):
        """Running stage 2 alone without prior checkpoints skips gracefully."""
        input_dir = _prepare_input(tmp_path)
        output_dir = tmp_path / "output"
        output_dir.mkdir()
        runner = CliRunner()

        result = runner.invoke(main, [
            "run", str(input_dir), "-o", str(output_dir),
            "--stages", "2", "-q",
        ])

        assert result.exit_code == 0, result.output
        assert "no input checkpoint found" in result.output

    def test_invalid_stage_number_shows_error(self, tmp_path):
        input_dir = _prepare_input(tmp_path)
        runner = CliRunner()

        result = runner.invoke(main, [
            "run", str(input_dir), "--stages", "99",
        ])

        assert result.exit_code != 0
        assert "Unknown stage number" in result.output

    def test_verbose_flag(self, tmp_path):
        input_dir = _prepare_input(tmp_path)
        output_dir = tmp_path / "output"
        runner = CliRunner()

        result = runner.invoke(main, [
            "run", str(input_dir), "-o", str(output_dir),
            "--stages", "0", "-v",
        ])

        assert result.exit_code == 0, result.output

    def test_default_output_dir(self, tmp_path):
        input_dir = _prepare_input(tmp_path)
        runner = CliRunner()

        result = runner.invoke(main, [
            "run", str(input_dir), "--stages", "0", "-q",
        ])

        assert result.exit_code == 0, result.output
        expected_output = input_dir.parent / f"{input_dir.name}_output"
        assert expected_output.is_dir()

    def test_saves_pipeline_state(self, tmp_path):
        input_dir = _prepare_input(tmp_path)
        output_dir = tmp_path / "output"
        runner = CliRunner()

        runner.invoke(main, [
            "run", str(input_dir), "-o", str(output_dir),
            "--stages", "0", "-q",
        ])

        assert (output_dir / "pipeline.json").exists()
