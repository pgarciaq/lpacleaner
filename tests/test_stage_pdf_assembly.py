"""Tests for Stage 12 (PDFAssemblyStage): PDF assembly.

Covers the BaseStage contract, JPEG and PNG compression, page count,
resume/skip, exclude_images, DPI, empty input, and input from any stage.
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from ghh.config import Config
from ghh.pipeline import BaseStage, PipelineState
from ghh.stages.pdf_assembly import PDFAssemblyStage, _collect_images

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _cfg(tmp_path: Path, **overrides) -> Config:
    """Create a Config rooted in tmp_path with optional overrides.

    Uses a fixed input_dir name so the PDF filename is predictable.
    """
    input_dir = tmp_path / "book"
    input_dir.mkdir(exist_ok=True)
    return Config(input_dir=input_dir, **overrides)


def _make_color_image(
    width: int = 200,
    height: int = 300,
    color: tuple[int, int, int] = (200, 180, 160),
) -> np.ndarray:
    return np.full((height, width, 3), color, dtype=np.uint8)


def _save_test_images(
    directory: Path,
    count: int = 3,
    prefix: str = "IMG_",
) -> list[Path]:
    directory.mkdir(parents=True, exist_ok=True)
    paths = []
    for i in range(count):
        name = f"{prefix}{i:04d}.png"
        img = _make_color_image(color=(200 - i * 20, 180, 160))
        p = directory / name
        cv2.imwrite(str(p), img)
        paths.append(p)
    return paths


def _read_pdf_page_count(pdf_path: Path) -> int:
    """Count pages by looking for /Type /Page entries in the PDF."""
    data = pdf_path.read_bytes()
    import re
    pages = re.findall(rb"/Type\s*/Page(?!s)", data)
    return len(pages)


# ---------------------------------------------------------------------------
# TestPDFAssemblyStageContract
# ---------------------------------------------------------------------------

class TestPDFAssemblyStageContract:
    """Verify that PDFAssemblyStage satisfies the BaseStage contract."""

    def test_has_correct_name(self):
        stage = PDFAssemblyStage()
        assert stage.name == "pdf_assembly"

    def test_has_correct_number(self):
        stage = PDFAssemblyStage()
        assert stage.number == 15

    def test_has_correct_checkpoint_name(self):
        stage = PDFAssemblyStage()
        assert stage.checkpoint_name == "15_pdf"

    def test_has_correct_error_class(self):
        stage = PDFAssemblyStage()
        assert stage.error_class == "fatal"

    def test_is_basestage_subclass(self):
        assert issubclass(PDFAssemblyStage, BaseStage)

    def test_process_image_raises(self, tmp_path):
        stage = PDFAssemblyStage()
        with pytest.raises(NotImplementedError):
            stage.process_image(
                np.zeros((10, 10, 3), dtype=np.uint8), {}, _cfg(tmp_path)
            )

    def test_registered_in_stage_registry(self):
        from ghh.stages import STAGE_BY_NUMBER
        assert 15 in STAGE_BY_NUMBER
        assert STAGE_BY_NUMBER[15] is PDFAssemblyStage


# ---------------------------------------------------------------------------
# TestPDFAssemblyJPEG
# ---------------------------------------------------------------------------

class TestPDFAssemblyJPEG:
    """PDF assembly with JPEG compression."""

    def test_produces_valid_pdf(self, tmp_path):
        input_dir = tmp_path / "input"
        _save_test_images(input_dir, count=3)
        output_dir = tmp_path / "output"
        output_dir.mkdir()

        cfg = _cfg(tmp_path, pdf_compression="jpeg", pdf_jpeg_quality=85)
        state = PipelineState(output_dir)
        stage = PDFAssemblyStage()

        result = stage.run(input_dir, output_dir, cfg, state)

        pdf_path = output_dir / "book.pdf"
        assert pdf_path.exists()
        assert pdf_path.read_bytes()[:5] == b"%PDF-"
        assert result.processed == 3

    def test_jpeg_smaller_than_png(self, tmp_path):
        input_dir = tmp_path / "input"
        input_dir.mkdir()
        rng = np.random.RandomState(42)
        for i in range(5):
            img = rng.randint(0, 256, (300, 200, 3), dtype=np.uint8)
            cv2.imwrite(str(input_dir / f"IMG_{i:04d}.png"), img)

        out_jpeg = tmp_path / "out_jpeg"
        out_jpeg.mkdir()
        cfg_jpeg = _cfg(tmp_path, pdf_compression="jpeg", pdf_jpeg_quality=85)
        state_j = PipelineState(out_jpeg)
        PDFAssemblyStage().run(input_dir, out_jpeg, cfg_jpeg, state_j)

        out_png = tmp_path / "out_png"
        out_png.mkdir()
        cfg_png = _cfg(tmp_path, pdf_compression="png")
        state_p = PipelineState(out_png)
        PDFAssemblyStage().run(input_dir, out_png, cfg_png, state_p)

        jpeg_size = (out_jpeg / "book.pdf").stat().st_size
        png_size = (out_png / "book.pdf").stat().st_size
        assert jpeg_size < png_size

    def test_jpeg_quality_affects_size(self, tmp_path):
        input_dir = tmp_path / "input"
        _save_test_images(input_dir, count=3)

        out_low = tmp_path / "out_low"
        out_low.mkdir()
        cfg_low = _cfg(tmp_path, pdf_compression="jpeg", pdf_jpeg_quality=30)
        state_low = PipelineState(out_low)
        PDFAssemblyStage().run(input_dir, out_low, cfg_low, state_low)

        out_high = tmp_path / "out_high"
        out_high.mkdir()
        cfg_high = _cfg(tmp_path, pdf_compression="jpeg", pdf_jpeg_quality=95)
        state_high = PipelineState(out_high)
        PDFAssemblyStage().run(input_dir, out_high, cfg_high, state_high)

        low_size = (out_low / "book.pdf").stat().st_size
        high_size = (out_high / "book.pdf").stat().st_size
        assert low_size < high_size


# ---------------------------------------------------------------------------
# TestPDFAssemblyPNG
# ---------------------------------------------------------------------------

class TestPDFAssemblyPNG:
    """PDF assembly with lossless (PNG) compression."""

    def test_produces_valid_pdf(self, tmp_path):
        input_dir = tmp_path / "input"
        _save_test_images(input_dir, count=3)
        output_dir = tmp_path / "output"
        output_dir.mkdir()

        cfg = _cfg(tmp_path, pdf_compression="png")
        state = PipelineState(output_dir)
        stage = PDFAssemblyStage()

        result = stage.run(input_dir, output_dir, cfg, state)

        pdf_path = output_dir / "book.pdf"
        assert pdf_path.exists()
        assert pdf_path.read_bytes()[:5] == b"%PDF-"
        assert result.processed == 3

    def test_page_count_matches_input(self, tmp_path):
        input_dir = tmp_path / "input"
        _save_test_images(input_dir, count=7)
        output_dir = tmp_path / "output"
        output_dir.mkdir()

        cfg = _cfg(tmp_path, pdf_compression="png")
        state = PipelineState(output_dir)
        PDFAssemblyStage().run(input_dir, output_dir, cfg, state)

        page_count = _read_pdf_page_count(output_dir / "book.pdf")
        assert page_count == 7


# ---------------------------------------------------------------------------
# TestPDFAssemblyCompressionCaseInsensitive
# ---------------------------------------------------------------------------

class TestPDFAssemblyCompressionCaseInsensitive:
    """Compression mode is case-insensitive."""

    @pytest.mark.parametrize("value", ["JPEG", "Jpeg", "jPeG"])
    def test_jpeg_variants(self, tmp_path, value):
        input_dir = tmp_path / "input"
        _save_test_images(input_dir, count=1)
        output_dir = tmp_path / "output"
        output_dir.mkdir()

        cfg = _cfg(tmp_path, pdf_compression=value)
        state = PipelineState(output_dir)
        result = PDFAssemblyStage().run(input_dir, output_dir, cfg, state)

        assert result.processed == 1
        assert (output_dir / "book.pdf").exists()

    @pytest.mark.parametrize("value", ["PNG", "Png", "pNg"])
    def test_png_variants(self, tmp_path, value):
        input_dir = tmp_path / "input"
        _save_test_images(input_dir, count=1)
        output_dir = tmp_path / "output"
        output_dir.mkdir()

        cfg = _cfg(tmp_path, pdf_compression=value)
        state = PipelineState(output_dir)
        result = PDFAssemblyStage().run(input_dir, output_dir, cfg, state)

        assert result.processed == 1
        assert (output_dir / "book.pdf").exists()


# ---------------------------------------------------------------------------
# TestPDFAssemblyResume
# ---------------------------------------------------------------------------

class TestPDFAssemblyResume:
    """Resume behavior: skips if already done."""

    def test_skips_when_already_done(self, tmp_path):
        input_dir = tmp_path / "input"
        _save_test_images(input_dir, count=3)
        output_dir = tmp_path / "output"
        output_dir.mkdir()

        cfg = _cfg(tmp_path, pdf_compression="jpeg")
        state = PipelineState(output_dir)
        stage = PDFAssemblyStage()

        result1 = stage.run(input_dir, output_dir, cfg, state)
        assert result1.processed == 3

        pdf_mtime = (output_dir / "book.pdf").stat().st_mtime

        result2 = stage.run(input_dir, output_dir, cfg, state)
        assert result2.skipped == 1
        assert result2.processed == 0

        assert (output_dir / "book.pdf").stat().st_mtime == pdf_mtime


# ---------------------------------------------------------------------------
# TestPDFAssemblyExclude
# ---------------------------------------------------------------------------

class TestPDFAssemblyExclude:
    """Respects cfg.exclude_images."""

    def test_excludes_by_filename(self, tmp_path):
        input_dir = tmp_path / "input"
        _save_test_images(input_dir, count=5)

        output_dir = tmp_path / "output"
        output_dir.mkdir()

        cfg = _cfg(
            tmp_path,
            pdf_compression="jpeg",
            exclude_images=["IMG_0001.png", "IMG_0003.png"],
        )
        state = PipelineState(output_dir)
        result = PDFAssemblyStage().run(input_dir, output_dir, cfg, state)

        assert result.processed == 3

    def test_excludes_by_stem(self, tmp_path):
        input_dir = tmp_path / "input"
        _save_test_images(input_dir, count=5)

        output_dir = tmp_path / "output"
        output_dir.mkdir()

        cfg = _cfg(
            tmp_path,
            pdf_compression="jpeg",
            exclude_images=["IMG_0002"],
        )
        state = PipelineState(output_dir)
        result = PDFAssemblyStage().run(input_dir, output_dir, cfg, state)

        assert result.processed == 4


# ---------------------------------------------------------------------------
# TestPDFAssemblyDPI
# ---------------------------------------------------------------------------

class TestPDFAssemblyDPI:
    """DPI setting affects the PDF."""

    def test_dpi_in_metadata(self, tmp_path):
        input_dir = tmp_path / "input"
        _save_test_images(input_dir, count=1)
        output_dir = tmp_path / "output"
        output_dir.mkdir()

        cfg = _cfg(tmp_path, pdf_compression="jpeg", pdf_dpi=150)
        state = PipelineState(output_dir)
        PDFAssemblyStage().run(input_dir, output_dir, cfg, state)

        sidecar = output_dir / "book.pdf.json"
        assert sidecar.exists()
        meta = json.loads(sidecar.read_text())
        assert meta["dpi"] == 150


# ---------------------------------------------------------------------------
# TestPDFAssemblyMetadata
# ---------------------------------------------------------------------------

class TestPDFAssemblyMetadata:
    """Sidecar metadata is correct."""

    def test_sidecar_contents(self, tmp_path):
        input_dir = tmp_path / "input"
        _save_test_images(input_dir, count=4)
        output_dir = tmp_path / "output"
        output_dir.mkdir()

        cfg = _cfg(tmp_path, pdf_compression="jpeg", pdf_jpeg_quality=92, pdf_dpi=300)
        state = PipelineState(output_dir)
        PDFAssemblyStage().run(input_dir, output_dir, cfg, state)

        sidecar = output_dir / "book.pdf.json"
        meta = json.loads(sidecar.read_text())

        assert meta["stage"] == "pdf_assembly"
        assert meta["page_count"] == 4
        assert meta["compression"] == "jpeg"
        assert meta["jpeg_quality"] == 92
        assert meta["dpi"] == 300
        assert meta["file_size_bytes"] > 0
        assert "input_dir" in meta

    def test_png_sidecar_no_quality(self, tmp_path):
        input_dir = tmp_path / "input"
        _save_test_images(input_dir, count=1)
        output_dir = tmp_path / "output"
        output_dir.mkdir()

        cfg = _cfg(tmp_path, pdf_compression="png")
        state = PipelineState(output_dir)
        PDFAssemblyStage().run(input_dir, output_dir, cfg, state)

        meta = json.loads((output_dir / "book.pdf.json").read_text())
        assert meta["compression"] == "png"
        assert meta["jpeg_quality"] is None


# ---------------------------------------------------------------------------
# TestPDFAssemblyEmptyInput
# ---------------------------------------------------------------------------

class TestPDFAssemblyEmptyInput:
    """Graceful handling of empty input."""

    def test_raises_on_no_images(self, tmp_path):
        input_dir = tmp_path / "input"
        input_dir.mkdir()
        output_dir = tmp_path / "output"
        output_dir.mkdir()

        cfg = _cfg(tmp_path)
        state = PipelineState(output_dir)
        stage = PDFAssemblyStage()

        with pytest.raises(RuntimeError, match="No images found"):
            stage.run(input_dir, output_dir, cfg, state)


# ---------------------------------------------------------------------------
# TestPDFAssemblyPageOrder
# ---------------------------------------------------------------------------

class TestPDFAssemblyPageOrder:
    """Pages are assembled in filename sort order."""

    def test_filename_sort_order(self, tmp_path):
        input_dir = tmp_path / "input"
        input_dir.mkdir()

        for name in ["C.png", "A.png", "B.png"]:
            img = _make_color_image()
            cv2.imwrite(str(input_dir / name), img)

        output_dir = tmp_path / "output"
        output_dir.mkdir()

        cfg = _cfg(tmp_path, pdf_compression="png")
        state = PipelineState(output_dir)
        PDFAssemblyStage().run(input_dir, output_dir, cfg, state)

        meta = json.loads((output_dir / "book.pdf.json").read_text())
        assert meta["page_count"] == 3


# ---------------------------------------------------------------------------
# TestPDFAssemblyInputFromAnyStage
# ---------------------------------------------------------------------------

class TestPDFAssemblyInputFromAnyStage:
    """Works with checkpoint directories from any stage number."""

    def test_works_with_stage_06_dir(self, tmp_path):
        checkpoint_dir = tmp_path / "06_content_area"
        _save_test_images(checkpoint_dir, count=2)
        output_dir = tmp_path / "output"
        output_dir.mkdir()

        cfg = _cfg(tmp_path, pdf_compression="jpeg")
        state = PipelineState(output_dir)
        result = PDFAssemblyStage().run(checkpoint_dir, output_dir, cfg, state)

        assert result.processed == 2
        assert (output_dir / "book.pdf").exists()

    def test_works_with_stage_05_dir(self, tmp_path):
        checkpoint_dir = tmp_path / "05_perspective"
        _save_test_images(checkpoint_dir, count=4)
        output_dir = tmp_path / "output"
        output_dir.mkdir()

        cfg = _cfg(tmp_path, pdf_compression="png")
        state = PipelineState(output_dir)
        result = PDFAssemblyStage().run(checkpoint_dir, output_dir, cfg, state)

        assert result.processed == 4


# ---------------------------------------------------------------------------
# TestCollectImages
# ---------------------------------------------------------------------------

class TestCollectImages:
    """Unit tests for _collect_images helper."""

    def test_collects_png_files(self, tmp_path):
        _save_test_images(tmp_path, count=3)
        files = _collect_images(tmp_path, _cfg(tmp_path))
        assert len(files) == 3

    def test_sorted_by_name(self, tmp_path):
        for name in ["Z.png", "A.png", "M.png"]:
            cv2.imwrite(str(tmp_path / name), _make_color_image())
        files = _collect_images(tmp_path, _cfg(tmp_path))
        names = [f.name for f in files]
        assert names == ["A.png", "M.png", "Z.png"]

    def test_ignores_non_image_files(self, tmp_path):
        _save_test_images(tmp_path, count=2)
        (tmp_path / "metadata.json").write_text("{}")
        (tmp_path / "notes.txt").write_text("hello")
        files = _collect_images(tmp_path, _cfg(tmp_path))
        assert len(files) == 2

    def test_excludes_by_name(self, tmp_path):
        _save_test_images(tmp_path, count=4)
        cfg = _cfg(tmp_path, exclude_images=["IMG_0001.png"])
        files = _collect_images(tmp_path, cfg)
        assert len(files) == 3
        assert all(f.name != "IMG_0001.png" for f in files)

    def test_excludes_by_stem(self, tmp_path):
        _save_test_images(tmp_path, count=4)
        cfg = _cfg(tmp_path, exclude_images=["IMG_0002"])
        files = _collect_images(tmp_path, cfg)
        assert len(files) == 3
        assert all(f.stem != "IMG_0002" for f in files)


# ---------------------------------------------------------------------------
# TestPDFAssemblyConfigFromTOML
# ---------------------------------------------------------------------------

class TestPDFAssemblyConfigFromTOML:
    """Config loading from TOML for pdf section."""

    def test_pdf_section_loaded(self, tmp_path):
        toml_file = tmp_path / "book.toml"
        toml_file.write_text(
            '[pdf]\n'
            'compression = "PNG"\n'
            'jpeg_quality = 75\n'
            'dpi = 150\n'
        )
        cfg = Config.from_toml(input_dir=tmp_path, toml_path=toml_file)
        assert cfg.pdf_compression == "png"
        assert cfg.pdf_jpeg_quality == 75
        assert cfg.pdf_dpi == 150

    def test_pdf_compression_case_insensitive(self, tmp_path):
        toml_file = tmp_path / "book.toml"
        toml_file.write_text('[pdf]\ncompression = "JpEg"\n')
        cfg = Config.from_toml(input_dir=tmp_path, toml_path=toml_file)
        assert cfg.pdf_compression == "jpeg"
