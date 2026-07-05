"""Tests for ghh.flipbook (flipbook generation) and CLI integration."""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pytest
from click.testing import CliRunner

from ghh.cli import main
from ghh.flipbook import (
    _downscale_image,
    _find_pdf,
    _find_source_images,
    _vendor_js_path,
    generate_flipbook,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_image(path: Path, w: int = 200, h: int = 300, color: int = 180) -> None:
    """Write a small solid-color PNG to *path*."""
    img = np.full((h, w, 3), color, dtype=np.uint8)
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), img)


def _make_pipeline_output(
    tmp_path: Path,
    *,
    stages: list[str] | None = None,
    stems: list[str] | None = None,
    with_pdf: bool = False,
) -> Path:
    """Create a minimal pipeline output layout and return output_dir."""
    if stages is None:
        stages = ["05_perspective"]
    if stems is None:
        stems = ["IMG_0001", "IMG_0002", "IMG_0003"]

    out = tmp_path / "book_output"
    out.mkdir()

    for stage_name in stages:
        for stem in stems:
            _make_image(out / stage_name / f"{stem}.png")

    if with_pdf:
        (out / "LPA-1.pdf").write_bytes(b"%PDF-1.4 fake pdf content")

    return out


# ---------------------------------------------------------------------------
# _vendor_js_path
# ---------------------------------------------------------------------------


class TestVendorJsPath:
    def test_file_exists(self):
        path = _vendor_js_path()
        assert path.exists()
        assert path.name == "page-flip.browser.js"
        assert path.stat().st_size > 1000


# ---------------------------------------------------------------------------
# _find_source_images
# ---------------------------------------------------------------------------


class TestFindSourceImages:
    def test_finds_images_from_highest_checkpoint(self, tmp_path: Path):
        out = _make_pipeline_output(
            tmp_path, stages=["00_preprocessed", "05_perspective"]
        )
        images = _find_source_images(out)
        assert len(images) == 3
        assert all("05_perspective" in str(p) for p in images)

    def test_falls_back_to_earlier_checkpoint(self, tmp_path: Path):
        out = _make_pipeline_output(tmp_path, stages=["00_preprocessed"])
        # Add an empty higher checkpoint
        (out / "07_deskewed").mkdir()
        images = _find_source_images(out)
        assert len(images) == 3
        assert all("00_preprocessed" in str(p) for p in images)

    def test_returns_empty_when_no_checkpoints(self, tmp_path: Path):
        out = tmp_path / "empty_output"
        out.mkdir()
        assert _find_source_images(out) == []

    def test_sorted_by_filename(self, tmp_path: Path):
        out = _make_pipeline_output(
            tmp_path, stems=["IMG_0003", "IMG_0001", "IMG_0002"]
        )
        images = _find_source_images(out)
        names = [p.stem for p in images]
        assert names == ["IMG_0001", "IMG_0002", "IMG_0003"]


# ---------------------------------------------------------------------------
# _find_pdf
# ---------------------------------------------------------------------------


class TestFindPdf:
    def test_finds_pdf(self, tmp_path: Path):
        out = _make_pipeline_output(tmp_path, with_pdf=True)
        pdf = _find_pdf(out)
        assert pdf is not None
        assert pdf.name == "LPA-1.pdf"

    def test_returns_none_when_no_pdf(self, tmp_path: Path):
        out = _make_pipeline_output(tmp_path, with_pdf=False)
        assert _find_pdf(out) is None


# ---------------------------------------------------------------------------
# _downscale_image
# ---------------------------------------------------------------------------


class TestDownscaleImage:
    def test_no_downscale_when_smaller(self, tmp_path: Path):
        src = tmp_path / "input.png"
        dst = tmp_path / "output.jpg"
        _make_image(src, w=100, h=150)
        w, h = _downscale_image(src, dst, max_width=200, jpeg_quality=85)
        assert w == 100
        assert h == 150
        assert dst.exists()

    def test_downscales_when_wider(self, tmp_path: Path):
        src = tmp_path / "input.png"
        dst = tmp_path / "output.jpg"
        _make_image(src, w=3200, h=2400)
        w, h = _downscale_image(src, dst, max_width=1600, jpeg_quality=85)
        assert w == 1600
        assert h == 1200

    def test_output_is_jpeg(self, tmp_path: Path):
        src = tmp_path / "input.png"
        dst = tmp_path / "output.jpg"
        _make_image(src, w=100, h=100)
        _downscale_image(src, dst, max_width=200, jpeg_quality=85)
        img = cv2.imread(str(dst))
        assert img is not None

    def test_raises_on_invalid_input(self, tmp_path: Path):
        src = tmp_path / "nonexistent.png"
        dst = tmp_path / "output.jpg"
        with pytest.raises(ValueError, match="Could not read"):
            _downscale_image(src, dst, max_width=200, jpeg_quality=85)


# ---------------------------------------------------------------------------
# generate_flipbook
# ---------------------------------------------------------------------------


class TestGenerateFlipbook:
    def test_basic_generation(self, tmp_path: Path):
        out = _make_pipeline_output(tmp_path)
        index = generate_flipbook(out)
        fb_dir = out / "flipbook"

        assert index.exists()
        assert index.name == "index.html"
        assert (fb_dir / "page-flip.browser.js").exists()
        assert (fb_dir / "pages").is_dir()
        assert (fb_dir / "flipbook.json").exists()

        pages = list((fb_dir / "pages").glob("*.jpg"))
        assert len(pages) == 3

    def test_pages_downscaled(self, tmp_path: Path):
        out = _make_pipeline_output(tmp_path)
        # Override images with larger ones
        stage_dir = out / "05_perspective"
        for f in stage_dir.glob("*.png"):
            _make_image(f, w=3200, h=2400)

        generate_flipbook(out, max_width=800)

        fb_dir = out / "flipbook"
        for jpg in (fb_dir / "pages").glob("*.jpg"):
            img = cv2.imread(str(jpg))
            assert img.shape[1] <= 800

    def test_custom_flipbook_dir(self, tmp_path: Path):
        out = _make_pipeline_output(tmp_path)
        custom_dir = tmp_path / "my_flipbook"
        index = generate_flipbook(out, custom_dir)
        assert index.parent == custom_dir
        assert (custom_dir / "pages").is_dir()

    def test_no_pdf_flag(self, tmp_path: Path):
        out = _make_pipeline_output(tmp_path, with_pdf=True)
        index = generate_flipbook(out, include_pdf=False)
        html = index.read_text()
        assert "LPA-1.pdf" not in html
        assert not (index.parent / "LPA-1.pdf").exists()

    def test_pdf_copied(self, tmp_path: Path):
        out = _make_pipeline_output(tmp_path, with_pdf=True)
        index = generate_flipbook(out)
        fb_dir = index.parent
        assert (fb_dir / "LPA-1.pdf").exists()
        html = index.read_text()
        assert "LPA-1.pdf" in html
        assert "Download PDF" in html

    def test_pdf_missing_warns(self, tmp_path: Path, caplog):
        out = _make_pipeline_output(tmp_path, with_pdf=False)
        index = generate_flipbook(out, include_pdf=True)
        html = index.read_text()
        assert "Download PDF" not in html
        assert not (index.parent / "LPA-1.pdf").exists()

    def test_metadata_sidecar(self, tmp_path: Path):
        out = _make_pipeline_output(tmp_path)
        generate_flipbook(out, title="Test Book")
        fb_dir = out / "flipbook"
        meta = json.loads((fb_dir / "flipbook.json").read_text())
        assert meta["page_count"] == 3
        assert meta["title"] == "Test Book"
        assert meta["has_pdf"] is False
        assert "generated_at" in meta
        assert meta["total_size_bytes"] > 0

    def test_raises_when_no_images(self, tmp_path: Path):
        out = tmp_path / "empty_output"
        out.mkdir()
        with pytest.raises(FileNotFoundError, match="No checkpoint images"):
            generate_flipbook(out)

    def test_title_in_html(self, tmp_path: Path):
        out = _make_pipeline_output(tmp_path)
        index = generate_flipbook(out, title="My Ancient Book")
        html = index.read_text()
        assert "My Ancient Book" in html

    def test_html_contains_stpageflip(self, tmp_path: Path):
        out = _make_pipeline_output(tmp_path)
        index = generate_flipbook(out)
        html = index.read_text()
        assert "St.PageFlip" in html
        assert "page-flip.browser.js" in html

    def test_page_numbering(self, tmp_path: Path):
        out = _make_pipeline_output(tmp_path, stems=["A", "B", "C", "D", "E"])
        index = generate_flipbook(out)
        html = index.read_text()
        assert "Page 1 of 5" in html

    def test_show_cover_false_by_default(self, tmp_path: Path):
        out = _make_pipeline_output(tmp_path)
        index = generate_flipbook(out)
        html = index.read_text()
        assert "showCover: false" in html

    def test_show_cover_true(self, tmp_path: Path):
        out = _make_pipeline_output(tmp_path)
        index = generate_flipbook(out, show_cover=True)
        html = index.read_text()
        assert "showCover: true" in html


# ---------------------------------------------------------------------------
# CLI: ghh flipbook
# ---------------------------------------------------------------------------


class TestFlipbookCLI:
    def test_basic(self, tmp_path: Path):
        out = _make_pipeline_output(tmp_path)
        runner = CliRunner()
        result = runner.invoke(main, ["flipbook", str(out), "--no-open"])
        assert result.exit_code == 0
        assert "Flipbook generated" in result.output
        assert (out / "flipbook" / "index.html").exists()

    def test_custom_output_dir(self, tmp_path: Path):
        out = _make_pipeline_output(tmp_path)
        fb_dir = tmp_path / "custom_fb"
        runner = CliRunner()
        result = runner.invoke(
            main, ["flipbook", str(out), str(fb_dir), "--no-open"]
        )
        assert result.exit_code == 0
        assert (fb_dir / "index.html").exists()

    def test_no_pdf(self, tmp_path: Path):
        out = _make_pipeline_output(tmp_path, with_pdf=True)
        runner = CliRunner()
        result = runner.invoke(main, ["flipbook", str(out), "--no-pdf", "--no-open"])
        assert result.exit_code == 0
        assert not (out / "flipbook" / "LPA-1.pdf").exists()

    def test_with_title(self, tmp_path: Path):
        out = _make_pipeline_output(tmp_path)
        runner = CliRunner()
        result = runner.invoke(
            main, ["flipbook", str(out), "--title", "My Title", "--no-open"]
        )
        assert result.exit_code == 0
        html = (out / "flipbook" / "index.html").read_text()
        assert "My Title" in html

    def test_cover_flag(self, tmp_path: Path):
        out = _make_pipeline_output(tmp_path)
        runner = CliRunner()
        result = runner.invoke(
            main, ["flipbook", str(out), "--cover", "--no-open"]
        )
        assert result.exit_code == 0
        html = (out / "flipbook" / "index.html").read_text()
        assert "showCover: true" in html

    def test_error_no_images(self, tmp_path: Path):
        out = tmp_path / "empty"
        out.mkdir()
        runner = CliRunner()
        result = runner.invoke(main, ["flipbook", str(out), "--no-open"])
        assert result.exit_code != 0

    def test_max_width_option(self, tmp_path: Path):
        out = _make_pipeline_output(tmp_path)
        stage_dir = out / "05_perspective"
        for f in stage_dir.glob("*.png"):
            _make_image(f, w=3200, h=2400)

        runner = CliRunner()
        result = runner.invoke(
            main, ["flipbook", str(out), "--max-width", "600", "--no-open"]
        )
        assert result.exit_code == 0
        for jpg in (out / "flipbook" / "pages").glob("*.jpg"):
            img = cv2.imread(str(jpg))
            assert img.shape[1] <= 600


# ---------------------------------------------------------------------------
# CLI: ghh publish --with-flipbook / --with-pdf
# ---------------------------------------------------------------------------


class TestPublishWithFlipbook:
    def test_with_flipbook(self, tmp_path: Path):
        out = _make_pipeline_output(tmp_path)
        pub_dir = tmp_path / "published"
        runner = CliRunner()
        result = runner.invoke(
            main,
            ["publish", str(out), str(pub_dir), "--with-flipbook", "--no-open"],
        )
        assert result.exit_code == 0
        assert (pub_dir / "flipbook" / "index.html").exists()
        assert (pub_dir / "flipbook" / "pages").is_dir()

    def test_with_pdf_implies_flipbook(self, tmp_path: Path):
        out = _make_pipeline_output(tmp_path, with_pdf=True)
        pub_dir = tmp_path / "published"
        runner = CliRunner()
        result = runner.invoke(
            main,
            ["publish", str(out), str(pub_dir), "--with-pdf", "--no-open"],
        )
        assert result.exit_code == 0
        fb_dir = pub_dir / "flipbook"
        assert fb_dir.exists()
        assert (fb_dir / "LPA-1.pdf").exists()
        html = (fb_dir / "index.html").read_text()
        assert "Download PDF" in html

    def test_without_flags_no_flipbook(self, tmp_path: Path):
        out = _make_pipeline_output(tmp_path)
        pub_dir = tmp_path / "published"
        runner = CliRunner()
        result = runner.invoke(
            main, ["publish", str(out), str(pub_dir), "--no-open"]
        )
        assert result.exit_code == 0
        assert not (pub_dir / "flipbook").exists()
