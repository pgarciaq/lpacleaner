"""Tests for lpacleaner.compare (compare and publish viewers)."""

from __future__ import annotations

import json
import re
from pathlib import Path

import cv2
import numpy as np

from lpacleaner.compare import (
    _THEME_COMPARE,
    _THEME_PUBLISH,
    _convert_image,
    _render_html,
    discover_book,
    generate_compare_html,
    infer_input_dir,
    publish_book,
    write_compare_html,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_image(path: Path, w: int = 80, h: int = 60, color: int = 200) -> None:
    """Write a small solid-color PNG to *path*."""
    img = np.full((h, w, 3), color, dtype=np.uint8)
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), img)


def _make_sidecar(path: Path, data: dict | None = None) -> None:
    """Write a JSON sidecar next to an image."""
    if data is None:
        data = {"angle": 1.2, "method": "staff_lines"}
    path.write_text(json.dumps(data))


def _make_book(tmp_path: Path, stages: list[str], stems: list[str],
               *, input_dir: bool = False) -> tuple[Path, Path | None]:
    """Create a minimal pipeline output layout.

    Returns (output_dir, input_dir_or_None).
    """
    out = tmp_path / "book_output"
    out.mkdir()

    for stage_name in stages:
        for stem in stems:
            _make_image(out / stage_name / f"{stem}.png")

    inp = None
    if input_dir:
        inp = tmp_path / "book"
        inp.mkdir()
        for stem in stems:
            _make_image(inp / f"{stem}.png")

    return out, inp


# ---------------------------------------------------------------------------
# discover_book
# ---------------------------------------------------------------------------

class TestDiscoverBook:
    def test_discovers_stages_and_images(self, tmp_path: Path):
        out, _ = _make_book(
            tmp_path,
            stages=["00_preprocessed", "05_perspective"],
            stems=["IMG_0001", "IMG_0002"],
        )
        book = discover_book(out)
        assert book["stages"] == ["0: Preprocess", "5: Perspective"]
        assert len(book["images"]) == 2
        assert book["images"][0]["stem"] == "IMG_0001"
        assert book["images"][1]["stem"] == "IMG_0002"

    def test_includes_originals_when_input_dir_given(self, tmp_path: Path):
        out, inp = _make_book(
            tmp_path,
            stages=["00_preprocessed"],
            stems=["IMG_0001"],
            input_dir=True,
        )
        book = discover_book(out, inp)
        assert book["stages"][0] == "Original"
        assert book["stages"][1] == "0: Preprocess"
        assert book["images"][0]["stages"][0] is not None
        assert "file://" in book["images"][0]["stages"][0]["src"]

    def test_skips_input_dir_when_none(self, tmp_path: Path):
        out, _ = _make_book(
            tmp_path,
            stages=["07_deskewed"],
            stems=["IMG_0001"],
        )
        book = discover_book(out, None)
        assert book["stages"] == ["7: Deskew"]

    def test_unknown_stage_number_uses_dirname(self, tmp_path: Path):
        out, _ = _make_book(tmp_path, stages=["99_custom"], stems=["A"])
        book = discover_book(out)
        assert book["stages"] == ["99: custom"]

    def test_missing_image_in_one_stage(self, tmp_path: Path):
        out = tmp_path / "out"
        out.mkdir()
        (out / "00_preprocessed").mkdir()
        (out / "05_perspective").mkdir()
        _make_image(out / "00_preprocessed" / "A.png")
        _make_image(out / "00_preprocessed" / "B.png")
        _make_image(out / "05_perspective" / "A.png")

        book = discover_book(out)
        assert len(book["images"]) == 2
        img_b = next(i for i in book["images"] if i["stem"] == "B")
        assert img_b["stages"][1] is None

    def test_reads_sidecar_metadata(self, tmp_path: Path):
        out, _ = _make_book(
            tmp_path, stages=["00_preprocessed"], stems=["X"],
        )
        _make_sidecar(out / "00_preprocessed" / "X.json")
        book = discover_book(out)
        assert book["images"][0]["stages"][0]["meta"]["angle"] == 1.2

    def test_ignores_malformed_sidecar(self, tmp_path: Path):
        out, _ = _make_book(
            tmp_path, stages=["00_preprocessed"], stems=["X"],
        )
        (out / "00_preprocessed" / "X.json").write_text("{bad json")
        book = discover_book(out)
        assert book["images"][0]["stages"][0]["meta"] is None

    def test_images_sorted_alphabetically(self, tmp_path: Path):
        out, _ = _make_book(
            tmp_path,
            stages=["00_preprocessed"],
            stems=["IMG_0030", "IMG_0010", "IMG_0020"],
        )
        book = discover_book(out)
        assert [i["stem"] for i in book["images"]] == [
            "IMG_0010", "IMG_0020", "IMG_0030",
        ]

    def test_empty_output_dir(self, tmp_path: Path):
        out = tmp_path / "empty_output"
        out.mkdir()
        book = discover_book(out)
        assert book["stages"] == []
        assert book["images"] == []

    def test_ignores_non_checkpoint_dirs(self, tmp_path: Path):
        out, _ = _make_book(
            tmp_path, stages=["00_preprocessed"], stems=["A"],
        )
        (out / "notes").mkdir()
        (out / ".hidden").mkdir()
        book = discover_book(out)
        assert len(book["stages"]) == 1


# ---------------------------------------------------------------------------
# infer_input_dir
# ---------------------------------------------------------------------------

class TestInferInputDir:
    def test_infers_from_output_convention(self, tmp_path: Path):
        inp = tmp_path / "mybook"
        inp.mkdir()
        out = tmp_path / "mybook_output"
        out.mkdir()
        assert infer_input_dir(out) == inp

    def test_returns_none_when_no_match(self, tmp_path: Path):
        out = tmp_path / "something"
        out.mkdir()
        assert infer_input_dir(out) is None

    def test_returns_none_when_candidate_missing(self, tmp_path: Path):
        out = tmp_path / "missing_output"
        out.mkdir()
        assert infer_input_dir(out) is None


# ---------------------------------------------------------------------------
# generate_compare_html / _render_html
# ---------------------------------------------------------------------------

class TestGenerateCompareHtml:
    def test_produces_valid_html(self, tmp_path: Path):
        out, _ = _make_book(
            tmp_path, stages=["00_preprocessed"], stems=["A"],
        )
        html = generate_compare_html(out)
        assert "<!DOCTYPE html>" in html
        assert "Compare mode" in html
        assert "__BOOK_JSON__" not in html
        assert "__TITLE__" not in html
        assert "__T_BG__" not in html

    def test_uses_compare_theme_colors(self, tmp_path: Path):
        out, _ = _make_book(
            tmp_path, stages=["00_preprocessed"], stems=["A"],
        )
        html = generate_compare_html(out)
        assert _THEME_COMPARE["bg"] in html
        assert _THEME_PUBLISH["bg"] not in html

    def test_initial_stem_sets_index(self, tmp_path: Path):
        out, _ = _make_book(
            tmp_path, stages=["00_preprocessed"],
            stems=["A", "B", "C"],
        )
        html = generate_compare_html(out, initial_stem="B")
        assert "let imgIdx = 1;" in html

    def test_strips_output_suffix_from_title(self, tmp_path: Path):
        out, _ = _make_book(
            tmp_path, stages=["00_preprocessed"], stems=["A"],
        )
        out_renamed = tmp_path / "LPA1_output"
        out.rename(out_renamed)
        html = generate_compare_html(out_renamed)
        assert "<title>" in html
        assert "LPA1" in html

    def test_embeds_book_json(self, tmp_path: Path):
        out, _ = _make_book(
            tmp_path, stages=["00_preprocessed"], stems=["X"],
        )
        html = generate_compare_html(out)
        match = re.search(r"const BOOK = ({.*?});", html)
        assert match is not None
        book = json.loads(match.group(1))
        assert book["images"][0]["stem"] == "X"

    def test_no_unreplaced_placeholders(self, tmp_path: Path):
        out, _ = _make_book(
            tmp_path, stages=["00_preprocessed"], stems=["A"],
        )
        html = generate_compare_html(out)
        assert "__" not in html or html.count("__") == 0


class TestRenderHtml:
    def test_compare_mode_badge(self):
        html = _render_html('{"stages":[],"images":[]}', "test", 0, "compare")
        assert "Compare mode" in html

    def test_publish_mode_badge(self):
        html = _render_html('{"stages":[],"images":[]}', "test", 0, "publish")
        assert "Published " in html
        assert " UTC" in html

    def test_publish_uses_amber_theme(self):
        html = _render_html('{"stages":[],"images":[]}', "test", 0, "publish")
        assert _THEME_PUBLISH["bg"] in html
        assert _THEME_COMPARE["bg"] not in html

    def test_all_theme_keys_replaced(self):
        html = _render_html('{"stages":[],"images":[]}', "t", 0, "compare")
        assert "__T_" not in html


# ---------------------------------------------------------------------------
# write_compare_html
# ---------------------------------------------------------------------------

class TestWriteCompareHtml:
    def test_writes_file(self, tmp_path: Path):
        out, _ = _make_book(
            tmp_path, stages=["00_preprocessed"], stems=["A"],
        )
        result = write_compare_html(out)
        assert result == out / "compare.html"
        assert result.exists()
        content = result.read_text()
        assert "Compare mode" in content

    def test_auto_infers_input_dir(self, tmp_path: Path):
        inp = tmp_path / "book"
        inp.mkdir()
        _make_image(inp / "A.png")

        out = tmp_path / "book_output"
        out.mkdir()
        _make_image(out / "00_preprocessed" / "A.png")

        result = write_compare_html(out)
        content = result.read_text()
        assert "Original" in content


# ---------------------------------------------------------------------------
# _convert_image
# ---------------------------------------------------------------------------

class TestConvertImage:
    def test_converts_png_to_jpeg(self, tmp_path: Path):
        src = tmp_path / "src"
        src.mkdir()
        _make_image(src / "IMG.png", w=100, h=80)
        images_dir = tmp_path / "images"
        images_dir.mkdir()

        entry = _convert_image(src, "IMG", images_dir, "00_stage", 1500, 85)
        assert entry is not None
        assert entry["src"] == "images/00_stage/IMG.jpg"
        assert (images_dir / "00_stage" / "IMG.jpg").exists()

    def test_downscales_large_images(self, tmp_path: Path):
        src = tmp_path / "src"
        src.mkdir()
        _make_image(src / "BIG.png", w=3000, h=2000)
        images_dir = tmp_path / "images"
        images_dir.mkdir()

        _convert_image(src, "BIG", images_dir, "stg", 1000, 85)
        result = cv2.imread(str(images_dir / "stg" / "BIG.jpg"))
        h, w = result.shape[:2]
        assert max(h, w) <= 1000

    def test_preserves_small_images(self, tmp_path: Path):
        src = tmp_path / "src"
        src.mkdir()
        _make_image(src / "SM.png", w=200, h=150)
        images_dir = tmp_path / "images"
        images_dir.mkdir()

        _convert_image(src, "SM", images_dir, "stg", 1500, 85)
        result = cv2.imread(str(images_dir / "stg" / "SM.jpg"))
        h, w = result.shape[:2]
        assert w == 200 and h == 150

    def test_returns_none_for_missing_image(self, tmp_path: Path):
        src = tmp_path / "empty"
        src.mkdir()
        images_dir = tmp_path / "images"
        images_dir.mkdir()
        assert _convert_image(src, "NOPE", images_dir, "stg", 1500, 85) is None

    def test_skips_existing_jpeg(self, tmp_path: Path):
        src = tmp_path / "src"
        src.mkdir()
        _make_image(src / "A.png", w=100, h=80, color=50)

        images_dir = tmp_path / "images"
        out_dir = images_dir / "stg"
        out_dir.mkdir(parents=True)
        existing = out_dir / "A.jpg"
        existing.write_bytes(b"sentinel")

        entry = _convert_image(src, "A", images_dir, "stg", 1500, 85)
        assert entry is not None
        assert existing.read_bytes() == b"sentinel"

    def test_reads_sidecar(self, tmp_path: Path):
        src = tmp_path / "src"
        src.mkdir()
        _make_image(src / "M.png")
        _make_sidecar(src / "M.json", {"key": "value"})
        images_dir = tmp_path / "images"
        images_dir.mkdir()

        entry = _convert_image(src, "M", images_dir, "stg", 1500, 85)
        assert entry["meta"] == {"key": "value"}


# ---------------------------------------------------------------------------
# publish_book
# ---------------------------------------------------------------------------

class TestPublishBook:
    def test_creates_index_html_and_images(self, tmp_path: Path):
        out, inp = _make_book(
            tmp_path,
            stages=["00_preprocessed", "05_perspective"],
            stems=["A", "B"],
            input_dir=True,
        )
        pub = tmp_path / "pub"
        result = publish_book(out, pub, input_dir=inp)
        assert result == pub / "index.html"
        assert result.exists()

        assert (pub / "images" / "original" / "A.jpg").exists()
        assert (pub / "images" / "00_preprocessed" / "B.jpg").exists()
        assert (pub / "images" / "05_perspective" / "A.jpg").exists()

    def test_uses_publish_theme(self, tmp_path: Path):
        out, _ = _make_book(
            tmp_path, stages=["00_preprocessed"], stems=["A"],
        )
        pub = tmp_path / "pub"
        result = publish_book(out, pub)
        html = result.read_text()
        assert _THEME_PUBLISH["bg"] in html
        assert "Published " in html

    def test_uses_relative_paths(self, tmp_path: Path):
        out, _ = _make_book(
            tmp_path, stages=["00_preprocessed"], stems=["A"],
        )
        pub = tmp_path / "pub"
        publish_book(out, pub)
        html = (pub / "index.html").read_text()
        assert "file://" not in html
        match = re.search(r"const BOOK = ({.*?});", html)
        book = json.loads(match.group(1))
        src = book["images"][0]["stages"][0]["src"]
        assert src.startswith("images/")

    def test_stage_filter(self, tmp_path: Path):
        out, _ = _make_book(
            tmp_path,
            stages=["00_preprocessed", "05_perspective", "07_deskewed"],
            stems=["A"],
        )
        pub = tmp_path / "pub"
        publish_book(out, pub, stage_filter={"00", "07", "orig"})
        html = (pub / "index.html").read_text()
        match = re.search(r"const BOOK = ({.*?});", html)
        book = json.loads(match.group(1))
        assert "0: Preprocess" in book["stages"]
        assert "7: Deskew" in book["stages"]
        assert "5: Perspective" not in book["stages"]

    def test_auto_infers_input_dir(self, tmp_path: Path):
        inp = tmp_path / "book"
        inp.mkdir()
        _make_image(inp / "A.png")
        out = tmp_path / "book_output"
        out.mkdir()
        _make_image(out / "00_preprocessed" / "A.png")

        pub = tmp_path / "pub"
        publish_book(out, pub)
        html = (pub / "index.html").read_text()
        assert "Original" in html


# ---------------------------------------------------------------------------
# CLI commands
# ---------------------------------------------------------------------------

class TestCompareCLI:
    def test_compare_generates_html(self, tmp_path: Path):
        from click.testing import CliRunner

        from lpacleaner.cli import compare

        out, _ = _make_book(
            tmp_path, stages=["00_preprocessed"], stems=["A"],
        )
        runner = CliRunner()
        result = runner.invoke(compare, [str(out), "--no-open"])
        assert result.exit_code == 0
        assert "compare.html" in result.output
        assert (out / "compare.html").exists()

    def test_compare_with_image_stem(self, tmp_path: Path):
        from click.testing import CliRunner

        from lpacleaner.cli import compare

        out, _ = _make_book(
            tmp_path, stages=["00_preprocessed"], stems=["A", "B", "C"],
        )
        runner = CliRunner()
        result = runner.invoke(compare, [str(out), "B", "--no-open"])
        assert result.exit_code == 0
        html = (out / "compare.html").read_text()
        assert "let imgIdx = 1;" in html

    def test_compare_empty_dir_exits_1(self, tmp_path: Path):
        from click.testing import CliRunner

        from lpacleaner.cli import compare

        out = tmp_path / "empty"
        out.mkdir()
        runner = CliRunner()
        result = runner.invoke(compare, [str(out), "--no-open"])
        assert result.exit_code != 0
        assert "No images found" in result.output


class TestPublishCLI:
    def test_publish_creates_site(self, tmp_path: Path):
        from click.testing import CliRunner

        from lpacleaner.cli import publish

        out, _ = _make_book(
            tmp_path, stages=["00_preprocessed"], stems=["A"],
        )
        pub = tmp_path / "pub"
        runner = CliRunner()
        result = runner.invoke(publish, [
            str(out), str(pub), "--no-open",
        ])
        assert result.exit_code == 0
        assert "Published to" in result.output
        assert (pub / "index.html").exists()
        assert (pub / "images" / "00_preprocessed" / "A.jpg").exists()

    def test_publish_with_stage_filter(self, tmp_path: Path):
        from click.testing import CliRunner

        from lpacleaner.cli import publish

        out, _ = _make_book(
            tmp_path,
            stages=["00_preprocessed", "05_perspective"],
            stems=["A"],
        )
        pub = tmp_path / "pub"
        runner = CliRunner()
        result = runner.invoke(publish, [
            str(out), str(pub), "--stages", "0", "--no-open",
        ])
        assert result.exit_code == 0
        assert (pub / "images" / "00_preprocessed" / "A.jpg").exists()
        assert not (pub / "images" / "05_perspective").exists()

    def test_publish_with_max_dim(self, tmp_path: Path):
        from click.testing import CliRunner

        from lpacleaner.cli import publish

        out, _ = _make_book(
            tmp_path, stages=["00_preprocessed"], stems=["A"],
        )
        _make_image(out / "00_preprocessed" / "A.png", w=2000, h=1500)

        pub = tmp_path / "pub"
        runner = CliRunner()
        result = runner.invoke(publish, [
            str(out), str(pub), "--max-dim", "500", "--no-open",
        ])
        assert result.exit_code == 0
        img = cv2.imread(str(pub / "images" / "00_preprocessed" / "A.jpg"))
        assert max(img.shape[:2]) <= 500
