"""TDD tests for stitch utilities and StitchStage.

Tests cover: ORB-based image grouping, non-content detection, retake
deduplication, stitching fallback chain, manual overrides, and the
StitchStage BaseStage integration.
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from ghh.config import Config


# ---------------------------------------------------------------------------
# Synthetic image generators for stitch testing
# ---------------------------------------------------------------------------

def _make_textured_page(h: int = 400, w: int = 600, seed: int = 0) -> np.ndarray:
    """Create a page with enough texture for ORB feature matching.

    Uses random colored rectangles on a beige background to create
    distinctive, matchable features.
    """
    rng = np.random.RandomState(seed)
    img = np.full((h, w, 3), (200, 220, 230), dtype=np.uint8)

    for _ in range(40):
        x1, y1 = rng.randint(0, w - 30), rng.randint(0, h - 30)
        x2, y2 = x1 + rng.randint(10, 60), y1 + rng.randint(10, 60)
        color = tuple(int(c) for c in rng.randint(0, 200, 3))
        cv2.rectangle(img, (x1, y1), (min(x2, w), min(y2, h)), color, -1)

    # Add some text-like features for more keypoints
    for i in range(8):
        y = 30 + i * 45
        cv2.putText(img, f"Line {seed}_{i}", (20, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (30, 30, 30), 2)

    return img


def _make_overlapping_pair(
    overlap_frac: float = 0.4,
    h: int = 400,
    w: int = 600,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Create two images that share an overlapping region.

    Generates a wide canvas, then crops two windows with the specified
    overlap fraction. Both images share the same textured content in
    the overlap zone, making them matchable by ORB.
    """
    overlap_px = int(w * overlap_frac)
    total_w = 2 * w - overlap_px

    canvas = _make_textured_page(h, total_w, seed=seed)
    img_a = canvas[:, :w].copy()
    img_b = canvas[:, (w - overlap_px):].copy()

    return img_a, img_b


def _make_dark_cover(h: int = 400, w: int = 600) -> np.ndarray:
    """Simulate a book cover: mostly dark/brown uniform pixels."""
    img = np.full((h, w, 3), (30, 25, 20), dtype=np.uint8)
    # Slight variation to avoid being perfectly uniform
    noise = np.random.RandomState(99).randint(0, 15, img.shape, dtype=np.uint8)
    return cv2.add(img, noise)


def _save_test_image(path: Path, img: np.ndarray) -> None:
    cv2.imwrite(str(path), img)


# ---------------------------------------------------------------------------
# TestGroupDetection
# ---------------------------------------------------------------------------

class TestGroupDetection:
    """Test ORB-based image grouping logic."""

    def test_detects_overlapping_pair(self):
        from ghh.utils.stitch import detect_groups

        img_a, img_b = _make_overlapping_pair(overlap_frac=0.4)
        images = {"IMG_0001": img_a, "IMG_0002": img_b}
        cfg = Config(input_dir=Path("/tmp"))

        groups = detect_groups(images, cfg)

        # Should produce one group containing both images
        assert len(groups) == 1
        assert set(groups[0]) == {"IMG_0001", "IMG_0002"}

    def test_standalone_images_are_separate_groups(self):
        from ghh.utils.stitch import detect_groups

        img_a = _make_textured_page(seed=1)
        img_b = _make_textured_page(seed=2)
        images = {"IMG_0001": img_a, "IMG_0002": img_b}
        cfg = Config(input_dir=Path("/tmp"))

        groups = detect_groups(images, cfg)

        # Two distinct images → two separate groups
        assert len(groups) == 2

    def test_transitive_grouping(self):
        """If A matches B and B matches C, all three form one group."""
        from ghh.utils.stitch import detect_groups

        # Create a chain of 3 overlapping crops from one wide canvas.
        # Each crop is 600px wide with 50% overlap between consecutive crops.
        h, w = 400, 600
        step = w // 2  # 300px step → 300px overlap between consecutive
        total_w = w + 2 * step  # 1200px canvas
        canvas = _make_textured_page(h, total_w, seed=77)

        img_a = canvas[:, 0:w].copy()           # columns 0-599
        img_b = canvas[:, step:step + w].copy()  # columns 300-899
        img_c = canvas[:, 2 * step:2 * step + w].copy()  # columns 600-1199

        images = {"IMG_0001": img_a, "IMG_0002": img_b, "IMG_0003": img_c}
        cfg = Config(
            input_dir=Path("/tmp"),
            stitch_min_matches=10,
            stitch_inlier_ratio=0.4,
        )

        groups = detect_groups(images, cfg)

        assert len(groups) == 1
        assert set(groups[0]) == {"IMG_0001", "IMG_0002", "IMG_0003"}

    def test_no_stitch_override_prevents_grouping(self):
        from ghh.utils.stitch import detect_groups

        img_a, img_b = _make_overlapping_pair(overlap_frac=0.4)
        images = {"IMG_0001": img_a, "IMG_0002": img_b}
        cfg = Config(
            input_dir=Path("/tmp"),
            no_stitch_images=["IMG_0001", "IMG_0002"],
        )

        groups = detect_groups(images, cfg)

        # no_stitch should force these into separate groups
        assert len(groups) == 2

    def test_manual_stitch_groups_override(self):
        from ghh.utils.stitch import detect_groups

        img_a = _make_textured_page(seed=1)
        img_b = _make_textured_page(seed=2)
        img_c = _make_textured_page(seed=3)
        images = {"IMG_0001": img_a, "IMG_0002": img_b, "IMG_0003": img_c}
        cfg = Config(
            input_dir=Path("/tmp"),
            stitch_groups=[["IMG_0001", "IMG_0002"]],
        )

        groups = detect_groups(images, cfg)

        group_sets = [set(g) for g in groups]
        assert {"IMG_0001", "IMG_0002"} in group_sets
        assert {"IMG_0003"} in group_sets

    def test_returns_sorted_names_within_groups(self):
        from ghh.utils.stitch import detect_groups

        img_a, img_b = _make_overlapping_pair(overlap_frac=0.4)
        images = {"IMG_0002": img_b, "IMG_0001": img_a}
        cfg = Config(input_dir=Path("/tmp"))

        groups = detect_groups(images, cfg)
        for group in groups:
            assert group == sorted(group)


# ---------------------------------------------------------------------------
# TestNonContentDetection
# ---------------------------------------------------------------------------

class TestNonContentDetection:
    """Test detection of book covers and non-content images."""

    def test_detects_dark_cover(self):
        from ghh.utils.stitch import is_non_content

        cover = _make_dark_cover()
        assert is_non_content(cover) is True

    def test_content_page_is_not_flagged(self):
        from ghh.utils.stitch import is_non_content

        page = _make_textured_page(seed=1)
        assert is_non_content(page) is False

    def test_bright_page_is_not_flagged(self):
        from ghh.utils.stitch import is_non_content

        page = np.full((400, 600, 3), (220, 230, 240), dtype=np.uint8)
        assert is_non_content(page) is False


# ---------------------------------------------------------------------------
# TestRetakeDedup
# ---------------------------------------------------------------------------

class TestRetakeDedup:
    """Test detection and removal of near-duplicate retakes."""

    def test_detects_identical_images_as_retakes(self):
        from ghh.utils.stitch import deduplicate_retakes

        img = _make_textured_page(seed=1)
        group_images = {"IMG_0001": img.copy(), "IMG_0002": img.copy()}
        cfg = Config(input_dir=Path("/tmp"))

        kept, discarded = deduplicate_retakes(group_images, cfg)

        assert len(kept) == 1
        assert len(discarded) == 1

    def test_keeps_sharper_retake(self):
        from ghh.utils.stitch import deduplicate_retakes

        sharp = _make_textured_page(seed=1)
        blurry = cv2.GaussianBlur(sharp, (15, 15), 5)
        group_images = {"IMG_0001": sharp.copy(), "IMG_0002": blurry.copy()}
        cfg = Config(input_dir=Path("/tmp"))

        kept, discarded = deduplicate_retakes(group_images, cfg)

        assert "IMG_0001" in kept
        assert "IMG_0002" in discarded

    def test_non_duplicates_are_all_kept(self):
        from ghh.utils.stitch import deduplicate_retakes

        img_a, img_b = _make_overlapping_pair(overlap_frac=0.4)
        group_images = {"IMG_0001": img_a, "IMG_0002": img_b}
        cfg = Config(input_dir=Path("/tmp"))

        kept, discarded = deduplicate_retakes(group_images, cfg)

        assert len(kept) == 2
        assert len(discarded) == 0


# ---------------------------------------------------------------------------
# TestStitching
# ---------------------------------------------------------------------------

class TestStitching:
    """Test the stitching fallback chain."""

    def test_stitches_overlapping_pair(self):
        from ghh.utils.stitch import stitch_images

        img_a, img_b = _make_overlapping_pair(overlap_frac=0.4)
        images = {"IMG_0001": img_a, "IMG_0002": img_b}
        cfg = Config(input_dir=Path("/tmp"))

        result, method, success = stitch_images(images, cfg)

        assert success is True
        assert result is not None
        assert result.shape[0] > 0 and result.shape[1] > 0
        assert method in ("panorama", "scans", "homography")

    def test_result_is_wider_than_either_input(self):
        from ghh.utils.stitch import stitch_images

        img_a, img_b = _make_overlapping_pair(overlap_frac=0.4)
        images = {"IMG_0001": img_a, "IMG_0002": img_b}
        cfg = Config(input_dir=Path("/tmp"))

        result, _, success = stitch_images(images, cfg)

        if success:
            # Stitched image should be wider than either input
            assert result.shape[1] > max(img_a.shape[1], img_b.shape[1]) * 0.9

    def test_single_image_returns_itself(self):
        from ghh.utils.stitch import stitch_images

        img = _make_textured_page(seed=1)
        images = {"IMG_0001": img}
        cfg = Config(input_dir=Path("/tmp"))

        result, method, success = stitch_images(images, cfg)

        assert success is True
        assert method == "single"
        assert result.shape == img.shape

    def test_fallback_to_best_single_on_unmatchable(self):
        """When images have no matchable features, fall back to best single."""
        from ghh.utils.stitch import stitch_images

        # Solid-color images with zero ORB features → all stitchers fail
        img_a = np.full((200, 300, 3), (180, 200, 210), dtype=np.uint8)
        img_b = np.full((200, 300, 3), (100, 120, 130), dtype=np.uint8)
        images = {"IMG_0001": img_a, "IMG_0002": img_b}
        cfg = Config(input_dir=Path("/tmp"))

        result, method, success = stitch_images(images, cfg)

        assert result is not None
        assert method == "best_single"
        assert success is False


# ---------------------------------------------------------------------------
# TestFocusMetric
# ---------------------------------------------------------------------------

class TestFocusMetric:
    """Test focus quality measurement (Laplacian variance)."""

    def test_sharp_has_higher_focus_than_blurry(self):
        from ghh.utils.stitch import compute_focus

        sharp = _make_textured_page(seed=1)
        blurry = cv2.GaussianBlur(sharp, (21, 21), 7)

        assert compute_focus(sharp) > compute_focus(blurry)

    def test_returns_float(self):
        from ghh.utils.stitch import compute_focus

        img = _make_textured_page(seed=1)
        assert isinstance(compute_focus(img), float)


# ---------------------------------------------------------------------------
# TestStitchStage
# ---------------------------------------------------------------------------

class TestStitchStageContract:
    """Verify StitchStage satisfies the BaseStage contract."""

    def test_has_correct_attributes(self):
        from ghh.stages.stitch import StitchStage

        stage = StitchStage()
        assert stage.name == "stitch"
        assert stage.number == 1
        assert stage.checkpoint_name == "01_stitched"
        assert stage.error_class == "skippable"

    def test_is_base_stage_subclass(self):
        from ghh.pipeline import BaseStage
        from ghh.stages.stitch import StitchStage

        assert issubclass(StitchStage, BaseStage)


@pytest.mark.slow
class TestStitchStageRun:
    """Integration tests for StitchStage.run()."""

    def test_standalone_images_pass_through(self, tmp_path):
        """When all images are standalone (no overlaps), they pass through."""
        from ghh.pipeline import PipelineState
        from ghh.stages.stitch import StitchStage

        input_dir = tmp_path / "00_preprocessed"
        input_dir.mkdir()
        for i in range(3):
            img = _make_textured_page(seed=i + 10)
            _save_test_image(input_dir / f"IMG_{i:04d}.png", img)

        cfg = Config(input_dir=tmp_path / "raw", output_dir=tmp_path)
        state = PipelineState(tmp_path)
        stage = StitchStage()

        result = stage.run(input_dir, tmp_path, cfg, state)

        out_dir = tmp_path / "01_stitched"
        assert out_dir.exists()
        assert len(list(out_dir.glob("*.png"))) == 3
        assert result.processed == 3

    def test_excluded_images_not_in_output(self, tmp_path):
        """Images in exclude list should be omitted from output."""
        from ghh.pipeline import PipelineState
        from ghh.stages.stitch import StitchStage

        input_dir = tmp_path / "00_preprocessed"
        input_dir.mkdir()
        _save_test_image(input_dir / "IMG_0001.png", _make_textured_page(seed=1))
        _save_test_image(input_dir / "IMG_0002.png", _make_dark_cover())
        _save_test_image(input_dir / "IMG_0003.png", _make_textured_page(seed=3))

        cfg = Config(
            input_dir=tmp_path / "raw",
            output_dir=tmp_path,
            exclude_images=["IMG_0002"],
        )
        state = PipelineState(tmp_path)
        stage = StitchStage()

        result = stage.run(input_dir, tmp_path, cfg, state)

        out_dir = tmp_path / "01_stitched"
        out_files = sorted(p.stem for p in out_dir.glob("*.png"))
        assert "IMG_0002" not in out_files
        assert "IMG_0001" in out_files
        assert "IMG_0003" in out_files

    def test_writes_metadata_sidecars(self, tmp_path):
        from ghh.pipeline import PipelineState
        from ghh.stages.stitch import StitchStage

        input_dir = tmp_path / "00_preprocessed"
        input_dir.mkdir()
        _save_test_image(input_dir / "IMG_0001.png", _make_textured_page(seed=1))

        cfg = Config(input_dir=tmp_path / "raw", output_dir=tmp_path)
        state = PipelineState(tmp_path)
        stage = StitchStage()

        stage.run(input_dir, tmp_path, cfg, state)

        sidecar = tmp_path / "01_stitched" / "IMG_0001.json"
        assert sidecar.exists()
        meta = json.loads(sidecar.read_text())
        assert meta["stage"] == "stitch"
