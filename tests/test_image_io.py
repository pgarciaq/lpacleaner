"""TDD tests for ghh.utils.image_io -- written RED (before implementation)."""

from __future__ import annotations

import json
import os
import signal
import threading
from pathlib import Path

import cv2
import numpy as np
import pytest
from PIL import Image

from ghh.utils.image_io import (
    ensure_checkpoint_dir,
    load_image,
    save_checkpoint,
)


class TestLoadImage:
    """Tests for load_image(): EXIF-aware loading, metadata extraction."""

    def test_loads_jpeg(self, sample_jpeg):
        img, meta = load_image(sample_jpeg)
        assert isinstance(img, np.ndarray)
        assert img.ndim == 3
        assert img.shape[2] == 3  # BGR

    def test_loads_png(self, sample_png):
        img, meta = load_image(sample_png)
        assert isinstance(img, np.ndarray)
        assert img.shape[2] == 3

    def test_applies_exif_rotation(self, sample_jpeg_rotated_90cw, music_page):
        """Image with EXIF orientation=6 should be rotated so the result
        has the same orientation as the original (un-rotated) music page."""
        img, meta = load_image(sample_jpeg_rotated_90cw)
        # After EXIF correction, landscape orientation should be restored
        assert img.shape[0] == music_page.shape[0]  # height matches
        assert img.shape[1] == music_page.shape[1]  # width matches

    def test_extracts_exif_metadata(self, sample_jpeg):
        _, meta = load_image(sample_jpeg)
        assert isinstance(meta, dict)
        assert "orientation" in meta

    def test_returns_empty_exif_for_png(self, sample_png):
        _, meta = load_image(sample_png)
        assert isinstance(meta, dict)
        # PNG has no EXIF -- metadata should still be a dict but orientation absent
        assert meta.get("orientation") is None

    def test_returns_bgr_uint8(self, sample_jpeg):
        img, _ = load_image(sample_jpeg)
        assert img.dtype == np.uint8

    def test_rejects_nonexistent_path(self, tmp_path):
        with pytest.raises((FileNotFoundError, OSError)):
            load_image(tmp_path / "does_not_exist.jpg")

    def test_accepts_path_or_string(self, sample_jpeg):
        img1, _ = load_image(sample_jpeg)
        img2, _ = load_image(str(sample_jpeg))
        np.testing.assert_array_equal(img1, img2)


class TestSaveCheckpoint:
    """Tests for save_checkpoint(): lossless PNG, atomic writes, metadata sidecars."""

    def test_saves_as_png(self, tmp_path, music_page):
        stage_dir = tmp_path / "02_oriented"
        stage_dir.mkdir()
        save_checkpoint(music_page, stage_dir, "IMG_0001")

        out_path = stage_dir / "IMG_0001.png"
        assert out_path.exists()
        loaded = cv2.imread(str(out_path), cv2.IMREAD_UNCHANGED)
        assert loaded is not None
        np.testing.assert_array_equal(loaded, music_page)

    def test_no_tmp_files_remain(self, tmp_path, music_page):
        stage_dir = tmp_path / "02_oriented"
        stage_dir.mkdir()
        save_checkpoint(music_page, stage_dir, "IMG_0001")

        tmp_files = list(stage_dir.glob("*.tmp.png"))
        assert tmp_files == []

    def test_writes_metadata_sidecar(self, tmp_path, music_page):
        stage_dir = tmp_path / "test_stage"
        stage_dir.mkdir()
        meta = {"orientation": 1, "focus_score": 42.5}
        save_checkpoint(music_page, stage_dir, "IMG_0001", metadata=meta)

        sidecar = stage_dir / "IMG_0001.json"
        assert sidecar.exists()
        loaded_meta = json.loads(sidecar.read_text())
        assert loaded_meta["orientation"] == 1
        assert loaded_meta["focus_score"] == 42.5

    def test_no_sidecar_when_no_metadata(self, tmp_path, music_page):
        stage_dir = tmp_path / "test_stage"
        stage_dir.mkdir()
        save_checkpoint(music_page, stage_dir, "IMG_0001")

        sidecar = stage_dir / "IMG_0001.json"
        assert not sidecar.exists()

    def test_overwrites_existing(self, tmp_path, music_page):
        stage_dir = tmp_path / "test_stage"
        stage_dir.mkdir()
        save_checkpoint(music_page, stage_dir, "IMG_0001")

        modified = music_page.copy()
        modified[:10, :10] = 0
        save_checkpoint(modified, stage_dir, "IMG_0001")

        loaded = cv2.imread(str(stage_dir / "IMG_0001.png"), cv2.IMREAD_UNCHANGED)
        np.testing.assert_array_equal(loaded, modified)

    def test_creates_lossless_output(self, tmp_path, music_page):
        """PNG should be perfectly lossless -- no pixel differences."""
        stage_dir = tmp_path / "test_stage"
        stage_dir.mkdir()
        save_checkpoint(music_page, stage_dir, "IMG_0001")

        loaded = cv2.imread(str(stage_dir / "IMG_0001.png"), cv2.IMREAD_UNCHANGED)
        assert np.array_equal(loaded, music_page)


class TestEnsureCheckpointDir:
    """Tests for ensure_checkpoint_dir()."""

    def test_creates_directory(self, tmp_path):
        result = ensure_checkpoint_dir(tmp_path, "02_oriented")
        assert result.exists()
        assert result.is_dir()
        assert result == tmp_path / "02_oriented"

    def test_idempotent(self, tmp_path):
        dir1 = ensure_checkpoint_dir(tmp_path, "02_oriented")
        dir2 = ensure_checkpoint_dir(tmp_path, "02_oriented")
        assert dir1 == dir2
        assert dir1.exists()

    def test_returns_path(self, tmp_path):
        result = ensure_checkpoint_dir(tmp_path, "09_enhanced")
        assert isinstance(result, Path)
