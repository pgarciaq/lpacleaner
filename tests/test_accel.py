"""TDD tests for ghh.utils.accel -- GPU/OpenCL detection and UMat wrappers."""

from __future__ import annotations

import numpy as np
import pytest

from ghh.utils.accel import (
    from_umat,
    gpu_canny,
    gpu_clahe,
    gpu_remap,
    has_opencl,
    has_openvino_gpu,
    to_umat,
)


class TestHasOpenCL:
    """Tests for has_opencl() detection."""

    def test_returns_bool(self):
        result = has_opencl()
        assert isinstance(result, bool)

    def test_is_deterministic(self):
        assert has_opencl() == has_opencl()


class TestHasOpenVINOGPU:
    """Tests for has_openvino_gpu() detection."""

    def test_returns_bool(self):
        result = has_openvino_gpu()
        assert isinstance(result, bool)

    def test_is_deterministic(self):
        assert has_openvino_gpu() == has_openvino_gpu()


class TestUMatConversion:
    """Tests for to_umat() / from_umat() round-trip."""

    def test_round_trip_preserves_data(self):
        img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        umat = to_umat(img)
        back = from_umat(umat)
        np.testing.assert_array_equal(back, img)

    def test_to_umat_returns_correct_type(self):
        import cv2
        img = np.zeros((50, 50, 3), dtype=np.uint8)
        result = to_umat(img)
        # Should be a UMat if OpenCL is available, otherwise ndarray fallback
        assert isinstance(result, (cv2.UMat, np.ndarray))

    def test_from_umat_returns_ndarray(self):
        img = np.zeros((50, 50, 3), dtype=np.uint8)
        umat = to_umat(img)
        result = from_umat(umat)
        assert isinstance(result, np.ndarray)

    def test_handles_grayscale(self):
        gray = np.random.randint(0, 255, (100, 100), dtype=np.uint8)
        umat = to_umat(gray)
        back = from_umat(umat)
        np.testing.assert_array_equal(back, gray)


class TestGPUCanny:
    """Tests for gpu_canny() -- Canny edge detection with GPU acceleration."""

    def test_returns_binary_image(self):
        gray = np.zeros((100, 100), dtype=np.uint8)
        cv2_import = __import__("cv2")
        cv2_import.rectangle(gray, (20, 20), (80, 80), 255, -1)
        edges = gpu_canny(gray, 50, 150)
        assert edges.dtype == np.uint8
        unique = np.unique(edges)
        assert all(v in (0, 255) for v in unique)

    def test_output_shape_matches_input(self):
        gray = np.zeros((120, 160), dtype=np.uint8)
        edges = gpu_canny(gray, 50, 150)
        assert edges.shape == gray.shape

    def test_detects_edges_of_rectangle(self):
        gray = np.zeros((100, 100), dtype=np.uint8)
        cv2_import = __import__("cv2")
        cv2_import.rectangle(gray, (30, 30), (70, 70), 200, -1)
        edges = gpu_canny(gray, 50, 150)
        assert edges.sum() > 0


class TestGPUCLAHE:
    """Tests for gpu_clahe() -- adaptive histogram equalization."""

    def test_returns_same_shape(self):
        gray = np.random.randint(50, 200, (100, 100), dtype=np.uint8)
        result = gpu_clahe(gray, clip=2.0, grid=(8, 8))
        assert result.shape == gray.shape
        assert result.dtype == np.uint8

    def test_increases_contrast(self):
        gray = np.random.randint(100, 150, (100, 100), dtype=np.uint8)
        result = gpu_clahe(gray, clip=4.0, grid=(8, 8))
        assert result.std() >= gray.std() * 0.9  # should not decrease significantly


class TestGPURemap:
    """Tests for gpu_remap() -- image remapping."""

    def test_identity_remap_preserves_image(self):
        img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        h, w = img.shape[:2]
        map_x = np.tile(np.arange(w, dtype=np.float32), (h, 1))
        map_y = np.tile(np.arange(h, dtype=np.float32).reshape(-1, 1), (1, w))
        result = gpu_remap(img, map_x, map_y)
        np.testing.assert_array_equal(result, img)

    def test_output_shape_matches_input(self):
        img = np.zeros((80, 120, 3), dtype=np.uint8)
        h, w = img.shape[:2]
        map_x = np.zeros((h, w), dtype=np.float32)
        map_y = np.zeros((h, w), dtype=np.float32)
        result = gpu_remap(img, map_x, map_y)
        assert result.shape == img.shape
