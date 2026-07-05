"""TDD tests for lpacleaner.utils.geometry -- quad ordering, homography, distance helpers."""

from __future__ import annotations

import numpy as np
import pytest

from lpacleaner.utils.geometry import (
    compute_target_size,
    get_perspective_transform,
    order_corners,
)


class TestOrderCorners:
    """Tests for order_corners(): returns [top-left, top-right, bottom-right, bottom-left]."""

    def test_already_ordered(self):
        pts = np.array([[10, 10], [90, 10], [90, 90], [10, 90]], dtype=np.float32)
        ordered = order_corners(pts)
        np.testing.assert_array_equal(ordered[0], [10, 10])  # TL
        np.testing.assert_array_equal(ordered[1], [90, 10])  # TR
        np.testing.assert_array_equal(ordered[2], [90, 90])  # BR
        np.testing.assert_array_equal(ordered[3], [10, 90])  # BL

    def test_shuffled_corners(self):
        # Same rectangle, points in random order
        pts = np.array([[90, 90], [10, 10], [10, 90], [90, 10]], dtype=np.float32)
        ordered = order_corners(pts)
        np.testing.assert_array_equal(ordered[0], [10, 10])  # TL
        np.testing.assert_array_equal(ordered[1], [90, 10])  # TR
        np.testing.assert_array_equal(ordered[2], [90, 90])  # BR
        np.testing.assert_array_equal(ordered[3], [10, 90])  # BL

    def test_near_rectangular(self):
        # Slightly skewed quadrilateral
        pts = np.array([[12, 8], [88, 12], [92, 88], [8, 92]], dtype=np.float32)
        ordered = order_corners(pts)
        # TL should be closest to (0,0): (12, 8)
        assert ordered[0][0] < ordered[1][0]  # TL.x < TR.x
        assert ordered[0][1] < ordered[3][1]  # TL.y < BL.y
        # BR should be at bottom-right
        assert ordered[2][0] > ordered[3][0]  # BR.x > BL.x
        assert ordered[2][1] > ordered[1][1]  # BR.y > TR.y

    def test_returns_float32(self):
        pts = np.array([[10, 10], [90, 10], [90, 90], [10, 90]], dtype=np.int32)
        ordered = order_corners(pts)
        assert ordered.dtype == np.float32

    def test_four_points_required(self):
        pts = np.array([[10, 10], [90, 10], [90, 90]], dtype=np.float32)
        with pytest.raises((ValueError, IndexError)):
            order_corners(pts)


class TestComputeTargetSize:
    """Tests for compute_target_size(): rectangle dimensions from quad corners."""

    def test_perfect_rectangle(self):
        corners = np.array([[0, 0], [100, 0], [100, 80], [0, 80]], dtype=np.float32)
        w, h = compute_target_size(corners)
        assert w == pytest.approx(100, abs=1)
        assert h == pytest.approx(80, abs=1)

    def test_skewed_quadrilateral(self):
        corners = np.array([[5, 3], [103, 7], [98, 85], [2, 82]], dtype=np.float32)
        w, h = compute_target_size(corners)
        # Width should be approximately 100, height ~80
        assert 90 < w < 110
        assert 70 < h < 90

    def test_returns_integers(self):
        corners = np.array([[0, 0], [100, 0], [100, 80], [0, 80]], dtype=np.float32)
        w, h = compute_target_size(corners)
        assert isinstance(w, int)
        assert isinstance(h, int)

    def test_uses_max_not_average(self):
        """Width/height should use the longer of opposite edges."""
        corners = np.array([
            [0, 0], [200, 0], [180, 100], [20, 100],
        ], dtype=np.float32)
        w, h = compute_target_size(corners)
        assert w == 200, "Should use max(top=200, bottom=160), not average"


class TestGetPerspectiveTransform:
    """Tests for get_perspective_transform(): 3x3 transformation matrix."""

    def test_returns_3x3_matrix(self):
        src = np.array([[10, 10], [90, 15], [85, 90], [15, 85]], dtype=np.float32)
        dst = np.array([[0, 0], [80, 0], [80, 80], [0, 80]], dtype=np.float32)
        M = get_perspective_transform(src, dst)
        assert M.shape == (3, 3)

    def test_identity_for_same_points(self):
        pts = np.array([[0, 0], [100, 0], [100, 100], [0, 100]], dtype=np.float32)
        M = get_perspective_transform(pts, pts)
        expected = np.eye(3, dtype=np.float64)
        np.testing.assert_array_almost_equal(M, expected, decimal=5)

    def test_transform_maps_corners_correctly(self):
        import cv2
        src = np.array([[20, 20], [80, 25], [75, 75], [25, 70]], dtype=np.float32)
        dst = np.array([[0, 0], [100, 0], [100, 100], [0, 100]], dtype=np.float32)
        M = get_perspective_transform(src, dst)

        # Apply transform to src points, should land on dst points
        for i in range(4):
            pt = np.array([src[i][0], src[i][1], 1.0])
            result = M @ pt
            result = result[:2] / result[2]
            np.testing.assert_array_almost_equal(result, dst[i], decimal=1)
