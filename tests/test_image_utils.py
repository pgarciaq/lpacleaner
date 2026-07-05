"""Tests for utils/image_utils.py: estimate_background and trim_to_content."""

from __future__ import annotations

import numpy as np
import pytest

from ghh.utils.image_utils import estimate_background, trim_to_content


# ---------------------------------------------------------------------------
# TestEstimateBackground
# ---------------------------------------------------------------------------

class TestEstimateBackground:

    def test_uniform_image_returns_that_color(self):
        img = np.full((100, 100, 3), (200, 180, 160), dtype=np.uint8)
        bg = estimate_background(img)
        assert bg == (200, 180, 160)

    def test_ignores_center_content(self):
        img = np.full((200, 200, 3), (230, 230, 230), dtype=np.uint8)
        img[40:160, 40:160] = (10, 10, 10)
        bg = estimate_background(img)
        assert all(c > 200 for c in bg)

    def test_returns_tuple_of_ints(self):
        img = np.random.randint(0, 256, (50, 50, 3), dtype=np.uint8)
        bg = estimate_background(img)
        assert isinstance(bg, tuple)
        assert all(isinstance(c, int) for c in bg)
        assert len(bg) == 3

    def test_grayscale_returns_single_value(self):
        img = np.full((100, 100), 128, dtype=np.uint8)
        bg = estimate_background(img)
        assert bg == (128,)
        assert len(bg) == 1

    def test_custom_border_frac(self):
        img = np.full((100, 100, 3), (100, 100, 100), dtype=np.uint8)
        img[:20, :] = (200, 200, 200)
        img[-20:, :] = (200, 200, 200)
        img[:, :20] = (200, 200, 200)
        img[:, -20:] = (200, 200, 200)
        bg = estimate_background(img, border_frac=0.20)
        assert all(c > 150 for c in bg)

    def test_small_image(self):
        img = np.full((5, 5, 3), (42, 42, 42), dtype=np.uint8)
        bg = estimate_background(img)
        assert bg == (42, 42, 42)


# ---------------------------------------------------------------------------
# TestTrimToContent
# ---------------------------------------------------------------------------

class TestTrimToContent:

    def test_trims_background_border(self):
        content = np.full((50, 50, 3), (0, 0, 0), dtype=np.uint8)
        img = np.full((100, 100, 3), (240, 240, 240), dtype=np.uint8)
        img[25:75, 25:75] = content

        result = trim_to_content(img, bg_color=(240, 240, 240))
        assert result.shape[0] < img.shape[0]
        assert result.shape[1] < img.shape[1]

    def test_adds_margin(self):
        content = np.full((50, 50, 3), (0, 0, 0), dtype=np.uint8)
        img = np.full((200, 200, 3), (240, 240, 240), dtype=np.uint8)
        img[75:125, 75:125] = content

        result = trim_to_content(img, bg_color=(240, 240, 240), margin_frac=0.05)
        assert result.shape[0] > 50
        assert result.shape[1] > 50

    def test_uniform_image_returns_unchanged(self):
        img = np.full((100, 100, 3), (200, 200, 200), dtype=np.uint8)
        result = trim_to_content(img, bg_color=(200, 200, 200))
        assert result.shape == img.shape

    def test_auto_estimates_background(self):
        img = np.full((100, 100, 3), (230, 230, 230), dtype=np.uint8)
        img[30:70, 30:70] = (10, 10, 10)
        result = trim_to_content(img)
        assert result.shape[0] < img.shape[0]

    def test_preserves_content(self):
        bg = (230, 230, 230)
        img = np.full((100, 100, 3), bg, dtype=np.uint8)
        img[40:60, 40:60] = (50, 50, 50)
        result = trim_to_content(img, bg_color=bg, margin_frac=0.0, threshold=30)
        center = result[result.shape[0] // 2, result.shape[1] // 2]
        assert all(c < 100 for c in center)

    def test_grayscale(self):
        img = np.full((100, 100), 200, dtype=np.uint8)
        img[30:70, 30:70] = 20
        result = trim_to_content(img, bg_color=(200,))
        assert result.shape[0] < 100

    def test_custom_threshold(self):
        img = np.full((100, 100, 3), (200, 200, 200), dtype=np.uint8)
        img[30:70, 30:70] = (180, 180, 180)
        result_strict = trim_to_content(img, bg_color=(200, 200, 200), threshold=10)
        result_loose = trim_to_content(img, bg_color=(200, 200, 200), threshold=100)
        assert result_strict.shape[0] < img.shape[0]
        assert result_loose.shape == img.shape
