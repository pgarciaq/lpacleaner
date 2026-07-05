"""TDD tests for utils/page_find.py -- simplified page quad detection.

Used by the analyze command to crop to the page region before measuring
ink color, layout, and condition. Not the full Stage 4 fallback chain.
"""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from ghh.utils.page_find import find_page_quad, crop_to_page


# ---------------------------------------------------------------------------
# Synthetic images
# ---------------------------------------------------------------------------

def _make_page_on_dark_bg(
    page_h: int = 300,
    page_w: int = 400,
    bg_h: int = 500,
    bg_w: int = 600,
    bg_color: tuple = (30, 25, 20),
    page_color: tuple = (220, 210, 200),
) -> np.ndarray:
    """A light-colored page centered on a dark background."""
    img = np.full((bg_h, bg_w, 3), bg_color, dtype=np.uint8)
    y0 = (bg_h - page_h) // 2
    x0 = (bg_w - page_w) // 2
    img[y0:y0 + page_h, x0:x0 + page_w] = page_color
    return img


def _make_uniform_image(h: int = 500, w: int = 600, color: tuple = (200, 200, 200)):
    """A uniform image with no discernible page boundary."""
    return np.full((h, w, 3), color, dtype=np.uint8)


# ---------------------------------------------------------------------------
# TestFindPageQuad
# ---------------------------------------------------------------------------

class TestFindPageQuad:
    """Test the simplified Otsu + largest-contour page quad detector."""

    def test_finds_page_on_dark_background(self):
        img = _make_page_on_dark_bg()
        quad = find_page_quad(img)

        assert quad is not None
        assert quad.shape == (4, 2)
        assert quad.dtype == np.float32

    def test_quad_corners_enclose_page_region(self):
        img = _make_page_on_dark_bg(page_h=300, page_w=400, bg_h=500, bg_w=600)
        quad = find_page_quad(img)

        assert quad is not None
        xs = quad[:, 0]
        ys = quad[:, 1]
        assert min(xs) >= 50
        assert max(xs) <= 550
        assert min(ys) >= 50
        assert max(ys) <= 450

    def test_returns_none_for_uniform_image(self):
        """When no page boundary is detectable, returns None."""
        img = _make_uniform_image()
        quad = find_page_quad(img)

        assert quad is None

    def test_handles_light_page_on_dark_bg(self):
        img = _make_page_on_dark_bg(bg_color=(20, 15, 10), page_color=(230, 220, 210))
        quad = find_page_quad(img)

        assert quad is not None
        assert quad.shape == (4, 2)

    def test_handles_dark_page_on_light_bg(self):
        img = _make_page_on_dark_bg(bg_color=(230, 230, 230), page_color=(60, 50, 40))
        quad = find_page_quad(img)

        assert quad is not None
        assert quad.shape == (4, 2)


# ---------------------------------------------------------------------------
# TestCropToPage
# ---------------------------------------------------------------------------

class TestCropToPage:
    """Test page cropping using detected quad or central fallback."""

    def test_crop_with_quad(self):
        img = _make_page_on_dark_bg(page_h=300, page_w=400, bg_h=500, bg_w=600)
        quad = find_page_quad(img)

        cropped = crop_to_page(img, quad)

        assert cropped is not None
        assert cropped.shape[0] > 0 and cropped.shape[1] > 0
        assert cropped.shape[0] < img.shape[0]
        assert cropped.shape[1] < img.shape[1]

    def test_crop_fallback_when_no_quad(self):
        """When quad is None, crop to central 80% of image."""
        img = _make_uniform_image(h=500, w=600)

        cropped = crop_to_page(img, quad=None)

        assert cropped.shape[0] == 400  # 80% of 500
        assert cropped.shape[1] == 480  # 80% of 600

    def test_crop_preserves_channels(self):
        img = _make_page_on_dark_bg()
        quad = find_page_quad(img)

        cropped = crop_to_page(img, quad)
        assert cropped.ndim == 3
        assert cropped.shape[2] == 3

    def test_crop_output_is_contiguous(self):
        img = _make_page_on_dark_bg()
        quad = find_page_quad(img)

        cropped = crop_to_page(img, quad)
        assert cropped.flags["C_CONTIGUOUS"]
