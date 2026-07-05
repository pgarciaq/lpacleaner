"""TDD tests for ghh.utils.preprocess -- hotspot removal (R1), finger detection (R8)."""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from tests.conftest import add_finger, add_hotspot, make_music_page
from ghh.config import Config
from ghh.utils.preprocess import (
    detect_fingers,
    remove_fingers,
    remove_hotspots,
)


@pytest.fixture
def default_cfg(tmp_path) -> Config:
    return Config(input_dir=tmp_path)


@pytest.fixture
def finger_cfg(tmp_path) -> Config:
    return Config(input_dir=tmp_path, fingers_detected=True)


# ---------------------------------------------------------------------------
# TestRemoveHotspots (R1)
# ---------------------------------------------------------------------------

class TestRemoveHotspots:
    """Tests for remove_hotspots(): flash hotspot / specular highlight removal."""

    def test_removes_bright_spot(self, default_cfg):
        page = make_music_page()
        with_hotspot = add_hotspot(page, center=(400, 300), radius=80)
        result, meta = remove_hotspots(with_hotspot, default_cfg)
        assert result.shape == with_hotspot.shape
        # The bright center should no longer be clipped white
        center_val = result[300, 400]
        assert not all(c > 250 for c in center_val), "Hotspot center should be inpainted"

    def test_records_hotspot_in_metadata(self, default_cfg):
        page = make_music_page()
        with_hotspot = add_hotspot(page, center=(400, 300), radius=80)
        _, meta = remove_hotspots(with_hotspot, default_cfg)
        assert meta.get("hotspot_detected") is True

    def test_no_change_when_no_hotspot(self, default_cfg):
        page = make_music_page()
        result, meta = remove_hotspots(page, default_cfg)
        assert meta.get("hotspot_detected") is False
        # Image should be unchanged (or very close)
        diff = cv2.absdiff(result, page)
        assert diff.sum() == 0, "Image without hotspot should not be modified"

    def test_preserves_image_dimensions(self, default_cfg):
        page = make_music_page()
        with_hotspot = add_hotspot(page, radius=30)
        result, _ = remove_hotspots(with_hotspot, default_cfg)
        assert result.shape == with_hotspot.shape
        assert result.dtype == np.uint8


# ---------------------------------------------------------------------------
# TestDetectFingers (R8)
# ---------------------------------------------------------------------------

class TestDetectFingers:
    """Tests for detect_fingers(): skin-colored region detection at borders."""

    def test_detects_finger_at_border(self, finger_cfg):
        page = make_music_page()
        with_finger = add_finger(page, position="top-right", size_frac=0.08)
        mask = detect_fingers(with_finger, finger_cfg)
        assert mask.dtype == np.uint8
        # Should detect some skin-colored pixels in top-right
        h, w = mask.shape[:2]
        tr_region = mask[0:h // 4, 3 * w // 4:]
        assert np.count_nonzero(tr_region) > 0, "Should detect finger in top-right"

    def test_no_detection_on_clean_page(self, finger_cfg):
        page = make_music_page()
        mask = detect_fingers(page, finger_cfg)
        finger_ratio = np.count_nonzero(mask) / mask.size
        assert finger_ratio < 0.01, "Clean page should have no significant finger detection"

    def test_returns_binary_mask(self, finger_cfg):
        page = make_music_page()
        with_finger = add_finger(page)
        mask = detect_fingers(with_finger, finger_cfg)
        unique = set(np.unique(mask))
        assert unique <= {0, 255}


# ---------------------------------------------------------------------------
# TestRemoveFingers (R8)
# ---------------------------------------------------------------------------

class TestRemoveFingers:
    """Tests for remove_fingers(): inpainting detected finger regions."""

    def test_removes_finger_region(self, finger_cfg):
        page = make_music_page()
        with_finger = add_finger(page, position="top-right", size_frac=0.08)
        mask = detect_fingers(with_finger, finger_cfg)
        if np.count_nonzero(mask) == 0:
            pytest.skip("Finger not detected by detect_fingers on this synthetic image")
        result = remove_fingers(with_finger, mask, finger_cfg)
        assert result.shape == with_finger.shape
        # The inpainted region should differ from the finger region
        diff = cv2.absdiff(result, with_finger)
        assert diff.sum() > 0, "Finger area should be inpainted (changed)"

    def test_no_change_with_empty_mask(self, finger_cfg):
        page = make_music_page()
        mask = np.zeros(page.shape[:2], dtype=np.uint8)
        result = remove_fingers(page, mask, finger_cfg)
        np.testing.assert_array_equal(result, page)

    def test_preserves_image_dimensions(self, finger_cfg):
        page = make_music_page()
        with_finger = add_finger(page, size_frac=0.06)
        mask = detect_fingers(with_finger, finger_cfg)
        result = remove_fingers(with_finger, mask, finger_cfg)
        assert result.shape == page.shape
        assert result.dtype == np.uint8
