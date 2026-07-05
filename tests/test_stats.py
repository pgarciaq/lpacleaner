"""TDD tests for utils/stats.py -- robust statistics with outlier rejection."""

from __future__ import annotations

import numpy as np
import pytest

from ghh.utils.stats import robust_median, adaptive_sample_count


# ---------------------------------------------------------------------------
# TestRobustMedian
# ---------------------------------------------------------------------------

class TestRobustMedian:
    """Test median computation with MAD-based outlier rejection."""

    def test_clean_data_returns_median(self):
        values = [10.0, 12.0, 11.0, 13.0, 10.5]
        result = robust_median(values)
        assert result == pytest.approx(11.0, abs=0.5)

    def test_rejects_outliers(self):
        values = [10.0, 11.0, 12.0, 10.5, 11.5, 100.0]
        result = robust_median(values)
        assert result < 15.0

    def test_empty_input_returns_none(self):
        result = robust_median([])
        assert result is None

    def test_single_value_returns_itself(self):
        result = robust_median([42.0])
        assert result == 42.0

    def test_all_identical_values(self):
        result = robust_median([5.0, 5.0, 5.0, 5.0])
        assert result == 5.0

    def test_two_values_returns_mean(self):
        result = robust_median([10.0, 20.0])
        assert result == pytest.approx(15.0)

    def test_respects_mad_threshold(self):
        """With a stricter threshold, more values are rejected."""
        values = [10.0, 11.0, 12.0, 10.5, 11.5, 20.0]
        strict = robust_median(values, mad_threshold=1.0)
        lenient = robust_median(values, mad_threshold=3.0)
        assert strict is not None and lenient is not None


# ---------------------------------------------------------------------------
# TestAdaptiveSampleCount
# ---------------------------------------------------------------------------

class TestAdaptiveSampleCount:
    """Test adaptive sample count formula: max(10, min(30, total // 10))."""

    def test_small_book_clamps_to_minimum(self):
        assert adaptive_sample_count(50) == 10

    def test_medium_book_uses_formula(self):
        assert adaptive_sample_count(225) == 22

    def test_large_book_clamps_to_maximum(self):
        assert adaptive_sample_count(800) == 30

    def test_exact_boundary_100(self):
        assert adaptive_sample_count(100) == 10

    def test_exact_boundary_300(self):
        assert adaptive_sample_count(300) == 30

    def test_very_small_book(self):
        """With fewer than 10 images, return total (can't sample more than exist)."""
        assert adaptive_sample_count(5) == 5

    def test_zero_images(self):
        assert adaptive_sample_count(0) == 0
