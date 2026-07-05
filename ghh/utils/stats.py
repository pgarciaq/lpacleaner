"""Robust statistics utilities for the analyze command.

Provides median-based aggregation with MAD (Median Absolute Deviation)
outlier rejection, designed for noisy measurements from raw book photos.
"""

from __future__ import annotations

import numpy as np


def robust_median(
    values: list[float] | np.ndarray,
    mad_threshold: float = 2.0,
) -> float | None:
    """Compute median after rejecting outliers beyond *mad_threshold* MADs.

    Parameters
    ----------
    values : list or array of floats
    mad_threshold : number of MADs from the median to keep (default 2.0)

    Returns
    -------
    float or None if input is empty.
    """
    if len(values) == 0:
        return None

    arr = np.asarray(values, dtype=np.float64)

    if len(arr) <= 2:
        return float(np.median(arr))

    med = float(np.median(arr))
    mad = float(np.median(np.abs(arr - med)))

    if mad < 1e-9:
        return med

    keep = np.abs(arr - med) <= mad_threshold * mad
    filtered = arr[keep]

    if len(filtered) == 0:
        return med

    return float(np.median(filtered))


def adaptive_sample_count(total_images: int) -> int:
    """Compute how many images to sample for analysis.

    Formula: max(10, min(30, total // 10)), clamped to total.
    """
    if total_images <= 0:
        return 0
    n = max(10, min(30, total_images // 10))
    return min(n, total_images)
