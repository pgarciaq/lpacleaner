"""GPU/OpenCL detection, UMat wrappers, and OpenVINO initialization.

All GPU-accelerated operations fall back to CPU transparently.
"""

from __future__ import annotations

import logging
from functools import lru_cache

import cv2
import numpy as np

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def has_opencl() -> bool:
    """Return True if OpenCL is available and usable."""
    try:
        if not cv2.ocl.haveOpenCL():
            return False
        cv2.ocl.setUseOpenCL(True)
        return cv2.ocl.useOpenCL()
    except Exception:
        return False


@lru_cache(maxsize=1)
def has_openvino_gpu() -> bool:
    """Return True if OpenVINO is installed and has a GPU device."""
    try:
        import openvino as ov
        core = ov.Core()
        return "GPU" in core.available_devices
    except Exception:
        return False


def to_umat(img: np.ndarray) -> cv2.UMat | np.ndarray:
    """Convert ndarray to UMat for GPU processing, or return as-is if no OpenCL."""
    if has_opencl():
        try:
            return cv2.UMat(img)
        except Exception:
            return img
    return img


def from_umat(umat: cv2.UMat | np.ndarray) -> np.ndarray:
    """Convert UMat back to ndarray, or return as-is if already ndarray."""
    if isinstance(umat, cv2.UMat):
        return umat.get()
    return umat


def gpu_canny(gray: np.ndarray, low: float, high: float) -> np.ndarray:
    """Canny edge detection, GPU-accelerated when available."""
    u = to_umat(gray)
    result = cv2.Canny(u, low, high)
    return from_umat(result)


def gpu_clahe(gray: np.ndarray, clip: float = 2.0,
              grid: tuple[int, int] = (8, 8)) -> np.ndarray:
    """CLAHE adaptive histogram equalization, GPU-accelerated when available."""
    clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=grid)
    u = to_umat(gray)
    result = clahe.apply(u)
    return from_umat(result)


def gpu_remap(img: np.ndarray, map_x: np.ndarray,
              map_y: np.ndarray) -> np.ndarray:
    """Image remapping, GPU-accelerated when available."""
    u = to_umat(img)
    mx = to_umat(map_x)
    my = to_umat(map_y)
    result = cv2.remap(u, mx, my, interpolation=cv2.INTER_LINEAR)
    return from_umat(result)
