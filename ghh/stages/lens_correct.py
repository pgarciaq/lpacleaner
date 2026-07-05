"""Stage 3: Lens distortion correction (optional).

Corrects radial (barrel/pincushion) distortion using coefficients from
``book.toml`` (typically set by the ``analyze`` command).  Runs before
page detection so that ``cv2.undistort`` can use the original optical
centre, which is lost after perspective correction.  Straighter page
edges also improve quad detection in Stage 4.

Skipped entirely when both ``lens_distortion_k1`` and
``lens_distortion_k2`` are zero (the default).
"""

from __future__ import annotations

import logging

import cv2
import numpy as np

from ghh.config import Config
from ghh.pipeline import BaseStage

logger = logging.getLogger(__name__)


class LensCorrectStage(BaseStage):
    name = "lens_correct"
    number = 3
    checkpoint_name = "03_lens_corrected"
    error_class = "skippable"

    def should_skip(self, cfg: Config) -> bool:
        if super().should_skip(cfg):
            return True
        if cfg.lens_distortion_k1 == 0.0 and cfg.lens_distortion_k2 == 0.0:
            logger.info("Lens correction skipped: k1=k2=0")
            return True
        return False

    def process_image(
        self,
        img: np.ndarray,
        metadata: dict,
        cfg: Config,
    ) -> tuple[np.ndarray, dict]:
        h, w = img.shape[:2]
        k1 = cfg.lens_distortion_k1
        k2 = cfg.lens_distortion_k2

        fx = fy = float(max(w, h))
        cx, cy = w / 2.0, h / 2.0
        camera_matrix = np.array(
            [[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64,
        )
        dist_coeffs = np.array([k1, k2, 0, 0, 0], dtype=np.float64)

        corrected = cv2.undistort(img, camera_matrix, dist_coeffs)

        meta = {
            "stage": "lens_correct",
            "k1": k1,
            "k2": k2,
            "focal_length_px": fx,
        }

        logger.debug("Lens correction applied: k1=%.4f k2=%.4f fx=%.0f", k1, k2, fx)
        return corrected, meta
