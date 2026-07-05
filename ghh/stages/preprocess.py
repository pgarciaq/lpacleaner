"""Stage 0: Pre-processing -- hotspot removal (R1) and finger masking (R8).

Runs before stitching so that flash artifacts and skin-colored regions
do not contaminate feature matching or blending in Stage 1.

This is a "skippable" stage: if processing fails on any image, the
original is passed through unchanged.
"""

from __future__ import annotations

import numpy as np

from ghh.config import Config
from ghh.pipeline import BaseStage
from ghh.utils.preprocess import detect_fingers, remove_fingers, remove_hotspots


class PreprocessStage(BaseStage):
    """Stage 0: flash hotspot removal and finger/hand masking."""

    name = "preprocess"
    number = 0
    checkpoint_name = "00_preprocessed"
    error_class = "skippable"

    def process_image(
        self,
        img: np.ndarray,
        metadata: dict,
        cfg: Config,
    ) -> tuple[np.ndarray, dict]:
        result = img
        metadata["stage"] = self.name

        # R1: hotspot removal (always runs -- cheap no-op if no hotspots)
        result, hotspot_meta = remove_hotspots(result, cfg)
        metadata.update(hotspot_meta)

        # R8: finger removal (only when analyze detected fingers)
        if cfg.fingers_detected:
            finger_mask = detect_fingers(result, cfg)
            if np.count_nonzero(finger_mask) > 0:
                result = remove_fingers(result, finger_mask, cfg)
                metadata["finger_removed"] = True
            else:
                metadata["finger_removed"] = False

        return result, metadata
