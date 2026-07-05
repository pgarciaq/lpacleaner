"""Stage 1: Image grouping and stitching.

Unlike most stages, stitch operates on the full image set at once
(for grouping detection), then processes each group. It overrides
BaseStage.run() to implement batch-level logic while still using
atomic checkpoints and per-image state tracking.
"""

from __future__ import annotations

import logging
from pathlib import Path

import cv2
import numpy as np

from ghh.config import Config
from ghh.pipeline import BaseStage, PipelineState, StageResult
from ghh.utils.image_io import ensure_checkpoint_dir, save_checkpoint
from ghh.utils.stitch import (
    compute_focus,
    deduplicate_retakes,
    detect_groups,
    is_non_content,
    stitch_images,
)

logger = logging.getLogger(__name__)


class StitchStage(BaseStage):
    """Stage 1: group overlapping photos and stitch them."""

    name = "stitch"
    number = 1
    checkpoint_name = "01_stitched"
    error_class = "skippable"

    def process_image(self, img, metadata, cfg):
        # Not used -- run() handles batch processing directly
        raise NotImplementedError("StitchStage uses batch run(), not per-image processing")

    def run(
        self,
        input_dir: Path,
        output_dir: Path,
        cfg: Config,
        state: PipelineState,
    ) -> StageResult:
        stage_dir = ensure_checkpoint_dir(output_dir, self.checkpoint_name)
        result = StageResult(stage_name=self.name)

        image_files = sorted(
            p for p in Path(input_dir).iterdir()
            if p.suffix.lower() in (".png", ".jpg", ".jpeg", ".tiff", ".tif")
        )

        if not image_files:
            return result

        # Load all images into memory
        all_images: dict[str, np.ndarray] = {}
        for p in image_files:
            img = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
            if img is not None:
                all_images[p.stem] = img

        # Filter out excluded images
        exclude_stems = set()
        if cfg.exclude_images:
            exclude_stems = {
                n.rsplit(".", 1)[0] if "." in n else n for n in cfg.exclude_images
            }
        for stem in exclude_stems:
            if stem in all_images:
                del all_images[stem]
                logger.info("Excluded image: %s (manual override)", stem)

        # Detect non-content images (covers, spine shots)
        non_content: list[str] = []
        for stem, img in list(all_images.items()):
            if is_non_content(img):
                non_content.append(stem)
                if not cfg.include_covers:
                    del all_images[stem]
                    logger.info("Excluded non-content image: %s", stem)

        # Detect groups
        groups = detect_groups(all_images, cfg)
        logger.info("Detected %d groups from %d images", len(groups), len(all_images))

        # Process each group
        for group in groups:
            output_stem = group[0]

            if state.is_image_done(self.checkpoint_name, output_stem):
                out_path = stage_dir / f"{output_stem}.png"
                if out_path.exists():
                    result.skipped += 1
                    continue

            try:
                group_images = {n: all_images[n] for n in group if n in all_images}

                if len(group_images) == 0:
                    continue

                meta: dict = {
                    "stage": self.name,
                    "group": group,
                    "non_content_detected": [s for s in group if s in non_content],
                }

                if len(group_images) > 1:
                    # Deduplicate retakes
                    kept, discarded = deduplicate_retakes(group_images, cfg)
                    if discarded:
                        meta["retakes_discarded"] = discarded
                        logger.info("Discarded retakes: %s", list(discarded.keys()))
                    group_images = kept

                if len(group_images) > 1:
                    # Stitch
                    stitched, method, success = stitch_images(group_images, cfg)
                    meta["stitch_method"] = method
                    meta["stitch_success"] = success
                    save_checkpoint(stitched, stage_dir, output_stem, metadata=meta)
                else:
                    # Single image (standalone or after dedup)
                    single_name = next(iter(group_images))
                    meta["stitch_method"] = "single"
                    meta["stitch_success"] = True
                    save_checkpoint(group_images[single_name], stage_dir, output_stem, metadata=meta)

                state.mark_image_done(self.checkpoint_name, output_stem)
                result.processed += 1

            except Exception as exc:
                logger.error(
                    "Stage %s failed on group %s: %s", self.name, group, exc,
                    exc_info=True,
                )
                # Skippable: pass through the first image in the group
                try:
                    fallback_name = group[0]
                    if fallback_name in all_images:
                        save_checkpoint(all_images[fallback_name], stage_dir, output_stem)
                        state.mark_image_done(self.checkpoint_name, output_stem)
                except Exception:
                    pass
                result.failed += 1

        return result
