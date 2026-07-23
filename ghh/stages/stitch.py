"""Stage 1: Image grouping and stitching.

Unlike most stages, stitch operates on the full image set at once
(for grouping detection), then processes each group. It overrides
BaseStage.run() to implement batch-level logic while still using
atomic checkpoints and per-image state tracking.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import cv2

from ghh.config import Config
from ghh.pipeline import BaseStage, PipelineState, StageResult
from ghh.utils.image_io import ensure_checkpoint_dir, save_checkpoint
from ghh.utils.stitch import (
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
    config_keys = ("exclude_images", "include_covers")

    def process_image(self, img, metadata, cfg):
        # Not used -- run() handles batch processing directly
        raise NotImplementedError("StitchStage uses batch run(), not per-image processing")

    def run(
        self,
        input_dir: Path,
        output_dir: Path,
        cfg: Config,
        state: PipelineState,
        progress_callback: callable | None = None,
        max_workers: int = 1,
    ) -> StageResult:
        stage_dir = ensure_checkpoint_dir(output_dir, self.checkpoint_name)
        result = StageResult(stage_name=self.name)

        image_files = sorted(
            p for p in Path(input_dir).iterdir()
            if p.suffix.lower() in (".png", ".jpg", ".jpeg", ".tiff", ".tif")
        )

        if not image_files:
            return result

        # Build stem→path mapping (no pixel data loaded yet)
        source_paths: dict[str, Path] = {p.stem: p for p in image_files}

        # Filter out excluded images
        if cfg.exclude_images:
            exclude_stems = {
                n.rsplit(".", 1)[0] if "." in n else n for n in cfg.exclude_images
            }
            for stem in list(source_paths):
                if stem in exclude_stems:
                    del source_paths[stem]
                    logger.info("Excluded image: %s (manual override)", stem)

        # Detect non-content images (load one at a time)
        non_content: list[str] = []
        for stem in list(source_paths):
            img = cv2.imread(str(source_paths[stem]), cv2.IMREAD_UNCHANGED)
            if img is not None and is_non_content(img):
                non_content.append(stem)
                if not cfg.include_covers:
                    del source_paths[stem]
                    logger.info("Excluded non-content image: %s", stem)

        # Detect groups (on-demand: loads consecutive pairs, not all at once)
        groups = detect_groups(
            None, cfg, image_paths=source_paths,
        )
        logger.info(
            "Detected %d groups from %d images", len(groups), len(source_paths),
        )

        # Process each group (load images only when needed)
        for group in groups:
            output_stem = group[0]

            if state.is_image_done(self.checkpoint_name, output_stem):
                out_path = stage_dir / f"{output_stem}.png"
                if out_path.exists():
                    result.skipped += 1
                    if progress_callback is not None:
                        progress_callback()
                    continue

            try:
                group_images = {}
                for n in group:
                    if n in source_paths:
                        loaded = cv2.imread(
                            str(source_paths[n]), cv2.IMREAD_UNCHANGED,
                        )
                        if loaded is not None:
                            group_images[n] = loaded

                if len(group_images) == 0:
                    continue

                meta: dict = {
                    "stage": self.name,
                    "group": group,
                    "non_content_detected": [s for s in group if s in non_content],
                }

                if len(group_images) > 1:
                    kept, discarded = deduplicate_retakes(group_images, cfg)
                    if discarded:
                        meta["retakes_discarded"] = discarded
                        logger.info("Discarded retakes: %s", list(discarded.keys()))
                    group_images = kept

                if len(group_images) > 1:
                    stitched, method, success = stitch_images(
                        group_images, cfg,
                    )
                    meta["stitch_method"] = method
                    meta["stitch_success"] = success
                    save_checkpoint(
                        stitched, stage_dir, output_stem,
                        metadata=meta,
                    )
                else:
                    single_name = next(iter(group_images))
                    meta["stitch_method"] = "single"
                    meta["stitch_success"] = True
                    src = source_paths.get(single_name)
                    if src is not None:
                        out_png = stage_dir / f"{output_stem}.png"
                        if out_png.is_symlink() or out_png.exists():
                            out_png.unlink()
                        out_png.symlink_to(src.resolve())
                    else:
                        save_checkpoint(
                            group_images[single_name],
                            stage_dir, output_stem,
                        )
                    sidecar = stage_dir / f"{output_stem}.json"
                    sidecar.write_text(
                        json.dumps(meta, indent=2, default=str),
                    )

                state.mark_image_done(self.checkpoint_name, output_stem)
                result.processed += 1
                if progress_callback is not None:
                    progress_callback()

            except Exception as exc:
                logger.error(
                    "Stage %s failed on group %s: %s", self.name, group, exc,
                    exc_info=True,
                )
                try:
                    fallback_name = group[0]
                    if fallback_name in source_paths:
                        fallback_img = cv2.imread(
                            str(source_paths[fallback_name]),
                            cv2.IMREAD_UNCHANGED,
                        )
                        if fallback_img is not None:
                            save_checkpoint(
                                fallback_img, stage_dir, output_stem,
                            )
                            state.mark_image_done(
                                self.checkpoint_name, output_stem,
                            )
                except Exception:
                    pass
                result.failed += 1
                if progress_callback is not None:
                    progress_callback()

        return result
