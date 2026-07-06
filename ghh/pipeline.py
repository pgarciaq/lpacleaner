"""Pipeline orchestrator: BaseStage contract, PipelineState, and stage runner.

Every stage subclasses BaseStage and implements process_image(). The run()
method in BaseStage handles: per-image iteration, resume (skip already-done
images), error handling per error_class, atomic checkpoint writes, and
metadata sidecar output.

PipelineState tracks per-image completion, config hashes for cache
invalidation, and stage results -- persisted as pipeline.json.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import threading
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from ghh.config import Config
from ghh.utils.image_io import ensure_checkpoint_dir, save_checkpoint

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# StageResult
# ---------------------------------------------------------------------------

@dataclass
class StageResult:
    """Outcome of running a stage across all images."""

    stage_name: str
    processed: int = 0
    skipped: int = 0     # already done (resume)
    failed: int = 0      # error class applied
    excluded: int = 0    # critical: image removed from pipeline

    def to_dict(self) -> dict:
        return {
            "stage_name": self.stage_name,
            "processed": self.processed,
            "skipped": self.skipped,
            "failed": self.failed,
            "excluded": self.excluded,
        }

    @classmethod
    def from_dict(cls, d: dict) -> StageResult:
        return cls(**d)


# ---------------------------------------------------------------------------
# PipelineState
# ---------------------------------------------------------------------------

class PipelineState:
    """Persistent state for the pipeline, stored as pipeline.json.

    Tracks:
    - Which images are done per stage (for resume)
    - Config hashes per stage (for cache invalidation)
    - Stage results (for end-of-run report)
    - Config source ("defaults" or "analyzed")
    """

    def __init__(self, output_dir: str | Path):
        self.output_dir = Path(output_dir)
        self._done: dict[str, set[str]] = {}       # stage -> set of done image stems
        self._config_hashes: dict[str, str] = {}    # stage -> hash
        self._results: dict[str, StageResult] = {}  # stage -> result
        self.config_source: str = "defaults"

    def mark_image_done(self, stage_name: str, image_stem: str) -> None:
        self._done.setdefault(stage_name, set()).add(image_stem)

    def is_image_done(self, stage_name: str, image_stem: str) -> bool:
        return image_stem in self._done.get(stage_name, set())

    def set_stage_hash(self, stage_name: str, hash_val: str) -> None:
        self._config_hashes[stage_name] = hash_val

    def get_stage_hash(self, stage_name: str) -> str | None:
        return self._config_hashes.get(stage_name)

    def is_stage_invalidated(self, stage_name: str, current_hash: str) -> bool:
        stored = self._config_hashes.get(stage_name)
        if stored is None:
            return True
        return stored != current_hash

    def invalidate_stage(self, stage_name: str) -> None:
        self._done.pop(stage_name, None)
        self._config_hashes.pop(stage_name, None)

    def record_result(self, result: StageResult) -> None:
        self._results[result.stage_name] = result

    def get_result(self, stage_name: str) -> StageResult | None:
        return self._results.get(stage_name)

    def save(self) -> None:
        data = {
            "config_source": self.config_source,
            "config_hashes": self._config_hashes,
            "done": {stage: sorted(stems) for stage, stems in self._done.items()},
            "results": {name: r.to_dict() for name, r in self._results.items()},
        }
        self.output_dir.mkdir(parents=True, exist_ok=True)
        path = self.output_dir / "pipeline.json"
        tmp = self.output_dir / "pipeline.json.tmp"
        tmp.write_text(json.dumps(data, indent=2))
        os.replace(str(tmp), str(path))

    @classmethod
    def load(cls, output_dir: str | Path) -> PipelineState:
        state = cls(output_dir)
        path = Path(output_dir) / "pipeline.json"
        if not path.exists():
            return state
        try:
            data = json.loads(path.read_text())
            state.config_source = data.get("config_source", "defaults")
            state._config_hashes = data.get("config_hashes", {})
            state._done = {
                stage: set(stems) for stage, stems in data.get("done", {}).items()
            }
            for name, rd in data.get("results", {}).items():
                state._results[name] = StageResult.from_dict(rd)
        except (json.JSONDecodeError, KeyError, TypeError) as exc:
            logger.warning("Corrupt pipeline.json, starting fresh: %s", exc)
        return state


# ---------------------------------------------------------------------------
# BaseStage
# ---------------------------------------------------------------------------

class BaseStage(ABC):
    """Abstract base for all pipeline stages.

    Subclasses set class attributes and implement process_image().
    The run() method handles orchestration: iteration, resume, error
    handling, checkpoint writing.
    """

    name: str                # e.g., "preprocess"
    number: int              # e.g., 0
    checkpoint_name: str     # e.g., "00_preprocessed"
    error_class: str         # "skippable", "critical", "fatal"
    writes_image: bool = True  # False → symlink to source image (saves disk)

    @abstractmethod
    def process_image(
        self,
        img: np.ndarray,
        metadata: dict,
        cfg: Config,
    ) -> tuple[np.ndarray, dict]:
        """Process a single image. Returns (processed_image, updated_metadata).

        Raise an exception to trigger the error_class policy.
        """

    def should_skip(self, cfg: Config) -> bool:
        """Whether this stage should be skipped entirely (profile/flags)."""
        return cfg.should_skip_stage(self.name)

    @staticmethod
    def count_images(input_dir: Path) -> int:
        """Count processable images in *input_dir*."""
        return sum(
            1 for p in Path(input_dir).iterdir()
            if p.suffix.lower() in (".png", ".jpg", ".jpeg", ".tiff", ".tif")
        )

    def run(
        self,
        input_dir: Path,
        output_dir: Path,
        cfg: Config,
        state: PipelineState,
        progress_callback: callable | None = None,
        max_workers: int = 1,
    ) -> StageResult:
        """Run the stage on all images in input_dir.

        Handles:
        - Resume: skips images already marked done in state
        - Error handling: applies error_class policy per image
        - Atomic checkpoint writes via save_checkpoint()
        - Metadata sidecar output
        - Parallel processing when *max_workers* > 1
        """
        stage_dir = ensure_checkpoint_dir(output_dir, self.checkpoint_name)
        result = StageResult(stage_name=self.name)

        image_files = sorted(
            p for p in Path(input_dir).iterdir()
            if p.suffix.lower() in (".png", ".jpg", ".jpeg", ".tiff", ".tif")
        )

        if max_workers > 1:
            lock = threading.Lock()
            fatal_exc: BaseException | None = None

            def _worker(img_path: Path) -> None:
                nonlocal fatal_exc
                self._process_one(
                    img_path, stage_dir, cfg, state, result,
                    lock, progress_callback,
                )

            workers = min(max_workers, len(image_files) or 1)
            with ThreadPoolExecutor(max_workers=workers) as pool:
                futures = {
                    pool.submit(_worker, p): p for p in image_files
                }
                for fut in as_completed(futures):
                    try:
                        fut.result()
                    except Exception as exc:
                        if self.error_class == "fatal":
                            fatal_exc = exc
                            for f in futures:
                                f.cancel()
                            break

            if fatal_exc is not None:
                raise fatal_exc
        else:
            for img_path in image_files:
                self._process_one(
                    img_path, stage_dir, cfg, state, result,
                    None, progress_callback,
                )

        return result

    def _process_one(
        self,
        img_path: Path,
        stage_dir: Path,
        cfg: Config,
        state: PipelineState,
        result: StageResult,
        lock: threading.Lock | None,
        progress_callback: callable | None,
    ) -> None:
        """Process a single image. Thread-safe when *lock* is provided."""
        stem = img_path.stem

        if state.is_image_done(self.checkpoint_name, stem):
            out_path = stage_dir / f"{stem}.png"
            if out_path.exists():
                if lock:
                    with lock:
                        result.skipped += 1
                else:
                    result.skipped += 1
                if progress_callback is not None:
                    progress_callback()
                return

        try:
            img = cv2.imread(str(img_path), cv2.IMREAD_UNCHANGED)
            if img is None:
                raise OSError(f"Cannot read image: {img_path}")

            metadata: dict = {}
            sidecar_path = img_path.with_suffix(".json")
            if sidecar_path.exists():
                try:
                    metadata = json.loads(sidecar_path.read_text())
                except (json.JSONDecodeError, ValueError, TypeError):
                    logger.debug("Could not read sidecar %s", sidecar_path)

            processed_img, metadata = self.process_image(img, metadata, cfg)

            if self.writes_image:
                save_checkpoint(
                    processed_img, stage_dir, stem,
                    metadata=metadata or None,
                )
            else:
                out_png = stage_dir / f"{stem}.png"
                if out_png.is_symlink() or out_png.exists():
                    out_png.unlink()
                out_png.symlink_to(img_path.resolve())
                if metadata:
                    sidecar = stage_dir / f"{stem}.json"
                    sidecar.write_text(
                        json.dumps(metadata, indent=2, default=str),
                    )

            if lock:
                with lock:
                    state.mark_image_done(self.checkpoint_name, stem)
                    result.processed += 1
            else:
                state.mark_image_done(self.checkpoint_name, stem)
                result.processed += 1
            if progress_callback is not None:
                progress_callback()

        except Exception as exc:
            logger.error(
                "Stage %s failed on %s: %s", self.name, stem, exc,
                exc_info=True,
            )

            if self.error_class == "skippable":
                try:
                    original = cv2.imread(
                        str(img_path), cv2.IMREAD_UNCHANGED,
                    )
                    if original is not None:
                        save_checkpoint(original, stage_dir, stem)
                        if lock:
                            with lock:
                                state.mark_image_done(
                                    self.checkpoint_name, stem,
                                )
                        else:
                            state.mark_image_done(
                                self.checkpoint_name, stem,
                            )
                except Exception:
                    pass
                if lock:
                    with lock:
                        result.failed += 1
                else:
                    result.failed += 1

            elif self.error_class == "critical":
                if lock:
                    with lock:
                        result.excluded += 1
                else:
                    result.excluded += 1

            elif self.error_class == "fatal":
                raise

            if progress_callback is not None:
                progress_callback()
