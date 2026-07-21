"""Stage 13: Optical Music Recognition (OMR).

Runs OpenVINO inference on music pages (as classified by stage 4
PageDetect) and produces ``.gabc`` transcriptions alongside symlinked
PNGs.  Non-music pages pass through with a skip sidecar.

Requires the ``chant-omr`` package (installed separately) and a
directory of exported OpenVINO IR models pointed to by
``cfg.omr_model_dir``.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import cv2
import numpy as np

from ghh.config import Config
from ghh.pipeline import BaseStage, PipelineState, StageResult
from ghh.utils.image_io import ensure_checkpoint_dir

logger = logging.getLogger(__name__)

try:
    from chant_omr.inference.ov_decode import (
        OvModelBundle,
        load_openvino_models,
        ov_predict_gabc_from_array,
    )
    from chant_omr.inference.preprocess import prepare_inference_numpy_from_array

    CHANT_OMR_AVAILABLE = True
except ImportError:
    CHANT_OMR_AVAILABLE = False

_IMAGE_EXTENSIONS = frozenset((".png", ".jpg", ".jpeg", ".tiff", ".tif"))


class OmrStage(BaseStage):
    name = "omr"
    number = 13
    checkpoint_name = "13_omr"
    error_class = "skippable"
    writes_image = False

    def process_image(
        self,
        img: np.ndarray,
        metadata: dict,
        cfg: Config,
    ) -> tuple[np.ndarray, dict]:
        raise NotImplementedError("OmrStage uses run() directly")

    def run(
        self,
        input_dir: Path,
        output_dir: Path,
        cfg: Config,
        state: PipelineState,
        progress_callback: callable | None = None,
        max_workers: int = 1,
    ) -> StageResult:
        result = StageResult(stage_name=self.name)
        input_dir = Path(input_dir)
        output_dir = Path(output_dir)
        stage_dir = ensure_checkpoint_dir(output_dir, self.checkpoint_name)

        if not cfg.omr_model_dir:
            logger.warning("omr_model_dir not set, skipping OMR stage")
            return result

        if not CHANT_OMR_AVAILABLE:
            raise RuntimeError(
                "chant-omr is not installed. Install it with: "
                "pip install -e ../chant-omr  (see ghh docs)"
            )

        model_path = Path(cfg.omr_model_dir)
        if not model_path.is_dir():
            raise FileNotFoundError(
                f"OMR model directory not found: {model_path}"
            )

        models = load_openvino_models(model_path, device=cfg.omr_device)

        image_files = sorted(
            p for p in input_dir.iterdir()
            if p.suffix.lower() in _IMAGE_EXTENSIONS
        )

        for img_path in image_files:
            stem = img_path.stem

            if state.is_image_done(self.checkpoint_name, stem):
                out_path = stage_dir / f"{stem}.png"
                if out_path.exists() or out_path.is_symlink():
                    result.skipped += 1
                    if progress_callback is not None:
                        progress_callback()
                    continue

            try:
                self._process_one_omr(
                    img_path, stage_dir, cfg, state, result, models,
                )
            except Exception as exc:
                logger.error(
                    "OMR failed on %s: %s", stem, exc, exc_info=True,
                )
                _symlink_passthrough(img_path, stage_dir, stem)
                result.failed += 1

            if progress_callback is not None:
                progress_callback()

        return result

    def _process_one_omr(
        self,
        img_path: Path,
        stage_dir: Path,
        cfg: Config,
        state: PipelineState,
        result: StageResult,
        models: "OvModelBundle",
    ) -> None:
        stem = img_path.stem

        metadata: dict = {}
        sidecar_in = img_path.with_suffix(".json")
        if sidecar_in.exists():
            try:
                metadata = json.loads(sidecar_in.read_text())
            except (json.JSONDecodeError, ValueError, TypeError):
                pass

        page_type = metadata.get("page_type", "unknown")
        if page_type != "music":
            _symlink_passthrough(img_path, stage_dir, stem)
            sidecar = stage_dir / f"{stem}.json"
            sidecar.write_text(json.dumps(
                {**metadata, "omr_status": "skipped_non_music"},
                indent=2, default=str,
            ))
            state.mark_image_done(self.checkpoint_name, stem)
            result.processed += 1
            return

        img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if img is None:
            raise OSError(f"Cannot read image: {img_path}")

        rgb = img[:, :, ::-1]
        pixel_values = prepare_inference_numpy_from_array(rgb)
        gabc_text = ov_predict_gabc_from_array(
            pixel_values,
            models,
            beam_width=cfg.omr_beam_width,
            name=stem,
        )

        gabc_path = stage_dir / f"{stem}.gabc"
        gabc_path.write_text(gabc_text, encoding="utf-8")

        _symlink_passthrough(img_path, stage_dir, stem)

        sidecar = stage_dir / f"{stem}.json"
        sidecar.write_text(json.dumps(
            {**metadata, "omr_status": "ok", "gabc_file": f"{stem}.gabc"},
            indent=2, default=str,
        ))
        state.mark_image_done(self.checkpoint_name, stem)
        result.processed += 1


def _symlink_passthrough(img_path: Path, stage_dir: Path, stem: str) -> None:
    """Create a symlink to the input image in the stage directory."""
    out_png = stage_dir / f"{stem}.png"
    if out_png.is_symlink() or out_png.exists():
        out_png.unlink()
    out_png.symlink_to(img_path.resolve())
