"""Stage 12: PDF assembly.

Collects all images from the latest completed pipeline stage and
assembles them into a single PDF file.  This is the terminal stage
-- it produces ``<input_dir_name>.pdf`` in the output directory rather
than per-image checkpoint files.

Two compression modes are supported:

- ``jpeg`` (default): PNG images are re-encoded as JPEG at a
  configurable quality (default 90) inside the PDF.  This is the
  only lossy step in the entire pipeline; one compression pass
  produces negligible quality loss.
- ``png``: images are embedded as-is (lossless).  Much larger file
  size but perfect quality.

The stage dynamically resolves its input: the CLI's
``_find_previous_checkpoint`` walks backward from stage 12 and uses
whichever checkpoint directory exists, so the PDF always contains
the output of the last completed stage.

If no input images are found the stage raises an error (error_class
is ``fatal``).
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path

import cv2
import img2pdf
import numpy as np

from ghh.config import Config
from ghh.pipeline import BaseStage, PipelineState, StageResult

logger = logging.getLogger(__name__)

_IMAGE_EXTENSIONS = frozenset((".png", ".jpg", ".jpeg", ".tiff", ".tif"))


def _make_layout_fun(dpi: int):
    """Return an img2pdf layout function that sizes pages at the given DPI."""
    def layout_fun(imgwidthpx, imgheightpx, ndpi):
        imgwidthpt = imgwidthpx * 72.0 / dpi
        imgheightpt = imgheightpx * 72.0 / dpi
        return imgwidthpt, imgheightpt, imgwidthpt, imgheightpt
    return layout_fun


class PDFAssemblyStage(BaseStage):
    name = "pdf_assembly"
    number = 12
    checkpoint_name = "12_pdf"
    error_class = "fatal"

    def process_image(
        self,
        img: np.ndarray,
        metadata: dict,
        cfg: Config,
    ) -> tuple[np.ndarray, dict]:
        raise NotImplementedError(
            "PDFAssemblyStage does not process individual images"
        )

    def run(
        self,
        input_dir: Path,
        output_dir: Path,
        cfg: Config,
        state: PipelineState,
        progress_callback: callable | None = None,
    ) -> StageResult:
        result = StageResult(stage_name=self.name)
        output_dir = Path(output_dir)
        input_dir = Path(input_dir)

        pdf_name = f"{cfg.input_dir.name}.pdf"
        pdf_path = output_dir / pdf_name

        if state.is_image_done(self.checkpoint_name, "output") and pdf_path.exists():
            result.skipped = 1
            return result

        image_files = _collect_images(input_dir, cfg)

        if not image_files:
            raise RuntimeError(
                f"No images found in {input_dir} for PDF assembly"
            )

        compression = cfg.pdf_compression.lower()
        if compression not in ("jpeg", "png"):
            logger.warning(
                "Unknown pdf_compression %r, defaulting to jpeg",
                compression,
            )
            compression = "jpeg"

        if compression == "jpeg":
            image_data = list(_jpeg_generator(image_files, cfg.pdf_jpeg_quality))
        else:
            image_data = [p.read_bytes() for p in image_files]

        dpi = cfg.pdf_dpi
        layout = _make_layout_fun(dpi)

        pdf_bytes = img2pdf.convert(image_data, layout_fun=layout)

        tmp_path = output_dir / f"{cfg.input_dir.name}.pdf.tmp"
        tmp_path.write_bytes(pdf_bytes)
        os.replace(str(tmp_path), str(pdf_path))

        file_size = pdf_path.stat().st_size

        meta = {
            "stage": "pdf_assembly",
            "page_count": len(image_files),
            "compression": compression,
            "jpeg_quality": cfg.pdf_jpeg_quality if compression == "jpeg" else None,
            "dpi": dpi,
            "file_size_bytes": file_size,
            "input_dir": str(input_dir),
        }
        sidecar = output_dir / f"{cfg.input_dir.name}.pdf.json"
        sidecar.write_text(json.dumps(meta, indent=2))

        state.mark_image_done(self.checkpoint_name, "output")
        result.processed = len(image_files)
        if progress_callback is not None:
            progress_callback()

        logger.info(
            "PDF assembled: %d pages, %s compression, %.1f MB",
            len(image_files),
            compression,
            file_size / (1024 * 1024),
        )
        return result


def _collect_images(
    input_dir: Path,
    cfg: Config,
) -> list[Path]:
    """Collect and sort image files, excluding any from cfg.exclude_images."""
    exclude = set(cfg.exclude_images or [])

    files = sorted(
        p for p in input_dir.iterdir()
        if p.suffix.lower() in _IMAGE_EXTENSIONS
        and p.name not in exclude
        and p.stem not in exclude
    )
    return files


def _jpeg_generator(
    image_files: list[Path],
    quality: int,
) -> list[bytes]:
    """Convert each image to JPEG bytes in memory."""
    for path in image_files:
        img = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if img is None:
            logger.warning("Cannot read %s, skipping", path)
            continue
        ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])
        if not ok:
            logger.warning("JPEG encoding failed for %s, skipping", path)
            continue
        yield buf.tobytes()
