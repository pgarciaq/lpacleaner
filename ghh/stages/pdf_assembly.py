"""Stage 15: PDF assembly.

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
``_find_previous_checkpoint`` walks backward from stage 15 and uses
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
    number = 15
    checkpoint_name = "15_pdf"
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
        max_workers: int = 1,
    ) -> StageResult:
        result = StageResult(stage_name=self.name)
        output_dir = Path(output_dir)
        input_dir = Path(input_dir)

        pdf_name = f"{cfg.input_dir.name}.pdf"
        pdf_path = output_dir / pdf_name

        if state.is_image_done(self.checkpoint_name, "output") and pdf_path.exists():
            result.skipped = 1
            return result

        # Two-source assembly: book pages + optional score annex
        book_images = _find_book_images(input_dir, output_dir, cfg)
        score_images = _find_score_annex_images(output_dir, cfg)

        image_files = book_images + score_images

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

        # Copy GABC source files alongside the PDF
        _copy_gabc_sources(output_dir, pdf_path)

        meta = {
            "stage": "pdf_assembly",
            "page_count": len(book_images),
            "score_annex_count": len(score_images),
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
            "PDF assembled: %d book pages + %d score pages, %s compression, %.1f MB",
            len(book_images),
            len(score_images),
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


def _find_book_images(
    input_dir: Path,
    output_dir: Path,
    cfg: Config,
) -> list[Path]:
    """Find book page images for the PDF.

    Prefers the book branch's latest checkpoint; falls back to the
    input_dir (which is the last common or flat-mode checkpoint).
    """
    book_dir = output_dir / "book"
    if book_dir.is_dir():
        from ghh.stages import STAGE_BY_NUMBER
        for n in sorted(STAGE_BY_NUMBER.keys(), reverse=True):
            if n >= 14:
                continue
            cls = STAGE_BY_NUMBER[n]
            candidate = book_dir / cls.checkpoint_name
            if candidate.is_dir() and any(
                p for p in candidate.iterdir()
                if p.suffix.lower() in _IMAGE_EXTENSIONS
            ):
                logger.info("PDF book pages from: %s", candidate)
                return _collect_images(candidate, cfg)

    if cfg.scores_only:
        return []

    return _collect_images(input_dir, cfg)


def _find_score_annex_images(
    output_dir: Path,
    cfg: Config,
) -> list[Path]:
    """Find rendered score images for the PDF annex."""
    if cfg.book_only:
        return []

    score_render_dir = output_dir / "14_score_render"
    if score_render_dir.is_dir():
        images = sorted(
            p for p in score_render_dir.iterdir()
            if p.suffix.lower() in _IMAGE_EXTENSIONS
        )
        if images:
            logger.info("PDF score annex: %d pages from %s", len(images), score_render_dir)
            return images

    return []


def _copy_gabc_sources(output_dir: Path, pdf_path: Path) -> None:
    """Copy GABC source files to a scores/ directory alongside the PDF."""
    import shutil

    omr_dir = output_dir / "score" / "13_omr"
    if not omr_dir.is_dir():
        return

    gabc_files = sorted(omr_dir.glob("*.gabc"))
    if not gabc_files:
        return

    scores_dir = output_dir / "scores"
    scores_dir.mkdir(exist_ok=True)
    for gabc in gabc_files:
        shutil.copy2(gabc, scores_dir / gabc.name)
    logger.info("Copied %d GABC files to %s", len(gabc_files), scores_dir)


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
