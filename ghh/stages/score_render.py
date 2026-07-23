"""Stage 14: Score Render.

Renders GABC files from the OMR stage (score/13_omr/) into typeset
notation images using Gregorio and LuaLaTeX.  If Gregorio is not
installed, the stage is skipped with a warning.
"""

from __future__ import annotations

import json
import logging
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np

from ghh.config import Config
from ghh.pipeline import BaseStage, PipelineState, StageResult
from ghh.utils.image_io import ensure_checkpoint_dir

logger = logging.getLogger(__name__)

_GABC_TEX_TEMPLATE = r"""\documentclass[preview]{standalone}
\usepackage{gregorio}
\begin{document}
\gregorioscore{__GABC_PATH__}
\end{document}
"""


def _gregorio_available() -> bool:
    """Check if lualatex (with gregorio) is installed."""
    try:
        subprocess.run(
            ["lualatex", "--version"],
            capture_output=True, timeout=10,
        )
        return True
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


class ScoreRenderStage(BaseStage):
    name = "score_render"
    number = 14
    checkpoint_name = "14_score_render"
    error_class = "skippable"
    config_keys = ("book_only",)
    writes_image = True

    def process_image(
        self,
        img: np.ndarray,
        metadata: dict,
        cfg: Config,
    ) -> tuple[np.ndarray, dict]:
        raise NotImplementedError("ScoreRenderStage uses custom run()")

    def should_skip(self, cfg: Config) -> bool:
        return cfg.book_only

    def run(
        self,
        input_dir: Path,
        output_dir: Path,
        cfg: Config,
        state: PipelineState,
        progress_callback: callable | None = None,
        max_workers: int = 1,
    ) -> StageResult:
        """Render GABC files from the OMR stage into notation images."""
        result = StageResult(stage_name=self.name)
        stage_dir = ensure_checkpoint_dir(output_dir, self.checkpoint_name)

        omr_dir = output_dir / "score" / "13_omr"
        if not omr_dir.is_dir():
            logger.info("No OMR output at %s, skipping score render", omr_dir)
            return result

        gabc_files = sorted(omr_dir.glob("*.gabc"))
        if not gabc_files:
            logger.info("No GABC files found in %s", omr_dir)
            return result

        if not _gregorio_available():
            logger.warning(
                "LuaLaTeX/Gregorio not installed; skipping score render. "
                "Install texlive-luatex and gregorio to enable."
            )
            meta = {"score_render_action": "skipped", "reason": "gregorio_not_installed"}
            meta_path = stage_dir / "_render_skipped.json"
            meta_path.write_text(json.dumps(meta, indent=2))
            return result

        for gabc_path in gabc_files:
            stem = gabc_path.stem

            if state.is_image_done(self.checkpoint_name, stem):
                out_path = stage_dir / f"{stem}.png"
                if out_path.exists():
                    result.skipped += 1
                    if progress_callback:
                        progress_callback()
                    continue

            try:
                rendered = self._render_gabc(gabc_path, stage_dir, stem)
                if rendered:
                    state.mark_image_done(self.checkpoint_name, stem)
                    result.processed += 1
                else:
                    result.failed += 1
            except Exception as exc:
                logger.error("Score render failed for %s: %s", stem, exc)
                result.failed += 1

            if progress_callback:
                progress_callback()

        return result

    def _render_gabc(self, gabc_path: Path, stage_dir: Path, stem: str) -> bool:
        """Render a single GABC file to a PNG image."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)

            gabc_copy = tmp / f"{stem}.gabc"
            shutil.copy2(gabc_path, gabc_copy)

            tex_content = _GABC_TEX_TEMPLATE.replace("__GABC_PATH__", str(gabc_copy))
            tex_path = tmp / f"{stem}.tex"
            tex_path.write_text(tex_content)

            subprocess.run(
                ["lualatex", "--interaction=nonstopmode", str(tex_path)],
                cwd=tmp,
                capture_output=True,
                timeout=60,
            )

            pdf_path = tmp / f"{stem}.pdf"
            if not pdf_path.exists():
                logger.warning("LuaLaTeX did not produce PDF for %s", stem)
                return False

            out_png = stage_dir / f"{stem}.png"
            subprocess.run(
                [
                    "gs", "-dBATCH", "-dNOPAUSE", "-sDEVICE=png16m",
                    "-r300", f"-sOutputFile={out_png}", str(pdf_path),
                ],
                capture_output=True,
                timeout=30,
            )

            if out_png.exists():
                meta = {
                    "score_render_action": "rendered",
                    "source_gabc": str(gabc_path),
                }
                meta_path = stage_dir / f"{stem}.json"
                meta_path.write_text(json.dumps(meta, indent=2))
                return True

            logger.warning("Ghostscript did not produce PNG for %s", stem)
            return False
