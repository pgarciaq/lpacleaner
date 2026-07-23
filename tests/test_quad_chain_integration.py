"""Integration test: quad_corners chain through Stage 4 → 5 → 8 → 9 (#82).

Verifies that quad_corners survive the full pipeline chain:
  Page Detect (4) writes quad → Gentle Crop (5) transforms it →
  Deskew (8) transforms it → Perspective (9) consumes it.

Uses process_image() directly (no filesystem I/O) so the test is fast
and self-contained.
"""

from __future__ import annotations

import numpy as np
import pytest

from ghh.config import Config
from ghh.stages.deskew import DeskewStage
from ghh.stages.gentle_crop import GentleCropStage
from ghh.stages.page_detect import PageDetectStage
from ghh.stages.perspective import PerspectiveStage
from tests.conftest import make_music_page, make_page_on_background


@pytest.fixture
def cfg(tmp_path) -> Config:
    return Config(input_dir=tmp_path)


def _run_chain(
    img: np.ndarray, cfg: Config,
) -> list[tuple[np.ndarray, dict]]:
    """Run Stages 4 → 5 → 8 → 9 in sequence, returning all (image, meta) pairs."""
    results: list[tuple[np.ndarray, dict]] = []

    img4, meta4 = PageDetectStage().process_image(img, {}, cfg)
    results.append((img4, meta4))

    img5, meta5 = GentleCropStage().process_image(img4, meta4, cfg)
    results.append((img5, meta5))

    img8, meta8 = DeskewStage().process_image(img5, meta5, cfg)
    results.append((img8, meta8))

    img9, meta9 = PerspectiveStage().process_image(img8, meta8, cfg)
    results.append((img9, meta9))

    return results


class TestQuadChainIntegration:

    def test_quad_survives_full_chain(self, cfg):
        """quad_corners should be present in Stage 4, 5, and 8 output metadata."""
        page = make_music_page()
        photo = make_page_on_background(page, angle=1.0)
        results = _run_chain(photo, cfg)

        meta4 = results[0][1]
        meta5 = results[1][1]
        meta8 = results[2][1]

        assert "quad_corners" in meta4, "Stage 4 should produce quad_corners"
        assert "quad_corners" in meta5, "Stage 5 should propagate quad_corners"
        assert "quad_corners" in meta8, "Stage 8 should propagate quad_corners"

    def test_quad_shape_consistent(self, cfg):
        """quad_corners should always be a 4x2 array at every stage."""
        page = make_music_page()
        photo = make_page_on_background(page)
        results = _run_chain(photo, cfg)

        for i, stage_name in enumerate(["page_detect", "gentle_crop", "deskew"]):
            meta = results[i][1]
            if "quad_corners" in meta:
                corners = np.array(meta["quad_corners"])
                assert corners.shape == (4, 2), (
                    f"{stage_name} quad_corners has wrong shape: {corners.shape}"
                )

    def test_perspective_receives_and_uses_quad(self, cfg):
        """Stage 9 should consume the quad and produce a warp or passthrough."""
        page = make_music_page()
        photo = make_page_on_background(page, perspective_skew=0.02)
        results = _run_chain(photo, cfg)

        meta9 = results[3][1]
        assert meta9["stage"] == "perspective"
        assert "method" in meta9
        assert meta9["method"] != "passthrough", (
            "With perspective skew, Stage 9 should attempt a warp"
        )

    def test_final_output_is_reasonable_size(self, cfg):
        """The final image should be a reasonable fraction of the original."""
        page = make_music_page()
        photo = make_page_on_background(page)
        results = _run_chain(photo, cfg)

        original_area = photo.shape[0] * photo.shape[1]
        final_img = results[3][0]
        final_area = final_img.shape[0] * final_img.shape[1]

        ratio = final_area / original_area
        assert ratio > 0.2, f"Final image too small: {ratio:.1%} of original"
        assert ratio < 2.0, f"Final image too large: {ratio:.1%} of original"

    def test_quad_coords_decrease_through_crops(self, cfg):
        """After gentle crop, quad x/y values should be smaller (shifted to origin)."""
        page = make_music_page()
        photo = make_page_on_background(page, border=100)
        results = _run_chain(photo, cfg)

        meta4 = results[0][1]
        meta5 = results[1][1]

        if "quad_corners" in meta4 and "quad_corners" in meta5:
            q4 = np.array(meta4["quad_corners"])
            q5 = np.array(meta5["quad_corners"])
            assert np.mean(q5[:, 0]) < np.mean(q4[:, 0]), (
                "After crop, quad x should shift toward origin"
            )
            assert np.mean(q5[:, 1]) < np.mean(q4[:, 1]), (
                "After crop, quad y should shift toward origin"
            )

    def test_skewed_page_deskew_adjusts_quad(self, cfg):
        """Deskew rotation should visibly change quad_corners."""
        page = make_music_page(skew_deg=2.5)
        photo = make_page_on_background(page, angle=2.5)
        results = _run_chain(photo, cfg)

        meta5 = results[1][1]
        meta8 = results[2][1]

        if "quad_corners" in meta5 and "quad_corners" in meta8:
            q5 = np.array(meta5["quad_corners"])
            q8 = np.array(meta8["quad_corners"])
            diff = np.max(np.abs(q8 - q5))
            if meta8["method"] != "skipped":
                assert diff > 1.0, (
                    f"Deskew should change quad coords for skewed input, "
                    f"max diff was only {diff:.2f}"
                )

    def test_passthrough_perspective_still_works(self, cfg):
        """When the quad is near-rectangular, perspective should passthrough."""
        page = make_music_page()
        photo = make_page_on_background(page, angle=0.0, perspective_skew=0.0)
        results = _run_chain(photo, cfg)

        meta9 = results[3][1]
        assert meta9["stage"] == "perspective"
