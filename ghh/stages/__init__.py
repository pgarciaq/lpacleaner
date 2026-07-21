"""Pipeline stages for image processing.

Provides a registry of all implemented stages, ordered by stage number.
Stages that are only stubs (empty module files) are excluded from the
registry until they are implemented.
"""

from __future__ import annotations

from ghh.pipeline import BaseStage
from ghh.stages.content_area import ContentAreaStage
from ghh.stages.deskew import DeskewStage
from ghh.stages.lens_correct import LensCorrectStage
from ghh.stages.omr import OmrStage
from ghh.stages.orientation import OrientationStage
from ghh.stages.page_detect import PageDetectStage
from ghh.stages.pdf_assembly import PDFAssemblyStage
from ghh.stages.perspective import PerspectiveStage
from ghh.stages.preprocess import PreprocessStage
from ghh.stages.score_render import ScoreRenderStage
from ghh.stages.staff_extract import StaffExtractStage
from ghh.stages.stitch import StitchStage

STAGE_CLASSES: list[type[BaseStage]] = [
    PreprocessStage,    # 0
    StitchStage,        # 1
    OrientationStage,   # 2
    LensCorrectStage,   # 3
    PageDetectStage,    # 4
    PerspectiveStage,   # 5
    ContentAreaStage,   # 6
    StaffExtractStage,  # 7
    DeskewStage,        # 8
    OmrStage,           # 13
    ScoreRenderStage,   # 14
    PDFAssemblyStage,   # 15
]

STAGE_BY_NUMBER: dict[int, type[BaseStage]] = {s.number: s for s in STAGE_CLASSES}
STAGE_BY_NAME: dict[str, type[BaseStage]] = {s.name: s for s in STAGE_CLASSES}

ALL_STAGE_NUMBERS: list[int] = sorted(STAGE_BY_NUMBER)

# Stage groupings for the forked pipeline architecture
COMMON_STAGE_NUMBERS = [0, 1, 2, 3, 4, 5]
BOOK_STAGE_NUMBERS = [8]  # only deskew implemented so far; will grow: 8, 9, 10, 11, 12
SCORE_STAGE_NUMBERS = [6, 7, 8, 13]  # content area, staff extract, deskew, omr
FINAL_STAGE_NUMBERS = [14, 15]  # score render + pdf assembly


def get_stages(numbers: list[int] | None = None) -> list[BaseStage]:
    """Return instantiated stages filtered by number, in pipeline order.

    If *numbers* is ``None``, returns all implemented stages.
    Raises ``ValueError`` for unknown stage numbers.
    """
    if numbers is None:
        numbers = ALL_STAGE_NUMBERS

    unknown = set(numbers) - set(STAGE_BY_NUMBER)
    if unknown:
        raise ValueError(
            f"Unknown stage number(s): {sorted(unknown)}. "
            f"Available: {ALL_STAGE_NUMBERS}"
        )

    return [STAGE_BY_NUMBER[n]() for n in sorted(numbers)]


def parse_stage_spec(spec: str) -> list[int]:
    """Parse a stage specifier string into a sorted list of stage numbers.

    Accepts comma-separated numbers and/or ranges::

        "0"       → [0]
        "0,2"     → [0, 2]
        "0-2"     → [0, 1, 2]
        "0,2-4,6" → [0, 2, 3, 4, 6]

    Raises ``click.BadParameter`` for invalid syntax.
    """
    import click

    result: set[int] = set()
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            bounds = part.split("-", 1)
            try:
                lo, hi = int(bounds[0]), int(bounds[1])
            except ValueError:
                raise click.BadParameter(
                    f"Invalid range {part!r} in stage spec {spec!r}"
                )
            if lo > hi:
                raise click.BadParameter(
                    f"Invalid range {part!r}: start > end"
                )
            result.update(range(lo, hi + 1))
        else:
            try:
                result.add(int(part))
            except ValueError:
                raise click.BadParameter(
                    f"Invalid stage number {part!r} in spec {spec!r}"
                )
    return sorted(result)
