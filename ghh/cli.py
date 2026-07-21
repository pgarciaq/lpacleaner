"""Click CLI: analyze, run, inspect, review, cleanup commands."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import click

from ghh import __version__

logger = logging.getLogger(__name__)


@click.group()
@click.version_option(version=__version__)
def main():
    """Guido's Helping Hand -- process photographed book pages into searchable PDFs."""


@main.command()
def stages():
    """List all implemented pipeline stages."""
    from ghh.stages import STAGE_CLASSES

    click.echo("Implemented pipeline stages:\n")
    click.echo(f"  {'#':>3}  {'Name':<16}  {'Checkpoint dir'}")
    click.echo(f"  {'---':>3}  {'----':<16}  {'--------------'}")
    for cls in sorted(STAGE_CLASSES, key=lambda c: c.number):
        click.echo(
            f"  {cls.number:>3}  {cls.name:<16}  {cls.checkpoint_name}"
        )
    click.echo(f"\n{len(STAGE_CLASSES)} stages. "
               f"Use --stages with 'ghh run' to select specific stages.")


@main.command()
@click.argument("input_dir", type=click.Path(exists=True, file_okay=False))
@click.option("-o", "--output-dir", type=click.Path(file_okay=False), default=None,
              help="Output directory (default: <input_dir>_output)")
@click.option("--config", "config_path", type=click.Path(exists=True, dir_okay=False),
              default=None, help="Path to book.toml configuration file")
@click.option("--stages", "stage_spec", type=str, default=None,
              help="Stages to run: comma-separated numbers or ranges (e.g. 0,1,2 or 0-2)")
@click.option("--profile", type=click.Choice(["full", "geometry", "clean", "quick"]),
              default="full")
@click.option("--preview", type=int, default=0, help="Process only N images")
@click.option("--book-only", is_flag=True, help="Run only the book branch (skip score branch)")
@click.option("--scores-only", is_flag=True, help="Run only the score branch (skip book branch)")
@click.option("--skip-dewarp", is_flag=True)
@click.option("--skip-deskew", is_flag=True)
@click.option("--skip-enhance", is_flag=True)
@click.option("--skip-normalize", is_flag=True)
@click.option("--skip-ocr", is_flag=True)
@click.option("--skip-omr", is_flag=True)
@click.option("--skip-content-area", is_flag=True)
@click.option("--model-dir", default=None, help="Path to chant-omr OpenVINO model directory")
@click.option("--ai-dewarp", is_flag=True)
@click.option("--binarize", is_flag=True)
@click.option("--cleanup", is_flag=True, help="Delete intermediate checkpoints after success")
@click.option("--on-error", type=click.Choice(["skip", "stop", "force"]), default="skip")
@click.option("-j", "--jobs", type=int, default=None,
              help="Parallel workers per stage (default: half of CPU cores, 1=sequential)")
@click.option("-v", "--verbose", is_flag=True)
@click.option("-q", "--quiet", is_flag=True)
def run(input_dir, output_dir, config_path, stage_spec, profile, preview,
        book_only, scores_only,
        skip_dewarp, skip_deskew, skip_enhance, skip_normalize, skip_ocr,
        skip_omr, skip_content_area, model_dir, ai_dewarp, binarize,
        cleanup, on_error, jobs, verbose, quiet):
    """Process book page photos through the pipeline.

    The pipeline has three phases: common preparation (stages 0-5),
    two parallel branches (book and score), and finalization (stages 14-15).

    Use --book-only or --scores-only to run a single branch.
    Use --stages to select specific stages (e.g. ``--stages 0-2``).
    """
    import os

    from ghh.config import Config
    from ghh.pipeline import PipelineState
    from ghh.stages import get_stages, parse_stage_spec

    _configure_logging(verbose, quiet)

    if book_only and scores_only:
        raise click.UsageError("Cannot use both --book-only and --scores-only")

    if jobs is None:
        jobs = max(1, (os.cpu_count() or 2) // 2)
    jobs = max(1, jobs)

    input_path = Path(input_dir)
    if output_dir is None:
        output_path = input_path.parent / f"{input_path.name}_output"
    else:
        output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    overrides = {
        "output_dir": output_path,
        "profile": profile,
        "preview": preview,
        "book_only": book_only,
        "scores_only": scores_only,
        "skip_dewarp": skip_dewarp,
        "skip_deskew": skip_deskew,
        "skip_enhance": skip_enhance,
        "skip_normalize": skip_normalize,
        "skip_ocr": skip_ocr,
        "skip_omr": skip_omr,
        "skip_content_area": skip_content_area,
        "ai_dewarp": ai_dewarp,
        "binarize": binarize,
        "on_error": on_error,
        "cleanup": cleanup,
        "verbose": verbose,
        "quiet": quiet,
    }
    if model_dir is not None:
        overrides["omr_model_dir"] = model_dir
    cfg = Config.from_toml(input_path, toml_path=config_path, overrides=overrides)

    # If user specified explicit stages, use flat (non-branched) execution
    if stage_spec is not None:
        stage_numbers = parse_stage_spec(stage_spec)
        try:
            selected = get_stages(stage_numbers)
        except ValueError as exc:
            raise click.BadParameter(str(exc), param_hint="--stages") from exc
        state = PipelineState.load(output_path)
        total_p, total_f = _run_stage_list(
            selected, input_path, output_path, cfg, state, jobs, quiet,
        )
    else:
        state = PipelineState.load(output_path)
        total_p, total_f = _run_forked_pipeline(
            input_path, output_path, cfg, state, jobs, quiet,
            book_only=book_only, scores_only=scores_only,
        )

    click.echo(f"Done. {total_p} images processed, {total_f} failures.")

    try:
        from ghh.compare import write_compare_html
        html_path = write_compare_html(output_path, input_path)
        click.echo(f"Comparison viewer: {html_path}")
    except Exception as exc:
        logger.debug("Could not generate comparison HTML: %s", exc)

    if total_f and on_error == "stop":
        sys.exit(1)


def _run_stage_list(
    stages: list,
    input_path: Path,
    output_path: Path,
    cfg,
    state,
    jobs: int,
    quiet: bool,
    branch_dir: Path | None = None,
) -> tuple[int, int]:
    """Run a flat list of stages sequentially (legacy / explicit --stages mode).

    When *branch_dir* is set, checkpoint directories are created under
    that subdirectory (e.g. ``output/book/``) and previous-checkpoint
    lookup starts there before falling back to the common output root.
    """
    import click
    from tqdm import tqdm

    from ghh.pipeline import BaseStage

    effective_output = branch_dir if branch_dir else output_path
    total_processed = 0
    total_failed = 0

    for stage in stages:
        if stage.should_skip(cfg):
            click.echo(f"  Skipping stage {stage.number} ({stage.name})")
            continue

        if stage.number == 0:
            stage_input = input_path
        else:
            prev = _find_previous_checkpoint(
                stage.number, stages, effective_output,
                fallback_dir=output_path if branch_dir else None,
            )
            if prev is None:
                click.echo(
                    f"  Stage {stage.number} ({stage.name}): "
                    f"no input checkpoint found, skipping",
                    err=True,
                )
                continue
            stage_input = prev

        n_images = BaseStage.count_images(stage_input)
        desc = f"Stage {stage.number} {stage.name}"
        bar = tqdm(
            total=n_images, desc=desc, unit="img",
            disable=quiet, leave=True,
        )
        result = stage.run(
            stage_input, effective_output, cfg, state,
            progress_callback=bar.update,
            max_workers=jobs,
        )
        bar.close()
        state.record_result(result)
        state.save()

        total_processed += result.processed
        total_failed += result.failed

        click.echo(
            f"    processed={result.processed} skipped={result.skipped} "
            f"failed={result.failed} excluded={result.excluded}"
        )

    return total_processed, total_failed


def _run_forked_pipeline(
    input_path: Path,
    output_path: Path,
    cfg,
    state,
    jobs: int,
    quiet: bool,
    book_only: bool = False,
    scores_only: bool = False,
) -> tuple[int, int]:
    """Execute the three-phase forked pipeline."""
    import click

    from ghh.stages import (
        BOOK_STAGE_NUMBERS,
        COMMON_STAGE_NUMBERS,
        FINAL_STAGE_NUMBERS,
        SCORE_STAGE_NUMBERS,
        get_stages,
    )

    total_p = 0
    total_f = 0

    # Phase 1: Common stages (0-5)
    common_stages = get_stages(
        [n for n in COMMON_STAGE_NUMBERS if n in _implemented_numbers()]
    )
    if common_stages:
        click.echo("─── Common preparation ───")
        p, f = _run_stage_list(
            common_stages, input_path, output_path, cfg, state, jobs, quiet,
        )
        total_p += p
        total_f += f

    # Phase 2: Branches
    if not scores_only:
        book_nums = [n for n in BOOK_STAGE_NUMBERS if n in _implemented_numbers()]
        if book_nums:
            book_stages = get_stages(book_nums)
            book_dir = output_path / "book"
            book_dir.mkdir(parents=True, exist_ok=True)
            cfg_book = cfg.for_branch("book")
            click.echo("─── Book branch (full page) ───")
            p, f = _run_stage_list(
                book_stages, input_path, output_path, cfg_book, state, jobs, quiet,
                branch_dir=book_dir,
            )
            total_p += p
            total_f += f

    if not book_only:
        score_nums = [n for n in SCORE_STAGE_NUMBERS if n in _implemented_numbers()]
        if score_nums:
            score_stages = get_stages(score_nums)
            score_dir = output_path / "score"
            score_dir.mkdir(parents=True, exist_ok=True)
            cfg_score = cfg.for_branch("score")
            click.echo("─── Score branch (content area) ───")
            p, f = _run_stage_list(
                score_stages, input_path, output_path, cfg_score, state, jobs, quiet,
                branch_dir=score_dir,
            )
            total_p += p
            total_f += f

    # Phase 3: Finalization
    final_nums = [n for n in FINAL_STAGE_NUMBERS if n in _implemented_numbers()]
    if final_nums:
        final_stages = get_stages(final_nums)
        click.echo("─── Finalization ───")
        p, f = _run_stage_list(
            final_stages, input_path, output_path, cfg, state, jobs, quiet,
        )
        total_p += p
        total_f += f

    return total_p, total_f


def _implemented_numbers() -> set[int]:
    """Return the set of currently implemented stage numbers."""
    from ghh.stages import STAGE_BY_NUMBER
    return set(STAGE_BY_NUMBER.keys())


def _find_previous_checkpoint(
    current_number: int,
    stages: list,
    output_path: Path,
    fallback_dir: Path | None = None,
) -> Path | None:
    """Find the checkpoint directory from the preceding stage.

    Walks backward from *current_number* looking for an existing checkpoint
    directory. When *fallback_dir* is set (branch execution), also checks
    the common output root for cross-phase continuity (e.g. score branch
    stage 6 reads from common ``05_perspective/``).
    """
    from ghh.stages import STAGE_BY_NUMBER

    for n in range(current_number - 1, -1, -1):
        cls = STAGE_BY_NUMBER.get(n)
        if cls is None:
            continue
        # Check primary directory first
        candidate = output_path / cls.checkpoint_name
        if candidate.is_dir() and any(candidate.iterdir()):
            return candidate
        # Check fallback (common output root) for branch entry points
        if fallback_dir is not None:
            fallback_candidate = fallback_dir / cls.checkpoint_name
            if fallback_candidate.is_dir() and any(fallback_candidate.iterdir()):
                return fallback_candidate
    return None


def _configure_logging(verbose: bool, quiet: bool) -> None:
    if verbose:
        level = logging.DEBUG
    elif quiet:
        level = logging.WARNING
    else:
        level = logging.INFO
    logging.basicConfig(
        level=level,
        format="%(levelname)s:%(name)s:%(message)s",
        force=True,
    )


@main.command()
@click.argument("input_dir", type=click.Path(exists=True, file_okay=False))
@click.option("-o", "--output-dir", type=click.Path(file_okay=False), default=None)
@click.option("--samples", type=int, default=15)
def analyze(input_dir, output_dir, samples):
    """Analyze book photos and generate book.toml configuration."""
    click.echo(f"Analyzing {input_dir}...")


@main.command()
@click.argument("image_path", type=click.Path(exists=True, dir_okay=False))
@click.option("--config", type=click.Path(exists=True, dir_okay=False), default=None)
def inspect(image_path, config):
    """Inspect a single image with diagnostic output."""
    click.echo(f"Inspecting {image_path}...")


@main.command()
@click.argument("output_dir", type=click.Path(exists=True, file_okay=False))
@click.option("--stage", type=str, default=None)
def review(output_dir, stage):
    """Review processed pages and generate contact sheet."""
    click.echo(f"Reviewing {output_dir}...")


@main.command()
@click.argument("output_dir", type=click.Path(exists=True, file_okay=False))
@click.argument("image_stem", type=str, required=False, default=None)
@click.option("--input-dir", type=click.Path(exists=True, file_okay=False), default=None,
              help="Original input directory (auto-detected if <input>_output convention)")
@click.option("--no-open", is_flag=True, help="Don't open the browser automatically")
def compare(output_dir, image_stem, input_dir, no_open):
    """Compare all pipeline stages locally in the browser.

    Generates an interactive HTML viewer with all images and all
    checkpoint stages, using file:// references to existing PNGs on
    disk (no copies, no conversion).

    Use PgUp/PgDn to navigate between images, Left/Right arrows to
    switch stages, and S for side-by-side mode.

    If IMAGE_STEM is provided (e.g. IMG_0012), the viewer opens at
    that image.  Otherwise it starts at the first image.
    """
    from ghh.compare import discover_book, generate_compare_html, infer_input_dir

    output_path = Path(output_dir)

    if input_dir is not None:
        input_path = Path(input_dir)
    else:
        input_path = infer_input_dir(output_path)

    book = discover_book(output_path, input_path)

    if not book["images"]:
        click.echo(f"No images found in {output_dir}", err=True)
        sys.exit(1)

    stem = Path(image_stem).stem if image_stem else None

    click.echo(
        f"Found {len(book['images'])} images across "
        f"{len(book['stages'])} stages"
    )

    html = generate_compare_html(output_path, input_path, stem)
    html_path = output_path / "compare.html"
    html_path.write_text(html)
    click.echo(f"Wrote {html_path}")

    if not no_open:
        import webbrowser
        webbrowser.open(f"file://{html_path.resolve()}")


@main.command()
@click.argument("output_dir", type=click.Path(exists=True, file_okay=False))
@click.argument("publish_dir", type=click.Path(file_okay=False))
@click.option("--input-dir", type=click.Path(exists=True, file_okay=False), default=None,
              help="Original input directory (auto-detected if <input>_output convention)")
@click.option("--max-dim", type=int, default=1500,
              help="Max pixel dimension for JPEGs (default: 1500)")
@click.option("--quality", type=int, default=85,
              help="JPEG compression quality (default: 85)")
@click.option("--stages", "stage_spec", type=str, default=None,
              help="Comma-separated stage numbers to include (e.g. '0,5,8')")
@click.option("--with-flipbook", is_flag=True, help="Include flipbook viewer")
@click.option("--with-pdf", is_flag=True,
              help="Include PDF download in flipbook (implies --with-flipbook)")
@click.option("--no-open", is_flag=True, help="Don't open the browser automatically")
def publish(output_dir, publish_dir, input_dir, max_dim, quality, stage_spec,
            with_flipbook, with_pdf, no_open):
    """Publish a web-friendly comparison site with downscaled JPEGs.

    Converts all pipeline stage images to downscaled JPEGs and writes
    them to PUBLISH_DIR alongside an index.html viewer.  The result
    is a self-contained directory ready for static web hosting
    (GitHub Pages, Netlify, S3, etc.).

    Use --with-flipbook to also generate a flipbook viewer in a
    flipbook/ subdirectory.  --with-pdf implies --with-flipbook and
    adds a PDF download link.
    """
    from ghh.compare import infer_input_dir, publish_book

    output_path = Path(output_dir)

    if input_dir is not None:
        input_path = Path(input_dir)
    else:
        input_path = infer_input_dir(output_path)

    stage_filter = None
    if stage_spec:
        stage_filter = {s.strip().zfill(2) for s in stage_spec.split(",")}
        stage_filter.add("orig")

    if with_pdf:
        with_flipbook = True

    extra_links = ""
    pdf_filename: str | None = None

    if with_flipbook:
        from ghh.flipbook import find_pdf, generate_flipbook

        fb_dir = Path(publish_dir) / "flipbook"
        click.echo("Generating flipbook...")
        try:
            fb_index = generate_flipbook(
                output_path,
                fb_dir,
                max_width=1600,
                jpeg_quality=quality,
                include_pdf=with_pdf,
            )
            click.echo(f"Flipbook added: {fb_index.parent}")
            extra_links += (
                '<a href="flipbook/index.html" class="extra-link" '
                'target="_blank">&#128214; Flipbook</a>'
            )
        except FileNotFoundError as exc:
            click.echo(f"Warning: {exc}", err=True)

        if with_pdf:
            pdf_src = find_pdf(output_path)
            if pdf_src is not None:
                pdf_filename = pdf_src.name
                extra_links += (
                    f'<a href="flipbook/{pdf_filename}" download '
                    f'class="extra-link">&#128196; Download PDF</a>'
                )

    click.echo("Publishing comparison site with downscaled JPEGs...")
    html_path = publish_book(
        output_path,
        Path(publish_dir),
        input_dir=input_path,
        max_dim=max_dim,
        quality=quality,
        stage_filter=stage_filter,
        extra_links=extra_links,
    )
    click.echo(f"Published to {html_path.parent}")

    if not no_open:
        import webbrowser
        webbrowser.open(f"file://{html_path.resolve()}")


@main.command(name="cleanup")
@click.argument("output_dir", type=click.Path(exists=True, file_okay=False))
@click.option("--keep", type=str, default=None, help="Comma-separated stage numbers to keep")
def cleanup_cmd(output_dir, keep):
    """Delete intermediate checkpoint directories."""
    click.echo(f"Cleaning up {output_dir}...")


@main.command()
@click.argument("output_dir", type=click.Path(exists=True, file_okay=False))
@click.argument("flipbook_dir", type=click.Path(file_okay=False), required=False, default=None)
@click.option("--input-dir", type=click.Path(exists=True, file_okay=False), default=None,
              help="Original input directory (unused, kept for consistency)")
@click.option("--max-width", type=int, default=1600,
              help="Max page width in pixels (default: 1600)")
@click.option("--quality", type=int, default=85,
              help="JPEG compression quality (default: 85)")
@click.option("--title", type=str, default="",
              help="Title displayed in the flipbook viewer")
@click.option("--no-pdf", is_flag=True, help="Omit PDF download link")
@click.option("--cover", is_flag=True,
              help="Treat the first page as a standalone cover (displayed alone before flipping)")
@click.option("--no-open", is_flag=True, help="Don't open the browser automatically")
def flipbook(output_dir, flipbook_dir, input_dir, max_width, quality, title, no_pdf, cover,
             no_open):
    """Generate a static HTML flipbook from processed pages.

    Reads images from the latest completed pipeline checkpoint in
    OUTPUT_DIR and generates a self-contained flipbook with page-turning
    animation.  The result can be uploaded to any static hosting.

    If FLIPBOOK_DIR is not specified, defaults to OUTPUT_DIR/flipbook/.

    By default, page 1 starts on the left (interior page layout).  Use
    --cover if the first image is a book cover that should be displayed
    alone on the right before flipping.
    """
    from ghh.flipbook import generate_flipbook

    output_path = Path(output_dir)
    fb_path = Path(flipbook_dir) if flipbook_dir else None

    click.echo("Generating flipbook...")
    try:
        index_path = generate_flipbook(
            output_path,
            fb_path,
            max_width=max_width,
            jpeg_quality=quality,
            title=title,
            include_pdf=not no_pdf,
            show_cover=cover,
        )
    except FileNotFoundError as exc:
        click.echo(str(exc), err=True)
        sys.exit(1)

    click.echo(f"Flipbook generated: {index_path.parent}")

    if not no_open:
        import webbrowser
        webbrowser.open(f"file://{index_path.resolve()}")
