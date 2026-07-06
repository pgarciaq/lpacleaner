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
@click.option("--skip-dewarp", is_flag=True)
@click.option("--skip-deskew", is_flag=True)
@click.option("--skip-enhance", is_flag=True)
@click.option("--skip-normalize", is_flag=True)
@click.option("--skip-ocr", is_flag=True)
@click.option("--skip-content-area", is_flag=True)
@click.option("--ai-dewarp", is_flag=True)
@click.option("--binarize", is_flag=True)
@click.option("--cleanup", is_flag=True, help="Delete intermediate checkpoints after success")
@click.option("--on-error", type=click.Choice(["skip", "stop", "force"]), default="skip")
@click.option("-v", "--verbose", is_flag=True)
@click.option("-q", "--quiet", is_flag=True)
def run(input_dir, output_dir, config_path, stage_spec, profile, preview,
        skip_dewarp, skip_deskew, skip_enhance, skip_normalize, skip_ocr,
        skip_content_area, ai_dewarp, binarize, cleanup, on_error, verbose, quiet):
    """Process book page photos through the pipeline.

    Runs all implemented stages by default.  Use --stages to select specific
    stages (e.g. ``--stages 0-2`` for preprocess, stitch, orientation).
    """
    from tqdm import tqdm

    from ghh.config import Config
    from ghh.pipeline import BaseStage, PipelineState
    from ghh.stages import get_stages, parse_stage_spec

    _configure_logging(verbose, quiet)

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
        "skip_dewarp": skip_dewarp,
        "skip_deskew": skip_deskew,
        "skip_enhance": skip_enhance,
        "skip_normalize": skip_normalize,
        "skip_ocr": skip_ocr,
        "skip_content_area": skip_content_area,
        "ai_dewarp": ai_dewarp,
        "binarize": binarize,
        "on_error": on_error,
        "cleanup": cleanup,
        "verbose": verbose,
        "quiet": quiet,
    }
    cfg = Config.from_toml(input_path, toml_path=config_path, overrides=overrides)

    if stage_spec is not None:
        stage_numbers = parse_stage_spec(stage_spec)
    else:
        stage_numbers = None

    try:
        stages = get_stages(stage_numbers)
    except ValueError as exc:
        raise click.BadParameter(str(exc), param_hint="--stages") from exc

    state = PipelineState.load(output_path)

    total_processed = 0
    total_failed = 0

    for stage in stages:
        if stage.should_skip(cfg):
            click.echo(f"  Skipping stage {stage.number} ({stage.name}) [profile={profile}]")
            continue

        # Determine input directory: first stage reads from input_dir,
        # subsequent stages read from the previous stage's checkpoint.
        if stage.number == 0:
            stage_input = input_path
        else:
            prev = _find_previous_checkpoint(stage.number, stages, output_path)
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
            stage_input, output_path, cfg, state,
            progress_callback=bar.update,
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

    click.echo(f"Done. {total_processed} images processed, {total_failed} failures.")

    try:
        from ghh.compare import write_compare_html
        html_path = write_compare_html(output_path, input_path)
        click.echo(f"Comparison viewer: {html_path}")
    except Exception as exc:
        logger.debug("Could not generate comparison HTML: %s", exc)

    if total_failed and on_error == "stop":
        sys.exit(1)


def _find_previous_checkpoint(
    current_number: int,
    stages: list,
    output_path: Path,
) -> Path | None:
    """Find the checkpoint directory from the preceding stage.

    Walks backward from *current_number* looking for an existing checkpoint
    directory, regardless of whether that stage was in the current run.
    """
    from ghh.stages import STAGE_BY_NUMBER

    for n in range(current_number - 1, -1, -1):
        cls = STAGE_BY_NUMBER.get(n)
        if cls is None:
            continue
        candidate = output_path / cls.checkpoint_name
        if candidate.is_dir() and any(candidate.iterdir()):
            return candidate
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
              help="Comma-separated stage numbers to include (e.g. '0,5,7')")
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
