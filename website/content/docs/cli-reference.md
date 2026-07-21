---
title: "CLI Reference"
weight: 20
description: "Complete reference for all ghh commands and options"
---

Guido's Helping Hand provides a set of commands for processing, inspecting,
comparing, and publishing book page images. The main entry point is `ghh`.

```bash
ghh --version        # show version
ghh --help           # list all commands
ghh COMMAND --help   # help for a specific command
```

## ghh run

Process book page photos through the pipeline.

```bash
ghh run INPUT_DIR [OPTIONS]
```

This is the primary command. It runs all implemented stages by default,
producing a clean PDF from raw photographs.

### Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `-o`, `--output-dir` | path | `INPUT_DIR_output` | Output directory |
| `--config` | path | auto-detected | Path to `book.toml` configuration file |
| `--stages` | string | all | Stages to run (e.g. `0,1,2` or `0-2,5,7`) |
| `--profile` | choice | `full` | Processing profile: `full`, `geometry`, `clean`, `quick` |
| `--preview` | integer | 0 | Process only N images (0 = all) |
| `--skip-content-area` | flag | | Skip content area detection (Stage 6) |
| `--skip-dewarp` | flag | | Skip dewarping (Stage 8) |
| `--skip-deskew` | flag | | Skip deskewing (Stage 8) |
| `--skip-enhance` | flag | | Skip image enhancement (Stage 9) |
| `--skip-normalize` | flag | | Skip cross-page normalization (Stage 10) |
| `--skip-ocr` | flag | | Skip OCR (Stage 11) |
| `--skip-omr` | flag | | Skip OMR (Stage 13) |
| `--model-dir` | path | from config | Path to chant-omr OpenVINO model directory |
| `--ai-dewarp` | flag | | Use AI-based dewarping |
| `--binarize` | flag | | Binarize output images |
| `--cleanup` | flag | | Delete intermediate checkpoints after success |
| `--on-error` | choice | `skip` | Error handling: `skip`, `stop`, `force` |
| `-v`, `--verbose` | flag | | Verbose logging (DEBUG level) |
| `-q`, `--quiet` | flag | | Quiet logging (WARNING level only) |

### Examples

```bash
# Process all photos with default settings
ghh run ~/photos/LPA-1

# Process first 10 images with geometry-only profile
ghh run ~/photos/LPA-1 --preview 10 --profile geometry

# Run specific stages
ghh run ~/photos/LPA-1 --stages 0-5,12

# Use a custom config file
ghh run ~/photos/LPA-1 --config ~/photos/LPA-1/book.toml

# Skip OCR and clean up intermediates
ghh run ~/photos/LPA-1 --skip-ocr --cleanup

# Run with OMR (requires chant-omr and exported model)
ghh run ~/photos/LPA-1 --model-dir ~/models/chant-omr
```

## ghh analyze

Analyze book photos and generate a `book.toml` configuration file.

```bash
ghh analyze INPUT_DIR [OPTIONS]
```

Samples a subset of images to auto-detect ink colors, page layout,
photography conditions, and physical condition. Writes the results
to `book.toml` in the input directory.

### Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `-o`, `--output-dir` | path | input dir | Where to write `book.toml` |
| `--samples` | integer | 15 | Number of sample images to analyze |

### Examples

```bash
# Analyze and generate config
ghh analyze ~/photos/LPA-1

# Analyze with more samples for better accuracy
ghh analyze ~/photos/LPA-1 --samples 30
```

## ghh compare

Open an interactive HTML comparison viewer in your browser.

```bash
ghh compare OUTPUT_DIR [IMAGE_STEM] [OPTIONS]
```

Displays all pipeline stages side-by-side for visual inspection.
Keyboard navigation lets you quickly flip between images and stages.

### Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `IMAGE_STEM` | argument | first image | Open at this image (e.g. `IMG_0012`) |
| `--input-dir` | path | auto-detected | Original input directory |
| `--no-open` | flag | | Don't open the browser automatically |

### Keyboard shortcuts

| Key | Action |
|-----|--------|
| **Up** / **Down** | Previous / next image (reset to stage 0) |
| **PgUp** / **PgDn** | Previous / next image (keep current stage) |
| **Left** / **Right** | Previous / next stage |
| **S** | Toggle side-by-side mode |
| **M** | Toggle metadata panel |
| **Z** | Toggle zoom |
| **1**--**9** | Jump to stage by number |

In side-by-side mode, each pane can show a different image:

| Key | Action |
|-----|--------|
| **Up** / **Down** | Left pane: previous / next image (reset to stage 0) |
| **Shift+Up** / **Shift+Down** | Right pane: previous / next image (reset to stage 0) |
| **PgUp** / **PgDn** | Left pane: previous / next image (keep stage) |
| **Shift+PgUp** / **Shift+PgDn** | Right pane: previous / next image (keep stage) |
| **Left** / **Right** | Change left pane stage |
| **Shift+Left** / **Shift+Right** | Change right pane stage |
| **1**--**9** | Set left pane stage |
| **Shift+1**--**Shift+9** | Set right pane stage |

### Examples

```bash
# Compare all stages
ghh compare ~/photos/LPA-1_output

# Open at a specific image
ghh compare ~/photos/LPA-1_output IMG_0042
```

## ghh publish

Generate a self-contained, web-publishable comparison site.

```bash
ghh publish OUTPUT_DIR PUBLISH_DIR [OPTIONS]
```

Creates a directory with downscaled JPEG images and an HTML viewer
suitable for uploading to any static web host.

### Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--input-dir` | path | auto-detected | Original input directory |
| `--max-dim` | integer | 1500 | Max pixel dimension for JPEGs |
| `--quality` | integer | 85 | JPEG compression quality (1--100) |
| `--stages` | string | all | Comma-separated stage numbers to include |
| `--with-flipbook` | flag | | Include the flipbook viewer |
| `--with-pdf` | flag | | Include PDF download (implies `--with-flipbook`) |
| `--no-open` | flag | | Don't open the browser automatically |

### Examples

```bash
# Publish with flipbook and PDF
ghh publish ~/photos/LPA-1_output ~/publish/LPA-1 \
    --with-flipbook --with-pdf

# Publish only specific stages, smaller images
ghh publish ~/photos/LPA-1_output ~/publish/LPA-1 \
    --stages 0,5,7 --max-dim 1000 --quality 75
```

## ghh flipbook

Generate a static HTML flipbook with page-turning animation.

```bash
ghh flipbook OUTPUT_DIR [FLIPBOOK_DIR] [OPTIONS]
```

Creates a standalone HTML page using the StPageFlip library.
By default includes a PDF download link.

### Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `FLIPBOOK_DIR` | argument | `OUTPUT_DIR/flipbook/` | Output directory |
| `--max-width` | integer | 1600 | Max page width in pixels |
| `--quality` | integer | 85 | JPEG compression quality |
| `--title` | string | directory name | Title displayed in the viewer |
| `--no-pdf` | flag | | Omit the PDF download link |
| `--cover` | flag | | Treat first page as a standalone cover |
| `--no-open` | flag | | Don't open the browser automatically |

### Examples

```bash
# Generate flipbook with defaults
ghh flipbook ~/photos/LPA-1_output

# Generate with custom title and smaller images
ghh flipbook ~/photos/LPA-1_output --title "LPA-1 Antiphonary" \
    --max-width 1200 --quality 80

# Generate without PDF link, treating first page as cover
ghh flipbook ~/photos/LPA-1_output --no-pdf --cover
```

## ghh inspect

Inspect a single image with diagnostic output.

```bash
ghh inspect IMAGE_PATH [OPTIONS]
```

### Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--config` | path | none | Path to a configuration file |

## ghh review

Review processed pages and generate a contact sheet.

```bash
ghh review OUTPUT_DIR [OPTIONS]
```

### Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--stage` | string | none | Specific stage directory to review |

## ghh cleanup

Delete intermediate checkpoint directories to save disk space.

```bash
ghh cleanup OUTPUT_DIR [OPTIONS]
```

### Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--keep` | string | none | Comma-separated stage numbers to keep |

### Examples

```bash
# Remove all intermediates, keep only the final PDF
ghh cleanup ~/photos/LPA-1_output

# Keep stages 0 (original preprocessing) and 7 (deskewed)
ghh cleanup ~/photos/LPA-1_output --keep 0,7
```

## ghh stages

List all implemented pipeline stages.

```bash
ghh stages
```

Displays a table with stage number, name, and checkpoint directory name.
No options.
