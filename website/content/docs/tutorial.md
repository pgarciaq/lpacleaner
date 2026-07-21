---
title: "Tutorial"
weight: 10
description: "Step-by-step guide to processing your first book"
---

This tutorial walks you through processing a set of smartphone photographs
of a historical music manuscript into a clean, searchable PDF. No prior
experience with image processing is needed.

## What Guido's Helping Hand does

You photograph the pages of a historical book -- an antiphonary, gradual,
or any bound manuscript -- with your smartphone or camera. The photos may
be crooked, have uneven lighting, show your fingers at the edges, or
include the table beneath the book. Guido's Helping Hand takes these raw
photographs and automatically:

- Detects and crops each page from its background
- Corrects perspective distortion from the camera angle
- Straightens skewed pages
- Produces a clean, consistently sized PDF

The entire process is automated and requires no manual configuration for
most books.

## Installation

### Prerequisites

- **Python 3.11 or later** (3.13 recommended)
- **Tesseract OCR** (optional, for searchable text)

### Install ghh

```bash
pip install .
```

Or with optional extras:

```bash
pip install ".[historical-ocr]"   # Kraken OCR for historical scripts
pip install ".[dev]"              # pytest, ruff (for contributors)
```

### Install system dependencies

Tesseract enables the OCR stage, which adds a searchable text layer to the
final PDF. Ghostscript is used by the PDF assembly stage. If Tesseract is
not installed, ghh skips OCR gracefully.

**Fedora / RHEL:**

```bash
sudo dnf install tesseract tesseract-langpack-lat ghostscript
```

**Ubuntu / Debian:**

```bash
sudo apt install tesseract-ocr tesseract-ocr-lat ghostscript
```

**macOS (Homebrew):**

```bash
brew install python@3.13 tesseract tesseract-lang ghostscript
```

ghh works on macOS (Intel and Apple Silicon) with no code changes.
OpenVINO's `--ai-dewarp` runs on CPU on macOS.

For the full dependency matrix, see the
[DEPENDENCIES.md](https://github.com/pgarciaq/ghh/blob/master/DEPENDENCIES.md)
file in the repository.

## Preparing your photos

Good input produces the best output. Here are some tips for photographing
book pages:

1. **One page per photo** -- photograph each page individually. If a page
   is too large, take overlapping photos (ghh will stitch them).
2. **Include some background** -- don't crop too tightly. ghh needs to see
   the page edges to detect and correct perspective.
3. **Consistent lighting** -- avoid harsh shadows across the page. Diffused
   natural light works well.
4. **Any orientation is fine** -- ghh automatically detects and corrects
   rotation, even upside-down pages.
5. **Don't worry about your fingers** -- if you need to hold the page
   down, ghh detects and handles finger intrusions at the edges.

## Processing your first book

Place all your photographs in a single directory. The filename order should
match the page order (most cameras do this automatically).

```bash
ghh run /path/to/photos
```

That's it. ghh will:

1. Analyze the images and detect book characteristics
2. Run each processing stage in sequence
3. Produce a PDF in the output directory

### Output directory

By default, ghh creates an output directory next to your input:

```
/path/to/photos/           # your original images
/path/to/photos_output/    # ghh output
  00_preprocessed/         # after hotspot/finger removal
  01_stitched/             # after grouping and stitching
  02_oriented/             # after rotation correction
  03_lens_corrected/       # after lens distortion fix
  04_page_detected/        # after page detection and crop
  05_perspective/          # after perspective correction
  06_content_area/         # after content area extraction
  07_deskewed/             # after straightening
  12_pdf/                  # final PDF
  photos.pdf               # the output PDF
```

Each numbered directory is a **checkpoint** -- you can inspect intermediate
results at any stage to see exactly what happened.

### Processing a subset

To test with just a few images first:

```bash
ghh run /path/to/photos --preview 5
```

This processes only the first 5 images, which is useful for checking
settings before running the full book.

## Viewing and comparing results

### Interactive comparison

After processing, use the compare tool to inspect every stage side-by-side
in your browser:

```bash
ghh compare /path/to/photos_output
```

This opens an interactive HTML viewer where you can:

- Navigate between images with **Up** / **Down** arrows (resets to stage 0)
  or **PgUp** / **PgDn** (keeps current stage)
- Switch between stages with the **Left** / **Right** arrows
- Press **S** for side-by-side view -- each pane can show a different image
  and stage independently (Shift variants control the right pane)
- Press **M** to view metadata for each stage
- Press **Z** to toggle zoom

### Generating a flipbook

Create a page-turning flipbook viewer for browsing on the web:

```bash
ghh flipbook /path/to/photos_output
```

### Publishing for colleagues

To share results with colleagues who don't have ghh installed, generate a
self-contained web-ready version with downscaled JPEG images:

```bash
ghh publish /path/to/photos_output /path/to/publish_dir \
    --with-flipbook --with-pdf
```

This creates a directory you can upload to any web server or share via
a file-sharing service.

## Using profiles

ghh offers four processing profiles that control which stages run:

| Profile | What it does | Use when |
|---------|-------------|----------|
| `full` | All stages (default) | Final production run |
| `geometry` | Geometry stages only (crop, perspective, straighten) | Quick geometry check |
| `clean` | Everything except OCR | OCR is not needed |
| `quick` | Geometry + enhancement, skip advanced stages | Fast preview |

```bash
ghh run /path/to/photos --profile geometry
```

## Next steps

- **[CLI Reference]({{< relref "cli-reference" >}})** -- all commands and
  options
- **[Configuration]({{< relref "configuration" >}})** -- customize
  processing with `book.toml`
- **[Pipeline Stages]({{< relref "pipeline" >}})** -- what each stage does
  and when to skip it
