# Guido's Helping Hand (ghh)

A Python pipeline that automatically processes photographed pages of historical music books (Gregorian chant) into searchable PDFs.

Named after the [Guidonian Hand](https://en.wikipedia.org/wiki/Guidonian_hand), the medieval mnemonic device used to teach singers the notes of the musical scale.

Designed for digitizing collections of 15+ books with varying ink colors, photography conditions, and physical condition -- coastal preservation, humidity damage, aging, iron gall corrosion, and more.

## Features

- **Zero configuration** -- just point it at a directory of photos and get a PDF
- **12-stage pipeline** -- orientation, page detection, perspective correction, dewarping, deskewing, enhancement, OCR, and PDF assembly
- **Auto-detection** -- ink colors, page boundaries, staff lines, and book characteristics are detected automatically per book
- **Checkpointed and resumable** -- interrupting mid-run loses no work; only incomplete images are reprocessed
- **Lossless intermediates** -- all checkpoints use PNG; JPEG compression happens only once during final PDF assembly
- **Hardware-accelerated** -- optional GPU acceleration via OpenCV UMat (OpenCL) and OpenVINO for AI dewarping
- **Graceful degradation** -- missing Tesseract? GPU unavailable? The pipeline adapts and keeps going

## Installation

Requires Python 3.11+.

```bash
pip install .
```

Optional extras:

```bash
pip install ".[ai]"             # OpenVINO for --ai-dewarp (Intel GPU)
pip install ".[omr]"            # OpenVINO for ghh omr (chant-omr model)
pip install ".[historical-ocr]" # Kraken OCR engine (requires Python <3.14)
pip install ".[dev]"            # pytest, ruff, coverage
```

Fedora RPMs (OCR + Intel GPU): see [docs/DEPENDENCIES.md](docs/DEPENDENCIES.md).

For OCR support, install [Tesseract](https://github.com/tesseract-ocr/tesseract):

```bash
# Fedora / RHEL
sudo dnf install tesseract

# Ubuntu / Debian
sudo apt install tesseract-ocr

# macOS
brew install tesseract
```

## Quick Start

Process a directory of book photos into a searchable PDF:

```bash
ghh run /path/to/book/photos
```

That's it. The pipeline auto-detects book characteristics, runs all 12 stages, and produces a PDF in a sibling output directory.

### Common Options

```bash
# Specify output directory
ghh run /path/to/photos -o /path/to/output

# Preview mode -- process only the first 5 images
ghh run /path/to/photos --preview 5

# Skip specific stages
ghh run /path/to/photos --skip-dewarp --skip-ocr

# Use AI-powered dewarping (requires openvino)
ghh run /path/to/photos --ai-dewarp

# Quick profile (skip optional stages for faster processing)
ghh run /path/to/photos --profile quick

# Delete intermediate files after successful processing
ghh run /path/to/photos --cleanup
```

### Other Commands

```bash
# Analyze a book and generate book.toml configuration (runs automatically if needed)
ghh analyze /path/to/photos

# Inspect a single image with diagnostic output
ghh inspect /path/to/image.jpg

# Review processed output with contact sheets
ghh review /path/to/output

# Compare pipeline stages locally in the browser
ghh compare /path/to/output

# Publish a web-friendly comparison site with downscaled JPEGs
ghh publish /path/to/output /var/www/mybook

# Clean up intermediate checkpoint directories
ghh cleanup /path/to/output
```

### Compare & Publish

After processing, `ghh compare` generates an interactive HTML viewer that lets you flip through every image across all pipeline stages. It's automatically created at the end of every `ghh run`.

```bash
# Open the full-book comparison viewer (auto-generated after run):
ghh compare /path/to/output

# Start at a specific image:
ghh compare /path/to/output IMG_0050
```

Keyboard shortcuts: **PgUp/PgDn** = prev/next image, **Left/Right** = prev/next stage, **S** = side-by-side, **M** = metadata panel, **Z** = zoom.

To share results with collaborators, `ghh publish` creates a self-contained directory with downscaled JPEGs suitable for any static web host:

```bash
# Publish all stages (1500px max, quality 85):
ghh publish /path/to/output /var/www/mybook

# Publish specific stages at a smaller size:
ghh publish /path/to/output ./pub --stages "0,5,7" --max-dim 1000
```

The two viewers have distinct color themes (blue for local compare, amber for published) so you can tell at a glance which one you're looking at.

## Pipeline Stages

| Stage | Name | Description |
|-------|------|-------------|
| 0 | Preprocess | Flash hotspot removal, finger detection |
| 1 | Stitch | Group and merge partial photos, deduplicate retakes |
| 2 | Orient | EXIF rotation + ink line angle + 180° disambiguation |
| 3 | Lens Correct | Barrel/pincushion distortion correction (auto-detected) |
| 4 | Page Detect | Page quad detection with multi-method fallback chain |
| 5 | Perspective | Perspective correction from detected quad corners |
| 6 | Content Area | Border frame detection and edge masking |
| 7 | Dewarp | Staff line polynomial mesh or AI dewarping |
| 8 | Deskew | Staff line angle or projection profile alignment |
| 9 | Enhance | Color cast, illumination, show-through, stain removal, CLAHE, sharpening |
| 10 | Normalize | Cross-page color and DPI normalization |
| 11 | OCR | Tesseract or Kraken text layer generation |
| 12 | PDF Assembly | Searchable PDF with optimized JPEG compression |

Stages marked as optional auto-skip when not needed (e.g., lens correction is skipped when no distortion is detected).

## Output Structure

```
output/
  book.toml              # Auto-generated book configuration
  00_preprocessed/       # Only if hotspots/fingers detected
  01_stitched/           # Only if partial photos exist
  02_oriented/           # EXIF-corrected images
  ...
  10_normalized/         # Final processed pages
  11_ocr/                # hOCR files
  output.pdf             # Final searchable PDF
  pipeline.json          # Stage status and parameters
  ghh.log                # Detailed processing log
```

## Processing Profiles

| Profile | Stages | Use Case |
|---------|--------|----------|
| `full` | All stages | Best quality (default) |
| `geometry` | Orientation through perspective | Straighten pages only |
| `clean` | Enhancement + normalization | Already-straight pages that need cleanup |
| `quick` | Skip dewarp, deskew, enhance | Fast preview |

## Development

Dependency matrix: **[docs/DEPENDENCIES.md](docs/DEPENDENCIES.md)**.

```bash
pip install -e ".[dev]"
pytest
```

**Fedora system packages (OCR + Intel GPU):**

```bash
sudo dnf install tesseract tesseract-langpack-lat
sudo dnf install intel-compute-runtime oneapi-level-zero   # OpenVINO GPU (dewarp + OMR)
sudo usermod -aG render "$USER"   # re-login for /dev/dri/renderD* access
```

Optional Python extras:

```bash
pip install -e ".[ai]"    # --ai-dewarp (OpenVINO)
pip install -e ".[omr]"   # ghh omr (OpenVINO, chant-omr model)
```

Run with coverage:

```bash
pytest --cov=ghh --cov-report=term-missing
```

Lint:

```bash
ruff check ghh tests
```

## License

MIT
