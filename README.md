# LPA Cleaner

A Python pipeline that automatically processes photographed pages of historical music books (Gregorian chant) into searchable PDFs.

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
pip install ".[ai]"             # OpenVINO + PyTorch for AI dewarping
pip install ".[historical-ocr]" # Kraken OCR engine for historical scripts
pip install ".[dev]"            # pytest, ruff, coverage
```

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
lpacleaner run /path/to/book/photos
```

That's it. The pipeline auto-detects book characteristics, runs all 12 stages, and produces a PDF in a sibling output directory.

### Common Options

```bash
# Specify output directory
lpacleaner run /path/to/photos -o /path/to/output

# Preview mode -- process only the first 5 images
lpacleaner run /path/to/photos --preview 5

# Skip specific stages
lpacleaner run /path/to/photos --skip-dewarp --skip-ocr

# Use AI-powered dewarping (requires openvino)
lpacleaner run /path/to/photos --ai-dewarp

# Quick profile (skip optional stages for faster processing)
lpacleaner run /path/to/photos --profile quick

# Delete intermediate files after successful processing
lpacleaner run /path/to/photos --cleanup
```

### Other Commands

```bash
# Analyze a book and generate book.toml configuration (runs automatically if needed)
lpacleaner analyze /path/to/photos

# Inspect a single image with diagnostic output
lpacleaner inspect /path/to/image.jpg

# Review processed output with contact sheets
lpacleaner review /path/to/output

# Clean up intermediate checkpoint directories
lpacleaner cleanup /path/to/output
```

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
  lpacleaner.log         # Detailed processing log
```

## Processing Profiles

| Profile | Stages | Use Case |
|---------|--------|----------|
| `full` | All stages | Best quality (default) |
| `geometry` | Orientation through perspective | Straighten pages only |
| `clean` | Enhancement + normalization | Already-straight pages that need cleanup |
| `quick` | Skip dewarp, deskew, enhance | Fast preview |

## Development

```bash
pip install -e ".[dev]"
pytest
```

Run with coverage:

```bash
pytest --cov=lpacleaner --cov-report=term-missing
```

Lint:

```bash
ruff check lpacleaner tests
```

## License

MIT
