---
title: "Pipeline Stages"
weight: 40
description: "What each processing stage does and when to skip it"
---

Guido's Helping Hand processes images through a sequence of stages,
numbered 0 through 15. The pipeline forks after common preparation
(stages 0--5) into a **Book branch** (full-page processing for PDF
production) and a **Score branch** (content-area extraction for OMR).
Each stage reads from the previous stage's output and writes to its own
checkpoint directory, making it easy to inspect intermediate results and
resume interrupted runs.

## How checkpointing works

Each stage writes its output to a numbered directory inside the output
folder:

```
LPA-1_output/
  00_preprocessed/
  01_stitched/
  02_oriented/
  ...
```

When you re-run `ghh run`, stages whose checkpoint directories already
exist and whose input images haven't changed are **skipped automatically**.
This means you can interrupt a run, fix a configuration issue, and resume
without re-processing everything from scratch.

**Config-hash invalidation:** Each stage declares the configuration
fields it depends on. When you change a setting in `book.toml` (e.g.
`deskew.max_angle`), the orchestrator detects the change and
automatically re-runs that stage *and all downstream stages*. No need
to manually delete checkpoint directories.

To force a stage to re-run anyway, delete its checkpoint directory.

## Running specific stages

You can run a subset of stages with the `--stages` option:

```bash
# Run only stages 0 through 5
ghh run /path/to/photos --stages 0-5

# Run specific stages
ghh run /path/to/photos --stages 0,4,5,12

# Mix ranges and individual stages
ghh run /path/to/photos --stages 0-2,5,7,12
```

## Stage reference

### Stage 0: Preprocess

**Checkpoint:** `00_preprocessed/`

Prepares raw photographs for processing:

- Removes flash hotspots (bright reflections from flash photography)
- Detects and masks finger intrusions at page edges
- Normalizes image format for downstream stages

**Skip when:** Your photos were taken without flash and without visible
fingers.

### Stage 1: Stitch

**Checkpoint:** `01_stitched/`

Groups and stitches partial photographs of the same page. If a page was
too large to photograph in one shot, multiple overlapping photos are
combined into a single image.

- Detects overlapping image groups using feature matching
- Stitches groups using homography estimation and multi-band blending
- Identifies and deduplicates retakes (near-identical photos)
- Detects non-content images (covers, spines) and flags them

**Skip when:** Every page was captured in a single photo.

**Key configuration:** `[stitch]` section, `[page_overrides]` for manual
grouping.

### Stage 2: Orientation

**Checkpoint:** `02_oriented/`

Corrects page rotation so all pages are right-side-up:

- Detects 90/270-degree rotation via staff line direction analysis
- Resolves 180-degree ambiguity using a cascading approach: letter shape
  analysis (Tesseract OSD), red title edge detection, and spine wear
  detection
- Computes a focus score (blur detection) for quality flagging

This stage cannot be skipped -- correct orientation is essential for all
downstream stages.

### Stage 3: Lens Correct

**Checkpoint:** `03_lens_corrected/`

Corrects barrel or pincushion distortion from the camera lens. Uses
distortion coefficients (k1, k2) from `book.toml`.

**Skip when:** `lens_distortion_k1` and `lens_distortion_k2` are both 0
(auto-skipped).

**Key configuration:** `[photography]` section: `lens_distortion_k1`,
`lens_distortion_k2`.

### Stage 4: Page Detect

**Checkpoint:** `04_page_detected/`

Finds the page within each photograph and crops to its boundaries:

- Cascading detection: Otsu thresholding, inverted Otsu, Canny edge
  detection, adaptive thresholding, full-image fallback
- Refines the detected quadrilateral with escalating epsilon values
- Classifies each page as "music" (has staff lines), "text" (no staff
  lines), or "blank"

This stage cannot be skipped.

**Key configuration:** `[page_detect]` section.

### Stage 5: Gentle Crop

**Checkpoint:** `05_gentle_crop/`

Crops the image to the page region detected by Stage 4 without applying
any perspective warp:

- Computes the axis-aligned bounding box of the Stage 4 quad
- Expands by a configurable margin (default 5%) to preserve all page content
- Crops the image to this expanded bounding box (no interpolation)
- Transforms `quad_corners` to the new coordinate system so downstream
  stages (Deskew, Perspective) can still use them

This replaces the former Perspective stage at position 5. The actual
perspective correction now runs at Stage 9, after Deskew has straightened
the page, producing a much cleaner result with less overcropping.

This stage cannot be skipped.

**Key configuration:** `[gentle_crop]` section: `margin_frac`.

### Stage 6: Content Area (Score branch only)

**Checkpoint:** `06_content/`

Extracts the actual content region within the page, removing outer
margins, border frames, and decorative edges:

- Detects printed border frames using Hough line detection
- Falls back to ink density analysis or a configurable inset fraction
- Applies feathered masking at edges for a smooth transition

**Skip when:** Pages have no border frame and you want to preserve
the full page including margins. Use `--skip-content-area`.

**Key configuration:** `[content_area]` section.

### Stage 7: Staff Extract (Score branch only)

**Checkpoint:** `07_staff_extract/`

Isolates music staff regions from mixed-content pages:

- Detects and extracts music staff areas, removing illustrations,
  decorative elements, and marginal annotations
- Produces cropped images containing only musical notation
- Pages without music pass through unchanged

**Status:** Not yet implemented.

**Key configuration:** `[staff_extract]` section (TBD).

### Stage 8: Deskew

**Checkpoint:** `08_deskewed/`

Straightens slightly tilted pages:

- Measures skew angle using staff line angles (music pages) or
  horizontal projection profile sharpness (text pages)
- Uses `quad_corners` from the sidecar as an ROI mask to avoid
  detecting angles in background regions outside the page
- Rotates the image to correct the skew
- Transforms `quad_corners` by the same rotation matrix and adjusts
  for any trim offset, preserving them for Stage 9 (Perspective)
- Trims background-filled borders after rotation

Runs in both branches: on the full page (Book branch) and on the
content area or staff extract output (Score branch).

**Skip when:** Pages are already straight. Use `--skip-deskew`.

**Key configuration:** `[deskew]` section: `max_angle`,
`skip_threshold`.

### Stage 9: Perspective (Book branch only)

**Checkpoint:** `09_perspective/`

Corrects perspective distortion (keystone effect) from the camera angle.
Now runs **after** Deskew, so it operates on an already-straightened
image:

- Reads the transformed `quad_corners` that have passed through
  Gentle Crop (Stage 5) and Deskew (Stage 8)
- Warps the quad into a rectangle using `cv2.warpPerspective`
- Sizes the output using the longest edges of the quad
- Fills any exposed corners with the estimated page background color
- Safety checks: skips if the quad is near-rectangular (common after
  Deskew has already straightened the page), if crop ratio is excessive,
  or if the homography would introduce tilt

Because Deskew has already removed skew, the perspective correction at
this stage is typically a small residual adjustment (trapezoid
correction), producing much cleaner results than the former pipeline
where Perspective ran before Deskew.

Skipped in the Score branch (Content Area already produces a clean crop).

**Key configuration:** `[perspective]` section: `max_skew_deg`,
`max_crop_frac`, `near_rect_threshold_deg`, `max_introduced_tilt_deg`.

### Stage 10: Dewarp

**Checkpoint:** `10_dewarped/`

Corrects page curl and waviness from the book's binding:

- Detects staff lines and fits polynomial curves
- Builds a deformation mesh to flatten curved lines
- Falls back to pass-through for pages without staff lines
- Optional AI-based dewarping with `--ai-dewarp`

**Skip when:** Pages are flat (not bound tightly). Use `--skip-dewarp`.

**Status:** Not yet implemented.

**Key configuration:** `--ai-dewarp` flag, `--skip-dewarp` flag.

### Stage 11: Enhance

**Checkpoint:** `11_enhanced/`

Applies a chain of image enhancement operations to improve readability:

1. Color cast correction
2. Illumination normalization
3. Shadow removal
4. Stain correction
5. Iron gall halo reduction
6. Show-through removal
7. White balance
8. CLAHE (adaptive contrast)
9. Salt deposit correction
10. Denoise
11. Sharpen

Each sub-step can be individually toggled in `[enhance]`. Severity values
from `[condition]` control how aggressively each correction operates.

**Skip when:** You want to preserve the original appearance. Use
`--skip-enhance`.

**Status:** Not yet implemented.

**Key configuration:** `[enhance]` section, `[condition]` section.

### Stage 12: Normalize

**Checkpoint:** `12_normalized/`

Ensures visual consistency across all pages in the book:

- Normalizes brightness and contrast across pages
- Standardizes page dimensions and DPI

This is a **batch stage** -- it processes all pages together rather than
one at a time, because it needs to compute global statistics first.

**Skip when:** Consistency across pages is not important. Use
`--skip-normalize`.

**Status:** Not yet implemented.

### OCR (Book branch only)

**Checkpoint:** (embedded in PDF)

Adds a searchable text layer to the final PDF using Tesseract or Kraken:

- Masks musical notation regions to avoid OCR on staff lines
- Extracts text regions between staves
- Produces an hOCR layer embedded in the PDF

Requires Tesseract to be installed. If missing, ghh skips this stage
gracefully.

**Skip when:** You don't need searchable text. Use `--skip-ocr`.

**Status:** Not yet implemented.

**Key configuration:** `[ocr]` section: `engine`, `language`.

### Stage 13: OMR (Score branch only)

**Checkpoint:** `13_omr/`

Transcribes music pages into GABC notation using
[ChantOMR](https://pgarciaq.github.io/chant-omr/), a deep learning model
for Gregorian chant:

- Runs OpenVINO inference on pages classified as "music" by Stage 4
- Produces `.gabc` files alongside symlinked PNG images
- Non-music pages (text, blank) pass through with skip metadata
- Models are loaded once and reused across all pages for efficiency

Requires the `chant-omr` package and a directory of exported OpenVINO IR
models. If `omr_model_dir` is not configured, the stage is skipped with a
warning.

**Skip when:** You don't need GABC transcriptions. Use `--skip-omr`.

**Key configuration:** `[omr]` section: `model_dir`, `beam_width`,
`device`. CLI: `--model-dir`.

### Stage 14: Score Render (Score branch only)

**Checkpoint:** `score/14_score_render/`

Renders GABC files from the OMR stage into music notation images using
Gregorio/LuaLaTeX:

- Converts `.gabc` files into engraved notation images
- Produces images suitable for inclusion as a score annex in the PDF
- Only runs as part of the Score branch (skipped with `--book-only`)

**Status:** Not yet implemented.

**Key configuration:** `[score_render]` section (TBD).

### Stage 15: PDF Assembly (Finalization)

**Checkpoint:** `15_pdf/`

Assembles the final PDF from processed page images:

- Uses `img2pdf` for lossless or JPEG-compressed assembly
- Applies configured DPI for correct physical sizing
- Names the output PDF after the input directory (e.g., `LPA-1.pdf`)
- Combines book pages from the Book branch with rendered scores from
  the Score branch as an annex

This stage cannot be skipped.

**Key configuration:** `[pdf]` section: `compression`, `jpeg_quality`,
`dpi`.
