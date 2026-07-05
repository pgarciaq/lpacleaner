# LPA Cleaner -- Technical Implementation Plan

A generic Python pipeline that automatically processes photographed pages of
historical music books (Gregorian chant) into searchable PDFs. Designed to
handle 15+ books with varying ink colors, photography conditions, and
physical condition (coastal preservation, humidity damage, aging).

## Design Principles

1. **Generic first**: No hardcoded colors or thresholds. All book-specific
   characteristics are auto-detected by the `analyze` command and stored in
   a per-book `book.toml` config.
2. **Defensive processing**: Every stage has fallback paths. Photography
   variations (flash, shadows, color casts, variable backgrounds) and
   physical damage (foxing, iron gall corrosion, water stains, salt deposits)
   are detected and handled automatically.
3. **Lossless intermediate format**: All inter-stage checkpoints use PNG
   (lossless). JPEG compression happens only once, at the very end (PDF
   assembly), avoiding cumulative generation loss from repeated lossy
   encode/decode cycles. Input images can be JPEG or PNG.
4. **Checkpointed and resumable**: Each stage writes its output to a numbered
   directory. The pipeline resumes at per-image granularity: if interrupted
   mid-stage, only incomplete images are reprocessed. Atomic writes prevent
   corrupt checkpoints.
5. **Hardware-accelerated**: Selective GPU acceleration via OpenCV UMat
   (Intel Arc OpenCL) for Canny, CLAHE, and remap operations. Optional
   AI dewarping via OpenVINO on GPU. Parallelism auto-scales to available
   RAM.

## Implementation Status

| Step | Component | Status | Tests | Commit |
|------|-----------|--------|-------|--------|
| 1 | Project scaffolding | **Done** | conftest.py + fixtures | `4cb5f87` |
| 2 | utils/image_io.py | **Done** | 17 tests | `4cb5f87` |
| 3 | utils/accel.py | **Done** | 15 tests | `4cb5f87` |
| 4 | utils/line_detect.py | **Done** | 21 tests | `d0ee9d4` |
| 5 | utils/preprocess.py | **Done** | 10 tests | `d0ee9d4` |
| 6 | config.py | **Done** | 21 tests | `4cb5f87` |
| -- | utils/geometry.py | **Done** | 11 tests | `d0ee9d4` |
| -- | pipeline.py (BaseStage, PipelineState) | **Done** | 16 tests | `d944406` |
| 7 | Stage 0 (preprocess) | **Done** | 18 tests | `9de99a7` |
| 8 | Stage 1 (stitch) | **Done** | 23 tests | `3632cdb` |
| -- | utils/page_find.py | **Done** | 9 tests | |
| -- | utils/stats.py | **Done** | 14 tests | |
| -- | Config: layout/condition fields | **Done** | 6 tests | |
| -- | Test tiers (@pytest.mark.slow) | **Done** | 16 tagged | |
| 9 | stages/analyze.py | **Done** | 18 tests | |
| 10 | Stage 2 (orientation) | **Done** | 31 tests | OSD + adaptive OSD + title + spine cascade; 224/224 on LPA-1 |
| 11 | Stage 3 (lens correct) | **Done** | 22 tests | Optional; skips when k1=k2=0; `cv2.undistort` with `max(w,h)` focal length |
| 12 | Stage 4 (page detect) | **Done** | 30 tests | Otsu→inverted Otsu→Canny→adaptive→full-image fallback; ink-aware classification; detect-only (no crop), quad in sidecar |
| 13 | Stage 5 (perspective) | **Done** | 28 tests | warpPerspective from Stage 4 quad; max-edge sizing; background fill; sidecar propagation |
| 14 | Stage 6 (content area) | **Done** | 29 tests | Hough border detection→ink density→inset fallback; feathered masking; margin padding; metadata forwarding |
| 15 | Stage 7 (deskew) | **Done** | 35 tests | Staff-line angle (184) + projection profile (27) + skipped (13); 224/224 on LPA-1 (4m12s); shared image_utils |
| 16 | Stage 8 (dewarp) | Pending | | |
| 17 | Stage 9 (enhance) | Pending | | |
| 18 | Stage 10 (normalize) | Pending | | |
| 19 | Stage 11 (OCR) | Pending | | |
| 20 | Stage 12 (PDF assembly) | **Done** | 35 tests | img2pdf; JPEG/PNG compression; case-insensitive; DPI layout; resume; exclude; 224-page LPA-1 PDF (311.9 MB) |
| 21 | Stage 13 (flipbook export) | Pending | | |
| -- | `compare` CLI command | **Done** | -- | Full-book HTML viewer; PgUp/PgDn image nav; side-by-side; auto-generated after `run` |
| 22 | Pipeline orchestrator | Pending | | |
| 23 | CLI polish | Pending | | |
| 24 | Integration tests | Pending | | |

**Totals:** 438 tests, all green.

---

## Reference Input (LPA-1 San Nicolas)

Measured from the first book to calibrate default parameters:

- 225 JPEG images, 4000x3000 (12MP), Canon PowerShot SX200 IS
- EXIF orientations: 144 normal (1), 43 rotated 90 CW (6), 38 rotated 90 CCW (8)
- Content: Gregorian chant on parchment -- 4-line red staves, black square notes, Latin text
- Staff ink: red (HSV hue ~5), occupies ~4.6% of pixel area
- Hough detection: ~100 horizontal segments (staff lines), ~550 vertical (bar lines)
- Page-to-background: Otsu threshold ~119, page area 71-98% of image
- Background: dark table surface
- Partial photos: some pages span multiple images (e.g., IMG_0232-0234 are 3 overlapping
  partials of the INDEX page); IMG_0231 is a non-content image (metal book cover)
- Fingers visible in some shots (e.g., IMG_0235 has a finger in the top-right corner)

These values serve as defaults; the `analyze` command recalibrates for each book.

---

## Project Structure

```
lpacleaner/
  pyproject.toml
  PLAN.md
  README.md
  tests/
    conftest.py               # Shared fixtures: synthetic images, temp dirs, Config
    fixtures/                  # Generated test images (not real photos)
    test_image_io.py
    test_line_detect.py
    test_geometry.py
    test_preprocess.py
    test_config.py
    test_stage_*.py            # One file per stage
    test_pipeline.py           # Integration tests
    test_cli.py                # CLI tests
  lpacleaner/
    __init__.py
    cli.py                  # Click CLI: analyze, run, inspect, compare commands
    compare.py              # Interactive HTML stage comparison viewer
    pipeline.py             # Orchestrator: run stages, manage checkpoints
    config.py               # Dataclass with all parameters + TOML loading
    stages/
      __init__.py
      analyze.py            # Analyze: auto-detect book characteristics, generate book.toml
      stitch.py             # Stage  1: image grouping, stitching partials, retake dedup
      orientation.py        # Stage  2: EXIF + ink line angle + 180deg disambiguation
      lens_correct.py       # Stage  3 (optional): barrel/pincushion correction (R7)
      page_detect.py        # Stage  4: page quad detection with fallback chain (R2)
      perspective.py        # Stage  5: perspective correction from quad corners
      content_area.py       # Stage  6: border frame detection, edge masking
      deskew.py             # Stage  7: staff line angle or projection profile
      dewarp.py             # Stage  8: staff line polynomial mesh or AI dewarping
      enhance.py            # Stage  9: R3 color cast, illumination, show-through,
                            #   shadows, stains, halos, salt, CLAHE, denoise, sharpen
      normalize.py          # Stage 10: cross-page color + DPI normalization
      ocr.py                # Stage 11: Tesseract/Kraken OCR
      pdf_assembly.py       # Stage 12: searchable PDF assembly
      flipbook_export.py    # Stage 13: static HTML flipbook for web publishing
    utils/
      __init__.py
      line_detect.py        # Generic ink detection, staff line detection, foxing filter (R9)
      geometry.py           # Quad ordering, homography, distance helpers
      image_io.py           # EXIF-aware load, save, checkpoint dir management
      image_utils.py        # Shared: estimate_background(), trim_to_content()
      accel.py              # GPU/OpenCL detection, UMat wrappers, OpenVINO init
      preprocess.py         # Flash hotspot removal (R1), finger detection (R8)
      page_find.py          # Simplified page quad detection (Otsu + largest contour)
      stats.py              # Median-based statistics, MAD outlier rejection
```

## Checkpoint Directory Layout

```
output/
  book.toml                  (auto-generated book config from analyze)
  00_preprocessed/           (hotspot removal, finger masking -- only if needed)
  01_stitched/               (partial photo groups merged -- only if partials exist)
  02_oriented/               IMG_0011.jpg, IMG_0012.jpg, ...
  03_lens_corrected/         IMG_0011.jpg, ...   (only if lens distortion detected)
  04_cropped/                IMG_0011.jpg, corners.json, ...
  05_perspective/            IMG_0011.jpg, ...
  06_content/                IMG_0011.jpg, ...   (content area cropped, edges masked)
  07_deskewed/               IMG_0011.jpg, ...
  08_dewarped/               IMG_0011.jpg, ...
  09_enhanced/               IMG_0011.jpg, ...
  10_normalized/             IMG_0011.jpg, ...   (cross-page color + DPI matched)
  11_ocr/                    IMG_0011.hocr, ...  (hOCR XML files)
  12_pdf/                    output.pdf, output.pdf.json
  13_flipbook/               index.html, pages/, page-flip.browser.js
  pipeline.json                                  (stage status, parameters used)
  lpacleaner.log                                 (detailed log, always verbose)
```

---

## Zero-Config Design

The tool must work out of the box with nothing more than an input directory:

```bash
lpacleaner run "/path/to/book/photos"
```

This single command must:

1. **Auto-detect output directory**: default to `{input_dir}/../{input_dir_name}_output/`
   (sibling of the input directory). Override with `-o`.
2. **Auto-run analyze if no `book.toml` exists**: if the output directory does not
   contain a `book.toml`, run the analyze step automatically before processing.
   This ensures ink color, layout, and condition are correctly detected for every
   book without manual intervention.
3. **Graceful dependency degradation**:
   - OCR: if `tesseract` is not installed, skip Stage 11 with a warning
     ("Tesseract not found, skipping OCR. Install with: dnf install tesseract").
     The PDF is produced without a text layer (img2pdf fallback).
   - GPU: if OpenCL is unavailable, fall back to CPU silently (already designed).
   - Kraken: if not installed and `--ocr-engine kraken` is requested, error with
     install instructions. Never auto-selected.
   - OpenVINO: if not installed and `--ai-dewarp` is requested, error with
     install instructions. Never auto-selected.
4. **Sane built-in defaults**: the profile is `full`, all optional stages are on,
   all auto-conditional stages self-skip when not needed. The pipeline produces
   the best possible output with zero user input.
5. **`analyze` remains available as a separate command** for users who want to
   review/edit `book.toml` before processing, or re-run analysis with different
   sample sizes. But it is never *required*.

### Fallback chain for ink detection (when analyze hasn't run)

If the pipeline runs without `book.toml` AND auto-analyze somehow fails to
detect ink color:

1. Try HSV histogram peak detection (default analyze algorithm)
2. Try channel-difference method (R-G, R-B for red; B-R, B-G for blue; etc.)
3. Last resort: skip color-dependent features (dewarp uses AI or pass-through,
   deskew uses projection profile instead of staff lines, content area uses
   fixed inset instead of border detection)

The pipeline never crashes due to missing configuration. It degrades gracefully
and flags affected pages for review.

---

## Analyze Command (`analyze.py`)

Scans sample images and generates a `book.toml` config file with auto-detected
book characteristics. Runs automatically as part of `lpacleaner run` if no
`book.toml` exists, but can also be run separately for manual review/editing.

### Usage

```
lpacleaner analyze INPUT_DIR [--output-dir OUTPUT_DIR] [--samples N]
```

`--samples N` overrides the adaptive default. When omitted, analyze samples
`max(10, min(30, total_images // 10))` images:

| Book size | Calculation | Samples |
|-----------|-------------|---------|
| 50 images | 50/10 = 5 → min clamp | **10** |
| 100 images | 100/10 = 10 | **10** |
| 225 images (LPA-1) | 225/10 = 22 | **22** |
| 400 images | 400/10 = 40 → max clamp | **30** |
| 800 images | 800/10 = 80 → max clamp | **30** |

Minimum 10 ensures median-based statistics are stable (after outlier rejection
we retain ≥7 samples). Maximum 30 avoids diminishing returns on large books.

### Robustness on Raw Images

The analyze command runs on raw, un-preprocessed images. These may contain
fingers, hotspots, partial overlaps, dark covers, camera distortions, and
large non-page areas (table surfaces). To produce reliable results:

- **Median-based statistics**: All measurements use medians (not means)
  across the sample set. A single outlier image (e.g., a cover shot, a
  heavily fingered page) does not skew the results.
- **Outlier rejection**: For each metric, discard values more than 2 MAD
  (median absolute deviations) from the median before computing the final
  calibration value.
- **Sample selection**: After partial photo and cover detection (steps 1-2),
  analyze only uses standalone page images. It skips covers, spine shots,
  and partial photos (which have distorted page geometry).
- **Simplified page quad detection**: Analyze uses a lightweight Otsu +
  largest-contour page finder (not the full Stage 4 fallback chain). This
  is sufficient for calibration purposes where sub-pixel accuracy is not
  needed. The simplified detector: (a) converts to grayscale, (b) applies
  Otsu threshold, (c) finds the largest contour, (d) approximates it with
  `cv2.approxPolyDP`, (e) returns 4 corners if the approximation is a
  quadrilateral, otherwise falls back to the central 80% of the image.
- **Coarse orientation before measurement**: Before ink and layout analysis,
  analyze determines the correct coarse orientation of sampled images (see
  "Coarse Orientation Detection" below). This ensures staff line detection
  works regardless of how the photos were taken.
- **Graceful degradation**: If fewer than 3 valid samples remain after
  filtering, analyze emits a WARNING and falls back to built-in defaults,
  setting `config_source = "defaults"` in pipeline.json.

### Coarse Orientation Detection

EXIF rotation handles most orientation issues, but some cameras write
incorrect or missing EXIF tags. For robustness, analyze performs a
coarse orientation check on a small subset (3-5 images):

```
1. Select 3-5 evenly-spaced sample images
2. For each image, try 4 cardinal rotations (0°, 90°, 180°, 270°):
   a. Downscale to 1000px wide
   b. Run detect_ink_mask() + HoughLinesP (±15° horizontal tolerance)
   c. Count near-horizontal staff line segments
3. For each image, the rotation with the most horizontal lines wins
4. Take the most common winning rotation across all subset images
5. If the winning rotation ≠ EXIF rotation, record a correction offset
```

This costs ~200ms per image (4 rotations × 50ms each on 1000px images).
Total cost for 5 images: ~1 second.

The ±15° tolerance in HoughLinesP handles real-world camera wobble.
Even a photo taken at 87° (3° off from 90°) has its staff lines brought
to ~3° from horizontal by the 90° rotation -- well within the detection
window. Fine-angle correction (the remaining 2-3°) is Stage 7's job.

### Algorithm

```
1. List all images in input directory, sorted by name

2. Adaptive sampling: select N = max(10, min(30, total // 10)) images,
   evenly spaced across the sorted file list

3. Partial photo detection (lightweight, on ALL consecutive pairs):
   a. Run ORB feature matching on consecutive image pairs
      (same algorithm as Stage 1 stitch grouping, but detection only)
   b. Identify groups of partial photos and retakes
   c. Exclude partial photos from sample set: only keep standalone
      images or the first image of each group (as a representative)
   d. Record detected groups in book.toml for Stage 1 to use

4. Filter non-content images (covers, spine shots) from sample set
   using is_non_content() from stitch utils

5. Coarse orientation detection (on 3-5 images from sample set):
   a. Try 4 cardinal rotations, count staff line segments
   b. Determine most common best rotation → coarse_rotation_offset
   c. If EXIF disagrees with detected orientation, record correction

6. For each sampled image:
   a. Apply EXIF rotation + coarse_rotation_offset (if any)
   b. Detect page quad (simplified: Otsu + largest contour)
   c. Crop to page quad (fallback: central 80% of image)

7. Ink color discovery:
   a. Convert cropped pages to HSV
   b. Mask out near-white (background, value > 200) and
      near-black (text/notes, value < 60)
   c. Histogram the remaining hue values
   d. Dominant hue peak = staff line ink color
   e. Compute optimal HSV range and channel-difference thresholds

8. Layout analysis:
   a. Detect border frame presence (consistent linear structures at page edges)
   b. Count staff lines per page (estimate expected range)
   c. Detect page number positions (top-right, top-left, bottom)
   d. Detect illustration presence (multi-colored regions)
   e. Compute median page aspect ratio

9. Photography condition analysis:
   a. Flash: check for clipped-white regions (> 250 in all channels)
   b. Color cast: compare per-channel means (gray-world deviation)
   c. Background contrast: which Otsu method works (normal vs inverted)
   d. Shadows: detect long straight high-gradient edges
   e. Lens distortion: measure edge curvature on page quads
   f. Fingers: check for skin-colored regions at image borders

10. Physical condition analysis:
    a. Foxing: count small reddish-brown blobs matching ink color but
       with low aspect ratio (round, not linear)
    b. Iron gall halos: measure brown halo width around text
    c. Stains: detect large-area brightness deviations
    d. Salt deposits: detect bright textured (not clipped) patches
    e. Show-through: detect faint reverse-page ink bleed
    f. Ink fading: compare staff ink saturation to expected range

11. Write book.toml
```

### Output: `book.toml`

```toml
[book]
name = "LPA-1 San Nicolas"       # informational (not consumed by stages)
type = "music"                    # informational

[ink]
staff_color_hue = 5              # HSV hue (0-180), ~red
staff_color_range = 15           # +/- hue tolerance
staff_saturation_min = 40
staff_value_min = 80
channel_diff_rg = 30             # R-G fallback threshold
channel_diff_rb = 30             # R-B fallback threshold

[layout]
has_border_frame = true
border_ink_matches_staff = true
page_number_position = "top-right"
expected_staff_lines_per_page = 16
has_illustrations = true
illustration_frequency = "rare"  # "none", "rare", "frequent"
median_aspect_ratio = 1.33       # width/height of detected page quads

[photography]
has_flash_hotspots = false
color_cast_detected = "slight_warm"  # "none", "slight_warm", "slight_cool", "strong_warm", "strong_cool"
background_contrast = "dark_on_light"  # "dark_on_light", "light_on_dark"
shadow_severity = "none"         # "none", "mild", "moderate", "severe"
lens_distortion_k1 = 0.0
lens_distortion_k2 = 0.0
fingers_detected = false
coarse_rotation_offset = 0       # 0, 90, 180, 270 -- correction beyond EXIF

[condition]
stain_severity = "mild"          # "none", "mild", "moderate", "severe"
ink_fading = "slight"            # "none", "slight", "moderate", "severe"
show_through_severity = "moderate"  # "none", "mild", "moderate", "severe"
foxing_severity = "mild"         # "none", "mild", "moderate", "severe"
iron_gall_halos = "slight"       # "none", "slight", "moderate", "severe"
salt_deposits = "none"           # "none", "mild", "moderate", "severe"

[ocr]
language = "lat"

[pipeline]
profile = "full"             # "full", "geometry", "clean", "quick"
# skip_content_area = false
# skip_dewarp = false
# skip_deskew = false
# skip_enhance = false
# skip_normalize = false
# skip_ocr = false
```

### Config Field Alignment

All `book.toml` sections except `[book]` are loaded by `Config.from_toml()`:

| TOML section | Config consumption | Purpose |
|---|---|---|
| `[book]` | **Informational only** -- not loaded into Config | Human-readable book identity |
| `[ink]` | Loaded → ink detection thresholds | Staff line color calibration |
| `[layout]` | Loaded → layout-aware stages | Border detection, staff count, page numbers |
| `[photography]` | Loaded → conditional stage behavior | Flash, fingers, lens, orientation correction |
| `[condition]` | Loaded → enhance aggressiveness | Severity drives sub-step intensity |
| `[ocr]` | Loaded → OCR configuration | Language, engine |
| `[pipeline]` | Loaded → profile and skip flags | Stage execution control |
| `[stitch]` | Loaded → stitch thresholds | Feature matching, overlap, grouping |
| `[page_overrides]` | Loaded → manual stitch control | Explicit grouping, exclusion |

Condition severities influence enhancement aggressiveness: `"none"` disables
the sub-step, `"mild"` uses conservative parameters, `"moderate"` and
`"severe"` use progressively more aggressive correction.

---

## Stage 0: Pre-processing (`preprocess.py`, conditional)

Runs only when `analyze` detects flash hotspots or fingers. Otherwise skipped.
Runs **before** stitching so that fingers and hotspots do not contaminate
feature matching or blending.

### R1. Flash Hotspot / Specular Highlight Removal

```
1. Detect clipped regions: mask = (B > 250) & (G > 250) & (R > 250)
2. Dilate mask with 5x5 kernel (catch hotspot edges)
3. If hotspot area > 0.5% of image: flag as flash-affected
4. Inpaint: cv2.inpaint(img, mask, inpaintRadius=5, flags=INPAINT_TELEA)
5. Record hotspot locations in metadata
```

### R8. Finger/Hand Detection and Masking

```
1. Convert to YCrCb color space
2. Skin mask: (133 < Cr < 173) & (77 < Cb < 127)
3. Filter: only regions touching image border, area > 1% of page
4. Inpaint with INPAINT_TELEA (radius=10)
```

### Parameters

```python
hotspot_clip_threshold: int = 250
hotspot_min_area_frac: float = 0.005
hotspot_inpaint_radius: int = 5
finger_detection: bool = False
finger_min_area_frac: float = 0.01
finger_inpaint_radius: int = 10
```

### Input/Output

- Input: raw JPGs from source directory
- Output: cleaned images in `00_preprocessed/`
- If neither hotspots nor fingers detected: stage is skipped, images pass through

---

## Stage 1: Image Grouping and Stitching (`stitch.py`)

Some pages are too large to capture in a single photo. The photographer
takes 2-4 overlapping partial shots from slightly different angles. These
must be detected and stitched into a single composite image before the
rest of the pipeline runs. Also detects and excludes non-content images
(book covers, spine shots).

Runs after pre-processing so that fingers and hotspots are already cleaned.

Example from LPA-1: IMG_0232, IMG_0233, IMG_0234 are three overlapping
partials of the INDEX page. IMG_0231 is the rusty metal book cover.

### Algorithm

```
1. Group detection (identify which images are partials of the same page):
   a. For each consecutive pair of images (N, N+1):
      - Extract ORB features (fast, rotation-invariant)
      - Match features with BFMatcher (Hamming distance)
      - Apply Lowe's ratio test (ratio < 0.75) to filter good matches
      - If > 30 good matches: compute homography (RANSAC)
      - If homography inlier ratio > 0.5 AND overlapping area > 20%:
        images N and N+1 belong to the same group
   b. Extend groups transitively: if A matches B and B matches C,
      then {A, B, C} is one group
   c. Single images (no matches) remain as standalone pages

2. Non-content detection:
   a. Images with no detectable page (simple Otsu threshold fails,
      no large bright contour): likely cover, spine, or equipment shots
   b. Images where > 80% of pixels are near-uniform dark/brown:
      likely book cover (e.g., IMG_0231)
   c. Flag as "non-content" in metadata; include or exclude based on
      book.toml setting:
      ```toml
      [page_overrides]
      include_covers = false
      exclude = ["IMG_0231.JPG"]
      ```

3. Stitching (for each group of 2+ images), fallback chain:
   a. Try cv2.Stitcher.create(cv2.Stitcher_PANORAMA)
      - Perspective model with bundle adjustment + multi-band blending
      - Best for angled partial photos (different camera positions)
      - Can fail on low overlap (<15%) or uniform textures
   b. If PANORAMA fails: try cv2.Stitcher.create(cv2.Stitcher_SCANS)
      - Affine model only (no perspective correction)
      - Better for flat photos taken from directly above the book
      - Simpler transform = fewer failure modes
   c. If SCANS fails: manual homography stitching
      - ORB features + BFMatcher(NORM_HAMMING)
      - findHomography(RANSAC, ransacReprojThreshold=5.0)
      - warpPerspective first image into second's coordinate frame
      - Feathered alpha blend at seam (20px gradient)
      - Gives full control over thresholds and error reporting
   d. If all stitching fails: keep single best image
      - Select the image from the group with the largest detected page area
      - Flag in pipeline.json for manual review
      - Log WARNING with match counts and failure reason
   e. Name output after first image in group (e.g., IMG_0232.jpg)
      and record the group composition + method used in metadata

4. Duplicate/retake detection (within groups):
   a. If two images in a group have > 90% overlap (nearly identical):
      they are retakes, not complementary partials
   b. Keep the one with better focus (higher Laplacian variance)
   c. Record discarded retake in metadata
```

### Parameters

```python
stitch_min_matches: int = 30        # minimum ORB feature matches
stitch_ratio_threshold: float = 0.75  # Lowe's ratio test
stitch_min_overlap_frac: float = 0.2  # minimum overlap area
stitch_inlier_ratio: float = 0.5    # RANSAC inlier ratio
retake_overlap_threshold: float = 0.9  # above this = retake, not partial
```

### Manual Override (`book.toml`)

```toml
[page_overrides]
# Explicitly define stitch groups (overrides auto-detection)
stitch_groups = [
    ["IMG_0232.JPG", "IMG_0233.JPG", "IMG_0234.JPG"],
]
# Exclude specific images entirely (book covers, test shots, etc.)
exclude = ["IMG_0231.JPG"]
# Images that look similar but are separate pages (disable auto-grouping)
no_stitch = ["IMG_0100.JPG", "IMG_0101.JPG"]
# Include book covers in the PDF (default: false)
include_covers = false
```

CLI flags:
- `--no-stitch`: skip Stage 1 entirely (treat all images as standalone pages)

### Input/Output

- Input: images from `00_preprocessed/` (or raw JPGs if Stage 0 skipped)
- Output: grouped/stitched images in `01_stitched/`
- Metadata: group composition, stitch method used, retakes discarded,
  non-content images, fallback reason (if any)
- If no groups detected: images are symlinked/copied unchanged

---

## Stage 2: Orientation Normalization (`orientation.py`)

Mandatory stage -- never skipped.  EXIF is intentionally **not** relied upon
(unreliable across camera models and transfer tools).

### Algorithm

Two-phase content-based orientation:

```
Phase 1 — Axis detection (0° vs 90°)
─────────────────────────────────────
1. Downscale to max 1200 px (speed).
2. Count near-horizontal line segments via Canny + HoughLinesP at 0°
   and at 90° CCW.
3. Pick the orientation with more horizontal lines, provided:
   a. max(h_lines_0, h_lines_90) >= 5     (_HORIZONTAL_LINE_MIN_COUNT)
   b. ratio of max / min >= 2.0            (confidence threshold)
4. Validate with _has_real_staff_lines():
   a. Build red ink mask (HSV hue within staff_color_range of
      staff_color_hue, saturation > 120).
   b. Reject if total red area < 0.5%      (_RED_AREA_MIN = 0.005)
      → page has no staff ink (blank/dirty), falls to portrait fallback.
   c. Apply horizontal morphological opening (kernel width = max(w/20, 30)).
   d. Reject if surviving red area > 5%    (_STAFF_AREA_MAX = 0.05)
      → textured surface (rusty cover), falls to portrait fallback.
5. If staff lines valid → apply axis rotation → proceed to Phase 2.
6. If no confident staff lines → portrait fallback:
   if w > h: rotate 90° CCW to enforce portrait → proceed to Phase 2.

Phase 2 — Polarity detection (0° vs 180°), cascading
─────────────────────────────────────────────────────
After axis correction the image should have staff lines horizontal (or
be portrait-enforced).  The question is whether it is right-side-up or
upside-down.  Four detectors are tried in order; the first one to
produce a confident result wins.

  2a. Tesseract OSD — standard pass
      ·  Downscale to max 1200 px.
      ·  Run pytesseract.image_to_osd() on the colour image.
      ·  If confidence >= 2.0 and degrees ∈ {0, 180}: use result.
         180° → rotate 180°.

  2b. Tesseract OSD — adaptive-threshold pass
      ·  Convert to grayscale.
      ·  Apply cv2.adaptiveThreshold(GAUSSIAN_C, blockSize=31, C=15).
      ·  Downscale to max 2000 px (higher resolution preserves small
         character detail in aged manuscripts).
      ·  Run pytesseract.image_to_osd(config="--dpi 300").
      ·  Same confidence threshold (>= 2.0).
      ·  Catches pages where the standard pass fails due to stains,
         ink bleed, decorative red shapes, or low character count.

  2c. Red title detection (chant-book-specific)
      ·  Build non-staff red mask (red_mask minus horizontal morph opening).
      ·  Build dark ink mask (V < 80).
      ·  For each row, compute:
         - red_per_row = fraction of non-staff red pixels
         - central_dark_per_row = fraction of dark pixels in central 80%
      ·  Title-eligible rows: red_per_row > 3% AND central_dark < 2.5%.
         (Titles are pure red lines with no dark text; body rubrics are
         always mixed with dark text in the same rows.)
      ·  Zero out non-eligible rows.
      ·  Compare weighted scores in the top 10% vs bottom 10% of the image
         (edge_frac = 0.10):
         - Weights: linear, highest at the extreme edge, decaying to 0 at
           the edge-zone boundary (np.linspace).
         - top_score = Σ (title_red[:top_cut] > 0) × top_weights
         - bot_score = Σ (title_red[bot_cut:] > 0) × bot_weights
      ·  If total_score (top + bot) < 50: insufficient signal → next detector.
      ·  If bot_score > top_score: upside-down → rotate 180°.

  2d. Spine detection (covers, blanks)
      ·  Convert to HSV.
      ·  Compare S/V ratio (saturation / max(brightness, 1)) of the left
         vs right 15% edge bands.
      ·  The more-saturated, darker edge is the spine (worn/oxidized).
      ·  Western convention: spine on the left.
      ·  If right S/V > left S/V and relative difference > 10%:
         rotate 180° to move spine to the left.

Phase 3 — Focus QA
───────────────────
·  Compute Laplacian variance on the central 80% of the image.
·  focus_score = cv2.Laplacian(gray, cv2.CV_64F).var()
·  If focus_score < threshold (default 100.0): flag as blurry.
·  Blur cannot be fixed programmatically; this flags pages for reshoot.
```

### Constants

| Constant | Value | Purpose |
|---|---|---|
| `_HORIZONTAL_LINE_MIN_COUNT` | 5 | Minimum HoughLinesP segments to consider staff lines present |
| `_STAFF_AREA_MAX` | 0.05 (5%) | Maximum red area after morph opening; above = textured surface |
| `_RED_AREA_MIN` | 0.005 (0.5%) | Minimum total red area; below = blank page (no staff ink) |
| `_OSD_MIN_CONFIDENCE` | 2.0 | Tesseract OSD confidence threshold for both passes |
| `_FOCUS_THRESHOLD_DEFAULT` | 100.0 | Laplacian variance below this = blurry |

### OSD implementation details

- Standard pass: 1200 px max, colour, default DPI.
- Adaptive pass: 2000 px max, Gaussian adaptive threshold (block=31, C=15),
  explicit `--dpi 300`.  The binarization removes stains, ink bleed, and
  parchment texture; the higher resolution preserves small character detail.
- Both passes use `pytesseract.image_to_osd()` which calls Tesseract in
  `--psm 0` mode (orientation and script detection from letter shapes).
- Graceful degradation: `TesseractError` is caught and treated as
  inconclusive (returns `(None, 0.0)`), falling through to the next detector.

### Parameters (Config)

```python
staff_color_hue: int = 0        # HSV hue of staff ink (red ≈ 0 or 180)
staff_color_range: int = 15     # ± hue tolerance for staff ink detection
focus_score_threshold: float = 100.0
```

### Input/Output

- Input: stitched image from `01_stitched/` (or `00_preprocessed/`)
- Output: correctly-oriented PNG in `02_oriented/`
- Metadata JSON sidecar with: `rotation_applied`, `orientation_method`,
  `focus_score`, `focus_threshold`, `is_blurry`
- `orientation_method` values: `staff_lines`, `staff_lines+polarity_flip`,
  `portrait_fallback`, `portrait_fallback+polarity_flip`

### Performance

- 224 images @ 4000×3000: ~5.5 minutes (Phase 1 ≈ 15%, Phase 2 OSD ≈ 70%,
  Phase 2 adaptive OSD ≈ 10%, Phase 3 ≈ 5%).
- Adaptive OSD only runs when standard OSD is inconclusive (~20% of images),
  so it adds minimal overhead for typical books.

---

## Stage 3: Lens Distortion Correction (`lens_correct.py`, optional) ✅

Runs only when `analyze` detects significant radial distortion (R7).
Must run **before** page detection: `cv2.undistort` requires the
original optical center (near image center), which is lost after
perspective correction. Correcting early also produces straighter
page edges, improving quad detection in Stage 4.

### Algorithm (implemented)

```
1. Check skip condition:
   - If lens_distortion_k1 == 0.0 AND lens_distortion_k2 == 0.0:
     skip this stage entirely (the default state).
   - Also skipped if profile/flag overrides say so.

2. Build camera matrix:
   fx = fy = max(width, height)  (reasonable default for unknown focal length)
   cx, cy = width / 2, height / 2  (optical center)

3. cv2.undistort(img, camera_matrix, dist_coeffs)
   where dist_coeffs = [k1, k2, 0, 0, 0]

4. Write metadata sidecar with: stage, k1, k2, focal_length_px
```

### Parameters (Config)

```python
lens_distortion_k1: float = 0.0    # radial distortion coefficient
lens_distortion_k2: float = 0.0    # higher-order radial coefficient
```

### Stage attributes

```python
name = "lens_correct"
number = 3
checkpoint_name = "03_lens_corrected"
error_class = "skippable"
```

### Input/Output

- Input: oriented image from `02_oriented/`
- Output: undistorted image in `03_lens_corrected/`
- If distortion coefficients are zero: stage is skipped, pipeline
  reads from `02_oriented/` for the next stage via
  `_find_previous_checkpoint()`

---

## Stage 4: Page Detection and Cropping (`page_detect.py`) ✅

### Algorithm (implemented)

```
1. Detection cascade (auto mode tries each in order):
   a. Otsu threshold (light page on dark background):
      · grayscale → Otsu → morph close (50×50) → findContours
      · Largest contour must be 30%–99.5% of image area
   b. Inverted Otsu (dark page on light background):
      · Same as (a) with THRESH_BINARY_INV
   c. Canny edge detection:
      · Canny(30,100) → dilate(5×5,3) → morph close → findContours
   d. Adaptive threshold:
      · adaptiveThreshold(Gaussian, block=51, C=10) → invert → morph close
   e. Full-image fallback:
      · Entire image treated as the page (always succeeds)

2. Quad refinement (from largest contour):
   a. approxPolyDP with escalating epsilon [0.02, 0.03 .. 0.08]
   b. If no epsilon yields 4 vertices: convexHull → try again
   c. Last resort: minAreaRect → boxPoints
   d. order_corners (TL, TR, BR, BL via sum/difference)

3. Quad expansion:
   a. Push each corner outward from centroid by page_detect_expand_frac × avg edge length
   b. Clamp to image bounds
   c. Compensates for Otsu contours that sit inside the actual page boundary

4. Pass-through (no crop):
   a. Full image passes through unchanged
   b. Quad corners stored in metadata sidecar for Stage 5
   c. Cropping deferred to Stage 5 (perspective correction), which inherently
      crops via warpPerspective — downstream stages (dewarp, deskew) need full
      page context including edges

5. Page type classification:
   a. detect_staff_lines() on quad ROI (ink-color-aware, not generic edge detection)
   b. ≥ 4 staff lines → "music"
   c. Low variance + bright mean → "blank"
   d. Otherwise → "text" or "other"

6. Metadata sidecar:
   stage, method, page_type, quad_corners
```

### Deferred features

- **Spread detection (K7)**: Not yet implemented.  Will be added when
  the dataset contains two-page spreads that need splitting.
- **GrabCut fallback**: Removed from cascade; the existing four methods
  plus the full-image fallback cover all observed cases.

### Stage attributes

```python
name = "page_detect"
number = 4
checkpoint_name = "04_page_detected"
error_class = "skippable"
```

### Parameters (Config)

```python
page_detect_method: str = "auto"       # "auto", "otsu", "otsu_inverted", "canny", "adaptive"
page_detect_morph_kernel: int = 50
page_detect_epsilon: float = 0.02
page_detect_min_area_frac: float = 0.30
page_detect_padding: int = 10          # reserved for future use
page_detect_expand_frac: float = 0.03  # fraction of avg edge length to expand quad outward
```

### Input/Output

- Input: image from `03_lens_corrected/` (or `02_oriented/` if lens correction skipped)
- Output: **full image unchanged** in `04_page_detected/`, metadata sidecar with quad corners
- Stage 5 reads the quad corners from the sidecar to apply perspective correction (and crop)

---

## Stage 5: Perspective Correction (`perspective.py`) ✅

### Algorithm (implemented)

```
1. Load quad corners from Stage 4 metadata sidecar:
   a. BaseStage.run() reads {stem}.json from input directory
   b. Passes metadata dict to process_image()
   c. If no quad_corners key → pass-through (image unchanged)
   d. If malformed (not 4x2) or degenerate (side < 10px) → pass-through

2. Order corners TL → TR → BR → BL via order_corners()

3. Compute target rectangle (max of opposite edges, not average):
   width  = max(dist(TL,TR), dist(BL,BR))
   height = max(dist(TL,BL), dist(TR,BR))
   Using max prevents content loss from the longer edge.

4. Build destination corners:
   dst = [[0,0], [width-1,0], [width-1,height-1], [0,height-1]]

5. Compute transform: M = getPerspectiveTransform(src, dst)

6. Estimate background color:
   a. Sample outermost 5% border pixels on all four sides
   b. Avoid double-counting corner rectangles
   c. Per-channel median → fill color tuple

7. Apply: warpPerspective(img, M, (width, height),
       INTER_LINEAR, BORDER_CONSTANT, bg_color)
```

Out-of-bounds pixels are filled with the estimated page background color,
not black. This prevents downstream stages (enhance, normalize) from
being confused by black corners.

### Stage attributes

```python
name = "perspective"
number = 5
checkpoint_name = "05_perspective"
error_class = "skippable"
```

### Input/Output

- Input: full image + quad corners from `04_page_detected/` (sidecar JSON)
- Output: rectangular image in `05_perspective/` (this is where the actual crop happens)
- Metadata sidecar: stage, method, src_quad, dst_size, background_color

---

## Stage 6: Content Area Detection (`content_area.py`) ✅

Detects the border frame that surrounds content on most pages, providing
a tighter crop than the page edge, and masks residual adjacent page edges.

### Algorithm (implemented)

```
1. Detect border frame (when has_border_frame is True):
   a. Isolate ink mask via detect_ink_mask (line_detect.py)
   b. Morphological close with 5x5 kernel to join fragments
   c. HoughLinesP: find long horizontal and vertical ink-colored lines
      (min length = 25% of min dimension, threshold=80, maxGap=30)
   d. Filter: horizontals within top/bottom 35%, verticals within left/right 35%
   e. Intersect 4 border lines to find content rectangle
   f. Fallback cascade:
      1. Ink density bounding box (bounding rect of dilated ink mask + 2% pad)
      2. Fixed inset (default 5% from all edges)

2. Mask adjacent page edges:
   a. Build alpha mask: 1.0 inside content rect, 0.0 outside
   b. Gaussian blur alpha (sigma=20px) for feathered transition
   c. Blend: result = img * alpha + bg_color * (1 - alpha)

3. Crop and add uniform margins:
   a. Crop to content rectangle
   b. Add padding (default 2% of width) filled with background color

4. Special cases:
   a. Blank pages (from Stage 4/5 metadata): pass through unchanged
   b. Content rect < 10% of image area: fall back to inset

5. Store content_rect, method, margin_px, background_color in metadata
```

### Stage attributes

```python
name = "content_area"
number = 6
checkpoint_name = "06_content"
error_class = "skippable"
```

### Parameters

```python
content_detect_inset_fallback: float = 0.05
content_margin_padding: float = 0.02
content_feather_sigma: int = 20
```

### Input/Output

- Input: perspective-corrected image from `05_perspective/` + sidecar metadata
- Output: content-cropped image with margins in `06_content/`
- Metadata sidecar: stage, method, content_rect, margin_px, background_color, page_type

---

## Stage 7: Deskew (`deskew.py`)

### Algorithm

```
1. Detect staff lines (line_detect.py with geometric filter)
2. If lines found: skew_angle = median angle of line segments
3. If no lines (text-only pages):
   a. Binary threshold (Otsu)
   b. Projection profile with coarse-to-fine search:
      - Coarse: test every 1.0 degree in [-max_angle, +max_angle]
      - Fine: refine ±1 degree around best in 0.1 steps
      - Score = variance of row sums. Best = max variance.
      - Image downscaled to 25% before computing row sums
        (accuracy is preserved, ~6x speedup).
4. Clamp angle: if |skew_angle| > deskew_max_angle, clamp and
   log WARNING (may indicate Stage 2 orientation failure).
5. Skip rotation if |skew_angle| < deskew_skip_threshold (0.1 deg).
6. Estimate background color: shared estimate_background() utility
   (median of border pixels, outermost 5%).
7. Rotate by -skew_angle via cv2.warpAffine with
   borderMode=cv2.BORDER_CONSTANT, borderValue=bg_color.
8. Post-geometry trim (shared trim_to_content() utility):
   a. Threshold: pixels significantly different from bg_color
   b. Find bounding rect of non-background region
   c. Crop to the content bounding box
   d. Add uniform margin padding (default 2% of width) filled with
      bg_color
   Note: Stage 8 (dewarp) also calls trim_to_content() at its end.
   Running it in both stages handles all skip combinations and the
   overhead is negligible (~5ms per image).
9. Forward page_type from input metadata.
```

### Parameters

```python
deskew_max_angle: float = 5.0
deskew_angle_step: float = 0.1
deskew_skip_threshold: float = 0.1
```

### Input/Output

- Input: image from `06_content/`
- Output: deskewed image in `07_deskewed/`

---

## Stage 8: Dewarping (`dewarp.py`)

Two paths: classical (default) and AI (optional).

### Classical Path: Staff-Line Polynomial Mesh

```
1. Detect staff lines (via line_detect.py):
   a. detect_ink_mask(img, cfg) -> color mask
   b. Geometric filter (R9): morph open with 1x30 horizontal kernel,
      discard components with aspect_ratio < 5:1 (foxing spots)
   c. Morphological close with 1x20 kernel (connect broken segments)
   d. HoughLinesP(mask, rho=1, theta=pi/180, threshold=80,
                  minLineLength=width*0.3, maxLineGap=30)
   e. Filter to lines within 15deg of horizontal
   f. Cluster by y-coordinate (binning with eps=20px)

2. Fit polynomial to each staff line:
   a. Collect all points from segments in each cluster
   b. Fit degree-3 polynomial: y = ax^3 + bx^2 + cx + d
   c. Evaluate at N evenly-spaced x positions

3. Build dewarping mesh:
   a. For each line i with polynomial p_i(x):
      target_y_i = p_i(width/2)
      dy = p_i(x) - target_y_i for each x
   b. Interpolate dy across full image height
   c. map_x[y,x] = x, map_y[y,x] = y - dy_interpolated[y,x]

4. Estimate background color: shared estimate_background() utility.
5. Apply: cv2.remap(img, map_x, map_y, INTER_CUBIC,
       borderMode=cv2.BORDER_CONSTANT, borderValue=bg_color)
   (Use UMat for GPU -- 1.5x speedup)

6. Pages with < 2 detected staff lines:
   - Flag in metadata
   - If --ai-dewarp: use AI path
   - Otherwise: pass through unchanged

7. Post-geometry trim: shared trim_to_content() utility
   (same as Stage 7 -- cleans up background artifacts from remap).
```

### AI Path: DocTr GeoTr (Optional, `--ai-dewarp`)

```
1. One-time: convert PyTorch model to OpenVINO IR (FP16)
2. Per image: resize to 288x288, normalize, infer on GPU (1.27ms)
3. Output: 288x288x2 warp field, upscale to original size
4. Apply via cv2.remap (GPU UMat)
```

### Parameters

```python
dewarp_hough_threshold: int = 80
dewarp_min_line_length_frac: float = 0.3
dewarp_hough_max_gap: int = 30
dewarp_cluster_eps: int = 20
dewarp_poly_degree: int = 3
dewarp_min_staff_lines: int = 2
dewarp_morph_kernel: tuple = (1, 20)
```

### Input/Output

- Input: deskewed image from `07_deskewed/`
- Output: dewarped image in `08_dewarped/`
- Metadata: staff positions, polynomial coefficients, method used

---

## Stage 9: Image Enhancement (`enhance.py`)

The most feature-rich stage, incorporating core enhancement plus 6
robustness features (R3, R5, R6, R10, R11, and show-through/sharpening).

### Algorithm

```
1. R3 Color cast correction:
   a. Gray-world: compute per-channel means, scale so all equal
   b. Or: sample background pixels, compute correction from their color
   c. Applied early, before any other enhancement

2. Illumination normalization:
   a. Background estimation: cv2.morphologyEx(gray, MORPH_CLOSE, 151)
   b. Normalized = gray / background * 255, per-channel for color

3. R5 Sharp shadow removal:
   a. Detect long straight high-gradient edges (> 50% of image span)
   b. Estimate illumination step across each shadow edge
   c. Apply local gain correction, blend smoothly

4. R6 Stain-aware enhancement:
   a. Detect large-area brightness deviations in 256x256 blocks
   b. Apply local normalization (51px kernel) in stain regions
   c. Smooth tide-mark boundaries with bilateral filtering

5. R10 Iron gall ink halo reduction:
   a. Detect text regions (dark ink threshold)
   b. Dilate by 15px to create halo zone
   c. Apply local contrast normalization (31px kernel) in halo zone
   d. Blend with global normalization via feathered mask (sigma=10)

6. Show-through / bleed-through removal:
   a. Gamma method (default): apply gamma=0.6 to mid-range [120-200]
      to push faint bleed-through toward background
   b. DoG method (alternative): difference-of-Gaussians isolates sharp
      foreground ink from smooth bleed-through

7. White balance:
   a. Sample background pixels (top 10% brightness)
   b. Scale each channel to target warm white (245, 240, 230)

8. CLAHE contrast enhancement:
   a. LAB color space, CLAHE on L channel (clipLimit=2.0, grid=8x8)
   b. Use UMat for GPU (6.7x speedup)

9. R11 Salt/efflorescence correction:
   a. Detect bright (> 230 L) but textured (stddev > 20) regions
   b. Apply localized CLAHE (clipLimit=4.0) in affected areas

10. Noise reduction:
    a. cv2.fastNlMeansDenoisingColored(h=5, template=7, search=21)

11. Sharpening (compensate resampling blur from Stages 3-5):
    a. Unsharp mask on L channel: alpha=0.3, sigma=1.5
    b. Clip to [0, 255]

12. Optional binarization (--binarize):
    a. cv2.adaptiveThreshold(blockSize=31, C=15)
```

### Parameters

```python
# Color cast
color_cast_correction: bool = True
color_cast_method: str = "gray_world"

# Illumination
enhance_bg_kernel: int = 151
enhance_white_target: tuple = (245, 240, 230)

# Shadow removal (R5)
shadow_correction: bool = True
shadow_detect_min_span_frac: float = 0.5
shadow_detect_gradient_threshold: int = 30

# Stain handling (R6)
stain_correction: bool = True
stain_detect_block_size: int = 256
stain_detect_stddev_threshold: float = 2.0
stain_local_kernel: int = 51

# Iron gall halos (R10)
halo_reduction: bool = True
halo_dilate_px: int = 15
halo_local_kernel: int = 31
halo_feather_sigma: int = 10

# Show-through
enhance_showthrough_gamma: float = 0.6
enhance_showthrough_method: str = "gamma"

# CLAHE
enhance_clahe_clip: float = 2.0
enhance_clahe_grid: int = 8

# Salt (R11)
salt_correction: bool = True
salt_brightness_threshold: int = 230
salt_texture_threshold: float = 20.0
salt_clahe_clip: float = 4.0

# Denoise
enhance_denoise_h: int = 5
enhance_denoise_template: int = 7
enhance_denoise_search: int = 21

# Sharpen
enhance_sharpen_alpha: float = 0.3
enhance_sharpen_sigma: float = 1.5

# Binarize
enhance_binarize_block: int = 31
enhance_binarize_c: int = 15
```

### Input/Output

- Input: dewarped image from `08_dewarped/`
- Output: enhanced image in `09_enhanced/`

### Future: Parchment Recto/Verso Handling (GitHub #2)

Historical manuscripts from ~1700s are written on parchment (animal skin),
which has distinct recto (hair-side, smoother, whiter) and verso (flesh-side,
rougher, yellower) characteristics. Within the same bifolium, alternating
pages naturally have different base colors and ink absorption.

Current approach: all pages treated identically. This produces acceptable
results but may over-correct recto pages or under-correct verso pages.

Future refinement: detect recto/verso automatically via base-color histogram
analysis, then apply enhancement parameters per-group. Stage 10 (normalize)
should normalize within recto and verso groups separately rather than forcing
all pages to the same white point.

---

## Stage 10: Cross-Page Normalization (`normalize.py`)

A **global pass** that runs after all individual pages are enhanced.
Ensures visual consistency across the full book.

### Algorithm

```
1. Color consistency:
   a. For each page: sample background pixels (top 20% brightness),
      record median background color (B, G, R)
   b. Compute global target: median of all per-page backgrounds
   c. Per page: scale channels to match global target

2. Resolution / DPI normalization:
   a. Compute median dimensions across all pages
   b. Resize all to median dimensions (INTER_LANCZOS4)
   c. Set DPI metadata to 300
```

### Parameters

```python
normalize_target_dpi: int = 300
normalize_color_method: str = "median"
normalize_bg_sample_percentile: float = 0.8
```

### Input/Output

- Input: all images from `09_enhanced/`
- Output: normalized images in `10_normalized/`

---

## Stage 11: OCR (`ocr.py`)

### Algorithm

```
1. Blank page detection: skip if grayscale stddev < 15

2. Notation masking (see K5):
   a. Load staff line positions from Stage 8 metadata
   b. For each staff group: mask the region from top staff line - margin
      to bottom staff line + margin with white
   c. This leaves only inter-staff text lines visible to Tesseract

3. Tesseract (default):
   hocr = pytesseract.image_to_pdf_or_hocr(masked_img, lang=cfg.ocr_lang,
          extension='hocr', config='--psm 6')
   Filter output: discard words with confidence < 40%

4. Kraken (optional, --ocr-engine kraken):
   segment + recognize + output hOCR/ALTO
```

### Parameters

```python
ocr_engine: str = "tesseract"
ocr_lang: str = "lat"                # configurable per book via book.toml
ocr_psm: int = 6
ocr_blank_stddev_threshold: float = 15
```

### Input/Output

- Input: normalized image from `10_normalized/`
- Output: hOCR file in `11_ocr/`

---

## Stage 12: PDF Assembly (`pdf_assembly.py`)

### Algorithm (implemented)

```
1. Resolve input: find the latest completed stage checkpoint directory.
   The CLI's _find_previous_checkpoint walks backward from stage 12.

2. Collect images from input directory (sorted by filename = page order).
   Exclude images listed in cfg.exclude_images.

3. Image compression for PDF:
   a. pdf_compression = "jpeg" (default): convert each PNG to JPEG in
      memory at cfg.pdf_jpeg_quality (default 90) using cv2.imencode.
      This is the ONLY lossy step in the entire pipeline.
      One compression pass = negligible quality loss.
      Result: ~312 MB for 224 pages at quality 90.
   b. pdf_compression = "png": pass PNG bytes directly to img2pdf.
      Perfect quality, larger file size.
   c. Compression mode is case-insensitive ("JPEG", "jpeg", "PNG", etc.).

4. Assembly: img2pdf.convert() with custom layout function setting
   page dimensions from image pixel dimensions and cfg.pdf_dpi (default 300).

5. Atomic write: output.pdf.tmp → output.pdf via os.replace().

6. Write metadata sidecar (output.pdf.json) with page count, compression,
   quality, DPI, file size, and input directory.

7. Resume: if output.pdf exists and stage is marked done in PipelineState,
   skip entirely. PDF assembly is all-or-nothing (no partial resume).
```

### Parameters

```python
pdf_compression: str = "jpeg"      # "jpeg" or "png" (case-insensitive)
pdf_jpeg_quality: int = 90
pdf_dpi: int = 300
```

### TOML configuration

```toml
[pdf]
compression = "jpeg"   # or "png"
jpeg_quality = 90
dpi = 300
```

### Input/Output

- Input: images from latest completed stage checkpoint
- Output: `output.pdf` + `output.pdf.json` sidecar

### Future enhancements (deferred)

- OCR layer via ocrmypdf (depends on Stage 11)
- JPEG 2000 compression mode
- Page reordering (OCR-based or manual via book.toml)
- Spread detection/splitting (K7)

---

## Stage 13: Flipbook Export (`flipbook_export.py`)

Generates a self-contained static HTML flipbook from the processed images,
suitable for publishing on a website. The output is a directory that can be
uploaded to any static hosting (GitHub Pages, Netlify, a parish website, etc.)
without server-side dependencies.

### Algorithm

```
1. Resolve input: find the latest completed image stage checkpoint
   (same logic as Stage 12 -- walk backward from stage 13).

2. Collect images (sorted by filename = page order).
   Exclude images from cfg.exclude_images.

3. Downscale images for web:
   a. flipbook_max_width (default 1600px): resize if wider, preserving
      aspect ratio.
   b. Export as JPEG at flipbook_jpeg_quality (default 85) into
      output_dir/pages/.
   c. Optionally generate thumbnail versions for lazy loading.

4. Generate index.html with embedded StPageFlip viewer:
   a. Bundle page-flip.browser.js (vendored, MIT license, ~50 KB).
   b. loadFromImages() with the page JPEG paths.
   c. Responsive layout: size="stretch", showCover=true.
   d. Touch/swipe support for mobile.
   e. Keyboard navigation (arrow keys, Home/End).
   f. Page number indicator.

5. Write metadata sidecar (flipbook.json) with page count, dimensions,
   total size, and generation timestamp.
```

### Parameters

```python
flipbook_max_width: int = 1600       # max page width in pixels for web
flipbook_jpeg_quality: int = 85      # JPEG quality for web images
flipbook_title: str = ""             # title shown in the viewer
```

### TOML configuration

```toml
[flipbook]
max_width = 1600
jpeg_quality = 85
title = "LPA 1 - San Nicolás"
```

### Input/Output

- Input: images from latest completed image stage checkpoint
- Output: `flipbook/` directory containing:
  - `index.html` (self-contained viewer)
  - `pages/` (resized JPEG images)
  - `page-flip.browser.js` (vendored library)
  - `flipbook.json` (sidecar metadata)

### Library choice: StPageFlip (`page-flip`)

- **MIT license**, zero dependencies, ~50 KB bundled
- Realistic 3D page-turning physics with canvas rendering
- Works on desktop and mobile (touch/swipe)
- Portrait and landscape support
- `loadFromImages()` API fits our use case perfectly
- React wrapper available (`react-pageflip`) if needed later
- The most actively used open-source flipbook library as of 2026
  (64K+ weekly npm downloads)

### Alternatives considered

| Library | Notes |
|---------|-------|
| Turn.js | jQuery dependency, commercial license for v4+ |
| Flipbook.js | Less maintained, smaller community |
| pdf.js + StPageFlip | Could render PDF directly, but pre-rendered images are faster and simpler |
| Zaya | Commercial with free tier; not fully open source |

### CLI integration

```bash
lpacleaner flipbook INPUT_DIR -o OUTPUT_DIR    # standalone command
lpacleaner run INPUT_DIR --stages 0-6,13       # or as part of pipeline
```

### Design notes

- Like Stage 12, this stage subclasses `BaseStage` but overrides `run()`
  entirely (batch output, not per-image checkpoints).
- The `page-flip.browser.js` file is vendored into the package to avoid
  requiring npm/node at runtime. Updated periodically.
- The generated flipbook is fully static -- no server, no build step, no
  Node.js required to view it. Just open `index.html` in a browser.

---

## Stage Contract (`pipeline.py`)

All 14 stages share the same lifecycle via `BaseStage`:

```python
class BaseStage(ABC):
    name: str              # "preprocess", "stitch", "orientation", ...
    number: int            # 0, 1, 2, ...
    checkpoint_name: str   # "00_preprocessed", "01_stitched", ...
    error_class: str       # "skippable", "critical", "fatal"

    @abstractmethod
    def process_image(self, img, metadata, cfg) -> (img, metadata):
        """Process a single image. Only this varies per stage."""

    def should_skip(self, cfg) -> bool:
        """Check profile/flags. Default delegates to cfg.should_skip_stage()."""

    def run(self, input_dir, output_dir, cfg, state) -> StageResult:
        """Orchestration loop: iterate images, resume, error handling,
        atomic checkpoint writes, metadata sidecars."""
```

`run()` handles:
1. List input images (PNG/JPG from previous stage's checkpoint dir)
2. Check `PipelineState` per-image: skip already-done images (resume)
3. Call `process_image()` inside try/except
4. On success: `save_checkpoint()` atomically, `mark_image_done()`
5. On failure: apply error_class policy:
   - skippable: copy input image through unchanged, log WARNING
   - critical: exclude image from pipeline, log ERROR
   - fatal: stop pipeline immediately
6. Return `StageResult` with counts (processed, skipped, failed, excluded)

Individual stages **only** implement `process_image()` and optionally
override `should_skip()`.

### PipelineState (`pipeline.json`)

Persisted state for resume, cache invalidation, and end-of-run reporting:

```json
{
  "config_source": "analyzed",
  "config_hashes": {
    "stage_2": "a1b2c3...",
    "stage_8": "d4e5f6..."
  },
  "done": {
    "02_oriented": ["IMG_0001", "IMG_0002", ...],
    "08_dewarped": ["IMG_0001"]
  },
  "results": {
    "orientation": {"processed": 220, "skipped": 0, "failed": 3, "excluded": 2}
  }
}
```

- `config_source`: `"defaults"` or `"analyzed"` -- tracked so the end-of-run
  report can warn when analyze hasn't run
- `config_hashes`: per-stage hash of dependent config fields; if changed,
  stage + downstream are invalidated
- `done`: per-stage set of completed image stems; used for per-image resume
- `results`: per-stage outcome counts for the end-of-run summary

---

## Pipeline Orchestrator

The `run_pipeline()` function in `pipeline.py` chains all stages in order,
passing each stage's output directory as the next stage's input directory.

### Stage Chain

```python
STAGE_ORDER = [
    PreprocessStage(),     # 0: hotspot + finger removal
    StitchStage(),         # 1: grouping + stitching
    OrientationStage(),    # 2: rotation + 180-deg disambiguation
    LensCorrectStage(),    # 3: barrel/pincushion correction
    PageDetectStage(),     # 4: page quad detection
    PerspectiveStage(),    # 5: perspective correction
    ContentAreaStage(),    # 6: border detection + crop
    DeSkewStage(),         # 7: staff line angle or projection profile
    DewarpStage(),         # 8: polynomial mesh or AI dewarping
    EnhanceStage(),        # 9: color correction, denoising, sharpening
    NormalizeStage(),      # 10: cross-page color + DPI
    OCRStage(),            # 11: Tesseract/Kraken OCR
    PDFAssemblyStage(),    # 12: final PDF output
]
```

### Orchestration Logic

```python
def run_pipeline(cfg: Config) -> PipelineState:
    state = PipelineState.load(cfg.output_dir)
    input_dir = cfg.input_dir

    # Auto-analyze if no book.toml exists
    if not (cfg.input_dir / "book.toml").exists():
        analyze(cfg)

    for stage in STAGE_ORDER:
        if stage.should_skip(cfg):
            input_dir = previous_stage_dir  # pass through
            continue

        result = stage.run(input_dir, cfg.output_dir, cfg, state)
        state.record_result(result)
        state.save()

        input_dir = cfg.output_dir / stage.checkpoint_name

    print_end_of_run_report(state)
    return state
```

### Progress Reporting

Each stage emits progress via `tqdm` (when available) or plain logging:

```
[Stage 0] Pre-processing... 225/225 [00:03, 75.0 img/s]
[Stage 1] Grouping & stitching... 220/220 [00:15, 14.7 img/s]
[Stage 2] Orienting images... 220/220 [00:12, 18.3 img/s]
...
[Stage 12] Assembling PDF... done (222 pages, 185 MB)

=== Pipeline Complete ===
Processed 222/225 images in 12m34s.
  2 flagged (soft fallback): IMG_0045 (low focus), IMG_0013 (no staff lines)
  1 excluded (critical failure): IMG_0080 (Stage 4: no page quad found)
Output: /path/to/output/output.pdf
Config source: analyzed (book.toml)
```

If `--quiet`: only errors, warnings, and the final summary.
If `--verbose`: per-image timing and parameter values.

---

## Shared Utilities

### line_detect.py

Generic ink color detection with foxing discrimination. Used by Stages 2
(orientation), 6 (content area), 7 (deskew), and 8 (dewarp).

```python
def detect_ink_mask(img: np.ndarray, cfg: Config) -> np.ndarray:
    """Isolate staff-line ink pixels using auto-detected color thresholds.
    
    Primary: HSV range from cfg (hue center +/- range, min sat, min value).
    Fallback: channel-difference method if HSV produces too few pixels.
    """

def detect_ink_mask_geometric(img: np.ndarray, cfg: Config) -> np.ndarray:
    """Ink mask filtered for line-like geometry (R9 foxing discrimination).
    
    1. detect_ink_mask() for color
    2. Morphological opening with 1x30 horizontal kernel
    3. Connected components: keep aspect_ratio > 5:1
    Removes foxing spots while preserving staff lines.
    """

def detect_staff_lines(img, cfg) -> list[StaffLine]:
    """Detect staff lines using geometric ink mask.
    Each StaffLine has: y_center, points, polynomial_coeffs, angle."""

def detect_dominant_angle(img, cfg) -> float:
    """Median angle of detected staff lines in degrees."""

def detect_illustration_regions(img, cfg) -> np.ndarray:
    """R4: Mask multi-colored regions (high local hue variance).
    Used to exclude illustrations from line detection."""
```

### geometry.py

```python
def order_corners(pts) -> np.ndarray: ...
def compute_target_size(corners) -> tuple[int, int]: ...
def get_perspective_transform(src, dst) -> np.ndarray: ...
```

### image_utils.py

Shared image utilities used by multiple geometric stages (7, 8) and
potentially other stages that need background estimation or content trim.

```python
def estimate_background(img: np.ndarray, border_frac: float = 0.05) -> tuple[int, int, int]:
    """Estimate background color from border pixels (outermost border_frac).

    Returns BGR color as a 3-tuple. Uses median of border pixel values
    for robustness against content near the edges.
    Used by: Stage 5 (perspective), Stage 7 (deskew), Stage 8 (dewarp).
    """

def trim_to_content(
    img: np.ndarray,
    bg_color: tuple[int, int, int] | None = None,
    margin_frac: float = 0.02,
    threshold: int = 30,
) -> np.ndarray:
    """Trim background-colored borders and add uniform margin padding.

    1. Estimate bg_color if not provided (via estimate_background).
    2. Threshold: pixels with L1 distance > threshold from bg_color.
    3. Find bounding rect of non-background region.
    4. Crop to bounding rect.
    5. Add uniform margin (margin_frac * width) filled with bg_color.
    Used by: Stage 7 (deskew), Stage 8 (dewarp).
    """
```

### image_io.py

```python
def load_image(path, cfg) -> tuple[np.ndarray, dict]:
    """Load image, extract and return EXIF metadata, apply EXIF rotation.
    
    Returns (image, exif_dict) where exif_dict contains:
    - orientation, camera_model, datetime, dpi, focal_length, etc.
    EXIF is extracted via Pillow before converting to numpy array.
    No color correction here -- R3 lives in Stage 9.
    Accepts JPEG, PNG, TIFF, or any format Pillow supports.
    """

def save_checkpoint(img, stage_dir, filename, metadata=None): ...
    """Save image as PNG (lossless) with atomic write.
    
    1. Write to {filename}.tmp in the stage directory
    2. Atomic rename to {filename}.png
    3. Write metadata sidecar to {filename}.json if metadata provided
    This prevents corrupt files if the process is interrupted.
    Filename extension is always .png regardless of input format.
    """

def ensure_checkpoint_dir(output_dir, stage_name) -> Path: ...
```

### accel.py

```python
def has_opencl() -> bool: ...
def has_openvino_gpu() -> bool: ...
def to_umat(img) -> cv2.UMat: ...
def from_umat(umat) -> np.ndarray: ...
def gpu_canny(img, low, high) -> np.ndarray: ...
def gpu_clahe(gray, clip, grid) -> np.ndarray: ...
def gpu_remap(img, map_x, map_y) -> np.ndarray: ...
```

### preprocess.py

```python
def remove_hotspots(img, cfg) -> tuple[np.ndarray, dict]: ...
def detect_fingers(img, cfg) -> np.ndarray: ...
def remove_fingers(img, mask, cfg) -> np.ndarray: ...
```

---

## Config (`config.py`)

### Stage Optionality

Each stage is classified as mandatory, auto-conditional, or optional:

| Stage | Name | Class | Default | Skip condition |
|-------|------|-------|---------|----------------|
| 0 | Preprocess | auto | skip | No hotspots or fingers detected by analyze |
| 1 | Stitch | auto | skip | No partial photo groups detected |
| 2 | Orientation | **mandatory** | on | Always runs (EXIF at minimum) |
| 3 | Lens correct | auto | skip | No distortion detected (k1 == k2 == 0) |
| 4 | Page detect | **mandatory** | on | Always runs (everything downstream needs it) |
| 5 | Perspective | **mandatory** | on | Always runs (produces rectangle) |
| 6 | Content area | optional | on | `skip_content_area = true` in book.toml |
| 7 | Deskew | optional | on | `skip_deskew = true` or angle < 0.1 degrees |
| 8 | Dewarp | optional | on | `skip_dewarp = true` or no staff lines + no AI |
| 9 | Enhance | optional | on | `skip_enhance = true` (sub-steps also toggleable) |
| 10 | Normalize | optional | on | `skip_normalize = true` |
| 11 | OCR | optional | on | `--no-ocr` flag or `skip_ocr = true` |
| 12 | PDF assembly | **mandatory** | on | Always runs (it's the output) |

- **Mandatory** stages cannot be skipped (pipeline produces incorrect output without them).
- **Auto** stages have built-in skip logic: they check a condition and pass through
  unchanged if not needed. No user configuration required.
- **Optional** stages default to on but can be disabled per-book or via CLI.

### Profiles (CLI shortcut `--profile NAME`)

Named presets for common workflows. The user can override individual
settings on top of a profile.

```toml
# book.toml
[pipeline]
profile = "full"   # default
```

| Profile | Description | Stages enabled | Use case |
|---------|-------------|----------------|----------|
| `full` | All stages, all enhancements | 0-12 (auto-skips apply) | Production-quality output |
| `geometry` | Flatten and crop only, no color processing | 0-5, 12 | Quick structural fix, user handles color elsewhere |
| `clean` | Geometry + enhancement, no OCR | 0-10, 12 | High-quality PDF without OCR overhead |
| `quick` | Geometry + light enhance, no dewarp/OCR | 0-5, 9 (denoise+sharpen only), 12 | Fast preview to check framing |

### Sub-step toggles for Stage 9 (enhance)

Each enhancement sub-step has an independent bool flag (already defined
in stage parameters). The profile or book.toml can disable specific
sub-steps without disabling the whole stage:

```toml
[enhance]
color_cast_correction = true     # R3
illumination_normalization = true
shadow_correction = true         # R5
stain_correction = true          # R6
halo_reduction = true            # R10
show_through_removal = true
white_balance = true
clahe = true
salt_correction = true           # R11
denoise = true
sharpen = true
binarize = false                 # only if explicitly requested
```

### Config dataclass

```python
@dataclass
class Config:
    input_dir: Path
    output_dir: Path = None       # default: {input_dir}_output/ (computed at init)
    profile: str = "full"
    preview: int = 0
    use_gpu: bool = True
    ai_dewarp: bool = False
    binarize: bool = False

    # Stage skip overrides (optional stages only)
    skip_content_area: bool = False
    skip_dewarp: bool = False
    skip_deskew: bool = False
    skip_enhance: bool = False
    skip_normalize: bool = False
    skip_ocr: bool = False

    # Error handling, cleanup, logging
    on_error: str = "skip"         # "skip" (per-stage classes), "stop", "force"
    cleanup: bool = False          # delete intermediate checkpoints after success
    keep_stages: list[str] = None  # if set, only keep these stage dirs
    verbose: bool = False
    quiet: bool = False

    # Book characteristics (from book.toml via analyze)
    staff_color_hue: int = 5
    staff_color_range: int = 15
    staff_saturation_min: int = 40
    staff_value_min: int = 80
    channel_diff_rg: int = 30
    channel_diff_rb: int = 30
    has_border_frame: bool = True
    page_number_position: str = "top-right"
    expected_staff_lines: int = 16

    # OCR
    ocr_engine: str = "tesseract"
    ocr_lang: str = "lat"

    # ... all stage-specific params from sections above ...
```

Loading priority: **CLI args > book.toml > profile defaults > built-in defaults**

---

## CLI Interface (`cli.py`)

```
lpacleaner run INPUT_DIR                           # just works
lpacleaner run INPUT_DIR -o OUTPUT_DIR             # explicit output dir
lpacleaner run INPUT_DIR --profile geometry        # flatten only
lpacleaner run INPUT_DIR --profile clean           # no OCR
lpacleaner run INPUT_DIR --profile quick           # fast preview
lpacleaner run INPUT_DIR --skip-dewarp --skip-ocr  # selective skip
lpacleaner run INPUT_DIR --stages 2,3,4,5          # explicit stage list (advanced)
lpacleaner run INPUT_DIR --preview 5               # process only 5 images
lpacleaner run INPUT_DIR --ai-dewarp
lpacleaner run INPUT_DIR --binarize
lpacleaner run INPUT_DIR --cleanup                 # delete intermediates after success
lpacleaner run INPUT_DIR --on-error stop           # halt on first failure
lpacleaner run INPUT_DIR --verbose                 # per-image debug output
lpacleaner run INPUT_DIR --quiet                   # warnings and errors only
lpacleaner analyze INPUT_DIR [-o OUTPUT_DIR] [--samples 15]
lpacleaner inspect IMAGE_PATH [--config book.toml]
lpacleaner review OUTPUT_DIR [--stage 09_enhanced]
lpacleaner compare OUTPUT_DIR                       # full-book HTML stage comparison (local)
lpacleaner compare OUTPUT_DIR IMG_0012              # open at specific image
lpacleaner compare OUTPUT_DIR --no-open             # generate without opening browser
lpacleaner compare OUTPUT_DIR --input-dir /path/to/originals
lpacleaner publish OUTPUT_DIR /var/www/lpa1         # publish with downscaled JPEGs
lpacleaner publish OUTPUT_DIR pub --stages "0,5,7"  # subset of stages
lpacleaner cleanup OUTPUT_DIR [--keep 07,09]        # delete intermediates post-hoc
```

### `compare` command

Generates a single interactive HTML viewer for the entire book, showing
all images across all executed pipeline stages.  Purely local: uses
`file://` references to existing PNGs on disk (no copies, no conversion),
so the HTML is lightweight (~400 KB for a 225-image book) and images
load on demand.

The output directory is scanned for all `NN_*` checkpoint directories
that actually exist (only stages that were run are included); the
original input is auto-detected from the `<input>_output` naming
convention (override with `--input-dir`).

The comparison HTML is also **automatically generated** at the end of
every `lpacleaner run` invocation.

The viewer has a **dark blue theme** and displays a "Compare mode"
badge in the top bar.

Keyboard shortcuts (shared with `publish` viewer):

| Key | Action |
|-----|--------|
| ← → | Previous / next stage (same image) |
| PgUp / PgDn | Previous / next image (resets to stage 0) |
| S | Toggle side-by-side mode |
| M | Toggle metadata panel (sidecar JSON) |
| Z | Toggle 100% zoom |
| 1-9 | Jump directly to stage N |

The top bar shows the current image name, a counter (e.g. "IMG_0050 --
40 / 225"), Prev/Next buttons, and a dropdown to jump to any image.

Side-by-side mode lets you pick a different stage in each pane for
direct visual comparison.  The metadata panel shows the JSON sidecar
(e.g. skew angle, detection method, page type).

### `publish` command

Generates a self-contained directory with downscaled JPEG thumbnails,
suitable for uploading to any static web host (GitHub Pages, Netlify,
an S3 bucket, etc.).  Uses relative paths so the whole folder can be
uploaded as-is.

The viewer has a **warm amber theme** and displays a "Published
YYYY-MM-DD HH:MM UTC" badge in the top bar, so it's immediately
distinguishable from the local `compare` viewer.

```bash
# Publish all stages at default settings (1500px max, quality 85):
lpacleaner publish OUTPUT_DIR /var/www/lpa1

# Publish only specific stages, smaller images for bandwidth:
lpacleaner publish OUTPUT_DIR ./pub \
    --stages "0,5,7" \
    --max-dim 1000 \
    --quality 80
```

Options:

| Option | Default | Description |
|--------|---------|-------------|
| `PUBLISH_DIR` | (required) | Target directory for the published site |
| `--max-dim` | 1500 | Max pixel dimension (longest side) for JPEGs |
| `--quality` | 85 | JPEG compression quality (1-100) |
| `--stages` | all | Comma-separated stage numbers to include |
| `--input-dir` | auto | Override original input directory |

Output structure:

```
PUBLISH_DIR/
├── index.html           # self-contained viewer (relative paths)
└── images/
    ├── original/        # input images
    │   ├── IMG_0011.jpg
    │   └── ...
    ├── 00_preprocessed/
    ├── 05_perspective/
    └── 07_deskewed/
```

### Workflow for a new book

```bash
# Simplest usage -- everything automatic:
lpacleaner run "/path/to/book/photos"
# Auto-detects book characteristics, processes all pages, produces PDF.
# Output goes to /path/to/book/photos_output/

# Advanced workflow -- review config before processing:
lpacleaner analyze "/path/to/book/photos"
# Review and edit the generated book.toml
nano "/path/to/book/photos_output/book.toml"
lpacleaner run "/path/to/book/photos"

# Quick preview to check framing before full run:
lpacleaner run "/path/to/book/photos" --profile quick --preview 5

# Review flagged pages after processing:
lpacleaner review "/path/to/book/photos_output"

# Browse all images across all stages (auto-generated after run):
lpacleaner compare "/path/to/book/photos_output"

# Open at a specific image:
lpacleaner compare "/path/to/book/photos_output" IMG_0012

# Publish a web-friendly version for sharing with researchers:
lpacleaner publish "/path/to/book/photos_output" /var/www/lpa1-compare \
    --stages "0,5,7" \
    --max-dim 1000
```

---

## Parameter Tuning and Diagnostics

### How to Detect Parameter Problems

Each stage computes per-image confidence metrics. The pipeline aggregates
these into stage-level statistics in the end-of-run report:

```
Stage  8 (dewarp): 210/225 OK, 12 soft fallback (< 2 staff lines), 3 AI fallback
Stage  9 (enhance): 222/225 OK, 3 flagged (high noise residual)
```

**Thresholds for review recommendations:**
- If >20% of pages trigger a soft fallback in any stage, the end-of-run
  report emits a `REVIEW` recommendation with specific advice:
  *"Staff line detection had low confidence on 48 pages. Run
  `lpacleaner inspect IMG_0045.jpg --stage dewarp` to visualize,
  then consider adjusting staff_color_hue in book.toml."*

### Diagnostic Visualization (`lpacleaner inspect`)

The `inspect` command renders annotated overlays for each stage:

```bash
lpacleaner inspect IMG_0045.jpg --stage dewarp --config book.toml
```

Output: an annotated image showing:
- Ink mask overlay (detected ink pixels in green)
- Staff lines drawn as colored polylines
- Polynomial curves fitted to each staff line
- Foxing spots circled in red (filtered out by R9)
- Page quad drawn in blue (from Stage 4)
- Detection parameters used and match counts

For ink detection, `inspect` also suggests parameter corrections:
*"Current: staff_color_hue=5 range=15. Histogram peak at hue=18.
Suggest: staff_color_hue=18, staff_color_range=12."*

### Parameter Adjustment Workflow

```
1. First run with defaults (or after analyze):
   lpacleaner run "/path/to/book/photos"

2. Review quality:
   lpacleaner review "/path/to/book/photos_output"
   → Shows flagged pages, confidence stats, contact sheet

3. Diagnose specific problems:
   lpacleaner inspect IMG_0045.jpg --stage dewarp
   → Shows detection overlay with parameter suggestions

4. Adjust configuration:
   nano "/path/to/book/photos_output/book.toml"
   → Edit staff_color_hue, thresholds, etc.

5. Re-run (only affected stages reprocess -- cache invalidation):
   lpacleaner run "/path/to/book/photos"
   → Only stages dependent on changed params are re-run
```

This workflow leverages the config-aware cache invalidation: changing
`staff_color_hue` in book.toml automatically invalidates stages 2, 6,
7, 8 and all downstream, without reprocessing stages 0-1 or 3-5.

### Adaptive Polynomial Fitting

Staff line polynomial fitting uses adaptive degree selection to avoid
ill-conditioned fits (numpy RankWarning) on tightly clustered segments:

```
1. Fit degree-1 (linear) -- handles skew
2. Compute RMS residual
3. If residual > 1.5px: try degree-2 -- handles spine curvature
4. If residual still > 1.5px: try degree-3 -- handles severe warping
5. Stop at the lowest degree with acceptable residual
```

Rationale (conservation perspective): Gregorian chant uses 4-line staves
(tetragram) drawn with a rastrum. On flat parchment these are remarkably
straight; curvature only appears from spine binding. A degree-1 fit
suffices for 80%+ of pages. Degree-3 is reserved for severely warped
pages near the spine.

---

## Error Handling

### Per-Image Error Policy

When a single image fails in any stage, the pipeline must not crash.
Failures are isolated per-image: the remaining images continue processing.

```
Error categories:

1. Soft failure (stage-internal fallback):
   - Example: dewarp finds no staff lines, falls back to passthrough
   - Handled internally by the stage (fallback paths already in plan)
   - Logged as WARNING, image flagged in pipeline.json
   - No user intervention needed

2. Hard failure (unrecoverable for this image in this stage):
   - Example: corrupt JPEG, cv2 exception, degenerate polygon
   - Caught by the pipeline orchestrator's per-image try/except
   - Logged as ERROR with full traceback
   - What happens next depends on the stage's error class (see below)

3. Fatal failure (pipeline cannot continue):
   - Example: output disk full, permissions error, all images fail
   - Pipeline stops with a clear error message
   - pipeline.json records the last successful state for resume
```

### Per-Stage Error Classes

Each stage has a built-in error class that determines what happens
when a hard failure occurs:

| Stage | Name | Error class | On failure |
|-------|------|-------------|------------|
| 0 | Preprocess | skippable | Pass through original image |
| 1 | Stitch | skippable | Use best single image from group |
| 2 | Orientation | critical | **Exclude image** from pipeline |
| 3 | Lens correct | skippable | Pass through uncorrected image |
| 4 | Page detect | critical | **Exclude image** from pipeline |
| 5 | Perspective | critical | **Exclude image** from pipeline |
| 6 | Content area | skippable | Pass through with fixed 5% inset |
| 7 | Deskew | skippable | Pass through unskewed image |
| 8 | Dewarp | skippable | Pass through undistorted image |
| 9 | Enhance | skippable | Pass through unenhanced image |
| 10 | Normalize | skippable | Pass through unnormalized image |
| 11 | OCR | skippable | No text layer for this page |
| 12 | PDF assembly | fatal | Pipeline stops |

- **skippable**: on hard failure, carry forward the image from the last
  successful stage. The image appears in the PDF but without this
  stage's processing. Logged as ERROR, flagged for review.
- **critical**: the image cannot produce a usable result without this
  stage (no valid crop, no valid rectangle). The image is excluded from
  all subsequent stages and from the PDF. Logged as ERROR, reported in
  end-of-run summary.
- **fatal**: the pipeline cannot produce any output. Stop immediately.

### CLI Override

```
--on-error skip    (default) use per-stage error classes as above
--on-error stop    halt on ANY hard failure in ANY stage (for debugging)
--on-error force   treat all stages as skippable (never exclude images,
                   carry forward whatever exists -- for maximum output)
```

Per-stage override in book.toml for edge cases:
```toml
[error_overrides]
stage_5 = "skippable"   # don't exclude images that fail perspective
stage_8 = "critical"    # I want perfect dewarping or nothing
```

### End-of-Run Report

```
Processed 222/225 images.
  2 flagged (soft fallback): IMG_0045 (low focus), IMG_0013 (no staff lines)
  1 excluded (critical failure): IMG_0080 (Stage 4: no page quad found)
  0 failed stages: all skippable failures recovered

Output: /path/to/output/output.pdf (222 pages, 185 MB)
Log: /path/to/output/lpacleaner.log
```

---

## Disk Space Management

### Checkpoint Storage Estimates

At 12MP (4000×3000), each PNG checkpoint is ~35MB. Per book:

| Stages kept | Images | Size per book | 15 books |
|-------------|--------|---------------|----------|
| All 13 | 225 | ~100 GB | ~1.5 TB |
| Final only (10_normalized) | 225 | ~8 GB | ~120 GB |
| Final + 3 key stages | 225 | ~30 GB | ~450 GB |

Default: keep all checkpoints (enables resume and debugging).
The CLI warns about estimated disk usage before processing starts:

```
"Estimated disk usage: ~105 GB for 225 images × 13 stages.
 Available: 230 GB. Proceed? [Y/n]"
```

### Cleanup Options

```bash
# Remove intermediate checkpoints after successful completion,
# keeping only 10_normalized/ (final images) and output.pdf
lpacleaner run INPUT_DIR --cleanup

# Keep only specific stages (for debugging a particular stage)
lpacleaner run INPUT_DIR --keep-stages 02,07,09,10

# Clean up a completed run after the fact
lpacleaner cleanup OUTPUT_DIR              # keeps 10_normalized + PDF
lpacleaner cleanup OUTPUT_DIR --keep 07,09 # keeps specific stages too
```

`--cleanup` deletes each stage's checkpoint directory after the next
stage completes successfully for ALL images in that stage. This means
resume goes back to the kept stages, not to the deleted intermediate.

---

## Logging

### Design

Structured logging via Python's `logging` module. Two outputs:

1. **Console** (stderr): human-readable, colorized, progress-focused.
   Controlled by `--verbose` / `--quiet`.
2. **Log file** (`output/lpacleaner.log`): machine-parseable, always
   verbose, includes timestamps and per-image details. Rotated at 50MB.

### Verbosity Levels

| CLI flag | Console level | What's shown |
|----------|---------------|--------------|
| `--quiet` | WARNING | Only errors and warnings |
| (default) | INFO | Stage progress, summary stats, flagged pages |
| `--verbose` | DEBUG | Per-image timing, parameter values, fallback decisions |

### Log Format

Console (default):
```
[Stage 2] Orienting images... 225/225 [00:12, 18.7 img/s]
[Stage 2]   3 images flagged: IMG_0080 (low focus: 45.2)
[Stage 8] Dewarping images... 222/222 [01:45, 2.1 img/s]
[Stage 8]   210 dewarped (staff lines), 12 passthrough (text pages)
[Stage 8]   ERROR: IMG_0080.JPG skipped (ValueError in polynomial fit)
```

Console (verbose):
```
[Stage 8] IMG_0011.JPG: 14 staff lines, poly R²=0.997, 0.42s
[Stage 8] IMG_0012.JPG: 16 staff lines, poly R²=0.999, 0.38s
[Stage 8] IMG_0013.JPG: 0 staff lines, passthrough (text page), 0.02s
```

Log file (always):
```
2026-07-04 04:30:12.345 INFO  stage=2 image=IMG_0011.JPG action=oriented method=exif+staff_lines angle=0 focus=342.5 elapsed_ms=55
2026-07-04 04:30:12.400 INFO  stage=2 image=IMG_0012.JPG action=oriented method=exif+staff_lines angle=90 focus=287.3 elapsed_ms=62
2026-07-04 04:30:15.123 WARN  stage=2 image=IMG_0080.JPG action=flagged reason=low_focus focus=45.2 threshold=100
2026-07-04 04:32:45.678 ERROR stage=8 image=IMG_0080.JPG action=failed error="ValueError: polynomial fit singular matrix" traceback="..."
```

### Per-Stage Summary

After each stage completes, a summary line is logged at INFO level:

```
[Stage 8] Complete: 210/222 dewarped, 12 passthrough, 0 failed. Total: 1m45s
```

This is also written to `pipeline.json` for programmatic access.

---

## Robustness Features Summary

| ID | Feature | Where | Default | Auto-detected |
|----|---------|-------|---------|---------------|
| R1 | Flash hotspot removal | Stage 0 / preprocess.py | Off (auto-enable) | Yes |
| R2 | Robust page detection fallback | Stage 4 | On (fallback chain) | Method: Yes |
| R3 | Color cast correction | Stage 9 / enhance.py | On | Severity: Yes |
| R4 | Illustration region exclusion | line_detect.py | On | Presence: Yes |
| R5 | Sharp shadow removal | Stage 9 | On | Severity: Yes |
| R6 | Stain-aware enhancement | Stage 9 | On | Severity: Yes |
| R7 | Lens distortion correction | Stage 3 | Off (auto-enable) | Yes |
| R8 | Finger/hand detection | Stage 0 / preprocess.py | Off (manual enable) | Yes |
| R9 | Foxing/rust discrimination | line_detect.py | On (geometric) | Severity: Yes |
| R10 | Iron gall ink halo reduction | Stage 9 | On | Severity: Yes |
| R11 | Salt/efflorescence correction | Stage 9 | On | Severity: Yes |

---

## AI/ML Strategy

### Current Approach: Classical Computer Vision

The pipeline is deliberately built on classical computer vision, not
deep learning. Every stage uses deterministic, interpretable algorithms:

| Technique | Used in | Purpose |
|-----------|---------|---------|
| Otsu / adaptive thresholding | Stage 4 | Page-background separation |
| Canny edge detection | Stage 4 (fallback) | Edge-based page detection |
| GrabCut | Stage 4 (fallback) | Foreground segmentation |
| Hough Line Transform | Stages 2, 6, 7, 8 | Staff line and border detection |
| HSV color filtering | line_detect.py | Ink color isolation |
| Morphological operations | Multiple stages | Mask cleaning, gap bridging |
| Polynomial fitting | Stage 8 | Staff line curvature modeling |
| Perspective transform | Stage 5 | Geometric correction |
| cv2.remap | Stage 8 | Dewarping via displacement mesh |
| CLAHE | Stage 9 | Adaptive contrast enhancement |
| ORB feature matching | Stage 1 | Stitch group detection |
| cv2.Stitcher | Stage 1 | Panoramic stitching (ORB-based, not neural) |
| cv2.inpaint (Telea) | Stage 0 | PDE-based inpainting for hotspots/fingers |
| Laplacian variance | Stage 2 | Focus quality metric |

### Why Classical First

1. **Deterministic**: Same input always produces same output. No model
   randomness, no batch normalization artifacts, no training data bias.
2. **Debuggable**: Every parameter is interpretable. When dewarp produces
   a bad result, you can inspect the polynomial coefficients, the staff
   line positions, the Hough thresholds. Neural models are opaque.
3. **Fast**: No model loading, no GPU memory allocation for inference,
   no batch size tuning. A Hough transform on a 4000x3000 image takes
   ~10ms. A neural segmentation model takes ~200ms+ after loading.
4. **No training data required**: These books are rare. There's no
   labeled dataset of "correct dewarping for 300-year-old Gregorian
   chant." Classical methods work from first principles.
5. **No model distribution**: No 500MB model files to download, no
   version compatibility issues, no ONNX/OpenVINO conversion steps.
6. **Runs everywhere**: CPU-only fallback works. No GPU required (GPU
   accelerates but isn't mandatory).

### Where AI/ML Is Used (Optional)

Only one place, only when explicitly requested:

- **Stage 8 dewarp, AI path** (`--ai-dewarp`): DocTr GeoTr model,
  converted to OpenVINO IR (FP16), runs on Intel Arc GPU. Used as
  fallback when classical dewarping finds fewer than 2 staff lines
  (text-only pages, decorative pages). Requires `openvino` and
  `torch` packages (optional dependency group `ai`).

### Where AI/ML Could Add Value (Not Currently Implemented)

These are potential future enhancements, ordered by impact:

1. **Kraken OCR** (Stage 11, already in plan as optional):
   Neural OCR for historical scripts. Much better than Tesseract on
   300-year-old handwritten Latin with abbreviations and ligatures.
   Available via `--ocr-engine kraken` and the `historical-ocr`
   dependency group. This is the highest-value ML addition.

2. **Neural binarization** (Stage 9):
   Models like Robin or DE-GAN produce cleaner binarization on
   degraded documents than adaptive thresholding. Useful if
   binarized output is needed for OMR or archival purposes.
   Low priority since most users want color output.

3. **Page segmentation** (Stage 4):
   A U-Net or similar model could segment page vs. background more
   reliably than Otsu/Canny on difficult backgrounds (textured
   tablecloths, cluttered surfaces). Not needed for the current
   books (dark table background with good contrast), but could help
   with varied photography conditions across 15+ books.

4. **Layout analysis**:
   Neural layout detection (staff regions, text regions, illustrations,
   marginalia) could replace the current heuristic approach (ink mask
   + Hough lines + aspect ratio filtering). Would primarily benefit
   OMR (see below) by providing precise region boundaries.

---

## Future: Optical Music Recognition (OMR)

OMR is a planned future capability: converting the cleaned, dewarped
music page images into machine-readable music notation (MEI, MusicXML,
or similar).

### Why It's Not in v1

OMR requires:
1. Very clean, well-dewarped, properly oriented input images
2. Accurate staff line detection and removal
3. Symbol segmentation and classification (neural network)
4. Musical context understanding (rhythm, key, clef awareness)

Items 1-2 are exactly what lpacleaner produces. Building the image
pipeline first gives OMR the best possible input. Attempting OMR on
raw, skewed, warped, poorly-lit photos would produce poor results.

### How lpacleaner Prepares for OMR

Several design decisions in the current pipeline are intentionally
OMR-friendly:

1. **Lossless checkpoints (PNG)**: OMR needs clean pixel data, not
   JPEG-compressed artifacts around note heads.
2. **Staff line detection metadata**: Stage 8 stores staff positions,
   polynomial coefficients, and cluster assignments. OMR needs exactly
   this data to locate staves.
3. **Page type classification**: Stage 4 classifies pages as "music",
   "text", "decorative", etc. OMR only processes "music" pages.
4. **Dewarped staff lines**: After Stage 8, staff lines are straight
   and horizontal -- the ideal input for OMR symbol detection.
5. **Notation masking** (Stage 11): The OCR stage already masks staff
   regions. The inverse mask (notation regions only) is exactly what
   OMR needs.
6. **Ink mask / geometric filtering**: The ink detection pipeline
   (line_detect.py) separates staff lines from notes, text, and
   damage. This segmentation is a prerequisite for OMR.

### Planned OMR Architecture (Future Iteration)

Two approaches, in order of feasibility:

#### Approach A: End-to-End Vision-Encoder-Decoder (Recommended)

Inspired by [Transcoda](https://huggingface.co/btrkeks/transcoda-59M-zeroshot-v1),
a 59M-parameter model that does end-to-end OMR for modern notation using a
ConvNeXt-V2 encoder + Transformer decoder. Transcoda itself won't work for
Gregorian chant (trained on 5-line modern staves, outputs `**kern`), but the
same architecture can be trained from scratch on square notation:

```
Encoder: ConvNeXt-V2 or similar (pretrained ImageNet, fine-tuned)
  ↓ patch grid
Projector: MLP (encoder dim → decoder dim)
  ↓
Decoder: Transformer (8 layers, BPE vocabulary over GABC tokens)
  ↓
Output: GABC notation (Gregorio format)
```

**Why GABC, not MEI or MusicXML**: GABC is the native notation format for
Gregorian chant, used by the [Gregorio project](https://gregorio-project.github.io/).
It's compact, human-readable, and directly renders via LaTeX. MEI is more
general but overkill for square notation.

**Training data strategy -- synthetic generation, not manual transcription:**

The 225 LPA-1 photographs (and photos from other books) are NOT suitable for
training. Training a vision-encoder-decoder requires thousands of (image, GABC)
pairs. Manual transcription of 225 pages would take weeks and still be far too
few (Transcoda used 310,000+ synthetic pairs). Instead:

1. **Source GABC files**: [GregoBase](https://gregobase.selapa.net/) contains
   ~10,000 chant transcriptions in GABC format, contributed by scholars.
   Additional sources: Gregorio project samples, GABC files from chant
   communities.
2. **Render to images**: Use [Gregorio](https://gregorio-project.github.io/)
   (a TeX package) to typeset each GABC file into a clean score image. This
   produces pixel-perfect (image, GABC) pairs automatically.
3. **Domain augmentation**: Apply transformations to make clean renders
   resemble real photographs:
   - Parchment/paper texture overlay
   - Red staff ink with variable hue/saturation (matching real books)
   - Ink bleeding, fading, and thickness variation
   - Foxing spots, water stains, iron gall corrosion simulation
   - Camera perspective distortion, barrel distortion
   - Uneven lighting, shadows, flash hotspots
   - Background (dark table surface)
   - JPEG compression artifacts
   These augmentations bridge the domain gap between synthetic renders and
   real photographs.
4. **Validation/test set from real photos**: Manually transcribe 20-30 pages
   from each real book into GABC to measure real-world performance (domain
   gap). The real photos are too precious to train on -- they're the ground
   truth benchmark.

**Model sizing**: A compact model (50-100M parameters) should suffice.
Square notation has a much smaller symbol vocabulary than modern notation
(~30 neume types vs hundreds of symbols), so the decoder can be smaller.

**Estimated effort**: Significant -- data pipeline (GABC → rendered images),
augmentation engine, model training, grammar-constrained decoding for valid
GABC output. This is a standalone research/engineering project.

#### Approach B: Classical Pipeline (Segmentation + Classification)

Fallback if end-to-end proves too complex:

```
Input: dewarped images from Stage 8 + staff line metadata

1. Staff line removal: use Stage 8 staff positions to precisely
   remove staff lines, leaving only notes, clefs, accidentals, text
2. Symbol segmentation: connected components or neural detection
   (YOLO-based or similar) to isolate individual symbols
3. Symbol classification: CNN classifier trained on Gregorian chant
   notation (square notation: neumes, punctum, virga, clivis, etc.)
4. Sequence assembly: left-to-right, top-to-bottom reading order
   using staff line positions as vertical reference
5. Output: GABC notation
```

This is more brittle (errors cascade between stages) but easier to debug
and doesn't require as much training data.

#### Existing Tools (Limited Applicability)

- **Transcoda** (btrkeks/transcoda-59M-zeroshot-v1): Modern notation only,
  outputs `**kern`. Architecture is reusable, weights are not.
- **OMMR4all**: End-to-end OMR for historical manuscripts, but focused on
  mensural notation (white notation), not square notation.
- **Rodan/Gamera**: Legacy framework, not actively maintained.
- **Kraken**: Already in our pipeline for text OCR. Could potentially be
  trained for neume recognition but not designed for music.

### OMR Impact on Current Design

No changes needed to the current pipeline for OMR readiness. The
architecture is already compatible:
- OMR would be a new Stage 14 (or a separate command `lpacleaner omr`)
- It consumes Stage 8 output (dewarped images) and Stage 8 metadata
  (staff positions)
- It runs after the image pipeline, optionally in parallel with
  enhancement/OCR
- It produces its own output files (GABC) alongside the PDF
- The augmentation engine for synthetic training data could reuse
  lpacleaner's own image processing stages (enhance, normalize) in reverse

---

## Resumability and Parallelism

### Per-Image Resume

The pipeline resumes at per-image granularity, not per-stage. If
interrupted mid-stage (Ctrl+C, crash, power loss), only incomplete
images are reprocessed on the next run.

```
Resume algorithm:
1. For each stage in the pipeline:
   a. List images that need processing (from previous stage's output)
   b. For each image, check if output already exists in this stage:
      - Output file exists AND is a valid PNG (file size > 0,
        header bytes are PNG magic): skip (already complete)
      - Output file does not exist: process
      - Output file exists but is invalid (truncated, zero bytes,
        no PNG header): delete and reprocess
   c. Process only the missing/invalid images
2. Stage is considered complete when all images have valid outputs

Atomic writes:
  save_checkpoint() writes to {filename}.tmp, then os.rename() to
  {filename}.png. On POSIX, rename is atomic -- the file either
  exists fully or doesn't. A crash during write leaves only the
  .tmp file, which is cleaned up on the next run.

Config-aware invalidation:
  Each stage declares which config fields it depends on. When the
  pipeline starts, it hashes the relevant config fields per stage
  and compares with the hashes stored in pipeline.json from the
  previous run.

  Stage dependency map (stored in code, not config):
    Stage 0:  [hotspot_*, finger_*]
    Stage 1:  [stitch_*]
    Stage 2:  [orient_*, staff_color_*, focus_*]
    Stage 3:  [lens_*]
    Stage 4:  [page_detect_*]
    Stage 5:  (none -- only depends on Stage 4 corners)
    Stage 6:  [content_*, staff_color_*, has_border_frame]
    Stage 7:  [deskew_*, staff_color_*]
    Stage 8:  [dewarp_*, staff_color_*, ai_dewarp]
    Stage 9:  [enhance_*, color_cast_*, shadow_*, stain_*, halo_*, salt_*]
    Stage 10: [normalize_*]
    Stage 11: [ocr_*]
    Stage 12: [pdf_*]

  If any field in a stage's dependency set has changed since the last
  run, ALL images in that stage are invalidated AND all downstream
  stages are invalidated too (since they consume this stage's output).

  Example: changing staff_color_hue invalidates stages 2, 6, 7, 8
  and everything downstream (9, 10, 11, 12).

  pipeline.json stores:
    "config_hashes": {
      "stage_2": "a1b2c3...",
      "stage_7": "d4e5f6...",
      ...
    }

Force re-run:
  --force-stage N   reprocesses all images in stage N (and downstream)
  --force-all       reprocesses everything from scratch
  Deleting a checkpoint directory also forces re-run of that stage.
```

### Adaptive Parallelism

Worker count auto-scales to available RAM, not just CPU count.

```
1. Detect available RAM: psutil.virtual_memory().available
   (or read /proc/meminfo on Linux if psutil unavailable)

2. Estimate memory per worker:
   - Base: ~150MB (Python process, loaded libraries)
   - Per image: width × height × 3 × 2 (input + output buffers)
     For 4000×3000: ~72MB per image
   - Stage overhead: dewarp mesh, morphological kernels, etc. ~50MB
   - Total per worker: ~270MB for 12MP images

3. max_workers = min(
       cpu_count // 2,                    # leave headroom for GPU
       available_ram_mb // mem_per_worker, # don't exhaust RAM
       cfg.max_workers                     # user override
   )

4. Special cases:
   - Stage 1 (stitch): single-threaded for stitch groups (high
     memory, internally parallelized by cv2.Stitcher), parallel
     for standalone images
   - Stage 10 (normalize): two-pass -- first pass collects stats
     (parallel), second pass applies normalization (parallel)
   - Stage 12 (PDF): single-threaded (sequential assembly)

5. Default --workers 0 = auto-detect. Explicit --workers N overrides.
```

### Pipeline State (`pipeline.json`)

```json
{
  "book": "LPA-1 San Nicolas",
  "started": "2026-07-04T04:30:00",
  "stages": {
    "0": {"status": "complete", "processed": 225, "skipped": 220},
    "1": {"status": "complete", "processed": 225, "stitched_groups": 1},
    "2": {"status": "complete", "processed": 222, "flagged": ["IMG_0080"]},
    "7": {"status": "running", "processed": 140, "total": 222}
  },
  "exif": {
    "IMG_0011.JPG": {"camera": "Canon PowerShot SX200 IS", "dpi": 180, ...}
  },
  "focus_scores": {
    "IMG_0011.JPG": 342.5,
    "IMG_0080.JPG": 45.2
  },
  "flags": ["IMG_0080.JPG: low focus score (45.2 < 100)"]
}
```

---

## Known Risks and Mitigations

### K1. 180-Degree Disambiguation — Resolved

**Risk**: The orientation stage detects 90-degree rotation reliably (staff
lines are either horizontal or vertical), but distinguishing right-side-up
from upside-down is much harder. Page numbers are small, faded, or absent.
If this fails, the page is upside down and every downstream stage fails
silently.

**Resolution** (implemented):

The 180° ambiguity is resolved by a cascading polarity detector that
achieved 224/224 (100%) accuracy on the LPA-1 test set:

1. **Tesseract OSD (letter shapes)**: Primary detector.  Analyses the
   shapes of individual characters (ascenders, descenders, letter
   geometry) to determine if text is at 0° or 180°.  Two passes:
   standard (1200 px, colour) and adaptive (2000 px, binarized,
   `--dpi 300`).  Handles ~98% of pages including aged manuscripts
   with stains and decorative elements.
2. **Red title edge comparison**: Chant-book-specific fallback.
   Compares proximity-weighted title-eligible red ink in the top 10%
   vs bottom 10% of the image.  Title-eligible = red rows with no
   dark text in the centre (distinguishes titles from body rubrics).
3. **Spine S/V detection**: Last-resort fallback for covers and blank
   pages.  Compares saturation-to-brightness ratio of left/right
   edges; the more-worn edge is placed on the left (Western
   convention).
4. **Manual override**: The `[orientation_overrides]` table in
   `book.toml` can force a specific rotation for individual images:
   ```toml
   [orientation_overrides]
   "IMG_0080.JPG" = 180  # force 180-degree rotation
   ```

**Design decisions**:
- EXIF is not used (unreliable across devices and transfer tools).
- OSD is preferred over page-number detection (page numbers are too
  small and faded in this corpus).
- Sequential consistency pass was not needed (per-page detection is
  accurate enough).

### K2. Stage 9 Enhancement Chain Ordering

**Risk**: The 12-step enhancement chain has order-dependent interactions.
Wrong ordering could cause one step to amplify artifacts from another.

**Mitigations**:

1. **Principled ordering** (coarse-to-fine, low-frequency-to-high-frequency):
   - First: color cast correction (global color shift)
   - Second: illumination normalization (low-frequency spatial)
   - Third: shadow removal (medium-frequency spatial)
   - Fourth: stain correction (medium-frequency spatial)
   - Fifth: halo reduction (localized contrast)
   - Sixth: show-through removal (pixel-level classification)
   - Seventh: white balance (global color adjustment on clean signal)
   - Eighth: CLAHE (adaptive contrast on clean signal)
   - Ninth: salt correction (localized CLAHE)
   - Tenth: denoise (high-frequency noise removal)
   - Eleventh: sharpen (high-frequency detail restoration)
   - Last: binarize (if requested)
2. **Checkpoint each sub-step during development**: Temporarily save
   intermediate results after each sub-step to visually verify the chain.
   Remove intermediate checkpoints once ordering is validated.
3. **A/B comparison**: The `inspect` command should show before/after for
   each sub-step to identify problematic interactions.

### K3. No QA/Validation Mechanism

**Risk**: With 225+ pages per book, failures on individual pages (wrong
orientation, bad crop, distorted dewarp) can go unnoticed until the final
PDF is reviewed.

**Mitigations**:

1. **Per-page confidence scores**: Each stage computes a confidence metric
   and writes it to `pipeline.json`:
   - Orientation: number of agreeing signals (0-3), focus score (Laplacian variance)
   - Page detection: contour area as fraction of image
   - Dewarping: number of staff lines found, polynomial fit R-squared
   - Deskewing: skew angle magnitude
   - Enhancement: background uniformity score
2. **Flagged pages**: Pages with low confidence in any stage are flagged.
   The CLI reports them at the end: `"3 pages flagged for review: IMG_0060,
   IMG_0080, IMG_0230"`.
3. **Contact sheet generation**: New `lpacleaner review OUTPUT_DIR` command
   that generates a single image showing thumbnails of all pages from a
   given stage (e.g., `--stage 09_enhanced`), so the user can visually
   scan all 225 pages at once and spot problems.
4. **Stage-level summary stats**: After each stage completes, log summary
   statistics:    "Stage 8: 210/225 pages dewarped (93%), 12 passed through
   (no lines), 3 used AI fallback."

### K4. Text-Only and Special Pages

**Risk**: Title pages, text-only pages, blank pages, index pages, and
heavily damaged pages have zero staff lines. The plan's fallback paths
exist but are secondary. In practice 10-20% of pages per book may be
non-standard.

**Mitigations**:

1. **Page type classification** (in Stage 4 or as part of analyze):
   - "music": has staff lines (detected via ink mask)
   - "text": has text but no staff lines
   - "decorative": title page, elaborate borders
   - "blank": stddev < threshold
   - "damaged": very low contrast or mostly stained
   Stored in metadata, used to select appropriate algorithms per page.
2. **First-class fallback paths**: The projection profile deskew and
   pass-through dewarp paths are not "fallbacks" -- they're the correct
   algorithms for text pages. Code and test them to the same standard as
   the staff-line paths.
3. **Analyze detects page type distribution**: `book.toml` records how
   many pages of each type were found in samples, so the user knows what
   to expect.

### K5. OCR Produces Garbage on Music Notation

**Risk**: Tesseract running on pages that are 70% musical notation produces
nonsense for the notation areas, polluting the searchable text layer.

**Mitigations**:

1. **Notation masking before OCR**: Detect staff line regions (using the
   same line_detect.py output from Stage 8) and mask them with white
   before feeding to Tesseract. This leaves only text lines visible to OCR.
2. **Text region extraction**: Use horizontal projection profiles to find
   text lines between staves. These inter-staff text bands are the only
   regions that should be OCR'd.
3. **PSM selection**: Use `--psm 6` (uniform block of text) on extracted
   text regions rather than full-page OCR.
4. **Confidence filtering**: Tesseract outputs per-word confidence. Discard
   words below a threshold (e.g., 40%) to avoid nonsense in the text layer.

### K6. Page Ordering May Not Match Filename Order

**Risk**: Filenames (IMG_0001 through IMG_0225) might include cover shots,
spine photos, retakes, or be out of order. Filename order != book page order.

**Mitigations**:

1. **Page number extraction**: After orientation is fixed, run a targeted
   OCR on the page number region (configured position, small crop) to
   extract the printed page number. Use this for ordering.
2. **Duplicate/retake detection**: If two images have the same extracted
   page number, flag as possible retake. Let the user choose which to keep
   via `book.toml`:
   ```toml
   [page_overrides]
   exclude = ["IMG_0045.JPG", "IMG_0046.JPG"]  # retakes, use IMG_0047
   ```
3. **Fallback to filename order**: If page number extraction fails (no
   number detected), fall back to filename order. Flag these pages.
4. **Manual ordering**: Support an optional `page_order` list in `book.toml`
   for fully manual control:
   ```toml
   page_order = ["IMG_0011.JPG", "IMG_0012.JPG", ...]
   ```

### K7. Possible Two-Page Spreads

**Risk**: Some photos might capture a full two-page spread (both pages
fully visible). The largest-contour approach grabs both as one page.

**Mitigations**:

1. **Spread detection** (in Stage 4): After finding the largest contour,
   check its aspect ratio. If width/height > 1.5 (landscape-oriented and
   much wider than a single page), suspect a spread.
2. **Vertical split detection**: Look for a vertical line/gap near the
   center of the detected quad (the book spine). Use edge detection or
   luminance valley to find the split point.
3. **Split into two pages**: If spread detected, split into left and right
   halves, process each as a separate page.
4. **The `analyze` command detects spreads** in sample images and records
   `has_spreads = true` in `book.toml`. If detected, the split logic
   activates automatically.

### K8. Performance at Scale

**Risk**: 225 images x 10+ stages = 2000+ operations per book. Some
operations are slow (denoise ~500ms, inpainting ~200ms, AI inference).
Total per-book: 30-60 minutes. With 15+ books: 8-15 hours.

**Mitigations**:

1. **Image-level parallelism**: Process multiple images simultaneously
   using `concurrent.futures.ProcessPoolExecutor`. Most stages are
   per-image with no inter-image dependencies.
   - Default: `--workers N` where N = CPU count / 2 (leave headroom for GPU)
   - Stage 10 (normalize) is the exception: it needs all images first
2. **Skip unchanged**: If a checkpoint exists, the source image mtime
   hasn't changed, AND the stage's config hash matches the previous run,
   skip that image. Config changes automatically invalidate affected
   stages and all downstream stages (see Resumability section).
3. **GPU batching**: For operations using UMat (CLAHE, remap), batch the
   GPU transfers to amortize overhead.
4. **Progress reporting**: `tqdm` progress bar per stage showing
   images/second throughput and ETA.
5. **Profile first**: Before optimizing, profile a 10-image run to find
   the actual bottleneck. It might not be where we expect.

### K9. Partial Photo Stitching Failures

**Risk**: Partial photographs of the same page may have significantly
different perspective, lighting, or white balance between shots. The
photographer may have moved the book between shots. Stitching may produce
visible seams, ghosting, or misalignment.

**Mitigations**:

1. **Pre-stitching homogenization**: Before stitching, normalize white
   balance and exposure across images in the same group using the
   overlapping region as reference. This reduces visible seams.
2. **Multi-band blending**: `cv2.Stitcher` uses multi-band blending by
   default, which handles gradual lighting differences well.
3. **Fallback to best single image**: If `cv2.Stitcher` fails (returns
   `ERR_NEED_MORE_IMGS` or `ERR_HOMOGRAPHY_EST_FAIL`), fall back to the
   single image in the group that captures the largest page area. Flag
   the page for manual review.
4. **Manual group override**: Allow explicit grouping in `book.toml`:
   ```toml
   [stitch_groups]
   "INDEX_1" = ["IMG_0232.JPG", "IMG_0233.JPG", "IMG_0234.JPG"]
   ```
   Also allow forcing images as standalone:
   ```toml
   [stitch_overrides]
   standalone = ["IMG_0233.JPG"]  # this is a retake, not a partial
   ```
5. **Visual QA**: The `review` command includes stitched results with
   overlay lines showing the seam positions, so the user can spot
   misalignment.

### K10. Incomplete Page Coverage (Cut Corners)

**Risk**: Many photographs do not capture the full page -- one or two
corners are outside the camera frame. This is common when photographing
thick bound books where the photographer cannot frame the page perfectly.
The missing corners affect multiple stages differently.

**Impact per stage**:

| Stage | Impact | Severity |
|-------|--------|----------|
| 4 (page detect) | Quad can't find real corners, falls back to full-image quad | Low |
| 5 (perspective) | Full-image fallback = near-identity warp, so minimal distortion | Low |
| 6 (content area) | Border detection may fail on cut side; ink-density fallback works; feathering hides the cut edge | Low |
| 7 (deskew) | Rotation shifts the cut corner, background fill makes it visible; `trim_to_content()` cleans up | Medium |
| 8 (dewarp) | Staff lines near the cut corner are shorter/missing, weakening polynomial fit on that side | Medium |

**Mitigations** (most already in place):

1. **Full-image quad fallback** (Stage 4): When the quad detection fails to
   find 4 real corners, it returns the full image bounds. This prevents
   incorrect perspective correction in Stage 5.
2. **`page_detect_expand_frac`** (Stage 4): Expands the detected quad
   outward by a fraction, reducing the chance of cropping at the edge.
3. **Feathered edge masking** (Stage 6): Gaussian feathering at image
   edges fades out the cut-corner region instead of creating a hard edge.
4. **Background fill** (Stages 5, 7, 8): All geometric transforms use
   `borderMode=BORDER_CONSTANT` with the estimated background color.
   Cut corners are filled with a plausible color rather than black.
5. **`trim_to_content()`** (Stages 7, 8): Post-geometry trim crops the
   background-filled artifacts from cut corners after rotation/dewarp.
6. **This is fundamentally a photography limitation**: The pipeline
   cannot recover content that was never captured. The mitigations make
   the result look clean, but the missing corner content is gone.

---

## Dependencies (`pyproject.toml`)

```toml
[project]
name = "lpacleaner"
version = "0.1.0"
requires-python = ">=3.11"
dependencies = [
    "opencv-python>=4.10",
    "numpy>=2.0",
    "scipy>=1.12",
    "pillow>=10.0",
    "click>=8.0",
    "img2pdf>=0.5",
    "ocrmypdf>=16.0",
    "tqdm>=4.60",
]

[project.optional-dependencies]
ai = ["openvino>=2024.0", "torch>=2.0"]
historical-ocr = ["kraken>=5.0"]
dev = ["pytest>=8.0", "pytest-cov>=5.0"]

[project.scripts]
lpacleaner = "lpacleaner.cli:main"
```

System packages: `tesseract`, `tesseract-langpack-lat` (via `dnf`)

---

## Testing Strategy (TDD)

Development follows strict red-green-refactor TDD: write a failing test
first, implement just enough code to make it pass, then refactor.

### Project Structure

```
tests/
  conftest.py               # Shared fixtures: test images, temp dirs, Config factory
  fixtures/                  # Synthetic test images (generated, not real photos)
    music_page_4x3.png       # Synthetic music page with red staff lines
    text_page_4x3.png        # Synthetic text-only page
    rotated_90cw.jpg         # Same page with EXIF orientation=6
    partial_top.png           # Top half of a page (for stitch testing)
    partial_bottom.png        # Bottom half with overlap
    blurry_page.png           # Synthetically blurred page
    finger_on_edge.png        # Page with skin-colored region at border
    barrel_distorted.png      # Synthetically distorted with known k1
    book_cover.png            # Dark uniform image (non-content)
  test_image_io.py
  test_line_detect.py
  test_geometry.py
  test_preprocess.py
  test_accel.py
  test_config.py
  test_stage_stitch.py
  test_stage_orientation.py
  test_stage_lens_correct.py
  test_stage_page_detect.py
  test_stage_perspective.py
  test_stage_content_area.py
  test_stage_dewarp.py
  test_stage_deskew.py
  test_stage_enhance.py
  test_stage_normalize.py
  test_stage_ocr.py
  test_stage_pdf_assembly.py
  test_pipeline.py           # Integration tests
  test_cli.py                # CLI invocation tests
```

### Test Fixtures: Synthetic Images

Test images are **generated programmatically**, not extracted from real
books (which are large, copyrighted, and non-reproducible). Each fixture
is a function that creates a controlled test image:

```python
def make_music_page(width=800, height=600, staff_color=(0, 0, 200),
                    num_staves=4, skew_deg=0, curve_amount=0,
                    noise_level=0, bg_color=(230, 220, 200)):
    """Generate a synthetic music page with staff lines, text, and border."""

def make_text_page(width=800, height=600, ...): ...
def make_page_on_background(page, angle=0, perspective_skew=0, bg=(40,30,25)): ...
def add_finger(img, position="top-right", size_frac=0.05): ...
def add_barrel_distortion(img, k1=0.3): ...
def add_hotspot(img, center, radius): ...
def blur_image(img, kernel_size=15): ...
```

Small images (800x600) for fast tests. A few 4000x3000 fixtures
(marked `@pytest.mark.slow`) for realistic integration tests.

### Test Tiers

Two tiers for developer workflow:

| Tier | Command | When to run | Target time |
|------|---------|-------------|-------------|
| **Fast** | `pytest -m "not slow"` | Every edit, during TDD | <15 seconds |
| **Full** | `pytest` | Before commit, in CI | <5 minutes |

Tests marked `@pytest.mark.slow` are those that:
- Process full-resolution (4000x3000) synthetic images
- Run multi-image stage integration tests (e.g., StitchStage with grouping)
- Perform batch operations (e.g., analyze over multiple samples)
- Individually take >2 seconds

Fast tests use small (800x600) images and test single-function behavior.
The `slow` marker is registered in `pyproject.toml`.

### Test Levels

#### 1. Unit Tests (per function, fast, <1s each)

Every public function in `utils/` gets tests before implementation:

```python
# test_image_io.py
class TestLoadImage:
    def test_loads_jpeg(self, tmp_path): ...
    def test_loads_png(self, tmp_path): ...
    def test_applies_exif_rotation(self): ...
    def test_extracts_exif_metadata(self): ...
    def test_returns_empty_exif_for_png(self): ...

class TestSaveCheckpoint:
    def test_saves_as_png(self, tmp_path): ...
    def test_atomic_write_no_partial_on_interrupt(self, tmp_path): ...
    def test_writes_metadata_sidecar(self, tmp_path): ...
    def test_cleans_up_tmp_files(self, tmp_path): ...

# test_line_detect.py
class TestDetectInkMask:
    def test_detects_red_staff_lines(self): ...
    def test_detects_brown_staff_lines(self): ...
    def test_ignores_background(self): ...
    def test_fallback_to_channel_difference(self): ...

class TestDetectInkMaskGeometric:
    def test_filters_round_foxing_spots(self): ...
    def test_preserves_horizontal_lines(self): ...

class TestDetectStaffLines:
    def test_finds_expected_number_of_lines(self): ...
    def test_returns_polynomial_coefficients(self): ...
    def test_returns_empty_for_text_page(self): ...

# test_geometry.py
class TestOrderCorners:
    def test_already_ordered(self): ...
    def test_shuffled_corners(self): ...
    def test_near_rectangular(self): ...
```

#### 2. Stage Tests (per stage, medium, <5s each)

Each stage is tested end-to-end with synthetic inputs:

```python
# test_stage_orientation.py
class TestOrientationStage:
    def test_corrects_90cw_rotation(self): ...
    def test_corrects_90ccw_rotation(self): ...
    def test_detects_upside_down_via_text_direction(self): ...
    def test_passthrough_when_already_correct(self): ...
    def test_computes_focus_score(self): ...
    def test_flags_blurry_image(self): ...

# test_stage_page_detect.py
class TestPageDetectStage:
    def test_finds_page_on_dark_background(self): ...
    def test_fallback_to_inverted_otsu(self): ...
    def test_fallback_to_canny(self): ...
    def test_detects_spread(self): ...
    def test_classifies_music_page(self): ...
    def test_classifies_text_page(self): ...
    def test_classifies_blank_page(self): ...

# test_stage_deskew.py
class TestDeskewStage:
    def test_corrects_3_degree_skew(self): ...
    def test_skips_when_angle_below_threshold(self): ...
    def test_uses_projection_profile_for_text_page(self): ...
    def test_fills_border_with_background_color(self): ...
    def test_post_geometry_trim(self): ...

# test_stage_enhance.py
class TestEnhanceStage:
    def test_sub_steps_run_in_correct_order(self): ...
    def test_skips_disabled_sub_steps(self): ...
    def test_color_cast_correction(self): ...
    def test_shadow_removal(self): ...
    def test_denoise_reduces_noise(self): ...
    def test_sharpen_increases_laplacian_variance(self): ...
```

#### 3. Integration Tests (multi-stage, slow, <30s each)

Test the pipeline end-to-end on synthetic data:

```python
# test_pipeline.py
class TestPipeline:
    def test_full_pipeline_produces_pdf(self, tmp_path): ...
    def test_geometry_profile_skips_enhance(self, tmp_path): ...
    def test_resume_after_interrupt(self, tmp_path): ...
    def test_auto_analyze_when_no_book_toml(self, tmp_path): ...
    def test_skip_ocr_when_tesseract_missing(self, tmp_path): ...
    def test_parallel_processing(self, tmp_path): ...

class TestPipelineOutputValidation:
    def test_pdf_has_correct_page_count(self, tmp_path): ...
    def test_pdf_pages_have_correct_dpi(self, tmp_path): ...
    def test_pdf_has_text_layer_when_ocr_enabled(self, tmp_path): ...
    def test_all_checkpoint_dirs_exist(self, tmp_path): ...
    def test_pipeline_json_has_all_stages(self, tmp_path): ...
    def test_flagged_pages_reported(self, tmp_path): ...

# test_cli.py
class TestCLI:
    def test_run_with_only_input_dir(self, tmp_path): ...
    def test_run_with_profile(self, tmp_path): ...
    def test_run_with_skip_flags(self, tmp_path): ...
    def test_analyze_generates_book_toml(self, tmp_path): ...
    def test_review_generates_contact_sheet(self, tmp_path): ...
```

#### 4. Real Image Smoke Test (after Stage 2)

Planned after Stage 2 (orientation) is complete. Runs Stages 0-1-2 on
a small set (5-10) of real LPA-1 images to validate that:

1. Stage 0: hotspot/finger detection doesn't produce artifacts on real photos
2. Stage 1: grouping correctly identifies the known partial photo set
   (IMG_0232-0234) and excludes the book cover (IMG_0231)
3. Stage 2: orientation produces correctly rotated pages

The smoke test is **not automated** -- it's a manual visual inspection of
checkpoint outputs. Results inform whether synthetic test parameters need
adjustment and which integration test paths to add (GitHub #3).

Select images that cover common scenarios:
- A normal standalone page
- The book cover (IMG_0231)
- The 3-image partial set (IMG_0232-0234)
- A page with visible finger at the border
- A page with uneven lighting

#### 5. Regression Tests (golden reference, run on real images)

Not part of the standard test suite (requires actual book photos),
but available via `pytest -m regression`:

```python
@pytest.mark.regression
class TestRealBookRegression:
    """Run on a small set of real images with known-good outputs.
    
    Golden references are stored in tests/golden/ as PNG files.
    Tests compare stage outputs against golden references using
    structural similarity (SSIM > 0.95) rather than pixel-exact
    comparison, allowing for minor algorithmic improvements.
    """
    def test_lpa1_orientation(self): ...
    def test_lpa1_page_detect(self): ...
    def test_lpa1_dewarp(self): ...
```

### TDD Workflow per Implementation Step

```
For each item in the Implementation Order:
1. RED:   Write test(s) for the function/stage (they fail)
2. GREEN: Implement just enough code to pass the tests
3. REFACTOR: Clean up, extract helpers, improve naming
4. VERIFY: Run full test suite (pytest), check no regressions
5. COMMIT: One commit per red-green-refactor cycle
```

### Running Tests

```bash
pytest -m "not slow"             # fast tests only (~15s) -- use during TDD
pytest                           # full suite (all tiers)
pytest -x                        # stop at first failure
pytest --cov=lpacleaner          # with coverage report
pytest -m slow                   # only slow tests (4000x3000 images)
pytest -m regression             # only regression tests (real images)
pytest tests/test_stage_deskew.py  # single test file
```

---

## Implementation Order

Development follows TDD. For each step: write failing tests first,
then implement, then refactor. Build bottom-up, testing each piece
on synthetic images:

1. ~~**Project scaffolding**~~: ✅ pyproject.toml, empty modules, CLI skeleton, conftest.py
2. ~~**utils/image_io.py**~~: ✅ EXIF-aware load, PNG save with atomic writes, checkpoints
3. ~~**utils/accel.py**~~: ✅ GPU detection, UMat wrappers
4. ~~**utils/line_detect.py**~~: ✅ ink mask, geometric filter (R9), illustration exclusion (R4), staff lines, adaptive polyfit
5. ~~**utils/preprocess.py**~~: ✅ hotspot removal (R1), finger detection (R8)
6. ~~**config.py**~~: ✅ Config dataclass with all params, TOML loading, profile resolution, stitch params
7. ~~**Stage 0** (preprocess)~~: ✅ wire preprocess.py into BaseStage (18 tests)
8. ~~**Stage 1** (stitch)~~: ✅ grouping, stitching fallback chain, retake dedup, non-content detection (23 tests)
9. ~~**stages/analyze.py**~~: ✅ auto-detect book characteristics, generate book.toml, adaptive sampling, coarse orientation, simplified quad detection, median-based robustness (18 tests)
10. ~~**Stage 2** (orientation)~~: ✅ content-based axis detection (HoughLinesP) + staff-area validation + cascading polarity (Tesseract OSD standard + adaptive, red title edges, spine S/V) + focus QA (31 tests, 224/224 on LPA-1)
11. ~~**Real image smoke test**~~: ✅ Stages 0-1-2 on full LPA-1 set (225 images), visual inspection confirmed 224/224 correct orientation
12. ~~**Stage 3** (lens correct, optional)~~: ✅ radial distortion correction (R7) via `cv2.undistort`, auto-skip when k1=k2=0, `max(w,h)` focal length (22 tests)
13. ~~**Stage 4** (page detect)~~: ✅ Otsu→inverted Otsu→Canny→adaptive→full-image cascade, quad refinement with escalating epsilon, ink-aware page classification, bounding-box crop (27 tests)
14. ~~**Stage 5** (perspective)~~: ✅ warpPerspective from Stage 4 quad, max-edge sizing, background fill (not black), sidecar propagation via BaseStage.run() (28 tests)
15. ~~**Stage 6** (content area)~~: ✅ Hough border detection→ink density→inset fallback; feathered masking; margin padding; sidecar forwarding (29 tests)
16. **Stage 7** (deskew): staff angle or projection profile, post-geometry trim
17. **Stage 8** (dewarp): polynomial mesh from staff lines, background fill (most complex stage)
18. **Stage 9** (enhance): R3 color cast, illumination, shadows (R5), stains (R6), halos (R10), show-through, CLAHE, salt (R11), denoise, sharpen
19. **Stage 10** (normalize): cross-page color + DPI (global pass, batch stage)
20. **Stage 11** (OCR): Tesseract integration, graceful skip if missing, Kraken optional
21. ~~**Stage 12** (PDF)~~: ✅ img2pdf assembly, JPEG/PNG compression, case-insensitive config, DPI layout, resume, exclude (35 tests)
22. **Stage 13** (flipbook): static HTML flipbook export with StPageFlip, web-optimized images (batch stage)
23. **pipeline.py**: orchestrator -- chain stages, progress reporting, end-of-run summary
24. **CLI polish**: tqdm progress bars, `inspect` + `review` commands, error handling UX
25. **Integration tests**: full-pipeline tests, CLI tests, output validation
26. **Performance**: memory optimization (GitHub #1), parallelism tuning
