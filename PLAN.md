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
3. **Checkpointed**: Each stage writes its output to a numbered directory.
   Stages can be re-run independently. The pipeline can resume from any point.
4. **Hardware-accelerated**: Selective GPU acceleration via OpenCV UMat
   (Intel Arc OpenCL) for Canny, CLAHE, and remap operations. Optional
   AI dewarping via OpenVINO on GPU.

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
  lpacleaner/
    __init__.py
    cli.py                  # Click CLI: analyze, run, inspect commands
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
      dewarp.py             # Stage  7: staff line polynomial mesh or AI dewarping
      deskew.py             # Stage  8: staff line angle or projection profile
      enhance.py            # Stage  9: R3 color cast, illumination, show-through,
                            #   shadows, stains, halos, salt, CLAHE, denoise, sharpen
      normalize.py          # Stage 10: cross-page color + DPI normalization
      ocr.py                # Stage 11: Tesseract/Kraken OCR
      pdf_assembly.py       # Stage 12: searchable PDF assembly
    utils/
      __init__.py
      line_detect.py        # Generic ink detection, staff line detection, foxing filter (R9)
      geometry.py           # Quad ordering, homography, distance helpers
      image_io.py           # EXIF-aware load, save, checkpoint dir management
      accel.py              # GPU/OpenCL detection, UMat wrappers, OpenVINO init
      preprocess.py         # Flash hotspot removal (R1), finger detection (R8)
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
  07_dewarped/               IMG_0011.jpg, ...
  08_deskewed/               IMG_0011.jpg, ...
  09_enhanced/               IMG_0011.jpg, ...
  10_normalized/             IMG_0011.jpg, ...   (cross-page color + DPI matched)
  11_ocr/                    IMG_0011.hocr, ...  (hOCR XML files)
  output.pdf                                     (final searchable PDF)
  pipeline.json                                  (stage status, parameters used)
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
lpacleaner analyze INPUT_DIR [--output-dir OUTPUT_DIR] [--samples 15]
```

### Algorithm

```
1. Load N evenly-spaced images from the input directory

2. Partial photo detection (lightweight, before sampling):
   a. Run ORB feature matching on ALL consecutive image pairs
      (same algorithm as Stage 1 stitch grouping, but detection only)
   b. Identify groups of partial photos and retakes
   c. Exclude partial photos from sampling: only sample standalone
      images or the first image of each group (as a representative)
   d. Record detected groups in book.toml for Stage 1 to use

3. For each sampled image:
   a. Apply EXIF rotation
   b. Detect page quad (try all page_detect methods, pick best)
   c. Crop to page

4. Ink color discovery:
   a. Convert cropped pages to HSV
   b. Mask out near-white (background, value > 200) and
      near-black (text/notes, value < 60)
   c. Histogram the remaining hue values
   d. Dominant hue peak = staff line ink color
   e. Compute optimal HSV range and channel-difference thresholds

5. Layout analysis:
   a. Detect border frame presence (consistent linear structures at page edges)
   b. Count staff lines per page (estimate expected range)
   c. Detect page number positions (top-right, top-left, bottom)
   d. Detect illustration presence (multi-colored regions)
   e. Compute median page aspect ratio

6. Photography condition analysis:
   a. Flash: check for clipped-white regions (> 250 in all channels)
   b. Color cast: compare per-channel means (gray-world deviation)
   c. Background contrast: which Otsu method works (normal vs inverted)
   d. Shadows: detect long straight high-gradient edges
   e. Lens distortion: measure edge curvature on page quads
   f. Fingers: check for skin-colored regions at image borders

7. Physical condition analysis:
   a. Foxing: count small reddish-brown blobs matching ink color but
      with low aspect ratio (round, not linear)
   b. Iron gall halos: measure brown halo width around text
   c. Stains: detect large-area brightness deviations
   d. Salt deposits: detect bright textured (not clipped) patches
   e. Show-through: detect faint reverse-page ink bleed
   f. Ink fading: compare staff ink saturation to expected range

8. Write book.toml
```

### Output: `book.toml`

```toml
[book]
name = "LPA-1 San Nicolas"
type = "music"

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

[photography]
has_flash_hotspots = false
color_cast_detected = "slight_warm"
background_contrast = "dark_on_light"
shadow_severity = "none"
lens_distortion_k1 = 0.0
lens_distortion_k2 = 0.0
fingers_detected = false

[condition]
stain_severity = "mild"
ink_fading = "slight"
show_through_severity = "moderate"
foxing_severity = "mild"
iron_gall_halos = "slight"
salt_deposits = "none"

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

3. Stitching (for each group of 2+ images):
   a. Use cv2.Stitcher.create(cv2.Stitcher_PANORAMA)
   b. Feed all images in the group
   c. Stitcher performs: feature detection, pairwise matching,
      homography estimation, bundle adjustment, blending
   d. Output: single composite image at full resolution
   e. If stitching fails (insufficient overlap, too different angles):
      fall back to using the single image with largest page area
   f. Name output after first image in group (e.g., IMG_0232.jpg)
      and record the group composition in metadata

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

### Input/Output

- Input: images from `00_preprocessed/` (or raw JPGs if Stage 0 skipped)
- Output: grouped/stitched images in `01_stitched/`
- Metadata: group composition, retakes discarded, non-content images
- If no groups detected: images are symlinked/copied unchanged

---

## Stage 2: Orientation Normalization (`orientation.py`)

### Algorithm

```
1. Read EXIF orientation tag (PIL.Image._getexif()[274])
2. Apply EXIF rotation:
     1 -> no-op, 6 -> 90 CW, 8 -> 90 CCW, 3 -> 180

3. Detect content orientation via staff lines:
   a. Downscale to 1000px wide (speed)
   b. Isolate staff ink: mask = detect_ink_mask(img, cfg)
   c. HoughLinesP(mask, rho=1, theta=pi/180, threshold=50,
                  minLineLength=100, maxLineGap=15)
   d. Compute angle of each line segment
   e. Count segments within 15deg of horizontal vs vertical
   f. If vertical > horizontal: image needs 90deg rotation
   g. Direction: check which edge has more staff lines

4. 180-degree disambiguation (see K1 for full risk analysis):
   a. Signal 1: page number detection in configured quadrant
   b. Signal 2: border frame asymmetry (if cfg.has_border_frame)
   c. Signal 3: text direction (ascender/descender distribution)
   d. Multi-signal voting: require 2+ signals to agree
   e. If ambiguous: flag for manual review

5. Sequential consistency pass (after all pages oriented):
   a. Check for isolated orientation flips
   b. If page N differs from N-1 and N+1, re-evaluate with
      lower confidence threshold
```

6. Focus quality detection (QA metric, see K3):
   a. Compute Laplacian variance on the central 80% of the image
      (avoid edges where blur is expected from depth of field)
   b. focus_score = cv2.Laplacian(gray, cv2.CV_64F).var()
   c. Record in metadata as focus_score
   d. If focus_score < threshold (default 100): flag as blurry
   e. For books where many pages are blurry (poor camera/lighting),
      the threshold is auto-calibrated by analyze to the 10th
      percentile of sampled focus scores
   Note: blur cannot be fixed programmatically. This detection
   exists to flag pages for the user to reshoot or accept.
```

### Parameters

```python
orient_downscale_width: int = 1000
orient_hough_threshold: int = 50
orient_hough_min_length: int = 100
orient_hough_max_gap: int = 15
focus_score_threshold: float = 100.0
```

### Input/Output

- Input: stitched image from `01_stitched/` (or earlier if stages skipped)
- Output: correctly-oriented JPG in `02_oriented/`
- Metadata: orientation method, confidence, focus_score, blur flag

---

## Stage 3: Lens Distortion Correction (`lens_correct.py`, optional)

Runs only when `analyze` detects significant radial distortion (R7).
Must run **before** page detection: `cv2.undistort` requires the
original optical center (near image center), which is lost after
perspective correction. Correcting early also produces straighter
page edges, improving quad detection in Stage 4.

### Algorithm

```
1. Estimate distortion from analyze results (book.toml k1, k2):
   - analyze detects distortion by measuring edge curvature of page
     quads across sample images
   - If k1 == k2 == 0.0: skip this stage entirely

2. Build camera matrix:
   fx = fy = max(width, height)  (reasonable default for unknown focal length)
   cx, cy = width / 2, height / 2  (optical center)

3. cv2.undistort(img, camera_matrix, dist_coeffs)
   where dist_coeffs = [k1, k2, 0, 0, 0]

4. Optional: if analyze stored per-image coefficients (e.g., zoom
   varied between shots), load per-image k1/k2 from metadata
```

### Parameters

```python
lens_correction: bool = False       # auto-enabled by analyze
lens_distortion_k1: float = 0.0
lens_distortion_k2: float = 0.0
```

### Input/Output

- Input: oriented image from `02_oriented/`
- Output: undistorted image in `03_lens_corrected/`
- If distortion coefficients are zero: stage is skipped, images pass through

---

## Stage 4: Page Detection and Cropping (`page_detect.py`)

### Algorithm

```
1. Primary method (Otsu):
   a. Convert to grayscale
   b. Otsu threshold -> binary mask
   c. Morphological close with 50x50 rect kernel
   d. findContours(RETR_EXTERNAL, CHAIN_APPROX_SIMPLE)
   e. Sort by area descending, take largest
   f. approxPolyDP(contour, epsilon=0.02*perimeter, closed=True)
   g. If result has 4 vertices -> page quad found

2. R2 Fallback chain (if contour area < 30% or > 99% of image):
   a. Try inverted Otsu (light background, dark page)
   b. Try Canny edge detection + findContours
   c. Try adaptive threshold (handles mixed lighting)
   d. Last resort: GrabCut with center 80% rectangle, or full image

3. Quad refinement:
   a. If not 4 vertices: convex hull, try epsilon [0.02..0.08]
   b. Fallback: minAreaRect -> boxPoints
   c. Order corners as [TL, TR, BR, BL] via sum/difference method

4. Spread detection (see K7):
   a. If detected quad aspect ratio width/height > 1.5: suspect spread
   b. Look for vertical luminance valley or edge near center
   c. If confirmed: split into two page quads (left, right)
   d. Process each half as a separate image for all subsequent stages

5. Page type classification (see K4):
   a. Run detect_ink_mask on cropped page
   b. Count staff line candidates via HoughLinesP
   c. Classify as "music", "text", "decorative", "blank", or "damaged"
   d. Record in metadata (used by downstream stages to select algorithms)
   Note: runs on lens-corrected but not yet perspective-corrected image.
   The trapezoid shape slightly reduces staff line count accuracy, but
   the classification only needs "has staff lines? yes/no" which is
   robust to moderate perspective distortion.

6. Store quad corners and page type in metadata JSON
```

### Corner Ordering (`geometry.py`)

```python
def order_corners(pts: np.ndarray) -> np.ndarray:
    """Order 4 points as [TL, TR, BR, BL]."""
    s = pts.sum(axis=1)
    d = np.diff(pts, axis=1).ravel()
    return np.array([
        pts[np.argmin(s)],   # top-left: smallest x+y
        pts[np.argmin(d)],   # top-right: smallest x-y
        pts[np.argmax(s)],   # bottom-right: largest x+y
        pts[np.argmax(d)],   # bottom-left: largest x-y
    ], dtype=np.float32)
```

### Parameters

```python
page_detect_method: str = "auto"       # "auto", "otsu", "otsu_inverted", "canny", "grabcut"
page_detect_morph_kernel: int = 50
page_detect_epsilon: float = 0.02
page_detect_min_area_frac: float = 0.3
```

### Input/Output

- Input: image from `03_lens_corrected/` (or `02_oriented/` if lens correction skipped)
- Output: cropped image + `corners.json` in `04_cropped/`

---

## Stage 5: Perspective Correction (`perspective.py`)

### Algorithm

```
1. Load quad corners from Stage 4 metadata
2. Compute target rectangle:
   width  = max(dist(TL,TR), dist(BL,BR))
   height = max(dist(TL,BL), dist(TR,BR))
3. dst = [[0,0], [width,0], [width,height], [0,height]]
4. M = cv2.getPerspectiveTransform(src_corners, dst)
5. Estimate background color: median of border pixels (outermost 5%)
6. result = cv2.warpPerspective(img, M, (width, height),
       borderMode=cv2.BORDER_CONSTANT, borderValue=bg_color)
```

Out-of-bounds pixels are filled with the estimated page background color,
not black. This prevents downstream stages (enhance, normalize) from
being confused by black corners.

### Input/Output

- Input: cropped image + corners from `04_cropped/`
- Output: rectangular image in `05_perspective/`

---

## Stage 6: Content Area Detection (`content_area.py`)

Detects the border frame that surrounds content on most pages, providing
a tighter crop than the page edge, and masks residual adjacent page edges.

### Algorithm

```
1. Detect border frame:
   a. Isolate ink mask (line_detect.py detect_ink_mask)
   b. Morphological close with 5x5 kernel
   c. HoughLinesP: find long horizontal and vertical ink-colored lines
   d. Filter: horizontals near top/bottom, verticals near left/right
   e. Intersect 4 border lines to find content rectangle
   f. Fallback if cfg.has_border_frame is False or < 4 lines found:
      ink-density bounding box with padding, or fixed 5% inset

2. Mask adjacent page edges:
   a. Detect sharp vertical luminance transitions outside content rect
   b. Fill outside content rect with median background color
      (Gaussian feathering, sigma=20px)

3. Add uniform margins:
   a. Crop to content rectangle
   b. Add padding (default 2% of width) filled with background color

4. Store content rectangle in metadata
```

### Parameters

```python
content_detect_inset_fallback: float = 0.05
content_margin_padding: float = 0.02
content_feather_sigma: int = 20
```

### Input/Output

- Input: perspective-corrected image from `05_perspective/`
- Output: content-cropped image in `06_content/`

---

## Stage 7: Dewarping (`dewarp.py`)

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

4. Estimate background color: median of border pixels (outermost 5%)
5. Apply: cv2.remap(img, map_x, map_y, INTER_CUBIC,
       borderMode=cv2.BORDER_CONSTANT, borderValue=bg_color)
   (Use UMat for GPU -- 1.5x speedup)

5. Pages with < 2 detected staff lines:
   - Flag in metadata
   - If --ai-dewarp: use AI path
   - Otherwise: pass through unchanged
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

- Input: image from `06_content/`
- Output: dewarped image in `07_dewarped/`
- Metadata: staff positions, polynomial coefficients, method used

---

## Stage 8: Deskew (`deskew.py`)

### Algorithm

```
1. Detect staff lines (line_detect.py with geometric filter)
2. If lines found: skew_angle = median angle of line segments
3. If no lines (text-only pages):
   a. Binary threshold (Otsu)
   b. Projection profile: for angles in [-5, +5] step 0.1,
      score = variance of row sums. Best = max variance.
4. Estimate background color: median of border pixels (outermost 5%)
5. Rotate by -skew_angle via cv2.warpAffine with
   borderMode=cv2.BORDER_CONSTANT, borderValue=bg_color
6. Skip rotation if |skew_angle| < 0.1 degrees

7. Post-geometry trim (runs even if rotation was skipped):
   a. The geometric transforms in Stages 5, 7, 8 may have shifted
      content boundaries. Re-detect the actual content bounding box:
      - Threshold: pixels significantly different from bg_color
      - Find bounding rect of non-background region
   b. Crop to the content bounding box
   c. Add uniform margin padding (default 2% of width) filled with
      bg_color
   d. This produces the cleanest possible input for enhancement
```

### Parameters

```python
deskew_max_angle: float = 5.0
deskew_angle_step: float = 0.1
deskew_skip_threshold: float = 0.1
```

### Input/Output

- Input: dewarped image from `07_dewarped/`
- Output: deskewed image in `08_deskewed/`

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

- Input: deskewed image from `08_deskewed/`
- Output: enhanced image in `09_enhanced/`

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
   a. Load staff line positions from Stage 7 metadata
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

### Algorithm

```
1. Determine page order (see K6):
   a. If page numbers extracted in Stage 11: order by page number
   b. If page_order specified in book.toml: use manual order
   c. Fallback: filename order
   d. Exclude images listed in book.toml [page_overrides] exclude

2. Detect and handle spreads (see K7):
   a. If spread detected in Stage 4: both halves included as separate pages

3. With OCR: ocrmypdf.ocr(images, output.pdf, language=cfg.ocr_lang)
4. Without OCR (--no-ocr): img2pdf.convert(images)
5. Page sizing: uniform size based on median aspect ratio
```

### Input/Output

- Input: normalized images + hOCR files
- Output: `output.pdf`

---

## Shared Utilities

### line_detect.py

Generic ink color detection with foxing discrimination. Used by Stages 2
(orientation), 6 (content area), 7 (dewarp), and 8 (deskew).

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

### image_io.py

```python
def load_image(path, cfg) -> np.ndarray:
    """Load with EXIF rotation. No color correction here -- R3 lives in Stage 9."""

def save_checkpoint(img, stage_dir, filename, metadata=None): ...
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
| 7 | Dewarp | optional | on | `skip_dewarp = true` or no staff lines + no AI |
| 8 | Deskew | optional | on | `skip_deskew = true` or angle < 0.1 degrees |
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
lpacleaner analyze INPUT_DIR [-o OUTPUT_DIR] [--samples 15]
lpacleaner inspect IMAGE_PATH [--config book.toml]
lpacleaner review OUTPUT_DIR [--stage 09_enhanced]
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
```

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

## Known Risks and Mitigations

### K1. 180-Degree Disambiguation is Fragile

**Risk**: The orientation stage detects 90-degree rotation reliably (staff
lines are either horizontal or vertical), but distinguishing right-side-up
from upside-down is much harder. Page numbers are small, faded, or absent.
If this fails, the page is upside down and every downstream stage fails
silently.

**Mitigations**:

1. **Sequential consistency**: After initial per-page orientation, run a
   second pass that enforces consistency. If pages N-1 and N+1 are both
   right-side-up, page N should be too. Isolated flips are almost certainly
   errors.
2. **Text direction detection**: Latin text has ascenders (b, d, f, h, k, l)
   that point upward and descenders (g, j, p, q, y) that point downward.
   Compute vertical distribution of dark pixels in the text regions: the
   top-heavy half is "up." This works independently of staff lines.
3. **Multi-signal voting**: Combine page number position, border asymmetry,
   text direction, and sequential consistency. Require 2+ signals to agree
   before flipping 180 degrees. If ambiguous, flag for manual review.
4. **Manual override**: The `inspect` command shows orientation detection
   results. The user can add `[orientation_overrides]` to `book.toml`:
   ```toml
   [orientation_overrides]
   "IMG_0080.JPG" = 180  # force 180-degree rotation
   ```

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
   statistics:    "Stage 7: 210/225 pages dewarped (93%), 12 passed through
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
   same line_detect.py output from Stage 7) and mask them with white
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
2. **Skip unchanged**: If a checkpoint exists and the source image hasn't
   changed (check mtime), skip that image for that stage. Enables fast
   re-runs after parameter tuning.
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
]

[project.optional-dependencies]
ai = ["openvino>=2024.0", "torch>=2.0"]
historical-ocr = ["kraken>=5.0"]

[project.scripts]
lpacleaner = "lpacleaner.cli:main"
```

System packages: `tesseract`, `tesseract-langpack-lat` (via `dnf`)

---

## Implementation Order

Build bottom-up, testing each piece on 5-10 sample images:

1. **Project scaffolding**: pyproject.toml, empty modules, CLI skeleton
2. **utils/image_io.py**: EXIF-aware load/save, checkpoints
3. **utils/accel.py**: GPU detection, UMat wrappers
4. **utils/line_detect.py**: generic ink mask, geometric filter (R9), illustration exclusion (R4), staff line detection
5. **utils/preprocess.py**: hotspot removal (R1), finger detection (R8)
6. **config.py**: Config dataclass with all params, TOML loading
7. **Stage 0** (preprocess): wire preprocess.py into pipeline (runs before stitch)
8. **Stage 1** (stitch): image grouping (ORB features, homography), cv2.Stitcher, retake dedup, non-content detection
9. **stages/analyze.py**: auto-detect book characteristics (including partial photo detection), generate book.toml
10. **Stage 2** (orientation): EXIF + ink line angle + 180deg disambiguation
11. **Stage 3** (lens correct, optional): radial distortion correction (R7) -- simple, build early
12. **Stage 4** (page detect): Otsu + fallback chain (R2), page type classification
13. **Stage 5** (perspective): homography from quad corners
14. **Stage 6** (content area): border detection, edge masking, margins
15. **Stage 8** (deskew): staff angle or projection profile (build before 7 to validate line_detect.py)
16. **Stage 7** (dewarp): polynomial mesh from staff lines (most complex stage)
17. **Stage 9** (enhance): R3 color cast, illumination, shadows (R5), stains (R6), halos (R10), show-through, CLAHE, salt (R11), denoise, sharpen
18. **Stage 10** (normalize): cross-page color + DPI (global pass)
19. **Stage 11** (OCR): Tesseract integration
20. **Stage 12** (PDF): ocrmypdf / img2pdf assembly
21. **pipeline.py**: orchestrator with checkpointing, parallelism (K8), skip-unchanged
22. **CLI polish**: progress bars, error handling, `inspect` + `review` commands
