---
title: "Configuration"
weight: 30
description: "Customize processing with book.toml and profiles"
---

Guido's Helping Hand works out of the box with no configuration. For fine
control, you can customize processing through a `book.toml` file and
command-line options.

## Configuration priority

Settings are resolved in this order (highest priority first):

1. **CLI arguments** -- flags passed to `ghh run`
2. **book.toml** -- per-book configuration file
3. **Profile defaults** -- the selected profile's skip/include rules
4. **Built-in defaults** -- sensible defaults for all parameters

## Auto-generating book.toml

The simplest way to create a configuration file is to let ghh analyze
your photos:

```bash
ghh analyze /path/to/photos
```

This samples 15 images (configurable with `--samples`) and detects:

- Ink colors and saturation
- Page layout (border frames, staff line count)
- Photography conditions (lighting, lens distortion)
- Physical condition (stains, fading, foxing)

The result is written to `book.toml` in the input directory. `ghh run`
picks it up automatically.

## Profiles

Profiles control which optional stages are included. Select one with
`--profile`:

| Profile | Skipped stages | Use when |
|---------|---------------|----------|
| `full` | none | Final production run |
| `geometry` | Content area, dewarp, deskew, enhance, normalize, OCR, OMR | Quick crop and straighten check |
| `clean` | OCR | No searchable text needed |
| `quick` | Content area, dewarp, deskew, normalize, OCR, OMR | Fast preview with enhancement |

```bash
ghh run /path/to/photos --profile geometry
```

Individual stages can also be skipped explicitly:

```bash
ghh run /path/to/photos --skip-ocr --skip-normalize
```

## book.toml reference

Below is a complete annotated `book.toml` with all supported sections and
parameters. Every parameter has a sensible default; you only need to
include values you want to override.

### [book] -- Informational metadata

```toml
[book]
name = "LPA-1 San Nicolas"
type = "music"
```

These fields are not consumed by processing stages. They are stored for
your reference.

### [ink] -- Staff line ink detection

Controls how ghh detects the colored ink used for staff lines (typically
red, brown, or dark brown in historical manuscripts).

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `staff_color_hue` | integer | 5 | HSV hue center (0--180). Red = ~5, brown = ~15 |
| `staff_color_range` | integer | 15 | Hue tolerance (+/-) |
| `staff_saturation_min` | integer | 40 | Minimum HSV saturation |
| `staff_value_min` | integer | 80 | Minimum HSV brightness |
| `channel_diff_rg` | integer | 30 | Red minus Green threshold (fallback detector) |
| `channel_diff_rb` | integer | 30 | Red minus Blue threshold (fallback detector) |

```toml
[ink]
staff_color_hue = 5
staff_color_range = 15
staff_saturation_min = 40
```

### [layout] -- Page layout characteristics

Describes the physical layout of pages in the book.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `has_border_frame` | boolean | true | Pages have a printed border frame |
| `border_ink_matches_staff` | boolean | true | Border uses same ink as staff lines |
| `page_number_position` | string | `"top-right"` | Where page numbers appear |
| `expected_staff_lines_per_page` | integer | 16 | Expected number of staff lines |
| `has_illustrations` | boolean | false | Pages contain illustrations |
| `illustration_frequency` | string | `"none"` | `"none"`, `"rare"`, or `"frequent"` |
| `median_aspect_ratio` | float | 0.0 | Width/height of detected pages (auto-computed) |

```toml
[layout]
has_border_frame = true
expected_staff_lines_per_page = 16
has_illustrations = true
illustration_frequency = "rare"
```

### [photography] -- Camera and lighting conditions

Characteristics of the photography setup, auto-detected by `ghh analyze`.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `has_flash_hotspots` | boolean | false | Flash reflections detected |
| `fingers_detected` | boolean | false | Fingers visible at page edges |
| `lens_distortion_k1` | float | 0.0 | Barrel/pincushion coefficient |
| `lens_distortion_k2` | float | 0.0 | Higher-order distortion coefficient |
| `color_cast_detected` | string | `"none"` | `"none"`, `"slight_warm"`, `"slight_cool"`, `"strong_warm"`, `"strong_cool"` |
| `background_contrast` | string | `"dark_on_light"` | `"dark_on_light"` or `"light_on_dark"` |
| `shadow_severity` | string | `"none"` | `"none"`, `"mild"`, `"moderate"`, `"severe"` |
| `coarse_rotation_offset` | integer | 0 | Pre-EXIF rotation: 0, 90, 180, or 270 |

### [condition] -- Physical condition of the book

Describes degradation of the physical pages. Severity values drive how
aggressively enhancement sub-steps operate; `"none"` disables the
corresponding sub-step.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `stain_severity` | string | `"none"` | Water stains, discoloration |
| `ink_fading` | string | `"none"` | Ink fading or loss |
| `show_through_severity` | string | `"none"` | Ink bleeding from the reverse side |
| `foxing_severity` | string | `"none"` | Brown spots from fungal growth |
| `iron_gall_halos` | string | `"none"` | Brown halos around iron gall ink |
| `salt_deposits` | string | `"none"` | White salt deposits (coastal books) |

All severity values: `"none"`, `"mild"` / `"slight"`, `"moderate"`, `"severe"`.

```toml
[condition]
stain_severity = "mild"
ink_fading = "slight"
show_through_severity = "moderate"
foxing_severity = "mild"
```

### [stitch] -- Image stitching (Stage 1)

Controls how partial photographs of the same page are grouped and stitched.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `min_matches` | integer | 30 | Minimum feature matches for stitching |
| `ratio_threshold` | float | 0.75 | Lowe's ratio test threshold |
| `min_overlap_frac` | float | 0.2 | Minimum overlap fraction |
| `inlier_ratio` | float | 0.5 | Minimum RANSAC inlier ratio |
| `retake_overlap_threshold` | float | 0.9 | Overlap above this = retake, not partial |

### [page_overrides] -- Manual image overrides

Override automatic grouping, exclusion, and ordering for specific images.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `stitch_groups` | list of lists | none | Manual grouping of images for stitching |
| `exclude` | list of strings | none | Images to exclude from processing |
| `no_stitch` | list of strings | none | Images that should not be stitched |
| `include_covers` | boolean | false | Include cover images in the output |

```toml
[page_overrides]
exclude = ["IMG_0045.JPG", "IMG_0046.JPG"]
stitch_groups = [["IMG_0232.JPG", "IMG_0233.JPG", "IMG_0234.JPG"]]
```

### [page_detect] -- Page detection (Stage 4)

Controls how ghh finds the page within each photograph.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `method` | string | `"auto"` | Detection method (`"auto"`, `"otsu"`, `"canny"`, `"adaptive"`) |
| `morph_kernel` | integer | 50 | Morphological kernel size for cleanup |
| `epsilon` | float | 0.02 | Contour approximation epsilon |
| `min_area_frac` | float | 0.30 | Minimum page area as fraction of image |
| `padding` | integer | 10 | Pixel padding around detected page |
| `expand_frac` | float | 0.03 | Outward expansion of detected quad |

### [content_area] -- Content area detection (Stage 6)

Controls how ghh finds the content region within a cropped page.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `inset_fallback` | float | 0.05 | Fallback inset as fraction of page size |
| `margin_padding` | float | 0.02 | Extra margin around detected content |
| `feather_sigma` | integer | 20 | Gaussian feathering sigma for edge masking |

### [deskew] -- Deskew (Stage 8)

Controls straightening of tilted pages.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `max_angle` | float | 5.0 | Maximum correction angle (degrees) |
| `angle_step` | float | 0.1 | Angle search resolution (degrees) |
| `skip_threshold` | float | 0.1 | Skip if detected angle is below this |

### [enhance] -- Enhancement sub-steps (Stage 9)

Toggle individual enhancement operations. All default to `true`.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `color_cast_correction` | boolean | true | Correct color cast |
| `illumination_normalization` | boolean | true | Normalize uneven lighting |
| `shadow_correction` | boolean | true | Remove shadows |
| `stain_correction` | boolean | true | Reduce stain visibility |
| `halo_reduction` | boolean | true | Reduce iron gall ink halos |
| `show_through_removal` | boolean | true | Remove show-through bleed |
| `white_balance` | boolean | true | Apply white balance |
| `clahe` | boolean | true | Adaptive contrast (CLAHE) |
| `salt_correction` | boolean | true | Reduce salt deposits |
| `denoise` | boolean | true | Noise reduction |
| `sharpen` | boolean | true | Sharpening |

```toml
[enhance]
show_through_removal = false   # disable for pages without show-through
sharpen = false                # skip if already sharp
```

### [pdf] -- PDF assembly (Stage 15)

Controls final PDF creation.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `compression` | string | `"jpeg"` | `"jpeg"` or `"png"` |
| `jpeg_quality` | integer | 90 | JPEG quality (1--100) |
| `dpi` | integer | 300 | Output resolution |

### [flipbook] -- Flipbook generation

Controls the HTML flipbook viewer.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `max_width` | integer | 1600 | Maximum page width in pixels |
| `jpeg_quality` | integer | 85 | JPEG quality for flipbook pages |
| `title` | string | directory name | Title shown in the viewer |

### [ocr] -- Optical Character Recognition (Stage 11)

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `engine` | string | `"tesseract"` | OCR engine (`"tesseract"` or `"kraken"`) |
| `language` | string | `"lat"` | Language code for OCR |

### [omr] -- Optical Music Recognition (Stage 13)

Controls OMR inference using [ChantOMR](https://pgarciaq.github.io/chant-omr/).
Requires the `chant-omr` package to be installed separately.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model_dir` | string | `""` | Path to exported OpenVINO model directory |
| `beam_width` | integer | 1 | Beam search width (1 = greedy decoding) |
| `device` | string | `"AUTO"` | OpenVINO device: `"AUTO"`, `"CPU"`, `"GPU"` |

```toml
[omr]
model_dir = "models/chant-omr"
beam_width = 1
device = "AUTO"
```

### [pipeline] -- Pipeline control

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `profile` | string | `"full"` | Default processing profile |

```toml
[pipeline]
profile = "full"
```

## Complete example

Here is a full `book.toml` as generated by `ghh analyze` for a typical
Gregorian chant antiphonary:

```toml
[book]
name = "LPA-1 San Nicolas"
type = "music"

[ink]
staff_color_hue = 5
staff_color_range = 15
staff_saturation_min = 40
staff_value_min = 80
channel_diff_rg = 30
channel_diff_rb = 30

[layout]
has_border_frame = true
border_ink_matches_staff = true
page_number_position = "top-right"
expected_staff_lines_per_page = 16
has_illustrations = true
illustration_frequency = "rare"
median_aspect_ratio = 1.33

[photography]
has_flash_hotspots = false
color_cast_detected = "slight_warm"
background_contrast = "dark_on_light"
shadow_severity = "none"
lens_distortion_k1 = 0.0
lens_distortion_k2 = 0.0
fingers_detected = false
coarse_rotation_offset = 0

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
profile = "full"
```
