# Stage 4: Page Border Line Refinement -- Implementation Plan

## Problem

Stage 4 (Page Detect) uses Otsu thresholding + largest contour to find the
page quad. This merges bright non-page regions (fore edges, adjacent page
margins, book spine, covers) with the page into a single connected blob,
producing a quad that includes non-page artifacts. Stage 5 then faithfully
maps this over-sized quad to a rectangle.

**Parent issue**: #69
**Sub-issues**: #70, #71, #72

## Design

### Architecture

The enhancement adds a **post-processing refinement step** after the existing
Otsu detection. The cascade remains untouched; we add a new `_refine_quad_with_borders()`
function that runs between `_find_page_quad()` and `_expand_quad()`:

```
Current flow:
  _find_page_quad() → order_corners() → _expand_quad()

New flow:
  _find_page_quad() → order_corners() → _refine_quad_with_borders() → _expand_quad()
```

If border lines are not detected (e.g., other books without red borders),
the refinement is a no-op and the quad passes through unchanged.

### Red Border Line Detection (`_detect_page_borders`)

Located in `ghh/stages/page_detect.py` (same module, private function).

**Algorithm:**

1. **Color segmentation**: Convert image region to HSV. Create a red mask
   covering both ends of the hue circle (H < 15 or H > 165, S > 40, V > 60).
   This captures faded and bright red lines while excluding parchment.

2. **Morphological cleanup**: Close small gaps (red lines may be broken by
   ink/neumes crossing them). Erode to thin to single-pixel width.

3. **Hough line detection**: Run `HoughLinesP` on the red mask. Filter for:
   - **Vertical lines**: angle within ±15° of vertical, length > 40% of
     image height. These are left/right page borders.
   - **Horizontal lines**: angle within ±15° of horizontal, length > 40% of
     image width. These are top/bottom page borders (rarer, optional).

4. **Clustering**: Group detected lines by x-position (vertical) or
   y-position (horizontal). Take the leftmost and rightmost vertical
   clusters as left/right borders. Take the topmost and bottommost horizontal
   clusters as top/bottom borders.

5. **Confidence scoring**: Return confidence based on:
   - Number of line segments supporting each border (more = higher)
   - Consistency of line positions within a cluster (tighter = higher)
   - Whether both left and right borders were found (both = high confidence)

**Returns**: `PageBorders` dataclass with `left_x`, `right_x`, `top_y`,
`bottom_y` (each `float | None`) and `confidence: float`.

### Quad Refinement (`_refine_quad_with_borders`)

Takes the Otsu quad and the detected borders, returns a refined quad.

**Algorithm:**

1. If confidence < threshold (0.5), return quad unchanged (fallback).

2. For each detected border (left, right, top, bottom):
   - If the Otsu quad edge is **outside** the border line (i.e., the quad
     extends beyond the page into fore edges/desk), clip the quad edge
     inward to the border position.
   - If the Otsu quad edge is **inside** the border line (i.e., the quad
     is too tight, missing page content), leave it unchanged -- the
     subsequent `_expand_quad()` will compensate.

3. Recompute corner intersections:
   - If both vertical borders detected: set TL.x = BL.x = left_x,
     TR.x = BR.x = right_x.
   - If both horizontal borders detected: set TL.y = TR.y = top_y,
     BL.y = BR.y = bottom_y.
   - If only vertical borders detected: keep original top/bottom y-positions
     from the Otsu quad.

4. Small outward offset (2-3px) from detected border positions to ensure
   the red line itself is included in the output.

### Configuration

Add to `Config`:

| Field | Default | Description |
|---|---|---|
| `page_detect_border_refinement` | `true` | Enable/disable border-based refinement |
| `page_detect_border_confidence_threshold` | `0.5` | Min confidence to use borders |
| `page_detect_border_hue_range` | `[0, 15, 165, 180]` | HSV hue range for red detection |

TOML section:
```toml
[page_detect]
border_refinement = true
border_confidence_threshold = 0.5
```

### Metadata

The sidecar JSON gets new fields when borders are detected:

```json
{
  "stage": "page_detect",
  "method": "otsu",
  "border_refinement": {
    "left_x": 45.2,
    "right_x": 732.8,
    "top_y": null,
    "bottom_y": null,
    "confidence": 0.82,
    "applied": true
  },
  "quad_corners": [[...], ...]
}
```

## Implementation Steps

### Step 1: Red border detection function (#70)

**File**: `ghh/stages/page_detect.py`

Add:
- `PageBorders` dataclass
- `_detect_page_borders(img, cfg)` function
- HSV color segmentation for red
- Hough line detection + clustering
- Confidence scoring

**Tests** (`tests/test_stage_page_detect.py`):
- `TestBorderDetection.test_detects_red_vertical_lines` -- synthetic image with red vertical lines on parchment
- `TestBorderDetection.test_ignores_red_initials` -- red decorative elements should not be detected as borders
- `TestBorderDetection.test_no_borders_returns_none` -- image without red lines returns None/low confidence
- `TestBorderDetection.test_partial_borders` -- only left or only right detected
- `TestBorderDetection.test_faded_red_lines` -- lower saturation still detected

### Step 2: Quad refinement using borders (#71)

**File**: `ghh/stages/page_detect.py`

Add:
- `_refine_quad_with_borders(quad, borders, img_h, img_w)` function
- Integrate into `process_image()` between `order_corners()` and `_expand_quad()`
- Add `border_refinement` metadata to sidecar

**Config** (`ghh/config.py`):
- `page_detect_border_refinement: bool = True`
- `page_detect_border_confidence_threshold: float = 0.5`
- TOML loading in `from_toml()`

**Tests** (`tests/test_stage_page_detect.py`):
- `TestBorderRefinement.test_clips_quad_to_borders` -- quad extending past borders is clipped
- `TestBorderRefinement.test_leaves_tight_quad_unchanged` -- quad inside borders is not changed
- `TestBorderRefinement.test_partial_border_refinement` -- only left border clips left edge
- `TestBorderRefinement.test_low_confidence_skips_refinement` -- below threshold = no-op
- `TestBorderRefinement.test_disabled_refinement` -- `border_refinement=false` skips entirely

### Step 3: Integration tests (#72)

**File**: `tests/test_stage_page_detect.py`

Add:
- `TestBorderRefinementIntegration.test_fore_edge_excluded` -- synthetic page with bright fore edge strip; verify quad excludes it
- `TestBorderRefinementIntegration.test_adjacent_page_margin_excluded` -- synthetic page with adjacent page visible; verify quad excludes it
- `TestBorderRefinementIntegration.test_no_regression_on_clean_page` -- page without artifacts; verify quad is not degraded
- `TestBorderRefinementIntegration.test_end_to_end_with_stage5` -- verify Stage 4 + Stage 5 together produce clean output

### Step 4: Test on LPA 1

Run `ghh run --stages 4-5` on LPA 1 and verify:
- IMG_0030, IMG_0031, IMG_0036, IMG_0037 no longer have artifacts
- IMG_0041 (already good) is not degraded
- Overall quality across all 225 pages is improved

## Risks and Mitigations

| Risk | Mitigation |
|---|---|
| Red initials/titles false-positive as borders | Filter by line length (borders span >40% of image height/width; initials don't) and by position (borders are near edges, not in the middle) |
| Faded/broken red lines not detected | Morphological closing bridges gaps; lenient Hough parameters; confidence threshold allows fallback |
| Books without red borders | `border_refinement` is additive; if no borders detected, Otsu quad passes through unchanged |
| Red border lines at angles (un-rectified image) | Allow ±15° from vertical/horizontal; this covers typical photography angles |
| Over-clipping when border detection is slightly off | 2-3px outward offset from detected position; subsequent expand_frac adds further margin |

## Dependencies

- OpenCV (already a dependency): HSV conversion, Hough lines
- No new external dependencies required

## Estimated Effort

- Step 1 (detection): ~150 lines of code + ~100 lines of tests
- Step 2 (refinement): ~80 lines of code + ~80 lines of tests
- Step 3 (integration): ~100 lines of tests
- Step 4 (validation): manual review
- **Total**: ~400 lines of code + ~280 lines of tests
