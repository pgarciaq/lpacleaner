"""Generate interactive HTML comparison viewers for pipeline stages.

Two commands produce distinct viewers:

* ``ghh compare`` -- local ``file://`` references to PNGs on
  disk, dark-blue theme, labelled "Compare mode".
* ``ghh publish`` -- self-contained directory with downscaled
  JPEGs and relative paths, warm amber theme, labelled with the
  publication timestamp.
"""

from __future__ import annotations

import json
import logging
import os
import re
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path

import cv2
from tqdm import tqdm

logger = logging.getLogger(__name__)

_CHECKPOINT_RE = re.compile(r"^(\d{2})_")
_IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".tiff", ".tif"}

_STAGE_LABELS = {
    "00": "0: Preprocess",
    "01": "1: Stitch",
    "02": "2: Orientation",
    "03": "3: Lens Correct",
    "04": "4: Page Detect",
    "05": "5: Perspective",
    "06": "6: Content Area",
    "07": "7: Staff Extract",
    "08": "8: Deskew",
    "09": "9: Dewarp",
    "10": "10: Enhance",
    "11": "11: Normalize",
    "12": "12: OCR",
    "13": "13: OMR",
    "14": "14: Score Render",
    "15": "15: PDF",
}


def discover_book(
    output_dir: Path,
    input_dir: Path | None = None,
) -> dict:
    """Discover all images and stages in a pipeline output directory.

    Returns::

        {
            "stages": ["Original", "0: Preprocess", ...],
            "images": [
                {
                    "stem": "IMG_0012",
                    "stages": [
                        {"src": "file://...", "meta": {...}},
                        ...
                    ]
                },
                ...
            ]
        }

    Each image's ``stages`` list is aligned with the top-level
    ``stages`` list (``None`` if that stage has no output for this
    image).
    """
    checkpoint_dirs = sorted(
        d for d in output_dir.iterdir()
        if d.is_dir() and _CHECKPOINT_RE.match(d.name)
    )

    # Also discover branch subdirectories (book/, score/)
    for branch_name in ("book", "score"):
        branch_dir = output_dir / branch_name
        if branch_dir.is_dir():
            for d in sorted(branch_dir.iterdir()):
                if d.is_dir() and _CHECKPOINT_RE.match(d.name):
                    checkpoint_dirs.append(d)

    # Re-sort all checkpoint dirs by their numeric prefix
    checkpoint_dirs.sort(key=lambda d: (d.parent.name if d.parent != output_dir else "", d.name))

    stage_labels: list[str] = []
    stage_dirs: list[Path | None] = []

    if input_dir is not None and input_dir.is_dir():
        stage_labels.append("Original")
        stage_dirs.append(input_dir)

    for d in checkpoint_dirs:
        num = d.name[:2]
        base_label = _STAGE_LABELS.get(num, f"{int(num)}: {d.name[3:]}")
        # Qualify label with branch name when inside a branch subdir
        if d.parent.name in ("book", "score"):
            label = f"{base_label} [{d.parent.name}]"
        else:
            label = base_label
        stage_labels.append(label)
        stage_dirs.append(d)

    all_stems: dict[str, None] = {}
    for d in checkpoint_dirs:
        for f in d.iterdir():
            if f.suffix.lower() in _IMAGE_EXTS:
                all_stems[f.stem] = None

    stems = sorted(all_stems)

    images = []
    for stem in tqdm(stems, desc="Scanning images", unit="img"):
        stage_entries: list[dict | None] = []
        for d in stage_dirs:
            entry = _find_image_entry(d, stem) if d else None
            stage_entries.append(entry)
        if any(e is not None for e in stage_entries):
            images.append({"stem": stem, "stages": stage_entries})

    return {"stages": stage_labels, "images": images}


def _find_image_entry(
    directory: Path,
    stem: str,
) -> dict | None:
    """Find an image and its sidecar metadata in a directory."""
    for ext in (".png", ".jpg", ".jpeg", ".tiff", ".tif",
                ".PNG", ".JPG", ".JPEG"):
        candidate = directory / f"{stem}{ext}"
        if candidate.exists():
            sidecar = None
            sidecar_path = directory / f"{stem}.json"
            if sidecar_path.exists():
                try:
                    sidecar = json.loads(sidecar_path.read_text())
                except (json.JSONDecodeError, ValueError):
                    pass
            return {
                "src": f"file://{candidate.resolve()}",
                "meta": sidecar,
            }
    return None


def generate_compare_html(
    output_dir: Path,
    input_dir: Path | None = None,
    initial_stem: str | None = None,
) -> str:
    """Generate a full-book HTML comparison viewer (local/compare mode).

    If *initial_stem* is provided, the viewer opens at that image.
    """
    book = discover_book(output_dir, input_dir)
    initial_idx = 0
    if initial_stem:
        for i, img in enumerate(book["images"]):
            if img["stem"] == initial_stem:
                initial_idx = i
                break

    book_js = json.dumps(book, indent=None, separators=(",", ":"))

    title = output_dir.name
    if title.endswith("_output"):
        title = title[:-7]

    return _render_html(book_js, title, initial_idx, mode="compare")


def infer_input_dir(output_dir: Path) -> Path | None:
    """Try to infer the original input directory from the output path.

    Convention: output is ``<input>_output``.
    """
    name = output_dir.name
    if name.endswith("_output"):
        candidate = output_dir.parent / name[: -len("_output")]
        if candidate.is_dir():
            return candidate
    return None


def write_compare_html(
    output_dir: Path,
    input_dir: Path | None = None,
) -> Path:
    """Generate and write the comparison HTML to the output directory.

    Returns the path to the written HTML file.
    """
    if input_dir is None:
        input_dir = infer_input_dir(output_dir)

    html = generate_compare_html(output_dir, input_dir)
    html_path = output_dir / "compare.html"
    html_path.write_text(html)
    logger.info("Wrote comparison viewer: %s", html_path)
    return html_path


def publish_book(
    output_dir: Path,
    publish_dir: Path,
    input_dir: Path | None = None,
    max_dim: int = 1500,
    quality: int = 85,
    stage_filter: set[str] | None = None,
    extra_links: str = "",
) -> Path:
    """Publish a self-contained comparison site with downscaled JPEGs.

    *stage_filter*, if provided, is a set of stage number prefixes
    (e.g. ``{"00", "05", "07"}``) to include.  ``None`` means all.

    Returns the path to the written HTML file.
    """
    if input_dir is None:
        input_dir = infer_input_dir(output_dir)

    checkpoint_dirs = sorted(
        d for d in output_dir.iterdir()
        if d.is_dir() and _CHECKPOINT_RE.match(d.name)
    )

    # Also discover branch subdirectories (book/, score/)
    for branch_name in ("book", "score"):
        branch_dir = output_dir / branch_name
        if branch_dir.is_dir():
            for d in sorted(branch_dir.iterdir()):
                if d.is_dir() and _CHECKPOINT_RE.match(d.name):
                    checkpoint_dirs.append(d)

    checkpoint_dirs.sort(key=lambda d: (d.parent.name if d.parent != output_dir else "", d.name))

    stage_labels: list[str] = []
    stage_dirs: list[tuple[str, Path]] = []

    if input_dir is not None and input_dir.is_dir():
        include_original = stage_filter is None or "orig" in stage_filter
        if include_original:
            stage_labels.append("Original")
            stage_dirs.append(("original", input_dir))

    for d in checkpoint_dirs:
        num = d.name[:2]
        if stage_filter is not None and num not in stage_filter:
            continue
        base_label = _STAGE_LABELS.get(num, f"{int(num)}: {d.name[3:]}")
        if d.parent.name in ("book", "score"):
            label = f"{base_label} [{d.parent.name}]"
            folder_name = f"{d.parent.name}_{d.name}"
        else:
            label = base_label
            folder_name = d.name
        stage_labels.append(label)
        stage_dirs.append((folder_name, d))

    all_stems: dict[str, None] = {}
    for _, d in stage_dirs:
        for f in d.iterdir():
            if f.suffix.lower() in _IMAGE_EXTS:
                all_stems[f.stem] = None
    stems = sorted(all_stems)

    images_dir = publish_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    def _process_stem(stem: str) -> dict | None:
        stage_entries: list[dict | None] = []
        count = 0
        for folder_name, src_dir in stage_dirs:
            entry = _convert_image(
                src_dir, stem, images_dir, folder_name,
                max_dim, quality,
            )
            stage_entries.append(entry)
            if entry is not None:
                count += 1
        if any(e is not None for e in stage_entries):
            return {"stem": stem, "stages": stage_entries, "_n": count}
        return None

    images = []
    total = 0
    n_workers = max(1, (os.cpu_count() or 2) // 2)
    with ThreadPoolExecutor(max_workers=n_workers) as pool:
        for result in tqdm(
            pool.map(_process_stem, stems),
            total=len(stems), desc="Publishing", unit="img",
        ):
            if result is not None:
                total += result.pop("_n")
                images.append(result)

    book = {"stages": stage_labels, "images": images}
    book_js = json.dumps(book, indent=None, separators=(",", ":"))

    title = output_dir.name
    if title.endswith("_output"):
        title = title[:-7]

    html = _render_html(book_js, title, 0, mode="publish", extra_links=extra_links)

    html_path = publish_dir / "index.html"
    html_path.write_text(html)

    logger.info(
        "Published %d images (%d JPEGs) to %s",
        len(images), total, publish_dir,
    )
    return html_path


def _convert_image(
    src_dir: Path,
    stem: str,
    images_dir: Path,
    folder_name: str,
    max_dim: int,
    quality: int,
) -> dict | None:
    """Find, downscale, and write a JPEG; return entry dict or None."""
    src_path = None
    for ext in (".png", ".jpg", ".jpeg", ".tiff", ".tif",
                ".PNG", ".JPG", ".JPEG"):
        candidate = src_dir / f"{stem}{ext}"
        if candidate.exists():
            src_path = candidate
            break
    if src_path is None:
        return None

    out_subdir = images_dir / folder_name
    out_subdir.mkdir(exist_ok=True)
    out_path = out_subdir / f"{stem}.jpg"

    if not out_path.exists():
        img = cv2.imread(str(src_path), cv2.IMREAD_UNCHANGED)
        if img is None:
            return None

        h, w = img.shape[:2]
        longest = max(h, w)
        if longest > max_dim:
            scale = max_dim / longest
            img = cv2.resize(
                img, None, fx=scale, fy=scale,
                interpolation=cv2.INTER_AREA,
            )

        cv2.imwrite(
            str(out_path), img,
            [cv2.IMWRITE_JPEG_QUALITY, quality],
        )

    sidecar = None
    sidecar_path = src_dir / f"{stem}.json"
    if sidecar_path.exists():
        try:
            sidecar = json.loads(sidecar_path.read_text())
        except (json.JSONDecodeError, ValueError):
            pass

    rel = out_path.relative_to(out_path.parent.parent.parent)
    return {"src": str(rel), "meta": sidecar}


_THEME_COMPARE = {
    "bg":        "#1a1a2e",
    "topbar":    "#0d1b2a",
    "topborder": "#1b2838",
    "header":    "#16213e",
    "hborder":   "#0f3460",
    "accent":    "#8ecae6",
    "tabbg":     "#0f3460",
    "tabbdr":    "#1a5276",
    "active":    "#e94560",
    "btnbg":     "#1b2838",
    "btnbdr":    "#2a4a6b",
    "badge":     "#0f3460",
    "badgetxt":  "#8ecae6",
}

_THEME_PUBLISH = {
    "bg":        "#2e2517",
    "topbar":    "#2a1f0d",
    "topborder": "#38301b",
    "header":    "#3e2f16",
    "hborder":   "#604a0f",
    "accent":    "#e6c88e",
    "tabbg":     "#604a0f",
    "tabbdr":    "#76521a",
    "active":    "#e99445",
    "btnbg":     "#38301b",
    "btnbdr":    "#6b5a2a",
    "badge":     "#604a0f",
    "badgetxt":  "#e6c88e",
}


def _render_html(
    book_js: str,
    title: str,
    initial_idx: int,
    mode: str,
    extra_links: str = "",
) -> str:
    """Fill the HTML template with data and the appropriate theme."""
    theme = _THEME_PUBLISH if mode == "publish" else _THEME_COMPARE

    if mode == "publish":
        now = datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC")
        badge = f"Published {now}"
    else:
        badge = "Compare mode"

    html = _HTML_TEMPLATE
    html = html.replace("__BOOK_JSON__", book_js)
    html = html.replace("__TITLE__", title)
    html = html.replace("__INITIAL_IDX__", str(initial_idx))
    html = html.replace("__MODE_BADGE__", badge)
    html = html.replace("__EXTRA_LINKS__", extra_links)
    for key, val in theme.items():
        html = html.replace(f"__T_{key.upper()}__", val)
    return html


# noqa: E501 -- HTML template contains long JS lines by necessity
_HTML_TEMPLATE = """\
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Guido's Helping Hand — __TITLE__</title>
<style>
* { margin: 0; padding: 0; box-sizing: border-box; }
body {
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto,
    sans-serif;
  background: __T_BG__; color: #eee;
  display: flex; flex-direction: column; height: 100vh;
}

/* --- Top bar: image navigation --- */
.topbar {
  background: __T_TOPBAR__; padding: 6px 16px;
  display: flex; align-items: center; gap: 12px;
  border-bottom: 1px solid __T_TOPBORDER__;
  flex-shrink: 0;
}
.topbar h1 {
  font-size: 15px; font-weight: 600;
  color: __T_ACCENT__; white-space: nowrap;
}
.mode-badge {
  font-size: 11px; padding: 2px 10px; border-radius: 10px;
  background: __T_BADGE__; color: __T_BADGETXT__;
  border: 1px solid __T_TABBDR__;
  white-space: nowrap; user-select: none;
}
.nav-btn {
  background: __T_BTNBG__; border: 1px solid __T_BTNBDR__;
  color: __T_ACCENT__;
  padding: 4px 14px; border-radius: 4px; cursor: pointer;
  font-size: 13px; user-select: none;
}
.nav-btn:hover { background: __T_BTNBDR__; color: #fff; }
.nav-btn:disabled { opacity: 0.3; cursor: default; }
.img-counter {
  font-size: 13px; color: __T_ACCENT__; white-space: nowrap;
  min-width: 100px; text-align: center;
}
.img-select {
  background: __T_BTNBG__; color: __T_ACCENT__;
  border: 1px solid __T_BTNBDR__;
  padding: 3px 6px; border-radius: 4px; font-size: 12px;
  max-width: 180px;
}
.extra-link {
  display: inline-flex; align-items: center; gap: 4px;
  padding: 3px 10px; font-size: 12px; text-decoration: none;
  border-radius: 3px; margin-left: 4px;
  background: __T_BTNBG__; color: __T_ACCENT__;
  border: 1px solid __T_BTNBDR__;
}
.extra-link:hover { opacity: 0.85; }

/* --- Stage tabs --- */
header {
  background: __T_HEADER__; padding: 6px 16px;
  display: flex; align-items: center; gap: 12px;
  border-bottom: 1px solid __T_HBORDER__;
  flex-shrink: 0;
}
.tabs { display: flex; gap: 4px; flex-wrap: wrap; }
.tab {
  padding: 4px 12px; border-radius: 4px; cursor: pointer;
  background: __T_TABBG__; border: 1px solid __T_TABBDR__;
  font-size: 13px;
  color: #bbb; transition: all 0.15s; user-select: none;
}
.tab:hover { background: __T_TABBDR__; color: #fff; }
.tab.active {
  background: __T_ACTIVE__; color: #fff; border-color: __T_ACTIVE__;
}
.tab.active-left {
  background: #1a6b5a; color: #fff; border-color: #1a6b5a;
  box-shadow: inset 0 -3px 0 #4dd0b8;
}
.tab.active-right {
  background: #6b4a1a; color: #fff; border-color: #6b4a1a;
  box-shadow: inset 0 -3px 0 #e6a040;
}
.tab.active-both {
  background: #4a4a4a; color: #fff; border-color: #4a4a4a;
  box-shadow: inset 3px -3px 0 #4dd0b8, inset -3px -3px 0 #e6a040;
}
.tab.missing { opacity: 0.3; cursor: default; }
.controls {
  margin-left: auto; display: flex; gap: 8px;
  align-items: center; flex-shrink: 0;
}
.controls label { font-size: 12px; cursor: pointer; }
.controls input[type=checkbox] { margin-right: 4px; }

/* --- Main image area --- */
.main {
  flex: 1; position: relative;
  overflow: hidden; min-height: 0;
}
.view-single {
  width: 100%; height: 100%;
  display: flex; justify-content: center; align-items: center;
  padding: 8px;
}
.view-single img {
  max-width: 100%; max-height: 100%; object-fit: contain;
}
.view-single img.zoomed {
  max-width: none; max-height: none;
  cursor: grab; object-fit: initial;
}
.view-side {
  width: 100%; height: 100%; display: flex;
}
.view-side .pane {
  flex: 1; display: flex; flex-direction: column;
  border-right: 1px solid #333; overflow: hidden;
  min-height: 0;
}
.view-side .pane:last-child { border-right: none; }
.pane-body {
  flex: 1; min-height: 0;
  display: flex; justify-content: center; align-items: center;
  padding: 8px;
}
.pane-label {
  position: absolute; top: 8px; padding: 3px 10px; font-size: 11px;
  font-weight: 600; border-radius: 3px; z-index: 5;
  pointer-events: none; opacity: 0.9;
}
.pane-label-left {
  left: 8px; background: rgba(26,107,90,0.85); color: #4dd0b8;
  border: 1px solid #4dd0b8;
}
.pane-label-right {
  right: 8px; background: rgba(107,74,26,0.85); color: #e6a040;
  border: 1px solid #e6a040;
}
.pane-body {
  position: relative;
}
.pane-body img {
  max-width: 100%; max-height: 100%; object-fit: contain;
}
.pane-body img.zoomed {
  max-width: none; max-height: none;
  cursor: grab; object-fit: initial;
}

/* --- Metadata panel --- */
.meta-panel {
  position: absolute; right: 0; top: 0; bottom: 0;
  width: 320px; background: __T_HEADER__; padding: 12px;
  overflow-y: auto; border-left: 1px solid __T_HBORDER__;
  font-size: 12px; font-family: monospace;
  transform: translateX(100%); transition: transform 0.2s;
  z-index: 10;
}
.meta-panel.open { transform: translateX(0); }
.meta-panel h3 {
  font-size: 13px; margin-bottom: 8px; color: __T_ACTIVE__;
}
.meta-panel pre {
  white-space: pre-wrap; word-break: break-all;
  color: #aaa; line-height: 1.5;
}

/* --- Footer --- */
footer {
  background: __T_HEADER__; padding: 4px 16px;
  font-size: 11px; color: #aaa;
  border-top: 1px solid __T_HBORDER__; flex-shrink: 0;
  display: flex; justify-content: space-between; align-items: center;
}
footer a { color: #aaa; text-decoration: none; }
footer a:hover { color: #ddd; }
footer kbd {
  background: __T_TABBG__; padding: 1px 5px; border-radius: 3px;
  border: 1px solid __T_TABBDR__; font-size: 10px;
}
</style>
</head>
<body>

<div class="topbar">
  <h1 id="bookTitle">__TITLE__</h1>
  <span class="mode-badge">__MODE_BADGE__</span>
  <button class="nav-btn" id="prevImg">&laquo; Prev</button>
  <span class="img-counter" id="imgCounter"></span>
  <button class="nav-btn" id="nextImg">Next &raquo;</button>
  <select class="img-select" id="imgSelect"></select>
  __EXTRA_LINKS__
</div>

<header>
  <div class="tabs" id="tabs"></div>
  <div class="controls">
    <label>
      <input type="checkbox" id="sideToggle"> Side-by-side
    </label>
    <label>
      <input type="checkbox" id="metaToggle"> Metadata
    </label>
  </div>
</header>

<div class="main" id="main"></div>

<div class="meta-panel" id="metaPanel">
  <h3 id="metaTitle"></h3>
  <pre id="metaContent"></pre>
</div>

<footer>
  <span id="footer">
    <kbd>\u2191</kbd> <kbd>\u2193</kbd> prev/next image (stage 0) &nbsp;
    <kbd>PgUp</kbd> <kbd>PgDn</kbd> prev/next image (keep stage) &nbsp;
    <kbd>\u2190</kbd> <kbd>\u2192</kbd> prev/next stage &nbsp;
    <kbd>S</kbd> side-by-side &nbsp;
    <kbd>M</kbd> metadata &nbsp;
    <kbd>Z</kbd> zoom &nbsp;
    <kbd>1</kbd>\u2013<kbd>9</kbd> jump to stage
  </span>
  <span>Generated by
    <a href="https://pgarciaq.github.io/ghh/"
       target="_blank">Guido's Helping Hand</a></span>
</footer>

<script>
const BOOK = __BOOK_JSON__;
const STAGES = BOOK.stages;
const IMAGES = BOOK.images;

let imgIdx = __INITIAL_IDX__;
let imgIdxRight = __INITIAL_IDX__;
let stgIdx = 0;
let sideLeft = 0;
let sideRight = Math.min(1, STAGES.length - 1);
let isSide = false;
let isMeta = false;
let isZoomed = false;

const $ = (id) => document.getElementById(id);
const $tabs = $("tabs");
const $main = $("main");
const $metaPanel = $("metaPanel");
const $metaTitle = $("metaTitle");
const $metaContent = $("metaContent");
const $sideToggle = $("sideToggle");
const $metaToggle = $("metaToggle");
const $prevImg = $("prevImg");
const $nextImg = $("nextImg");
const $imgCounter = $("imgCounter");
const $imgSelect = $("imgSelect");

/* --- Image selector dropdown --- */
IMAGES.forEach((img, i) => {
  const opt = document.createElement("option");
  opt.value = i;
  opt.textContent = img.stem;
  $imgSelect.appendChild(opt);
});
$imgSelect.addEventListener("change", () => {
  setImage(parseInt($imgSelect.value));
});

$prevImg.addEventListener("click", () => setImage(imgIdx - 1));
$nextImg.addEventListener("click", () => setImage(imgIdx + 1));

function setImage(i) {
  imgIdx = clampImg(i);
  $imgSelect.value = imgIdx;
  updateImgCounter();
  buildTabs();
  render();
  updateMeta();
  pushHash();
}

function updateImgCounter() {
  if (isSide && imgIdx !== imgIdxRight) {
    $imgCounter.textContent =
      "L: " + IMAGES[imgIdx].stem +
      " (" + (imgIdx + 1) + "/" + IMAGES.length + ")" +
      "  R: " + IMAGES[imgIdxRight].stem +
      " (" + (imgIdxRight + 1) + "/" + IMAGES.length + ")";
  } else {
    $imgCounter.textContent = IMAGES[imgIdx].stem +
      " (" + (imgIdx + 1) + " / " + IMAGES.length + ")";
  }
  $prevImg.disabled = imgIdx === 0;
  $nextImg.disabled = imgIdx === IMAGES.length - 1;
}

function clampImg(i) {
  return Math.max(0, Math.min(i, IMAGES.length - 1));
}
function cur() { return IMAGES[imgIdx]; }
function stg(si) {
  const s = cur().stages[si];
  return s || null;
}

const FOOTER_SINGLE =
  '<kbd>\u2191</kbd> <kbd>\u2193</kbd> prev/next image (stage 0) &nbsp;' +
  '<kbd>PgUp</kbd> <kbd>PgDn</kbd> prev/next image (keep stage) &nbsp;' +
  '<kbd>\u2190</kbd> <kbd>\u2192</kbd> prev/next stage &nbsp;' +
  '<kbd>S</kbd> side-by-side &nbsp;' +
  '<kbd>M</kbd> metadata &nbsp;' +
  '<kbd>Z</kbd> zoom &nbsp;' +
  '<kbd>1</kbd>\u2013<kbd>9</kbd> jump to stage';
const FOOTER_SIDE =
  '<kbd>\u2191</kbd> <kbd>\u2193</kbd> left image (\u2192 stage 0) &nbsp;' +
  '<kbd>Shift+\u2191\u2193</kbd> right image (\u2192 stage 0) &nbsp;' +
  '<kbd>PgUp</kbd> <kbd>PgDn</kbd> left image (keep stage) &nbsp;' +
  '<kbd>Shift+PgUp/Dn</kbd> right image (keep stage) &nbsp;' +
  '<kbd>\u2190</kbd> <kbd>\u2192</kbd> left stage &nbsp;' +
  '<kbd>Shift+\u2190\u2192</kbd> right stage &nbsp;' +
  '<kbd>S</kbd> side-by-side &nbsp;' +
  '<kbd>Z</kbd> zoom &nbsp;' +
  '<kbd>1</kbd>\u2013<kbd>9</kbd> left &nbsp;' +
  '<kbd>Shift+1</kbd>\u2013<kbd>9</kbd> right';
const $footer = $("footer");

/* --- Stage tabs --- */
function buildTabs() {
  $tabs.innerHTML = "";
  STAGES.forEach((label, i) => {
    const t = document.createElement("div");
    const existsLeft = IMAGES[imgIdx].stages[i] != null;
    const existsRight = isSide
      ? IMAGES[imgIdxRight].stages[i] != null : false;
    const exists = isSide ? (existsLeft || existsRight) : (stg(i) !== null);
    let cls = "tab";
    if (isSide) {
      const isL = i === sideLeft;
      const isR = i === sideRight;
      if (isL && isR) cls += " active-both";
      else if (isL) cls += " active-left";
      else if (isR) cls += " active-right";
    } else {
      if (i === stgIdx) cls += " active";
    }
    if (!exists) cls += " missing";
    t.className = cls;
    t.textContent = label;
    if (exists) {
      t.addEventListener("click", (ev) => {
        if (isSide) {
          if (ev.shiftKey) { setSideRight(i); }
          else { setSideLeft(i); }
        } else {
          setStage(i);
        }
      });
    }
    $tabs.appendChild(t);
  });
}

function setStage(i) {
  stgIdx = Math.max(0, Math.min(i, STAGES.length - 1));
  buildTabs();
  render();
  updateMeta();
  pushHash();
}

function setSideLeft(i) {
  sideLeft = Math.max(0, Math.min(i, STAGES.length - 1));
  stgIdx = sideLeft;
  buildTabs();
  render();
  updateMeta();
  pushHash();
}

function setSideRight(i) {
  sideRight = Math.max(0, Math.min(i, STAGES.length - 1));
  buildTabs();
  render();
  updateMeta();
  pushHash();
}

/* --- Rendering --- */
function render() {
  if (isSide) renderSide(); else renderSingle();
}

function imgTag(si) {
  const s = stg(si);
  if (!s) return '<div style="color:#666;padding:40px">' +
    'No output for this stage</div>';
  return '<img src="' + s.src + '"' +
    (isZoomed ? ' class="zoomed"' : "") + ">";
}

function renderSingle() {
  $main.innerHTML =
    '<div class="view-single">' +
    imgTag(stgIdx) + "</div>";
}

function imgTagFor(imgI, stgI) {
  const s = IMAGES[imgI].stages[stgI];
  if (!s) return '<div style="color:#666;padding:40px">' +
    'No output for this stage</div>';
  return '<img src="' + s.src + '"' +
    (isZoomed ? ' class="zoomed"' : "") + ">";
}

function renderSide() {
  if (sideLeft >= STAGES.length) sideLeft = 0;
  if (sideRight >= STAGES.length)
    sideRight = Math.min(1, STAGES.length - 1);

  function pane(imgI, stgI, side) {
    const labelCls = "pane-label pane-label-" + side;
    const prefix = side === "left" ? "L" : "R";
    const stem = IMAGES[imgI].stem;
    const labelText = prefix + ": " + STAGES[stgI] + " — " + stem;
    return '<div class="pane">' +
      '<div class="pane-body">' +
      '<span class="' + labelCls + '">' + labelText + '</span>' +
      imgTagFor(imgI, stgI) +
      "</div></div>";
  }

  $main.innerHTML = '<div class="view-side">' +
    pane(imgIdx, sideLeft, "left") +
    pane(imgIdxRight, sideRight, "right") +
    "</div>";
}

function updateMeta() {
  const s = stg(stgIdx);
  $metaTitle.textContent = STAGES[stgIdx] +
    " — " + cur().stem;
  $metaContent.textContent = s && s.meta
    ? JSON.stringify(s.meta, null, 2) : "(no metadata)";
}

/* --- Toggles --- */
function toggleSide() {
  isSide = !isSide;
  $sideToggle.checked = isSide;
  if (isSide) {
    sideLeft = Math.max(0, stgIdx - 1);
    sideRight = stgIdx;
    imgIdxRight = imgIdx;
    isMeta = false;
    $metaToggle.checked = false;
    $metaPanel.classList.remove("open");
  }
  updateFooter();
  buildTabs();
  render();
  pushHash();
}
function toggleMeta() {
  isMeta = !isMeta;
  $metaToggle.checked = isMeta;
  $metaPanel.classList.toggle("open", isMeta);
}
function toggleZoom() { isZoomed = !isZoomed; render(); }

function updateFooter() {
  $footer.innerHTML = isSide ? FOOTER_SIDE : FOOTER_SINGLE;
}

$sideToggle.addEventListener("change", () => {
  isSide = $sideToggle.checked;
  if (isSide) {
    sideLeft = Math.max(0, stgIdx - 1);
    sideRight = stgIdx;
    imgIdxRight = imgIdx;
    isMeta = false;
    $metaToggle.checked = false;
    $metaPanel.classList.remove("open");
  }
  updateFooter();
  buildTabs();
  render();
  pushHash();
  $sideToggle.blur();
});
$metaToggle.addEventListener("change", () => {
  isMeta = $metaToggle.checked;
  $metaPanel.classList.toggle("open", isMeta);
  $metaToggle.blur();
});

/* --- Keyboard --- */
document.addEventListener("keydown", (e) => {
  const tag = (e.target || {}).tagName;
  if (tag === "SELECT" || tag === "INPUT") return;

  if (e.key === "ArrowLeft") {
    e.preventDefault();
    if (isSide) {
      if (e.shiftKey) setSideRight(sideRight - 1);
      else setSideLeft(sideLeft - 1);
    } else { setStage(stgIdx - 1); }
  } else if (e.key === "ArrowRight") {
    e.preventDefault();
    if (isSide) {
      if (e.shiftKey) setSideRight(sideRight + 1);
      else setSideLeft(sideLeft + 1);
    } else { setStage(stgIdx + 1); }
  } else if (e.key === "ArrowUp" || e.key === "ArrowDown") {
    e.preventDefault();
    const dir = e.key === "ArrowUp" ? -1 : 1;
    if (isSide) {
      if (e.shiftKey) {
        sideRight = 0;
        imgIdxRight = clampImg(imgIdxRight + dir);
        updateImgCounter(); buildTabs(); render(); updateMeta(); pushHash();
      } else {
        sideLeft = 0;
        setImage(imgIdx + dir);
      }
    } else {
      stgIdx = 0;
      setImage(imgIdx + dir);
    }
  } else if (e.key === "PageUp" || e.key === "PageDown") {
    e.preventDefault();
    const dir = e.key === "PageUp" ? -1 : 1;
    if (isSide) {
      if (e.shiftKey) {
        imgIdxRight = clampImg(imgIdxRight + dir);
        updateImgCounter(); buildTabs(); render(); updateMeta(); pushHash();
      } else {
        setImage(imgIdx + dir);
      }
    } else {
      setImage(imgIdx + dir);
    }
  } else if (e.key === "s" && !e.shiftKey) {
    toggleSide();
  } else if ((e.key === "m" || e.key === "M") && !e.shiftKey) {
    toggleMeta();
  } else if ((e.key === "z" || e.key === "Z") && !e.shiftKey) {
    toggleZoom();
  } else if (e.code && e.code >= "Digit1" && e.code <= "Digit9") {
    const idx = parseInt(e.code.charAt(5)) - 1;
    if (idx < STAGES.length && stg(idx)) {
      if (isSide) {
        if (e.shiftKey) setSideRight(idx);
        else setSideLeft(idx);
      } else {
        setStage(idx);
      }
    }
  }
});

/* --- Deep linking via URL hash --- */
let _suppressHash = false;
function pushHash() {
  let h;
  if (isSide) {
    h = "#" + IMAGES[imgIdx].stem + "/" + sideLeft +
        "+" + IMAGES[imgIdxRight].stem + "/" + sideRight;
  } else {
    h = "#" + IMAGES[imgIdx].stem + "/" + stgIdx;
  }
  if (location.hash !== h) {
    _suppressHash = true;
    history.replaceState(null, "", h);
  }
}
function findImg(stem) {
  return IMAGES.findIndex(img => img.stem === stem);
}
function clampStg(i) {
  return Math.max(0, Math.min(i, STAGES.length - 1));
}
function applyHash() {
  const h = location.hash;
  const sideMatch = h.match(
    /^#([^/+]+)[/](\\d+)\\+([^/+]+)[/](\\d+)$/);
  if (sideMatch) {
    const li = findImg(sideMatch[1]);
    const ri = findImg(sideMatch[3]);
    if (li < 0 || ri < 0) return;
    imgIdx = li;
    imgIdxRight = ri;
    sideLeft = clampStg(parseInt(sideMatch[2]));
    sideRight = clampStg(parseInt(sideMatch[4]));
    stgIdx = sideLeft;
    isSide = true;
    $sideToggle.checked = true;
    isMeta = false;
    $metaToggle.checked = false;
    $metaPanel.classList.remove("open");
    updateFooter();
    return;
  }
  const m = h.match(/^#([^/+]+)(?:[/](\\d+))?$/);
  if (!m) return;
  const idx = findImg(m[1]);
  if (idx < 0) return;
  imgIdx = idx;
  stgIdx = clampStg(m[2] != null ? parseInt(m[2]) : 0);
}
window.addEventListener("hashchange", () => {
  if (_suppressHash) { _suppressHash = false; return; }
  applyHash();
  $imgSelect.value = imgIdx;
  updateImgCounter();
  buildTabs();
  render();
  updateMeta();
});

/* --- Init --- */
applyHash();
setImage(imgIdx);
</script>
</body>
</html>
"""
