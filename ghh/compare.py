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
import re
from datetime import UTC, datetime
from pathlib import Path

import cv2

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
    "07": "7: Deskew",
    "08": "8: Dewarp",
    "09": "9: Enhance",
    "10": "10: Normalize",
    "11": "11: OCR",
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

    stage_labels: list[str] = []
    stage_dirs: list[Path | None] = []

    if input_dir is not None and input_dir.is_dir():
        stage_labels.append("Original")
        stage_dirs.append(input_dir)

    for d in checkpoint_dirs:
        num = d.name[:2]
        label = _STAGE_LABELS.get(num, f"{int(num)}: {d.name[3:]}")
        stage_labels.append(label)
        stage_dirs.append(d)

    all_stems: dict[str, None] = {}
    for d in checkpoint_dirs:
        for f in d.iterdir():
            if f.suffix.lower() in _IMAGE_EXTS:
                all_stems[f.stem] = None

    stems = sorted(all_stems)

    images = []
    for stem in stems:
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
        label = _STAGE_LABELS.get(num, f"{int(num)}: {d.name[3:]}")
        stage_labels.append(label)
        stage_dirs.append((d.name, d))

    all_stems: dict[str, None] = {}
    for _, d in stage_dirs:
        for f in d.iterdir():
            if f.suffix.lower() in _IMAGE_EXTS:
                all_stems[f.stem] = None
    stems = sorted(all_stems)

    images_dir = publish_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    images = []
    total = 0
    for stem in stems:
        stage_entries: list[dict | None] = []
        for folder_name, src_dir in stage_dirs:
            entry = _convert_image(
                src_dir, stem, images_dir, folder_name,
                max_dim, quality,
            )
            stage_entries.append(entry)
            if entry is not None:
                total += 1

        if any(e is not None for e in stage_entries):
            images.append({"stem": stem, "stages": stage_entries})

    book = {"stages": stage_labels, "images": images}
    book_js = json.dumps(book, indent=None, separators=(",", ":"))

    title = output_dir.name
    if title.endswith("_output"):
        title = title[:-7]

    html = _render_html(book_js, title, 0, mode="publish")

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
.pane-header {
  background: __T_HEADER__; padding: 4px 8px; font-size: 12px;
  font-weight: 600; text-align: center;
  border-bottom: 1px solid #333;
  display: flex; gap: 3px; justify-content: center; flex-wrap: wrap;
  flex-shrink: 0;
}
.pane-header .tab { padding: 2px 8px; font-size: 11px; }
.pane-body {
  flex: 1; min-height: 0;
  display: flex; justify-content: center; align-items: center;
  padding: 8px;
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
  font-size: 11px; color: #666;
  border-top: 1px solid __T_HBORDER__; flex-shrink: 0;
}
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
  <kbd>PgUp</kbd> <kbd>PgDn</kbd> prev/next image &nbsp;
  <kbd>\u2190</kbd> <kbd>\u2192</kbd> prev/next stage &nbsp;
  <kbd>S</kbd> side-by-side &nbsp;
  <kbd>M</kbd> metadata &nbsp;
  <kbd>Z</kbd> zoom &nbsp;
  <kbd>1</kbd>\u2013<kbd>9</kbd> jump to stage
</footer>

<script>
const BOOK = __BOOK_JSON__;
const STAGES = BOOK.stages;
const IMAGES = BOOK.images;

let imgIdx = __INITIAL_IDX__;
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
  imgIdx = Math.max(0, Math.min(i, IMAGES.length - 1));
  $imgSelect.value = imgIdx;
  $imgCounter.textContent = IMAGES[imgIdx].stem +
    " (" + (imgIdx + 1) + " / " + IMAGES.length + ")";
  $prevImg.disabled = imgIdx === 0;
  $nextImg.disabled = imgIdx === IMAGES.length - 1;
  buildTabs();
  render();
  updateMeta();
}

function cur() { return IMAGES[imgIdx]; }
function stg(si) {
  const s = cur().stages[si];
  return s || null;
}

/* --- Stage tabs --- */
function buildTabs() {
  $tabs.innerHTML = "";
  STAGES.forEach((label, i) => {
    const t = document.createElement("div");
    const exists = stg(i) !== null;
    let cls = "tab";
    if (i === stgIdx) cls += " active";
    if (!exists) cls += " missing";
    t.className = cls;
    t.textContent = label;
    if (exists) t.onclick = () => setStage(i);
    $tabs.appendChild(t);
  });
}

function setStage(i) {
  stgIdx = Math.max(0, Math.min(i, STAGES.length - 1));
  buildTabs();
  render();
  updateMeta();
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

function renderSide() {
  if (sideLeft >= STAGES.length) sideLeft = 0;
  if (sideRight >= STAGES.length)
    sideRight = Math.min(1, STAGES.length - 1);

  function pane(idx, side) {
    let hdr = "";
    STAGES.forEach((label, i) => {
      const exists = stg(i) !== null;
      let cls = "tab";
      if (i === idx) cls += " active";
      if (!exists) cls += " missing";
      hdr += '<div class="' + cls + '"' +
        (exists ? ' data-side="' + side +
        '" data-idx="' + i + '"' : "") +
        ">" + label + "</div>";
    });
    return '<div class="pane">' +
      '<div class="pane-header">' + hdr + "</div>" +
      '<div class="pane-body">' + imgTag(idx) +
      "</div></div>";
  }

  $main.innerHTML = '<div class="view-side">' +
    pane(sideLeft, "left") + pane(sideRight, "right") +
    "</div>";

  $main.querySelectorAll(".pane-header .tab[data-idx]")
    .forEach(t => {
      t.addEventListener("click", () => {
        const side = t.dataset.side;
        const idx = parseInt(t.dataset.idx);
        if (side === "left") sideLeft = idx;
        else sideRight = idx;
        stgIdx = idx;
        buildTabs();
        renderSide();
        updateMeta();
      });
    });
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
  }
  render();
}
function toggleMeta() {
  isMeta = !isMeta;
  $metaToggle.checked = isMeta;
  $metaPanel.classList.toggle("open", isMeta);
}
function toggleZoom() { isZoomed = !isZoomed; render(); }

$sideToggle.addEventListener("change", () => {
  isSide = $sideToggle.checked; render();
});
$metaToggle.addEventListener("change", () => {
  isMeta = $metaToggle.checked;
  $metaPanel.classList.toggle("open", isMeta);
});

/* --- Keyboard --- */
document.addEventListener("keydown", (e) => {
  const tag = (e.target || {}).tagName;
  if (tag === "SELECT" || tag === "INPUT") return;

  if (e.key === "ArrowLeft") {
    setStage(stgIdx - 1); e.preventDefault();
  } else if (e.key === "ArrowRight") {
    setStage(stgIdx + 1); e.preventDefault();
  } else if (e.key === "PageUp") {
    stgIdx = 0; setImage(imgIdx - 1); e.preventDefault();
  } else if (e.key === "PageDown") {
    stgIdx = 0; setImage(imgIdx + 1); e.preventDefault();
  } else if (e.key === "s" || e.key === "S") {
    toggleSide();
  } else if (e.key === "m" || e.key === "M") {
    toggleMeta();
  } else if (e.key === "z" || e.key === "Z") {
    toggleZoom();
  } else if (e.key >= "1" && e.key <= "9") {
    const idx = parseInt(e.key) - 1;
    if (idx < STAGES.length && stg(idx)) setStage(idx);
  }
});

/* --- Init --- */
setImage(imgIdx);
</script>
</body>
</html>
"""
