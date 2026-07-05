"""Generate a self-contained static HTML flipbook from processed images.

Uses StPageFlip (MIT license) for realistic page-turning animation.
The output is a directory uploadable to any static hosting without
server-side dependencies.
"""

from __future__ import annotations

import json
import logging
import re
import shutil
from datetime import UTC, datetime
from importlib import resources
from pathlib import Path

import cv2

logger = logging.getLogger(__name__)

_CHECKPOINT_RE = re.compile(r"^(\d{2})_")
_IMAGE_EXTS = frozenset((".png", ".jpg", ".jpeg", ".tiff", ".tif"))


def _vendor_js_path() -> Path:
    """Return the path to the vendored page-flip.browser.js."""
    ref = resources.files("ghh.vendor") / "page-flip.browser.js"
    return Path(str(ref))


def _find_source_images(output_dir: Path) -> list[Path]:
    """Find images from the latest completed checkpoint in output_dir.

    Walks backward from the highest-numbered checkpoint directory.
    Returns sorted list of image paths.
    """
    checkpoint_dirs = sorted(
        (d for d in output_dir.iterdir() if d.is_dir() and _CHECKPOINT_RE.match(d.name)),
        reverse=True,
    )

    for d in checkpoint_dirs:
        images = sorted(
            f for f in d.iterdir() if f.is_file() and f.suffix.lower() in _IMAGE_EXTS
        )
        if images:
            logger.info("Using images from checkpoint: %s (%d images)", d.name, len(images))
            return images

    return []


def _find_pdf(output_dir: Path) -> Path | None:
    """Find the PDF in the output directory (produced by Stage 12)."""
    for f in output_dir.iterdir():
        if f.is_file() and f.suffix.lower() == ".pdf":
            return f
    return None


def _downscale_image(src: Path, dst: Path, max_width: int, jpeg_quality: int) -> tuple[int, int]:
    """Read src image, downscale if wider than max_width, save as JPEG to dst.

    Returns (width, height) of the saved image.
    """
    img = cv2.imread(str(src), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"Could not read image: {src}")

    h, w = img.shape[:2]
    if w > max_width:
        scale = max_width / w
        new_w = max_width
        new_h = int(h * scale)
        img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
        h, w = new_h, new_w

    cv2.imwrite(str(dst), img, [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality])
    return w, h


def generate_flipbook(
    output_dir: Path,
    flipbook_dir: Path | None = None,
    *,
    max_width: int = 1600,
    jpeg_quality: int = 85,
    title: str = "",
    include_pdf: bool = True,
    show_cover: bool = False,
) -> Path:
    """Generate a flipbook from pipeline output.

    Args:
        output_dir: Pipeline output directory containing checkpoint dirs.
        flipbook_dir: Where to write the flipbook. Defaults to output_dir/flipbook/.
        max_width: Maximum page width in pixels.
        jpeg_quality: JPEG compression quality (1-100).
        title: Title displayed in the viewer.
        include_pdf: Whether to include a PDF download link.
        show_cover: If True, the first image is treated as a standalone cover
            (displayed alone on the right before flipping). If False (default),
            page 1 starts on the left as a normal interior page.

    Returns:
        Path to the generated index.html.
    """
    if flipbook_dir is None:
        flipbook_dir = output_dir / "flipbook"

    flipbook_dir.mkdir(parents=True, exist_ok=True)
    pages_dir = flipbook_dir / "pages"
    pages_dir.mkdir(exist_ok=True)

    source_images = _find_source_images(output_dir)
    if not source_images:
        raise FileNotFoundError(
            f"No checkpoint images found in {output_dir}. "
            "Run the pipeline first to generate images."
        )

    page_paths: list[str] = []
    dimensions: list[tuple[int, int]] = []

    for i, src in enumerate(source_images, start=1):
        dst = pages_dir / f"{i:03d}.jpg"
        w, h = _downscale_image(src, dst, max_width, jpeg_quality)
        page_paths.append(f"pages/{dst.name}")
        dimensions.append((w, h))

    js_src = _vendor_js_path()
    js_dst = flipbook_dir / "page-flip.browser.js"
    shutil.copy2(js_src, js_dst)

    pdf_filename: str | None = None
    if include_pdf:
        pdf_src = _find_pdf(output_dir)
        if pdf_src is not None:
            pdf_filename = pdf_src.name
            shutil.copy2(pdf_src, flipbook_dir / pdf_filename)
            logger.info("Copied PDF: %s", pdf_src.name)
        else:
            logger.warning("No PDF found in %s/12_pdf/; omitting download link", output_dir)

    if dimensions:
        med_idx = len(dimensions) // 2
        page_w, page_h = dimensions[med_idx]
    else:
        page_w, page_h = 800, 1200

    html = _render_html(
        page_paths=page_paths,
        title=title,
        pdf_filename=pdf_filename,
        page_width=page_w,
        page_height=page_h,
        show_cover=show_cover,
    )

    index_path = flipbook_dir / "index.html"
    index_path.write_text(html)

    metadata = {
        "generated_at": datetime.now(UTC).isoformat(),
        "page_count": len(page_paths),
        "max_dimensions": {"width": page_w, "height": page_h},
        "jpeg_quality": jpeg_quality,
        "max_width_setting": max_width,
        "title": title,
        "has_pdf": pdf_filename is not None,
        "total_size_bytes": sum(
            f.stat().st_size for f in flipbook_dir.rglob("*") if f.is_file()
        ),
    }
    (flipbook_dir / "flipbook.json").write_text(json.dumps(metadata, indent=2))

    logger.info(
        "Generated flipbook: %d pages, %s",
        len(page_paths),
        index_path,
    )

    return index_path


def _render_html(
    page_paths: list[str],
    title: str,
    pdf_filename: str | None,
    page_width: int,
    page_height: int,
    show_cover: bool = False,
) -> str:
    """Render the flipbook HTML from template."""
    pages_json = json.dumps(page_paths)

    pdf_button = ""
    if pdf_filename:
        pdf_button = (
            f'<a href="{pdf_filename}" download class="pdf-btn">'
            "&#x1F4E5; Download PDF</a>"
        )

    title_html = f"<h1>{title}</h1>" if title else ""

    return _FLIPBOOK_HTML_TEMPLATE.replace("{{TITLE_TAG}}", title or "Flipbook").replace(
        "{{TITLE_HTML}}", title_html
    ).replace(
        "{{PDF_BUTTON}}", pdf_button
    ).replace(
        "{{PAGES_JSON}}", pages_json
    ).replace(
        "{{PAGE_WIDTH}}", str(page_width)
    ).replace(
        "{{PAGE_HEIGHT}}", str(page_height)
    ).replace(
        "{{PAGE_COUNT}}", str(len(page_paths))
    ).replace(
        "{{SHOW_COVER}}", "true" if show_cover else "false"
    )


_FLIPBOOK_HTML_TEMPLATE = """\
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>{{TITLE_TAG}}</title>
<style>
* { margin: 0; padding: 0; box-sizing: border-box; }
body {
    background: #1a1a2e;
    color: #eee;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    display: flex;
    flex-direction: column;
    height: 100vh;
    overflow: hidden;
}
header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 0.5rem 1.5rem;
    background: #16213e;
    border-bottom: 1px solid #0f3460;
    flex-shrink: 0;
}
header h1 {
    font-size: 1.1rem;
    font-weight: 500;
    color: #e0e0e0;
}
.header-controls {
    display: flex;
    align-items: center;
    gap: 1rem;
}
.pdf-btn {
    display: inline-flex;
    align-items: center;
    gap: 0.4rem;
    padding: 0.4rem 0.9rem;
    background: #0f3460;
    color: #90caf9;
    text-decoration: none;
    border-radius: 4px;
    font-size: 0.85rem;
    transition: background 0.2s;
}
.pdf-btn:hover { background: #1a4f8a; }
.page-indicator {
    font-size: 0.85rem;
    color: #90caf9;
    min-width: 8rem;
    text-align: center;
}
#flipbook-container {
    flex: 1;
    display: flex;
    align-items: center;
    justify-content: center;
    padding: 1rem;
    min-height: 0;
}
#flipbook {
    box-shadow: 0 8px 40px rgba(0, 0, 0, 0.5);
}
.nav-controls {
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 1rem;
    padding: 0.6rem;
    background: #16213e;
    border-top: 1px solid #0f3460;
    flex-shrink: 0;
}
.nav-btn {
    padding: 0.4rem 1rem;
    background: #0f3460;
    color: #90caf9;
    border: none;
    border-radius: 4px;
    cursor: pointer;
    font-size: 0.85rem;
    transition: background 0.2s;
}
.nav-btn:hover { background: #1a4f8a; }
.nav-btn:disabled { opacity: 0.4; cursor: default; }
footer {
    text-align: center;
    padding: 0.3rem;
    font-size: 0.7rem;
    color: #999;
    flex-shrink: 0;
}
</style>
</head>
<body>
<header>
    <div>{{TITLE_HTML}}</div>
    <div class="header-controls">
        <span class="page-indicator" id="pageIndicator">Page 1 of {{PAGE_COUNT}}</span>
        {{PDF_BUTTON}}
    </div>
</header>

<div id="flipbook-container">
    <div id="flipbook"></div>
</div>

<div class="nav-controls">
    <button class="nav-btn" id="btnFirst" title="Home">&#x23EE; First</button>
    <button class="nav-btn" id="btnPrev" title="Left arrow">&#x25C0; Prev</button>
    <button class="nav-btn" id="btnNext" title="Right arrow">Next &#x25B6;</button>
    <button class="nav-btn" id="btnLast" title="End">Last &#x23ED;</button>
</div>

<footer>
    Generated by Guido's Helping Hand &middot; StPageFlip (MIT)
</footer>

<script src="page-flip.browser.js"></script>
<script>
(function() {
    const pages = {{PAGES_JSON}};
    const pageCount = pages.length;
    const container = document.getElementById('flipbook');
    const indicator = document.getElementById('pageIndicator');

    const containerEl = document.getElementById('flipbook-container');
    const availW = containerEl.clientWidth - 40;
    const availH = containerEl.clientHeight - 40;

    const pageAspect = {{PAGE_WIDTH}} / {{PAGE_HEIGHT}};
    let baseH = Math.min({{PAGE_HEIGHT}}, availH);
    let baseW = Math.round(baseH * pageAspect);
    if (baseW * 2 > availW) {
        baseW = Math.floor(availW / 2);
        baseH = Math.round(baseW / pageAspect);
    }

    const pageFlip = new St.PageFlip(container, {
        width: baseW,
        height: baseH,
        size: "fixed",
        minWidth: 200,
        maxWidth: baseW,
        minHeight: 300,
        maxHeight: baseH,
        showCover: {{SHOW_COVER}},
        maxShadowOpacity: 0.5,
        mobileScrollSupport: false,
        flippingTime: 800,
        usePortrait: true,
        startZIndex: 0,
        autoSize: true,
        drawShadow: true,
        startPage: 0,
    });

    pageFlip.loadFromImages(pages);

    function updateIndicator() {
        const current = pageFlip.getCurrentPageIndex() + 1;
        indicator.textContent = 'Page ' + current + ' of ' + pageCount;
    }
    pageFlip.on('flip', updateIndicator);
    updateIndicator();

    document.getElementById('btnFirst').addEventListener('click', function() {
        pageFlip.flip(0);
    });
    document.getElementById('btnPrev').addEventListener('click', function() {
        pageFlip.flipPrev();
    });
    document.getElementById('btnNext').addEventListener('click', function() {
        pageFlip.flipNext();
    });
    document.getElementById('btnLast').addEventListener('click', function() {
        pageFlip.flip(pageCount - 1);
    });

    document.addEventListener('keydown', function(e) {
        switch(e.key) {
            case 'ArrowLeft':
                pageFlip.flipPrev();
                e.preventDefault();
                break;
            case 'ArrowRight':
                pageFlip.flipNext();
                e.preventDefault();
                break;
            case 'Home':
                pageFlip.flip(0);
                e.preventDefault();
                break;
            case 'End':
                pageFlip.flip(pageCount - 1);
                e.preventDefault();
                break;
        }
    });
})();
</script>
</body>
</html>
"""
