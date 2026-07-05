"""EXIF-aware image loading, lossless checkpoint saving, and directory management."""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path

import cv2
import numpy as np
from PIL import Image as PILImage
from PIL import ExifTags

logger = logging.getLogger(__name__)

# Pillow EXIF orientation tag ID
_ORIENTATION_TAG = 0x0112

# Map EXIF orientation values to the rotation needed to normalize them.
# After applying, the image should be in its "correct" visual orientation.
_EXIF_ROTATIONS: dict[int, int | None] = {
    1: None,               # Normal
    2: None,               # Mirrored horizontally (handled separately)
    3: cv2.ROTATE_180,     # Rotated 180
    4: None,               # Mirrored vertically (handled separately)
    5: cv2.ROTATE_90_COUNTERCLOCKWISE,  # Mirrored + 90 CW
    6: cv2.ROTATE_90_CLOCKWISE,         # Rotated 90 CW
    7: cv2.ROTATE_90_CLOCKWISE,         # Mirrored + 90 CCW
    8: cv2.ROTATE_90_COUNTERCLOCKWISE,  # Rotated 90 CCW
}

_EXIF_FLIPS: dict[int, int | None] = {
    1: None,
    2: 1,   # Flip horizontal
    3: None,
    4: 0,   # Flip vertical
    5: 1,   # Flip horizontal (before rotation)
    6: None,
    7: 1,   # Flip horizontal (before rotation)
    8: None,
}

# EXIF tags we want to extract
_WANTED_TAGS = {
    0x0112: "orientation",
    0x0110: "camera_model",
    0x9003: "datetime_original",
    0x011A: "x_resolution",
    0x011B: "y_resolution",
    0x0128: "resolution_unit",
    0x920A: "focal_length",
    0x8827: "iso",
    0x829A: "exposure_time",
}


def load_image(path: str | Path) -> tuple[np.ndarray, dict]:
    """Load an image, apply EXIF rotation, and extract metadata.

    Uses Pillow for EXIF extraction (cv2.imread strips it), then converts
    to a BGR numpy array for OpenCV processing.

    Returns:
        (image, metadata) where image is a BGR uint8 ndarray and metadata
        is a dict of extracted EXIF fields (empty dict if no EXIF).

    Raises:
        FileNotFoundError: If the path does not exist.
        OSError: If the file cannot be read as an image.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {path}")

    meta = {}

    try:
        pil_img = PILImage.open(path)
        exif_data = pil_img.getexif()

        for tag_id, field_name in _WANTED_TAGS.items():
            if tag_id in exif_data:
                val = exif_data[tag_id]
                # Convert IFDRational and similar types to plain Python types
                if hasattr(val, "numerator") and hasattr(val, "denominator"):
                    val = float(val)
                meta[field_name] = val

        # Convert Pillow image to numpy BGR
        if pil_img.mode == "L":
            rgb = pil_img.convert("RGB")
        elif pil_img.mode != "RGB":
            rgb = pil_img.convert("RGB")
        else:
            rgb = pil_img
        img = np.array(rgb)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    except Exception as exc:
        raise OSError(f"Cannot read image: {path} ({exc})") from exc

    orientation = meta.get("orientation", 1)

    # Apply EXIF flip
    flip_code = _EXIF_FLIPS.get(orientation)
    if flip_code is not None:
        img = cv2.flip(img, flip_code)

    # Apply EXIF rotation
    rotation = _EXIF_ROTATIONS.get(orientation)
    if rotation is not None:
        img = cv2.rotate(img, rotation)

    return img, meta


def save_checkpoint(
    img: np.ndarray,
    stage_dir: str | Path,
    filename: str,
    metadata: dict | None = None,
) -> Path:
    """Save an image as lossless PNG with atomic write.

    1. Writes to ``{filename}.tmp`` in ``stage_dir``
    2. Atomically renames to ``{filename}.png``
    3. If ``metadata`` is provided, writes ``{filename}.json`` sidecar

    Returns the final PNG path.
    """
    stage_dir = Path(stage_dir)
    stem = Path(filename).stem
    final_path = stage_dir / f"{stem}.png"
    tmp_path = stage_dir / f"{stem}.tmp.png"

    cv2.imwrite(str(tmp_path), img)
    os.replace(str(tmp_path), str(final_path))

    if metadata is not None:
        sidecar = stage_dir / f"{stem}.json"
        sidecar.write_text(json.dumps(metadata, indent=2, default=str))

    return final_path


def ensure_checkpoint_dir(output_dir: str | Path, stage_name: str) -> Path:
    """Create and return a checkpoint directory for a pipeline stage.

    Idempotent: safe to call multiple times.
    """
    d = Path(output_dir) / stage_name
    d.mkdir(parents=True, exist_ok=True)
    return d
