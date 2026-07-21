"""Shared image utilities used across multiple pipeline stages.

Provides background color estimation and post-geometry content trimming.
Used by Stage 5 (perspective), Stage 8 (deskew), and Stage 9 (dewarp).
"""

from __future__ import annotations

import numpy as np


def estimate_background(
    img: np.ndarray,
    border_frac: float = 0.05,
) -> tuple[int, ...]:
    """Estimate background color from border pixels.

    Samples the outermost *border_frac* (default 5%) of the image on
    all four sides and returns the per-channel median.  Avoids
    double-counting the four corner rectangles by only including the
    left/right strips from the interior rows.

    Returns a tuple of ints (B, G, R) for color images, or (gray,)
    for grayscale.
    """
    h, w = img.shape[:2]
    bh = max(1, int(h * border_frac))
    bw = max(1, int(w * border_frac))

    strips = [
        img[:bh, :],
        img[-bh:, :],
    ]
    if h > 2 * bh:
        strips.append(img[bh:-bh, :bw])
        strips.append(img[bh:-bh, -bw:])

    if img.ndim == 3:
        pixels = np.concatenate([s.reshape(-1, img.shape[2]) for s in strips])
        return tuple(int(np.median(pixels[:, c])) for c in range(img.shape[2]))

    pixels = np.concatenate([s.ravel() for s in strips])
    return (int(np.median(pixels)),)


def trim_to_content(
    img: np.ndarray,
    bg_color: tuple[int, ...] | None = None,
    margin_frac: float = 0.0,
    threshold: int = 30,
) -> np.ndarray:
    """Trim background-colored borders and add uniform margin padding.

    1. Estimate bg_color if not provided.
    2. Compute per-pixel L1 distance from bg_color.
    3. Find bounding rect of pixels exceeding *threshold*.
    4. Crop to that rect.
    5. Add uniform margin (margin_frac * width) filled with bg_color.

    Returns the trimmed image.  If no content is found (entire image
    is background), returns the original image unchanged.
    """
    if bg_color is None:
        bg_color = estimate_background(img)

    bg_arr = np.array(bg_color, dtype=np.float32)
    if img.ndim == 3:
        diff = np.sum(np.abs(img.astype(np.float32) - bg_arr), axis=2)
    else:
        diff = np.abs(img.astype(np.float32) - bg_arr[0])

    mask = diff > threshold
    coords = np.argwhere(mask)
    if coords.size == 0:
        return img

    y0, x0 = coords.min(axis=0)
    y1, x1 = coords.max(axis=0) + 1

    cropped = img[y0:y1, x0:x1]

    margin = max(1, int(cropped.shape[1] * margin_frac))

    if img.ndim == 3:
        padded = np.full(
            (cropped.shape[0] + 2 * margin, cropped.shape[1] + 2 * margin, img.shape[2]),
            bg_color,
            dtype=img.dtype,
        )
    else:
        padded = np.full(
            (cropped.shape[0] + 2 * margin, cropped.shape[1] + 2 * margin),
            bg_color[0],
            dtype=img.dtype,
        )

    padded[margin : margin + cropped.shape[0], margin : margin + cropped.shape[1]] = cropped
    return padded
