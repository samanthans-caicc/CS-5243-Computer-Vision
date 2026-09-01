"""Student implementations for A1. Do not add dependencies or use bypass libraries."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def image_summary(image: np.ndarray) -> dict[str, object]:
    """Return shape, height, width, channels, dtype, range, mean, and nbytes."""
    arr = np.asarray(image)
    if arr.ndim == 2:                       # grayscale: no channel axis
        height, width = arr.shape
        channels = 1
    elif arr.ndim == 3:                     # color: (H, W, C) with C == 3 or 4
        height, width, channels = arr.shape
    else:
        raise ValueError(f"Malformed image array. Expected a 2-D or 3-D image array, got shape {arr.shape!r}")
    return {
        "shape": tuple(int(n) for n in arr.shape),
        "height": int(height),
        "width": int(width),
        "channels": int(channels),
        "dtype": str(arr.dtype),
        "range": (arr.min().item(), arr.max().item()),
        "mean": float(arr.mean()),
        "nbytes": int(arr.nbytes),
    }


def _check_rectangle(image: np.ndarray, top: int, left: int, height: int, width: int) -> None:
    """Raise ValueError unless (top, left, height, width) is a positive, in-bounds rectangle."""
    if min(int(top), int(left)) < 0 or min(int(height), int(width)) <= 0:
        raise ValueError(f"rectangle needs origin >= 0 and positive size, got "
                         f"top={top}, left={left}, height={height}, width={width}")
    if top + height > image.shape[0] or left + width > image.shape[1]:
        raise ValueError(f"rectangle (top={top}, left={left}, h={height}, w={width}) "
                         f"is out of bounds for image {image.shape[:2]}")


def crop_image(image: np.ndarray, top: int, left: int, height: int, width: int) -> np.ndarray:
    """Return an independent NumPy crop; reject nonpositive or out-of-bounds rectangles."""
    arr = np.asarray(image)
    _check_rectangle(arr, top, left, height, width)
    return arr[top:top + height, left:left + width].copy()


def flip_horizontal(image: np.ndarray) -> np.ndarray:
    """Return a horizontally flipped view or copy using array operations."""
    return np.asarray(image)[:, ::-1].copy()


def extract_channel(image: np.ndarray, channel: int) -> np.ndarray:
    """Return one 2-D channel without changing its values."""
    arr = np.asarray(image)
    if arr.ndim != 3:
        raise ValueError(f"expected a 3-D (H, W, C) image, got shape {arr.shape}")
    if not 0 <= channel < arr.shape[2]:
        raise ValueError(f"channel {channel} out of range for {arr.shape[2]} channels")
    return arr[:, :, channel].copy()


def bgr_to_rgb(image: np.ndarray) -> np.ndarray:
    """Reorder a three-channel BGR array to RGB using NumPy."""
    arr = np.asarray(image)
    if arr.ndim != 3 or arr.shape[2] != 3:
        raise ValueError(f"expected a 3-channel image, got shape {arr.shape}")
    return arr[:, :, ::-1].copy()


def replace_region(image: np.ndarray, top: int, left: int, height: int, width: int,
                   value: int | Sequence[int]) -> np.ndarray:
    """Return a copy with a positive, in-bounds rectangle replaced."""
    arr = np.asarray(image)
    _check_rectangle(arr, top, left, height, width)
    out = arr.copy()
    out[top:top + height, left:left + width] = value
    return out


def contact_sheet(images: Sequence[np.ndarray], columns: int, fill_value: int = 0) -> np.ndarray:
    """Top-left-align all-grayscale or all-RGB uint8 images with an 8-pixel border/gutter."""
    items = [np.asarray(im) for im in images]
    if not items:
        raise ValueError("contact_sheet needs at least one image")
    if int(columns) <= 0:
        raise ValueError(f"columns must be positive, got {columns}")

    is_rgb = items[0].ndim == 3
    for im in items:
        if im.dtype != np.uint8:
            raise ValueError(f"all images must be uint8, got {im.dtype}")
        ok = (im.ndim == 3 and im.shape[2] == 3) if is_rgb else (im.ndim == 2)
        if not ok:
            raise ValueError("contact_sheet requires all-grayscale OR all-RGB uint8 images")

    border = gutter = 8
    cell_h = max(im.shape[0] for im in items)
    cell_w = max(im.shape[1] for im in items)
    rows = (len(items) + columns - 1) // columns
    height = 2 * border + rows * cell_h + (rows - 1) * gutter
    width = 2 * border + columns * cell_w + (columns - 1) * gutter
    shape = (height, width, 3) if is_rgb else (height, width)
    sheet = np.full(shape, fill_value, dtype=np.uint8)

    for idx, im in enumerate(items):
        r, c = divmod(idx, columns)
        y0 = border + r * (cell_h + gutter)
        x0 = border + c * (cell_w + gutter)
        sheet[y0:y0 + im.shape[0], x0:x0 + im.shape[1]] = im
    return sheet


def _check_uint8(image: np.ndarray) -> np.ndarray:
    """Return the image as an array, rejecting anything that is not uint8."""
    arr = np.asarray(image)
    if arr.dtype != np.uint8:
        raise ValueError(f"expected a uint8 image, got {arr.dtype}")
    return arr


def brighten_loop(image: np.ndarray, offset: float) -> np.ndarray:
    """Brighten uint8 values with loops, nearest-even rounding, and clipping to [0, 255]."""
    arr = _check_uint8(image)
    shift = float(offset)
    source = arr.reshape(-1)                      # one flat loop covers 2-D and 3-D alike
    result = np.empty(source.size, dtype=np.uint8)
    for i in range(source.size):
        value = float(np.rint(float(source[i]) + shift))   # round half to even, then clip
        if value < 0.0:
            value = 0.0
        elif value > 255.0:
            value = 255.0
        result[i] = value
    return result.reshape(arr.shape)


def brighten_vectorized(image: np.ndarray, offset: float) -> np.ndarray:
    """Brighten uint8 values vectorially, nearest-even rounding, and clipping to [0, 255]."""
    arr = _check_uint8(image)
    # float64 matches the loop's Python-float arithmetic, so both paths round identically
    shifted = arr.astype(np.float64) + float(offset)
    return np.clip(np.rint(shifted), 0.0, 255.0).astype(np.uint8)


def to_float01(image: np.ndarray) -> np.ndarray:
    """Convert uint8 or uint16 values to float32 in [0, 1]."""
    arr = np.asarray(image)
    if arr.dtype not in (np.uint8, np.uint16):
        raise ValueError(f"expected a uint8 or uint16 image, got {arr.dtype}")
    # divide by the dtype maximum, not by the observed maximum: the scale is a property
    # of the representation, so the same pixel maps to the same float in every image
    return (arr.astype(np.float32) / np.float32(np.iinfo(arr.dtype).max)).astype(np.float32)


def to_uint8_safe(image: np.ndarray) -> np.ndarray:
    """Convert floating [0, 1] values to uint8 with clipping and rounding."""
    arr = np.asarray(image)
    if arr.dtype.kind != "f":
        raise ValueError(f"expected a floating-point image, got {arr.dtype}")
    # clip BEFORE scaling so out-of-range values saturate instead of wrapping,
    # and round in float64 so k/255 -> k round-trips exactly
    scaled = np.clip(arr.astype(np.float64), 0.0, 1.0) * 255.0
    return np.rint(scaled).astype(np.uint8)


def _check_rgb(image: np.ndarray) -> np.ndarray:
    """Return the image as an array, rejecting anything that is not (H, W, 3)."""
    arr = np.asarray(image)
    if arr.ndim != 3 or arr.shape[2] != 3:
        raise ValueError(f"expected an (H, W, 3) RGB image, got shape {arr.shape}")
    return arr


def grayscale_mean(image_rgb: np.ndarray) -> np.ndarray:
    """Return the unweighted channel mean as float32."""
    arr = _check_rgb(image_rgb)
    # float32 accumulation, so the caller's value scale (0-255 or 0-1) is preserved
    return arr.astype(np.float32).mean(axis=2, dtype=np.float32)


def grayscale_luminance(image_rgb: np.ndarray) -> np.ndarray:
    """Return 0.299R + 0.587G + 0.114B as float32."""
    arr = _check_rgb(image_rgb).astype(np.float32)
    weights = np.array([0.299, 0.587, 0.114], dtype=np.float32)   # ITU-R BT.601 luma
    return (arr @ weights).astype(np.float32)


def hsv_rule(hsv_image: np.ndarray, lower: Sequence[int], upper: Sequence[int]) -> np.ndarray:
    """Return a boolean mask for inclusive OpenCV-HSV bounds, including wrapped hue ranges."""
    arr = _check_rgb(hsv_image)
    low = np.asarray(lower, dtype=np.int32)
    high = np.asarray(upper, dtype=np.int32)
    if low.shape != (3,) or high.shape != (3,):
        raise ValueError(f"lower and upper must each hold 3 values, got {low.shape} and {high.shape}")
    hue, sat, val = (arr[:, :, i].astype(np.int32) for i in range(3))

    # saturation and value are ordinary inclusive intervals
    mask = (sat >= low[1]) & (sat <= high[1]) & (val >= low[2]) & (val <= high[2])
    if low[0] <= high[0]:
        hue_mask = (hue >= low[0]) & (hue <= high[0])
    else:
        # OpenCV hue is 0-179 and circular: lower > upper means the interval crosses 0
        # (for example 170 -> 10 is "red"), so accept either side of the wrap point
        hue_mask = (hue >= low[0]) | (hue <= high[0])
    return mask & hue_mask


def transform_frame(frame_rgb: np.ndarray) -> np.ndarray:
    """Flip RGB horizontally and multiply red by 0.65, preserving HxWx3 uint8."""
    arr = _check_rgb(frame_rgb)
    if arr.dtype != np.uint8:
        raise ValueError(f"expected a uint8 frame, got {arr.dtype}")
    out = arr[:, ::-1, :].astype(np.float64)
    out[:, :, 0] *= 0.65                                  # scale red only
    return np.clip(np.rint(out), 0.0, 255.0).astype(np.uint8)
