"""Student implementations for A2. Do not add dependencies or bypass required work."""

from __future__ import annotations

from collections.abc import Sequence
import math
import cv2
import numpy as np


_PAD_MODES = {"reflect": "reflect", "edge": "edge", "constant": "constant"}


def _pad_spatial(image: np.ndarray, radius: int, border_mode: str) -> np.ndarray:
    """Pad only the two spatial axes of a 2-D or HxWxC array."""
    if border_mode not in _PAD_MODES:
        raise ValueError(f"Unsupported border_mode {border_mode!r}; use reflect, edge, or constant")
    pad = [(radius, radius), (radius, radius)] + [(0, 0)] * (image.ndim - 2)
    if border_mode == "constant":
        return np.pad(image, pad, mode="constant", constant_values=0)
    return np.pad(image, pad, mode=_PAD_MODES[border_mode])


def _check_image(image: np.ndarray) -> np.ndarray:
    a = np.asarray(image, dtype=np.float32)
    if a.ndim not in (2, 3):
        raise ValueError("image must be 2-D (H, W) or 3-D (H, W, C)")
    return a


def _check_odd_size(size: int, name: str = "kernel size") -> int:
    if not isinstance(size, (int, np.integer)) or isinstance(size, bool):
        raise ValueError(f"{name} must be an integer")
    if size <= 0 or size % 2 == 0:
        raise ValueError(f"{name} must be a positive odd integer, got {size}")
    return int(size)


def convolve2d_manual(image: np.ndarray, kernel: np.ndarray, border_mode: str = "reflect") -> np.ndarray:
    """Convolve a 2-D or HxWxC image with an odd 2-D kernel; return float32.

    Apply true mathematical convolution, including spatial reversal of the kernel.
    Support ``reflect``, ``edge``, and ``constant`` borders. Do not delegate the
    core operation to a library convolution, correlation, or neural-network layer.
    """
    img = _check_image(image)
    k = np.asarray(kernel, dtype=np.float32)
    if k.ndim != 2:
        raise ValueError("kernel must be 2-D")
    kh, kw = k.shape
    _check_odd_size(kh, "kernel height")
    _check_odd_size(kw, "kernel width")
    rh, rw = kh // 2, kw // 2
    radius = max(rh, rw)
    padded = _pad_spatial(img, radius, border_mode)
    H, W = img.shape[:2]
    # True convolution: out[y, x] = sum_{i,j} k[i, j] * img[y - (i - rh), x - (j - rw)].
    # Reversing the kernel turns this into a sum of weighted, shifted image slices.
    flipped = k[::-1, ::-1]
    out = np.zeros(img.shape, dtype=np.float32)
    for i in range(kh):
        for j in range(kw):
            w = flipped[i, j]
            if w == 0:
                continue
            y0 = radius - rh + i
            x0 = radius - rw + j
            out += w * padded[y0:y0 + H, x0:x0 + W]
    return out.astype(np.float32)


def gaussian_kernel(size: int, sigma: float) -> np.ndarray:
    """Return a nonnegative, symmetric, normalized float32 square Gaussian kernel."""
    size = _check_odd_size(size, "size")
    if not (sigma > 0):
        raise ValueError(f"sigma must be positive, got {sigma}")
    r = size // 2
    ax = np.arange(-r, r + 1, dtype=np.float64)
    g1 = np.exp(-(ax ** 2) / (2.0 * float(sigma) ** 2))
    k = np.outer(g1, g1)
    k /= k.sum()
    return k.astype(np.float32)


def gaussian_support(sigma: float) -> int:
    """Kernel size convention used by ``sharpen_image``: ``max(3, 2 * ceil(3 * sigma) + 1)``."""
    return max(3, 2 * int(math.ceil(3.0 * float(sigma))) + 1)


def sharpen_image(image: np.ndarray, amount: float, sigma: float) -> np.ndarray:
    """Return unclipped float32 unsharp masking; require amount >= 0 and sigma > 0.

    Compute ``image + amount * (image - blur)`` for a Gaussian blur of ``sigma``,
    so ``amount = 0`` returns the input unchanged. Choose the Gaussian support
    yourself and state your choice; see the A2 README.
    """
    if not (amount >= 0):
        raise ValueError(f"amount must be nonnegative, got {amount}")
    if not (sigma > 0):
        raise ValueError(f"sigma must be positive, got {sigma}")
    img = _check_image(image)
    blur = convolve2d_manual(img, gaussian_kernel(gaussian_support(sigma), sigma), "reflect")
    return (img + np.float32(amount) * (img - blur)).astype(np.float32)


def add_gaussian_noise(image: np.ndarray, sigma: float, rng: np.random.Generator) -> np.ndarray:
    """Add supplied-RNG Gaussian noise to float [0,1]; require sigma >= 0 and clip."""
    if not (sigma >= 0):
        raise ValueError(f"sigma must be nonnegative, got {sigma}")
    img = _check_image(image)
    noise = rng.normal(0.0, float(sigma), size=img.shape).astype(np.float32)
    return np.clip(img + noise, 0.0, 1.0).astype(np.float32)


def add_impulse_noise(image: np.ndarray, probability: float, rng: np.random.Generator) -> np.ndarray:
    """Apply equiprobable salt or pepper to independently selected spatial pixels.

    Select each pixel with ``probability`` using only ``rng``. Set every channel of
    a selected color pixel to the same value, either 0 or 1 with equal probability.
    """
    if not (0.0 <= probability <= 1.0):
        raise ValueError(f"probability must be in [0, 1], got {probability}")
    img = _check_image(image)
    H, W = img.shape[:2]
    selected = rng.random((H, W)) < probability
    salt = rng.random((H, W)) < 0.5
    values = np.where(salt, np.float32(1.0), np.float32(0.0))
    out = img.copy()
    if img.ndim == 3:
        out[selected] = values[selected][:, None]
    else:
        out[selected] = values[selected]
    return out.astype(np.float32)


def median_filter_manual(image: np.ndarray, kernel_size: int, border_mode: str = "reflect") -> np.ndarray:
    """Median-filter 2-D or HxWxC data with an odd window; do not delegate filtering."""
    k = _check_odd_size(kernel_size, "kernel_size")
    img = _check_image(image)
    r = k // 2
    padded = _pad_spatial(img, r, border_mode)
    H, W = img.shape[:2]
    # Stack every window offset along a new leading axis, then take the median per pixel.
    stack = np.empty((k * k,) + img.shape, dtype=np.float32)
    idx = 0
    for i in range(k):
        for j in range(k):
            stack[idx] = padded[i:i + H, j:j + W]
            idx += 1
    return np.median(stack, axis=0).astype(np.float32)


def normalize_contrast(image: np.ndarray, low_percentile: float, high_percentile: float) -> np.ndarray:
    """Globally stretch all image values between two valid percentiles to [0,1].

    Use one pair of percentile values jointly across all channels, apply one affine
    mapping, and clip. Raise ``ValueError`` for invalid or equal computed bounds.
    """
    if not (0.0 <= low_percentile < high_percentile <= 100.0):
        raise ValueError(f"require 0 <= low < high <= 100, got {low_percentile}, {high_percentile}")
    img = _check_image(image)
    lo, hi = np.percentile(img.astype(np.float64), [low_percentile, high_percentile])
    if hi == lo:
        raise ValueError(f"computed percentile bounds are equal ({lo}); cannot stretch")
    out = (img.astype(np.float64) - lo) / (hi - lo)
    return np.clip(out, 0.0, 1.0).astype(np.float32)


def histogram_equalize_gray(image: np.ndarray) -> np.ndarray:
    """Equalize a 2-D uint8 image with a 256-bin CDF; return 2-D uint8.

    Subtract the first nonzero CDF value, map to the full uint8 range with
    deterministic nearest-integer rounding, and return constant images unchanged.
    """
    a = np.asarray(image)
    if a.ndim != 2 or a.dtype != np.uint8:
        raise ValueError("histogram_equalize_gray expects a 2-D uint8 image")
    hist = np.bincount(a.ravel(), minlength=256).astype(np.float64)
    cdf = np.cumsum(hist)
    cdf_min = cdf[np.nonzero(cdf)[0][0]]
    total = cdf[-1]
    if total == cdf_min:  # a single gray level: nothing to equalize
        return a.copy()
    lut = np.rint((cdf - cdf_min) / (total - cdf_min) * 255.0)
    lut = np.clip(lut, 0, 255).astype(np.uint8)
    return lut[a]


def apply_gamma(image: np.ndarray, gamma: float) -> np.ndarray:
    """Return ``image ** gamma`` for float [0,1] input as float32 [0,1].

    Positive gamma below 1 brightens; gamma above 1 darkens.
    """
    if not (gamma > 0):
        raise ValueError(f"gamma must be positive, got {gamma}")
    img = np.clip(_check_image(image), 0.0, 1.0)
    return np.power(img, np.float32(gamma)).astype(np.float32)


# ---- pyramids -----------------------------------------------------------------------------

def _restore_channels(result: np.ndarray, like: np.ndarray) -> np.ndarray:
    """OpenCV drops a trailing singleton channel axis; put it back when the input had one."""
    if like.ndim == 3 and result.ndim == 2:
        return result[..., None]
    return result


def reduce_image(image: np.ndarray) -> np.ndarray:
    """One Gaussian-prefiltered ceiling-half reduction (approved ``cv2.pyrDown``)."""
    img = _check_image(image)
    H, W = img.shape[:2]
    out = cv2.pyrDown(img, dstsize=((W + 1) // 2, (H + 1) // 2))
    return _restore_channels(out, img).astype(np.float32)


def expand_image(image: np.ndarray, target_shape: Sequence[int]) -> np.ndarray:
    """Gaussian expansion (approved ``cv2.pyrUp``) to the recorded finer spatial shape."""
    img = _check_image(image)
    H, W = int(target_shape[0]), int(target_shape[1])
    out = cv2.pyrUp(img, dstsize=(W, H))
    return _restore_channels(out, img).astype(np.float32)


def _check_levels(levels: int) -> int:
    if not isinstance(levels, (int, np.integer)) or isinstance(levels, bool) or levels < 1:
        raise ValueError(f"levels must be a positive integer, got {levels}")
    return int(levels)


def gaussian_pyramid(image: np.ndarray, levels: int) -> list[np.ndarray]:
    """Return finest-first Gaussian levels using approved ceiling-half reductions."""
    levels = _check_levels(levels)
    current = _check_image(image)
    pyramid = [current]
    for _ in range(levels - 1):
        current = reduce_image(current)
        pyramid.append(current)
    return pyramid


def laplacian_pyramid(image: np.ndarray, levels: int) -> list[np.ndarray]:
    """Return fine-to-coarse Laplacian levels with the coarsest Gaussian last."""
    gauss = gaussian_pyramid(image, levels)
    pyramid = []
    for finer, coarser in zip(gauss[:-1], gauss[1:]):
        pyramid.append((finer - expand_image(coarser, finer.shape)).astype(np.float32))
    pyramid.append(gauss[-1])
    return pyramid


def reconstruct_laplacian_pyramid(pyramid: Sequence[np.ndarray]) -> np.ndarray:
    """Reconstruct a float32 image, including pyramids with odd spatial dimensions."""
    if len(pyramid) == 0:
        raise ValueError("pyramid must contain at least one level")
    current = np.asarray(pyramid[-1], dtype=np.float32)
    for level in reversed(pyramid[:-1]):
        level = np.asarray(level, dtype=np.float32)
        current = (expand_image(current, level.shape) + level).astype(np.float32)
    return current


def laplacian_pyramid_blend(image_a: np.ndarray, image_b: np.ndarray, mask: np.ndarray, levels: int) -> np.ndarray:
    """Laplacian pyramid blend of equal-size float [0,1] images with a float [0,1] spatial mask.

    Mask 0 selects ``image_a``; mask 1 selects ``image_b``; intermediate values blend.
    """
    a = _check_image(image_a)
    b = _check_image(image_b)
    m = np.asarray(mask, dtype=np.float32)
    if a.shape != b.shape:
        raise ValueError(f"image_a and image_b must share a shape, got {a.shape} and {b.shape}")
    if m.shape[:2] != a.shape[:2]:
        raise ValueError(f"mask spatial shape {m.shape[:2]} must match images {a.shape[:2]}")
    if m.ndim == 3 and m.shape[2] == 1:
        m = m[..., 0]
    if m.ndim != 2:
        raise ValueError("mask must be a 2-D spatial mask")
    lap_a = laplacian_pyramid(a, levels)
    lap_b = laplacian_pyramid(b, levels)
    gauss_m = gaussian_pyramid(m, levels)
    blended = []
    for la, lb, gm in zip(lap_a, lap_b, gauss_m):
        weight = gm[..., None] if la.ndim == 3 else gm
        blended.append(((1.0 - weight) * la + weight * lb).astype(np.float32))
    return reconstruct_laplacian_pyramid(blended)
