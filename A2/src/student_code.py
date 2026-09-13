"""Student implementations for A2. Do not add dependencies or bypass required work."""

from __future__ import annotations

from collections.abc import Sequence
import numpy as np


def convolve2d_manual(image: np.ndarray, kernel: np.ndarray, border_mode: str = "reflect") -> np.ndarray:
    """Convolve a 2-D or HxWxC image with an odd 2-D kernel; return float32.

    Apply true mathematical convolution, including spatial reversal of the kernel.
    Support ``reflect``, ``edge``, and ``constant`` borders. Do not delegate the
    core operation to a library convolution, correlation, or neural-network layer.
    """
    raise NotImplementedError


def gaussian_kernel(size: int, sigma: float) -> np.ndarray:
    """Return a nonnegative, symmetric, normalized float32 square Gaussian kernel."""
    raise NotImplementedError


def sharpen_image(image: np.ndarray, amount: float, sigma: float) -> np.ndarray:
    """Return unclipped float32 unsharp masking; require amount >= 0 and sigma > 0.

    Compute ``image + amount * (image - blur)`` for a Gaussian blur of ``sigma``,
    so ``amount = 0`` returns the input unchanged. Choose the Gaussian support
    yourself and state your choice; see the A2 README.
    """
    raise NotImplementedError


def add_gaussian_noise(image: np.ndarray, sigma: float, rng: np.random.Generator) -> np.ndarray:
    """Add supplied-RNG Gaussian noise to float [0,1]; require sigma >= 0 and clip."""
    raise NotImplementedError


def add_impulse_noise(image: np.ndarray, probability: float, rng: np.random.Generator) -> np.ndarray:
    """Apply equiprobable salt or pepper to independently selected spatial pixels.

    Select each pixel with ``probability`` using only ``rng``. Set every channel of
    a selected color pixel to the same value, either 0 or 1 with equal probability.
    """
    raise NotImplementedError


def median_filter_manual(image: np.ndarray, kernel_size: int, border_mode: str = "reflect") -> np.ndarray:
    """Median-filter 2-D or HxWxC data with an odd window; do not delegate filtering."""
    raise NotImplementedError


def normalize_contrast(image: np.ndarray, low_percentile: float, high_percentile: float) -> np.ndarray:
    """Globally stretch all image values between two valid percentiles to [0,1].

    Use one pair of percentile values jointly across all channels, apply one affine
    mapping, and clip. Raise ``ValueError`` for invalid or equal computed bounds.
    """
    raise NotImplementedError


def histogram_equalize_gray(image: np.ndarray) -> np.ndarray:
    """Equalize a 2-D uint8 image with a 256-bin CDF; return 2-D uint8.

    Subtract the first nonzero CDF value, map to the full uint8 range with
    deterministic nearest-integer rounding, and return constant images unchanged.
    """
    raise NotImplementedError


def apply_gamma(image: np.ndarray, gamma: float) -> np.ndarray:
    """Return ``image ** gamma`` for float [0,1] input as float32 [0,1].

    Positive gamma below 1 brightens; gamma above 1 darkens.
    """
    raise NotImplementedError


def gaussian_pyramid(image: np.ndarray, levels: int) -> list[np.ndarray]:
    """Return finest-first Gaussian levels using approved ceiling-half reductions."""
    raise NotImplementedError


def laplacian_pyramid(image: np.ndarray, levels: int) -> list[np.ndarray]:
    """Return fine-to-coarse Laplacian levels with the coarsest Gaussian last."""
    raise NotImplementedError


def reconstruct_laplacian_pyramid(pyramid: Sequence[np.ndarray]) -> np.ndarray:
    """Reconstruct a float32 image, including pyramids with odd spatial dimensions."""
    raise NotImplementedError


def laplacian_pyramid_blend(image_a: np.ndarray, image_b: np.ndarray, mask: np.ndarray, levels: int) -> np.ndarray:
    """Laplacian pyramid blend of equal-size float [0,1] images with a float [0,1] spatial mask.

    Mask 0 selects ``image_a``; mask 1 selects ``image_b``; intermediate values blend.
    """
    raise NotImplementedError
