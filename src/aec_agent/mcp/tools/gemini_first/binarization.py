"""
Ensemble Binarization Module.

Implements pixel-voting binarization using multiple algorithms to achieve
robust foreground/background separation even under uneven illumination,
watermarks, and document degradation.

Based on research from PDF_TO_VECTOR_NEW.md:
- Aggregates 7 binarization algorithms
- Majority vote per pixel
- Statistically rejects localized noise
"""

from dataclasses import dataclass
from enum import Enum
from typing import List, Optional, Tuple

import numpy as np
import structlog

try:
    import cv2
    OPENCV_AVAILABLE = True
except ImportError:
    OPENCV_AVAILABLE = False
    cv2 = None

logger = structlog.get_logger(__name__)


class BinarizationMethod(str, Enum):
    """Available binarization algorithms."""
    OTSU = "otsu"
    ADAPTIVE_GAUSSIAN = "adaptive_gaussian"
    ADAPTIVE_MEAN = "adaptive_mean"
    NIBLACK = "niblack"
    SAUVOLA = "sauvola"
    WOLF = "wolf"
    BRADLEY = "bradley"


@dataclass
class BinarizationConfig:
    """Configuration for ensemble binarization."""
    # Which methods to use in ensemble
    methods: List[BinarizationMethod] = None

    # Adaptive threshold parameters
    adaptive_block_size: int = 15  # Must be odd
    adaptive_c_gaussian: int = 10
    adaptive_c_mean: int = 12

    # Niblack parameters
    niblack_window_size: int = 25
    niblack_k: float = -0.2  # Typically negative for dark text on light bg

    # Sauvola parameters
    sauvola_window_size: int = 25
    sauvola_k: float = 0.2
    sauvola_r: float = 128.0  # Dynamic range of std deviation

    # Wolf parameters
    wolf_window_size: int = 25
    wolf_k: float = 0.5

    # Bradley parameters (integral image based)
    bradley_window_size: int = 15
    bradley_t: float = 0.15  # Threshold percentage

    # Voting threshold (0.5 = majority)
    voting_threshold: float = 0.5

    # Output inversion (True = black foreground on white background)
    invert_output: bool = True

    def __post_init__(self):
        if self.methods is None:
            # Default: use all methods
            self.methods = list(BinarizationMethod)


@dataclass
class BinarizationResult:
    """Result of ensemble binarization."""
    binary_image: np.ndarray
    method_results: dict  # Method name -> binary image
    vote_counts: np.ndarray  # Per-pixel vote counts
    config: BinarizationConfig

    @property
    def consensus_map(self) -> np.ndarray:
        """Map showing consensus level (0-1) per pixel."""
        num_methods = len(self.method_results)
        return self.vote_counts / num_methods


def _ensure_grayscale(image: np.ndarray) -> np.ndarray:
    """Convert image to grayscale if needed."""
    if len(image.shape) == 3:
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return image


def binarize_otsu(gray: np.ndarray) -> np.ndarray:
    """
    Otsu's global thresholding.

    Automatically finds optimal threshold by minimizing intra-class variance.
    Works well for bimodal histograms but fails with uneven illumination.
    """
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return binary


def binarize_adaptive_gaussian(
    gray: np.ndarray,
    block_size: int = 15,
    c: int = 10,
) -> np.ndarray:
    """
    Adaptive thresholding with Gaussian-weighted neighborhood.

    Threshold is computed as weighted sum of neighborhood pixels
    using a Gaussian window. Good for varying illumination.
    """
    # Ensure block_size is odd
    if block_size % 2 == 0:
        block_size += 1

    return cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        block_size,
        c,
    )


def binarize_adaptive_mean(
    gray: np.ndarray,
    block_size: int = 15,
    c: int = 12,
) -> np.ndarray:
    """
    Adaptive thresholding with mean of neighborhood.

    Simpler than Gaussian but less smooth transitions.
    """
    if block_size % 2 == 0:
        block_size += 1

    return cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_MEAN_C,
        cv2.THRESH_BINARY,
        block_size,
        c,
    )


def binarize_niblack(
    gray: np.ndarray,
    window_size: int = 25,
    k: float = -0.2,
) -> np.ndarray:
    """
    Niblack's local thresholding.

    T(x,y) = mean(x,y) + k * std(x,y)

    Works well for document images but sensitive to noise.
    k is typically negative for dark text on light background.
    """
    if window_size % 2 == 0:
        window_size += 1

    # Compute local mean using box filter
    mean = cv2.blur(gray.astype(np.float64), (window_size, window_size))

    # Compute local standard deviation
    mean_sq = cv2.blur(gray.astype(np.float64) ** 2, (window_size, window_size))
    std = np.sqrt(np.maximum(mean_sq - mean ** 2, 0))

    # Niblack threshold
    threshold = mean + k * std

    # Apply threshold
    binary = np.where(gray > threshold, 255, 0).astype(np.uint8)
    return binary


def binarize_sauvola(
    gray: np.ndarray,
    window_size: int = 25,
    k: float = 0.2,
    r: float = 128.0,
) -> np.ndarray:
    """
    Sauvola's local thresholding.

    T(x,y) = mean(x,y) * (1 + k * (std(x,y)/r - 1))

    Improvement over Niblack, normalizes by dynamic range.
    Better for documents with varying contrast.
    """
    if window_size % 2 == 0:
        window_size += 1

    # Compute local mean
    mean = cv2.blur(gray.astype(np.float64), (window_size, window_size))

    # Compute local standard deviation
    mean_sq = cv2.blur(gray.astype(np.float64) ** 2, (window_size, window_size))
    std = np.sqrt(np.maximum(mean_sq - mean ** 2, 0))

    # Sauvola threshold
    threshold = mean * (1.0 + k * (std / r - 1.0))

    # Apply threshold
    binary = np.where(gray > threshold, 255, 0).astype(np.uint8)
    return binary


def binarize_wolf(
    gray: np.ndarray,
    window_size: int = 25,
    k: float = 0.5,
) -> np.ndarray:
    """
    Wolf-Jolion local thresholding.

    T(x,y) = mean(x,y) + k * (mean(x,y) - min_gray) * (std(x,y) / max_std - 1)

    Combines Niblack with global statistics for better robustness.
    """
    if window_size % 2 == 0:
        window_size += 1

    # Global statistics
    min_gray = float(np.min(gray))

    # Compute local mean
    mean = cv2.blur(gray.astype(np.float64), (window_size, window_size))

    # Compute local standard deviation
    mean_sq = cv2.blur(gray.astype(np.float64) ** 2, (window_size, window_size))
    std = np.sqrt(np.maximum(mean_sq - mean ** 2, 0))

    # Maximum std for normalization
    max_std = np.max(std)
    if max_std == 0:
        max_std = 1.0

    # Wolf threshold
    threshold = mean + k * (mean - min_gray) * (std / max_std - 1.0)

    # Apply threshold
    binary = np.where(gray > threshold, 255, 0).astype(np.uint8)
    return binary


def binarize_bradley(
    gray: np.ndarray,
    window_size: int = 15,
    t: float = 0.15,
) -> np.ndarray:
    """
    Bradley adaptive thresholding using integral images.

    Fast O(1) per-pixel computation using summed area tables.
    Pixel is foreground if value < mean * (1 - t).

    Very fast and good for real-time applications.
    """
    height, width = gray.shape

    # Compute integral image
    integral = cv2.integral(gray)

    # Half window size
    s = window_size // 2

    # Output binary image
    binary = np.zeros_like(gray)

    # For each pixel, compute local mean from integral image
    for y in range(height):
        for x in range(width):
            # Window bounds (clamped to image)
            y1 = max(0, y - s)
            y2 = min(height - 1, y + s)
            x1 = max(0, x - s)
            x2 = min(width - 1, x + s)

            # Area of window
            count = (y2 - y1 + 1) * (x2 - x1 + 1)

            # Sum from integral image (note: integral is 1-indexed)
            sum_val = (
                integral[y2 + 1, x2 + 1]
                - integral[y1, x2 + 1]
                - integral[y2 + 1, x1]
                + integral[y1, x1]
            )

            # Local mean
            mean_val = sum_val / count

            # Bradley threshold: foreground if pixel < mean * (1 - t)
            if gray[y, x] * count <= sum_val * (1.0 - t):
                binary[y, x] = 0  # Foreground (dark)
            else:
                binary[y, x] = 255  # Background (light)

    return binary


def binarize_bradley_fast(
    gray: np.ndarray,
    window_size: int = 15,
    t: float = 0.15,
) -> np.ndarray:
    """
    Vectorized Bradley thresholding (much faster than loop version).
    """
    height, width = gray.shape
    s = window_size // 2

    # Compute integral image
    integral = cv2.integral(gray.astype(np.float64))

    # Create coordinate grids
    y, x = np.ogrid[0:height, 0:width]

    # Window bounds (clamped)
    y1 = np.maximum(0, y - s)
    y2 = np.minimum(height - 1, y + s)
    x1 = np.maximum(0, x - s)
    x2 = np.minimum(width - 1, x + s)

    # Count pixels in each window
    count = (y2 - y1 + 1) * (x2 - x1 + 1)

    # Extract sums using integral image (vectorized lookup)
    # Note: This is a simplification; for full vectorization we need
    # to handle the indexing differently

    # For efficiency, use cv2.blur as approximation
    local_mean = cv2.blur(gray.astype(np.float64), (window_size, window_size))

    # Bradley threshold
    threshold = local_mean * (1.0 - t)

    # Apply threshold
    binary = np.where(gray < threshold, 0, 255).astype(np.uint8)
    return binary


def ensemble_binarize(
    image: np.ndarray,
    config: Optional[BinarizationConfig] = None,
) -> BinarizationResult:
    """
    Perform ensemble binarization using pixel voting.

    Aggregates results from multiple binarization algorithms and
    determines each pixel's final state by majority vote.

    Args:
        image: Input image (grayscale or BGR)
        config: Binarization configuration

    Returns:
        BinarizationResult with binary image and per-method results

    Example:
        >>> result = ensemble_binarize(image)
        >>> cv2.imwrite("binary.png", result.binary_image)
        >>> print(f"Consensus: {result.consensus_map.mean():.2%}")
    """
    if not OPENCV_AVAILABLE:
        raise ImportError("OpenCV is required for binarization")

    config = config or BinarizationConfig()
    gray = _ensure_grayscale(image)

    method_results = {}

    # Apply each method
    for method in config.methods:
        try:
            if method == BinarizationMethod.OTSU:
                result = binarize_otsu(gray)
            elif method == BinarizationMethod.ADAPTIVE_GAUSSIAN:
                result = binarize_adaptive_gaussian(
                    gray,
                    config.adaptive_block_size,
                    config.adaptive_c_gaussian,
                )
            elif method == BinarizationMethod.ADAPTIVE_MEAN:
                result = binarize_adaptive_mean(
                    gray,
                    config.adaptive_block_size,
                    config.adaptive_c_mean,
                )
            elif method == BinarizationMethod.NIBLACK:
                result = binarize_niblack(
                    gray,
                    config.niblack_window_size,
                    config.niblack_k,
                )
            elif method == BinarizationMethod.SAUVOLA:
                result = binarize_sauvola(
                    gray,
                    config.sauvola_window_size,
                    config.sauvola_k,
                    config.sauvola_r,
                )
            elif method == BinarizationMethod.WOLF:
                result = binarize_wolf(
                    gray,
                    config.wolf_window_size,
                    config.wolf_k,
                )
            elif method == BinarizationMethod.BRADLEY:
                result = binarize_bradley_fast(
                    gray,
                    config.bradley_window_size,
                    config.bradley_t,
                )
            else:
                logger.warning(f"Unknown binarization method: {method}")
                continue

            method_results[method.value] = result

        except Exception as e:
            logger.warning(f"Binarization method {method} failed: {e}")
            continue

    if not method_results:
        raise ValueError("All binarization methods failed")

    # Stack all results and count votes
    # Normalize to 0/1 (assuming 255 = foreground)
    stacked = np.stack([
        (r > 127).astype(np.uint8)
        for r in method_results.values()
    ], axis=0)

    # Count votes for foreground (value = 1)
    vote_counts = np.sum(stacked, axis=0)

    # Apply voting threshold
    num_methods = len(method_results)
    threshold_count = int(num_methods * config.voting_threshold)

    # Pixel is foreground if votes exceed threshold
    binary = np.where(vote_counts > threshold_count, 255, 0).astype(np.uint8)

    # Optionally invert (make foreground black, background white)
    if config.invert_output:
        binary = 255 - binary

    logger.info(
        "ensemble_binarization_complete",
        methods_used=len(method_results),
        voting_threshold=config.voting_threshold,
        foreground_ratio=float(np.mean(binary < 128)),
    )

    return BinarizationResult(
        binary_image=binary,
        method_results=method_results,
        vote_counts=vote_counts,
        config=config,
    )


def quick_binarize(
    image: np.ndarray,
    methods: Optional[List[str]] = None,
) -> np.ndarray:
    """
    Quick binarization with sensible defaults.

    Args:
        image: Input image
        methods: List of method names (default: all 7)

    Returns:
        Binary image (black foreground, white background)
    """
    config = BinarizationConfig()

    if methods:
        config.methods = [BinarizationMethod(m) for m in methods]

    result = ensemble_binarize(image, config)
    return result.binary_image


# Fast 3-method ensemble for speed-critical applications
def fast_binarize(image: np.ndarray) -> np.ndarray:
    """
    Fast binarization using only 3 methods.

    Uses Otsu + Adaptive Gaussian + Sauvola for good balance
    of speed and quality.
    """
    config = BinarizationConfig(
        methods=[
            BinarizationMethod.OTSU,
            BinarizationMethod.ADAPTIVE_GAUSSIAN,
            BinarizationMethod.SAUVOLA,
        ]
    )
    result = ensemble_binarize(image, config)
    return result.binary_image
