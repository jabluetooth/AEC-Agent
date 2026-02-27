"""
Image Preprocessing Module.

Implements advanced preprocessing operations for raster-to-vector conversion:
- Deskewing (rotation correction)
- Noise reduction
- Contrast enhancement
- Border removal

Based on research from PDF_TO_VECTOR_NEW.md.
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


class SkewDetectionMethod(str, Enum):
    """Methods for detecting image skew angle."""
    HOUGH_LINES = "hough_lines"
    PROJECTION_PROFILE = "projection_profile"
    MOMENTS = "moments"
    FFT = "fft"


@dataclass
class DeskewConfig:
    """Configuration for deskewing operation."""
    # Detection method
    method: SkewDetectionMethod = SkewDetectionMethod.HOUGH_LINES

    # Hough parameters
    hough_threshold: int = 200
    hough_min_line_length: int = 100
    hough_max_line_gap: int = 10

    # Angle constraints
    max_skew_angle: float = 15.0  # Maximum angle to correct (degrees)
    min_skew_angle: float = 0.5   # Minimum angle to bother correcting

    # Projection profile parameters
    projection_angle_range: float = 15.0  # Range to search (±degrees)
    projection_angle_step: float = 0.5    # Step size (degrees)

    # Output options
    border_color: Tuple[int, int, int] = (255, 255, 255)  # Fill color for borders
    crop_to_content: bool = False  # Crop to remove added borders


@dataclass
class DeskewResult:
    """Result of deskewing operation."""
    image: np.ndarray
    skew_angle: float  # Detected angle in degrees
    was_corrected: bool
    method_used: str
    confidence: float  # 0-1


@dataclass
class PreprocessingConfig:
    """Configuration for full preprocessing pipeline."""
    # Deskewing
    enable_deskew: bool = True
    deskew_config: DeskewConfig = None

    # Noise reduction
    enable_denoise: bool = True
    denoise_strength: int = 10  # h parameter for fastNlMeansDenoising

    # Contrast enhancement
    enable_contrast: bool = True
    contrast_clip_limit: float = 2.0  # CLAHE clip limit
    contrast_grid_size: int = 8  # CLAHE tile grid size

    # Border removal
    enable_border_removal: bool = True
    border_threshold: int = 250  # Pixels brighter than this are border

    def __post_init__(self):
        if self.deskew_config is None:
            self.deskew_config = DeskewConfig()


@dataclass
class PreprocessingResult:
    """Result of full preprocessing pipeline."""
    image: np.ndarray
    original_size: Tuple[int, int]
    final_size: Tuple[int, int]
    skew_angle: float
    operations_applied: List[str]


def _ensure_grayscale(image: np.ndarray) -> np.ndarray:
    """Convert image to grayscale if needed."""
    if len(image.shape) == 3:
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return image


def detect_skew_hough(
    image: np.ndarray,
    config: DeskewConfig,
) -> Tuple[float, float]:
    """
    Detect skew angle using Hough line transform.

    Finds dominant line directions and calculates median angle.

    Args:
        image: Input image (grayscale or BGR)
        config: Deskew configuration

    Returns:
        Tuple of (angle_degrees, confidence)
    """
    gray = _ensure_grayscale(image)

    # Edge detection
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)

    # Hough line detection
    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180,
        threshold=config.hough_threshold,
        minLineLength=config.hough_min_line_length,
        maxLineGap=config.hough_max_line_gap,
    )

    if lines is None or len(lines) == 0:
        return 0.0, 0.0

    # Calculate angles for all lines
    angles = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        if x2 - x1 == 0:
            continue  # Vertical line

        angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))

        # Normalize to -45 to +45 range (we care about small deviations)
        while angle > 45:
            angle -= 90
        while angle < -45:
            angle += 90

        angles.append(angle)

    if not angles:
        return 0.0, 0.0

    # Use median angle (robust to outliers)
    median_angle = float(np.median(angles))

    # Confidence based on angle consistency
    angle_std = float(np.std(angles))
    confidence = max(0.0, 1.0 - angle_std / 10.0)

    return median_angle, confidence


def detect_skew_projection(
    image: np.ndarray,
    config: DeskewConfig,
) -> Tuple[float, float]:
    """
    Detect skew using projection profile analysis.

    Rotates image at various angles and finds the one with
    maximum variance in horizontal projection (sharpest text lines).

    Args:
        image: Input image
        config: Deskew configuration

    Returns:
        Tuple of (angle_degrees, confidence)
    """
    gray = _ensure_grayscale(image)

    # Binarize for cleaner projection
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    best_angle = 0.0
    best_variance = 0.0

    # Search range
    angle_range = config.projection_angle_range
    angle_step = config.projection_angle_step

    angles = np.arange(-angle_range, angle_range + angle_step, angle_step)

    for angle in angles:
        # Rotate image
        h, w = binary.shape
        center = (w // 2, h // 2)
        matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(binary, matrix, (w, h), borderValue=0)

        # Compute horizontal projection (sum of each row)
        projection = np.sum(rotated, axis=1)

        # Variance of projection (higher = better alignment)
        variance = float(np.var(projection))

        if variance > best_variance:
            best_variance = variance
            best_angle = angle

    # Confidence based on how much better the best angle is
    all_variances = []
    for angle in angles:
        h, w = binary.shape
        center = (w // 2, h // 2)
        matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(binary, matrix, (w, h), borderValue=0)
        projection = np.sum(rotated, axis=1)
        all_variances.append(np.var(projection))

    mean_variance = np.mean(all_variances)
    confidence = min(1.0, (best_variance - mean_variance) / mean_variance) if mean_variance > 0 else 0.0

    return best_angle, confidence


def detect_skew_moments(
    image: np.ndarray,
    config: DeskewConfig,
) -> Tuple[float, float]:
    """
    Detect skew using image moments.

    Calculates orientation from second-order central moments.
    Fast but less accurate for complex documents.

    Args:
        image: Input image
        config: Deskew configuration

    Returns:
        Tuple of (angle_degrees, confidence)
    """
    gray = _ensure_grayscale(image)

    # Binarize
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Calculate moments
    moments = cv2.moments(binary)

    # Calculate orientation angle from moments
    if moments["mu20"] - moments["mu02"] == 0:
        return 0.0, 0.0

    angle = 0.5 * np.degrees(np.arctan2(
        2 * moments["mu11"],
        moments["mu20"] - moments["mu02"]
    ))

    # Normalize to small angle range
    while angle > 45:
        angle -= 90
    while angle < -45:
        angle += 90

    # Confidence based on moment ratio
    total_moment = abs(moments["mu20"]) + abs(moments["mu02"])
    if total_moment == 0:
        return 0.0, 0.0

    confidence = abs(moments["mu20"] - moments["mu02"]) / total_moment

    return float(angle), float(confidence)


def rotate_image(
    image: np.ndarray,
    angle: float,
    border_color: Tuple[int, ...] = (255, 255, 255),
) -> np.ndarray:
    """
    Rotate image by given angle.

    Args:
        image: Input image
        angle: Rotation angle in degrees (positive = counter-clockwise)
        border_color: Color to fill borders

    Returns:
        Rotated image
    """
    h, w = image.shape[:2]
    center = (w // 2, h // 2)

    # Get rotation matrix
    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)

    # Calculate new image bounds
    cos = abs(matrix[0, 0])
    sin = abs(matrix[0, 1])
    new_w = int(h * sin + w * cos)
    new_h = int(h * cos + w * sin)

    # Adjust matrix for new size
    matrix[0, 2] += (new_w - w) / 2
    matrix[1, 2] += (new_h - h) / 2

    # Rotate with border fill
    if len(image.shape) == 3:
        border = border_color
    else:
        border = border_color[0] if isinstance(border_color, tuple) else border_color

    rotated = cv2.warpAffine(
        image,
        matrix,
        (new_w, new_h),
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=border,
    )

    return rotated


def deskew(
    image: np.ndarray,
    config: Optional[DeskewConfig] = None,
) -> DeskewResult:
    """
    Detect and correct image skew.

    Args:
        image: Input image
        config: Deskew configuration

    Returns:
        DeskewResult with corrected image and angle

    Example:
        >>> result = deskew(image)
        >>> if result.was_corrected:
        ...     print(f"Corrected {result.skew_angle:.2f}° skew")
        >>> cv2.imwrite("deskewed.png", result.image)
    """
    if not OPENCV_AVAILABLE:
        raise ImportError("OpenCV is required for deskewing")

    config = config or DeskewConfig()

    # Detect skew angle
    if config.method == SkewDetectionMethod.HOUGH_LINES:
        angle, confidence = detect_skew_hough(image, config)
    elif config.method == SkewDetectionMethod.PROJECTION_PROFILE:
        angle, confidence = detect_skew_projection(image, config)
    elif config.method == SkewDetectionMethod.MOMENTS:
        angle, confidence = detect_skew_moments(image, config)
    else:
        # Default to Hough
        angle, confidence = detect_skew_hough(image, config)

    # Check if correction is needed
    abs_angle = abs(angle)
    should_correct = (
        abs_angle >= config.min_skew_angle and
        abs_angle <= config.max_skew_angle and
        confidence > 0.3
    )

    method_str = config.method.value if hasattr(config.method, 'value') else str(config.method)

    if should_correct:
        corrected = rotate_image(image, angle, config.border_color)
        logger.info(
            "deskew_applied",
            angle=angle,
            confidence=confidence,
            method=method_str,
        )
    else:
        corrected = image
        logger.debug(
            "deskew_skipped",
            angle=angle,
            confidence=confidence,
            reason="angle_out_of_range" if abs_angle > config.max_skew_angle else "too_small",
        )

    return DeskewResult(
        image=corrected,
        skew_angle=angle,
        was_corrected=should_correct,
        method_used=method_str,
        confidence=confidence,
    )


def denoise(
    image: np.ndarray,
    strength: int = 10,
) -> np.ndarray:
    """
    Apply non-local means denoising.

    Args:
        image: Input image
        strength: Denoising strength (h parameter)

    Returns:
        Denoised image
    """
    if len(image.shape) == 3:
        return cv2.fastNlMeansDenoisingColored(image, None, strength, strength, 7, 21)
    else:
        return cv2.fastNlMeansDenoising(image, None, strength, 7, 21)


def enhance_contrast(
    image: np.ndarray,
    clip_limit: float = 2.0,
    grid_size: int = 8,
) -> np.ndarray:
    """
    Apply CLAHE (Contrast Limited Adaptive Histogram Equalization).

    Args:
        image: Input image
        clip_limit: Threshold for contrast limiting
        grid_size: Size of grid for histogram equalization

    Returns:
        Contrast-enhanced image
    """
    if len(image.shape) == 3:
        # Convert to LAB and apply CLAHE to L channel
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l_channel = lab[:, :, 0]

        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(grid_size, grid_size))
        lab[:, :, 0] = clahe.apply(l_channel)

        return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    else:
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(grid_size, grid_size))
        return clahe.apply(image)


def remove_borders(
    image: np.ndarray,
    threshold: int = 250,
) -> np.ndarray:
    """
    Remove white/light borders from image.

    Args:
        image: Input image
        threshold: Pixels brighter than this are considered border

    Returns:
        Cropped image
    """
    gray = _ensure_grayscale(image)

    # Find non-border pixels
    mask = gray < threshold

    # Find bounding box of content
    coords = np.column_stack(np.where(mask))
    if len(coords) == 0:
        return image

    y_min, x_min = coords.min(axis=0)
    y_max, x_max = coords.max(axis=0)

    # Add small margin
    margin = 5
    h, w = gray.shape
    y_min = max(0, y_min - margin)
    x_min = max(0, x_min - margin)
    y_max = min(h, y_max + margin)
    x_max = min(w, x_max + margin)

    return image[y_min:y_max, x_min:x_max]


def preprocess_image(
    image: np.ndarray,
    config: Optional[PreprocessingConfig] = None,
) -> PreprocessingResult:
    """
    Apply full preprocessing pipeline.

    Args:
        image: Input image
        config: Preprocessing configuration

    Returns:
        PreprocessingResult with processed image and metadata

    Example:
        >>> result = preprocess_image(image)
        >>> print(f"Applied: {result.operations_applied}")
        >>> cv2.imwrite("preprocessed.png", result.image)
    """
    if not OPENCV_AVAILABLE:
        raise ImportError("OpenCV is required for preprocessing")

    config = config or PreprocessingConfig()
    current = image.copy()
    operations = []
    skew_angle = 0.0
    original_size = image.shape[:2]

    # 1. Deskewing (first, before other operations)
    if config.enable_deskew:
        deskew_result = deskew(current, config.deskew_config)
        current = deskew_result.image
        skew_angle = deskew_result.skew_angle
        if deskew_result.was_corrected:
            operations.append(f"deskew({skew_angle:.2f}°)")

    # 2. Border removal
    if config.enable_border_removal:
        before_size = current.shape[:2]
        current = remove_borders(current, config.border_threshold)
        after_size = current.shape[:2]
        if before_size != after_size:
            operations.append("remove_borders")

    # 3. Denoising
    if config.enable_denoise:
        current = denoise(current, config.denoise_strength)
        operations.append("denoise")

    # 4. Contrast enhancement
    if config.enable_contrast:
        current = enhance_contrast(
            current,
            config.contrast_clip_limit,
            config.contrast_grid_size,
        )
        operations.append("enhance_contrast")

    final_size = current.shape[:2]

    logger.info(
        "preprocessing_complete",
        operations=operations,
        original_size=original_size,
        final_size=final_size,
        skew_angle=skew_angle,
    )

    return PreprocessingResult(
        image=current,
        original_size=original_size,
        final_size=final_size,
        skew_angle=skew_angle,
        operations_applied=operations,
    )


# Convenience functions
def quick_deskew(image: np.ndarray) -> np.ndarray:
    """Quick deskew with default settings."""
    result = deskew(image)
    return result.image


def quick_preprocess(image: np.ndarray) -> np.ndarray:
    """Quick preprocessing with default settings."""
    result = preprocess_image(image)
    return result.image
