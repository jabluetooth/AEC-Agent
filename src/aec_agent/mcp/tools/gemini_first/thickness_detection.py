"""
Line Thickness Detection Module.

Detects line thickness/width in binary images using distance transform and
morphological analysis. Maps detected widths to AutoCAD lineweights.

This enables proper visual hierarchy preservation when vectorizing drawings,
distinguishing between thick wall lines, medium object lines, and thin
dimension/leader lines.

Algorithm:
1. Skeletonize the line to find centerline
2. Use distance transform to measure width at each point
3. Sample widths along the line
4. Map pixel width to AutoCAD lineweight

AutoCAD Lineweights (in mm):
0.00, 0.05, 0.09, 0.13, 0.15, 0.18, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50,
0.53, 0.60, 0.70, 0.80, 0.90, 1.00, 1.06, 1.20, 1.40, 1.58, 2.00, 2.11
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Tuple
import math

import numpy as np
from scipy import ndimage
import cv2
import structlog

logger = structlog.get_logger(__name__)


# AutoCAD standard lineweights in mm
AUTOCAD_LINEWEIGHTS = [
    0.00, 0.05, 0.09, 0.13, 0.15, 0.18, 0.20, 0.25,
    0.30, 0.35, 0.40, 0.50, 0.53, 0.60, 0.70, 0.80,
    0.90, 1.00, 1.06, 1.20, 1.40, 1.58, 2.00, 2.11
]

# Semantic thickness categories
class ThicknessCategory(str, Enum):
    """Semantic line thickness categories."""
    HAIRLINE = "hairline"  # 0.00-0.05 mm - dimension lines, leaders
    THIN = "thin"  # 0.09-0.18 mm - hidden lines, center lines
    MEDIUM = "medium"  # 0.20-0.35 mm - object lines
    THICK = "thick"  # 0.40-0.70 mm - cutting planes, walls
    EXTRA_THICK = "extra_thick"  # 0.80+ mm - title block borders


@dataclass
class ThicknessResult:
    """Result of thickness detection for a line."""
    width_pixels: float  # Average width in pixels
    width_mm: float  # Converted to mm (requires DPI)
    lineweight: float  # Nearest AutoCAD lineweight in mm
    category: ThicknessCategory
    confidence: float  # 0.0 to 1.0
    samples: List[float] = field(default_factory=list)  # Width samples along line

    def to_dict(self) -> dict:
        return {
            "width_pixels": self.width_pixels,
            "width_mm": self.width_mm,
            "lineweight": self.lineweight,
            "category": self.category.value,
            "confidence": self.confidence,
            "sample_count": len(self.samples),
        }


@dataclass
class ThicknessConfig:
    """Configuration for thickness detection."""
    dpi: int = 300  # Image DPI for pixel-to-mm conversion
    min_samples: int = 5  # Minimum samples for reliable detection
    sample_spacing: int = 5  # Pixels between samples
    use_median: bool = True  # Use median instead of mean (robust to noise)
    max_width_pixels: int = 50  # Maximum expected line width
    min_width_pixels: int = 1  # Minimum line width


def pixels_to_mm(pixels: float, dpi: int = 300) -> float:
    """Convert pixel width to millimeters."""
    inches = pixels / dpi
    return inches * 25.4


def mm_to_lineweight(width_mm: float) -> float:
    """
    Map width in mm to nearest AutoCAD lineweight.

    Args:
        width_mm: Line width in millimeters

    Returns:
        Nearest standard AutoCAD lineweight in mm
    """
    if width_mm <= 0:
        return 0.0

    # Find nearest lineweight
    nearest = min(AUTOCAD_LINEWEIGHTS, key=lambda lw: abs(lw - width_mm))
    return nearest


def categorize_thickness(lineweight: float) -> ThicknessCategory:
    """
    Categorize lineweight into semantic categories.

    Args:
        lineweight: AutoCAD lineweight in mm

    Returns:
        ThicknessCategory enum value
    """
    if lineweight <= 0.05:
        return ThicknessCategory.HAIRLINE
    elif lineweight <= 0.18:
        return ThicknessCategory.THIN
    elif lineweight <= 0.35:
        return ThicknessCategory.MEDIUM
    elif lineweight <= 0.70:
        return ThicknessCategory.THICK
    else:
        return ThicknessCategory.EXTRA_THICK


def compute_distance_transform(binary_image: np.ndarray) -> np.ndarray:
    """
    Compute distance transform of binary image.

    Args:
        binary_image: Binary image (0 = background, 255 = foreground)

    Returns:
        Distance transform array (distance to nearest background pixel)
    """
    # Ensure binary
    _, binary = cv2.threshold(binary_image, 127, 255, cv2.THRESH_BINARY)

    # Distance transform
    dist = cv2.distanceTransform(binary, cv2.DIST_L2, 5)

    return dist


def skeletonize(binary_image: np.ndarray) -> np.ndarray:
    """
    Compute skeleton (medial axis) of binary image.

    Args:
        binary_image: Binary image

    Returns:
        Skeleton image (single-pixel width lines)
    """
    from skimage.morphology import skeletonize as sk_skeletonize

    # Normalize to 0/1
    binary_01 = (binary_image > 127).astype(np.uint8)

    # Skeletonize
    skeleton = sk_skeletonize(binary_01)

    return (skeleton * 255).astype(np.uint8)


def sample_width_along_line(
    dist_transform: np.ndarray,
    skeleton: np.ndarray,
    start: Tuple[int, int],
    end: Tuple[int, int],
    sample_spacing: int = 5,
) -> List[float]:
    """
    Sample line width along a line segment using the skeleton and distance transform.

    Args:
        dist_transform: Distance transform of the line region
        skeleton: Skeletonized image
        start: (x, y) start point
        end: (x, y) end point
        sample_spacing: Pixels between samples

    Returns:
        List of width values (diameter = 2 * distance) at sample points
    """
    x1, y1 = start
    x2, y2 = end

    dx = x2 - x1
    dy = y2 - y1
    length = int(math.sqrt(dx * dx + dy * dy))

    if length < 2:
        return []

    dx_norm = dx / length
    dy_norm = dy / length

    h, w = dist_transform.shape[:2]
    widths = []

    # Sample along the line
    for i in range(0, length, sample_spacing):
        px = int(x1 + dx_norm * i)
        py = int(y1 + dy_norm * i)

        if 0 <= px < w and 0 <= py < h:
            # Get distance at skeleton point (this is the radius)
            dist = dist_transform[py, px]

            if dist > 0:
                # Width = 2 * radius (diameter)
                width = 2 * dist
                widths.append(width)

    return widths


def extract_line_region(
    binary_image: np.ndarray,
    start: Tuple[int, int],
    end: Tuple[int, int],
    margin: int = 10,
) -> Tuple[np.ndarray, Tuple[int, int, int, int]]:
    """
    Extract a region of interest around a line segment.

    Args:
        binary_image: Full binary image
        start: (x, y) start point
        end: (x, y) end point
        margin: Extra pixels around the line

    Returns:
        Tuple of (cropped image, (x_offset, y_offset, width, height))
    """
    h, w = binary_image.shape[:2]

    x1, y1 = start
    x2, y2 = end

    # Bounding box with margin
    min_x = max(0, min(x1, x2) - margin)
    min_y = max(0, min(y1, y2) - margin)
    max_x = min(w, max(x1, x2) + margin)
    max_y = min(h, max(y1, y2) + margin)

    region = binary_image[min_y:max_y, min_x:max_x].copy()

    return region, (min_x, min_y, max_x - min_x, max_y - min_y)


def detect_line_thickness(
    binary_image: np.ndarray,
    line_start: Tuple[int, int],
    line_end: Tuple[int, int],
    config: Optional[ThicknessConfig] = None,
) -> ThicknessResult:
    """
    Detect the thickness of a line segment in a binary image.

    This is the main entry point for thickness detection. It analyzes the
    line width using distance transform and skeleton analysis.

    Args:
        binary_image: Binary image (grayscale, 0=background, 255=foreground)
        line_start: (x, y) start point of the line
        line_end: (x, y) end point of the line
        config: Optional configuration parameters

    Returns:
        ThicknessResult with detected width and AutoCAD lineweight

    Example:
        >>> result = detect_line_thickness(image, (10, 10), (200, 10))
        >>> print(f"Width: {result.width_mm:.2f}mm, Lineweight: {result.lineweight}")
    """
    if config is None:
        config = ThicknessConfig()

    try:
        # Extract region around the line
        region, (x_off, y_off, _, _) = extract_line_region(
            binary_image, line_start, line_end, margin=config.max_width_pixels
        )

        if region.size == 0:
            return _default_result(config)

        # Adjust coordinates to local region
        local_start = (line_start[0] - x_off, line_start[1] - y_off)
        local_end = (line_end[0] - x_off, line_end[1] - y_off)

        # Compute distance transform
        dist_transform = compute_distance_transform(region)

        # Compute skeleton
        try:
            skeleton = skeletonize(region)
        except ImportError:
            # Fallback without skimage
            skeleton = None

        # Sample widths along the line
        if skeleton is not None:
            widths = sample_width_along_line(
                dist_transform, skeleton, local_start, local_end,
                sample_spacing=config.sample_spacing
            )
        else:
            # Simpler approach: sample directly on the line
            widths = _sample_width_direct(
                dist_transform, local_start, local_end,
                sample_spacing=config.sample_spacing
            )

        if len(widths) < config.min_samples:
            return _default_result(config)

        # Compute representative width
        if config.use_median:
            width_pixels = float(np.median(widths))
        else:
            width_pixels = float(np.mean(widths))

        # Clamp to valid range
        width_pixels = max(config.min_width_pixels, min(config.max_width_pixels, width_pixels))

        # Convert to mm
        width_mm = pixels_to_mm(width_pixels, config.dpi)

        # Map to AutoCAD lineweight
        lineweight = mm_to_lineweight(width_mm)

        # Categorize
        category = categorize_thickness(lineweight)

        # Compute confidence based on sample consistency
        if len(widths) > 1:
            std_dev = np.std(widths)
            mean_width = np.mean(widths)
            cv = std_dev / mean_width if mean_width > 0 else 1.0
            confidence = max(0.0, min(1.0, 1.0 - cv))
        else:
            confidence = 0.5

        return ThicknessResult(
            width_pixels=width_pixels,
            width_mm=width_mm,
            lineweight=lineweight,
            category=category,
            confidence=confidence,
            samples=widths,
        )

    except Exception as e:
        logger.warning("thickness_detection_failed", error=str(e))
        return _default_result(config)


def _sample_width_direct(
    dist_transform: np.ndarray,
    start: Tuple[int, int],
    end: Tuple[int, int],
    sample_spacing: int = 5,
) -> List[float]:
    """
    Sample width directly along the line using distance transform.

    Fallback method when skeleton is not available.
    """
    x1, y1 = start
    x2, y2 = end

    dx = x2 - x1
    dy = y2 - y1
    length = int(math.sqrt(dx * dx + dy * dy))

    if length < 2:
        return []

    dx_norm = dx / length
    dy_norm = dy / length

    # Perpendicular direction
    perp_x = -dy_norm
    perp_y = dx_norm

    h, w = dist_transform.shape[:2]
    widths = []

    for i in range(0, length, sample_spacing):
        px = int(x1 + dx_norm * i)
        py = int(y1 + dy_norm * i)

        if 0 <= px < w and 0 <= py < h:
            # Find maximum distance across perpendicular
            max_dist = 0
            for offset in range(-20, 21):
                ox = int(px + perp_x * offset)
                oy = int(py + perp_y * offset)
                if 0 <= ox < w and 0 <= oy < h:
                    dist = dist_transform[oy, ox]
                    if dist > max_dist:
                        max_dist = dist

            if max_dist > 0:
                widths.append(2 * max_dist)

    return widths


def _default_result(config: ThicknessConfig) -> ThicknessResult:
    """Return a default result for failed detection."""
    default_width = 2.0  # pixels
    width_mm = pixels_to_mm(default_width, config.dpi)
    lineweight = mm_to_lineweight(width_mm)

    return ThicknessResult(
        width_pixels=default_width,
        width_mm=width_mm,
        lineweight=lineweight,
        category=ThicknessCategory.THIN,
        confidence=0.3,
    )


def detect_thickness_batch(
    binary_image: np.ndarray,
    lines: List[Tuple[Tuple[int, int], Tuple[int, int]]],
    config: Optional[ThicknessConfig] = None,
) -> List[ThicknessResult]:
    """
    Detect thickness for multiple lines efficiently.

    Pre-computes distance transform for the entire image.

    Args:
        binary_image: Binary image
        lines: List of ((x1, y1), (x2, y2)) line segments
        config: Optional configuration

    Returns:
        List of ThicknessResult for each line
    """
    if config is None:
        config = ThicknessConfig()

    # Pre-compute full distance transform
    dist_transform = compute_distance_transform(binary_image)

    # Try to compute skeleton once
    try:
        skeleton = skeletonize(binary_image)
    except ImportError:
        skeleton = None

    results = []

    for start, end in lines:
        try:
            if skeleton is not None:
                widths = sample_width_along_line(
                    dist_transform, skeleton, start, end,
                    sample_spacing=config.sample_spacing
                )
            else:
                widths = _sample_width_direct(
                    dist_transform, start, end,
                    sample_spacing=config.sample_spacing
                )

            if len(widths) < config.min_samples:
                results.append(_default_result(config))
                continue

            width_pixels = float(np.median(widths) if config.use_median else np.mean(widths))
            width_pixels = max(config.min_width_pixels, min(config.max_width_pixels, width_pixels))
            width_mm = pixels_to_mm(width_pixels, config.dpi)
            lineweight = mm_to_lineweight(width_mm)
            category = categorize_thickness(lineweight)

            # Confidence
            if len(widths) > 1:
                cv = np.std(widths) / np.mean(widths) if np.mean(widths) > 0 else 1.0
                confidence = max(0.0, min(1.0, 1.0 - cv))
            else:
                confidence = 0.5

            results.append(ThicknessResult(
                width_pixels=width_pixels,
                width_mm=width_mm,
                lineweight=lineweight,
                category=category,
                confidence=confidence,
                samples=widths,
            ))

        except Exception as e:
            logger.debug(f"Thickness detection failed for line: {e}")
            results.append(_default_result(config))

    return results


def suggest_layer_from_thickness(thickness: ThicknessResult) -> str:
    """
    Suggest an NCS layer name based on line thickness.

    Thick lines often represent walls, medium for objects, thin for annotations.

    Args:
        thickness: ThicknessResult from detection

    Returns:
        Suggested layer name
    """
    if thickness.category == ThicknessCategory.EXTRA_THICK:
        return "A-WALL"
    elif thickness.category == ThicknessCategory.THICK:
        return "A-WALL"
    elif thickness.category == ThicknessCategory.MEDIUM:
        return "A-FLOR"  # General floor plan
    elif thickness.category == ThicknessCategory.THIN:
        return "A-ANNO-DIMS"  # Annotations
    else:  # HAIRLINE
        return "A-ANNO-NOTE"  # Notes, leaders
