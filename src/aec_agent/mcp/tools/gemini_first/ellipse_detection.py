"""
Ellipse Detection Module.

Detects ellipses in binary images using:
1. Contour analysis + ellipse fitting
2. RANSAC-based ellipse fitting for partial ellipses
3. Arc-to-ellipse matching

Supports:
- Full ellipses
- Partial ellipses (elliptical arcs)
- Tilted ellipses
- Nested ellipses
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
import math

import numpy as np
import structlog

logger = structlog.get_logger(__name__)

# Check for OpenCV
try:
    import cv2
    HAS_OPENCV = True
except ImportError:
    HAS_OPENCV = False


def is_ellipse_detection_available() -> bool:
    """Check if ellipse detection is available."""
    return HAS_OPENCV


@dataclass
class DetectedEllipse:
    """Detected ellipse parameters."""
    center: Tuple[float, float]  # (cx, cy)
    semi_major: float  # Semi-major axis length
    semi_minor: float  # Semi-minor axis length
    rotation: float  # Rotation angle in degrees (0-180)
    confidence: float = 1.0
    is_partial: bool = False
    start_angle: float = 0.0  # For partial ellipses (degrees)
    end_angle: float = 360.0  # For partial ellipses (degrees)
    source_contour_idx: Optional[int] = None

    @property
    def aspect_ratio(self) -> float:
        """Get aspect ratio (major/minor)."""
        if self.semi_minor == 0:
            return float('inf')
        return self.semi_major / self.semi_minor

    @property
    def is_circle(self) -> bool:
        """Check if ellipse is effectively a circle."""
        return self.aspect_ratio < 1.1  # Within 10%

    @property
    def area(self) -> float:
        """Get ellipse area."""
        return math.pi * self.semi_major * self.semi_minor

    @property
    def perimeter(self) -> float:
        """Approximate ellipse perimeter (Ramanujan's formula)."""
        a, b = self.semi_major, self.semi_minor
        h = ((a - b) ** 2) / ((a + b) ** 2)
        return math.pi * (a + b) * (1 + (3 * h) / (10 + math.sqrt(4 - 3 * h)))

    def to_dict(self) -> dict:
        return {
            "center": list(self.center),
            "semi_major": self.semi_major,
            "semi_minor": self.semi_minor,
            "rotation": self.rotation,
            "confidence": self.confidence,
            "is_partial": self.is_partial,
            "start_angle": self.start_angle,
            "end_angle": self.end_angle,
            "aspect_ratio": self.aspect_ratio,
            "is_circle": self.is_circle,
        }


@dataclass
class EllipseDetectionConfig:
    """Configuration for ellipse detection."""
    # Size constraints
    min_radius: int = 5  # Minimum semi-axis in pixels
    max_radius: int = 1000  # Maximum semi-axis in pixels
    min_aspect_ratio: float = 1.0  # Minimum aspect ratio (1.0 = circle)
    max_aspect_ratio: float = 10.0  # Maximum aspect ratio

    # Quality thresholds
    min_points: int = 5  # Minimum contour points for fitting
    min_confidence: float = 0.7  # Minimum fit confidence
    circularity_threshold: float = 0.85  # Below this, not ellipse-like

    # Detection method
    use_ransac: bool = True  # Use RANSAC for partial ellipses
    ransac_iterations: int = 100
    ransac_threshold: float = 5.0  # Distance threshold in pixels

    # Post-processing
    merge_similar: bool = True
    merge_distance: float = 10.0  # Center distance to consider same
    merge_size_ratio: float = 0.1  # Size difference ratio to merge


@dataclass
class EllipseDetectionResult:
    """Result of ellipse detection."""
    ellipses: List[DetectedEllipse]
    full_count: int = 0
    partial_count: int = 0
    circles_filtered: int = 0  # Ellipses that are actually circles
    duration_ms: float = 0.0

    def to_dict(self) -> dict:
        return {
            "ellipse_count": len(self.ellipses),
            "full_count": self.full_count,
            "partial_count": self.partial_count,
            "circles_filtered": self.circles_filtered,
            "duration_ms": self.duration_ms,
            "ellipses": [e.to_dict() for e in self.ellipses],
        }


def _fit_ellipse_to_contour(
    contour: np.ndarray,
    config: EllipseDetectionConfig,
) -> Optional[DetectedEllipse]:
    """
    Fit ellipse to a contour using OpenCV's fitEllipse.

    Args:
        contour: Contour points (N, 1, 2) or (N, 2)
        config: Detection configuration

    Returns:
        DetectedEllipse or None if fitting fails
    """
    if not HAS_OPENCV:
        return None

    # Need at least 5 points for ellipse fitting
    if len(contour) < 5:
        return None

    try:
        # Reshape contour if needed
        if contour.ndim == 3:
            contour = contour.reshape(-1, 2)

        # Fit ellipse
        ellipse = cv2.fitEllipse(contour.astype(np.float32))
        (cx, cy), (width, height), angle = ellipse

        # Convert to semi-axes (OpenCV returns full width/height)
        semi_major = max(width, height) / 2
        semi_minor = min(width, height) / 2

        # Adjust angle if height > width
        if height > width:
            angle = (angle + 90) % 180

        # Check size constraints
        if semi_major < config.min_radius or semi_major > config.max_radius:
            return None
        if semi_minor < config.min_radius:
            return None

        # Check aspect ratio
        aspect = semi_major / semi_minor if semi_minor > 0 else float('inf')
        if aspect < config.min_aspect_ratio or aspect > config.max_aspect_ratio:
            return None

        # Calculate fit quality (how well contour matches ellipse)
        confidence = _calculate_ellipse_fit_quality(contour, ellipse)
        if confidence < config.min_confidence:
            return None

        return DetectedEllipse(
            center=(cx, cy),
            semi_major=semi_major,
            semi_minor=semi_minor,
            rotation=angle,
            confidence=confidence,
            is_partial=False,
        )

    except cv2.error:
        return None


def _calculate_ellipse_fit_quality(
    contour: np.ndarray,
    ellipse: Tuple[Tuple[float, float], Tuple[float, float], float],
) -> float:
    """
    Calculate how well a contour fits an ellipse.

    Returns value between 0 and 1, where 1 is perfect fit.
    """
    (cx, cy), (width, height), angle = ellipse

    # Convert angle to radians
    angle_rad = math.radians(angle)
    cos_a = math.cos(angle_rad)
    sin_a = math.sin(angle_rad)

    a = width / 2  # Semi-major
    b = height / 2  # Semi-minor

    if a == 0 or b == 0:
        return 0.0

    # Calculate distance of each contour point to ellipse
    total_error = 0.0

    for point in contour:
        if contour.ndim == 3:
            px, py = point[0]
        else:
            px, py = point

        # Translate to ellipse center
        dx = px - cx
        dy = py - cy

        # Rotate to ellipse axes
        x_rot = dx * cos_a + dy * sin_a
        y_rot = -dx * sin_a + dy * cos_a

        # Calculate normalized distance from ellipse
        # For a point on ellipse: (x/a)^2 + (y/b)^2 = 1
        dist_normalized = (x_rot / a) ** 2 + (y_rot / b) ** 2
        error = abs(dist_normalized - 1.0)
        total_error += error

    avg_error = total_error / len(contour)

    # Convert error to confidence (lower error = higher confidence)
    # Error of 0 = confidence 1.0, error of 0.5 = confidence ~0.5
    confidence = 1.0 / (1.0 + avg_error * 2)

    return confidence


def _ransac_ellipse_fit(
    points: np.ndarray,
    config: EllipseDetectionConfig,
) -> Optional[DetectedEllipse]:
    """
    RANSAC-based ellipse fitting for partial/noisy data.

    Args:
        points: Point array (N, 2)
        config: Detection configuration

    Returns:
        DetectedEllipse or None
    """
    if not HAS_OPENCV or len(points) < 5:
        return None

    best_ellipse = None
    best_inliers = 0
    best_confidence = 0.0

    n_points = len(points)

    for _ in range(config.ransac_iterations):
        # Randomly sample 5 points
        indices = np.random.choice(n_points, min(5, n_points), replace=False)
        sample = points[indices]

        try:
            # Fit ellipse to sample
            ellipse = cv2.fitEllipse(sample.astype(np.float32))
            (cx, cy), (width, height), angle = ellipse

            # Count inliers
            inliers = _count_ellipse_inliers(
                points, ellipse, config.ransac_threshold
            )

            if inliers > best_inliers:
                best_inliers = inliers
                semi_major = max(width, height) / 2
                semi_minor = min(width, height) / 2

                if height > width:
                    angle = (angle + 90) % 180

                # Calculate confidence based on inlier ratio
                inlier_ratio = inliers / n_points
                confidence = inlier_ratio * _calculate_ellipse_fit_quality(
                    points, ellipse
                )

                if confidence > best_confidence:
                    best_confidence = confidence
                    best_ellipse = DetectedEllipse(
                        center=(cx, cy),
                        semi_major=semi_major,
                        semi_minor=semi_minor,
                        rotation=angle,
                        confidence=confidence,
                        is_partial=inlier_ratio < 0.9,  # Partial if not all points fit
                    )

        except cv2.error:
            continue

    return best_ellipse


def _count_ellipse_inliers(
    points: np.ndarray,
    ellipse: Tuple[Tuple[float, float], Tuple[float, float], float],
    threshold: float,
) -> int:
    """Count points within threshold distance of ellipse."""
    (cx, cy), (width, height), angle = ellipse

    angle_rad = math.radians(angle)
    cos_a = math.cos(angle_rad)
    sin_a = math.sin(angle_rad)

    a = width / 2
    b = height / 2

    if a == 0 or b == 0:
        return 0

    inliers = 0

    for point in points:
        px, py = point[:2]

        # Translate and rotate
        dx = px - cx
        dy = py - cy
        x_rot = dx * cos_a + dy * sin_a
        y_rot = -dx * sin_a + dy * cos_a

        # Calculate approximate distance to ellipse
        # Using parametric distance approximation
        dist_normalized = (x_rot / a) ** 2 + (y_rot / b) ** 2
        dist_to_ellipse = abs(math.sqrt(dist_normalized) - 1.0) * min(a, b)

        if dist_to_ellipse <= threshold:
            inliers += 1

    return inliers


def _detect_partial_ellipse_angles(
    contour: np.ndarray,
    ellipse: DetectedEllipse,
) -> Tuple[float, float]:
    """
    Detect start and end angles for partial ellipse.

    Returns:
        Tuple of (start_angle, end_angle) in degrees
    """
    if len(contour) < 2:
        return 0.0, 360.0

    # Get first and last points
    if contour.ndim == 3:
        first_point = contour[0][0]
        last_point = contour[-1][0]
    else:
        first_point = contour[0]
        last_point = contour[-1]

    # Calculate angles from center
    cx, cy = ellipse.center
    angle_rad = math.radians(ellipse.rotation)
    cos_a = math.cos(-angle_rad)  # Inverse rotation
    sin_a = math.sin(-angle_rad)

    def point_to_angle(px, py):
        dx = px - cx
        dy = py - cy
        # Rotate to ellipse coordinate system
        x_rot = dx * cos_a + dy * sin_a
        y_rot = -dx * sin_a + dy * cos_a
        # Normalize by semi-axes
        x_norm = x_rot / ellipse.semi_major if ellipse.semi_major else 0
        y_norm = y_rot / ellipse.semi_minor if ellipse.semi_minor else 0
        # Calculate angle
        return math.degrees(math.atan2(y_norm, x_norm)) % 360

    start_angle = point_to_angle(first_point[0], first_point[1])
    end_angle = point_to_angle(last_point[0], last_point[1])

    return start_angle, end_angle


def _merge_similar_ellipses(
    ellipses: List[DetectedEllipse],
    config: EllipseDetectionConfig,
) -> List[DetectedEllipse]:
    """Merge ellipses that are very similar (duplicates)."""
    if not ellipses:
        return []

    merged = []
    used = set()

    for i, e1 in enumerate(ellipses):
        if i in used:
            continue

        # Find similar ellipses
        similar = [e1]
        for j, e2 in enumerate(ellipses[i + 1:], start=i + 1):
            if j in used:
                continue

            # Check center distance
            center_dist = math.sqrt(
                (e1.center[0] - e2.center[0]) ** 2 +
                (e1.center[1] - e2.center[1]) ** 2
            )

            # Check size similarity
            size_diff = abs(e1.semi_major - e2.semi_major) / max(e1.semi_major, 1)

            if (center_dist < config.merge_distance and
                size_diff < config.merge_size_ratio):
                similar.append(e2)
                used.add(j)

        # Keep the one with highest confidence
        best = max(similar, key=lambda e: e.confidence)
        merged.append(best)
        used.add(i)

    return merged


def detect_ellipses(
    image: np.ndarray,
    config: Optional[EllipseDetectionConfig] = None,
) -> EllipseDetectionResult:
    """
    Detect ellipses in a binary image.

    Args:
        image: Binary image (white objects on black background)
        config: Detection configuration

    Returns:
        EllipseDetectionResult with detected ellipses
    """
    import time
    start_time = time.perf_counter()

    if config is None:
        config = EllipseDetectionConfig()

    result = EllipseDetectionResult(ellipses=[])

    if not HAS_OPENCV:
        logger.warning("ellipse_detection_unavailable", reason="OpenCV not installed")
        return result

    # Ensure binary image
    if len(image.shape) > 2:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    _, binary = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)

    # Find contours
    contours, _ = cv2.findContours(
        binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE
    )

    ellipses = []
    circles_filtered = 0

    for idx, contour in enumerate(contours):
        if len(contour) < config.min_points:
            continue

        # Check if contour is closed enough to be an ellipse
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)

        if perimeter == 0:
            continue

        circularity = 4 * math.pi * area / (perimeter ** 2)

        if circularity < config.circularity_threshold:
            # Too irregular to be an ellipse
            continue

        # Try standard ellipse fitting
        ellipse = _fit_ellipse_to_contour(contour, config)

        if ellipse is None and config.use_ransac:
            # Try RANSAC for partial ellipses
            points = contour.reshape(-1, 2)
            ellipse = _ransac_ellipse_fit(points, config)

        if ellipse is not None:
            ellipse.source_contour_idx = idx

            # Check if it's really a circle (filter out if needed)
            if ellipse.is_circle:
                circles_filtered += 1
                continue  # Skip circles, handle separately

            # Detect partial ellipse angles
            if ellipse.is_partial:
                start, end = _detect_partial_ellipse_angles(contour, ellipse)
                ellipse.start_angle = start
                ellipse.end_angle = end
                result.partial_count += 1
            else:
                result.full_count += 1

            ellipses.append(ellipse)

    # Merge similar ellipses
    if config.merge_similar:
        ellipses = _merge_similar_ellipses(ellipses, config)

    result.ellipses = ellipses
    result.circles_filtered = circles_filtered
    result.duration_ms = (time.perf_counter() - start_time) * 1000

    logger.info(
        "ellipse_detection_complete",
        total=len(ellipses),
        full=result.full_count,
        partial=result.partial_count,
        circles_filtered=circles_filtered,
        duration_ms=f"{result.duration_ms:.1f}",
    )

    return result


def ellipse_to_entity(
    ellipse: DetectedEllipse,
    scale_factor: float = 1.0,
    layer: str = "0",
) -> Dict[str, Any]:
    """
    Convert detected ellipse to EntityToCreate-compatible dict.

    Args:
        ellipse: Detected ellipse
        scale_factor: Scale factor to apply
        layer: Layer name

    Returns:
        Dict with entity properties
    """
    cx, cy = ellipse.center
    cx_scaled = cx * scale_factor
    cy_scaled = cy * scale_factor
    major_scaled = ellipse.semi_major * scale_factor
    minor_scaled = ellipse.semi_minor * scale_factor

    if ellipse.is_partial:
        # Return as elliptical arc
        return {
            "entity_type": "ellipse",
            "layer": layer,
            "properties": {
                "center": (cx_scaled, cy_scaled, 0),
                "major_axis": (
                    major_scaled * math.cos(math.radians(ellipse.rotation)),
                    major_scaled * math.sin(math.radians(ellipse.rotation)),
                    0,
                ),
                "ratio": minor_scaled / major_scaled if major_scaled else 1.0,
                "start_param": math.radians(ellipse.start_angle),
                "end_param": math.radians(ellipse.end_angle),
            },
            "source": "opencv_ellipse",
            "confidence": ellipse.confidence,
        }
    else:
        # Return as full ellipse
        return {
            "entity_type": "ellipse",
            "layer": layer,
            "properties": {
                "center": (cx_scaled, cy_scaled, 0),
                "major_axis": (
                    major_scaled * math.cos(math.radians(ellipse.rotation)),
                    major_scaled * math.sin(math.radians(ellipse.rotation)),
                    0,
                ),
                "ratio": minor_scaled / major_scaled if major_scaled else 1.0,
                "start_param": 0,
                "end_param": 2 * math.pi,
            },
            "source": "opencv_ellipse",
            "confidence": ellipse.confidence,
        }


def detect_ellipses_in_regions(
    image: np.ndarray,
    regions: List[Tuple[int, int, int, int]],  # (x, y, w, h) bounding boxes
    config: Optional[EllipseDetectionConfig] = None,
) -> List[DetectedEllipse]:
    """
    Detect ellipses in specific regions of an image.

    Useful for targeted detection in symbol areas or known locations.

    Args:
        image: Full image
        regions: List of (x, y, width, height) bounding boxes
        config: Detection configuration

    Returns:
        List of detected ellipses with coordinates in full image space
    """
    if config is None:
        config = EllipseDetectionConfig()

    all_ellipses = []

    for x, y, w, h in regions:
        # Extract region
        region = image[y:y+h, x:x+w]

        if region.size == 0:
            continue

        # Detect ellipses in region
        result = detect_ellipses(region, config)

        # Adjust coordinates to full image space
        for ellipse in result.ellipses:
            ellipse.center = (
                ellipse.center[0] + x,
                ellipse.center[1] + y,
            )
            all_ellipses.append(ellipse)

    return all_ellipses


def fit_ellipse_to_arcs(
    arcs: List[Dict[str, Any]],
    tolerance: float = 5.0,
) -> Optional[DetectedEllipse]:
    """
    Try to fit an ellipse to a collection of arcs.

    Some drawings represent ellipses as multiple connected arcs.
    This function attempts to recognize that pattern.

    Args:
        arcs: List of arc dictionaries with center, radius, start_angle, end_angle
        tolerance: Distance tolerance for matching

    Returns:
        DetectedEllipse if arcs form an ellipse, None otherwise
    """
    if len(arcs) < 2:
        return None

    # Collect all arc points
    points = []

    for arc in arcs:
        center = arc.get("center", (0, 0))
        radius = arc.get("radius", 0)
        start = arc.get("start_angle", 0)
        end = arc.get("end_angle", 360)

        # Generate points along arc
        n_points = max(10, int(abs(end - start) / 10))
        for i in range(n_points):
            angle = math.radians(start + (end - start) * i / n_points)
            x = center[0] + radius * math.cos(angle)
            y = center[1] + radius * math.sin(angle)
            points.append([x, y])

    if len(points) < 5:
        return None

    points_array = np.array(points, dtype=np.float32)

    config = EllipseDetectionConfig(
        min_confidence=0.8,  # Higher threshold for arc-to-ellipse
        use_ransac=True,
    )

    return _ransac_ellipse_fit(points_array, config)
