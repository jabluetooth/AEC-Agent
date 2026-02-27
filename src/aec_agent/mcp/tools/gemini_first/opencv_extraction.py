"""
OpenCV Extraction Utilities for Hybrid Pipeline.

This module provides advanced computer vision functions for precise geometric
extraction from raster images. It's designed to work alongside Gemini's semantic
understanding to provide pixel-perfect accuracy.

Key Capabilities:
- Edge detection with adaptive thresholding
- Line extraction with sub-pixel accuracy using LSD
- Circle/arc detection with Hough transforms
- Contour extraction and polyline fitting
- Region-of-interest (ROI) processing guided by Gemini

The hybrid approach:
1. Gemini identifies WHAT to extract and WHERE (semantic understanding)
2. OpenCV extracts with geometric precision (pixel accuracy)
3. Results are validated and merged

Usage:
    >>> extractor = OpenCVExtractor(image)
    >>> lines = extractor.extract_lines(roi=(100, 100, 500, 500))
    >>> polylines = extractor.extract_polylines(min_length=50)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, List, Optional, Tuple, Union

import numpy as np
import structlog

logger = structlog.get_logger(__name__)

# Try to import OpenCV
try:
    import cv2
    OPENCV_AVAILABLE = True
except ImportError:
    OPENCV_AVAILABLE = False
    logger.warning("OpenCV not available. Install with: pip install opencv-python")


class LineType(str, Enum):
    """Types of lines detected."""
    CONTINUOUS = "continuous"
    DASHED = "dashed"
    DOTTED = "dotted"
    CENTER = "center"
    UNKNOWN = "unknown"


@dataclass
class ExtractedLine:
    """Line extracted by OpenCV."""
    start: Tuple[float, float]  # (x, y) in pixels
    end: Tuple[float, float]
    thickness: float = 1.0
    line_type: LineType = LineType.CONTINUOUS
    confidence: float = 1.0

    @property
    def length(self) -> float:
        """Calculate line length in pixels."""
        dx = self.end[0] - self.start[0]
        dy = self.end[1] - self.start[1]
        return np.sqrt(dx * dx + dy * dy)

    @property
    def angle(self) -> float:
        """Calculate line angle in degrees (0-180)."""
        dx = self.end[0] - self.start[0]
        dy = self.end[1] - self.start[1]
        return np.degrees(np.arctan2(dy, dx)) % 180

    def to_dict(self) -> dict:
        return {
            "start": list(self.start),
            "end": list(self.end),
            "thickness": self.thickness,
            "line_type": self.line_type.value if hasattr(self.line_type, 'value') else str(self.line_type),
            "confidence": self.confidence,
            "length": self.length,
            "angle": self.angle,
        }


@dataclass
class ExtractedCircle:
    """Circle extracted by OpenCV."""
    center: Tuple[float, float]  # (x, y) in pixels
    radius: float
    confidence: float = 1.0

    def to_dict(self) -> dict:
        return {
            "center": list(self.center),
            "radius": self.radius,
            "confidence": self.confidence,
        }


@dataclass
class ExtractedArc:
    """Arc extracted by OpenCV."""
    center: Tuple[float, float]
    radius: float
    start_angle: float  # degrees
    end_angle: float  # degrees
    confidence: float = 1.0

    def to_dict(self) -> dict:
        return {
            "center": list(self.center),
            "radius": self.radius,
            "start_angle": self.start_angle,
            "end_angle": self.end_angle,
            "confidence": self.confidence,
        }


@dataclass
class ExtractedPolyline:
    """Polyline (connected line segments) extracted by OpenCV."""
    points: List[Tuple[float, float]]
    is_closed: bool = False
    thickness: float = 1.0
    confidence: float = 1.0

    @property
    def num_points(self) -> int:
        return len(self.points)

    @property
    def total_length(self) -> float:
        """Calculate total polyline length."""
        if len(self.points) < 2:
            return 0.0
        length = 0.0
        for i in range(len(self.points) - 1):
            dx = self.points[i + 1][0] - self.points[i][0]
            dy = self.points[i + 1][1] - self.points[i][1]
            length += np.sqrt(dx * dx + dy * dy)
        return length

    def to_dict(self) -> dict:
        return {
            "points": [list(p) for p in self.points],
            "is_closed": self.is_closed,
            "thickness": self.thickness,
            "confidence": self.confidence,
            "num_points": self.num_points,
            "total_length": self.total_length,
        }


@dataclass
class ExtractedContour:
    """Contour (closed boundary) extracted by OpenCV."""
    points: List[Tuple[float, float]]
    area: float = 0.0
    perimeter: float = 0.0
    bounding_box: Tuple[int, int, int, int] = (0, 0, 0, 0)  # x, y, w, h
    confidence: float = 1.0

    def to_dict(self) -> dict:
        return {
            "points": [list(p) for p in self.points],
            "area": self.area,
            "perimeter": self.perimeter,
            "bounding_box": list(self.bounding_box),
            "confidence": self.confidence,
        }


@dataclass
class OpenCVExtractionResult:
    """Result of OpenCV extraction."""
    lines: List[ExtractedLine] = field(default_factory=list)
    circles: List[ExtractedCircle] = field(default_factory=list)
    arcs: List[ExtractedArc] = field(default_factory=list)
    polylines: List[ExtractedPolyline] = field(default_factory=list)
    contours: List[ExtractedContour] = field(default_factory=list)

    # Processing metadata
    image_size: Tuple[int, int] = (0, 0)  # (width, height)
    processing_roi: Optional[Tuple[int, int, int, int]] = None

    @property
    def total_elements(self) -> int:
        return (
            len(self.lines)
            + len(self.circles)
            + len(self.arcs)
            + len(self.polylines)
            + len(self.contours)
        )

    def to_dict(self) -> dict:
        return {
            "lines": [l.to_dict() for l in self.lines],
            "circles": [c.to_dict() for c in self.circles],
            "arcs": [a.to_dict() for a in self.arcs],
            "polylines": [p.to_dict() for p in self.polylines],
            "contours": [c.to_dict() for c in self.contours],
            "statistics": {
                "total_elements": self.total_elements,
                "lines": len(self.lines),
                "circles": len(self.circles),
                "arcs": len(self.arcs),
                "polylines": len(self.polylines),
                "contours": len(self.contours),
            },
            "image_size": list(self.image_size),
            "processing_roi": list(self.processing_roi) if self.processing_roi else None,
        }


class OpenCVExtractor:
    """
    Advanced OpenCV extractor for precise geometric extraction.

    This class provides methods to extract geometric primitives from
    raster images with high accuracy. It's designed to complement
    Gemini's semantic understanding with pixel-perfect precision.

    Args:
        image: Input image (BGR or grayscale numpy array)
        dpi: Image DPI for coordinate conversion (default: 300)

    Example:
        >>> extractor = OpenCVExtractor(cv2.imread("drawing.png"))
        >>> result = extractor.extract_all()
        >>> print(f"Found {len(result.lines)} lines")
    """

    def __init__(
        self,
        image: np.ndarray,
        dpi: int = 300,
    ):
        if not OPENCV_AVAILABLE:
            raise ImportError("OpenCV is required. Install with: pip install opencv-python")

        self.original_image = image
        self.dpi = dpi
        self.height, self.width = image.shape[:2]

        # Prepare grayscale and binary images
        if len(image.shape) == 3:
            self.gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            self.gray = image.copy()

        # Binary image (black lines on white background)
        self.binary = self._create_binary()

        logger.debug(
            "opencv_extractor_initialized",
            width=self.width,
            height=self.height,
            dpi=dpi,
        )

    def _create_binary(self) -> np.ndarray:
        """Create binary image using adaptive thresholding."""
        # Apply adaptive threshold
        binary = cv2.adaptiveThreshold(
            self.gray,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            blockSize=15,
            C=10,
        )
        return binary

    def _get_roi(
        self,
        image: np.ndarray,
        roi: Optional[Tuple[int, int, int, int]],
    ) -> Tuple[np.ndarray, int, int]:
        """
        Extract region of interest from image.

        Args:
            image: Source image
            roi: Optional (x1, y1, x2, y2) bounds

        Returns:
            Tuple of (cropped_image, offset_x, offset_y)
        """
        if roi is None:
            return image, 0, 0

        x1, y1, x2, y2 = roi
        x1 = max(0, min(x1, self.width))
        x2 = max(0, min(x2, self.width))
        y1 = max(0, min(y1, self.height))
        y2 = max(0, min(y2, self.height))

        if x2 <= x1 or y2 <= y1:
            return image, 0, 0

        return image[y1:y2, x1:x2], x1, y1

    def extract_lines_hough(
        self,
        roi: Optional[Tuple[int, int, int, int]] = None,
        min_length: int = 30,
        max_gap: int = 10,
        threshold: int = 50,
    ) -> List[ExtractedLine]:
        """
        Extract lines using Probabilistic Hough Transform.

        This is the standard line detection method, good for
        detecting straight line segments.

        Args:
            roi: Optional region of interest (x1, y1, x2, y2)
            min_length: Minimum line length in pixels
            max_gap: Maximum gap between line segments to merge
            threshold: Accumulator threshold

        Returns:
            List of ExtractedLine
        """
        cropped, offset_x, offset_y = self._get_roi(self.binary, roi)

        if cropped.size == 0:
            return []

        # Apply edge detection for cleaner lines
        edges = cv2.Canny(cropped, 50, 150, apertureSize=3)

        # Detect lines
        lines = cv2.HoughLinesP(
            edges,
            rho=1,
            theta=np.pi / 180,
            threshold=threshold,
            minLineLength=min_length,
            maxLineGap=max_gap,
        )

        extracted: List[ExtractedLine] = []

        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                extracted.append(
                    ExtractedLine(
                        start=(float(x1 + offset_x), float(y1 + offset_y)),
                        end=(float(x2 + offset_x), float(y2 + offset_y)),
                        confidence=0.9,
                    )
                )

        logger.debug("hough_lines_extracted", count=len(extracted), roi=roi)
        return extracted

    def extract_lines_lsd(
        self,
        roi: Optional[Tuple[int, int, int, int]] = None,
        min_length: int = 20,
    ) -> List[ExtractedLine]:
        """
        Extract lines using Line Segment Detector (LSD).

        LSD provides sub-pixel accuracy and is better for
        detecting lines in complex drawings.

        Args:
            roi: Optional region of interest (x1, y1, x2, y2)
            min_length: Minimum line length in pixels

        Returns:
            List of ExtractedLine with thickness information
        """
        cropped, offset_x, offset_y = self._get_roi(self.gray, roi)

        if cropped.size == 0:
            return []

        # Create LSD detector
        lsd = cv2.createLineSegmentDetector(cv2.LSD_REFINE_STD)

        # Detect lines
        lines, widths, _, _ = lsd.detect(cropped)

        extracted: List[ExtractedLine] = []

        if lines is not None:
            for i, line in enumerate(lines):
                x1, y1, x2, y2 = line[0]
                length = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

                if length < min_length:
                    continue

                thickness = widths[i][0] if widths is not None else 1.0

                extracted.append(
                    ExtractedLine(
                        start=(float(x1 + offset_x), float(y1 + offset_y)),
                        end=(float(x2 + offset_x), float(y2 + offset_y)),
                        thickness=float(thickness),
                        confidence=0.95,  # LSD is more accurate
                    )
                )

        logger.debug("lsd_lines_extracted", count=len(extracted), roi=roi)
        return extracted

    def extract_circles(
        self,
        roi: Optional[Tuple[int, int, int, int]] = None,
        min_radius: int = 5,
        max_radius: int = 200,
        min_dist: int = 20,
    ) -> List[ExtractedCircle]:
        """
        Extract circles using Hough Circle Transform.

        Args:
            roi: Optional region of interest
            min_radius: Minimum circle radius
            max_radius: Maximum circle radius
            min_dist: Minimum distance between circle centers

        Returns:
            List of ExtractedCircle
        """
        cropped, offset_x, offset_y = self._get_roi(self.gray, roi)

        if cropped.size == 0:
            return []

        # Apply blur to reduce noise
        blurred = cv2.GaussianBlur(cropped, (5, 5), 0)

        # Detect circles
        # param2 controls accumulator threshold - higher = fewer false positives
        # Increased to 70 to reduce excessive circle detection
        circles = cv2.HoughCircles(
            blurred,
            cv2.HOUGH_GRADIENT,
            dp=1,
            minDist=min_dist,
            param1=50,
            param2=70,  # Higher threshold to reduce false positives (was 50)
            minRadius=min_radius,
            maxRadius=max_radius,
        )

        extracted: List[ExtractedCircle] = []

        if circles is not None:
            circles = np.uint16(np.around(circles))
            for circle in circles[0, :]:
                x, y, r = circle
                extracted.append(
                    ExtractedCircle(
                        center=(float(x + offset_x), float(y + offset_y)),
                        radius=float(r),
                        confidence=0.85,
                    )
                )

        logger.debug("circles_extracted", count=len(extracted), roi=roi)
        return extracted

    def extract_contours(
        self,
        roi: Optional[Tuple[int, int, int, int]] = None,
        min_area: float = 100,
        max_area: Optional[float] = None,
        simplify_epsilon: float = 2.0,
    ) -> List[ExtractedContour]:
        """
        Extract contours (closed boundaries) from the image.

        Args:
            roi: Optional region of interest
            min_area: Minimum contour area in pixels
            max_area: Maximum contour area (None = no limit)
            simplify_epsilon: Douglas-Peucker simplification factor

        Returns:
            List of ExtractedContour
        """
        cropped, offset_x, offset_y = self._get_roi(self.binary, roi)

        if cropped.size == 0:
            return []

        # Find contours
        contours, hierarchy = cv2.findContours(
            cropped,
            cv2.RETR_TREE,
            cv2.CHAIN_APPROX_SIMPLE,
        )

        extracted: List[ExtractedContour] = []

        for contour in contours:
            area = cv2.contourArea(contour)

            # Filter by area
            if area < min_area:
                continue
            if max_area is not None and area > max_area:
                continue

            # Simplify contour
            epsilon = simplify_epsilon
            approx = cv2.approxPolyDP(contour, epsilon, True)

            # Get bounding box
            x, y, w, h = cv2.boundingRect(contour)

            # Convert points
            points = [
                (float(pt[0][0] + offset_x), float(pt[0][1] + offset_y))
                for pt in approx
            ]

            extracted.append(
                ExtractedContour(
                    points=points,
                    area=float(area),
                    perimeter=float(cv2.arcLength(contour, True)),
                    bounding_box=(x + offset_x, y + offset_y, w, h),
                    confidence=0.9,
                )
            )

        logger.debug("contours_extracted", count=len(extracted), roi=roi)
        return extracted

    def extract_polylines(
        self,
        roi: Optional[Tuple[int, int, int, int]] = None,
        min_length: float = 50,
        angle_threshold: float = 15.0,
    ) -> List[ExtractedPolyline]:
        """
        Extract polylines by connecting line segments.

        This method detects lines and then merges collinear
        segments into polylines.

        Args:
            roi: Optional region of interest
            min_length: Minimum total polyline length
            angle_threshold: Max angle difference to merge segments (degrees)

        Returns:
            List of ExtractedPolyline
        """
        # First extract lines using LSD
        lines = self.extract_lines_lsd(roi, min_length=10)

        if not lines:
            return []

        # Group lines by connectivity
        polylines = self._merge_lines_to_polylines(lines, angle_threshold)

        # Filter by length
        filtered = [p for p in polylines if p.total_length >= min_length]

        logger.debug(
            "polylines_extracted",
            count=len(filtered),
            roi=roi,
            from_lines=len(lines),
        )
        return filtered

    def _merge_lines_to_polylines(
        self,
        lines: List[ExtractedLine],
        angle_threshold: float,
    ) -> List[ExtractedPolyline]:
        """Merge connected line segments into polylines."""
        if not lines:
            return []

        # Distance threshold for considering lines connected
        connect_dist = 5.0

        # Build connectivity graph
        used = [False] * len(lines)
        polylines: List[ExtractedPolyline] = []

        for i, line in enumerate(lines):
            if used[i]:
                continue

            # Start new polyline
            points = [line.start, line.end]
            used[i] = True
            current_angle = line.angle

            # Try to extend the polyline
            changed = True
            while changed:
                changed = False
                for j, other in enumerate(lines):
                    if used[j]:
                        continue

                    # Check angle compatibility
                    angle_diff = abs(other.angle - current_angle)
                    if angle_diff > 90:
                        angle_diff = 180 - angle_diff
                    if angle_diff > angle_threshold:
                        continue

                    # Check if connected to either end
                    start_pt = np.array(points[0])
                    end_pt = np.array(points[-1])
                    other_start = np.array(other.start)
                    other_end = np.array(other.end)

                    # Connect to end
                    if np.linalg.norm(end_pt - other_start) < connect_dist:
                        points.append(other.end)
                        used[j] = True
                        changed = True
                    elif np.linalg.norm(end_pt - other_end) < connect_dist:
                        points.append(other.start)
                        used[j] = True
                        changed = True
                    # Connect to start
                    elif np.linalg.norm(start_pt - other_end) < connect_dist:
                        points.insert(0, other.start)
                        used[j] = True
                        changed = True
                    elif np.linalg.norm(start_pt - other_start) < connect_dist:
                        points.insert(0, other.end)
                        used[j] = True
                        changed = True

            polylines.append(
                ExtractedPolyline(
                    points=points,
                    is_closed=np.linalg.norm(
                        np.array(points[0]) - np.array(points[-1])
                    ) < connect_dist,
                    confidence=0.85,
                )
            )

        return polylines

    def detect_line_type(
        self,
        line: ExtractedLine,
        sample_width: int = 3,
    ) -> LineType:
        """
        Analyze a line to determine if it's dashed, dotted, center, etc.

        Detects line patterns by sampling pixels along the line and
        analyzing the ink/gap pattern:
        - Continuous: No gaps
        - Dashed: Regular long dashes with gaps
        - Dotted: Short dots with gaps
        - Center: Long-short-long pattern (dash-dot-dash)

        Args:
            line: The line to analyze
            sample_width: Width of sampling corridor

        Returns:
            LineType enum
        """
        # Sample pixels along the line
        x1, y1 = line.start
        x2, y2 = line.end
        length = line.length

        if length < 20:
            return LineType.CONTINUOUS

        num_samples = int(length / 2)
        if num_samples < 10:
            num_samples = int(length)

        samples = []

        for i in range(num_samples):
            t = i / num_samples
            x = int(x1 + t * (x2 - x1))
            y = int(y1 + t * (y2 - y1))

            if 0 <= x < self.width and 0 <= y < self.height:
                samples.append(self.binary[y, x] > 127)

        if not samples or len(samples) < 5:
            return LineType.UNKNOWN

        # Count transitions (ink to no-ink)
        transitions = sum(
            1 for i in range(len(samples) - 1)
            if samples[i] != samples[i + 1]
        )

        # Analyze pattern
        ink_ratio = sum(samples) / len(samples)

        # Find run lengths (consecutive ink or gap segments)
        runs = []
        current_run = 1
        for i in range(1, len(samples)):
            if samples[i] == samples[i - 1]:
                current_run += 1
            else:
                runs.append((samples[i - 1], current_run))
                current_run = 1
        runs.append((samples[-1], current_run))

        # Analyze run pattern for line type
        if transitions < 2:
            return LineType.CONTINUOUS

        # Get ink runs and gap runs
        ink_runs = [r[1] for r in runs if r[0]]
        gap_runs = [r[1] for r in runs if not r[0]]

        if not ink_runs or not gap_runs:
            return LineType.CONTINUOUS

        avg_ink = sum(ink_runs) / len(ink_runs)
        avg_gap = sum(gap_runs) / len(gap_runs)

        # Dotted: short ink, regular gaps
        if avg_ink < 4 and transitions > 8:
            return LineType.DOTTED

        # Center line: alternating long-short pattern (dash-dot-dash)
        # Look for variation in ink run lengths
        if len(ink_runs) >= 3:
            ink_variance = max(ink_runs) / (min(ink_runs) + 0.1)
            if ink_variance > 2.5 and transitions > 4:
                return LineType.CENTER

        # Dashed: regular dashes with gaps
        if transitions > 3 and avg_ink > avg_gap:
            return LineType.DASHED

        # Hidden line (shorter dashes)
        if transitions > 4 and 0.3 < ink_ratio < 0.7:
            return LineType.DASHED

        return LineType.CONTINUOUS

    def extract_arcs(
        self,
        roi: Optional[Tuple[int, int, int, int]] = None,
        min_arc_length: int = 20,
        min_radius: int = 10,
        max_radius: int = 500,
        arc_angle_threshold: float = 30.0,
    ) -> List[ExtractedArc]:
        """
        Extract arcs (partial circles) using contour analysis and ellipse fitting.

        Detects fillets, half-circles, quarter-circles, and other arc segments
        by analyzing contours that have circular curvature but don't form
        complete circles.

        Args:
            roi: Optional region of interest (x1, y1, x2, y2)
            min_arc_length: Minimum arc length in pixels
            min_radius: Minimum arc radius
            max_radius: Maximum arc radius
            arc_angle_threshold: Minimum arc angle span in degrees to be considered an arc

        Returns:
            List of ExtractedArc
        """
        cropped, offset_x, offset_y = self._get_roi(self.binary, roi)

        if cropped.size == 0:
            return []

        # Find contours
        contours, _ = cv2.findContours(
            cropped,
            cv2.RETR_LIST,
            cv2.CHAIN_APPROX_NONE,  # Get all points for accurate arc detection
        )

        extracted: List[ExtractedArc] = []

        for contour in contours:
            # Need at least 5 points for ellipse fitting
            if len(contour) < 5:
                continue

            # Calculate contour arc length
            arc_length = cv2.arcLength(contour, closed=False)
            if arc_length < min_arc_length:
                continue

            # Calculate area to determine if it's closed (full circle) or open (arc)
            area = cv2.contourArea(contour)

            # Fit ellipse to the contour
            try:
                ellipse = cv2.fitEllipse(contour)
                (center_x, center_y), (axis_a, axis_b), angle = ellipse

                # Calculate average radius
                radius = (axis_a + axis_b) / 4  # Divided by 4 because axes are diameters

                # Skip if radius outside bounds
                if radius < min_radius or radius > max_radius:
                    continue

                # Check circularity - for arcs, the aspect ratio should be reasonable
                aspect_ratio = min(axis_a, axis_b) / max(axis_a, axis_b) if max(axis_a, axis_b) > 0 else 0

                # Skip highly elongated shapes (not circular)
                if aspect_ratio < 0.5:
                    continue

                # Calculate the expected area for a full circle
                full_circle_area = np.pi * radius * radius

                # If area is close to full circle, it's a circle, not an arc
                area_ratio = area / full_circle_area if full_circle_area > 0 else 0

                # Arcs have area ratio between 0.1 and 0.9 (not too small, not full circle)
                # Small area ratio indicates an arc, not a closed circle
                if 0.05 < area_ratio < 0.85:
                    # Calculate arc angles by analyzing contour points
                    points = contour.reshape(-1, 2)

                    # Calculate angles from center for each point
                    angles = np.arctan2(
                        points[:, 1] - center_y,
                        points[:, 0] - center_x
                    )
                    angles_deg = np.degrees(angles)

                    # Normalize to 0-360
                    angles_deg = (angles_deg + 360) % 360

                    # Find the angular span
                    min_angle = np.min(angles_deg)
                    max_angle = np.max(angles_deg)

                    # Handle wraparound at 0/360 degrees
                    angle_span = max_angle - min_angle
                    if angle_span > 180:
                        # Wraps around 0 degrees
                        sorted_angles = np.sort(angles_deg)
                        gaps = np.diff(sorted_angles)
                        largest_gap_idx = np.argmax(gaps)
                        start_angle = sorted_angles[largest_gap_idx + 1] if largest_gap_idx + 1 < len(sorted_angles) else sorted_angles[0]
                        end_angle = sorted_angles[largest_gap_idx]
                    else:
                        start_angle = min_angle
                        end_angle = max_angle

                    actual_span = (end_angle - start_angle) % 360
                    if actual_span > 180:
                        actual_span = 360 - actual_span

                    # Skip very small arcs
                    if actual_span < arc_angle_threshold:
                        continue

                    # Skip nearly complete circles (>300 degrees)
                    if actual_span > 300:
                        continue

                    extracted.append(
                        ExtractedArc(
                            center=(float(center_x + offset_x), float(center_y + offset_y)),
                            radius=float(radius),
                            start_angle=float(start_angle),
                            end_angle=float(end_angle),
                            confidence=float(aspect_ratio * 0.9),  # Higher aspect ratio = more circular = higher confidence
                        )
                    )

            except cv2.error:
                # Ellipse fitting can fail for some contours
                continue

        logger.debug("arcs_extracted", count=len(extracted), roi=roi)
        return extracted

    def extract_all(
        self,
        roi: Optional[Tuple[int, int, int, int]] = None,
        use_lsd: bool = True,
    ) -> OpenCVExtractionResult:
        """
        Extract all geometric primitives from the image.

        Args:
            roi: Optional region of interest
            use_lsd: Use LSD for line detection (more accurate)

        Returns:
            OpenCVExtractionResult with all extracted elements
        """
        result = OpenCVExtractionResult(
            image_size=(self.width, self.height),
            processing_roi=roi,
        )

        # Extract lines
        if use_lsd:
            result.lines = self.extract_lines_lsd(roi)
        else:
            result.lines = self.extract_lines_hough(roi)

        # Detect line types
        for line in result.lines:
            line.line_type = self.detect_line_type(line)

        # Extract circles
        result.circles = self.extract_circles(roi)

        # Extract arcs (partial circles, fillets)
        result.arcs = self.extract_arcs(roi)

        # Extract contours
        result.contours = self.extract_contours(roi)

        # Extract polylines
        result.polylines = self.extract_polylines(roi)

        logger.info(
            "opencv_extraction_complete",
            lines=len(result.lines),
            circles=len(result.circles),
            arcs=len(result.arcs),
            contours=len(result.contours),
            polylines=len(result.polylines),
            total=result.total_elements,
        )

        return result


def extract_from_image(
    image_path: Union[str, Path],
    roi: Optional[Tuple[int, int, int, int]] = None,
    dpi: int = 300,
) -> OpenCVExtractionResult:
    """
    Convenience function to extract from an image file.

    Args:
        image_path: Path to image file
        roi: Optional region of interest
        dpi: Image DPI

    Returns:
        OpenCVExtractionResult
    """
    if not OPENCV_AVAILABLE:
        logger.error("OpenCV not available")
        return OpenCVExtractionResult()

    image = cv2.imread(str(image_path))
    if image is None:
        logger.error("failed_to_load_image", path=str(image_path))
        return OpenCVExtractionResult()

    extractor = OpenCVExtractor(image, dpi=dpi)
    return extractor.extract_all(roi=roi)


def extract_lines_from_region(
    image: np.ndarray,
    roi: Tuple[int, int, int, int],
    min_length: int = 20,
) -> List[ExtractedLine]:
    """
    Extract lines from a specific region.

    This is a convenience function for targeted extraction.

    Args:
        image: Source image
        roi: Region bounds (x1, y1, x2, y2)
        min_length: Minimum line length

    Returns:
        List of ExtractedLine
    """
    if not OPENCV_AVAILABLE:
        return []

    extractor = OpenCVExtractor(image)
    return extractor.extract_lines_lsd(roi=roi, min_length=min_length)


def extract_circles_from_region(
    image: np.ndarray,
    roi: Tuple[int, int, int, int],
    min_radius: int = 5,
    max_radius: int = 100,
) -> List[ExtractedCircle]:
    """
    Extract circles from a specific region.

    Args:
        image: Source image
        roi: Region bounds (x1, y1, x2, y2)
        min_radius: Minimum circle radius
        max_radius: Maximum circle radius

    Returns:
        List of ExtractedCircle
    """
    if not OPENCV_AVAILABLE:
        return []

    extractor = OpenCVExtractor(image)
    return extractor.extract_circles(roi=roi, min_radius=min_radius, max_radius=max_radius)
