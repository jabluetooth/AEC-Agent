"""
Python-side image vectorization using OpenCV.

Detects geometric features (lines, circles, arcs, contours/polylines) from
bitonal TIFF images and returns them as coordinate lists ready to be sent
to AutoCAD via draw_line / draw_polyline / draw_circle sidecar commands.

This replaces AutoCAD Raster Design VTools (vline, vpline, varc, vcircle)
which are interactive and cannot be automated via SendStringToExecute.

Coordinate conversion:
    pixel (x, y) → drawing units: x_dwg = px_x / dpi, y_dwg = (height - px_y) / dpi
    The Y-axis is flipped because images have origin at top-left,
    AutoCAD has origin at bottom-left.
"""

import os
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import structlog

logger = structlog.get_logger(__name__)


@dataclass
class DetectedLine:
    """A line segment detected from raster data."""
    start: Tuple[float, float]
    end: Tuple[float, float]


@dataclass
class DetectedCircle:
    """A circle detected from raster data."""
    center: Tuple[float, float]
    radius: float


@dataclass
class DetectedPolyline:
    """A polyline (contour) detected from raster data."""
    points: List[Tuple[float, float]]
    closed: bool = False


@dataclass
class VectorizationResult:
    """Results from image vectorization."""
    lines: List[DetectedLine] = field(default_factory=list)
    circles: List[DetectedCircle] = field(default_factory=list)
    polylines: List[DetectedPolyline] = field(default_factory=list)
    image_width_px: int = 0
    image_height_px: int = 0
    dpi: int = 300


def vectorize_bitonal_image(
    image_path: str,
    dpi: int = 300,
    min_line_length: int = 50,
    max_line_gap: int = 10,
    hough_threshold: int = 80,
    min_circle_radius: int = 10,
    max_circle_radius: int = 500,
    contour_epsilon_factor: float = 0.005,
    min_contour_points: int = 5,
    min_contour_area: float = 100.0,
) -> VectorizationResult:
    """
    Detect geometric features from a bitonal image using OpenCV.

    Args:
        image_path: Path to the bitonal TIFF image.
        dpi: Image DPI for coordinate conversion (default 300).
        min_line_length: Minimum line length in pixels for HoughLinesP (default 50).
        max_line_gap: Maximum gap between line segments to merge (default 10).
        hough_threshold: Accumulator threshold for HoughLinesP (default 80).
        min_circle_radius: Minimum circle radius in pixels (default 10).
        max_circle_radius: Maximum circle radius in pixels, 0=unlimited (default 500).
        contour_epsilon_factor: Polyline approximation tolerance as fraction
                                of contour perimeter (default 0.005).
        min_contour_points: Minimum points for a contour to be kept (default 5).
        min_contour_area: Minimum contour area in pixels to filter noise (default 100).

    Returns:
        VectorizationResult with detected lines, circles, and polylines
        in drawing-unit coordinates (inches, based on DPI).

    Raises:
        FileNotFoundError: If the image file does not exist.
        RuntimeError: If vectorization fails.
    """
    import cv2
    import numpy as np

    image_path = os.path.abspath(image_path)
    if not os.path.isfile(image_path):
        raise FileNotFoundError(f"Image file not found: {image_path}")

    logger.info("Starting image vectorization", image_path=image_path, dpi=dpi)

    try:
        # Load image as grayscale
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise RuntimeError(f"Failed to load image: {image_path}")

        height, width = img.shape[:2]
        result = VectorizationResult(
            image_width_px=width,
            image_height_px=height,
            dpi=dpi,
        )

        # Ensure binary (bitonal)
        _, binary = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY_INV)

        # Helper: convert pixel coords to drawing units
        # Origin flips from top-left (image) to bottom-left (AutoCAD)
        def px_to_dwg(px_x: float, px_y: float) -> Tuple[float, float]:
            return (px_x / dpi, (height - px_y) / dpi)

        # --- Line Detection (HoughLinesP) ---
        edges = cv2.Canny(binary, 50, 150)
        lines = cv2.HoughLinesP(
            edges,
            rho=1,
            theta=np.pi / 180,
            threshold=hough_threshold,
            minLineLength=min_line_length,
            maxLineGap=max_line_gap,
        )
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                start = px_to_dwg(float(x1), float(y1))
                end = px_to_dwg(float(x2), float(y2))
                result.lines.append(DetectedLine(start=start, end=end))

        logger.info(f"Detected {len(result.lines)} lines")

        # --- Circle Detection (HoughCircles) ---
        # Use median blur to reduce noise for circle detection
        blurred = cv2.medianBlur(img, 5)
        circles = cv2.HoughCircles(
            blurred,
            cv2.HOUGH_GRADIENT,
            dp=1.2,
            minDist=max(20, min_circle_radius * 2),
            param1=100,
            param2=40,
            minRadius=min_circle_radius,
            maxRadius=max_circle_radius if max_circle_radius > 0 else 0,
        )
        if circles is not None:
            circles_rounded = np.uint16(np.around(circles))
            for c in circles_rounded[0, :]:
                center = px_to_dwg(float(c[0]), float(c[1]))
                radius = float(c[2]) / dpi  # Convert radius to drawing units
                result.circles.append(DetectedCircle(center=center, radius=radius))

        logger.info(f"Detected {len(result.circles)} circles")

        # --- Contour Detection (polylines) ---
        contours, _ = cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < min_contour_area:
                continue

            perimeter = cv2.arcLength(contour, True)
            epsilon = contour_epsilon_factor * perimeter
            approx = cv2.approxPolyDP(contour, epsilon, True)

            if len(approx) < min_contour_points:
                continue

            points = []
            for pt in approx:
                dwg_pt = px_to_dwg(float(pt[0][0]), float(pt[0][1]))
                points.append(dwg_pt)

            # Check if contour is closed
            is_closed = cv2.isContourConvex(approx) or (
                len(approx) > 2 and
                np.linalg.norm(approx[0][0] - approx[-1][0]) < max_line_gap
            )

            result.polylines.append(DetectedPolyline(
                points=points,
                closed=is_closed,
            ))

        logger.info(f"Detected {len(result.polylines)} polylines/contours")

        total = len(result.lines) + len(result.circles) + len(result.polylines)
        logger.info(
            "Vectorization complete",
            total_features=total,
            lines=len(result.lines),
            circles=len(result.circles),
            polylines=len(result.polylines),
        )

        return result

    except (FileNotFoundError, RuntimeError):
        raise
    except Exception as e:
        raise RuntimeError(f"Image vectorization failed: {e}") from e
