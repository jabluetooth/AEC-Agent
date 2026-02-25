"""
VTracer Extraction Module for Hybrid Pipeline.

This module provides raster-to-vector conversion using VTracer, a Rust-based
vectorization library with O(n) complexity (vs Potrace's O(n²)).

Key Capabilities:
- High-fidelity raster-to-vector conversion
- Bézier curve fitting with precise corner detection
- Hierarchical color clustering for multi-color images
- SVG path generation with configurable precision

VTracer produces clean vector paths that preserve:
- Sharp corners (better than Potrace's smoothing)
- Accurate curves with Bézier fitting
- Proper topology without arbitrary holes

Usage:
    >>> extractor = VTracerExtractor(binary_image)
    >>> paths = extractor.extract()
    >>> entities = extractor.paths_to_entities(paths, calibration)
"""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Tuple, Union

import numpy as np
import structlog

from aec_agent.config.settings import get_settings

logger = structlog.get_logger(__name__)

# Try to import VTracer
try:
    import vtracer
    VTRACER_AVAILABLE = True
except ImportError:
    VTRACER_AVAILABLE = False
    vtracer = None  # type: ignore
    logger.warning(
        "VTracer not available. Install with: pip install vtracer",
        install_cmd="pip install 'aec-agent[phase_b]'",
    )


class VTracerColorMode(str, Enum):
    """VTracer color mode options."""
    BINARY = "binary"   # Black/white only
    COLOR = "color"     # Full color with clustering


class VTracerMode(str, Enum):
    """VTracer output mode options."""
    SPLINE = "spline"   # Bézier curves (recommended)
    POLYGON = "polygon" # Straight line segments only
    NONE = "none"       # Path only, no fitting


class PathCommand(str, Enum):
    """SVG path command types."""
    MOVE = "M"
    LINE = "L"
    CUBIC_BEZIER = "C"
    QUADRATIC_BEZIER = "Q"
    ARC = "A"
    CLOSE = "Z"
    MOVE_REL = "m"
    LINE_REL = "l"
    CUBIC_BEZIER_REL = "c"
    QUADRATIC_BEZIER_REL = "q"
    HORIZONTAL = "H"
    VERTICAL = "V"


@dataclass
class VTracerConfig:
    """Configuration for VTracer extraction.

    Attributes:
        colormode: Color handling mode (binary or color)
        mode: Output mode (spline for curves, polygon for straight lines)
        filter_speckle: Remove noise smaller than N pixels
        corner_threshold: Angle threshold for corner detection (degrees)
        length_threshold: Minimum segment length for curve fitting
        splice_threshold: Angle threshold for splicing splines
        path_precision: Decimal places for SVG coordinates
    """
    colormode: VTracerColorMode = VTracerColorMode.BINARY
    mode: VTracerMode = VTracerMode.SPLINE
    filter_speckle: int = 4
    corner_threshold: int = 60
    length_threshold: float = 4.0
    splice_threshold: int = 45
    path_precision: int = 3

    @classmethod
    def from_settings(cls) -> VTracerConfig:
        """Create config from application settings."""
        settings = get_settings()
        return cls(
            colormode=VTracerColorMode.BINARY,
            mode=VTracerMode(settings.vtracer_mode),
            filter_speckle=settings.vtracer_filter_speckle,
            corner_threshold=settings.vtracer_corner_threshold,
            length_threshold=settings.vtracer_length_threshold,
            splice_threshold=settings.vtracer_splice_threshold,
            path_precision=settings.vtracer_path_precision,
        )


@dataclass
class BezierSegment:
    """A cubic Bézier curve segment."""
    start: Tuple[float, float]
    control1: Tuple[float, float]
    control2: Tuple[float, float]
    end: Tuple[float, float]

    def to_dict(self) -> dict:
        return {
            "type": "cubic_bezier",
            "start": list(self.start),
            "control1": list(self.control1),
            "control2": list(self.control2),
            "end": list(self.end),
        }

    def approximate_length(self, segments: int = 10) -> float:
        """Approximate arc length using linear interpolation."""
        length = 0.0
        prev = self.start
        for i in range(1, segments + 1):
            t = i / segments
            # De Casteljau's algorithm for point on curve
            point = self._evaluate(t)
            dx = point[0] - prev[0]
            dy = point[1] - prev[1]
            length += np.sqrt(dx * dx + dy * dy)
            prev = point
        return length

    def _evaluate(self, t: float) -> Tuple[float, float]:
        """Evaluate Bézier curve at parameter t."""
        t2 = t * t
        t3 = t2 * t
        mt = 1 - t
        mt2 = mt * mt
        mt3 = mt2 * mt

        x = (mt3 * self.start[0] +
             3 * mt2 * t * self.control1[0] +
             3 * mt * t2 * self.control2[0] +
             t3 * self.end[0])
        y = (mt3 * self.start[1] +
             3 * mt2 * t * self.control1[1] +
             3 * mt * t2 * self.control2[1] +
             t3 * self.end[1])
        return (x, y)


@dataclass
class LineSegment:
    """A straight line segment."""
    start: Tuple[float, float]
    end: Tuple[float, float]

    @property
    def length(self) -> float:
        dx = self.end[0] - self.start[0]
        dy = self.end[1] - self.start[1]
        return np.sqrt(dx * dx + dy * dy)

    @property
    def angle(self) -> float:
        """Angle in degrees (0-180)."""
        dx = self.end[0] - self.start[0]
        dy = self.end[1] - self.start[1]
        return np.degrees(np.arctan2(dy, dx)) % 180

    def to_dict(self) -> dict:
        return {
            "type": "line",
            "start": list(self.start),
            "end": list(self.end),
            "length": self.length,
            "angle": self.angle,
        }


@dataclass
class ExtractedPath:
    """A vector path extracted by VTracer.

    Can contain a mix of line segments and Bézier curves.
    """
    segments: List[Union[LineSegment, BezierSegment]] = field(default_factory=list)
    is_closed: bool = False
    fill_color: Optional[str] = None  # Hex color or None
    stroke_color: Optional[str] = None
    stroke_width: float = 1.0

    @property
    def num_segments(self) -> int:
        return len(self.segments)

    @property
    def total_length(self) -> float:
        """Approximate total path length."""
        total = 0.0
        for seg in self.segments:
            if isinstance(seg, LineSegment):
                total += seg.length
            elif isinstance(seg, BezierSegment):
                total += seg.approximate_length()
        return total

    @property
    def bounding_box(self) -> Tuple[float, float, float, float]:
        """Get bounding box (min_x, min_y, max_x, max_y)."""
        if not self.segments:
            return (0, 0, 0, 0)

        points = []
        for seg in self.segments:
            if isinstance(seg, LineSegment):
                points.extend([seg.start, seg.end])
            elif isinstance(seg, BezierSegment):
                points.extend([seg.start, seg.control1, seg.control2, seg.end])

        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        return (min(xs), min(ys), max(xs), max(ys))

    def to_dict(self) -> dict:
        return {
            "segments": [s.to_dict() for s in self.segments],
            "is_closed": self.is_closed,
            "fill_color": self.fill_color,
            "stroke_color": self.stroke_color,
            "stroke_width": self.stroke_width,
            "num_segments": self.num_segments,
            "total_length": self.total_length,
            "bounding_box": self.bounding_box,
        }


@dataclass
class VTracerExtractionResult:
    """Result of VTracer extraction."""
    paths: List[ExtractedPath] = field(default_factory=list)
    svg_content: str = ""
    image_width: int = 0
    image_height: int = 0
    config: Optional[VTracerConfig] = None

    @property
    def num_paths(self) -> int:
        return len(self.paths)

    @property
    def total_segments(self) -> int:
        return sum(p.num_segments for p in self.paths)

    def to_dict(self) -> dict:
        return {
            "num_paths": self.num_paths,
            "total_segments": self.total_segments,
            "image_size": (self.image_width, self.image_height),
            "paths": [p.to_dict() for p in self.paths],
        }


class VTracerExtractor:
    """Extract vector paths from raster images using VTracer.

    VTracer uses a linear-time algorithm (O(n)) compared to Potrace's O(n²),
    making it much faster for high-resolution images while maintaining
    high fidelity to the original raster.

    Example:
        >>> extractor = VTracerExtractor(binary_image)
        >>> if extractor.is_available:
        ...     result = extractor.extract()
        ...     print(f"Extracted {result.num_paths} paths")
    """

    def __init__(
        self,
        image: Optional[np.ndarray] = None,
        config: Optional[VTracerConfig] = None,
    ):
        """Initialize the extractor.

        Args:
            image: Binary or grayscale image as numpy array
            config: VTracer configuration options
        """
        self._image = image
        self._config = config or VTracerConfig.from_settings()
        self._last_result: Optional[VTracerExtractionResult] = None

    @property
    def is_available(self) -> bool:
        """Check if VTracer is available."""
        return VTRACER_AVAILABLE

    @property
    def config(self) -> VTracerConfig:
        """Get current configuration."""
        return self._config

    def set_image(self, image: np.ndarray) -> None:
        """Set the image to process."""
        self._image = image
        self._last_result = None

    def extract(
        self,
        image: Optional[np.ndarray] = None,
        config: Optional[VTracerConfig] = None,
    ) -> VTracerExtractionResult:
        """Extract vector paths from the image.

        Args:
            image: Optional image to process (uses stored image if None)
            config: Optional config override

        Returns:
            VTracerExtractionResult with extracted paths
        """
        if not VTRACER_AVAILABLE:
            logger.warning("VTracer not available, returning empty result")
            return VTracerExtractionResult()

        img = image if image is not None else self._image
        if img is None:
            logger.error("No image provided for VTracer extraction")
            return VTracerExtractionResult()

        cfg = config or self._config

        # Ensure image is proper format
        if len(img.shape) == 2:
            # Grayscale - convert to RGB for VTracer
            img_rgb = np.stack([img, img, img], axis=-1)
        elif len(img.shape) == 3 and img.shape[2] == 3:
            img_rgb = img
        elif len(img.shape) == 3 and img.shape[2] == 4:
            # RGBA - drop alpha
            img_rgb = img[:, :, :3]
        else:
            logger.error("Unsupported image format", shape=img.shape)
            return VTracerExtractionResult()

        height, width = img_rgb.shape[:2]

        # Convert numpy array to bytes
        try:
            from PIL import Image
            pil_image = Image.fromarray(img_rgb.astype(np.uint8))
        except ImportError:
            logger.error("PIL required for VTracer extraction")
            return VTracerExtractionResult()

        # Run VTracer
        try:
            svg_content = vtracer.convert_pixels_to_svg(
                np.array(pil_image).flatten().tolist(),
                size=(width, height),
                colormode=cfg.colormode.value,
                mode=cfg.mode.value,
                filter_speckle=cfg.filter_speckle,
                corner_threshold=cfg.corner_threshold,
                length_threshold=cfg.length_threshold,
                splice_threshold=cfg.splice_threshold,
                path_precision=cfg.path_precision,
            )

            # Parse SVG to extract paths
            paths = self._parse_svg_paths(svg_content)

            result = VTracerExtractionResult(
                paths=paths,
                svg_content=svg_content,
                image_width=width,
                image_height=height,
                config=cfg,
            )

            self._last_result = result

            logger.info(
                "VTracer extraction complete",
                num_paths=result.num_paths,
                total_segments=result.total_segments,
                image_size=(width, height),
            )

            return result

        except Exception as e:
            logger.error("VTracer extraction failed", error=str(e), exc_info=True)
            return VTracerExtractionResult()

    def extract_to_svg(
        self,
        image: Optional[np.ndarray] = None,
        config: Optional[VTracerConfig] = None,
    ) -> str:
        """Extract and return raw SVG content.

        Args:
            image: Optional image to process
            config: Optional config override

        Returns:
            SVG content as string
        """
        result = self.extract(image, config)
        return result.svg_content

    def _parse_svg_paths(self, svg_content: str) -> List[ExtractedPath]:
        """Parse SVG content to extract path data.

        Args:
            svg_content: Raw SVG string from VTracer

        Returns:
            List of ExtractedPath objects
        """
        paths = []

        try:
            # Parse SVG XML
            root = ET.fromstring(svg_content)

            # Handle SVG namespace
            ns = {"svg": "http://www.w3.org/2000/svg"}

            # Find all path elements
            for path_elem in root.findall(".//path", ns) + root.findall(".//svg:path", ns):
                d = path_elem.get("d", "")
                if not d:
                    continue

                # Parse fill/stroke
                fill = path_elem.get("fill")
                stroke = path_elem.get("stroke")
                stroke_width = float(path_elem.get("stroke-width", "1"))

                # Parse path data
                extracted_path = self._parse_path_data(d)
                extracted_path.fill_color = fill
                extracted_path.stroke_color = stroke
                extracted_path.stroke_width = stroke_width

                if extracted_path.segments:
                    paths.append(extracted_path)

            # Also check for paths without namespace
            for path_elem in root.iter():
                if path_elem.tag.endswith("path") and "path" not in str(path_elem.tag).lower():
                    continue
                if path_elem.tag == "path":
                    d = path_elem.get("d", "")
                    if not d:
                        continue

                    fill = path_elem.get("fill")
                    stroke = path_elem.get("stroke")
                    stroke_width = float(path_elem.get("stroke-width", "1"))

                    extracted_path = self._parse_path_data(d)
                    extracted_path.fill_color = fill
                    extracted_path.stroke_color = stroke
                    extracted_path.stroke_width = stroke_width

                    # Avoid duplicates
                    if extracted_path.segments and extracted_path not in paths:
                        paths.append(extracted_path)

        except ET.ParseError as e:
            logger.warning("Failed to parse SVG XML", error=str(e))
            # Fallback: regex-based extraction
            paths = self._parse_paths_regex(svg_content)

        return paths

    def _parse_path_data(self, d: str) -> ExtractedPath:
        """Parse SVG path 'd' attribute into segments.

        Handles: M, L, C, Q, Z (and lowercase relative versions)
        """
        path = ExtractedPath()

        # Tokenize path data
        # Split on command letters while keeping them
        tokens = re.findall(r'[MLCQAZHVmlcqazhv]|[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?', d)

        current_pos = (0.0, 0.0)
        start_pos = (0.0, 0.0)
        i = 0

        while i < len(tokens):
            cmd = tokens[i]

            if cmd in ('M', 'm'):
                # Move to
                x = float(tokens[i + 1])
                y = float(tokens[i + 2])
                if cmd == 'm':  # Relative
                    x += current_pos[0]
                    y += current_pos[1]
                current_pos = (x, y)
                start_pos = current_pos
                i += 3

            elif cmd in ('L', 'l'):
                # Line to
                x = float(tokens[i + 1])
                y = float(tokens[i + 2])
                if cmd == 'l':  # Relative
                    x += current_pos[0]
                    y += current_pos[1]
                path.segments.append(LineSegment(
                    start=current_pos,
                    end=(x, y),
                ))
                current_pos = (x, y)
                i += 3

            elif cmd in ('H', 'h'):
                # Horizontal line
                x = float(tokens[i + 1])
                if cmd == 'h':
                    x += current_pos[0]
                path.segments.append(LineSegment(
                    start=current_pos,
                    end=(x, current_pos[1]),
                ))
                current_pos = (x, current_pos[1])
                i += 2

            elif cmd in ('V', 'v'):
                # Vertical line
                y = float(tokens[i + 1])
                if cmd == 'v':
                    y += current_pos[1]
                path.segments.append(LineSegment(
                    start=current_pos,
                    end=(current_pos[0], y),
                ))
                current_pos = (current_pos[0], y)
                i += 2

            elif cmd in ('C', 'c'):
                # Cubic Bézier
                c1x = float(tokens[i + 1])
                c1y = float(tokens[i + 2])
                c2x = float(tokens[i + 3])
                c2y = float(tokens[i + 4])
                x = float(tokens[i + 5])
                y = float(tokens[i + 6])
                if cmd == 'c':  # Relative
                    c1x += current_pos[0]
                    c1y += current_pos[1]
                    c2x += current_pos[0]
                    c2y += current_pos[1]
                    x += current_pos[0]
                    y += current_pos[1]
                path.segments.append(BezierSegment(
                    start=current_pos,
                    control1=(c1x, c1y),
                    control2=(c2x, c2y),
                    end=(x, y),
                ))
                current_pos = (x, y)
                i += 7

            elif cmd in ('Q', 'q'):
                # Quadratic Bézier - convert to cubic
                cx = float(tokens[i + 1])
                cy = float(tokens[i + 2])
                x = float(tokens[i + 3])
                y = float(tokens[i + 4])
                if cmd == 'q':  # Relative
                    cx += current_pos[0]
                    cy += current_pos[1]
                    x += current_pos[0]
                    y += current_pos[1]
                # Convert quadratic to cubic
                c1x = current_pos[0] + 2/3 * (cx - current_pos[0])
                c1y = current_pos[1] + 2/3 * (cy - current_pos[1])
                c2x = x + 2/3 * (cx - x)
                c2y = y + 2/3 * (cy - y)
                path.segments.append(BezierSegment(
                    start=current_pos,
                    control1=(c1x, c1y),
                    control2=(c2x, c2y),
                    end=(x, y),
                ))
                current_pos = (x, y)
                i += 5

            elif cmd in ('Z', 'z'):
                # Close path
                if current_pos != start_pos:
                    path.segments.append(LineSegment(
                        start=current_pos,
                        end=start_pos,
                    ))
                path.is_closed = True
                current_pos = start_pos
                i += 1

            else:
                # Unknown command or number, skip
                i += 1

        return path

    def _parse_paths_regex(self, svg_content: str) -> List[ExtractedPath]:
        """Fallback regex-based path extraction."""
        paths = []

        # Find all path d attributes
        path_pattern = r'd="([^"]+)"'
        for match in re.finditer(path_pattern, svg_content):
            d = match.group(1)
            extracted_path = self._parse_path_data(d)
            if extracted_path.segments:
                paths.append(extracted_path)

        return paths


def extract_with_vtracer(
    image: np.ndarray,
    config: Optional[VTracerConfig] = None,
) -> VTracerExtractionResult:
    """Convenience function to extract paths from an image.

    Args:
        image: Input image (binary or grayscale)
        config: Optional VTracer configuration

    Returns:
        VTracerExtractionResult with extracted paths
    """
    extractor = VTracerExtractor(image, config)
    return extractor.extract()


def is_vtracer_available() -> bool:
    """Check if VTracer library is available."""
    return VTRACER_AVAILABLE
