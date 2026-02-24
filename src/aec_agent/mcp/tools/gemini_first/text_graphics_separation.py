"""
Fletcher-Kasturi Text/Graphics Separation Module.

This module implements the Fletcher-Kasturi algorithm for separating text
regions from graphical elements in engineering drawings. This enables
targeted extraction strategies for each content type.

Key Capabilities:
- Connected component analysis for region classification
- Hough Transform-based text line grouping
- Skeleton segmentation for overlapping text/graphics
- Binary masks for text and graphics regions

Algorithm Overview:
1. Compute connected components in binarized image
2. Classify components by bounding box area and aspect ratio
3. Group text components into lines using Hough Transform
4. Handle intersections via skeleton segmentation
5. Generate separate masks for text and graphics

Reference:
Fletcher, L.A. and Kasturi, R. (1988). "A Robust Algorithm for Text String
Separation from Mixed Text/Graphics Images." IEEE TPAMI.

Usage:
    >>> separator = FletcherKasturiSeparator(binary_image)
    >>> result = separator.separate()
    >>> text_mask = result.text_mask
    >>> graphics_mask = result.graphics_mask
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, List, Optional, Tuple

import numpy as np
import structlog

from aec_agent.config.settings import get_settings

logger = structlog.get_logger(__name__)

# Try to import OpenCV
try:
    import cv2
    OPENCV_AVAILABLE = True
except ImportError:
    OPENCV_AVAILABLE = False
    cv2 = None  # type: ignore
    logger.warning("OpenCV not available for Fletcher-Kasturi separation")


class ComponentType(str, Enum):
    """Classification of connected components."""
    TEXT = "text"
    GRAPHICS = "graphics"
    MIXED = "mixed"
    NOISE = "noise"


@dataclass
class SeparationConfig:
    """Configuration for Fletcher-Kasturi separation.

    Attributes:
        area_threshold_factor: Factor for area threshold (T_a = factor * median_area)
        max_elongation: Maximum aspect ratio for text classification
        min_text_height: Minimum text height in pixels
        max_text_height: Maximum text height in pixels
        min_component_area: Minimum area to consider (noise filter)
        hough_step: Step size for Hough text line detection
        gap_threshold_factor: Word gap threshold (factor * avg_char_width)
        enable_skeleton_segmentation: Handle overlapping text/graphics
    """
    area_threshold_factor: float = 3.0
    max_elongation: float = 10.0
    min_text_height: int = 8
    max_text_height: int = 100
    min_component_area: int = 10
    hough_step: float = 1.0
    gap_threshold_factor: float = 2.0
    enable_skeleton_segmentation: bool = True

    @classmethod
    def from_settings(cls) -> SeparationConfig:
        """Create config from application settings."""
        settings = get_settings()
        return cls(
            area_threshold_factor=settings.text_graphics_area_factor,
            max_elongation=settings.text_graphics_elongation_max,
            min_text_height=settings.text_graphics_min_text_height,
            gap_threshold_factor=settings.text_graphics_gap_factor,
        )


@dataclass
class ConnectedComponent:
    """A connected component in the image."""
    label: int
    bbox: Tuple[int, int, int, int]  # (x, y, width, height)
    area: int
    centroid: Tuple[float, float]
    pixels: Optional[np.ndarray] = None  # Mask of the component

    @property
    def x(self) -> int:
        return self.bbox[0]

    @property
    def y(self) -> int:
        return self.bbox[1]

    @property
    def width(self) -> int:
        return self.bbox[2]

    @property
    def height(self) -> int:
        return self.bbox[3]

    @property
    def aspect_ratio(self) -> float:
        """Width / Height ratio."""
        if self.height == 0:
            return float("inf")
        return self.width / self.height

    @property
    def elongation(self) -> float:
        """Max(w/h, h/w) - how stretched the component is."""
        if self.width == 0 or self.height == 0:
            return float("inf")
        return max(self.width / self.height, self.height / self.width)


@dataclass
class TextComponent(ConnectedComponent):
    """A text component with additional properties."""
    component_type: ComponentType = ComponentType.TEXT
    confidence: float = 1.0


@dataclass
class GraphicsComponent(ConnectedComponent):
    """A graphics component with additional properties."""
    component_type: ComponentType = ComponentType.GRAPHICS
    confidence: float = 1.0


@dataclass
class TextLine:
    """A group of text components forming a line."""
    components: List[TextComponent] = field(default_factory=list)
    baseline_y: float = 0.0
    angle: float = 0.0  # degrees

    @property
    def bbox(self) -> Tuple[int, int, int, int]:
        """Bounding box of the entire line."""
        if not self.components:
            return (0, 0, 0, 0)
        x_min = min(c.x for c in self.components)
        y_min = min(c.y for c in self.components)
        x_max = max(c.x + c.width for c in self.components)
        y_max = max(c.y + c.height for c in self.components)
        return (x_min, y_min, x_max - x_min, y_max - y_min)

    @property
    def text_height(self) -> float:
        """Average height of text components."""
        if not self.components:
            return 0.0
        return sum(c.height for c in self.components) / len(self.components)


@dataclass
class SeparationResult:
    """Result of text/graphics separation."""
    text_mask: np.ndarray  # Binary mask of text regions
    graphics_mask: np.ndarray  # Binary mask of graphics regions
    text_components: List[TextComponent] = field(default_factory=list)
    graphics_components: List[GraphicsComponent] = field(default_factory=list)
    text_lines: List[TextLine] = field(default_factory=list)
    mixed_regions: List[ConnectedComponent] = field(default_factory=list)
    noise_filtered: int = 0

    @property
    def num_text_components(self) -> int:
        return len(self.text_components)

    @property
    def num_graphics_components(self) -> int:
        return len(self.graphics_components)

    @property
    def num_text_lines(self) -> int:
        return len(self.text_lines)

    def to_dict(self) -> dict:
        return {
            "num_text_components": self.num_text_components,
            "num_graphics_components": self.num_graphics_components,
            "num_text_lines": self.num_text_lines,
            "num_mixed_regions": len(self.mixed_regions),
            "noise_filtered": self.noise_filtered,
        }


class FletcherKasturiSeparator:
    """Separate text from graphics using the Fletcher-Kasturi algorithm.

    This algorithm uses connected component analysis to classify regions
    as text or graphics based on their geometric properties (area, aspect
    ratio, elongation).

    Example:
        >>> separator = FletcherKasturiSeparator(binary_image)
        >>> result = separator.separate()
        >>> # Use text_mask for OCR, graphics_mask for vectorization
    """

    def __init__(
        self,
        image: Optional[np.ndarray] = None,
        config: Optional[SeparationConfig] = None,
    ):
        """Initialize the separator.

        Args:
            image: Binary image (0 = background, 255 = foreground)
            config: Separation configuration
        """
        self._image = image
        self._config = config or SeparationConfig.from_settings()
        self._components: List[ConnectedComponent] = []
        self._median_area: float = 0.0
        self._avg_char_width: float = 0.0

    @property
    def is_available(self) -> bool:
        """Check if OpenCV is available for separation."""
        return OPENCV_AVAILABLE

    def set_image(self, image: np.ndarray) -> None:
        """Set the image to process."""
        self._image = image
        self._components = []

    def separate(
        self,
        image: Optional[np.ndarray] = None,
        config: Optional[SeparationConfig] = None,
    ) -> SeparationResult:
        """Separate text from graphics in the image.

        Args:
            image: Optional image override
            config: Optional config override

        Returns:
            SeparationResult with text and graphics masks
        """
        if not OPENCV_AVAILABLE:
            logger.warning("OpenCV not available for Fletcher-Kasturi separation")
            return self._empty_result()

        img = image if image is not None else self._image
        if img is None:
            logger.error("No image provided for separation")
            return self._empty_result()

        cfg = config or self._config

        # Ensure binary image
        if len(img.shape) == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if img.max() > 1:
            _, img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)

        height, width = img.shape

        # Step 1: Compute connected components
        self._components = self._compute_connected_components(img)

        # Filter noise
        noise_filtered = 0
        filtered_components = []
        for comp in self._components:
            if comp.area < cfg.min_component_area:
                noise_filtered += 1
            else:
                filtered_components.append(comp)
        self._components = filtered_components

        if not self._components:
            logger.debug("No components found after noise filtering")
            return SeparationResult(
                text_mask=np.zeros((height, width), dtype=np.uint8),
                graphics_mask=np.zeros((height, width), dtype=np.uint8),
                noise_filtered=noise_filtered,
            )

        # Compute statistics
        areas = [c.area for c in self._components]
        self._median_area = np.median(areas)

        # Step 2: Classify components
        text_components, graphics_components, mixed = self._classify_components(
            self._components, cfg
        )

        # Step 3: Group text into lines
        text_lines = self._group_text_into_lines(text_components, cfg)

        # Step 4: Handle mixed/intersecting regions
        if cfg.enable_skeleton_segmentation and mixed:
            text_from_mixed, graphics_from_mixed = self._handle_intersections(
                mixed, img, cfg
            )
            text_components.extend(text_from_mixed)
            graphics_components.extend(graphics_from_mixed)

        # Step 5: Generate masks
        text_mask = self._create_mask(text_components, (height, width))
        graphics_mask = self._create_mask(graphics_components, (height, width))

        # Ensure masks don't overlap
        overlap = cv2.bitwise_and(text_mask, graphics_mask)
        if np.any(overlap):
            # Remove overlap from graphics (prefer text)
            graphics_mask = cv2.bitwise_and(
                graphics_mask,
                cv2.bitwise_not(overlap),
            )

        logger.info(
            "fletcher_kasturi_separation_complete",
            text_components=len(text_components),
            graphics_components=len(graphics_components),
            text_lines=len(text_lines),
            noise_filtered=noise_filtered,
        )

        return SeparationResult(
            text_mask=text_mask,
            graphics_mask=graphics_mask,
            text_components=text_components,
            graphics_components=graphics_components,
            text_lines=text_lines,
            mixed_regions=mixed,
            noise_filtered=noise_filtered,
        )

    def _empty_result(self) -> SeparationResult:
        """Return an empty result."""
        return SeparationResult(
            text_mask=np.array([]),
            graphics_mask=np.array([]),
        )

    def _compute_connected_components(
        self,
        binary_image: np.ndarray,
    ) -> List[ConnectedComponent]:
        """Compute connected components in the binary image."""
        # Use cv2.connectedComponentsWithStats
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            binary_image, connectivity=8
        )

        components = []
        for i in range(1, num_labels):  # Skip background (label 0)
            x = stats[i, cv2.CC_STAT_LEFT]
            y = stats[i, cv2.CC_STAT_TOP]
            w = stats[i, cv2.CC_STAT_WIDTH]
            h = stats[i, cv2.CC_STAT_HEIGHT]
            area = stats[i, cv2.CC_STAT_AREA]
            centroid = (centroids[i][0], centroids[i][1])

            # Create component mask
            mask = (labels == i).astype(np.uint8) * 255

            components.append(ConnectedComponent(
                label=i,
                bbox=(x, y, w, h),
                area=area,
                centroid=centroid,
                pixels=mask,
            ))

        return components

    def _classify_components(
        self,
        components: List[ConnectedComponent],
        config: SeparationConfig,
    ) -> Tuple[List[TextComponent], List[GraphicsComponent], List[ConnectedComponent]]:
        """Classify components as text, graphics, or mixed.

        Classification rules (Fletcher-Kasturi):
        1. Text components have area < T_a (threshold based on median)
        2. Text components have elongation < max_elongation
        3. Text components have height in [min_text_height, max_text_height]
        """
        area_threshold = config.area_threshold_factor * self._median_area

        text_components: List[TextComponent] = []
        graphics_components: List[GraphicsComponent] = []
        mixed_components: List[ConnectedComponent] = []

        for comp in components:
            # Rule 1: Area check
            is_small_area = comp.area < area_threshold

            # Rule 2: Elongation check
            is_not_too_elongated = comp.elongation < config.max_elongation

            # Rule 3: Height check
            is_valid_height = (
                config.min_text_height <= comp.height <= config.max_text_height
            )

            # Classification logic
            if is_small_area and is_not_too_elongated and is_valid_height:
                # Likely text
                text_comp = TextComponent(
                    label=comp.label,
                    bbox=comp.bbox,
                    area=comp.area,
                    centroid=comp.centroid,
                    pixels=comp.pixels,
                    component_type=ComponentType.TEXT,
                    confidence=0.9 if is_not_too_elongated else 0.7,
                )
                text_components.append(text_comp)

            elif not is_small_area and comp.elongation > config.max_elongation * 2:
                # Likely graphics (lines, curves)
                graphics_comp = GraphicsComponent(
                    label=comp.label,
                    bbox=comp.bbox,
                    area=comp.area,
                    centroid=comp.centroid,
                    pixels=comp.pixels,
                    component_type=ComponentType.GRAPHICS,
                    confidence=0.9,
                )
                graphics_components.append(graphics_comp)

            elif not is_small_area:
                # Large component - could be graphics or mixed
                if comp.height > config.max_text_height:
                    graphics_comp = GraphicsComponent(
                        label=comp.label,
                        bbox=comp.bbox,
                        area=comp.area,
                        centroid=comp.centroid,
                        pixels=comp.pixels,
                        component_type=ComponentType.GRAPHICS,
                        confidence=0.8,
                    )
                    graphics_components.append(graphics_comp)
                else:
                    # Could be text or graphics - mark as mixed
                    mixed_components.append(comp)

            else:
                # Default to graphics for ambiguous cases
                graphics_comp = GraphicsComponent(
                    label=comp.label,
                    bbox=comp.bbox,
                    area=comp.area,
                    centroid=comp.centroid,
                    pixels=comp.pixels,
                    component_type=ComponentType.GRAPHICS,
                    confidence=0.6,
                )
                graphics_components.append(graphics_comp)

        return text_components, graphics_components, mixed_components

    def _group_text_into_lines(
        self,
        text_components: List[TextComponent],
        config: SeparationConfig,
    ) -> List[TextLine]:
        """Group text components into text lines using Hough Transform.

        This groups characters that are:
        1. Roughly on the same baseline
        2. Close together horizontally
        """
        if not text_components:
            return []

        # Sort by y-coordinate (top to bottom)
        sorted_components = sorted(text_components, key=lambda c: c.y)

        # Estimate average character width
        if text_components:
            self._avg_char_width = np.mean([c.width for c in text_components])

        # Group by baseline proximity
        lines: List[TextLine] = []
        used = set()

        for comp in sorted_components:
            if comp.label in used:
                continue

            # Start a new line
            line = TextLine(components=[comp])
            used.add(comp.label)

            # Find components on the same baseline
            baseline_y = comp.y + comp.height  # Bottom of character
            tolerance = comp.height * 0.5

            for other in sorted_components:
                if other.label in used:
                    continue

                other_baseline = other.y + other.height
                if abs(other_baseline - baseline_y) < tolerance:
                    # Check horizontal proximity
                    gap = self._horizontal_gap(comp, other)
                    max_gap = config.gap_threshold_factor * self._avg_char_width

                    if gap < max_gap:
                        line.components.append(other)
                        used.add(other.label)

            # Sort components in line by x-coordinate
            line.components.sort(key=lambda c: c.x)
            line.baseline_y = baseline_y

            lines.append(line)

        return lines

    def _horizontal_gap(
        self,
        comp1: ConnectedComponent,
        comp2: ConnectedComponent,
    ) -> float:
        """Calculate horizontal gap between two components."""
        left1, right1 = comp1.x, comp1.x + comp1.width
        left2, right2 = comp2.x, comp2.x + comp2.width

        if right1 < left2:
            return left2 - right1
        elif right2 < left1:
            return left1 - right2
        else:
            return 0  # Overlapping

    def _handle_intersections(
        self,
        mixed_regions: List[ConnectedComponent],
        image: np.ndarray,
        config: SeparationConfig,
    ) -> Tuple[List[TextComponent], List[GraphicsComponent]]:
        """Handle mixed regions using skeleton segmentation.

        When text and graphics intersect, we use morphological operations
        to separate them.
        """
        text_from_mixed: List[TextComponent] = []
        graphics_from_mixed: List[GraphicsComponent] = []

        for region in mixed_regions:
            if region.pixels is None:
                continue

            # Extract region
            x, y, w, h = region.bbox
            roi = region.pixels[y:y+h, x:x+w]

            # Skeletonize to find thin structures (graphics)
            try:
                from skimage.morphology import skeletonize
                skeleton = skeletonize(roi > 0).astype(np.uint8) * 255
            except ImportError:
                # Fallback: use morphological thinning
                skeleton = cv2.ximgproc.thinning(roi) if hasattr(cv2, "ximgproc") else roi

            # Dilate skeleton to get graphics region
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
            graphics_region = cv2.dilate(skeleton, kernel, iterations=1)

            # Subtract from original to get potential text
            text_region = cv2.subtract(roi, graphics_region)

            # Re-analyze the separated regions
            if np.sum(graphics_region) > config.min_component_area:
                graphics_comp = GraphicsComponent(
                    label=region.label,
                    bbox=region.bbox,
                    area=int(np.sum(graphics_region) / 255),
                    centroid=region.centroid,
                    pixels=self._embed_roi(graphics_region, region.bbox, image.shape),
                    component_type=ComponentType.GRAPHICS,
                    confidence=0.7,
                )
                graphics_from_mixed.append(graphics_comp)

            if np.sum(text_region) > config.min_component_area:
                text_comp = TextComponent(
                    label=region.label + 10000,  # Unique label
                    bbox=region.bbox,
                    area=int(np.sum(text_region) / 255),
                    centroid=region.centroid,
                    pixels=self._embed_roi(text_region, region.bbox, image.shape),
                    component_type=ComponentType.TEXT,
                    confidence=0.6,
                )
                text_from_mixed.append(text_comp)

        return text_from_mixed, graphics_from_mixed

    def _embed_roi(
        self,
        roi: np.ndarray,
        bbox: Tuple[int, int, int, int],
        image_shape: Tuple[int, ...],
    ) -> np.ndarray:
        """Embed a ROI back into a full-sized mask."""
        x, y, w, h = bbox
        full_mask = np.zeros(image_shape[:2], dtype=np.uint8)
        full_mask[y:y+h, x:x+w] = roi
        return full_mask

    def _create_mask(
        self,
        components: List[ConnectedComponent],
        shape: Tuple[int, int],
    ) -> np.ndarray:
        """Create a binary mask from components."""
        mask = np.zeros(shape, dtype=np.uint8)
        for comp in components:
            if comp.pixels is not None:
                mask = cv2.bitwise_or(mask, comp.pixels)
        return mask


def separate_text_from_graphics(
    binary_image: np.ndarray,
    config: Optional[SeparationConfig] = None,
) -> SeparationResult:
    """Convenience function to separate text from graphics.

    Args:
        binary_image: Binary image (0 = background, 255 = foreground)
        config: Optional separation configuration

    Returns:
        SeparationResult with text and graphics masks
    """
    separator = FletcherKasturiSeparator(binary_image, config)
    return separator.separate()


def is_separation_available() -> bool:
    """Check if text/graphics separation is available."""
    return OPENCV_AVAILABLE
