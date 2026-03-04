"""
Preview Generation Module.

Generates visual preview images of extracted entities overlaid on the
original drawing. This allows users to inspect vectorization results
before committing to AutoCAD.

Features:
- Color-coded entity visualization by type
- Layer-based coloring
- Transparency overlay on source image
- Confidence-based opacity
- Entity count legend
- Interactive zoom regions (future)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import math

import numpy as np
import cv2
import structlog

logger = structlog.get_logger(__name__)


# Default colors for entity types (BGR format for OpenCV)
ENTITY_COLORS = {
    "line": (255, 0, 0),      # Blue
    "circle": (0, 255, 0),    # Green
    "arc": (0, 255, 255),     # Yellow
    "mtext": (0, 0, 255),     # Red
    "text": (0, 0, 255),      # Red
    "block": (255, 0, 255),   # Magenta
    "dimension": (255, 255, 0),  # Cyan
    "polyline": (128, 128, 255),  # Light red
    "lwpolyline": (128, 128, 255),  # Light red
    "spline": (255, 128, 0),  # Orange-blue
    "ellipse": (0, 128, 255),  # Orange
    "hatch": (128, 255, 128),  # Light green
}

# Layer prefix colors (similar to AutoCAD ACI)
LAYER_COLORS = {
    "A-": (0, 255, 0),     # Green - Architectural
    "E-": (0, 255, 255),   # Yellow - Electrical
    "M-": (255, 0, 255),   # Magenta - Mechanical
    "P-": (255, 0, 0),     # Blue - Plumbing
    "S-": (0, 128, 255),   # Orange - Structural
    "L-": (0, 255, 128),   # Spring green - Landscape
    "C-": (255, 255, 0),   # Cyan - Civil
    "F-": (0, 0, 255),     # Red - Fire
}


@dataclass
class PreviewConfig:
    """Configuration for preview generation."""
    # Colors
    use_entity_colors: bool = True  # Color by entity type
    use_layer_colors: bool = False  # Color by layer prefix
    default_color: Tuple[int, int, int] = (0, 255, 0)  # Green

    # Overlay
    overlay_alpha: float = 0.6  # Entity overlay opacity (0-1)
    background_dim: float = 0.3  # Background dimming factor

    # Line rendering
    line_thickness: int = 2
    point_radius: int = 4
    text_scale: float = 0.5
    text_thickness: int = 1

    # Legend
    include_legend: bool = True
    legend_position: str = "top-right"  # top-left, top-right, bottom-left, bottom-right
    legend_padding: int = 10
    legend_line_height: int = 20

    # Confidence-based opacity
    use_confidence_opacity: bool = False
    min_confidence_opacity: float = 0.3

    # Output
    output_format: str = "png"
    jpeg_quality: int = 95
    max_dimension: int = 4096  # Resize if larger


@dataclass
class PreviewResult:
    """Result of preview generation."""
    success: bool
    output_path: Optional[str] = None
    entity_count: int = 0
    entity_counts_by_type: Dict[str, int] = field(default_factory=dict)
    width: int = 0
    height: int = 0
    error: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "success": self.success,
            "output_path": self.output_path,
            "entity_count": self.entity_count,
            "entity_counts_by_type": self.entity_counts_by_type,
            "width": self.width,
            "height": self.height,
            "error": self.error,
        }


def _get_color_for_entity(
    entity: Any,
    config: PreviewConfig,
) -> Tuple[int, int, int]:
    """Get the color to use for an entity."""
    if config.use_layer_colors:
        layer = getattr(entity, 'layer', '0')
        for prefix, color in LAYER_COLORS.items():
            if layer.upper().startswith(prefix):
                return color

    if config.use_entity_colors:
        entity_type = entity.entity_type
        if hasattr(entity_type, 'value'):
            entity_type = entity_type.value
        return ENTITY_COLORS.get(entity_type.lower(), config.default_color)

    return config.default_color


def _get_opacity_for_entity(
    entity: Any,
    config: PreviewConfig,
) -> float:
    """Get opacity based on entity confidence."""
    if not config.use_confidence_opacity:
        return config.overlay_alpha

    confidence = getattr(entity, 'confidence', 1.0)
    min_opacity = config.min_confidence_opacity
    return min_opacity + (config.overlay_alpha - min_opacity) * confidence


def _draw_line(
    image: np.ndarray,
    entity: Any,
    color: Tuple[int, int, int],
    config: PreviewConfig,
) -> None:
    """Draw a line entity."""
    props = entity.properties
    start = props.get("start", (0, 0))
    end = props.get("end", (0, 0))

    pt1 = (int(start[0]), int(start[1]))
    pt2 = (int(end[0]), int(end[1]))

    cv2.line(image, pt1, pt2, color, config.line_thickness, cv2.LINE_AA)


def _draw_circle(
    image: np.ndarray,
    entity: Any,
    color: Tuple[int, int, int],
    config: PreviewConfig,
) -> None:
    """Draw a circle entity."""
    props = entity.properties
    center = props.get("center", (0, 0))
    radius = props.get("radius", 10)

    center_pt = (int(center[0]), int(center[1]))
    cv2.circle(image, center_pt, int(radius), color, config.line_thickness, cv2.LINE_AA)


def _draw_arc(
    image: np.ndarray,
    entity: Any,
    color: Tuple[int, int, int],
    config: PreviewConfig,
) -> None:
    """Draw an arc entity."""
    props = entity.properties
    center = props.get("center", (0, 0))
    radius = props.get("radius", 10)
    start_angle = props.get("start_angle", 0)
    end_angle = props.get("end_angle", 90)

    center_pt = (int(center[0]), int(center[1]))
    axes = (int(radius), int(radius))

    cv2.ellipse(
        image,
        center_pt,
        axes,
        0,  # rotation
        start_angle,
        end_angle,
        color,
        config.line_thickness,
        cv2.LINE_AA,
    )


def _draw_ellipse(
    image: np.ndarray,
    entity: Any,
    color: Tuple[int, int, int],
    config: PreviewConfig,
) -> None:
    """Draw an ellipse entity."""
    props = entity.properties
    center = props.get("center", (0, 0))
    major_axis = props.get("major_axis", (10, 0))
    ratio = props.get("ratio", 0.5)

    center_pt = (int(center[0]), int(center[1]))

    # Calculate axes
    major_len = math.sqrt(major_axis[0] ** 2 + major_axis[1] ** 2)
    minor_len = major_len * ratio
    angle = math.degrees(math.atan2(major_axis[1], major_axis[0]))

    axes = (int(major_len), int(minor_len))

    cv2.ellipse(
        image,
        center_pt,
        axes,
        angle,
        0,
        360,
        color,
        config.line_thickness,
        cv2.LINE_AA,
    )


def _draw_text(
    image: np.ndarray,
    entity: Any,
    color: Tuple[int, int, int],
    config: PreviewConfig,
) -> None:
    """Draw a text/mtext entity."""
    props = entity.properties
    content = props.get("content", props.get("text", ""))
    position = props.get("position", props.get("insertion_point", (0, 0)))

    if not content:
        return

    pt = (int(position[0]), int(position[1]))

    # Draw text with background for visibility
    (text_w, text_h), baseline = cv2.getTextSize(
        content[:50],  # Truncate long text
        cv2.FONT_HERSHEY_SIMPLEX,
        config.text_scale,
        config.text_thickness,
    )

    # Background rectangle
    cv2.rectangle(
        image,
        (pt[0] - 2, pt[1] - text_h - 2),
        (pt[0] + text_w + 2, pt[1] + 2),
        (255, 255, 255),
        -1,
    )

    # Text
    cv2.putText(
        image,
        content[:50],
        pt,
        cv2.FONT_HERSHEY_SIMPLEX,
        config.text_scale,
        color,
        config.text_thickness,
        cv2.LINE_AA,
    )


def _draw_polyline(
    image: np.ndarray,
    entity: Any,
    color: Tuple[int, int, int],
    config: PreviewConfig,
) -> None:
    """Draw a polyline entity."""
    props = entity.properties
    points = props.get("points", props.get("vertices", []))
    is_closed = props.get("closed", False)

    if len(points) < 2:
        return

    pts = np.array([(int(p[0]), int(p[1])) for p in points], dtype=np.int32)

    if is_closed:
        cv2.polylines(image, [pts], True, color, config.line_thickness, cv2.LINE_AA)
    else:
        cv2.polylines(image, [pts], False, color, config.line_thickness, cv2.LINE_AA)


def _draw_block(
    image: np.ndarray,
    entity: Any,
    color: Tuple[int, int, int],
    config: PreviewConfig,
) -> None:
    """Draw a block/symbol entity as a marker."""
    props = entity.properties
    position = props.get("position", props.get("insertion_point", (0, 0)))
    name = props.get("block_name", props.get("name", "BLOCK"))

    pt = (int(position[0]), int(position[1]))

    # Draw cross marker
    size = config.point_radius * 2
    cv2.line(image, (pt[0] - size, pt[1]), (pt[0] + size, pt[1]), color, config.line_thickness)
    cv2.line(image, (pt[0], pt[1] - size), (pt[0], pt[1] + size), color, config.line_thickness)

    # Draw circle around
    cv2.circle(image, pt, size, color, config.line_thickness, cv2.LINE_AA)

    # Label
    cv2.putText(
        image,
        name[:10],
        (pt[0] + size + 2, pt[1]),
        cv2.FONT_HERSHEY_SIMPLEX,
        config.text_scale * 0.8,
        color,
        config.text_thickness,
        cv2.LINE_AA,
    )


def _draw_dimension(
    image: np.ndarray,
    entity: Any,
    color: Tuple[int, int, int],
    config: PreviewConfig,
) -> None:
    """Draw a dimension entity."""
    props = entity.properties
    start = props.get("start", (0, 0))
    end = props.get("end", (0, 0))
    value = props.get("value", "")

    pt1 = (int(start[0]), int(start[1]))
    pt2 = (int(end[0]), int(end[1]))

    # Draw dimension line
    cv2.line(image, pt1, pt2, color, config.line_thickness, cv2.LINE_AA)

    # Draw arrows at endpoints
    _draw_arrow(image, pt1, pt2, color, config)
    _draw_arrow(image, pt2, pt1, color, config)

    # Draw value text at midpoint
    mid = ((pt1[0] + pt2[0]) // 2, (pt1[1] + pt2[1]) // 2 - 10)
    cv2.putText(
        image,
        str(value),
        mid,
        cv2.FONT_HERSHEY_SIMPLEX,
        config.text_scale,
        color,
        config.text_thickness,
        cv2.LINE_AA,
    )


def _draw_arrow(
    image: np.ndarray,
    from_pt: Tuple[int, int],
    to_pt: Tuple[int, int],
    color: Tuple[int, int, int],
    config: PreviewConfig,
) -> None:
    """Draw an arrowhead at from_pt pointing toward to_pt."""
    arrow_length = 10
    arrow_angle = math.pi / 6  # 30 degrees

    # Direction from from_pt to to_pt
    dx = to_pt[0] - from_pt[0]
    dy = to_pt[1] - from_pt[1]
    length = math.sqrt(dx * dx + dy * dy)

    if length < 1:
        return

    # Unit vector
    ux = dx / length
    uy = dy / length

    # Arrow points
    angle1 = math.atan2(uy, ux) + math.pi - arrow_angle
    angle2 = math.atan2(uy, ux) + math.pi + arrow_angle

    p1 = (
        int(from_pt[0] + arrow_length * math.cos(angle1)),
        int(from_pt[1] + arrow_length * math.sin(angle1)),
    )
    p2 = (
        int(from_pt[0] + arrow_length * math.cos(angle2)),
        int(from_pt[1] + arrow_length * math.sin(angle2)),
    )

    cv2.line(image, from_pt, p1, color, config.line_thickness, cv2.LINE_AA)
    cv2.line(image, from_pt, p2, color, config.line_thickness, cv2.LINE_AA)


def _draw_legend(
    image: np.ndarray,
    entity_counts: Dict[str, int],
    config: PreviewConfig,
) -> None:
    """Draw a legend showing entity types and counts."""
    if not entity_counts:
        return

    h, w = image.shape[:2]

    # Calculate legend size
    max_label_width = 0
    items = []
    for entity_type, count in sorted(entity_counts.items()):
        label = f"{entity_type}: {count}"
        (text_w, text_h), _ = cv2.getTextSize(
            label,
            cv2.FONT_HERSHEY_SIMPLEX,
            config.text_scale,
            config.text_thickness,
        )
        max_label_width = max(max_label_width, text_w)
        items.append((entity_type, count, label))

    legend_width = max_label_width + 40  # Color box + padding
    legend_height = len(items) * config.legend_line_height + config.legend_padding * 2

    # Position
    if config.legend_position == "top-left":
        x = config.legend_padding
        y = config.legend_padding
    elif config.legend_position == "top-right":
        x = w - legend_width - config.legend_padding
        y = config.legend_padding
    elif config.legend_position == "bottom-left":
        x = config.legend_padding
        y = h - legend_height - config.legend_padding
    else:  # bottom-right
        x = w - legend_width - config.legend_padding
        y = h - legend_height - config.legend_padding

    # Background
    cv2.rectangle(
        image,
        (x, y),
        (x + legend_width, y + legend_height),
        (255, 255, 255),
        -1,
    )
    cv2.rectangle(
        image,
        (x, y),
        (x + legend_width, y + legend_height),
        (0, 0, 0),
        1,
    )

    # Items
    for i, (entity_type, count, label) in enumerate(items):
        item_y = y + config.legend_padding + i * config.legend_line_height + 15
        color = ENTITY_COLORS.get(entity_type.lower(), config.default_color)

        # Color box
        cv2.rectangle(
            image,
            (x + 5, item_y - 10),
            (x + 20, item_y + 2),
            color,
            -1,
        )

        # Label
        cv2.putText(
            image,
            label,
            (x + 25, item_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            config.text_scale,
            (0, 0, 0),
            config.text_thickness,
            cv2.LINE_AA,
        )


def generate_preview(
    source_image: Union[str, Path, np.ndarray],
    entities: List[Any],
    output_path: Optional[Union[str, Path]] = None,
    config: Optional[PreviewConfig] = None,
) -> PreviewResult:
    """
    Generate a preview image with entities overlaid.

    Args:
        source_image: Path to source image or numpy array
        entities: List of EntityToCreate objects
        output_path: Optional output path (auto-generated if not provided)
        config: Optional configuration

    Returns:
        PreviewResult with output path and statistics

    Example:
        >>> result = generate_preview("drawing.png", entities, "preview.png")
        >>> if result.success:
        ...     print(f"Preview saved to {result.output_path}")
    """
    if config is None:
        config = PreviewConfig()

    result = PreviewResult(success=False)

    try:
        # Load source image
        if isinstance(source_image, (str, Path)):
            image = cv2.imread(str(source_image))
            if image is None:
                raise FileNotFoundError(f"Could not load image: {source_image}")
        else:
            image = source_image.copy()

        h, w = image.shape[:2]
        result.width = w
        result.height = h

        # Resize if too large
        if max(h, w) > config.max_dimension:
            scale = config.max_dimension / max(h, w)
            image = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
            h, w = image.shape[:2]
            result.width = w
            result.height = h

        # Create overlay layer
        overlay = image.copy()

        # Dim background
        if config.background_dim > 0:
            overlay = (overlay * (1 - config.background_dim)).astype(np.uint8)

        # Count entities by type
        entity_counts: Dict[str, int] = {}

        # Draw entities
        for entity in entities:
            entity_type = entity.entity_type
            if hasattr(entity_type, 'value'):
                entity_type = entity_type.value
            entity_type = entity_type.lower()

            # Update counts
            entity_counts[entity_type] = entity_counts.get(entity_type, 0) + 1

            # Get color
            color = _get_color_for_entity(entity, config)

            # Draw based on type
            if entity_type == "line":
                _draw_line(overlay, entity, color, config)
            elif entity_type == "circle":
                _draw_circle(overlay, entity, color, config)
            elif entity_type == "arc":
                _draw_arc(overlay, entity, color, config)
            elif entity_type == "ellipse":
                _draw_ellipse(overlay, entity, color, config)
            elif entity_type in ("mtext", "text"):
                _draw_text(overlay, entity, color, config)
            elif entity_type in ("polyline", "lwpolyline"):
                _draw_polyline(overlay, entity, color, config)
            elif entity_type in ("block", "insert"):
                _draw_block(overlay, entity, color, config)
            elif entity_type == "dimension":
                _draw_dimension(overlay, entity, color, config)
            elif entity_type == "spline":
                _draw_polyline(overlay, entity, color, config)  # Approximate as polyline

        # Blend overlay with original
        output = cv2.addWeighted(image, 1 - config.overlay_alpha, overlay, config.overlay_alpha, 0)

        # Draw legend
        if config.include_legend and entity_counts:
            _draw_legend(output, entity_counts, config)

        # Save output
        if output_path is None:
            if isinstance(source_image, (str, Path)):
                source_path = Path(source_image)
                output_path = source_path.with_name(f"{source_path.stem}_preview.{config.output_format}")
            else:
                output_path = Path("preview.png")

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if config.output_format.lower() in ("jpg", "jpeg"):
            cv2.imwrite(
                str(output_path),
                output,
                [cv2.IMWRITE_JPEG_QUALITY, config.jpeg_quality],
            )
        else:
            cv2.imwrite(str(output_path), output)

        result.success = True
        result.output_path = str(output_path)
        result.entity_count = len(entities)
        result.entity_counts_by_type = entity_counts

        logger.info(
            "preview_generated",
            output_path=str(output_path),
            entity_count=len(entities),
            size=f"{w}x{h}",
        )

    except Exception as e:
        result.error = str(e)
        logger.exception("preview_generation_failed", error=str(e))

    return result


def generate_preview_from_pipeline(
    pipeline_result: Any,
    output_path: Optional[Union[str, Path]] = None,
    config: Optional[PreviewConfig] = None,
) -> PreviewResult:
    """
    Generate preview from a UnifiedPipeline result.

    Args:
        pipeline_result: Result from UnifiedPipeline.process()
        output_path: Optional output path
        config: Optional configuration

    Returns:
        PreviewResult
    """
    if not hasattr(pipeline_result, 'image_path') or not hasattr(pipeline_result, 'entities'):
        return PreviewResult(
            success=False,
            error="Invalid pipeline result: missing image_path or entities",
        )

    return generate_preview(
        source_image=pipeline_result.image_path,
        entities=pipeline_result.entities or [],
        output_path=output_path,
        config=config,
    )
