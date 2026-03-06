"""
Phase 4: Adaptive Extraction - Direct, Guided, and Selective Extraction Strategies.

This module extracts AutoCAD entities from Gemini's drawing analysis using
the optimal strategy for each element type:

1. **Direct Extraction**: Convert Gemini coordinates directly to DWG entities
2. **Guided Rasterization**: Generate Raster Design commands for complex tracing
3. **Selective OpenCV**: Use OpenCV for specific regions (hatching, patterns)

Key Features:
- Strategy selection based on Gemini's recommendations
- NCS-compliant layer assignment
- Block name mapping for MEP symbols
- Coordinate transformation using Phase 3 calibration
- Support for lines, arcs, circles, text, and block inserts

Usage:
    >>> result = await extract_all(analysis, calibration, image_path)
    >>> print(f"Extracted {len(result.entities)} entities")
    >>> print(f"Raster commands: {len(result.raster_commands)}")
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING

import numpy as np
import structlog

from .gemini_understanding import (
    DrawingAnalysis,
    DrawingElements,
    DetectedLine,
    DetectedArc,
    DetectedCircle,
    DetectedText,
    DetectedSymbol,
    DetectedDimension,
    SpecialRegion,
    ExtractionStrategy,
)
from .coordinate_calibration import ScaleCalibration

# Lazy imports for optional dependencies
if TYPE_CHECKING:
    from .opencv_extraction import OpenCVExtractor, OpenCVExtractionResult
    from .vtracer_extraction import VTracerExtractor, VTracerExtractionResult
    from ..yolo_detection import YOLOSymbolDetector, DetectedBlock

logger = structlog.get_logger(__name__)


class ExtractionSource(str, Enum):
    """Source/method used for extraction."""
    DIRECT = "direct"
    GUIDED_RASTERIZATION = "guided_rasterization"
    SELECTIVE_OPENCV = "selective_opencv"
    VTRACER = "vtracer"  # Phase B: O(n) vectorization
    HYBRID = "hybrid"


class EntityType(str, Enum):
    """AutoCAD entity types that can be created."""
    LINE = "line"
    ARC = "arc"
    CIRCLE = "circle"
    POLYLINE = "polyline"
    MTEXT = "mtext"
    TEXT = "text"
    BLOCK = "block"
    DIMENSION = "dimension"
    HATCH = "hatch"
    # Phase A additions
    SPLINE = "spline"
    ELLIPSE = "ellipse"
    LWPOLYLINE = "lwpolyline"


# =============================================================================
# Layer Mapping (NCS-compliant)
# =============================================================================

# Map element types to NCS layer prefixes
ELEMENT_TYPE_TO_LAYER = {
    # Architectural
    "wall": "A-WALL",
    "door": "A-DOOR",
    "door_swing": "A-DOOR",
    "window": "A-GLAZ",
    "column": "A-COLS",
    "room": "A-AREA",
    # Mechanical
    "duct": "M-DUCT",
    "diffuser": "M-DIFF",
    "supply_diffuser": "M-DIFF-SUPP",
    "return_diffuser": "M-DIFF-RETN",
    "damper": "M-DAMP",
    "vav": "M-EQPM",
    "ahu": "M-EQPM",
    # Electrical
    "wire": "E-POWR",
    "conduit": "E-COND",
    "outlet": "E-POWR-OUTL",
    "switch": "E-LITE-SWCH",
    "light": "E-LITE",
    "panel": "E-POWR-PANL",
    # Plumbing
    "pipe": "P-PIPE",
    "valve": "P-VALV",
    "fixture": "P-FIXT",
    "equipment": "P-EQPM",
    # Fire
    "sprinkler": "F-SPRL",
    "detector": "F-ALRM-DETC",
    "horn_strobe": "F-ANUN",
    "fire_alarm": "F-ALRM",
    # Low Voltage
    "data": "T-DATA",
    "security": "T-SECU",
    "camera": "T-SECU-CCTV",
    # General
    "dimension": "G-ANNO-DIMS",
    "leader": "G-ANNO-LEAD",
    "note": "G-ANNO-NOTE",
    "title": "G-ANNO-TITL",
    "label": "G-ANNO-TEXT",
    "other": "0",
}

# Map text types to layers
TEXT_TYPE_TO_LAYER = {
    "room_name": "A-AREA-IDEN",
    "dimension": "G-ANNO-DIMS",
    "equipment_tag": "G-ANNO-TAGS",
    "note": "G-ANNO-NOTE",
    "title": "G-ANNO-TITL",
    "label": "G-ANNO-TEXT",
    "other": "G-ANNO-TEXT",
}

# Map symbol types to layers
SYMBOL_TYPE_TO_LAYER = {
    "diffuser": "M-DIFF",
    "outlet": "E-POWR-OUTL",
    "switch": "E-LITE-SWCH",
    "valve": "P-VALV",
    "fixture": "P-FIXT",
    "detector": "F-ALRM-DETC",
    "device": "E-POWR",
    "equipment": "M-EQPM",
    "other": "0",
}

# =============================================================================
# Block Name Mapping
# =============================================================================

# Map (symbol_type, subtype) to block names
SYMBOL_TO_BLOCK = {
    # Mechanical - Diffusers
    ("diffuser", "supply_square"): "M-DIFF-SQ-S",
    ("diffuser", "supply_round"): "M-DIFF-RD-S",
    ("diffuser", "supply_linear"): "M-DIFF-LN-S",
    ("diffuser", "return_square"): "M-DIFF-SQ-R",
    ("diffuser", "return_round"): "M-DIFF-RD-R",
    ("diffuser", "exhaust"): "M-DIFF-EXH",
    ("diffuser", None): "M-DIFF-GEN",
    # Electrical - Outlets
    ("outlet", "duplex"): "E-OUTL-DUP",
    ("outlet", "gfci"): "E-OUTL-GFCI",
    ("outlet", "quad"): "E-OUTL-QUD",
    ("outlet", "floor"): "E-OUTL-FLR",
    ("outlet", "dedicated"): "E-OUTL-DED",
    ("outlet", None): "E-OUTL-GEN",
    # Electrical - Switches
    ("switch", "single_pole"): "E-SWCH-SP",
    ("switch", "three_way"): "E-SWCH-3W",
    ("switch", "dimmer"): "E-SWCH-DIM",
    ("switch", None): "E-SWCH-GEN",
    # Plumbing - Valves
    ("valve", "gate"): "P-VALV-GATE",
    ("valve", "ball"): "P-VALV-BALL",
    ("valve", "butterfly"): "P-VALV-BTRF",
    ("valve", "check"): "P-VALV-CHCK",
    ("valve", "globe"): "P-VALV-GLOB",
    ("valve", None): "P-VALV-GEN",
    # Plumbing - Fixtures
    ("fixture", "sink"): "P-FIXT-SINK",
    ("fixture", "toilet"): "P-FIXT-WC",
    ("fixture", "lavatory"): "P-FIXT-LAV",
    ("fixture", "urinal"): "P-FIXT-URNL",
    ("fixture", "floor_drain"): "P-FIXT-FD",
    ("fixture", None): "P-FIXT-GEN",
    # Fire - Detectors
    ("detector", "smoke"): "F-DETC-SMOK",
    ("detector", "heat"): "F-DETC-HEAT",
    ("detector", "duct"): "F-DETC-DUCT",
    ("detector", None): "F-DETC-GEN",
    # Fire - Notification
    ("horn_strobe", None): "F-ANUN-HS",
    ("sprinkler", "pendant"): "F-SPKL-PND",
    ("sprinkler", "upright"): "F-SPKL-UPR",
    ("sprinkler", "sidewall"): "F-SPKL-SID",
    ("sprinkler", None): "F-SPKL-GEN",
    # Low Voltage
    ("data", "rj45"): "T-DATA-RJ45",
    ("data", None): "T-DATA-GEN",
    ("camera", "dome"): "T-SECU-CAM-D",
    ("camera", "bullet"): "T-SECU-CAM-B",
    ("camera", None): "T-SECU-CAM",
    # General equipment
    ("equipment", None): "EQ-GEN",
    ("device", None): "DEV-GEN",
}

# Map VTool types for rasterization
VTOOL_MAPPING = {
    "polyline": "VFPLINE",
    "contour": "VFCONTOUR",
    "arc": "VARC",
    "circle": "VCIRCLE",
    "line": "VLINE",
    "rectangle": "VRECT",
    "3d_polyline": "VF3DPOLY",
}


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class EntityToCreate:
    """
    Represents an AutoCAD entity to be created.

    Contains all information needed to create the entity including
    coordinates (already in DWG units), layer, and type-specific properties.
    """
    entity_type: EntityType | str
    layer: str
    properties: Dict[str, Any]
    source: ExtractionSource | str = ExtractionSource.DIRECT

    # Optional metadata
    confidence: float = 1.0
    source_element: Optional[str] = None  # Original element ID/description

    # Linetype and lineweight (Phase A additions)
    linetype: str = "Continuous"  # AutoCAD linetype name
    lineweight: float = 0.0  # Lineweight in mm (0 = default/bylayer)

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "entity_type": self.entity_type.value if isinstance(self.entity_type, EntityType) else self.entity_type,
            "layer": self.layer,
            "properties": self.properties,
            "source": self.source.value if isinstance(self.source, ExtractionSource) else self.source,
            "confidence": self.confidence,
            "source_element": self.source_element,
            "linetype": self.linetype,
            "lineweight": self.lineweight,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "EntityToCreate":
        """Create from dictionary."""
        # Ensure properties is a dict (guard against malformed JSON)
        props = data.get("properties", {})
        if not isinstance(props, dict):
            props = {}
        return cls(
            entity_type=data["entity_type"],
            layer=data["layer"],
            properties=props,
            source=data.get("source", "direct"),
            confidence=data.get("confidence", 1.0),
            source_element=data.get("source_element"),
            linetype=data.get("linetype", "Continuous"),
            lineweight=data.get("lineweight", 0.0),
        )


@dataclass
class RasterCommand:
    """
    Represents a Raster Design VTool command for guided tracing.

    These commands are executed in AutoCAD with Raster Design to
    trace complex curves that are difficult to extract directly.
    """
    tool: str  # VTool command name (VFPLINE, VARC, etc.)
    start_point: Tuple[float, float]  # DWG coordinates
    layer: str
    options: Dict[str, Any] = field(default_factory=dict)

    # Optional guidance
    expected_end: Optional[Tuple[float, float]] = None
    path_type: str = "polyline"

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "tool": self.tool,
            "start_point": list(self.start_point),
            "layer": self.layer,
            "options": self.options,
            "expected_end": list(self.expected_end) if self.expected_end else None,
            "path_type": self.path_type,
        }

    def to_command_string(self) -> str:
        """Generate AutoCAD command string for SendStringToExecute."""
        # Format: TOOL start_x,start_y options...
        cmd = f"{self.tool} {self.start_point[0]:.6f},{self.start_point[1]:.6f}"
        return cmd


@dataclass
class ExtractionResult:
    """
    Result of the adaptive extraction process.

    Contains all entities to create and raster commands to execute,
    along with statistics about the extraction.
    """
    entities: List[EntityToCreate] = field(default_factory=list)
    raster_commands: List[RasterCommand] = field(default_factory=list)

    # Statistics
    direct_count: int = 0
    guided_count: int = 0
    opencv_count: int = 0

    # Metadata
    primary_strategy: str = "direct"
    drawing_type: str = ""
    calibration_method: str = ""
    calibration_confidence: float = 0.0

    @property
    def total_entities(self) -> int:
        return len(self.entities)

    @property
    def total_commands(self) -> int:
        return len(self.raster_commands)

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "entities": [e.to_dict() for e in self.entities],
            "raster_commands": [r.to_dict() for r in self.raster_commands],
            "statistics": {
                "total_entities": self.total_entities,
                "total_raster_commands": self.total_commands,
                "direct_count": self.direct_count,
                "guided_count": self.guided_count,
                "opencv_count": self.opencv_count,
            },
            "metadata": {
                "primary_strategy": self.primary_strategy,
                "drawing_type": self.drawing_type,
                "calibration_method": self.calibration_method,
                "calibration_confidence": self.calibration_confidence,
            },
        }


# =============================================================================
# Helper Functions
# =============================================================================

def get_layer_for_element_type(element_type: str, subtype: Optional[str] = None) -> str:
    """
    Get the NCS-compliant layer name for an element type.

    Args:
        element_type: Type of element (wall, duct, outlet, etc.)
        subtype: Optional subtype for more specific layer

    Returns:
        Layer name string
    """
    element_type = element_type.lower().replace("-", "_").replace(" ", "_")

    # Try with subtype first
    if subtype:
        subtype = subtype.lower().replace("-", "_").replace(" ", "_")
        combined = f"{element_type}_{subtype}"
        if combined in ELEMENT_TYPE_TO_LAYER:
            return ELEMENT_TYPE_TO_LAYER[combined]

    # Try base type
    if element_type in ELEMENT_TYPE_TO_LAYER:
        return ELEMENT_TYPE_TO_LAYER[element_type]

    return "0"  # Default layer


def get_layer_for_text(text_type: str) -> str:
    """Get layer for text based on its type."""
    text_type = text_type.lower().replace("-", "_").replace(" ", "_")
    return TEXT_TYPE_TO_LAYER.get(text_type, "G-ANNO-TEXT")


def get_layer_for_symbol(symbol_type: str) -> str:
    """Get layer for symbol based on its type."""
    symbol_type = symbol_type.lower().replace("-", "_").replace(" ", "_")
    return SYMBOL_TYPE_TO_LAYER.get(symbol_type, "0")


def get_block_name(symbol_type: str, subtype: Optional[str] = None) -> str:
    """
    Get block name for a symbol.

    Args:
        symbol_type: Type of symbol
        subtype: Optional subtype

    Returns:
        Block name string
    """
    symbol_type = symbol_type.lower().replace("-", "_").replace(" ", "_")
    subtype = subtype.lower().replace("-", "_").replace(" ", "_") if subtype else None

    # Try exact match
    if (symbol_type, subtype) in SYMBOL_TO_BLOCK:
        return SYMBOL_TO_BLOCK[(symbol_type, subtype)]

    # Try without subtype
    if (symbol_type, None) in SYMBOL_TO_BLOCK:
        return SYMBOL_TO_BLOCK[(symbol_type, None)]

    # Generate default block name
    if subtype:
        return f"{symbol_type.upper()}-{subtype.upper()}"
    return f"{symbol_type.upper()}-GEN"


def get_vtool_for_path_type(path_type: str) -> str:
    """Get Raster Design VTool command for a path type."""
    path_type = path_type.lower().replace("-", "_").replace(" ", "_")
    return VTOOL_MAPPING.get(path_type, "VFPLINE")


# =============================================================================
# Strategy A: Direct Extraction
# =============================================================================

def extract_lines_direct(
    elements: DrawingElements,
    calibration: ScaleCalibration,
) -> List[EntityToCreate]:
    """Extract lines directly from Gemini's analysis."""
    entities = []

    for line in elements.lines:
        # Convert coordinates
        start_dwg = calibration.to_dwg(*line.start)
        end_dwg = calibration.to_dwg(*line.end)

        # Get layer
        layer = line.layer_suggestion or get_layer_for_element_type(line.line_type)

        # Map linetype
        linetype = line.linetype.upper()
        if linetype == "CONTINUOUS":
            linetype = "Continuous"
        elif linetype in ("DASHED", "HIDDEN"):
            linetype = "HIDDEN"
        elif linetype == "CENTER":
            linetype = "CENTER"

        entities.append(EntityToCreate(
            entity_type=EntityType.LINE,
            layer=layer,
            properties={
                "start": start_dwg,
                "end": end_dwg,
                "linetype": linetype,
            },
            source=ExtractionSource.DIRECT,
            source_element=f"line_{line.line_type}",
        ))

    return entities


def extract_arcs_direct(
    elements: DrawingElements,
    calibration: ScaleCalibration,
) -> List[EntityToCreate]:
    """Extract arcs directly from Gemini's analysis."""
    entities = []

    for arc in elements.arcs:
        # Convert coordinates
        center_dwg = calibration.to_dwg(*arc.center)
        radius_dwg = calibration.scale_length(arc.radius)

        # Get layer
        layer = get_layer_for_element_type(arc.arc_type)

        entities.append(EntityToCreate(
            entity_type=EntityType.ARC,
            layer=layer,
            properties={
                "center": center_dwg,
                "radius": radius_dwg,
                "start_angle": arc.start_angle,
                "end_angle": arc.end_angle,
            },
            source=ExtractionSource.DIRECT,
            source_element=f"arc_{arc.arc_type}",
        ))

    return entities


def extract_circles_direct(
    elements: DrawingElements,
    calibration: ScaleCalibration,
) -> List[EntityToCreate]:
    """Extract circles directly from Gemini's analysis."""
    entities = []

    for circle in elements.circles:
        # Convert coordinates
        center_dwg = calibration.to_dwg(*circle.center)
        radius_dwg = calibration.scale_length(circle.radius)

        # Get layer
        layer = get_layer_for_element_type(circle.circle_type)

        entities.append(EntityToCreate(
            entity_type=EntityType.CIRCLE,
            layer=layer,
            properties={
                "center": center_dwg,
                "radius": radius_dwg,
            },
            source=ExtractionSource.DIRECT,
            source_element=f"circle_{circle.circle_type}",
        ))

    return entities


def extract_text_direct(
    elements: DrawingElements,
    calibration: ScaleCalibration,
) -> List[EntityToCreate]:
    """Extract text directly from Gemini's analysis."""
    entities = []

    for text in elements.text:
        # Convert coordinates
        position_dwg = calibration.to_dwg(*text.position)
        height_dwg = calibration.scale_length(text.height_px)

        # Ensure minimum readable text height (0.09375" = 3/32" is standard minimum)
        # For feet units, this becomes 0.0078125'
        min_height = 0.09375 if calibration.units in ("inches", "in") else 0.125
        height_dwg = max(height_dwg, min_height)

        # Get layer
        layer = get_layer_for_text(text.text_type)

        entities.append(EntityToCreate(
            entity_type=EntityType.MTEXT,
            layer=layer,
            properties={
                "content": text.content,
                "position": position_dwg,
                "height": height_dwg,
                "text_type": text.text_type,
            },
            source=ExtractionSource.DIRECT,
            source_element=f"text_{text.text_type}",
        ))

    return entities


def extract_symbols_direct(
    elements: DrawingElements,
    calibration: ScaleCalibration,
) -> List[EntityToCreate]:
    """Extract symbols as block references from Gemini's analysis."""
    entities = []

    for symbol in elements.symbols:
        # Convert coordinates
        position_dwg = calibration.to_dwg(*symbol.position)

        # Get block name and layer
        block_name = get_block_name(symbol.symbol_type, symbol.subtype)
        layer = get_layer_for_symbol(symbol.symbol_type)

        # Build attributes
        attributes = {}
        if symbol.tag:
            attributes["TAG"] = symbol.tag
        if symbol.size:
            attributes["SIZE"] = symbol.size
        for i, text in enumerate(symbol.associated_text):
            attributes[f"TEXT{i+1}"] = text

        entities.append(EntityToCreate(
            entity_type=EntityType.BLOCK,
            layer=layer,
            properties={
                "block_name": block_name,
                "position": position_dwg,
                "rotation": symbol.rotation,
                "scale": 1.0,
                "attributes": attributes,
            },
            source=ExtractionSource.DIRECT,
            source_element=f"symbol_{symbol.symbol_type}_{symbol.subtype or 'generic'}",
        ))

    return entities


def extract_dimensions_direct(
    elements: DrawingElements,
    calibration: ScaleCalibration,
) -> List[EntityToCreate]:
    """Extract dimensions from Gemini's analysis."""
    entities = []

    for dim in elements.dimensions:
        # Convert coordinates
        start_dwg = calibration.to_dwg(*dim.start)
        end_dwg = calibration.to_dwg(*dim.end)
        text_pos_dwg = calibration.to_dwg(*dim.text_position)

        entities.append(EntityToCreate(
            entity_type=EntityType.DIMENSION,
            layer="G-ANNO-DIMS",
            properties={
                "start": start_dwg,
                "end": end_dwg,
                "text_position": text_pos_dwg,
                "text": dim.value,
                "numeric_value": dim.numeric_value,
                "unit": dim.unit,
            },
            source=ExtractionSource.DIRECT,
            source_element="dimension",
        ))

    return entities


async def direct_extraction(
    analysis: DrawingAnalysis,
    calibration: ScaleCalibration,
) -> List[EntityToCreate]:
    """
    Extract all elements directly from Gemini's coordinate output.

    This is the primary extraction method for clean drawings with
    clear geometry that Gemini can accurately locate.

    Args:
        analysis: DrawingAnalysis from Phase 2
        calibration: ScaleCalibration from Phase 3

    Returns:
        List of EntityToCreate ready for AutoCAD
    """
    entities = []

    # Extract each element type
    entities.extend(extract_lines_direct(analysis.elements, calibration))
    entities.extend(extract_arcs_direct(analysis.elements, calibration))
    entities.extend(extract_circles_direct(analysis.elements, calibration))
    entities.extend(extract_text_direct(analysis.elements, calibration))
    entities.extend(extract_symbols_direct(analysis.elements, calibration))
    entities.extend(extract_dimensions_direct(analysis.elements, calibration))

    logger.info(
        "direct_extraction_complete",
        lines=len(analysis.elements.lines),
        arcs=len(analysis.elements.arcs),
        circles=len(analysis.elements.circles),
        text=len(analysis.elements.text),
        symbols=len(analysis.elements.symbols),
        dimensions=len(analysis.elements.dimensions),
        total_entities=len(entities),
    )

    return entities


# =============================================================================
# Strategy B: Guided Rasterization
# =============================================================================

async def guided_rasterization(
    analysis: DrawingAnalysis,
    calibration: ScaleCalibration,
    image_path: Path,
) -> List[RasterCommand]:
    """
    Generate Raster Design VTool commands for complex regions.

    Gemini provides the starting points and expected paths;
    Raster Design VTools do the precise pixel-level tracing.

    Args:
        analysis: DrawingAnalysis from Phase 2
        calibration: ScaleCalibration from Phase 3
        image_path: Path to the source image

    Returns:
        List of RasterCommand for AutoCAD execution
    """
    commands = []

    # Process special regions marked for guided rasterization
    for region in analysis.extraction_strategy.special_regions:
        if region.strategy != "guided_rasterization":
            continue

        # Get the VTool for this region
        tool = get_vtool_for_path_type(region.expected_pattern or "polyline")

        # Calculate starting point (center of region by default)
        center_x = (region.bounds[0] + region.bounds[2]) / 2
        center_y = (region.bounds[1] + region.bounds[3]) / 2
        start_dwg = calibration.to_dwg(center_x, center_y)

        # Determine layer based on region context
        layer = "0"  # Default - would be enhanced with region analysis

        commands.append(RasterCommand(
            tool=tool,
            start_point=start_dwg,
            layer=layer,
            options={
                "gap_jump": 3,
                "corner_threshold": 45,
            },
            path_type=region.expected_pattern or "polyline",
        ))

        logger.debug(
            "guided_rasterization_command",
            tool=tool,
            region_bounds=region.bounds,
            reason=region.reason,
        )

    logger.info(
        "guided_rasterization_complete",
        commands_generated=len(commands),
    )

    return commands


# =============================================================================
# Strategy C: Selective OpenCV
# =============================================================================

async def selective_opencv(
    analysis: DrawingAnalysis,
    calibration: ScaleCalibration,
    image_path: Path,
) -> List[EntityToCreate]:
    """
    Use OpenCV for specific regions where it excels.

    Gemini pre-filters regions; OpenCV batch-detects patterns;
    Results are converted to entities with proper coordinates.

    Args:
        analysis: DrawingAnalysis from Phase 2
        calibration: ScaleCalibration from Phase 3
        image_path: Path to the source image

    Returns:
        List of EntityToCreate from OpenCV detection
    """
    entities = []

    # Check if we have any regions for OpenCV processing
    opencv_regions = [
        r for r in analysis.extraction_strategy.special_regions
        if r.strategy == "selective_opencv"
    ]

    if not opencv_regions:
        logger.debug("no_opencv_regions", count=0)
        return entities

    # Try to import OpenCV
    try:
        import cv2
        import numpy as np
    except ImportError:
        logger.warning(
            "opencv_not_available",
            message="OpenCV not installed, skipping selective extraction",
        )
        return entities

    # Load image
    try:
        image = cv2.imread(str(image_path))
        if image is None:
            logger.warning("failed_to_load_image", path=str(image_path))
            return entities
    except Exception as e:
        logger.warning("image_load_error", error=str(e))
        return entities

    for region in opencv_regions:
        try:
            entities.extend(
                _process_opencv_region(region, image, calibration)
            )
        except Exception as e:
            logger.warning(
                "opencv_region_error",
                region_bounds=region.bounds,
                error=str(e),
            )

    logger.info(
        "selective_opencv_complete",
        regions_processed=len(opencv_regions),
        entities_extracted=len(entities),
    )

    return entities


def _process_opencv_region(
    region: SpecialRegion,
    image: np.ndarray,
    calibration: ScaleCalibration,
) -> List[EntityToCreate]:
    """
    Process a single region with OpenCV for line/circle detection.

    Args:
        region: SpecialRegion defining the area to process
        image: Source image as numpy array (BGR or grayscale)
        calibration: ScaleCalibration for coordinate conversion

    Returns:
        List of EntityToCreate extracted from the region
    """
    import cv2

    entities: List[EntityToCreate] = []
    bounds = region.bounds

    # Validate bounds
    if len(bounds) != 4:
        return entities

    # Cast bounds to int for array slicing
    x1, y1, x2, y2 = int(bounds[0]), int(bounds[1]), int(bounds[2]), int(bounds[3])
    if x2 <= x1 or y2 <= y1:
        return entities

    # Crop region from image (clamp to image dimensions)
    h, w = image.shape[:2]
    x1 = max(0, min(x1, w))
    x2 = max(0, min(x2, w))
    y1 = max(0, min(y1, h))
    y2 = max(0, min(y2, h))

    cropped = image[y1:y2, x1:x2]
    if cropped.size == 0:
        return entities

    # Convert to grayscale
    if len(cropped.shape) == 3:
        gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
    else:
        gray = cropped

    # Apply adaptive threshold
    processed = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        blockSize=11, C=2
    )

    # Detect based on expected pattern
    pattern = region.expected_pattern or "lines"

    if pattern == "parallel_lines" or pattern == "lines":
        detected = _detect_lines_opencv(processed)
    elif pattern == "hatching":
        detected = _detect_lines_opencv(processed)  # Hatching is lines
    elif pattern == "grid":
        detected = _detect_lines_opencv(processed)  # Grid is also lines
    elif pattern == "circles":
        detected = _detect_circles_opencv(gray)
    else:
        detected = []

    # Convert detections to entities
    layer = "0"  # Default layer for OpenCV extractions

    for item in detected:
        if item["type"] == "line":
            # Add region offset
            start = (item["start"][0] + x1, item["start"][1] + y1)
            end = (item["end"][0] + x1, item["end"][1] + y1)

            entities.append(EntityToCreate(
                entity_type=EntityType.LINE,
                layer=layer,
                properties={
                    "start": calibration.to_dwg(*start),
                    "end": calibration.to_dwg(*end),
                    "linetype": "Continuous",
                },
                source=ExtractionSource.SELECTIVE_OPENCV,
                source_element=f"opencv_line_{pattern}",
            ))

        elif item["type"] == "circle":
            center = (item["center"][0] + x1, item["center"][1] + y1)
            radius = item["radius"]

            entities.append(EntityToCreate(
                entity_type=EntityType.CIRCLE,
                layer=layer,
                properties={
                    "center": calibration.to_dwg(*center),
                    "radius": calibration.scale_length(radius),
                },
                source=ExtractionSource.SELECTIVE_OPENCV,
                source_element="opencv_circle",
            ))

    return entities


def _detect_lines_opencv(processed_image: np.ndarray) -> List[Dict[str, Any]]:
    """
    Detect lines using OpenCV HoughLinesP.

    Args:
        processed_image: Binary/thresholded image for line detection

    Returns:
        List of dicts with 'type', 'start', and 'end' keys
    """
    import cv2

    detected: List[Dict[str, Any]] = []

    # Detect lines using probabilistic Hough transform
    lines = cv2.HoughLinesP(
        processed_image,
        rho=1,
        theta=np.pi / 180,
        threshold=50,
        minLineLength=20,
        maxLineGap=10,
    )

    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            detected.append({
                "type": "line",
                "start": (int(x1), int(y1)),
                "end": (int(x2), int(y2)),
            })

    return detected


def _detect_circles_opencv(gray_image: np.ndarray) -> List[Dict[str, Any]]:
    """
    Detect circles using OpenCV HoughCircles.

    Args:
        gray_image: Grayscale image for circle detection

    Returns:
        List of dicts with 'type', 'center', and 'radius' keys
    """
    import cv2

    detected: List[Dict[str, Any]] = []

    # Detect circles using Hough transform
    # param2 controls accumulator threshold - higher = fewer false positives
    circles = cv2.HoughCircles(
        gray_image,
        cv2.HOUGH_GRADIENT,
        dp=1,
        minDist=30,  # Increased from 20 to reduce overlapping detections
        param1=50,
        param2=80,  # Increased from 30 to reduce false positives (was too sensitive)
        minRadius=8,  # Increased from 5 to ignore tiny noise
        maxRadius=100,
    )

    if circles is not None:
        circles = np.uint16(np.around(circles))
        for circle in circles[0, :]:
            x, y, r = circle
            detected.append({
                "type": "circle",
                "center": (int(x), int(y)),
                "radius": int(r),
            })

    return detected


# =============================================================================
# Phase B: VTracer Extraction
# =============================================================================


def _detect_with_vtracer(
    image: np.ndarray,
    calibration: ScaleCalibration,
) -> List[EntityToCreate]:
    """
    Detect lines and curves using VTracer (O(n) vectorizer).

    VTracer produces clean Bézier paths that are converted to AutoCAD entities.
    Falls back to empty list if VTracer is not installed.

    Args:
        image: Binary or grayscale image for vectorization
        calibration: Coordinate calibration for pixel-to-DWG conversion

    Returns:
        List of EntityToCreate objects from VTracer paths
    """
    from aec_agent.config.settings import get_settings

    settings = get_settings()
    if not settings.vtracer_enabled:
        return []

    try:
        from .vtracer_extraction import (
            VTracerExtractor,
            VTracerConfig,
            is_vtracer_available,
            LineSegment,
            BezierSegment,
        )
    except ImportError:
        logger.warning("VTracer module not available")
        return []

    if not is_vtracer_available():
        logger.debug("VTracer library not installed, skipping")
        return []

    entities: List[EntityToCreate] = []

    try:
        config = VTracerConfig.from_settings()
        extractor = VTracerExtractor(image, config)
        result = extractor.extract()

        if not result.paths:
            logger.debug("VTracer found no paths")
            return []

        # Convert VTracer paths to entities
        for path in result.paths:
            path_entities = _vtracer_path_to_entities(path, calibration)
            entities.extend(path_entities)

        logger.info(
            "VTracer extraction complete",
            num_paths=result.num_paths,
            entities_created=len(entities),
        )

    except Exception as e:
        logger.error("VTracer extraction failed", error=str(e), exc_info=True)

    return entities


def _vtracer_path_to_entities(
    path: Any,  # ExtractedPath type
    calibration: ScaleCalibration,
) -> List[EntityToCreate]:
    """
    Convert a VTracer ExtractedPath to AutoCAD entities.

    Line segments become LINE entities, Bézier curves become POLYLINE or SPLINE.

    Args:
        path: VTracer ExtractedPath with segments
        calibration: Coordinate calibration

    Returns:
        List of EntityToCreate objects
    """
    from .vtracer_extraction import LineSegment, BezierSegment

    entities: List[EntityToCreate] = []

    # Group consecutive line segments into polylines
    line_segments: List[Tuple[float, float]] = []

    for segment in path.segments:
        if isinstance(segment, LineSegment):
            # Convert to DWG coordinates
            start_dwg = calibration.pixel_to_dwg(segment.start[0], segment.start[1])
            end_dwg = calibration.pixel_to_dwg(segment.end[0], segment.end[1])

            if not line_segments:
                line_segments.append(start_dwg)
            line_segments.append(end_dwg)

        elif isinstance(segment, BezierSegment):
            # Flush any accumulated line segments first
            if line_segments:
                if len(line_segments) == 2:
                    # Single line
                    entities.append(EntityToCreate(
                        entity_type=EntityType.LINE,
                        layer="0",  # Default layer, will be assigned later
                        properties={
                            "start": line_segments[0],
                            "end": line_segments[1],
                        },
                        source=ExtractionSource.VTRACER,
                    ))
                elif len(line_segments) > 2:
                    # Polyline
                    entities.append(EntityToCreate(
                        entity_type=EntityType.POLYLINE,
                        layer="0",
                        properties={
                            "points": line_segments,
                            "closed": False,
                        },
                        source=ExtractionSource.VTRACER,
                    ))
                line_segments = []

            # Convert Bézier to approximate polyline (for AutoCAD compatibility)
            # Sample the curve at regular intervals
            bezier_points = _sample_bezier(segment, calibration, num_samples=10)
            if len(bezier_points) >= 2:
                entities.append(EntityToCreate(
                    entity_type=EntityType.POLYLINE,
                    layer="0",
                    properties={
                        "points": bezier_points,
                        "closed": False,
                        "fit_type": "spline",  # Hint for fitting
                    },
                    source=ExtractionSource.VTRACER,
                ))

    # Flush remaining line segments
    if line_segments:
        if len(line_segments) == 2:
            entities.append(EntityToCreate(
                entity_type=EntityType.LINE,
                layer="0",
                properties={
                    "start": line_segments[0],
                    "end": line_segments[1],
                },
                source=ExtractionSource.VTRACER,
            ))
        elif len(line_segments) > 2:
            entities.append(EntityToCreate(
                entity_type=EntityType.POLYLINE,
                layer="0",
                properties={
                    "points": line_segments,
                    "closed": path.is_closed,
                },
                source=ExtractionSource.VTRACER,
            ))

    return entities


def _sample_bezier(
    bezier: Any,  # BezierSegment
    calibration: ScaleCalibration,
    num_samples: int = 10,
) -> List[Tuple[float, float]]:
    """
    Sample a cubic Bézier curve at regular intervals.

    Args:
        bezier: BezierSegment with start, control1, control2, end
        calibration: Coordinate calibration
        num_samples: Number of sample points

    Returns:
        List of (x, y) DWG coordinates
    """
    points = []

    for i in range(num_samples + 1):
        t = i / num_samples
        t2 = t * t
        t3 = t2 * t
        mt = 1 - t
        mt2 = mt * mt
        mt3 = mt2 * mt

        # De Casteljau's algorithm
        px = (mt3 * bezier.start[0] +
              3 * mt2 * t * bezier.control1[0] +
              3 * mt * t2 * bezier.control2[0] +
              t3 * bezier.end[0])
        py = (mt3 * bezier.start[1] +
              3 * mt2 * t * bezier.control1[1] +
              3 * mt * t2 * bezier.control2[1] +
              t3 * bezier.end[1])

        dwg_point = calibration.pixel_to_dwg(px, py)
        points.append(dwg_point)

    return points


# =============================================================================
# Coordinator: Extract All
# =============================================================================

async def extract_all(
    analysis: DrawingAnalysis,
    calibration: ScaleCalibration,
    image_path: Optional[Path] = None,
) -> ExtractionResult:
    """
    Coordinate all extraction strategies based on Gemini's recommendations.

    This is the main entry point for Phase 4. It selects the appropriate
    extraction strategy for each part of the drawing and combines the results.

    Args:
        analysis: DrawingAnalysis from Phase 2
        calibration: ScaleCalibration from Phase 3
        image_path: Optional path to source image (needed for OpenCV)

    Returns:
        ExtractionResult with all entities and raster commands

    Example:
        >>> result = await extract_all(analysis, calibration, image_path)
        >>> print(f"Direct: {result.direct_count}, OpenCV: {result.opencv_count}")
    """
    result = ExtractionResult(
        primary_strategy=analysis.extraction_strategy.primary_strategy,
        drawing_type=analysis.drawing_type,
        calibration_method=calibration.method,
        calibration_confidence=calibration.confidence,
    )

    logger.info(
        "extraction_starting",
        primary_strategy=result.primary_strategy,
        drawing_type=result.drawing_type,
        total_elements=analysis.total_elements,
    )

    # Get primary strategy
    primary = analysis.extraction_strategy.primary_strategy.lower()

    # 1. Direct extraction (always done unless strategy is pure guided)
    if primary in ("direct", "direct_extraction", "hybrid", ""):
        direct_entities = await direct_extraction(analysis, calibration)
        result.entities.extend(direct_entities)
        result.direct_count = len(direct_entities)

    # 2. Guided rasterization for complex regions
    if primary in ("guided_rasterization", "guided", "hybrid"):
        if image_path:
            raster_cmds = await guided_rasterization(analysis, calibration, image_path)
            result.raster_commands.extend(raster_cmds)
            result.guided_count = len(raster_cmds)

    # 3. Selective OpenCV for special regions (only if strategy calls for it)
    # Skip OpenCV for "direct" strategy to avoid false positives
    if image_path and primary in ("guided_rasterization", "guided", "hybrid", "selective"):
        opencv_entities = await selective_opencv(analysis, calibration, image_path)
        result.entities.extend(opencv_entities)
        result.opencv_count = len(opencv_entities)

    logger.info(
        "extraction_complete",
        total_entities=result.total_entities,
        total_raster_commands=result.total_commands,
        direct_count=result.direct_count,
        guided_count=result.guided_count,
        opencv_count=result.opencv_count,
    )

    return result


async def extract_direct_only(
    analysis: DrawingAnalysis,
    calibration: ScaleCalibration,
) -> ExtractionResult:
    """
    Extract using direct strategy only (no image processing).

    Use this when you only want Gemini's coordinate-based extraction
    without any OpenCV or Raster Design processing.

    Args:
        analysis: DrawingAnalysis from Phase 2
        calibration: ScaleCalibration from Phase 3

    Returns:
        ExtractionResult with direct entities only
    """
    result = ExtractionResult(
        primary_strategy="direct",
        drawing_type=analysis.drawing_type,
        calibration_method=calibration.method,
        calibration_confidence=calibration.confidence,
    )

    direct_entities = await direct_extraction(analysis, calibration)
    result.entities.extend(direct_entities)
    result.direct_count = len(direct_entities)

    return result


def get_entities_by_type(result: ExtractionResult) -> Dict[str, List[EntityToCreate]]:
    """
    Group entities by their type.

    Args:
        result: ExtractionResult from extraction

    Returns:
        Dictionary mapping entity type to list of entities
    """
    grouped: Dict[str, List[EntityToCreate]] = {}

    for entity in result.entities:
        entity_type = entity.entity_type
        if isinstance(entity_type, EntityType):
            entity_type = entity_type.value

        if entity_type not in grouped:
            grouped[entity_type] = []
        grouped[entity_type].append(entity)

    return grouped


def get_entities_by_layer(result: ExtractionResult) -> Dict[str, List[EntityToCreate]]:
    """
    Group entities by their layer.

    Args:
        result: ExtractionResult from extraction

    Returns:
        Dictionary mapping layer name to list of entities
    """
    grouped: Dict[str, List[EntityToCreate]] = {}

    for entity in result.entities:
        layer = entity.layer

        if layer not in grouped:
            grouped[layer] = []
        grouped[layer].append(entity)

    return grouped


def get_required_layers(result: ExtractionResult) -> List[str]:
    """
    Get list of unique layers needed for all entities.

    Args:
        result: ExtractionResult from extraction

    Returns:
        Sorted list of unique layer names
    """
    layers = set(e.layer for e in result.entities)
    return sorted(layers)


def get_required_blocks(result: ExtractionResult) -> List[str]:
    """
    Get list of unique block names needed for all block entities.

    Args:
        result: ExtractionResult from extraction

    Returns:
        Sorted list of unique block names
    """
    blocks = set()
    for entity in result.entities:
        if entity.entity_type == EntityType.BLOCK or entity.entity_type == "block":
            block_name = entity.properties.get("block_name")
            if block_name:
                blocks.add(block_name)

    return sorted(blocks)


# =============================================================================
# HYBRID EXTRACTION: Gemini + OpenCV + YOLO Fusion
# =============================================================================

@dataclass
class DraftCleanupConfig:
    """
    Configuration for draftsman-like cleanup of extracted entities.

    Controls which cleanup operations to apply and their thresholds.
    Like a draftsman reviewing a tracing, this removes noise and
    incomplete elements.
    """
    # Enable/disable cleanup
    enabled: bool = True

    # Short segment removal (noise)
    remove_short_segments: bool = True
    min_line_length_px: float = 15.0  # Minimum line length in pixels
    min_arc_length_deg: float = 15.0  # Minimum arc span in degrees

    # Text intersection removal
    remove_lines_through_text: bool = True
    text_intersection_threshold: float = 0.3  # Max % of line that can cross text

    # Incomplete curve cleanup
    remove_incomplete_curves: bool = True
    min_arc_completeness: float = 0.15  # Min arc as fraction of full circle (15%)

    # Duplicate removal (stricter than merge)
    remove_near_duplicates: bool = True
    duplicate_distance_threshold: float = 3.0  # Pixels

    # Confidence filtering
    filter_low_confidence: bool = True
    min_confidence_threshold: float = 0.3

    # Isolated point removal
    remove_isolated_circles: bool = False  # Very small circles (dots/noise)
    max_isolated_circle_radius: float = 2.0  # Pixels

    # Edge artifact removal
    remove_edge_artifacts: bool = True
    edge_margin_px: int = 5  # Distance from image edge


@dataclass
class HybridExtractionConfig:
    """Configuration for hybrid extraction pipeline."""
    # Strategy selection
    use_opencv_for_lines: bool = True
    use_opencv_for_circles: bool = True
    use_yolo_for_symbols: bool = True
    use_gemini_for_text: bool = True  # Gemini is best for text/OCR
    use_gemini_for_semantic: bool = True  # Layer assignment, classification

    # OpenCV parameters
    opencv_line_min_length: int = 50  # Increased from 30 to reduce noise
    opencv_circle_min_radius: int = 10  # Increased from 5 to reduce tiny circle noise
    opencv_circle_max_radius: int = 200

    # YOLO parameters
    yolo_confidence_threshold: float = 0.5
    yolo_iou_threshold: float = 0.45
    yolo_model_path: Optional[str] = None

    # Fusion parameters
    coordinate_tolerance: float = 5.0  # Pixels - for merging duplicates
    prefer_opencv_geometry: bool = True  # When conflict, prefer OpenCV coords

    # OCR Text Anchoring (NEW)
    use_ocr_for_text_positions: bool = True  # Anchor text to OCR-detected positions
    ocr_min_confidence: float = 60.0  # Min OCR confidence threshold
    ocr_min_similarity: float = 0.5  # Min text similarity for matching

    # Gemini Refinement (NEW)
    enable_refinement: bool = True  # Enable Gemini refinement pass
    refine_snap_to_grid: bool = True
    refine_connect_endpoints: bool = True
    refine_align_parallel: bool = True
    refine_remove_duplicates: bool = True

    # Validation
    enable_gemini_validation: bool = True
    max_validation_iterations: int = 2

    # Entity limits (prevent runaway extraction)
    max_entities_per_source: int = 5000  # Max entities from OpenCV/YOLO each
    max_total_entities: int = 10000  # Max total entities after merge

    # Masking parameters for OpenCV (to exclude text/symbols from line detection)
    # Increased defaults for better text masking
    text_mask_padding: int = 8  # Extra padding around text bounding box
    text_mask_char_width: int = 8  # Estimated width per character in pixels
    text_mask_min_width: int = 20  # Minimum mask width
    text_mask_min_height: int = 16  # Minimum mask height
    symbol_mask_half_size: int = 30  # Half-size of symbol masking square
    symbol_mask_padding: int = 5  # Extra padding around symbols

    # Default DPI when calibration doesn't provide one
    default_dpi: int = 300

    # Draftsman-like cleanup
    draft_cleanup: DraftCleanupConfig = field(default_factory=DraftCleanupConfig)

    # Phase A: Advanced Preprocessing (from VECTORIZATION_IMPROVEMENT_ROADMAP)
    enable_preprocessing: bool = True
    enable_deskew: bool = True
    enable_ensemble_binarization: bool = True
    enable_denoise: bool = False  # Can blur fine lines, disabled by default
    enable_contrast_enhancement: bool = True

    # Phase A: Line Simplification
    enable_simplification: bool = True
    simplification_epsilon: float = 1.5  # RDP tolerance in pixels (lower = more detail)
    simplify_polylines_only: bool = True  # Only simplify polylines, not individual lines

    # Phase B: VTracer Vectorization (O(n) alternative to OpenCV)
    use_vtracer: bool = False  # Enable VTracer for line/curve extraction
    vtracer_fallback_to_opencv: bool = True  # Fall back to OpenCV if VTracer unavailable

    # Phase B: Text/Graphics Separation (Fletcher-Kasturi)
    enable_text_graphics_separation: bool = False  # Enable Fletcher-Kasturi separation


def _get_dpi(calibration: ScaleCalibration, config: Optional["HybridExtractionConfig"] = None) -> int:
    """
    Get DPI from calibration or fall back to config/default.

    Args:
        calibration: ScaleCalibration object
        config: Optional HybridExtractionConfig for default DPI

    Returns:
        DPI value as integer
    """
    if hasattr(calibration, 'dpi') and calibration.dpi:
        return int(calibration.dpi)
    if config is not None:
        return config.default_dpi
    return 300


@dataclass
class HybridExtractionResult(ExtractionResult):
    """Extended result with hybrid extraction statistics."""
    # Source breakdown
    gemini_entities: int = 0
    opencv_entities: int = 0
    yolo_entities: int = 0

    # Fusion statistics
    duplicates_merged: int = 0
    conflicts_resolved: int = 0

    # OCR Text Anchoring statistics (NEW)
    ocr_text_anchored: int = 0
    ocr_text_fallback: int = 0
    ocr_avg_offset: float = 0.0  # Average pixel offset corrected

    # Refinement statistics (NEW)
    refinement_applied: bool = False
    refinement_adjustments: int = 0
    endpoints_connected: int = 0
    lines_snapped: int = 0
    lines_aligned: int = 0

    # Draftsman cleanup statistics (NEW)
    cleanup_applied: bool = False
    cleanup_stats: Optional[CleanupStatistics] = None

    # Validation results
    validation_passed: bool = False
    validation_accuracy: float = 0.0
    corrections_applied: int = 0

    def to_dict(self) -> dict:
        base = super().to_dict()
        base["hybrid_statistics"] = {
            "gemini_entities": self.gemini_entities,
            "opencv_entities": self.opencv_entities,
            "yolo_entities": self.yolo_entities,
            "duplicates_merged": self.duplicates_merged,
            "conflicts_resolved": self.conflicts_resolved,
            "ocr_text_anchored": self.ocr_text_anchored,
            "ocr_text_fallback": self.ocr_text_fallback,
            "ocr_avg_offset": self.ocr_avg_offset,
            "refinement_applied": self.refinement_applied,
            "refinement_adjustments": self.refinement_adjustments,
            "endpoints_connected": self.endpoints_connected,
            "lines_snapped": self.lines_snapped,
            "lines_aligned": self.lines_aligned,
            "cleanup_applied": self.cleanup_applied,
            "validation_passed": self.validation_passed,
            "validation_accuracy": self.validation_accuracy,
            "corrections_applied": self.corrections_applied,
        }
        # Add cleanup details if available
        if self.cleanup_stats:
            base["cleanup_statistics"] = {
                "short_segments_removed": self.cleanup_stats.short_segments_removed,
                "text_intersections_removed": self.cleanup_stats.text_intersections_removed,
                "incomplete_curves_removed": self.cleanup_stats.incomplete_curves_removed,
                "duplicates_removed": self.cleanup_stats.duplicates_removed,
                "low_confidence_removed": self.cleanup_stats.low_confidence_removed,
                "edge_artifacts_removed": self.cleanup_stats.edge_artifacts_removed,
                "total_removed": self.cleanup_stats.total_removed,
                "total_kept": self.cleanup_stats.total_kept,
            }
        return base


def _load_image(image_path: Path) -> Optional[np.ndarray]:
    """Load image using OpenCV."""
    try:
        import cv2
        image = cv2.imread(str(image_path))
        return image
    except Exception as e:
        logger.warning("failed_to_load_image", path=str(image_path), error=str(e))
        return None


def _parse_patterns(pattern_string: Optional[str]) -> set:
    """
    Parse a pattern string into a set of individual patterns.

    Supports both comma-separated ("lines,circles") and single patterns ("lines").

    Args:
        pattern_string: Pattern string like "lines", "circles", or "lines,circles"

    Returns:
        Set of lowercase pattern strings
    """
    if not pattern_string:
        return {"lines"}  # Default pattern
    return {p.strip().lower() for p in pattern_string.split(",")}


def _pattern_matches(patterns: set, *keywords: str) -> bool:
    """
    Check if any pattern matches any of the given keywords.

    Args:
        patterns: Set of patterns from _parse_patterns()
        *keywords: Keywords to check for (e.g., "lines", "walls")

    Returns:
        True if any pattern matches any keyword
    """
    return bool(patterns & set(keywords))


# AutoCAD linetype mapping
LINETYPE_MAP = {
    "continuous": "Continuous",
    "dashed": "DASHED",
    "dotted": "DOT",
    "center": "CENTER",
    "hidden": "HIDDEN",
    "phantom": "PHANTOM",
    "dashdot": "DASHDOT",
    "border": "BORDER",
    "divide": "DIVIDE",
    "unknown": "Continuous",
}


def _map_line_type_to_autocad(line_type: str) -> str:
    """
    Map detected line type to AutoCAD linetype name.

    Args:
        line_type: Detected line type (continuous, dashed, dotted, center, etc.)

    Returns:
        AutoCAD linetype name
    """
    return LINETYPE_MAP.get(line_type.lower(), "Continuous")


async def hybrid_opencv_extraction(
    image: np.ndarray,
    analysis: DrawingAnalysis,
    calibration: ScaleCalibration,
    config: HybridExtractionConfig,
) -> List[EntityToCreate]:
    """
    Extract entities using OpenCV with Gemini guidance.

    Gemini tells us WHAT regions to process and WHAT type of elements
    to look for. OpenCV extracts with pixel-perfect accuracy.

    Args:
        image: Source image (BGR numpy array)
        analysis: Gemini's drawing analysis
        calibration: Coordinate calibration
        config: Hybrid extraction configuration

    Returns:
        List of EntityToCreate from OpenCV
    """
    entities: List[EntityToCreate] = []

    try:
        from .opencv_extraction import OpenCVExtractor, ExtractedLine, ExtractedCircle
    except ImportError:
        logger.warning("opencv_extraction_module_not_available")
        return entities

    # =================================================================
    # IMPROVED: Mask out text and symbols to prevent noise
    # Calculate proper bounding boxes from text content and height
    # =================================================================
    import cv2
    processed_image = image.copy()
    height, width = processed_image.shape[:2]

    text_regions_masked = 0
    symbol_regions_masked = 0

    # Mask text regions (white rectangle over text)
    # This prevents OpenCV from detecting text characters as lines
    if hasattr(analysis, 'elements') and hasattr(analysis.elements, 'text'):
        for text in analysis.elements.text:
            if not hasattr(text, 'position'):
                continue

            cx, cy = text.position

            # Calculate bounding box from text content and height
            text_height = getattr(text, 'height_px', 12)
            text_content = getattr(text, 'content', '')

            # Estimate width based on character count and height
            # Average char width is roughly 0.6 * height for most fonts
            char_width = max(config.text_mask_char_width, int(text_height * 0.6))
            estimated_width = max(
                config.text_mask_min_width,
                len(text_content) * char_width
            )

            # Use text height with minimum
            estimated_height = max(config.text_mask_min_height, int(text_height * 1.5))

            # Add padding
            padding = config.text_mask_padding
            half_w = (estimated_width // 2) + padding
            half_h = (estimated_height // 2) + padding

            # Calculate bounds (position is typically at baseline-left or center)
            # Assume position is at center-left of text
            x1 = int(cx - padding)
            y1 = int(cy - half_h)
            x2 = int(cx + estimated_width + padding)
            y2 = int(cy + half_h)

            # Clamp to image bounds
            x1 = max(0, min(x1, width - 1))
            y1 = max(0, min(y1, height - 1))
            x2 = max(0, min(x2, width))
            y2 = max(0, min(y2, height))

            if x2 > x1 and y2 > y1:
                cv2.rectangle(processed_image, (x1, y1), (x2, y2), (255, 255, 255), -1)
                text_regions_masked += 1

    # Mask symbol regions (white rectangle over symbols)
    if hasattr(analysis, 'elements') and hasattr(analysis.elements, 'symbols'):
        for symbol in analysis.elements.symbols:
            if not hasattr(symbol, 'position'):
                continue

            cx, cy = symbol.position
            half_size = config.symbol_mask_half_size + config.symbol_mask_padding

            x1 = max(0, int(cx - half_size))
            y1 = max(0, int(cy - half_size))
            x2 = min(width, int(cx + half_size))
            y2 = min(height, int(cy + half_size))

            if x2 > x1 and y2 > y1:
                cv2.rectangle(processed_image, (x1, y1), (x2, y2), (255, 255, 255), -1)
                symbol_regions_masked += 1

    logger.debug(
        "opencv_masking_applied",
        text_regions_masked=text_regions_masked,
        symbol_regions_masked=symbol_regions_masked,
    )

    height, width = processed_image.shape[:2]
    extractor = OpenCVExtractor(processed_image, dpi=_get_dpi(calibration, config))

    # Process regions marked for OpenCV
    opencv_regions = [
        r for r in analysis.extraction_strategy.special_regions
        if r.strategy in ("selective_opencv", "hybrid")
    ]

    # If no specific regions, process entire image for lines AND circles
    # CHANGED: Added "circles" to expected_pattern to enable circle detection by default
    if not opencv_regions and (config.use_opencv_for_lines or config.use_opencv_for_circles):
        opencv_regions = [SpecialRegion(
            bounds=[0, 0, width, height],
            reason="full_image",
            strategy="selective_opencv",
            expected_pattern="lines,circles",  # CHANGED: Enable circles
        )]

    for region in opencv_regions:
        bounds = region.bounds
        if len(bounds) != 4:
            continue

        x1, y1, x2, y2 = bounds
        roi = (int(x1), int(y1), int(x2), int(y2))

        # Parse expected patterns (supports comma-separated: "lines,circles")
        patterns = _parse_patterns(region.expected_pattern)
        is_full_image = (region.reason or "").lower() == "full_image"

        # Check for lines - extract if pattern includes lines/walls or it's a full image scan
        should_extract_lines = (
            _pattern_matches(patterns, "lines", "walls") or is_full_image
        ) and config.use_opencv_for_lines

        if should_extract_lines:
            lines = extractor.extract_lines_lsd(
                roi=roi,
                min_length=config.opencv_line_min_length,
            )

            # Detect line types for each line
            for line in lines:
                line.line_type = extractor.detect_line_type(line)

            for line in lines:
                # Convert to DWG coordinates
                start_dwg = calibration.to_dwg(*line.start)
                end_dwg = calibration.to_dwg(*line.end)

                # Determine layer based on Gemini's analysis
                layer = _infer_layer_for_region(region, "line", analysis)

                # Map OpenCV line type to AutoCAD linetype
                line_type_str = (
                    line.line_type.value
                    if hasattr(line.line_type, 'value')
                    else str(line.line_type)
                )
                linetype = _map_line_type_to_autocad(line_type_str)

                # Scale thickness from pixels to DWG units
                thickness_dwg = calibration.scale_length(line.thickness) if line.thickness > 1.0 else 0.0

                entities.append(EntityToCreate(
                    entity_type=EntityType.LINE,
                    layer=layer,
                    properties={
                        "start": start_dwg,
                        "end": end_dwg,
                        "linetype": linetype,
                        "thickness": thickness_dwg,  # Lineweight from OpenCV
                    },
                    source=ExtractionSource.SELECTIVE_OPENCV,
                    confidence=line.confidence,
                    source_element=f"opencv_line_{region.expected_pattern or 'default'}_{linetype.lower()}",
                ))

        # Check for circles - extract if pattern includes circles/columns/equipment
        should_extract_circles = (
            _pattern_matches(patterns, "circles", "columns", "equipment")
        ) and config.use_opencv_for_circles

        if should_extract_circles:
            circles = extractor.extract_circles(
                roi=roi,
                min_radius=config.opencv_circle_min_radius,
                max_radius=config.opencv_circle_max_radius,
            )

            for circle in circles:
                center_dwg = calibration.to_dwg(*circle.center)
                radius_dwg = calibration.scale_length(circle.radius)

                layer = _infer_layer_for_region(region, "circle", analysis)

                entities.append(EntityToCreate(
                    entity_type=EntityType.CIRCLE,
                    layer=layer,
                    properties={
                        "center": center_dwg,
                        "radius": radius_dwg,
                    },
                    source=ExtractionSource.SELECTIVE_OPENCV,
                    confidence=circle.confidence,
                    source_element="opencv_circle",
                ))

        # Check for arcs (fillets, half-circles) - extract alongside circles
        should_extract_arcs = (
            _pattern_matches(patterns, "circles", "arcs", "fillets", "curves")
        ) and config.use_opencv_for_circles  # Reuse circle flag for arcs

        if should_extract_arcs:
            arcs = extractor.extract_arcs(
                roi=roi,
                min_radius=config.opencv_circle_min_radius,
                max_radius=config.opencv_circle_max_radius,
            )

            for arc in arcs:
                center_dwg = calibration.to_dwg(*arc.center)
                radius_dwg = calibration.scale_length(arc.radius)

                layer = _infer_layer_for_region(region, "arc", analysis)

                entities.append(EntityToCreate(
                    entity_type=EntityType.ARC,
                    layer=layer,
                    properties={
                        "center": center_dwg,
                        "radius": radius_dwg,
                        "start_angle": arc.start_angle,
                        "end_angle": arc.end_angle,
                    },
                    source=ExtractionSource.SELECTIVE_OPENCV,
                    confidence=arc.confidence,
                    source_element="opencv_arc",
                ))

    # Count line types for logging
    linetype_counts = {}
    for e in entities:
        if e.entity_type in (EntityType.LINE, "line"):
            lt = e.properties.get("linetype", "Continuous")
            linetype_counts[lt] = linetype_counts.get(lt, 0) + 1

    logger.info(
        "hybrid_opencv_extraction_complete",
        regions_processed=len(opencv_regions),
        entities_extracted=len(entities),
        linetypes=linetype_counts,
    )

    return entities


async def hybrid_yolo_extraction(
    image: np.ndarray,
    analysis: DrawingAnalysis,
    calibration: ScaleCalibration,
    config: HybridExtractionConfig,
) -> List[EntityToCreate]:
    """
    Extract symbols using YOLO with Gemini semantic enhancement.

    YOLO detects symbol bounding boxes with trained accuracy.
    Gemini provides context for layer assignment and attribute inference.

    Args:
        image: Source image
        analysis: Gemini's drawing analysis
        calibration: Coordinate calibration
        config: Hybrid extraction configuration

    Returns:
        List of EntityToCreate (block references)
    """
    entities: List[EntityToCreate] = []

    if not config.use_yolo_for_symbols:
        return entities

    try:
        from ..yolo_detection import detect_symbols_yolo, DetectedBlock
    except ImportError:
        logger.warning("yolo_detection_not_available")
        return entities

    # Convert to grayscale for YOLO if needed
    if len(image.shape) == 3:
        import cv2
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image

    # Calculate scale factor using DPI helper
    scale = 1.0 / _get_dpi(calibration, config)

    # Run YOLO detection
    _, detected_blocks = detect_symbols_yolo(
        image=gray,
        scale=scale,
        confidence=config.yolo_confidence_threshold,
        iou_threshold=config.yolo_iou_threshold,
        model_path=config.yolo_model_path,
        mask_detections=False,
    )

    # Convert YOLO detections to entities
    for block in detected_blocks:
        # Get block name - YOLO provides this
        block_name = block.block_name

        # Get layer based on category
        layer = get_layer_for_symbol(block.category)

        # Cross-reference with Gemini's symbol detection for attributes
        attributes = _enrich_symbol_attributes(block, analysis)

        entities.append(EntityToCreate(
            entity_type=EntityType.BLOCK,
            layer=layer,
            properties={
                "block_name": block_name,
                "position": block.position,  # Already in DWG coords from YOLO
                "rotation": block.rotation,
                "scale": block.scale,
                "attributes": attributes,
            },
            source=ExtractionSource.HYBRID,
            confidence=block.confidence,
            source_element=f"yolo_{block.category}_{block.block_name}",
        ))

    logger.info(
        "hybrid_yolo_extraction_complete",
        symbols_detected=len(detected_blocks),
        entities_created=len(entities),
    )

    return entities


def _infer_layer_for_region(
    region: SpecialRegion,
    entity_type: str,
    analysis: DrawingAnalysis,
) -> str:
    """Infer the appropriate layer for entities in a region based on Gemini analysis."""
    # Use Gemini's layer suggestion if available
    if hasattr(region, 'layer_suggestion') and region.layer_suggestion:
        return region.layer_suggestion

    # Infer from region context
    reason = (region.reason or "").lower()
    pattern = (region.expected_pattern or "").lower()

    # Map common patterns to layers
    if "wall" in reason or "wall" in pattern:
        return "A-WALL"
    elif "duct" in reason or "duct" in pattern:
        return "M-DUCT"
    elif "pipe" in reason or "pipe" in pattern:
        return "P-PIPE"
    elif "wire" in reason or "electrical" in pattern:
        return "E-POWR"
    elif "column" in reason:
        return "A-COLS"

    # Fall back to drawing type inference
    drawing_type = analysis.drawing_type.lower() if analysis.drawing_type else ""

    if drawing_type == "mechanical":
        return "M-DUCT" if entity_type == "line" else "M-EQPM"
    elif drawing_type == "electrical":
        return "E-POWR" if entity_type == "line" else "E-POWR-OUTL"
    elif drawing_type == "plumbing":
        return "P-PIPE" if entity_type == "line" else "P-FIXT"
    elif drawing_type == "fire_alarm":
        return "F-ALRM" if entity_type == "line" else "F-ALRM-DETC"

    return "0"


def _enrich_symbol_attributes(
    block: "DetectedBlock",
    analysis: DrawingAnalysis,
) -> Dict[str, str]:
    """
    Enrich YOLO-detected symbol with attributes from Gemini analysis.

    Cross-references YOLO detection position with Gemini's symbol list
    to get semantic attributes like tags, sizes, etc.
    """
    attributes: Dict[str, str] = {}

    # Try to find matching symbol in Gemini's analysis
    pos_x, pos_y = block.position
    tolerance = 50  # Pixels

    for symbol in analysis.elements.symbols:
        sym_x, sym_y = symbol.position
        distance = np.sqrt((sym_x - pos_x) ** 2 + (sym_y - pos_y) ** 2)

        if distance < tolerance:
            # Found a match - use Gemini's attributes
            if symbol.tag:
                attributes["TAG"] = symbol.tag
            if symbol.size:
                attributes["SIZE"] = symbol.size
            for i, text in enumerate(symbol.associated_text):
                attributes[f"TEXT{i + 1}"] = text
            break

    return attributes


def _merge_duplicate_entities(
    entities: List[EntityToCreate],
    tolerance: float = 5.0,
    prefer_opencv: bool = True,
) -> Tuple[List[EntityToCreate], int]:
    """
    Merge duplicate entities from different sources.

    When Gemini and OpenCV both detect the same line, keep the more
    accurate one (typically OpenCV).

    Args:
        entities: List of entities from all sources
        tolerance: Distance tolerance for considering duplicates (pixels)
        prefer_opencv: When duplicates found, prefer OpenCV coordinates

    Returns:
        Tuple of (merged entities, number of duplicates removed)
    """
    if not entities:
        return entities, 0

    merged: List[EntityToCreate] = []
    removed = 0

    # Group by entity type
    by_type: Dict[str, List[EntityToCreate]] = {}
    for e in entities:
        etype = e.entity_type.value if isinstance(e.entity_type, EntityType) else e.entity_type
        if etype not in by_type:
            by_type[etype] = []
        by_type[etype].append(e)

    for etype, group in by_type.items():
        if etype == "line":
            merged_group, count = _merge_lines(group, tolerance, prefer_opencv)
            merged.extend(merged_group)
            removed += count
        elif etype == "circle":
            merged_group, count = _merge_circles(group, tolerance, prefer_opencv)
            merged.extend(merged_group)
            removed += count
        else:
            # For other types, keep all
            merged.extend(group)

    return merged, removed


def _merge_lines(
    lines: List[EntityToCreate],
    tolerance: float,
    prefer_opencv: bool,
) -> Tuple[List[EntityToCreate], int]:
    """
    Merge duplicate lines from different extraction sources.

    When both Gemini and OpenCV detect the same line, this function keeps
    one copy with the best attributes from both:
    - OpenCV provides pixel-accurate geometry
    - Gemini provides semantic layer assignment and linetype

    Args:
        lines: List of line entities from various sources
        tolerance: Maximum distance (in DWG units) to consider lines as duplicates
        prefer_opencv: If True, use OpenCV coordinates when duplicates found

    Returns:
        Tuple of (merged line list, count of duplicates removed)
    """
    if len(lines) <= 1:
        return lines, 0

    kept: List[EntityToCreate] = []
    removed = 0

    # Sort lines to process Gemini (direct) lines first, then OpenCV lines
    # This ensures we have metadata in 'kept' before processing geometric matches
    # 'direct' comes before 'selective_opencv' alphabetically, but let's be explicit
    # We want: [Gemini lines, OpenCV lines]
    # So when we process OpenCV line, we find its Gemini match in 'kept'
    sorted_lines = sorted(
        lines, 
        key=lambda x: 0 if x.source == ExtractionSource.DIRECT else 1
    )

    for line in sorted_lines:
        is_duplicate = False
        start = np.array(line.properties.get("start", (0, 0)))
        end = np.array(line.properties.get("end", (0, 0)))

        for i, existing in enumerate(kept):
            ex_start = np.array(existing.properties.get("start", (0, 0)))
            ex_end = np.array(existing.properties.get("end", (0, 0)))

            # Check if endpoints match (either direction)
            dist1 = np.linalg.norm(start - ex_start)
            dist2 = np.linalg.norm(end - ex_end)
            dist3 = np.linalg.norm(start - ex_end)
            dist4 = np.linalg.norm(end - ex_start)

            if (dist1 < tolerance and dist2 < tolerance) or (dist3 < tolerance and dist4 < tolerance):
                is_duplicate = True

                # If we prefer OpenCV and the current line IS OpenCV (and existing is likely Gemini)
                if prefer_opencv and line.source == ExtractionSource.SELECTIVE_OPENCV:
                    # Create a copy to avoid mutating the original input entity
                    merged_line = deepcopy(line)

                    # Capture metadata from the existing Gemini line
                    semantic_layer = existing.layer
                    # Only transfer if semantic layer is better than "0"
                    if semantic_layer and semantic_layer != "0":
                        merged_line.layer = semantic_layer

                    # Capture linetype if Gemini has a specific one and OpenCV is just Continuous (or default)
                    semantic_linetype = existing.properties.get("linetype")
                    geometric_linetype = merged_line.properties.get("linetype")

                    if semantic_linetype and semantic_linetype != "Continuous" and (not geometric_linetype or geometric_linetype == "Continuous"):
                        merged_line.properties["linetype"] = semantic_linetype

                    # Replace existing with this new, improved OpenCV line
                    kept[i] = merged_line

                # If we don't prefer OpenCV, or if current line is Gemini and we already have one
                # We just drop the current line (do nothing), effectively keeping 'existing'
                break

        if not is_duplicate:
            kept.append(line)
        else:
            removed += 1

    return kept, removed


def _merge_circles(
    circles: List[EntityToCreate],
    tolerance: float,
    prefer_opencv: bool,
) -> Tuple[List[EntityToCreate], int]:
    """
    Merge duplicate circles from different extraction sources.

    When both Gemini and OpenCV detect the same circle, this function keeps
    one copy with the best attributes from both:
    - OpenCV provides pixel-accurate center and radius
    - Gemini provides semantic layer assignment

    Args:
        circles: List of circle entities from various sources
        tolerance: Maximum distance (in DWG units) for center/radius to consider duplicates
        prefer_opencv: If True, use OpenCV coordinates when duplicates found

    Returns:
        Tuple of (merged circle list, count of duplicates removed)
    """
    if len(circles) <= 1:
        return circles, 0

    kept: List[EntityToCreate] = []
    removed = 0
    
    # Sort: Gemini first, then OpenCV
    sorted_circles = sorted(
        circles, 
        key=lambda x: 0 if x.source == ExtractionSource.DIRECT else 1
    )

    for circle in sorted_circles:
        is_duplicate = False
        center = np.array(circle.properties.get("center", (0, 0)))
        radius = circle.properties.get("radius", 0)

        for i, existing in enumerate(kept):
            ex_center = np.array(existing.properties.get("center", (0, 0)))
            ex_radius = existing.properties.get("radius", 0)

            center_dist = np.linalg.norm(center - ex_center)
            radius_diff = abs(radius - ex_radius)

            if center_dist < tolerance and radius_diff < tolerance:
                is_duplicate = True

                if prefer_opencv and circle.source == ExtractionSource.SELECTIVE_OPENCV:
                    # Create a copy to avoid mutating the original input entity
                    merged_circle = deepcopy(circle)

                    # Transfer Layer from Gemini's semantic analysis
                    semantic_layer = existing.layer
                    if semantic_layer and semantic_layer != "0":
                        merged_circle.layer = semantic_layer

                    # Replace existing with merged circle
                    kept[i] = merged_circle

                break

        if not is_duplicate:
            kept.append(circle)
        else:
            removed += 1

    return kept, removed


# =============================================================================
# DRAFTSMAN-LIKE CLEANUP FUNCTIONS
# =============================================================================

@dataclass
class CleanupStatistics:
    """Statistics from draftsman cleanup operations."""
    short_segments_removed: int = 0
    text_intersections_removed: int = 0
    incomplete_curves_removed: int = 0
    duplicates_removed: int = 0
    low_confidence_removed: int = 0
    edge_artifacts_removed: int = 0
    isolated_circles_removed: int = 0
    total_removed: int = 0
    total_kept: int = 0


def _calculate_text_bounding_boxes(
    analysis: "DrawingAnalysis",
    config: "HybridExtractionConfig",
) -> List[Tuple[int, int, int, int]]:
    """
    Calculate bounding boxes for all text elements.

    Args:
        analysis: Drawing analysis containing text elements
        config: Configuration with text sizing parameters

    Returns:
        List of (x1, y1, x2, y2) bounding boxes in pixels
    """
    boxes: List[Tuple[int, int, int, int]] = []

    if not hasattr(analysis, 'elements') or not hasattr(analysis.elements, 'text'):
        return boxes

    for text in analysis.elements.text:
        if not hasattr(text, 'position'):
            continue

        cx, cy = text.position
        text_height = getattr(text, 'height_px', 12)
        text_content = getattr(text, 'content', '')

        # Estimate dimensions
        char_width = max(config.text_mask_char_width, int(text_height * 0.6))
        estimated_width = max(
            config.text_mask_min_width,
            len(text_content) * char_width
        )
        estimated_height = max(config.text_mask_min_height, int(text_height * 1.5))

        padding = config.text_mask_padding
        x1 = int(cx - padding)
        y1 = int(cy - estimated_height // 2 - padding)
        x2 = int(cx + estimated_width + padding)
        y2 = int(cy + estimated_height // 2 + padding)

        boxes.append((x1, y1, x2, y2))

    return boxes


def _line_intersects_box(
    start: Tuple[float, float],
    end: Tuple[float, float],
    box: Tuple[int, int, int, int],
) -> float:
    """
    Calculate what fraction of a line segment passes through a bounding box.

    Uses Liang-Barsky algorithm for line-box intersection.

    Args:
        start: Line start point (x, y)
        end: Line end point (x, y)
        box: Bounding box (x1, y1, x2, y2)

    Returns:
        Fraction of line inside the box (0.0 to 1.0)
    """
    x1, y1, x2, y2 = box
    sx, sy = start
    ex, ey = end

    dx = ex - sx
    dy = ey - sy

    # Check if line is completely outside box (quick reject)
    if max(sx, ex) < x1 or min(sx, ex) > x2:
        return 0.0
    if max(sy, ey) < y1 or min(sy, ey) > y2:
        return 0.0

    # Liang-Barsky parameters
    p = [-dx, dx, -dy, dy]
    q = [sx - x1, x2 - sx, sy - y1, y2 - sy]

    t0, t1 = 0.0, 1.0

    for i in range(4):
        if p[i] == 0:
            if q[i] < 0:
                return 0.0  # Line parallel and outside
        else:
            t = q[i] / p[i]
            if p[i] < 0:
                t0 = max(t0, t)
            else:
                t1 = min(t1, t)

    if t0 > t1:
        return 0.0

    # Return fraction of line inside box
    return t1 - t0


def _remove_short_segments(
    entities: List[EntityToCreate],
    min_line_length: float,
    min_arc_span: float,
    calibration: "ScaleCalibration",
) -> Tuple[List[EntityToCreate], int]:
    """
    Remove line segments and arcs that are too short (likely noise).

    Args:
        entities: List of entities to filter
        min_line_length: Minimum line length in pixels
        min_arc_span: Minimum arc span in degrees
        calibration: For converting DWG units back to pixels

    Returns:
        Tuple of (filtered entities, count removed)
    """
    kept: List[EntityToCreate] = []
    removed = 0

    for entity in entities:
        etype = entity.entity_type
        if isinstance(etype, EntityType):
            etype = etype.value

        if etype == "line":
            props = entity.properties
            start = np.array(props.get("start", (0, 0)))
            end = np.array(props.get("end", (0, 0)))
            length_dwg = float(np.linalg.norm(end - start))

            # Convert back to pixels for comparison
            # Use inverse of scale_length
            if hasattr(calibration, 'pixels_per_unit') and calibration.pixels_per_unit:
                length_px = length_dwg * calibration.pixels_per_unit
            else:
                length_px = length_dwg * 10  # Rough estimate

            if length_px >= min_line_length:
                kept.append(entity)
            else:
                removed += 1

        elif etype == "arc":
            props = entity.properties
            start_angle = props.get("start_angle", 0)
            end_angle = props.get("end_angle", 0)

            # Calculate arc span
            span = abs(end_angle - start_angle)
            if span > 180:
                span = 360 - span

            if span >= min_arc_span:
                kept.append(entity)
            else:
                removed += 1

        else:
            # Keep all other entity types
            kept.append(entity)

    return kept, removed


def _remove_lines_through_text(
    entities: List[EntityToCreate],
    text_boxes: List[Tuple[int, int, int, int]],
    threshold: float,
    calibration: "ScaleCalibration",
) -> Tuple[List[EntityToCreate], int]:
    """
    Remove lines that pass through text regions.

    A draftsman wouldn't draw lines through text - these are likely
    detection artifacts from text characters.

    Args:
        entities: List of entities to filter
        text_boxes: List of text bounding boxes in pixels
        threshold: Maximum fraction of line that can intersect text
        calibration: For coordinate conversion

    Returns:
        Tuple of (filtered entities, count removed)
    """
    if not text_boxes:
        return entities, 0

    kept: List[EntityToCreate] = []
    removed = 0

    for entity in entities:
        etype = entity.entity_type
        if isinstance(etype, EntityType):
            etype = etype.value

        if etype == "line":
            props = entity.properties
            start_dwg = props.get("start", (0, 0))
            end_dwg = props.get("end", (0, 0))

            # Convert DWG coordinates back to pixels
            # This is approximate - we use inverse transform
            if hasattr(calibration, 'to_pixels'):
                start_px = calibration.to_pixels(*start_dwg)
                end_px = calibration.to_pixels(*end_dwg)
            else:
                # Fallback: assume 1:1 if no inverse available
                start_px = start_dwg
                end_px = end_dwg

            # Check intersection with each text box
            max_intersection = 0.0
            for box in text_boxes:
                intersection = _line_intersects_box(start_px, end_px, box)
                max_intersection = max(max_intersection, intersection)

            if max_intersection <= threshold:
                kept.append(entity)
            else:
                removed += 1
        else:
            kept.append(entity)

    return kept, removed


def _remove_incomplete_curves(
    entities: List[EntityToCreate],
    min_completeness: float,
) -> Tuple[List[EntityToCreate], int]:
    """
    Remove arcs that are too small a fraction of a complete circle.

    Very small arcs are often noise from corner detection or text.

    Args:
        entities: List of entities to filter
        min_completeness: Minimum arc as fraction of 360 degrees

    Returns:
        Tuple of (filtered entities, count removed)
    """
    kept: List[EntityToCreate] = []
    removed = 0

    min_span = min_completeness * 360.0  # Convert to degrees

    for entity in entities:
        etype = entity.entity_type
        if isinstance(etype, EntityType):
            etype = etype.value

        if etype == "arc":
            props = entity.properties
            start_angle = props.get("start_angle", 0)
            end_angle = props.get("end_angle", 0)

            span = abs(end_angle - start_angle)
            if span > 180:
                span = 360 - span

            if span >= min_span:
                kept.append(entity)
            else:
                removed += 1
        else:
            kept.append(entity)

    return kept, removed


def _remove_low_confidence(
    entities: List[EntityToCreate],
    min_confidence: float,
) -> Tuple[List[EntityToCreate], int]:
    """
    Remove entities with confidence below threshold.

    Args:
        entities: List of entities to filter
        min_confidence: Minimum confidence threshold (0.0 to 1.0)

    Returns:
        Tuple of (filtered entities, count removed)
    """
    kept: List[EntityToCreate] = []
    removed = 0

    for entity in entities:
        if entity.confidence >= min_confidence:
            kept.append(entity)
        else:
            removed += 1

    return kept, removed


def _remove_edge_artifacts(
    entities: List[EntityToCreate],
    image_size: Tuple[int, int],
    margin: int,
    calibration: "ScaleCalibration",
) -> Tuple[List[EntityToCreate], int]:
    """
    Remove entities that are at the edge of the image (likely artifacts).

    Args:
        entities: List of entities to filter
        image_size: (width, height) of the source image
        margin: Pixel distance from edge to consider as artifact
        calibration: For coordinate conversion

    Returns:
        Tuple of (filtered entities, count removed)
    """
    width, height = image_size
    kept: List[EntityToCreate] = []
    removed = 0

    for entity in entities:
        etype = entity.entity_type
        if isinstance(etype, EntityType):
            etype = etype.value

        props = entity.properties
        is_edge_artifact = False

        if etype == "line":
            start = props.get("start", (0, 0))
            end = props.get("end", (0, 0))

            # Convert to pixels
            if hasattr(calibration, 'to_pixels'):
                start_px = calibration.to_pixels(*start)
                end_px = calibration.to_pixels(*end)
            else:
                start_px, end_px = start, end

            # Check if both endpoints are near edge
            start_near_edge = (
                start_px[0] < margin or start_px[0] > width - margin or
                start_px[1] < margin or start_px[1] > height - margin
            )
            end_near_edge = (
                end_px[0] < margin or end_px[0] > width - margin or
                end_px[1] < margin or end_px[1] > height - margin
            )

            # Only remove if BOTH endpoints are near edge (border line)
            is_edge_artifact = start_near_edge and end_near_edge

        elif etype == "circle":
            center = props.get("center", (0, 0))
            radius = props.get("radius", 0)

            if hasattr(calibration, 'to_pixels'):
                center_px = calibration.to_pixels(*center)
            else:
                center_px = center

            # Check if circle touches edge
            is_edge_artifact = (
                center_px[0] - radius < margin or
                center_px[0] + radius > width - margin or
                center_px[1] - radius < margin or
                center_px[1] + radius > height - margin
            )

        if not is_edge_artifact:
            kept.append(entity)
        else:
            removed += 1

    return kept, removed


def _remove_isolated_small_circles(
    entities: List[EntityToCreate],
    max_radius: float,
    calibration: "ScaleCalibration",
) -> Tuple[List[EntityToCreate], int]:
    """
    Remove very small circles that are likely noise (dots, specks).

    Args:
        entities: List of entities to filter
        max_radius: Maximum radius in pixels to consider as isolated
        calibration: For coordinate conversion

    Returns:
        Tuple of (filtered entities, count removed)
    """
    kept: List[EntityToCreate] = []
    removed = 0

    for entity in entities:
        etype = entity.entity_type
        if isinstance(etype, EntityType):
            etype = etype.value

        if etype == "circle":
            radius_dwg = entity.properties.get("radius", 0)

            # Convert to pixels
            if hasattr(calibration, 'pixels_per_unit') and calibration.pixels_per_unit:
                radius_px = radius_dwg * calibration.pixels_per_unit
            else:
                radius_px = radius_dwg * 10

            if radius_px > max_radius:
                kept.append(entity)
            else:
                removed += 1
        else:
            kept.append(entity)

    return kept, removed


def clean_entities_like_draftsman(
    entities: List[EntityToCreate],
    analysis: "DrawingAnalysis",
    calibration: "ScaleCalibration",
    config: "HybridExtractionConfig",
    image_size: Optional[Tuple[int, int]] = None,
) -> Tuple[List[EntityToCreate], CleanupStatistics]:
    """
    Clean extracted entities like a draftsman would review a tracing.

    Removes noise, artifacts, incomplete curves, and lines through text.
    This produces cleaner output suitable for AutoCAD creation.

    Args:
        entities: List of extracted entities
        analysis: Drawing analysis for text regions
        calibration: Coordinate calibration
        config: Hybrid extraction configuration
        image_size: Optional (width, height) for edge artifact removal

    Returns:
        Tuple of (cleaned entities, cleanup statistics)

    Example:
        >>> cleaned, stats = clean_entities_like_draftsman(
        ...     entities, analysis, calibration, config
        ... )
        >>> print(f"Removed {stats.total_removed} artifacts")
    """
    stats = CleanupStatistics()
    cleanup_config = config.draft_cleanup

    if not cleanup_config.enabled:
        stats.total_kept = len(entities)
        return entities, stats

    current = entities

    # 1. Remove short segments (noise)
    if cleanup_config.remove_short_segments:
        current, count = _remove_short_segments(
            current,
            cleanup_config.min_line_length_px,
            cleanup_config.min_arc_length_deg,
            calibration,
        )
        stats.short_segments_removed = count

    # 2. Remove lines through text
    if cleanup_config.remove_lines_through_text:
        text_boxes = _calculate_text_bounding_boxes(analysis, config)
        current, count = _remove_lines_through_text(
            current,
            text_boxes,
            cleanup_config.text_intersection_threshold,
            calibration,
        )
        stats.text_intersections_removed = count

    # 3. Remove incomplete curves
    if cleanup_config.remove_incomplete_curves:
        current, count = _remove_incomplete_curves(
            current,
            cleanup_config.min_arc_completeness,
        )
        stats.incomplete_curves_removed = count

    # 4. Filter low confidence
    if cleanup_config.filter_low_confidence:
        current, count = _remove_low_confidence(
            current,
            cleanup_config.min_confidence_threshold,
        )
        stats.low_confidence_removed = count

    # 5. Remove edge artifacts
    if cleanup_config.remove_edge_artifacts and image_size:
        current, count = _remove_edge_artifacts(
            current,
            image_size,
            cleanup_config.edge_margin_px,
            calibration,
        )
        stats.edge_artifacts_removed = count

    # 6. Remove isolated small circles
    if cleanup_config.remove_isolated_circles:
        current, count = _remove_isolated_small_circles(
            current,
            cleanup_config.max_isolated_circle_radius,
            calibration,
        )
        stats.isolated_circles_removed = count

    stats.total_removed = (
        stats.short_segments_removed +
        stats.text_intersections_removed +
        stats.incomplete_curves_removed +
        stats.low_confidence_removed +
        stats.edge_artifacts_removed +
        stats.isolated_circles_removed
    )
    stats.total_kept = len(current)

    logger.info(
        "draftsman_cleanup_complete",
        short_segments_removed=stats.short_segments_removed,
        text_intersections_removed=stats.text_intersections_removed,
        incomplete_curves_removed=stats.incomplete_curves_removed,
        low_confidence_removed=stats.low_confidence_removed,
        edge_artifacts_removed=stats.edge_artifacts_removed,
        total_removed=stats.total_removed,
        total_kept=stats.total_kept,
    )

    return current, stats


@dataclass
class _TextExtractionResult:
    """Result of text extraction with OCR anchoring."""
    entities: List[EntityToCreate]
    ocr_anchored: int = 0
    ocr_fallback: int = 0


async def _extract_text_entities(
    analysis: DrawingAnalysis,
    calibration: ScaleCalibration,
    gemini_entities: List[EntityToCreate],
    image_path: Optional[Path],
    config: HybridExtractionConfig,
) -> _TextExtractionResult:
    """
    Extract text entities with optional OCR position anchoring.

    Uses OCR (Tesseract) to get pixel-accurate text positions when available,
    falling back to Gemini's estimated positions otherwise.

    Args:
        analysis: Drawing analysis from Gemini
        calibration: Coordinate calibration
        gemini_entities: Pre-extracted Gemini entities (for fallback)
        image_path: Path to source image (required for OCR)
        config: Hybrid extraction configuration

    Returns:
        _TextExtractionResult with text entities and statistics
    """
    result = _TextExtractionResult(entities=[])

    # Filter text entities from Gemini for fallback
    def get_gemini_text() -> List[EntityToCreate]:
        text_entities = [
            e for e in gemini_entities
            if e.entity_type in (EntityType.MTEXT, EntityType.TEXT, "mtext", "text")
        ]
        logger.debug(
            "gemini_text_entities_found",
            total_gemini_entities=len(gemini_entities),
            text_entities=len(text_entities),
        )
        return text_entities

    # Log Gemini's raw text count from analysis
    logger.info(
        "text_extraction_starting",
        gemini_text_elements=len(analysis.elements.text),
        use_ocr=config.use_ocr_for_text_positions,
        has_image_path=image_path is not None,
    )

    # Try OCR anchoring if enabled and image available
    if config.use_ocr_for_text_positions and image_path:
        try:
            from .ocr_text_anchoring import anchor_text_positions, is_ocr_available

            if is_ocr_available():
                text_entities = await anchor_text_positions(
                    analysis=analysis,
                    image_path=image_path,
                    calibration=calibration,
                    min_similarity=config.ocr_min_similarity,
                    min_ocr_confidence=config.ocr_min_confidence,
                )

                # Count anchored vs fallback
                result.entities = text_entities
                result.ocr_anchored = sum(
                    1 for e in text_entities
                    if e.source == ExtractionSource.HYBRID
                )
                result.ocr_fallback = len(text_entities) - result.ocr_anchored

                logger.info(
                    "ocr_text_anchoring_applied",
                    total_text=len(text_entities),
                    ocr_anchored=result.ocr_anchored,
                    fallback=result.ocr_fallback,
                )
                return result
            else:
                logger.warning("ocr_not_available", message="Using Gemini text positions")

        except Exception as e:
            logger.warning("ocr_text_anchoring_failed", error=str(e))

    # Fallback: use Gemini's direct text extraction
    gemini_text = get_gemini_text()
    result.entities = gemini_text
    result.ocr_fallback = len(gemini_text)

    logger.info(
        "text_extraction_fallback_to_gemini",
        text_entities=len(gemini_text),
        gemini_analysis_text=len(analysis.elements.text),
    )

    return result


async def hybrid_extract_all(
    analysis: DrawingAnalysis,
    calibration: ScaleCalibration,
    image_path: Optional[Path] = None,
    config: Optional[HybridExtractionConfig] = None,
) -> HybridExtractionResult:
    """
    Hybrid extraction combining Gemini, OpenCV, and YOLO.

    This is the optimal extraction approach:
    1. Gemini provides semantic understanding (what & where)
    2. OpenCV extracts geometry with pixel-perfect accuracy
    3. YOLO detects symbols with trained precision
    4. Results are merged and deduplicated
    5. Optional Gemini validation pass

    Args:
        analysis: DrawingAnalysis from Gemini (Phase 2)
        calibration: ScaleCalibration (Phase 3)
        image_path: Path to source image
        config: Optional HybridExtractionConfig

    Returns:
        HybridExtractionResult with all extracted entities

    Example:
        >>> result = await hybrid_extract_all(analysis, calibration, image_path)
        >>> print(f"Gemini: {result.gemini_entities}, OpenCV: {result.opencv_entities}")
        >>> print(f"YOLO: {result.yolo_entities}, Total: {result.total_entities}")
    """
    config = config or HybridExtractionConfig()

    result = HybridExtractionResult(
        primary_strategy="hybrid",
        drawing_type=analysis.drawing_type,
        calibration_method=calibration.method,
        calibration_confidence=calibration.confidence,
    )

    logger.info(
        "hybrid_extraction_starting",
        drawing_type=analysis.drawing_type,
        total_gemini_elements=analysis.total_elements,
        use_opencv=config.use_opencv_for_lines or config.use_opencv_for_circles,
        use_yolo=config.use_yolo_for_symbols,
        use_ocr_text_anchoring=config.use_ocr_for_text_positions,
    )

    all_entities: List[EntityToCreate] = []
    image: Optional[np.ndarray] = None  # Loaded lazily when needed

    # Get all Gemini entities once (used for text, dimensions, symbols, geometry fallback)
    gemini_entities = await direct_extraction(analysis, calibration)

    # 1. Text extraction with OCR anchoring
    text_result = await _extract_text_entities(
        analysis=analysis,
        calibration=calibration,
        gemini_entities=gemini_entities,
        image_path=image_path,
        config=config,
    )
    all_entities.extend(text_result.entities)
    result.gemini_entities = len(text_result.entities)
    result.ocr_text_anchored = text_result.ocr_anchored
    result.ocr_text_fallback = text_result.ocr_fallback

    # 1b. Dimensions (always from Gemini)
    gemini_dims = [
        e for e in gemini_entities
        if e.entity_type in (EntityType.DIMENSION, "dimension")
    ]
    all_entities.extend(gemini_dims)
    result.gemini_entities += len(gemini_dims)

    # Also include Gemini's symbols if not using YOLO
    if not config.use_yolo_for_symbols:
        gemini_symbols = [
            e for e in gemini_entities
            if e.entity_type in (EntityType.BLOCK, "block")
        ]
        all_entities.extend(gemini_symbols)
        result.gemini_entities += len(gemini_symbols)

    # Also include Gemini geometry if not using OpenCV
    if not config.use_opencv_for_lines and not config.use_opencv_for_circles:
        gemini_geometry = [
            e for e in gemini_entities
            if e.entity_type in (EntityType.LINE, EntityType.ARC, EntityType.CIRCLE,
                                "line", "arc", "circle")
        ]
        all_entities.extend(gemini_geometry)
        result.gemini_entities += len(gemini_geometry)

    # 1b. Text/Graphics Separation (Phase B: Fletcher-Kasturi)
    text_mask = None
    graphics_mask = None
    if image_path and config.enable_text_graphics_separation:
        try:
            from .text_graphics_separation import (
                FletcherKasturiSeparator,
                SeparationConfig,
                is_separation_available,
            )
            from .binarization import ensemble_binarize

            if is_separation_available():
                if image is None:
                    image = _load_image(image_path)

                if image is not None:
                    # Binarize first
                    binary = ensemble_binarize(image)
                    binary_image = binary.binary_image

                    # Separate text from graphics
                    separation_config = SeparationConfig.from_settings()
                    separator = FletcherKasturiSeparator(binary_image, separation_config)
                    separation_result = separator.separate()

                    text_mask = separation_result.text_mask
                    graphics_mask = separation_result.graphics_mask

                    logger.info(
                        "text_graphics_separation_complete",
                        text_components=separation_result.num_text_components,
                        graphics_components=separation_result.num_graphics_components,
                        text_lines=separation_result.num_text_lines,
                    )

        except Exception as e:
            logger.warning("text_graphics_separation_failed", error=str(e))

    # 2. OpenCV extraction (lines, circles - pixel perfect)
    if image_path and (config.use_opencv_for_lines or config.use_opencv_for_circles):
        image = _load_image(image_path)
        if image is not None:
            # Phase A: Apply preprocessing if enabled
            if config.enable_preprocessing:
                try:
                    from .preprocessing import preprocess_image, PreprocessingConfig, DeskewConfig
                    from .binarization import ensemble_binarize, BinarizationConfig

                    # Preprocess: deskew, denoise, contrast
                    preprocess_config = PreprocessingConfig(
                        enable_deskew=config.enable_deskew,
                        enable_denoise=config.enable_denoise,
                        enable_contrast=config.enable_contrast_enhancement,
                        enable_border_removal=False,  # Don't crop for vectorization
                    )
                    preprocess_result = preprocess_image(image, preprocess_config)
                    image = preprocess_result.image

                    logger.info(
                        "preprocessing_applied",
                        operations=preprocess_result.operations_applied,
                        skew_angle=preprocess_result.skew_angle,
                    )

                except Exception as e:
                    logger.warning("preprocessing_failed", error=str(e))

            opencv_entities = await hybrid_opencv_extraction(
                image, analysis, calibration, config
            )

            # Enforce entity limit to prevent crashes
            if len(opencv_entities) > config.max_entities_per_source:
                logger.warning(
                    "opencv_entities_limited",
                    original_count=len(opencv_entities),
                    limit=config.max_entities_per_source,
                    reason="Too many entities detected, keeping highest confidence"
                )
                # Sort by confidence and keep top N
                opencv_entities.sort(key=lambda e: e.confidence, reverse=True)
                opencv_entities = opencv_entities[:config.max_entities_per_source]

            all_entities.extend(opencv_entities)
            result.opencv_entities = len(opencv_entities)
            result.opencv_count = len(opencv_entities)

    # 2b. VTracer extraction (Phase B - O(n) vectorization, alternative to OpenCV)
    vtracer_used = False
    if image_path and config.use_vtracer:
        from aec_agent.config.settings import get_settings
        settings = get_settings()

        if settings.vtracer_enabled:
            if image is None:
                image = _load_image(image_path)

            if image is not None:
                vtracer_entities = _detect_with_vtracer(image, calibration)
                if vtracer_entities:
                    all_entities.extend(vtracer_entities)
                    vtracer_used = True
                    logger.info(
                        "vtracer_extraction_complete",
                        entities=len(vtracer_entities),
                    )
                elif config.vtracer_fallback_to_opencv:
                    logger.info("vtracer_fallback_to_opencv", reason="no entities from VTracer")

    # 3. YOLO extraction (symbols - trained detection)
    if image_path and config.use_yolo_for_symbols:
        if image is None:
            image = _load_image(image_path)
        if image is not None:
            yolo_entities = await hybrid_yolo_extraction(
                image, analysis, calibration, config
            )
            all_entities.extend(yolo_entities)
            result.yolo_entities = len(yolo_entities)

    # 4. Merge duplicates
    merged_entities, duplicates = _merge_duplicate_entities(
        all_entities,
        tolerance=config.coordinate_tolerance,
        prefer_opencv=config.prefer_opencv_geometry,
    )

    # Enforce total entity limit
    if len(merged_entities) > config.max_total_entities:
        logger.warning(
            "total_entities_limited",
            original_count=len(merged_entities),
            limit=config.max_total_entities,
            reason="Too many total entities, keeping highest confidence"
        )
        merged_entities.sort(key=lambda e: e.confidence, reverse=True)
        merged_entities = merged_entities[:config.max_total_entities]

    result.entities = merged_entities
    result.duplicates_merged = duplicates
    result.direct_count = result.gemini_entities

    # 5. Gemini Refinement Pass
    if config.enable_refinement and result.entities:
        try:
            from .gemini_refinement import refine_entities_with_gemini, RefinementConfig

            refinement_config = RefinementConfig(
                snap_to_grid=config.refine_snap_to_grid,
                connect_endpoints=config.refine_connect_endpoints,
                align_parallel_lines=config.refine_align_parallel,
                remove_duplicates=config.refine_remove_duplicates,
            )

            refinement_result = await refine_entities_with_gemini(
                entities=result.entities,
                image_path=image_path,
                calibration=calibration,
                config=refinement_config,
            )

            result.entities = refinement_result.refined_entities
            result.refinement_applied = True
            result.refinement_adjustments = refinement_result.adjustments_made
            result.endpoints_connected = refinement_result.endpoints_connected
            result.lines_snapped = refinement_result.lines_snapped

            logger.info(
                "refinement_pass_complete",
                adjustments=result.refinement_adjustments,
                endpoints_connected=result.endpoints_connected,
                lines_snapped=result.lines_snapped,
            )

        except Exception as e:
            logger.warning("refinement_pass_failed", error=str(e))
            result.refinement_applied = False

    # 6. Draftsman-like cleanup (remove noise, artifacts, incomplete curves)
    if config.draft_cleanup.enabled and result.entities:
        # Get image size for edge artifact detection
        image_size: Optional[Tuple[int, int]] = None
        if image is not None:
            image_size = (image.shape[1], image.shape[0])  # (width, height)

        cleaned_entities, cleanup_stats = clean_entities_like_draftsman(
            entities=result.entities,
            analysis=analysis,
            calibration=calibration,
            config=config,
            image_size=image_size,
        )

        result.entities = cleaned_entities
        result.cleanup_applied = True
        result.cleanup_stats = cleanup_stats

        logger.info(
            "draftsman_cleanup_applied",
            entities_before=cleanup_stats.total_kept + cleanup_stats.total_removed,
            entities_after=cleanup_stats.total_kept,
            total_removed=cleanup_stats.total_removed,
        )

    # 7. Line simplification (reduce vertex count)
    if config.enable_simplification and result.entities:
        try:
            from .simplification import (
                simplify_line,
                SimplificationConfig,
                SimplificationMethod,
            )

            simplified_entities = []
            total_points_before = 0
            total_points_after = 0

            for entity in result.entities:
                etype = entity.entity_type
                if isinstance(etype, EntityType):
                    etype = etype.value

                # Only simplify polylines (multi-point entities)
                if etype == "polyline" and not config.simplify_polylines_only:
                    points = entity.properties.get("points", [])
                    if len(points) > 2:
                        total_points_before += len(points)

                        # Convert to tuples
                        point_tuples = [(p[0], p[1]) for p in points]

                        simplify_config = SimplificationConfig(
                            method=SimplificationMethod.RDP,
                            rdp_epsilon=config.simplification_epsilon,
                        )
                        simplify_result = simplify_line(point_tuples, simplify_config)

                        # Update entity with simplified points
                        simplified_entity = EntityToCreate(
                            entity_type=entity.entity_type,
                            layer=entity.layer,
                            properties={
                                **entity.properties,
                                "points": [[p[0], p[1]] for p in simplify_result.points],
                            },
                            source=entity.source,
                            confidence=entity.confidence,
                        )
                        simplified_entities.append(simplified_entity)
                        total_points_after += len(simplify_result.points)
                    else:
                        simplified_entities.append(entity)
                        total_points_after += len(points) if etype == "polyline" else 0
                else:
                    simplified_entities.append(entity)

            result.entities = simplified_entities

            if total_points_before > 0:
                reduction = 1.0 - (total_points_after / total_points_before)
                logger.info(
                    "line_simplification_applied",
                    points_before=total_points_before,
                    points_after=total_points_after,
                    reduction_ratio=reduction,
                )

        except Exception as e:
            logger.warning("line_simplification_failed", error=str(e))

    logger.info(
        "hybrid_extraction_complete",
        gemini_entities=result.gemini_entities,
        opencv_entities=result.opencv_entities,
        yolo_entities=result.yolo_entities,
        duplicates_merged=result.duplicates_merged,
        refinement_adjustments=result.refinement_adjustments,
        cleanup_applied=result.cleanup_applied,
        total_entities=len(result.entities),
    )

    return result


# Alias for backward compatibility
extract_hybrid = hybrid_extract_all
