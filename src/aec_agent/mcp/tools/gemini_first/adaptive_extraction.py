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

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

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
)
from .coordinate_calibration import ScaleCalibration

logger = structlog.get_logger(__name__)


class ExtractionSource(str, Enum):
    """Source/method used for extraction."""
    DIRECT = "direct"
    GUIDED_RASTERIZATION = "guided_rasterization"
    SELECTIVE_OPENCV = "selective_opencv"
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

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "entity_type": self.entity_type.value if isinstance(self.entity_type, EntityType) else self.entity_type,
            "layer": self.layer,
            "properties": self.properties,
            "source": self.source.value if isinstance(self.source, ExtractionSource) else self.source,
            "confidence": self.confidence,
            "source_element": self.source_element,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "EntityToCreate":
        """Create from dictionary."""
        return cls(
            entity_type=data["entity_type"],
            layer=data["layer"],
            properties=data["properties"],
            source=data.get("source", "direct"),
            confidence=data.get("confidence", 1.0),
            source_element=data.get("source_element"),
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

        # Ensure minimum height
        height_dwg = max(height_dwg, 0.1)

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
    image,  # np.ndarray
    calibration: ScaleCalibration,
) -> List[EntityToCreate]:
    """Process a single region with OpenCV."""
    import cv2
    import numpy as np

    entities = []
    bounds = region.bounds

    # Validate bounds
    if len(bounds) != 4:
        return entities

    x1, y1, x2, y2 = bounds
    if x2 <= x1 or y2 <= y1:
        return entities

    # Crop region from image
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


def _detect_lines_opencv(processed_image) -> List[dict]:
    """Detect lines using OpenCV HoughLinesP."""
    import cv2
    import numpy as np

    detected = []

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
                "start": (x1, y1),
                "end": (x2, y2),
            })

    return detected


def _detect_circles_opencv(gray_image) -> List[dict]:
    """Detect circles using OpenCV HoughCircles."""
    import cv2
    import numpy as np

    detected = []

    # Detect circles using Hough transform
    circles = cv2.HoughCircles(
        gray_image,
        cv2.HOUGH_GRADIENT,
        dp=1,
        minDist=20,
        param1=50,
        param2=30,
        minRadius=5,
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

    # 3. Selective OpenCV for special regions
    if image_path:
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
