"""
Phase 5: AutoCAD Entity Creation - Create entities in AutoCAD from extraction results.

This module takes the ExtractionResult from Phase 4 and creates actual entities
in AutoCAD using the existing MCP tools / sidecar commands.

Key Features:
- Batch entity creation with progress tracking
- Automatic layer creation for required layers
- Support for lines, arcs, circles, text, and block insertions
- Error handling with detailed failure tracking
- Linetype assignment support
- Optional raster command execution

Usage:
    >>> from .autocad_creation import create_entities_in_autocad
    >>> result = await create_entities_in_autocad(extraction_result)
    >>> print(f"Created {result.success_count} entities")
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

import structlog

from .adaptive_extraction import (
    EntityToCreate,
    EntityType,
    ExtractionResult,
    RasterCommand,
    get_required_layers,
    get_required_blocks,
)

logger = structlog.get_logger(__name__)


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class EntityCreationResult:
    """
    Result of creating a single entity.

    Tracks the outcome of attempting to create one AutoCAD entity,
    including success/failure status and any error messages.
    """
    entity_type: str
    layer: str
    success: bool
    handle: Optional[str] = None  # AutoCAD entity handle if created
    error: Optional[str] = None
    source_element: Optional[str] = None

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "entity_type": self.entity_type,
            "layer": self.layer,
            "success": self.success,
            "handle": self.handle,
            "error": self.error,
            "source_element": self.source_element,
        }


@dataclass
class LayerCreationResult:
    """Result of creating a layer."""
    name: str
    success: bool
    already_existed: bool = False
    error: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "success": self.success,
            "already_existed": self.already_existed,
            "error": self.error,
        }


@dataclass
class CreationStatistics:
    """Statistics about the entity creation process."""
    total_entities: int = 0
    success_count: int = 0
    failure_count: int = 0

    # By entity type
    lines_created: int = 0
    arcs_created: int = 0
    circles_created: int = 0
    texts_created: int = 0
    blocks_created: int = 0
    dimensions_created: int = 0
    polylines_created: int = 0
    other_created: int = 0

    # Layers
    layers_created: int = 0
    layers_existed: int = 0

    # Raster commands
    raster_commands_executed: int = 0
    raster_commands_failed: int = 0

    def to_dict(self) -> dict:
        return {
            "total_entities": self.total_entities,
            "success_count": self.success_count,
            "failure_count": self.failure_count,
            "by_type": {
                "lines": self.lines_created,
                "arcs": self.arcs_created,
                "circles": self.circles_created,
                "texts": self.texts_created,
                "blocks": self.blocks_created,
                "dimensions": self.dimensions_created,
                "polylines": self.polylines_created,
                "other": self.other_created,
            },
            "layers": {
                "created": self.layers_created,
                "already_existed": self.layers_existed,
            },
            "raster_commands": {
                "executed": self.raster_commands_executed,
                "failed": self.raster_commands_failed,
            },
        }


@dataclass
class AutoCADCreationResult:
    """
    Complete result of AutoCAD entity creation process.

    Contains all results from creating entities, layers, and executing
    raster commands, along with aggregate statistics.
    """
    success: bool  # True if majority of entities created successfully
    statistics: CreationStatistics = field(default_factory=CreationStatistics)

    entity_results: List[EntityCreationResult] = field(default_factory=list)
    layer_results: List[LayerCreationResult] = field(default_factory=list)
    raster_results: List[dict] = field(default_factory=list)

    # Summary lists
    created_handles: List[str] = field(default_factory=list)
    failed_entities: List[dict] = field(default_factory=list)

    # Metadata
    drawing_type: str = ""
    extraction_source: str = ""

    @property
    def success_count(self) -> int:
        return self.statistics.success_count

    @property
    def failure_count(self) -> int:
        return self.statistics.failure_count

    @property
    def success_rate(self) -> float:
        total = self.statistics.total_entities
        if total == 0:
            return 1.0
        return self.statistics.success_count / total

    def to_dict(self) -> dict:
        return {
            "success": self.success,
            "statistics": self.statistics.to_dict(),
            "created_handles": self.created_handles,
            "failed_entities": self.failed_entities[:50],  # Limit for large results
            "layer_results": [r.to_dict() for r in self.layer_results],
            "raster_results": self.raster_results,
            "metadata": {
                "drawing_type": self.drawing_type,
                "extraction_source": self.extraction_source,
                "success_rate": f"{self.success_rate:.1%}",
            },
        }


# =============================================================================
# Layer Colors (NCS-based)
# =============================================================================

# Default colors for layer prefixes (NCS discipline colors)
LAYER_PREFIX_COLORS = {
    "A-": 7,   # Architectural - White
    "M-": 4,   # Mechanical - Cyan
    "E-": 1,   # Electrical - Red
    "P-": 5,   # Plumbing - Blue
    "F-": 1,   # Fire - Red
    "T-": 3,   # Telecom/Low Voltage - Green
    "G-": 7,   # General - White
    "S-": 6,   # Structural - Magenta
    "C-": 2,   # Civil - Yellow
}

def get_color_for_layer(layer_name: str) -> int:
    """Get default color for a layer based on its prefix."""
    for prefix, color in LAYER_PREFIX_COLORS.items():
        if layer_name.upper().startswith(prefix):
            return color
    return 7  # Default to white


# =============================================================================
# Entity Creation Functions
# =============================================================================

async def create_layer_if_needed(
    layer_name: str,
    call_command: Callable,
    existing_layers: Optional[Set[str]] = None,
) -> LayerCreationResult:
    """
    Create a layer if it doesn't already exist.

    Args:
        layer_name: Name of the layer to create
        call_command: Async function to call AutoCAD commands
        existing_layers: Set of known existing layer names (for optimization)

    Returns:
        LayerCreationResult with creation status
    """
    # Skip layer "0" - it always exists
    if layer_name == "0":
        return LayerCreationResult(
            name=layer_name,
            success=True,
            already_existed=True,
        )

    # Check if we already know it exists
    if existing_layers and layer_name in existing_layers:
        return LayerCreationResult(
            name=layer_name,
            success=True,
            already_existed=True,
        )

    try:
        color = get_color_for_layer(layer_name)
        result = await call_command(
            "create_layer",
            {"name": layer_name, "color": color}
        )

        # Check if layer already existed
        if result.get("success", False):
            return LayerCreationResult(
                name=layer_name,
                success=True,
                already_existed=False,
            )
        elif "already exists" in str(result.get("error", {}).get("message", "")).lower():
            return LayerCreationResult(
                name=layer_name,
                success=True,
                already_existed=True,
            )
        else:
            return LayerCreationResult(
                name=layer_name,
                success=False,
                error=str(result.get("error", {}).get("message", "Unknown error")),
            )

    except Exception as e:
        return LayerCreationResult(
            name=layer_name,
            success=False,
            error=str(e),
        )


async def create_line_entity(
    entity: EntityToCreate,
    call_command: Callable,
) -> EntityCreationResult:
    """Create a line entity in AutoCAD."""
    props = entity.properties

    start = props.get("start", (0, 0))
    end = props.get("end", (0, 0))

    # Handle both tuple and list formats
    if isinstance(start, (list, tuple)) and len(start) >= 2:
        start_x, start_y = start[0], start[1]
    else:
        return EntityCreationResult(
            entity_type="line",
            layer=entity.layer,
            success=False,
            error=f"Invalid start point format: {start}",
        )

    if isinstance(end, (list, tuple)) and len(end) >= 2:
        end_x, end_y = end[0], end[1]
    else:
        return EntityCreationResult(
            entity_type="line",
            layer=entity.layer,
            success=False,
            error=f"Invalid end point format: {end}",
        )

    try:
        params = {
            "start": [float(start_x), float(start_y)],
            "end": [float(end_x), float(end_y)],
            "layer": entity.layer,
        }

        # Add linetype if specified
        linetype = props.get("linetype")
        if linetype and linetype != "Continuous":
            params["linetype"] = linetype

        # Add lineweight/thickness if specified
        thickness = props.get("thickness")
        if thickness and thickness > 0:
            params["lineweight"] = float(thickness)

        result = await call_command("draw_line", params)

        if result.get("success", False):
            return EntityCreationResult(
                entity_type="line",
                layer=entity.layer,
                success=True,
                handle=result.get("data", {}).get("handle"),
                source_element=entity.source_element,
            )
        else:
            return EntityCreationResult(
                entity_type="line",
                layer=entity.layer,
                success=False,
                error=str(result.get("error", {}).get("message", "Unknown error")),
                source_element=entity.source_element,
            )

    except Exception as e:
        return EntityCreationResult(
            entity_type="line",
            layer=entity.layer,
            success=False,
            error=str(e),
            source_element=entity.source_element,
        )


async def create_arc_entity(
    entity: EntityToCreate,
    call_command: Callable,
) -> EntityCreationResult:
    """Create an arc entity in AutoCAD."""
    props = entity.properties

    center = props.get("center", (0, 0))
    radius = props.get("radius", 1.0)
    start_angle = props.get("start_angle", 0.0)
    end_angle = props.get("end_angle", 90.0)

    # Handle center point format
    if isinstance(center, (list, tuple)) and len(center) >= 2:
        center_x, center_y = center[0], center[1]
    else:
        return EntityCreationResult(
            entity_type="arc",
            layer=entity.layer,
            success=False,
            error=f"Invalid center point format: {center}",
        )

    try:
        params = {
            "center": [float(center_x), float(center_y), 0.0],
            "radius": float(radius),
            "start_angle": float(start_angle),
            "end_angle": float(end_angle),
            "layer": entity.layer,
        }

        result = await call_command("draw_arc", params)

        if result.get("success", False):
            return EntityCreationResult(
                entity_type="arc",
                layer=entity.layer,
                success=True,
                handle=result.get("data", {}).get("handle"),
                source_element=entity.source_element,
            )
        else:
            return EntityCreationResult(
                entity_type="arc",
                layer=entity.layer,
                success=False,
                error=str(result.get("error", {}).get("message", "Unknown error")),
                source_element=entity.source_element,
            )

    except Exception as e:
        return EntityCreationResult(
            entity_type="arc",
            layer=entity.layer,
            success=False,
            error=str(e),
            source_element=entity.source_element,
        )


async def create_circle_entity(
    entity: EntityToCreate,
    call_command: Callable,
) -> EntityCreationResult:
    """Create a circle entity in AutoCAD."""
    props = entity.properties

    center = props.get("center", (0, 0))
    radius = props.get("radius", 1.0)

    # Handle center point format
    if isinstance(center, (list, tuple)) and len(center) >= 2:
        center_x, center_y = center[0], center[1]
    else:
        return EntityCreationResult(
            entity_type="circle",
            layer=entity.layer,
            success=False,
            error=f"Invalid center point format: {center}",
        )

    if radius <= 0:
        return EntityCreationResult(
            entity_type="circle",
            layer=entity.layer,
            success=False,
            error=f"Invalid radius: {radius}",
        )

    try:
        params = {
            "center": [float(center_x), float(center_y)],
            "radius": float(radius),
            "layer": entity.layer,
        }

        result = await call_command("draw_circle", params)

        if result.get("success", False):
            return EntityCreationResult(
                entity_type="circle",
                layer=entity.layer,
                success=True,
                handle=result.get("data", {}).get("handle"),
                source_element=entity.source_element,
            )
        else:
            return EntityCreationResult(
                entity_type="circle",
                layer=entity.layer,
                success=False,
                error=str(result.get("error", {}).get("message", "Unknown error")),
                source_element=entity.source_element,
            )

    except Exception as e:
        return EntityCreationResult(
            entity_type="circle",
            layer=entity.layer,
            success=False,
            error=str(e),
            source_element=entity.source_element,
        )


async def create_text_entity(
    entity: EntityToCreate,
    call_command: Callable,
) -> EntityCreationResult:
    """Create a text (MTEXT) entity in AutoCAD."""
    props = entity.properties

    content = props.get("content", "")
    position = props.get("position", (0, 0))
    height = props.get("height", 0.125)  # Default 1/8"

    if not content:
        return EntityCreationResult(
            entity_type="mtext",
            layer=entity.layer,
            success=False,
            error="Empty text content",
        )

    # Handle position format
    if isinstance(position, (list, tuple)) and len(position) >= 2:
        pos_x, pos_y = position[0], position[1]
    else:
        return EntityCreationResult(
            entity_type="mtext",
            layer=entity.layer,
            success=False,
            error=f"Invalid position format: {position}",
        )

    try:
        params = {
            "text": str(content),  # C# sidecar expects "text", not "content"
            "position": [float(pos_x), float(pos_y), 0.0],
            "height": max(float(height), 0.0625),  # Minimum 1/16"
            "layer": entity.layer,
        }

        result = await call_command("draw_mtext", params)

        if result.get("success", False):
            return EntityCreationResult(
                entity_type="mtext",
                layer=entity.layer,
                success=True,
                handle=result.get("data", {}).get("handle"),
                source_element=entity.source_element,
            )
        else:
            return EntityCreationResult(
                entity_type="mtext",
                layer=entity.layer,
                success=False,
                error=str(result.get("error", {}).get("message", "Unknown error")),
                source_element=entity.source_element,
            )

    except Exception as e:
        return EntityCreationResult(
            entity_type="mtext",
            layer=entity.layer,
            success=False,
            error=str(e),
            source_element=entity.source_element,
        )


async def create_block_entity(
    entity: EntityToCreate,
    call_command: Callable,
) -> EntityCreationResult:
    """Create a block reference entity in AutoCAD."""
    props = entity.properties

    block_name = props.get("block_name", "")
    position = props.get("position", (0, 0))
    rotation = props.get("rotation", 0.0)
    scale = props.get("scale", 1.0)
    attributes = props.get("attributes", {})

    if not block_name:
        return EntityCreationResult(
            entity_type="block",
            layer=entity.layer,
            success=False,
            error="Missing block name",
        )

    # Handle position format
    if isinstance(position, (list, tuple)) and len(position) >= 2:
        pos_x, pos_y = position[0], position[1]
    else:
        return EntityCreationResult(
            entity_type="block",
            layer=entity.layer,
            success=False,
            error=f"Invalid position format: {position}",
        )

    try:
        params = {
            "block_name": block_name,
            "position": [float(pos_x), float(pos_y), 0.0],
            "rotation": float(rotation),
            "scale": float(scale) if scale > 0 else 1.0,
            "layer": entity.layer,
        }

        # Add attributes if any
        if attributes:
            params["attributes"] = attributes

        result = await call_command("insert_block", params)

        if result.get("success", False):
            return EntityCreationResult(
                entity_type="block",
                layer=entity.layer,
                success=True,
                handle=result.get("data", {}).get("handle"),
                source_element=entity.source_element,
            )
        else:
            # Check for missing block definition
            error_msg = str(result.get("error", {}).get("message", "Unknown error"))
            if "not found" in error_msg.lower() or "undefined" in error_msg.lower():
                error_msg = f"Block '{block_name}' not defined in drawing. {error_msg}"

            return EntityCreationResult(
                entity_type="block",
                layer=entity.layer,
                success=False,
                error=error_msg,
                source_element=entity.source_element,
            )

    except Exception as e:
        return EntityCreationResult(
            entity_type="block",
            layer=entity.layer,
            success=False,
            error=str(e),
            source_element=entity.source_element,
        )


async def create_polyline_entity(
    entity: EntityToCreate,
    call_command: Callable,
) -> EntityCreationResult:
    """Create a polyline entity in AutoCAD."""
    props = entity.properties

    points = props.get("points", [])
    closed = props.get("closed", False)

    if len(points) < 2:
        return EntityCreationResult(
            entity_type="polyline",
            layer=entity.layer,
            success=False,
            error="Polyline requires at least 2 points",
        )

    try:
        # Convert points to 3D format
        pts_3d = []
        for pt in points:
            if isinstance(pt, (list, tuple)) and len(pt) >= 2:
                pts_3d.append([float(pt[0]), float(pt[1]), 0.0])
            else:
                return EntityCreationResult(
                    entity_type="polyline",
                    layer=entity.layer,
                    success=False,
                    error=f"Invalid point format: {pt}",
                )

        params = {
            "points": pts_3d,
            "closed": closed,
            "layer": entity.layer,
        }

        result = await call_command("draw_polyline", params)

        if result.get("success", False):
            return EntityCreationResult(
                entity_type="polyline",
                layer=entity.layer,
                success=True,
                handle=result.get("data", {}).get("handle"),
                source_element=entity.source_element,
            )
        else:
            return EntityCreationResult(
                entity_type="polyline",
                layer=entity.layer,
                success=False,
                error=str(result.get("error", {}).get("message", "Unknown error")),
                source_element=entity.source_element,
            )

    except Exception as e:
        return EntityCreationResult(
            entity_type="polyline",
            layer=entity.layer,
            success=False,
            error=str(e),
            source_element=entity.source_element,
        )


# =============================================================================
# Entity Type Dispatch
# =============================================================================

ENTITY_CREATORS = {
    EntityType.LINE: create_line_entity,
    EntityType.ARC: create_arc_entity,
    EntityType.CIRCLE: create_circle_entity,
    EntityType.MTEXT: create_text_entity,
    EntityType.TEXT: create_text_entity,
    EntityType.BLOCK: create_block_entity,
    EntityType.POLYLINE: create_polyline_entity,
    # String versions for compatibility
    "line": create_line_entity,
    "arc": create_arc_entity,
    "circle": create_circle_entity,
    "mtext": create_text_entity,
    "text": create_text_entity,
    "block": create_block_entity,
    "polyline": create_polyline_entity,
}


async def create_single_entity(
    entity: EntityToCreate,
    call_command: Callable,
) -> EntityCreationResult:
    """
    Create a single entity in AutoCAD.

    Dispatches to the appropriate creation function based on entity type.

    Args:
        entity: EntityToCreate with type, layer, and properties
        call_command: Async function to call AutoCAD commands

    Returns:
        EntityCreationResult with creation status
    """
    entity_type = entity.entity_type

    # Get the creator function
    creator = ENTITY_CREATORS.get(entity_type)

    if creator is None:
        return EntityCreationResult(
            entity_type=str(entity_type),
            layer=entity.layer,
            success=False,
            error=f"Unsupported entity type: {entity_type}",
            source_element=entity.source_element,
        )

    return await creator(entity, call_command)


# =============================================================================
# Main Entry Point
# =============================================================================

async def create_entities_in_autocad(
    extraction_result: ExtractionResult,
    call_command: Optional[Callable] = None,
    create_layers: bool = True,
    execute_raster_commands: bool = False,
    progress_callback: Optional[Callable[[int, int], None]] = None,
) -> AutoCADCreationResult:
    """
    Create all extracted entities in AutoCAD.

    This is the main entry point for Phase 5. It takes the ExtractionResult
    from Phase 4 and creates all entities in the active AutoCAD drawing.

    Args:
        extraction_result: ExtractionResult from Phase 4 containing entities
        call_command: Async function to call AutoCAD commands. If None,
                      imports call_autocad_command from sidecar_client.
        create_layers: Whether to create layers that don't exist (default True)
        execute_raster_commands: Whether to execute raster commands (default False)
        progress_callback: Optional callback(current, total) for progress updates

    Returns:
        AutoCADCreationResult with all creation results and statistics

    Example:
        >>> from .adaptive_extraction import extract_all
        >>> extraction = await extract_all(analysis, calibration, image_path)
        >>> result = await create_entities_in_autocad(extraction)
        >>> print(f"Created {result.success_count} of {result.statistics.total_entities}")
    """
    # Import sidecar client if not provided
    if call_command is None:
        from aec_agent.mcp.sidecar_client import call_autocad_command
        call_command = call_autocad_command

    # Initialize result
    result = AutoCADCreationResult(
        success=True,
        drawing_type=extraction_result.drawing_type,
        extraction_source=extraction_result.primary_strategy,
    )

    total_entities = len(extraction_result.entities)
    result.statistics.total_entities = total_entities

    logger.info(
        "autocad_creation_starting",
        total_entities=total_entities,
        raster_commands=len(extraction_result.raster_commands),
        create_layers=create_layers,
    )

    # Step 1: Create required layers
    if create_layers:
        required_layers = get_required_layers(extraction_result)
        existing_layers: Set[str] = set()

        for layer_name in required_layers:
            layer_result = await create_layer_if_needed(
                layer_name, call_command, existing_layers
            )
            result.layer_results.append(layer_result)

            if layer_result.success:
                existing_layers.add(layer_name)
                if layer_result.already_existed:
                    result.statistics.layers_existed += 1
                else:
                    result.statistics.layers_created += 1

        logger.info(
            "layers_created",
            created=result.statistics.layers_created,
            existed=result.statistics.layers_existed,
        )

    # Step 2: Create entities
    for i, entity in enumerate(extraction_result.entities):
        # Progress callback
        if progress_callback:
            progress_callback(i + 1, total_entities)

        # Create the entity
        entity_result = await create_single_entity(entity, call_command)
        result.entity_results.append(entity_result)

        if entity_result.success:
            result.statistics.success_count += 1
            if entity_result.handle:
                result.created_handles.append(entity_result.handle)

            # Update type-specific counters
            et = entity_result.entity_type
            if et == "line":
                result.statistics.lines_created += 1
            elif et == "arc":
                result.statistics.arcs_created += 1
            elif et == "circle":
                result.statistics.circles_created += 1
            elif et in ("mtext", "text"):
                result.statistics.texts_created += 1
            elif et == "block":
                result.statistics.blocks_created += 1
            elif et == "polyline":
                result.statistics.polylines_created += 1
            elif et == "dimension":
                result.statistics.dimensions_created += 1
            else:
                result.statistics.other_created += 1
        else:
            result.statistics.failure_count += 1
            result.failed_entities.append({
                "entity_type": entity_result.entity_type,
                "layer": entity_result.layer,
                "error": entity_result.error,
                "source_element": entity_result.source_element,
            })

    # Step 3: Execute raster commands (if enabled)
    if execute_raster_commands and extraction_result.raster_commands:
        for raster_cmd in extraction_result.raster_commands:
            try:
                # Execute raster command via SendStringToExecute
                cmd_string = raster_cmd.to_command_string()
                raster_result = await call_command(
                    "send_string",
                    {"command": cmd_string, "wait": True}
                )

                if raster_result.get("success", False):
                    result.statistics.raster_commands_executed += 1
                    result.raster_results.append({
                        "tool": raster_cmd.tool,
                        "success": True,
                    })
                else:
                    result.statistics.raster_commands_failed += 1
                    result.raster_results.append({
                        "tool": raster_cmd.tool,
                        "success": False,
                        "error": str(raster_result.get("error", {}).get("message", "")),
                    })

            except Exception as e:
                result.statistics.raster_commands_failed += 1
                result.raster_results.append({
                    "tool": raster_cmd.tool,
                    "success": False,
                    "error": str(e),
                })

    # Determine overall success (>50% entities created)
    result.success = result.success_rate > 0.5

    logger.info(
        "autocad_creation_complete",
        success=result.success,
        success_count=result.statistics.success_count,
        failure_count=result.statistics.failure_count,
        success_rate=f"{result.success_rate:.1%}",
    )

    return result


async def create_entities_batch(
    entities: List[EntityToCreate],
    call_command: Optional[Callable] = None,
    create_layers: bool = True,
) -> AutoCADCreationResult:
    """
    Create a batch of entities in AutoCAD.

    Convenience function for creating entities without a full ExtractionResult.

    Args:
        entities: List of EntityToCreate objects
        call_command: Async function to call AutoCAD commands
        create_layers: Whether to create layers that don't exist

    Returns:
        AutoCADCreationResult with creation results
    """
    # Create a minimal extraction result
    extraction = ExtractionResult(entities=entities)
    return await create_entities_in_autocad(
        extraction,
        call_command=call_command,
        create_layers=create_layers,
    )


def get_entity_type_stats(result: AutoCADCreationResult) -> Dict[str, int]:
    """
    Get statistics grouped by entity type.

    Args:
        result: AutoCADCreationResult from creation process

    Returns:
        Dictionary mapping entity type to count
    """
    return result.statistics.to_dict()["by_type"]


def get_failed_by_type(result: AutoCADCreationResult) -> Dict[str, List[dict]]:
    """
    Get failed entities grouped by type.

    Args:
        result: AutoCADCreationResult from creation process

    Returns:
        Dictionary mapping entity type to list of failed entities
    """
    grouped: Dict[str, List[dict]] = {}

    for failed in result.failed_entities:
        entity_type = failed.get("entity_type", "unknown")
        if entity_type not in grouped:
            grouped[entity_type] = []
        grouped[entity_type].append(failed)

    return grouped
