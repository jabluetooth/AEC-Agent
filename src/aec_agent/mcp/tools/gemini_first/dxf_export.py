"""
DXF Export Module.

Exports extracted entities directly to DXF format without requiring AutoCAD.
This enables offline workflows and standalone vectorization.

Uses ezdxf library for DXF R2018 format generation with full support for:
- Lines, Circles, Arcs, Ellipses
- MTEXT and TEXT
- SPLINE (from Bezier curves)
- Layers with colors and linetypes
- Polylines (LWPOLYLINE)
- Block references (INSERT)

Dependencies:
    pip install ezdxf>=1.0.0
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import math

import structlog

try:
    import ezdxf
    from ezdxf import colors as ezdxf_colors
    from ezdxf.enums import TextEntityAlignment
    EZDXF_AVAILABLE = True
except ImportError:
    EZDXF_AVAILABLE = False

logger = structlog.get_logger(__name__)


def is_ezdxf_available() -> bool:
    """Check if ezdxf is installed."""
    return EZDXF_AVAILABLE


class DXFVersion(str, Enum):
    """Supported DXF versions."""
    R12 = "R12"
    R2000 = "R2000"
    R2004 = "R2004"
    R2007 = "R2007"
    R2010 = "R2010"
    R2013 = "R2013"
    R2018 = "R2018"


class Units(str, Enum):
    """Drawing units."""
    UNITLESS = "Unitless"
    INCHES = "Inches"
    FEET = "Feet"
    MILLIMETERS = "Millimeters"
    CENTIMETERS = "Centimeters"
    METERS = "Meters"


# ezdxf units mapping
UNITS_MAP = {
    Units.UNITLESS: 0,
    Units.INCHES: 1,
    Units.FEET: 2,
    Units.MILLIMETERS: 4,
    Units.CENTIMETERS: 5,
    Units.METERS: 6,
}


# Standard AutoCAD colors (ACI - AutoCAD Color Index)
ACI_COLORS = {
    "red": 1,
    "yellow": 2,
    "green": 3,
    "cyan": 4,
    "blue": 5,
    "magenta": 6,
    "white": 7,
    "gray": 8,
    "grey": 8,
    "black": 0,
}

# Layer color suggestions based on NCS layer names
LAYER_COLORS = {
    "A-WALL": 3,  # Green
    "A-DOOR": 1,  # Red
    "A-GLAZ": 4,  # Cyan
    "A-FLOR": 8,  # Gray
    "E-LITE": 2,  # Yellow
    "E-POWR": 1,  # Red
    "M-DUCT": 6,  # Magenta
    "P-PIPE": 5,  # Blue
    "0": 7,  # White (default layer)
}


@dataclass
class DXFExportConfig:
    """Configuration for DXF export."""
    version: DXFVersion = DXFVersion.R2018
    units: Units = Units.INCHES
    create_layers: bool = True
    include_linetypes: bool = True
    default_text_height: float = 0.125  # inches
    default_text_style: str = "Standard"
    precision: int = 6  # Decimal places for coordinates


@dataclass
class DXFExportResult:
    """Result of DXF export operation."""
    success: bool
    output_path: Optional[str] = None
    entity_count: int = 0
    layer_count: int = 0
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "success": self.success,
            "output_path": self.output_path,
            "entity_count": self.entity_count,
            "layer_count": self.layer_count,
            "errors": self.errors,
            "warnings": self.warnings,
        }


class DXFExporter:
    """
    Exports entities to DXF format.

    Example:
        >>> from aec_agent.mcp.tools.gemini_first.dxf_export import DXFExporter
        >>> exporter = DXFExporter()
        >>> result = exporter.export(entities, "output.dxf")
    """

    def __init__(self, config: Optional[DXFExportConfig] = None):
        if not EZDXF_AVAILABLE:
            raise ImportError("ezdxf is required for DXF export. Install with: pip install ezdxf")

        self.config = config or DXFExportConfig()
        self._doc = None
        self._msp = None
        self._created_layers = set()
        self._created_linetypes = set()

    def _create_document(self) -> None:
        """Create a new DXF document."""
        self._doc = ezdxf.new(dxfversion=self.config.version.value)
        self._msp = self._doc.modelspace()

        # Set units
        self._doc.units = UNITS_MAP.get(self.config.units, 1)

        # Ensure standard linetypes are loaded
        if self.config.include_linetypes:
            self._setup_linetypes()

    def _setup_linetypes(self) -> None:
        """Setup standard linetypes."""
        # Standard linetypes to ensure exist
        standard_linetypes = {
            "CONTINUOUS": [],
            "DASHED": [0.5, -0.25],
            "HIDDEN": [0.25, -0.125],
            "CENTER": [1.25, -0.25, 0.25, -0.25],
            "PHANTOM": [1.25, -0.25, 0.25, -0.25, 0.25, -0.25],
            "DOT": [0.0, -0.25],
            "DASHDOT": [0.5, -0.25, 0.0, -0.25],
        }

        for name, pattern in standard_linetypes.items():
            if name not in self._doc.linetypes:
                try:
                    if pattern:
                        self._doc.linetypes.add(
                            name,
                            pattern=pattern,
                            description=f"{name} linetype"
                        )
                    self._created_linetypes.add(name)
                except Exception as e:
                    logger.debug(f"Linetype {name} already exists or error: {e}")

    def _ensure_layer(self, layer_name: str) -> str:
        """Ensure layer exists, create if needed."""
        if not layer_name:
            layer_name = "0"

        if layer_name not in self._created_layers:
            if layer_name not in self._doc.layers:
                color = LAYER_COLORS.get(layer_name.upper(), 7)
                self._doc.layers.add(layer_name, color=color)
            self._created_layers.add(layer_name)

        return layer_name

    def _add_line(self, entity: Any) -> bool:
        """Add a LINE entity."""
        try:
            props = entity.properties
            start = props.get("start", (0, 0))
            end = props.get("end", (0, 0))
            layer = self._ensure_layer(entity.layer)
            linetype = props.get("linetype", "CONTINUOUS")

            line = self._msp.add_line(
                start=(start[0], start[1]),
                end=(end[0], end[1]),
                dxfattribs={
                    "layer": layer,
                    "linetype": linetype,
                }
            )

            # Set lineweight if specified
            if "lineweight" in props:
                line.dxf.lineweight = int(props["lineweight"] * 100)  # Convert mm to 1/100 mm

            return True
        except Exception as e:
            logger.warning("dxf_add_line_failed", error=str(e))
            return False

    def _add_circle(self, entity: Any) -> bool:
        """Add a CIRCLE entity."""
        try:
            props = entity.properties
            center = props.get("center", (0, 0))
            radius = props.get("radius", 1.0)
            layer = self._ensure_layer(entity.layer)

            self._msp.add_circle(
                center=(center[0], center[1]),
                radius=radius,
                dxfattribs={"layer": layer}
            )
            return True
        except Exception as e:
            logger.warning("dxf_add_circle_failed", error=str(e))
            return False

    def _add_arc(self, entity: Any) -> bool:
        """Add an ARC entity."""
        try:
            props = entity.properties
            center = props.get("center", (0, 0))
            radius = props.get("radius", 1.0)
            start_angle = props.get("start_angle", 0)
            end_angle = props.get("end_angle", 90)
            layer = self._ensure_layer(entity.layer)

            self._msp.add_arc(
                center=(center[0], center[1]),
                radius=radius,
                start_angle=start_angle,
                end_angle=end_angle,
                dxfattribs={"layer": layer}
            )
            return True
        except Exception as e:
            logger.warning("dxf_add_arc_failed", error=str(e))
            return False

    def _add_ellipse(self, entity: Any) -> bool:
        """Add an ELLIPSE entity."""
        try:
            props = entity.properties
            center = props.get("center", (0, 0))
            major_axis = props.get("major_axis", (1.0, 0.0))
            ratio = props.get("ratio", 0.5)  # Minor/major ratio
            start_param = props.get("start_param", 0)
            end_param = props.get("end_param", math.tau)  # Full ellipse
            layer = self._ensure_layer(entity.layer)

            self._msp.add_ellipse(
                center=(center[0], center[1], 0),
                major_axis=(major_axis[0], major_axis[1], 0),
                ratio=ratio,
                start_param=start_param,
                end_param=end_param,
                dxfattribs={"layer": layer}
            )
            return True
        except Exception as e:
            logger.warning("dxf_add_ellipse_failed", error=str(e))
            return False

    def _add_mtext(self, entity: Any) -> bool:
        """Add an MTEXT entity."""
        try:
            props = entity.properties
            content = props.get("content", props.get("text", ""))
            position = props.get("position", props.get("insertion_point", (0, 0)))
            height = props.get("height", self.config.default_text_height)
            rotation = props.get("rotation", 0)
            layer = self._ensure_layer(entity.layer)

            mtext = self._msp.add_mtext(
                content,
                dxfattribs={
                    "layer": layer,
                    "char_height": height,
                    "rotation": rotation,
                }
            )
            mtext.set_location((position[0], position[1]))
            return True
        except Exception as e:
            logger.warning("dxf_add_mtext_failed", error=str(e))
            return False

    def _add_text(self, entity: Any) -> bool:
        """Add a TEXT entity."""
        try:
            props = entity.properties
            content = props.get("content", props.get("text", ""))
            position = props.get("position", props.get("insertion_point", (0, 0)))
            height = props.get("height", self.config.default_text_height)
            rotation = props.get("rotation", 0)
            layer = self._ensure_layer(entity.layer)

            self._msp.add_text(
                content,
                height=height,
                rotation=rotation,
                dxfattribs={
                    "layer": layer,
                    "insert": (position[0], position[1]),
                }
            )
            return True
        except Exception as e:
            logger.warning("dxf_add_text_failed", error=str(e))
            return False

    def _add_spline(self, entity: Any) -> bool:
        """Add a SPLINE entity (for Bezier curves)."""
        try:
            props = entity.properties
            control_points = props.get("control_points", [])
            fit_points = props.get("fit_points", [])
            degree = props.get("degree", 3)
            layer = self._ensure_layer(entity.layer)

            if fit_points:
                # Create spline from fit points
                pts = [(p[0], p[1], 0) for p in fit_points]
                self._msp.add_spline(
                    fit_points=pts,
                    degree=degree,
                    dxfattribs={"layer": layer}
                )
            elif control_points:
                # Create spline from control points
                pts = [(p[0], p[1], 0) for p in control_points]
                self._msp.add_spline(
                    control_points=pts,
                    degree=degree,
                    dxfattribs={"layer": layer}
                )
            else:
                return False

            return True
        except Exception as e:
            logger.warning("dxf_add_spline_failed", error=str(e))
            return False

    def _add_polyline(self, entity: Any) -> bool:
        """Add an LWPOLYLINE entity."""
        try:
            props = entity.properties
            points = props.get("points", props.get("vertices", []))
            is_closed = props.get("closed", props.get("is_closed", False))
            layer = self._ensure_layer(entity.layer)

            if len(points) < 2:
                return False

            # Convert to 2D points
            pts = [(p[0], p[1]) for p in points]

            self._msp.add_lwpolyline(
                pts,
                close=is_closed,
                dxfattribs={"layer": layer}
            )
            return True
        except Exception as e:
            logger.warning("dxf_add_polyline_failed", error=str(e))
            return False

    def _add_block_reference(self, entity: Any) -> bool:
        """Add a block reference (INSERT)."""
        try:
            props = entity.properties
            block_name = props.get("block_name", props.get("name", "UNKNOWN"))
            position = props.get("position", props.get("insertion_point", (0, 0)))
            scale = props.get("scale", 1.0)
            rotation = props.get("rotation", 0)
            layer = self._ensure_layer(entity.layer)

            # Check if block exists, create placeholder if not
            if block_name not in self._doc.blocks:
                # Create a simple placeholder block (cross symbol)
                block = self._doc.blocks.new(name=block_name)
                size = 0.5
                block.add_line((-size, 0), (size, 0))
                block.add_line((0, -size), (0, size))

            self._msp.add_blockref(
                block_name,
                insert=(position[0], position[1]),
                dxfattribs={
                    "layer": layer,
                    "xscale": scale,
                    "yscale": scale,
                    "rotation": rotation,
                }
            )
            return True
        except Exception as e:
            logger.warning("dxf_add_block_failed", error=str(e))
            return False

    def _add_dimension(self, entity: Any) -> bool:
        """Add a linear dimension."""
        try:
            props = entity.properties
            start = props.get("start", (0, 0))
            end = props.get("end", (0, 0))
            text_position = props.get("text_position")
            layer = self._ensure_layer(entity.layer)

            # Calculate text midpoint if not specified
            if text_position is None:
                text_position = (
                    (start[0] + end[0]) / 2,
                    (start[1] + end[1]) / 2 + 0.5  # Offset above
                )

            self._msp.add_linear_dim(
                base=(start[0], start[1]),
                p1=(start[0], start[1]),
                p2=(end[0], end[1]),
                dimstyle="Standard",
                dxfattribs={"layer": layer}
            ).render()
            return True
        except Exception as e:
            logger.warning("dxf_add_dimension_failed", error=str(e))
            return False

    def _add_hatch(self, entity: Any) -> bool:
        """Add a HATCH entity."""
        try:
            props = entity.properties
            pattern = props.get("pattern", "SOLID")
            boundary_points = props.get("boundary", props.get("points", []))
            layer = self._ensure_layer(entity.layer)

            if len(boundary_points) < 3:
                return False

            hatch = self._msp.add_hatch(dxfattribs={"layer": layer})
            hatch.set_pattern_fill(pattern, scale=1.0)

            # Add boundary path
            pts = [(p[0], p[1]) for p in boundary_points]
            hatch.paths.add_polyline_path(pts, is_closed=True)

            return True
        except Exception as e:
            logger.warning("dxf_add_hatch_failed", error=str(e))
            return False

    def export(
        self,
        entities: List[Any],
        output_path: Union[str, Path],
    ) -> DXFExportResult:
        """
        Export entities to a DXF file.

        Args:
            entities: List of EntityToCreate objects
            output_path: Path for output DXF file

        Returns:
            DXFExportResult with success status and statistics
        """
        result = DXFExportResult(success=False)
        output_path = Path(output_path)

        try:
            self._create_document()
            entity_count = 0
            errors = []

            for entity in entities:
                entity_type = getattr(entity, 'entity_type', None)
                if entity_type is None:
                    continue

                # Get the type value (handle both enum and string)
                type_value = entity_type.value if hasattr(entity_type, 'value') else str(entity_type)
                type_lower = type_value.lower()

                success = False

                if type_lower == "line":
                    success = self._add_line(entity)
                elif type_lower == "circle":
                    success = self._add_circle(entity)
                elif type_lower == "arc":
                    success = self._add_arc(entity)
                elif type_lower == "ellipse":
                    success = self._add_ellipse(entity)
                elif type_lower in ("mtext", "text"):
                    success = self._add_mtext(entity)
                elif type_lower == "spline":
                    success = self._add_spline(entity)
                elif type_lower in ("polyline", "lwpolyline"):
                    success = self._add_polyline(entity)
                elif type_lower in ("block", "insert"):
                    success = self._add_block_reference(entity)
                elif type_lower == "dimension":
                    success = self._add_dimension(entity)
                elif type_lower == "hatch":
                    success = self._add_hatch(entity)
                else:
                    result.warnings.append(f"Unsupported entity type: {type_value}")
                    continue

                if success:
                    entity_count += 1
                else:
                    errors.append(f"Failed to add {type_value} entity")

            # Save the document
            output_path.parent.mkdir(parents=True, exist_ok=True)
            self._doc.saveas(str(output_path))

            result.success = True
            result.output_path = str(output_path)
            result.entity_count = entity_count
            result.layer_count = len(self._created_layers)
            result.errors = errors

            logger.info(
                "dxf_export_complete",
                output_path=str(output_path),
                entity_count=entity_count,
                layer_count=len(self._created_layers),
            )

        except Exception as e:
            result.errors.append(str(e))
            logger.exception("dxf_export_failed", error=str(e))

        return result


def export_to_dxf(
    entities: List[Any],
    output_path: Union[str, Path],
    version: str = "R2018",
    units: str = "Inches",
    config: Optional[DXFExportConfig] = None,
) -> DXFExportResult:
    """
    Export entities to DXF file without AutoCAD.

    This is a convenience function for simple exports.

    Args:
        entities: List of EntityToCreate objects
        output_path: Path for output DXF file
        version: DXF version (R12, R2000, R2004, R2007, R2010, R2013, R2018)
        units: Drawing units (Inches, Feet, Millimeters, Centimeters, Meters)
        config: Optional full configuration

    Returns:
        DXFExportResult with success status and statistics

    Example:
        >>> from aec_agent.mcp.tools.gemini_first import export_to_dxf
        >>> result = export_to_dxf(entities, "output.dxf", units="Inches")
        >>> if result.success:
        ...     print(f"Exported {result.entity_count} entities")
    """
    if not EZDXF_AVAILABLE:
        return DXFExportResult(
            success=False,
            errors=["ezdxf is not installed. Run: pip install ezdxf"]
        )

    if config is None:
        config = DXFExportConfig(
            version=DXFVersion(version),
            units=Units(units) if units in [u.value for u in Units] else Units.INCHES,
        )

    exporter = DXFExporter(config)
    return exporter.export(entities, output_path)
