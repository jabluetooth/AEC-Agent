"""
Geometry Classification for AEC Drawings.

Pattern-based classification of raw geometry (lines, polylines, circles)
into meaningful AEC elements like walls, ducts, pipes, conduits, etc.

This is Phase D: Geometry Intelligence in the Semantic Intelligence Pipeline.

The classifier uses:
1. Geometric patterns (parallel lines, spacing, orthogonality)
2. Drawing context (floor plan vs MEP plan)
3. Nearby symbols (diffuser suggests duct, valve suggests pipe)
4. Nearby text (size annotations, system labels)

Classification Rules:
- WALLS: Parallel line pairs with 4-8" spacing, orthogonal, floor plans
- DUCTS: Parallel lines connected to diffuser/grille symbols
- PIPES: Single lines connecting valve/fitting symbols
- CONDUITS: Lines connecting electrical outlets/panels
- DIMENSION_LINE: Lines with arrow endpoints and numeric text midpoint
- LEADER: Line ending at symbol/text with arrow/dot at other end
- HIDDEN_LINE: Dashed linetype (centerlines, hidden)
"""

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import structlog

from aec_agent.mcp.tools.document_classifier import DrawingType
from aec_agent.mcp.tools.image_vectorizer import (
    DetectedCircle,
    DetectedLine,
    DetectedPolyline,
)

logger = structlog.get_logger(__name__)


class GeometryType(str, Enum):
    """Type of classified geometry element."""
    # Architectural elements
    WALL = "wall"
    WALL_CENTERLINE = "wall_centerline"
    DOOR_SWING = "door_swing"
    WINDOW = "window"
    STAIR = "stair"

    # MEP elements - Mechanical
    DUCT = "duct"
    DUCT_RECTANGULAR = "duct_rectangular"
    DUCT_ROUND = "duct_round"
    DUCT_CENTERLINE = "duct_centerline"

    # MEP elements - Plumbing
    PIPE = "pipe"
    PIPE_CENTERLINE = "pipe_centerline"
    DRAIN = "drain"

    # MEP elements - Electrical
    CONDUIT = "conduit"
    WIRE_RUN = "wire_run"
    BUS_BAR = "bus_bar"

    # MEP elements - Fire
    FIRE_ALARM_WIRE = "fire_alarm_wire"
    SPRINKLER_PIPE = "sprinkler_pipe"

    # Annotation elements
    DIMENSION_LINE = "dimension"
    LEADER = "leader"
    SECTION_CUT = "section_cut"
    BREAK_LINE = "break_line"
    GRID_LINE = "grid"
    MATCH_LINE = "match_line"

    # Linetype-based
    HIDDEN_LINE = "hidden"
    CENTERLINE = "centerline"
    PROPERTY_LINE = "property"
    SETBACK_LINE = "setback"

    # Other
    BOUNDARY = "boundary"
    ROOM_BOUNDARY = "room_boundary"
    UNKNOWN = "unknown"


class GeometrySystem(str, Enum):
    """MEP system classification for geometry."""
    # HVAC systems
    SUPPLY_AIR = "supply_air"
    RETURN_AIR = "return_air"
    EXHAUST_AIR = "exhaust_air"
    OUTSIDE_AIR = "outside_air"

    # Plumbing systems
    DOMESTIC_COLD = "domestic_cold"
    DOMESTIC_HOT = "domestic_hot"
    STORM_DRAIN = "storm_drain"
    SANITARY = "sanitary"
    VENT = "vent"
    NATURAL_GAS = "natural_gas"

    # Electrical systems
    POWER = "power"
    LIGHTING = "lighting"
    LOW_VOLTAGE = "low_voltage"

    # Fire systems
    FIRE_ALARM = "fire_alarm"
    FIRE_PROTECTION = "fire_protection"

    UNKNOWN = "unknown"


@dataclass
class GeometryClassification:
    """Classification result for a geometry element."""
    geometry_type: GeometryType
    system: GeometrySystem = GeometrySystem.UNKNOWN
    confidence: float = 0.0
    size: str | None = None  # e.g., "12x8", "2\"", "3/4"
    layer_suggestion: str | None = None
    reasoning: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ClassifiedLine(DetectedLine):
    """A line with classification information."""
    classification: GeometryClassification = field(
        default_factory=lambda: GeometryClassification(GeometryType.UNKNOWN)
    )
    merged_from: list[int] = field(default_factory=list)  # Original line indices
    pair_index: int | None = None  # Index of parallel pair partner


@dataclass
class ClassifiedPolyline(DetectedPolyline):
    """A polyline with classification information."""
    classification: GeometryClassification = field(
        default_factory=lambda: GeometryClassification(GeometryType.UNKNOWN)
    )


@dataclass
class ClassifiedCircle(DetectedCircle):
    """A circle with classification information."""
    classification: GeometryClassification = field(
        default_factory=lambda: GeometryClassification(GeometryType.UNKNOWN)
    )


@dataclass
class ParallelLinePair:
    """A pair of parallel lines detected as potential wall/duct/pipe boundary."""
    line1_index: int
    line2_index: int
    spacing: float  # Distance between lines (in drawing units)
    centerline: DetectedLine  # Computed centerline
    length: float  # Average length of the pair
    is_orthogonal: bool  # True if horizontal or vertical
    angle: float  # Angle in degrees


@dataclass
class GeometryClassifierResult:
    """Result from geometry classification."""
    classified_lines: list[ClassifiedLine] = field(default_factory=list)
    classified_polylines: list[ClassifiedPolyline] = field(default_factory=list)
    classified_circles: list[ClassifiedCircle] = field(default_factory=list)
    parallel_pairs: list[ParallelLinePair] = field(default_factory=list)
    wall_centerlines: list[DetectedLine] = field(default_factory=list)
    duct_boundaries: list[ParallelLinePair] = field(default_factory=list)
    pipe_runs: list[ClassifiedLine] = field(default_factory=list)
    dimension_lines: list[ClassifiedLine] = field(default_factory=list)
    leaders: list[ClassifiedLine] = field(default_factory=list)
    statistics: dict[str, Any] = field(default_factory=dict)


class GeometryClassifier:
    """
    Pattern-based geometry classifier for AEC drawings.

    Uses geometric analysis and drawing context to classify raw
    lines, polylines, and circles into meaningful AEC elements.

    Args:
        scale: Drawing scale (units per pixel, default 1.0)
        wall_thickness_min: Minimum wall thickness in drawing units (default 3")
        wall_thickness_max: Maximum wall thickness in drawing units (default 12")
        duct_size_min: Minimum duct dimension (default 4")
        duct_size_max: Maximum duct dimension (default 48")
        parallel_tolerance: Angle tolerance for parallel detection (degrees)
        proximity_tolerance: Distance tolerance for endpoint proximity
        orthogonal_tolerance: Angle tolerance for orthogonal detection (degrees)

    Example:
        >>> classifier = GeometryClassifier()
        >>> result = await classifier.classify(
        ...     lines=detected_lines,
        ...     polylines=detected_polylines,
        ...     circles=detected_circles,
        ...     drawing_type=DrawingType.FLOOR_PLAN,
        ... )
        >>> for line in result.classified_lines:
        ...     print(f"{line.classification.geometry_type}: {line.start} to {line.end}")
    """

    # Standard wall thicknesses (in inches converted to drawing units)
    WALL_THICKNESSES = [4, 5, 6, 8, 10, 12]  # Common wall widths in inches

    # Standard duct sizes (rectangular, in inches)
    DUCT_WIDTHS = [6, 8, 10, 12, 14, 16, 18, 20, 24, 30, 36, 42, 48]
    DUCT_HEIGHTS = [6, 8, 10, 12, 14, 16, 18, 20, 24]

    # Standard pipe sizes (in inches)
    PIPE_SIZES = [0.5, 0.75, 1, 1.25, 1.5, 2, 2.5, 3, 4, 6, 8, 10, 12]

    def __init__(
        self,
        scale: float = 1.0,
        wall_thickness_min: float = 3.0,
        wall_thickness_max: float = 12.0,
        duct_size_min: float = 4.0,
        duct_size_max: float = 48.0,
        parallel_tolerance: float = 3.0,
        proximity_tolerance: float = 10.0,
        orthogonal_tolerance: float = 5.0,
    ):
        self.scale = scale
        self.wall_thickness_min = wall_thickness_min
        self.wall_thickness_max = wall_thickness_max
        self.duct_size_min = duct_size_min
        self.duct_size_max = duct_size_max
        self.parallel_tolerance = parallel_tolerance
        self.proximity_tolerance = proximity_tolerance
        self.orthogonal_tolerance = orthogonal_tolerance

    async def classify(
        self,
        lines: list[DetectedLine],
        polylines: list[DetectedPolyline] | None = None,
        circles: list[DetectedCircle] | None = None,
        drawing_type: DrawingType = DrawingType.UNKNOWN,
        nearby_symbols: list[Any] | None = None,  # List[SmartSymbol]
        nearby_text: list[Any] | None = None,  # List[ParsedAnnotation]
    ) -> GeometryClassifierResult:
        """
        Classify geometry elements into AEC categories.

        Args:
            lines: Detected lines from vectorization
            polylines: Detected polylines/contours
            circles: Detected circles
            drawing_type: Type of drawing (affects classification rules)
            nearby_symbols: Detected symbols for context
            nearby_text: Parsed text annotations for context

        Returns:
            GeometryClassifierResult with classified elements
        """
        polylines = polylines or []
        circles = circles or []
        nearby_symbols = nearby_symbols or []
        nearby_text = nearby_text or []

        result = GeometryClassifierResult()

        logger.info(
            "Classifying geometry",
            lines=len(lines),
            polylines=len(polylines),
            circles=len(circles),
            drawing_type=drawing_type.value,
        )

        # Step 1: Classify lines by linetype (dashed, hidden, etc.)
        classified_lines = self._classify_by_linetype(lines)

        # Step 2: Find parallel line pairs (potential walls, ducts)
        parallel_pairs = self._find_parallel_pairs(classified_lines)
        result.parallel_pairs = parallel_pairs

        # Step 3: Classify parallel pairs based on drawing type
        if drawing_type in [DrawingType.FLOOR_PLAN, DrawingType.UNKNOWN]:
            wall_pairs = self._classify_wall_pairs(parallel_pairs)
            result.wall_centerlines = [p.centerline for p in wall_pairs]
            for pair in wall_pairs:
                classified_lines[pair.line1_index].classification = GeometryClassification(
                    geometry_type=GeometryType.WALL,
                    confidence=0.85,
                    size=f"{pair.spacing:.1f}\"",
                    layer_suggestion="A-WALL",
                    reasoning="Parallel lines with wall-like spacing",
                )
                classified_lines[pair.line2_index].classification = GeometryClassification(
                    geometry_type=GeometryType.WALL,
                    confidence=0.85,
                    size=f"{pair.spacing:.1f}\"",
                    layer_suggestion="A-WALL",
                    reasoning="Parallel lines with wall-like spacing",
                )

        if drawing_type in [DrawingType.HVAC_PLAN, DrawingType.MEP_PLAN, DrawingType.UNKNOWN]:
            duct_pairs = self._classify_duct_pairs(parallel_pairs, nearby_symbols)
            result.duct_boundaries = duct_pairs
            for pair in duct_pairs:
                classified_lines[pair.line1_index].classification = GeometryClassification(
                    geometry_type=GeometryType.DUCT_RECTANGULAR,
                    system=GeometrySystem.SUPPLY_AIR,  # Default, could be inferred
                    confidence=0.80,
                    size=f"{pair.spacing:.0f}\"",
                    layer_suggestion="M-DUCT-SUPP",
                    reasoning="Parallel lines with duct dimensions near diffuser symbols",
                )
                classified_lines[pair.line2_index].classification = GeometryClassification(
                    geometry_type=GeometryType.DUCT_RECTANGULAR,
                    system=GeometrySystem.SUPPLY_AIR,
                    confidence=0.80,
                    size=f"{pair.spacing:.0f}\"",
                    layer_suggestion="M-DUCT-SUPP",
                    reasoning="Parallel lines with duct dimensions near diffuser symbols",
                )

        # Step 4: Classify single lines (pipes, conduits)
        if drawing_type in [DrawingType.PLUMBING_PLAN, DrawingType.MEP_PLAN, DrawingType.UNKNOWN]:
            pipe_lines = self._classify_pipe_lines(
                classified_lines, nearby_symbols, nearby_text
            )
            result.pipe_runs = pipe_lines

        if drawing_type in [DrawingType.ELECTRICAL_PLAN, DrawingType.FIRE_ALARM, DrawingType.MEP_PLAN]:
            self._classify_conduit_lines(classified_lines, nearby_symbols)

        # Step 5: Classify dimension lines and leaders
        dim_lines = self._classify_dimension_lines(classified_lines, nearby_text)
        result.dimension_lines = dim_lines

        leaders = self._classify_leaders(classified_lines, nearby_symbols, nearby_text)
        result.leaders = leaders

        # Step 6: Classify arcs (door swings, curved ducts)
        for polyline in polylines:
            classification = self._classify_polyline(polyline, drawing_type, nearby_symbols)
            result.classified_polylines.append(
                ClassifiedPolyline(
                    points=polyline.points,
                    closed=polyline.closed,
                    bulges=polyline.bulges,
                    classification=classification,
                )
            )

        # Step 7: Classify circles
        for circle in circles:
            classification = self._classify_circle(circle, drawing_type, nearby_symbols)
            result.classified_circles.append(
                ClassifiedCircle(
                    center=circle.center,
                    radius=circle.radius,
                    classification=classification,
                )
            )

        result.classified_lines = classified_lines

        # Compute statistics
        result.statistics = self._compute_statistics(result)

        logger.info(
            "Geometry classification complete",
            walls=len(result.wall_centerlines),
            ducts=len(result.duct_boundaries),
            pipes=len(result.pipe_runs),
            dimensions=len(result.dimension_lines),
            leaders=len(result.leaders),
        )

        return result

    def _classify_by_linetype(
        self, lines: list[DetectedLine]
    ) -> list[ClassifiedLine]:
        """Classify lines by their linetype (continuous, dashed, etc.)."""
        classified = []

        for line in lines:
            classification = GeometryClassification(
                geometry_type=GeometryType.UNKNOWN,
                confidence=0.5,
            )

            # Check linetype
            if line.linetype in ["DASHED", "HIDDEN", "HIDDEN2"]:
                classification = GeometryClassification(
                    geometry_type=GeometryType.HIDDEN_LINE,
                    confidence=0.9,
                    layer_suggestion="HIDDEN",
                    reasoning=f"Linetype is {line.linetype}",
                )
            elif line.linetype in ["CENTER", "CENTER2", "CENTERLINE"]:
                classification = GeometryClassification(
                    geometry_type=GeometryType.CENTERLINE,
                    confidence=0.9,
                    layer_suggestion="CENTER",
                    reasoning=f"Linetype is {line.linetype}",
                )
            elif line.linetype in ["PHANTOM", "PHANTOM2"]:
                classification = GeometryClassification(
                    geometry_type=GeometryType.PROPERTY_LINE,
                    confidence=0.7,
                    layer_suggestion="PROPERTY",
                    reasoning=f"Linetype is {line.linetype}",
                )

            classified.append(
                ClassifiedLine(
                    start=line.start,
                    end=line.end,
                    linetype=line.linetype,
                    classification=classification,
                )
            )

        return classified

    def _find_parallel_pairs(
        self, lines: list[ClassifiedLine]
    ) -> list[ParallelLinePair]:
        """Find pairs of parallel lines that could be walls, ducts, or pipes."""
        pairs = []
        n = len(lines)

        for i in range(n):
            for j in range(i + 1, n):
                line1 = lines[i]
                line2 = lines[j]

                # Skip if already classified as non-geometry (e.g., dimension)
                if line1.classification.geometry_type in [
                    GeometryType.DIMENSION_LINE,
                    GeometryType.LEADER,
                    GeometryType.HIDDEN_LINE,
                    GeometryType.CENTERLINE,
                ]:
                    continue

                if line2.classification.geometry_type in [
                    GeometryType.DIMENSION_LINE,
                    GeometryType.LEADER,
                    GeometryType.HIDDEN_LINE,
                    GeometryType.CENTERLINE,
                ]:
                    continue

                # Check if parallel
                angle1 = self._line_angle(line1)
                angle2 = self._line_angle(line2)

                angle_diff = abs(angle1 - angle2)
                if angle_diff > 180:
                    angle_diff = 360 - angle_diff

                if angle_diff > self.parallel_tolerance and abs(angle_diff - 180) > self.parallel_tolerance:
                    continue

                # Check spacing (perpendicular distance)
                spacing = self._perpendicular_distance(line1, line2)
                if spacing < 1.0 or spacing > self.duct_size_max * 1.5:
                    continue

                # Check overlap (lines should be adjacent, not end-to-end)
                if not self._lines_overlap(line1, line2):
                    continue

                # Compute centerline
                centerline = self._compute_centerline(line1, line2)

                # Check if orthogonal
                is_orthogonal = (
                    abs(angle1 % 90) < self.orthogonal_tolerance
                    or abs(angle1 % 90 - 90) < self.orthogonal_tolerance
                )

                pairs.append(
                    ParallelLinePair(
                        line1_index=i,
                        line2_index=j,
                        spacing=spacing,
                        centerline=centerline,
                        length=(self._line_length(line1) + self._line_length(line2)) / 2,
                        is_orthogonal=is_orthogonal,
                        angle=angle1,
                    )
                )

        return pairs

    def _classify_wall_pairs(
        self, pairs: list[ParallelLinePair]
    ) -> list[ParallelLinePair]:
        """Classify parallel pairs as walls based on spacing."""
        wall_pairs = []

        for pair in pairs:
            # Check if spacing matches wall thickness
            if self.wall_thickness_min <= pair.spacing <= self.wall_thickness_max:
                # Prefer orthogonal pairs
                if pair.is_orthogonal or pair.length > 100:  # Long walls may be angled
                    wall_pairs.append(pair)

        return wall_pairs

    def _classify_duct_pairs(
        self,
        pairs: list[ParallelLinePair],
        nearby_symbols: list[Any],
    ) -> list[ParallelLinePair]:
        """Classify parallel pairs as ducts based on spacing and nearby symbols."""
        duct_pairs = []

        # Get diffuser/grille positions
        diffuser_positions = []
        for symbol in nearby_symbols:
            if hasattr(symbol, 'category') and symbol.category == "mechanical":
                if hasattr(symbol, 'type') and symbol.type in ["diffuser", "grille", "register"]:
                    if hasattr(symbol, 'position'):
                        diffuser_positions.append(symbol.position)

        for pair in pairs:
            # Check if spacing matches duct sizes
            if self.duct_size_min <= pair.spacing <= self.duct_size_max:
                # Bonus: check if near a diffuser
                near_diffuser = False
                if diffuser_positions:
                    centerline_midpoint = (
                        (pair.centerline.start[0] + pair.centerline.end[0]) / 2,
                        (pair.centerline.start[1] + pair.centerline.end[1]) / 2,
                    )
                    for pos in diffuser_positions:
                        dist = self._point_distance(centerline_midpoint, pos)
                        if dist < 100:  # Within 100 units
                            near_diffuser = True
                            break

                # Ducts are typically orthogonal and larger than walls
                if pair.is_orthogonal and pair.spacing >= self.duct_size_min:
                    duct_pairs.append(pair)

        return duct_pairs

    def _classify_pipe_lines(
        self,
        lines: list[ClassifiedLine],
        nearby_symbols: list[Any],
        nearby_text: list[Any],
    ) -> list[ClassifiedLine]:
        """Classify single lines as pipes based on connected symbols."""
        pipe_lines = []

        # Get valve/fitting positions
        valve_positions = []
        for symbol in nearby_symbols:
            if hasattr(symbol, 'category') and symbol.category == "plumbing":
                if hasattr(symbol, 'type') and symbol.type in ["valve", "fitting", "fixture"]:
                    if hasattr(symbol, 'position'):
                        valve_positions.append(symbol.position)

        for line in lines:
            if line.classification.geometry_type != GeometryType.UNKNOWN:
                continue

            # Check if endpoints are near valves
            start_near_valve = False
            end_near_valve = False

            for pos in valve_positions:
                if self._point_distance(line.start, pos) < self.proximity_tolerance:
                    start_near_valve = True
                if self._point_distance(line.end, pos) < self.proximity_tolerance:
                    end_near_valve = True

            if start_near_valve or end_near_valve:
                # Look for pipe size annotation nearby
                pipe_size = self._find_nearby_size_text(line, nearby_text)

                line.classification = GeometryClassification(
                    geometry_type=GeometryType.PIPE,
                    system=GeometrySystem.UNKNOWN,
                    confidence=0.75 if (start_near_valve and end_near_valve) else 0.60,
                    size=pipe_size,
                    layer_suggestion="P-PIPE",
                    reasoning="Line connects to valve/fitting symbols",
                )
                pipe_lines.append(line)

        return pipe_lines

    def _classify_conduit_lines(
        self,
        lines: list[ClassifiedLine],
        nearby_symbols: list[Any],
    ) -> None:
        """Classify single lines as conduits based on connected symbols."""
        # Get outlet/switch/panel positions
        electrical_positions = []
        for symbol in nearby_symbols:
            if hasattr(symbol, 'category') and symbol.category == "electrical":
                if hasattr(symbol, 'position'):
                    electrical_positions.append(symbol.position)

        for line in lines:
            if line.classification.geometry_type != GeometryType.UNKNOWN:
                continue

            # Check if endpoints are near electrical symbols
            start_near = False
            end_near = False

            for pos in electrical_positions:
                if self._point_distance(line.start, pos) < self.proximity_tolerance:
                    start_near = True
                if self._point_distance(line.end, pos) < self.proximity_tolerance:
                    end_near = True

            if start_near or end_near:
                line.classification = GeometryClassification(
                    geometry_type=GeometryType.CONDUIT,
                    system=GeometrySystem.POWER,
                    confidence=0.70 if (start_near and end_near) else 0.55,
                    layer_suggestion="E-POWR-CIRC",
                    reasoning="Line connects to electrical symbols",
                )

    def _classify_dimension_lines(
        self,
        lines: list[ClassifiedLine],
        nearby_text: list[Any],
    ) -> list[ClassifiedLine]:
        """Classify lines as dimension lines based on text and arrow patterns."""
        dim_lines = []

        for line in lines:
            if line.classification.geometry_type != GeometryType.UNKNOWN:
                continue

            # Check for numeric text near midpoint
            midpoint = (
                (line.start[0] + line.end[0]) / 2,
                (line.start[1] + line.end[1]) / 2,
            )

            has_numeric_text = False
            for text in nearby_text:
                if hasattr(text, 'position') and hasattr(text, 'text'):
                    text_pos = text.position
                    dist = self._point_distance(midpoint, text_pos)
                    if dist < 30:  # Near midpoint
                        # Check if text looks like a dimension
                        if self._is_dimension_text(text.text):
                            has_numeric_text = True
                            break

            if has_numeric_text:
                line.classification = GeometryClassification(
                    geometry_type=GeometryType.DIMENSION_LINE,
                    confidence=0.85,
                    layer_suggestion="DIMS",
                    reasoning="Line with numeric text at midpoint",
                )
                dim_lines.append(line)

        return dim_lines

    def _classify_leaders(
        self,
        lines: list[ClassifiedLine],
        nearby_symbols: list[Any],
        nearby_text: list[Any],
    ) -> list[ClassifiedLine]:
        """Classify lines as leaders based on endpoint patterns."""
        leaders = []

        for line in lines:
            if line.classification.geometry_type != GeometryType.UNKNOWN:
                continue

            # Check if one endpoint is near text and other is near a symbol
            start_near_text = False
            end_near_symbol = False

            for text in nearby_text:
                if hasattr(text, 'position'):
                    if self._point_distance(line.start, text.position) < 20:
                        start_near_text = True
                    if self._point_distance(line.end, text.position) < 20:
                        start_near_text = True

            for symbol in nearby_symbols:
                if hasattr(symbol, 'position'):
                    if self._point_distance(line.start, symbol.position) < 30:
                        end_near_symbol = True
                    if self._point_distance(line.end, symbol.position) < 30:
                        end_near_symbol = True

            if start_near_text and end_near_symbol:
                line.classification = GeometryClassification(
                    geometry_type=GeometryType.LEADER,
                    confidence=0.80,
                    layer_suggestion="ANNO-LEAD",
                    reasoning="Line connecting text to symbol",
                )
                leaders.append(line)

        return leaders

    def _classify_polyline(
        self,
        polyline: DetectedPolyline,
        drawing_type: DrawingType,
        nearby_symbols: list[Any],
    ) -> GeometryClassification:
        """Classify a polyline based on shape and context."""
        # Check for door swing (90-degree arc)
        if self._is_door_swing(polyline, nearby_symbols):
            return GeometryClassification(
                geometry_type=GeometryType.DOOR_SWING,
                confidence=0.85,
                layer_suggestion="A-DOOR",
                reasoning="Quarter-circle arc near door symbol",
            )

        # Check for room boundary (closed polyline)
        if polyline.closed and len(polyline.points) >= 4:
            if drawing_type in [DrawingType.FLOOR_PLAN, DrawingType.MEP_PLAN]:
                return GeometryClassification(
                    geometry_type=GeometryType.ROOM_BOUNDARY,
                    confidence=0.70,
                    layer_suggestion="A-AREA-ROOM",
                    reasoning="Closed polyline in floor plan",
                )

        return GeometryClassification(
            geometry_type=GeometryType.UNKNOWN,
            confidence=0.0,
        )

    def _classify_circle(
        self,
        circle: DetectedCircle,
        drawing_type: DrawingType,
        nearby_symbols: list[Any],
    ) -> GeometryClassification:
        """Classify a circle based on size and context."""
        # Small circles might be bubble markers or symbols
        if circle.radius < 10:
            return GeometryClassification(
                geometry_type=GeometryType.UNKNOWN,
                confidence=0.3,
                reasoning="Small circle - could be symbol or marker",
            )

        # Medium circles in HVAC might be round ducts
        if drawing_type in [DrawingType.HVAC_PLAN, DrawingType.MEP_PLAN]:
            if 3 <= circle.radius <= 24:  # 6" to 48" diameter
                return GeometryClassification(
                    geometry_type=GeometryType.DUCT_ROUND,
                    system=GeometrySystem.SUPPLY_AIR,
                    confidence=0.65,
                    size=f"{circle.radius * 2:.0f}\"",
                    layer_suggestion="M-DUCT-SUPP",
                    reasoning="Circle with duct-like diameter",
                )

        return GeometryClassification(
            geometry_type=GeometryType.UNKNOWN,
            confidence=0.0,
        )

    def _is_door_swing(
        self,
        polyline: DetectedPolyline,
        nearby_symbols: list[Any],
    ) -> bool:
        """Check if polyline is a door swing arc."""
        # Door swings are quarter circles (90 degrees)
        if len(polyline.points) < 3:
            return False

        # Check if it's an arc (has bulges)
        if not polyline.bulges or all(b == 0 for b in polyline.bulges):
            # No bulges, not an arc
            return False

        # Check if near a door symbol
        for symbol in nearby_symbols:
            if hasattr(symbol, 'type') and symbol.type == "door":
                if hasattr(symbol, 'position'):
                    # Check if polyline start or end is near door
                    for pt in [polyline.points[0], polyline.points[-1]]:
                        if self._point_distance(pt, symbol.position) < 50:
                            return True

        return False

    def _find_nearby_size_text(
        self,
        line: ClassifiedLine,
        nearby_text: list[Any],
    ) -> str | None:
        """Find size annotation text near a line."""
        midpoint = (
            (line.start[0] + line.end[0]) / 2,
            (line.start[1] + line.end[1]) / 2,
        )

        for text in nearby_text:
            if hasattr(text, 'position') and hasattr(text, 'text'):
                dist = self._point_distance(midpoint, text.position)
                if dist < 50:
                    # Check if text looks like a pipe size
                    if self._is_pipe_size_text(text.text):
                        return text.text

        return None

    def _is_dimension_text(self, text: str) -> bool:
        """Check if text looks like a dimension value."""
        import re
        # Patterns: "12'-0\"", "3.5", "1/2", "6\"", "10'", etc.
        patterns = [
            r"^\d+'-\d+\"?$",  # 12'-0"
            r"^\d+\.?\d*\"$",  # 6"
            r"^\d+\.?\d*'$",   # 10'
            r"^\d+\.?\d*$",    # 3.5
            r"^\d+/\d+$",      # 1/2
            r"^\d+-\d+/\d+$",  # 1-1/2
        ]
        for pattern in patterns:
            if re.match(pattern, text.strip()):
                return True
        return False

    def _is_pipe_size_text(self, text: str) -> bool:
        """Check if text looks like a pipe size."""
        import re
        # Patterns: "3/4\"", "1\"", "2\"", "1-1/2\"", etc.
        patterns = [
            r"^\d+/\d+\"?$",    # 3/4"
            r"^\d+\"$",         # 2"
            r"^\d+-\d+/\d+\"?$", # 1-1/2"
        ]
        for pattern in patterns:
            if re.match(pattern, text.strip()):
                return True
        return False

    def _line_angle(self, line: DetectedLine | ClassifiedLine) -> float:
        """Calculate angle of a line in degrees (0-360)."""
        dx = line.end[0] - line.start[0]
        dy = line.end[1] - line.start[1]
        angle = math.degrees(math.atan2(dy, dx))
        if angle < 0:
            angle += 360
        return angle

    def _line_length(self, line: DetectedLine | ClassifiedLine) -> float:
        """Calculate length of a line."""
        dx = line.end[0] - line.start[0]
        dy = line.end[1] - line.start[1]
        return math.sqrt(dx * dx + dy * dy)

    def _perpendicular_distance(
        self,
        line1: DetectedLine | ClassifiedLine,
        line2: DetectedLine | ClassifiedLine,
    ) -> float:
        """Calculate perpendicular distance between two parallel lines."""
        # Use point-to-line distance from line2's midpoint to line1
        mid2 = (
            (line2.start[0] + line2.end[0]) / 2,
            (line2.start[1] + line2.end[1]) / 2,
        )

        # Line1: from start to end
        x1, y1 = line1.start
        x2, y2 = line1.end
        px, py = mid2

        # Perpendicular distance formula
        dx = x2 - x1
        dy = y2 - y1
        length = math.sqrt(dx * dx + dy * dy)
        if length == 0:
            return float('inf')

        dist = abs((py - y1) * dx - (px - x1) * dy) / length
        return dist

    def _lines_overlap(
        self,
        line1: DetectedLine | ClassifiedLine,
        line2: DetectedLine | ClassifiedLine,
    ) -> bool:
        """Check if two parallel lines have overlapping projections."""
        # Project both lines onto their shared direction
        angle = self._line_angle(line1)
        cos_a = math.cos(math.radians(angle))
        sin_a = math.sin(math.radians(angle))

        # Project line1 endpoints
        proj1_start = line1.start[0] * cos_a + line1.start[1] * sin_a
        proj1_end = line1.end[0] * cos_a + line1.end[1] * sin_a
        if proj1_start > proj1_end:
            proj1_start, proj1_end = proj1_end, proj1_start

        # Project line2 endpoints
        proj2_start = line2.start[0] * cos_a + line2.start[1] * sin_a
        proj2_end = line2.end[0] * cos_a + line2.end[1] * sin_a
        if proj2_start > proj2_end:
            proj2_start, proj2_end = proj2_end, proj2_start

        # Check overlap
        overlap = min(proj1_end, proj2_end) - max(proj1_start, proj2_start)
        min_length = min(proj1_end - proj1_start, proj2_end - proj2_start)

        return overlap > min_length * 0.3  # At least 30% overlap

    def _compute_centerline(
        self,
        line1: DetectedLine | ClassifiedLine,
        line2: DetectedLine | ClassifiedLine,
    ) -> DetectedLine:
        """Compute the centerline between two parallel lines."""
        # Average the endpoints
        start = (
            (line1.start[0] + line2.start[0]) / 2,
            (line1.start[1] + line2.start[1]) / 2,
        )
        end = (
            (line1.end[0] + line2.end[0]) / 2,
            (line1.end[1] + line2.end[1]) / 2,
        )
        return DetectedLine(start=start, end=end, linetype="CENTER")

    def _point_distance(
        self,
        p1: tuple[float, float],
        p2: tuple[float, float] | Any,
    ) -> float:
        """Calculate distance between two points."""
        if isinstance(p2, tuple):
            return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)
        # Handle point objects with x, y attributes
        if hasattr(p2, 'x') and hasattr(p2, 'y'):
            return math.sqrt((p1[0] - p2.x)**2 + (p1[1] - p2.y)**2)
        return float('inf')

    def _compute_statistics(
        self,
        result: GeometryClassifierResult,
    ) -> dict[str, Any]:
        """Compute classification statistics."""
        type_counts: dict[str, int] = {}
        system_counts: dict[str, int] = {}

        for line in result.classified_lines:
            geo_type = line.classification.geometry_type.value
            type_counts[geo_type] = type_counts.get(geo_type, 0) + 1

            if line.classification.system != GeometrySystem.UNKNOWN:
                system = line.classification.system.value
                system_counts[system] = system_counts.get(system, 0) + 1

        for poly in result.classified_polylines:
            geo_type = poly.classification.geometry_type.value
            type_counts[geo_type] = type_counts.get(geo_type, 0) + 1

        for circle in result.classified_circles:
            geo_type = circle.classification.geometry_type.value
            type_counts[geo_type] = type_counts.get(geo_type, 0) + 1

        total = len(result.classified_lines) + len(result.classified_polylines) + len(result.classified_circles)
        classified = total - type_counts.get(GeometryType.UNKNOWN.value, 0)

        return {
            "total_elements": total,
            "classified_elements": classified,
            "classification_rate": classified / total if total > 0 else 0,
            "type_counts": type_counts,
            "system_counts": system_counts,
            "parallel_pairs_found": len(result.parallel_pairs),
        }


# Convenience function for direct usage
async def classify_geometry(
    lines: list[DetectedLine],
    circles: list[DetectedCircle] | None = None,
    polylines: list[DetectedPolyline] | None = None,
    drawing_type: DrawingType = DrawingType.UNKNOWN,
    nearby_symbols: list[Any] | None = None,
    nearby_text: list[Any] | None = None,
    **classifier_kwargs: Any,
) -> GeometryClassifierResult:
    """
    Classify raw geometry into meaningful AEC elements.

    This is the main entry point for geometry classification.

    Args:
        lines: Detected lines from vectorization
        circles: Detected circles
        polylines: Detected polylines/contours
        drawing_type: Type of drawing (affects classification rules)
        nearby_symbols: Detected symbols for context
        nearby_text: Parsed text annotations for context
        **classifier_kwargs: Additional arguments for GeometryClassifier

    Returns:
        GeometryClassifierResult with classified elements

    Example:
        >>> from aec_agent.mcp.tools.geometry_classifier import classify_geometry
        >>> from aec_agent.mcp.tools.document_classifier import DrawingType
        >>> result = await classify_geometry(
        ...     lines=vectorization_result.lines,
        ...     circles=vectorization_result.circles,
        ...     drawing_type=DrawingType.HVAC_PLAN,
        ... )
        >>> print(f"Found {len(result.duct_boundaries)} duct runs")
    """
    classifier = GeometryClassifier(**classifier_kwargs)
    return await classifier.classify(
        lines=lines,
        polylines=polylines,
        circles=circles,
        drawing_type=drawing_type,
        nearby_symbols=nearby_symbols,
        nearby_text=nearby_text,
    )
