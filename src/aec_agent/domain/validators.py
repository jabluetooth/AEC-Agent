"""
Specialized MEP Validators for design verification.

Provides specific validation logic for common MEP checks:
- Clearance validation (distances between systems)
- Sizing validation (duct/pipe sizing for flow)
- Routing validation (slopes, directions, conflicts)
- Coverage validation (diffuser/sprinkler spacing)
"""

import logging
import math
from dataclasses import dataclass, field
from typing import Any, Optional
from uuid import UUID, uuid4

logger = logging.getLogger(__name__)


@dataclass
class Point3D:
    """3D point for spatial calculations."""
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0

    def distance_to(self, other: "Point3D") -> float:
        """Calculate Euclidean distance to another point."""
        return math.sqrt(
            (self.x - other.x) ** 2 +
            (self.y - other.y) ** 2 +
            (self.z - other.z) ** 2
        )

    def distance_2d(self, other: "Point3D") -> float:
        """Calculate 2D distance (ignoring Z)."""
        return math.sqrt(
            (self.x - other.x) ** 2 +
            (self.y - other.y) ** 2
        )


@dataclass
class ValidationIssue:
    """A single validation issue found."""
    element_id: Optional[UUID] = None
    element_name: str = ""
    issue_type: str = ""      # clearance, sizing, routing, coverage
    severity: str = "warning"  # error, warning, info
    message: str = ""
    location: Optional[Point3D] = None
    details: dict[str, Any] = field(default_factory=dict)


# =============================================================================
# Clearance Validator
# =============================================================================

class ClearanceValidator:
    """
    Validates clearances between MEP elements.

    Checks:
    - Minimum distances between systems
    - Insulation clearances
    - Structural clearances
    - Access/service clearances
    """

    # Default clearance requirements (meters)
    DEFAULT_CLEARANCES = {
        # Duct to structure
        ("duct", "beam"): 0.15,          # 6 inches
        ("duct", "column"): 0.10,        # 4 inches
        ("duct", "slab"): 0.10,          # 4 inches

        # Duct to duct (with insulation)
        ("supply_duct", "supply_duct"): 0.05,
        ("supply_duct", "return_duct"): 0.05,

        # Pipe to structure
        ("pipe", "beam"): 0.10,
        ("pipe", "column"): 0.05,

        # Conduit clearances
        ("conduit", "pipe"): 0.15,       # Hot water pipes
        ("conduit", "duct"): 0.05,

        # Equipment access
        ("ahu", "wall"): 0.45,           # 18 inches for service
        ("vav", "ceiling"): 0.30,        # 12 inches access panel

        # Sprinkler clearances
        ("sprinkler", "obstruction"): 0.45,  # NFPA 13 requirement

        # Default fallback
        ("default", "default"): 0.10,
    }

    def __init__(self, custom_clearances: Optional[dict] = None):
        """
        Initialize clearance validator.

        Args:
            custom_clearances: Optional dict of custom clearance requirements
        """
        self.clearances = self.DEFAULT_CLEARANCES.copy()
        if custom_clearances:
            self.clearances.update(custom_clearances)

    def get_required_clearance(
        self,
        element_type: str,
        near_type: str,
    ) -> float:
        """Get required clearance between two element types."""
        key = (element_type.lower(), near_type.lower())
        if key in self.clearances:
            return self.clearances[key]

        # Try reversed
        rev_key = (near_type.lower(), element_type.lower())
        if rev_key in self.clearances:
            return self.clearances[rev_key]

        # Fallback to default
        return self.clearances.get(("default", "default"), 0.10)

    def check_clearance(
        self,
        element: dict[str, Any],
        nearby_element: dict[str, Any],
    ) -> Optional[ValidationIssue]:
        """
        Check clearance between two elements.

        Args:
            element: Primary element dict with position
            nearby_element: Nearby element to check against

        Returns:
            ValidationIssue if clearance violated, None otherwise
        """
        # Get positions
        elem_pos = self._get_position(element)
        near_pos = self._get_position(nearby_element)

        if not elem_pos or not near_pos:
            return None  # Can't validate without positions

        # Calculate distance
        distance = elem_pos.distance_to(near_pos)

        # Get required clearance
        elem_type = element.get("category", "default")
        near_type = nearby_element.get("category", "default")
        required = self.get_required_clearance(elem_type, near_type)

        if distance < required:
            return ValidationIssue(
                element_id=UUID(element["id"]) if element.get("id") else None,
                element_name=element.get("name", "Unknown"),
                issue_type="clearance",
                severity="error" if distance < required * 0.5 else "warning",
                message=f"Clearance violation: {distance:.3f}m < {required:.3f}m required between {elem_type} and {near_type}",
                location=elem_pos,
                details={
                    "actual_distance": distance,
                    "required_distance": required,
                    "element_type": elem_type,
                    "near_type": near_type,
                    "near_element_id": nearby_element.get("id"),
                },
            )

        return None

    def validate_elements(
        self,
        elements: list[dict[str, Any]],
        search_radius: float = 2.0,
    ) -> list[ValidationIssue]:
        """
        Validate clearances for all elements.

        Args:
            elements: List of element dicts
            search_radius: Max distance to check for conflicts

        Returns:
            List of ValidationIssues
        """
        issues = []

        for i, element in enumerate(elements):
            elem_pos = self._get_position(element)
            if not elem_pos:
                continue

            # Check against all other elements
            for j, other in enumerate(elements):
                if i >= j:  # Skip self and already-checked pairs
                    continue

                other_pos = self._get_position(other)
                if not other_pos:
                    continue

                # Quick distance check
                if elem_pos.distance_to(other_pos) > search_radius:
                    continue

                issue = self.check_clearance(element, other)
                if issue:
                    issues.append(issue)

        return issues

    def _get_position(self, element: dict[str, Any]) -> Optional[Point3D]:
        """Extract position from element dict."""
        props = element.get("properties", element)

        # Try various position field names
        for pos_field in ["position", "location", "origin", "center"]:
            pos = props.get(pos_field)
            if pos:
                if isinstance(pos, dict):
                    return Point3D(
                        x=pos.get("x", pos.get("X", 0)),
                        y=pos.get("y", pos.get("Y", 0)),
                        z=pos.get("z", pos.get("Z", 0)),
                    )
                elif isinstance(pos, (list, tuple)) and len(pos) >= 2:
                    return Point3D(
                        x=pos[0],
                        y=pos[1],
                        z=pos[2] if len(pos) > 2 else 0,
                    )

        # Try individual coordinates
        if "x" in props and "y" in props:
            return Point3D(
                x=props.get("x", 0),
                y=props.get("y", 0),
                z=props.get("z", 0),
            )

        return None


# =============================================================================
# Sizing Validator
# =============================================================================

class SizingValidator:
    """
    Validates MEP element sizing for flow requirements.

    Checks:
    - Duct velocities (noise, pressure drop)
    - Pipe velocities (erosion, noise)
    - Equipment capacities
    """

    # Velocity limits (fpm for ducts, fps for pipes)
    VELOCITY_LIMITS = {
        # Duct velocities by type and location
        "supply_main": {"min": 1000, "max": 4000, "recommended": 2500},
        "supply_branch": {"min": 600, "max": 2000, "recommended": 1200},
        "return_main": {"min": 1000, "max": 4500, "recommended": 3000},
        "return_branch": {"min": 600, "max": 2400, "recommended": 1400},
        "exhaust": {"min": 800, "max": 3000, "recommended": 2000},

        # Space type modifiers (multiply max by this)
        "office": 0.7,      # Lower velocities for quiet spaces
        "lobby": 0.6,
        "conference": 0.5,
        "mechanical": 1.2,  # Higher velocities OK

        # Pipe velocities (fps)
        "cold_water": {"min": 2, "max": 8, "recommended": 5},
        "hot_water": {"min": 2, "max": 5, "recommended": 4},
        "chilled_water": {"min": 3, "max": 10, "recommended": 6},
        "condenser_water": {"min": 4, "max": 12, "recommended": 8},
        "steam": {"min": 60, "max": 120, "recommended": 80},  # Steam is ft/min
    }

    def __init__(self, custom_limits: Optional[dict] = None):
        """
        Initialize sizing validator.

        Args:
            custom_limits: Optional custom velocity limits
        """
        self.limits = self.VELOCITY_LIMITS.copy()
        if custom_limits:
            self.limits.update(custom_limits)

    def check_duct_velocity(
        self,
        element: dict[str, Any],
        space_type: str = "general",
    ) -> Optional[ValidationIssue]:
        """
        Check duct velocity against limits.

        Args:
            element: Duct element with flow and size data
            space_type: Type of space (office, mechanical, etc.)

        Returns:
            ValidationIssue if velocity is out of range
        """
        props = element.get("properties", element)

        # Get velocity or calculate from flow and area
        velocity = props.get("velocity", props.get("air_velocity"))

        if velocity is None:
            # Try to calculate from CFM and area
            cfm = props.get("cfm", props.get("flow"))
            area = props.get("area")  # sq ft

            if cfm and area and area > 0:
                velocity = cfm / area  # fpm
            else:
                return None  # Can't validate without velocity

        # Determine duct type
        duct_type = self._classify_duct(element)
        limits = self.limits.get(duct_type, self.limits.get("supply_branch"))

        # Apply space type modifier
        space_modifier = self.limits.get(space_type.lower(), 1.0)
        if isinstance(space_modifier, (int, float)):
            max_velocity = limits["max"] * space_modifier
        else:
            max_velocity = limits["max"]

        # Check velocity
        if velocity > max_velocity:
            return ValidationIssue(
                element_id=UUID(element["id"]) if element.get("id") else None,
                element_name=element.get("name", "Unknown"),
                issue_type="sizing",
                severity="warning" if velocity < max_velocity * 1.2 else "error",
                message=f"High velocity: {velocity:.0f} fpm exceeds {max_velocity:.0f} fpm limit for {duct_type} in {space_type}",
                details={
                    "velocity": velocity,
                    "max_velocity": max_velocity,
                    "duct_type": duct_type,
                    "space_type": space_type,
                    "recommended": limits["recommended"],
                },
            )

        if velocity < limits["min"]:
            return ValidationIssue(
                element_id=UUID(element["id"]) if element.get("id") else None,
                element_name=element.get("name", "Unknown"),
                issue_type="sizing",
                severity="info",
                message=f"Low velocity: {velocity:.0f} fpm below {limits['min']:.0f} fpm minimum - duct may be oversized",
                details={
                    "velocity": velocity,
                    "min_velocity": limits["min"],
                    "duct_type": duct_type,
                },
            )

        return None

    def check_pipe_velocity(
        self,
        element: dict[str, Any],
    ) -> Optional[ValidationIssue]:
        """
        Check pipe velocity against limits.

        Args:
            element: Pipe element with flow and size data

        Returns:
            ValidationIssue if velocity is out of range
        """
        props = element.get("properties", element)

        velocity = props.get("velocity")
        if velocity is None:
            # Calculate from GPM and diameter
            gpm = props.get("gpm", props.get("flow"))
            diameter = props.get("diameter")  # inches

            if gpm and diameter and diameter > 0:
                # Velocity = 0.4085 * GPM / D^2 (fps)
                velocity = 0.4085 * gpm / (diameter ** 2)
            else:
                return None

        # Determine pipe type
        pipe_type = self._classify_pipe(element)
        limits = self.limits.get(pipe_type, self.limits.get("cold_water"))

        if not isinstance(limits, dict):
            return None

        if velocity > limits["max"]:
            return ValidationIssue(
                element_id=UUID(element["id"]) if element.get("id") else None,
                element_name=element.get("name", "Unknown"),
                issue_type="sizing",
                severity="warning" if velocity < limits["max"] * 1.2 else "error",
                message=f"High pipe velocity: {velocity:.1f} fps exceeds {limits['max']} fps limit for {pipe_type}",
                details={
                    "velocity": velocity,
                    "max_velocity": limits["max"],
                    "pipe_type": pipe_type,
                    "concern": "erosion/noise" if velocity > limits["max"] * 1.5 else "noise",
                },
            )

        return None

    def _classify_duct(self, element: dict[str, Any]) -> str:
        """Classify duct type from element properties."""
        props = element.get("properties", element)
        name = str(element.get("name", "")).lower()
        system = str(props.get("system", props.get("system_type", ""))).lower()

        is_main = any(kw in name for kw in ["main", "trunk", "riser"])

        if "supply" in name or "supply" in system:
            return "supply_main" if is_main else "supply_branch"
        elif "return" in name or "return" in system:
            return "return_main" if is_main else "return_branch"
        elif "exhaust" in name or "exhaust" in system:
            return "exhaust"

        return "supply_branch"  # Default

    def _classify_pipe(self, element: dict[str, Any]) -> str:
        """Classify pipe type from element properties."""
        props = element.get("properties", element)
        name = str(element.get("name", "")).lower()
        system = str(props.get("system", props.get("system_type", ""))).lower()

        if "hot" in name or "hwp" in system or "hot" in system:
            return "hot_water"
        elif "chill" in name or "chw" in system:
            return "chilled_water"
        elif "condenser" in name or "cw" in system:
            return "condenser_water"
        elif "steam" in name or "steam" in system:
            return "steam"

        return "cold_water"


# =============================================================================
# Routing Validator
# =============================================================================

class RoutingValidator:
    """
    Validates MEP routing requirements.

    Checks:
    - Drain slopes
    - Duct/pipe directions relative to structure
    - Transition angles
    - Offset requirements
    """

    # Minimum slopes (percent)
    MIN_SLOPES = {
        "condensate": 1.0,       # 1/8" per foot = ~1%
        "gravity_drain": 2.0,    # 1/4" per foot = ~2%
        "storm_drain": 1.0,
        "sanitary": 2.0,
        "vent": 0,               # Vents can be level
    }

    def __init__(self, custom_slopes: Optional[dict] = None):
        """
        Initialize routing validator.

        Args:
            custom_slopes: Optional custom slope requirements
        """
        self.slopes = self.MIN_SLOPES.copy()
        if custom_slopes:
            self.slopes.update(custom_slopes)

    def check_slope(
        self,
        element: dict[str, Any],
        start_point: Optional[Point3D] = None,
        end_point: Optional[Point3D] = None,
    ) -> Optional[ValidationIssue]:
        """
        Check that element has adequate slope.

        Args:
            element: Element dict
            start_point: Optional start point
            end_point: Optional end point

        Returns:
            ValidationIssue if slope is insufficient
        """
        props = element.get("properties", element)

        # Get slope from properties or calculate
        slope = props.get("slope")

        if slope is None and start_point and end_point:
            horizontal = start_point.distance_2d(end_point)
            if horizontal > 0:
                vertical = end_point.z - start_point.z
                slope = abs(vertical / horizontal) * 100  # percent

        if slope is None:
            return None  # Can't validate

        # Determine drain type
        drain_type = self._classify_drain(element)
        min_slope = self.slopes.get(drain_type, 0)

        if slope < min_slope:
            return ValidationIssue(
                element_id=UUID(element["id"]) if element.get("id") else None,
                element_name=element.get("name", "Unknown"),
                issue_type="routing",
                severity="error",
                message=f"Insufficient slope: {slope:.2f}% < {min_slope}% required for {drain_type}",
                details={
                    "actual_slope": slope,
                    "min_slope": min_slope,
                    "drain_type": drain_type,
                },
            )

        return None

    def check_offset(
        self,
        element: dict[str, Any],
        max_offset_angle: float = 45,
    ) -> Optional[ValidationIssue]:
        """
        Check that duct/pipe offsets are within limits.

        Args:
            element: Element dict with offset angle
            max_offset_angle: Maximum allowed offset angle

        Returns:
            ValidationIssue if offset is too steep
        """
        props = element.get("properties", element)
        offset_angle = props.get("offset_angle", props.get("angle"))

        if offset_angle is None:
            return None

        if offset_angle > max_offset_angle:
            return ValidationIssue(
                element_id=UUID(element["id"]) if element.get("id") else None,
                element_name=element.get("name", "Unknown"),
                issue_type="routing",
                severity="warning",
                message=f"Steep offset: {offset_angle}° exceeds {max_offset_angle}° recommended maximum",
                details={
                    "offset_angle": offset_angle,
                    "max_angle": max_offset_angle,
                    "recommendation": "Use 45° or flatter offsets for better flow",
                },
            )

        return None

    def _classify_drain(self, element: dict[str, Any]) -> str:
        """Classify drain type from element."""
        name = str(element.get("name", "")).lower()
        props = element.get("properties", element)
        system = str(props.get("system", "")).lower()

        if "condensate" in name or "condensate" in system:
            return "condensate"
        elif "storm" in name or "storm" in system:
            return "storm_drain"
        elif "sanitary" in name or "sanitary" in system:
            return "sanitary"
        elif "vent" in name or "vent" in system:
            return "vent"

        return "gravity_drain"


# =============================================================================
# Coverage Validator
# =============================================================================

class CoverageValidator:
    """
    Validates coverage for diffusers and sprinklers.

    Checks:
    - Diffuser spacing for air distribution
    - Sprinkler spacing per NFPA 13
    - Dead zones in coverage
    """

    # Coverage standards
    COVERAGE = {
        # Diffuser coverage (sq ft per diffuser by type)
        "diffuser_linear": 150,
        "diffuser_square": 200,
        "diffuser_round": 175,

        # Max spacing (feet)
        "diffuser_max_spacing": 15,

        # Sprinkler coverage (sq ft per head)
        "sprinkler_light": 225,
        "sprinkler_ordinary": 130,
        "sprinkler_extra": 100,

        # Sprinkler max spacing (feet)
        "sprinkler_max_spacing": 15,
    }

    def __init__(self, custom_coverage: Optional[dict] = None):
        """
        Initialize coverage validator.

        Args:
            custom_coverage: Optional custom coverage requirements
        """
        self.coverage = self.COVERAGE.copy()
        if custom_coverage:
            self.coverage.update(custom_coverage)

    def check_diffuser_coverage(
        self,
        room_area: float,
        diffuser_count: int,
        diffuser_type: str = "square",
    ) -> Optional[ValidationIssue]:
        """
        Check diffuser coverage for a room.

        Args:
            room_area: Room area in sq ft
            diffuser_count: Number of diffusers
            diffuser_type: Type of diffuser

        Returns:
            ValidationIssue if coverage is inadequate
        """
        if diffuser_count <= 0:
            return ValidationIssue(
                issue_type="coverage",
                severity="error",
                message="No diffusers in room",
                details={"room_area": room_area},
            )

        coverage_per_diffuser = room_area / diffuser_count
        max_coverage = self.coverage.get(f"diffuser_{diffuser_type}", 200)

        if coverage_per_diffuser > max_coverage:
            return ValidationIssue(
                issue_type="coverage",
                severity="warning",
                message=f"Insufficient diffuser coverage: {coverage_per_diffuser:.0f} sq ft/diffuser exceeds {max_coverage} sq ft max",
                details={
                    "room_area": room_area,
                    "diffuser_count": diffuser_count,
                    "coverage_per_diffuser": coverage_per_diffuser,
                    "max_coverage": max_coverage,
                    "recommended_count": math.ceil(room_area / max_coverage),
                },
            )

        return None

    def check_diffuser_spacing(
        self,
        diffusers: list[dict[str, Any]],
    ) -> list[ValidationIssue]:
        """
        Check spacing between diffusers.

        Args:
            diffusers: List of diffuser elements with positions

        Returns:
            List of ValidationIssues for spacing problems
        """
        issues = []
        max_spacing = self.coverage.get("diffuser_max_spacing", 15)

        # Check spacing between each pair
        for i, diff1 in enumerate(diffusers):
            pos1 = self._get_position_2d(diff1)
            if not pos1:
                continue

            for j, diff2 in enumerate(diffusers):
                if i >= j:
                    continue

                pos2 = self._get_position_2d(diff2)
                if not pos2:
                    continue

                distance = pos1.distance_2d(pos2)

                # Convert to feet if needed (assuming meters)
                distance_ft = distance * 3.281

                if distance_ft > max_spacing:
                    issues.append(ValidationIssue(
                        element_id=UUID(diff1["id"]) if diff1.get("id") else None,
                        issue_type="coverage",
                        severity="warning",
                        message=f"Diffuser spacing {distance_ft:.1f} ft exceeds {max_spacing} ft maximum",
                        details={
                            "distance_ft": distance_ft,
                            "max_spacing": max_spacing,
                            "diffuser_1": diff1.get("name"),
                            "diffuser_2": diff2.get("name"),
                        },
                    ))

        return issues

    def check_sprinkler_coverage(
        self,
        area: float,
        sprinkler_count: int,
        hazard_class: str = "ordinary",
    ) -> Optional[ValidationIssue]:
        """
        Check sprinkler coverage per NFPA 13.

        Args:
            area: Coverage area in sq ft
            sprinkler_count: Number of sprinkler heads
            hazard_class: light, ordinary, or extra hazard

        Returns:
            ValidationIssue if coverage is inadequate
        """
        if sprinkler_count <= 0:
            return ValidationIssue(
                issue_type="coverage",
                severity="error",
                message="No sprinklers in area",
                details={"area": area, "hazard_class": hazard_class},
            )

        coverage_per_head = area / sprinkler_count
        max_coverage = self.coverage.get(f"sprinkler_{hazard_class}", 130)

        if coverage_per_head > max_coverage:
            return ValidationIssue(
                issue_type="coverage",
                severity="error",  # Sprinkler coverage is code-required
                message=f"NFPA 13 violation: {coverage_per_head:.0f} sq ft/head exceeds {max_coverage} sq ft max for {hazard_class} hazard",
                details={
                    "area": area,
                    "sprinkler_count": sprinkler_count,
                    "coverage_per_head": coverage_per_head,
                    "max_coverage": max_coverage,
                    "hazard_class": hazard_class,
                    "required_heads": math.ceil(area / max_coverage),
                },
            )

        return None

    def _get_position_2d(self, element: dict[str, Any]) -> Optional[Point3D]:
        """Extract 2D position from element."""
        props = element.get("properties", element)

        for pos_field in ["position", "location", "origin"]:
            pos = props.get(pos_field)
            if pos:
                if isinstance(pos, dict):
                    return Point3D(
                        x=pos.get("x", pos.get("X", 0)),
                        y=pos.get("y", pos.get("Y", 0)),
                        z=0,
                    )
                elif isinstance(pos, (list, tuple)) and len(pos) >= 2:
                    return Point3D(x=pos[0], y=pos[1], z=0)

        return None
