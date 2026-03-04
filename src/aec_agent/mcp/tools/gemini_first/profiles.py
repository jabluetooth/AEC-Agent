"""
Conversion Profiles Module.

Provides pre-configured settings for different drawing types:
- ARCHITECTURAL: Floor plans, elevations, sections
- MECHANICAL: Machine drawings, assemblies
- ELECTRICAL: Schematic diagrams, wiring diagrams
- STRUCTURAL: Beam layouts, foundation plans
- CIVIL: Site plans, grading plans
- HVAC: Ductwork layouts, equipment schedules
- PLUMBING: Pipe layouts, riser diagrams
- FIRE_ALARM: Device layouts, riser diagrams

Each profile adjusts:
- Scale factors and tolerances
- Layer naming conventions
- Entity type priorities
- Extraction sensitivity
- Post-processing options
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple


class ProfileType(str, Enum):
    """Available conversion profile types."""
    ARCHITECTURAL = "architectural"
    MECHANICAL = "mechanical"
    ELECTRICAL = "electrical"
    STRUCTURAL = "structural"
    CIVIL = "civil"
    HVAC = "hvac"
    PLUMBING = "plumbing"
    FIRE_ALARM = "fire_alarm"
    GENERAL = "general"  # Default catch-all
    CUSTOM = "custom"  # User-defined


@dataclass
class LayerMapping:
    """Layer naming configuration."""
    prefix: str = ""
    walls: str = "A-WALL"
    doors: str = "A-DOOR"
    windows: str = "A-GLAZ"
    dimensions: str = "A-ANNO-DIMS"
    text: str = "A-ANNO-TEXT"
    furniture: str = "A-FURN"
    equipment: str = "M-EQUIP"
    electrical: str = "E-LITE"
    plumbing: str = "P-FIXT"
    hvac: str = "M-DUCT"
    fire_alarm: str = "F-ALRM"
    structural: str = "S-COLS"
    centerlines: str = "0-CENTER"
    hidden: str = "0-HIDDEN"
    misc: str = "0-MISC"

    def get_layer(self, entity_type: str) -> str:
        """Get layer name for entity type."""
        mapping = {
            "wall": self.walls,
            "door": self.doors,
            "window": self.windows,
            "dimension": self.dimensions,
            "text": self.text,
            "furniture": self.furniture,
            "equipment": self.equipment,
            "electrical": self.electrical,
            "plumbing": self.plumbing,
            "hvac": self.hvac,
            "fire_alarm": self.fire_alarm,
            "structural": self.structural,
            "centerline": self.centerlines,
            "hidden": self.hidden,
        }
        layer = mapping.get(entity_type.lower(), self.misc)
        return f"{self.prefix}{layer}" if self.prefix else layer


@dataclass
class ExtractionSettings:
    """Extraction sensitivity settings."""
    # Line detection
    min_line_length_px: int = 10
    max_line_gap_px: int = 5
    line_thickness_px: Tuple[int, int] = (1, 20)

    # Circle/arc detection
    min_circle_radius_px: int = 5
    max_circle_radius_px: int = 500
    circle_accuracy: float = 0.9

    # Text detection
    min_text_height_px: int = 8
    max_text_height_px: int = 200
    text_confidence_threshold: float = 0.7

    # Symbol detection
    min_symbol_size_px: int = 10
    max_symbol_size_px: int = 200
    symbol_confidence_threshold: float = 0.6

    # Tolerances
    endpoint_snap_tolerance: float = 0.01  # Drawing units
    collinear_angle_tolerance: float = 0.1  # Radians

    # Post-processing
    join_connected_lines: bool = True
    remove_duplicates: bool = True
    simplify_geometry: bool = True
    detect_linetypes: bool = True
    detect_lineweights: bool = True


@dataclass
class ScaleSettings:
    """Scale and unit settings."""
    input_dpi: int = 300
    drawing_units: str = "inches"  # inches, feet, mm, cm, m
    default_scale: str = "1:1"  # Common: 1/4"=1'-0", 1:100, etc.
    scale_factor: float = 1.0  # Computed from scale
    auto_detect_scale: bool = True

    @staticmethod
    def parse_architectural_scale(scale_str: str) -> float:
        """Parse architectural scale string to factor."""
        # Common architectural scales
        scales = {
            "1:1": 1.0,
            "1/16\"=1'-0\"": 192.0,
            "3/32\"=1'-0\"": 128.0,
            "1/8\"=1'-0\"": 96.0,
            "3/16\"=1'-0\"": 64.0,
            "1/4\"=1'-0\"": 48.0,
            "3/8\"=1'-0\"": 32.0,
            "1/2\"=1'-0\"": 24.0,
            "3/4\"=1'-0\"": 16.0,
            "1\"=1'-0\"": 12.0,
            "1-1/2\"=1'-0\"": 8.0,
            "3\"=1'-0\"": 4.0,
            "FULL": 1.0,
            # Metric scales
            "1:10": 10.0,
            "1:20": 20.0,
            "1:25": 25.0,
            "1:50": 50.0,
            "1:100": 100.0,
            "1:200": 200.0,
            "1:500": 500.0,
            "1:1000": 1000.0,
        }
        return scales.get(scale_str.upper(), 1.0)


@dataclass
class OutputSettings:
    """Output format settings."""
    create_in_autocad: bool = False
    export_dxf: bool = True
    dxf_version: str = "R2018"
    generate_preview: bool = True
    preview_format: str = "png"  # png, jpg, svg
    preview_scale: float = 1.0
    generate_report: bool = False
    report_format: str = "json"  # json, csv, txt


@dataclass
class ConversionProfile:
    """Complete conversion profile configuration."""
    name: str
    profile_type: ProfileType
    description: str = ""

    # Component settings
    layers: LayerMapping = field(default_factory=LayerMapping)
    extraction: ExtractionSettings = field(default_factory=ExtractionSettings)
    scale: ScaleSettings = field(default_factory=ScaleSettings)
    output: OutputSettings = field(default_factory=OutputSettings)

    # Entity priorities (what to extract)
    priority_entities: List[str] = field(default_factory=lambda: [
        "line", "circle", "arc", "text", "dimension"
    ])

    # Gemini prompts customization
    analysis_prompt_hints: List[str] = field(default_factory=list)

    # Custom settings
    custom_settings: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        """Convert profile to dictionary."""
        return {
            "name": self.name,
            "profile_type": self.profile_type.value,
            "description": self.description,
            "layers": {
                "prefix": self.layers.prefix,
                "walls": self.layers.walls,
                "doors": self.layers.doors,
                "text": self.layers.text,
            },
            "extraction": {
                "min_line_length_px": self.extraction.min_line_length_px,
                "text_confidence_threshold": self.extraction.text_confidence_threshold,
                "join_connected_lines": self.extraction.join_connected_lines,
            },
            "scale": {
                "input_dpi": self.scale.input_dpi,
                "drawing_units": self.scale.drawing_units,
                "default_scale": self.scale.default_scale,
            },
            "output": {
                "export_dxf": self.output.export_dxf,
                "generate_preview": self.output.generate_preview,
            },
            "priority_entities": self.priority_entities,
        }


# ============================================================================
# Pre-defined Profiles
# ============================================================================

def get_architectural_profile() -> ConversionProfile:
    """Get profile optimized for architectural drawings."""
    return ConversionProfile(
        name="Architectural",
        profile_type=ProfileType.ARCHITECTURAL,
        description="Optimized for floor plans, elevations, and sections",
        layers=LayerMapping(
            prefix="A-",
            walls="WALL",
            doors="DOOR",
            windows="GLAZ",
            dimensions="ANNO-DIMS",
            text="ANNO-TEXT",
            furniture="FURN",
        ),
        extraction=ExtractionSettings(
            min_line_length_px=15,  # Larger for wall lines
            max_line_gap_px=3,
            join_connected_lines=True,
            detect_linetypes=True,
            detect_lineweights=True,
            text_confidence_threshold=0.8,
        ),
        scale=ScaleSettings(
            input_dpi=300,
            drawing_units="feet",
            default_scale="1/4\"=1'-0\"",
            scale_factor=48.0,
            auto_detect_scale=True,
        ),
        output=OutputSettings(
            export_dxf=True,
            generate_preview=True,
        ),
        priority_entities=[
            "line", "polyline", "text", "dimension",
            "door", "window", "wall", "circle", "arc"
        ],
        analysis_prompt_hints=[
            "This is an architectural floor plan",
            "Look for walls, doors, windows, and room labels",
            "Dimensions are typically in feet and inches",
            "Door swings indicate door direction",
        ],
    )


def get_mechanical_profile() -> ConversionProfile:
    """Get profile optimized for mechanical drawings."""
    return ConversionProfile(
        name="Mechanical",
        profile_type=ProfileType.MECHANICAL,
        description="Optimized for machine parts, assemblies, and details",
        layers=LayerMapping(
            prefix="M-",
            equipment="EQUIP",
            centerlines="CENTER",
            hidden="HIDDEN",
            dimensions="DIMS",
            text="TEXT",
        ),
        extraction=ExtractionSettings(
            min_line_length_px=5,  # Smaller for detailed parts
            max_line_gap_px=2,
            circle_accuracy=0.95,  # Higher for precise holes
            detect_linetypes=True,  # Important for centerlines
            detect_lineweights=True,
            collinear_angle_tolerance=0.05,  # Tighter tolerance
        ),
        scale=ScaleSettings(
            input_dpi=400,  # Higher DPI for detail
            drawing_units="inches",
            default_scale="1:1",
            scale_factor=1.0,
        ),
        output=OutputSettings(
            export_dxf=True,
            generate_preview=True,
        ),
        priority_entities=[
            "line", "circle", "arc", "dimension",
            "centerline", "text", "spline"
        ],
        analysis_prompt_hints=[
            "This is a mechanical/machine drawing",
            "Look for centerlines (long-short-long dash pattern)",
            "Hidden lines (dashed) show internal features",
            "Dimensions are typically in inches or mm",
            "Circles may represent holes, shafts, or threads",
        ],
    )


def get_electrical_profile() -> ConversionProfile:
    """Get profile optimized for electrical schematic diagrams."""
    return ConversionProfile(
        name="Electrical",
        profile_type=ProfileType.ELECTRICAL,
        description="Optimized for electrical schematics and wiring diagrams",
        layers=LayerMapping(
            prefix="E-",
            electrical="LITE",
            equipment="POWR",
            text="ANNO",
        ),
        extraction=ExtractionSettings(
            min_line_length_px=8,
            max_line_gap_px=5,  # Allow gaps for symbols
            min_symbol_size_px=15,
            max_symbol_size_px=100,
            symbol_confidence_threshold=0.7,
            join_connected_lines=False,  # Don't join across symbols
        ),
        scale=ScaleSettings(
            input_dpi=300,
            drawing_units="inches",
            default_scale="1:1",
            auto_detect_scale=False,  # Schematics not to scale
        ),
        output=OutputSettings(
            export_dxf=True,
            generate_preview=True,
        ),
        priority_entities=[
            "line", "circle", "arc", "text", "symbol",
            "polyline", "wire", "component"
        ],
        analysis_prompt_hints=[
            "This is an electrical schematic diagram",
            "Look for standard electrical symbols (resistors, capacitors, etc.)",
            "Wires connect components - identify junction points",
            "Component values and reference designators are important",
            "Ground symbols typically point downward",
        ],
    )


def get_structural_profile() -> ConversionProfile:
    """Get profile optimized for structural drawings."""
    return ConversionProfile(
        name="Structural",
        profile_type=ProfileType.STRUCTURAL,
        description="Optimized for structural plans and details",
        layers=LayerMapping(
            prefix="S-",
            structural="COLS",
            walls="WALL",
            dimensions="DIMS",
            text="NOTE",
            centerlines="GRID",
        ),
        extraction=ExtractionSettings(
            min_line_length_px=20,  # Structural elements are larger
            line_thickness_px=(2, 30),  # Thicker lines
            detect_linetypes=True,
            detect_lineweights=True,
            join_connected_lines=True,
        ),
        scale=ScaleSettings(
            input_dpi=300,
            drawing_units="feet",
            default_scale="1/4\"=1'-0\"",
            scale_factor=48.0,
        ),
        output=OutputSettings(
            export_dxf=True,
            generate_preview=True,
        ),
        priority_entities=[
            "line", "polyline", "text", "dimension",
            "column", "beam", "footing", "circle"
        ],
        analysis_prompt_hints=[
            "This is a structural drawing",
            "Columns appear as rectangles or circles on plan",
            "Grid lines have bubble annotations (A, B, 1, 2, etc.)",
            "Look for beam sizes and reinforcement callouts",
            "Foundation elements may have hatching",
        ],
    )


def get_civil_profile() -> ConversionProfile:
    """Get profile optimized for civil/site drawings."""
    return ConversionProfile(
        name="Civil",
        profile_type=ProfileType.CIVIL,
        description="Optimized for site plans and grading plans",
        layers=LayerMapping(
            prefix="C-",
            misc="TOPO",
            text="ANNO",
            dimensions="DIMS",
        ),
        extraction=ExtractionSettings(
            min_line_length_px=10,
            max_line_gap_px=10,  # Contour lines may have gaps
            detect_linetypes=True,
            simplify_geometry=True,  # Smooth contours
        ),
        scale=ScaleSettings(
            input_dpi=300,
            drawing_units="feet",
            default_scale="1\"=20'-0\"",
            scale_factor=240.0,
        ),
        output=OutputSettings(
            export_dxf=True,
            generate_preview=True,
        ),
        priority_entities=[
            "line", "polyline", "text", "arc", "spline",
            "contour", "property_line", "easement"
        ],
        analysis_prompt_hints=[
            "This is a civil/site plan",
            "Contour lines show elevation (numbers indicate height)",
            "Property lines may be dashed or have tick marks",
            "Look for setback lines and easements",
            "North arrow indicates orientation",
        ],
    )


def get_hvac_profile() -> ConversionProfile:
    """Get profile optimized for HVAC ductwork drawings."""
    return ConversionProfile(
        name="HVAC",
        profile_type=ProfileType.HVAC,
        description="Optimized for ductwork layouts and equipment",
        layers=LayerMapping(
            prefix="M-",
            hvac="DUCT",
            equipment="EQUIP",
            text="ANNO",
        ),
        extraction=ExtractionSettings(
            min_line_length_px=15,
            line_thickness_px=(1, 15),
            min_symbol_size_px=20,
            max_symbol_size_px=150,
            join_connected_lines=True,
        ),
        scale=ScaleSettings(
            input_dpi=300,
            drawing_units="feet",
            default_scale="1/4\"=1'-0\"",
            scale_factor=48.0,
        ),
        output=OutputSettings(
            export_dxf=True,
            generate_preview=True,
        ),
        priority_entities=[
            "line", "polyline", "rectangle", "text",
            "diffuser", "grille", "equipment", "duct"
        ],
        analysis_prompt_hints=[
            "This is an HVAC ductwork drawing",
            "Ducts are typically shown as parallel lines",
            "Diffusers and grilles have standard symbols",
            "CFM values indicate airflow",
            "Equipment symbols may include fans, AHUs, VAVs",
        ],
    )


def get_plumbing_profile() -> ConversionProfile:
    """Get profile optimized for plumbing drawings."""
    return ConversionProfile(
        name="Plumbing",
        profile_type=ProfileType.PLUMBING,
        description="Optimized for pipe layouts and fixture drawings",
        layers=LayerMapping(
            prefix="P-",
            plumbing="PIPE",
            equipment="FIXT",
            text="ANNO",
        ),
        extraction=ExtractionSettings(
            min_line_length_px=10,
            line_thickness_px=(1, 10),
            detect_linetypes=True,  # Different pipe types
            min_symbol_size_px=15,
            max_symbol_size_px=100,
        ),
        scale=ScaleSettings(
            input_dpi=300,
            drawing_units="feet",
            default_scale="1/4\"=1'-0\"",
            scale_factor=48.0,
        ),
        output=OutputSettings(
            export_dxf=True,
            generate_preview=True,
        ),
        priority_entities=[
            "line", "polyline", "circle", "text",
            "fixture", "valve", "pipe", "fitting"
        ],
        analysis_prompt_hints=[
            "This is a plumbing drawing",
            "Different line types indicate pipe types (CW, HW, sanitary, etc.)",
            "Fixture symbols include toilets, sinks, water heaters",
            "Pipe sizes are typically noted",
            "Valves appear as symbols along pipe runs",
        ],
    )


def get_fire_alarm_profile() -> ConversionProfile:
    """Get profile optimized for fire alarm drawings."""
    return ConversionProfile(
        name="Fire Alarm",
        profile_type=ProfileType.FIRE_ALARM,
        description="Optimized for fire alarm device layouts",
        layers=LayerMapping(
            prefix="F-",
            fire_alarm="ALRM",
            equipment="PANEL",
            text="ANNO",
        ),
        extraction=ExtractionSettings(
            min_line_length_px=8,  # Conduit runs
            min_symbol_size_px=10,  # Small device symbols
            max_symbol_size_px=80,
            symbol_confidence_threshold=0.65,
            join_connected_lines=True,  # Connect conduit runs
        ),
        scale=ScaleSettings(
            input_dpi=300,
            drawing_units="feet",
            default_scale="1/4\"=1'-0\"",
            scale_factor=48.0,
        ),
        output=OutputSettings(
            export_dxf=True,
            generate_preview=True,
        ),
        priority_entities=[
            "line", "circle", "text", "symbol",
            "smoke_detector", "pull_station", "horn_strobe",
            "duct_detector", "control_panel"
        ],
        analysis_prompt_hints=[
            "This is a fire alarm layout drawing",
            "Smoke detectors are typically circles with 'SD' or similar label",
            "Pull stations are at exits",
            "Horn/strobes are notification devices",
            "Lines between devices represent conduit runs",
            "Device addresses may be noted (e.g., L1-001)",
        ],
    )


def get_general_profile() -> ConversionProfile:
    """Get default general-purpose profile."""
    return ConversionProfile(
        name="General",
        profile_type=ProfileType.GENERAL,
        description="General-purpose profile for unknown drawing types",
        layers=LayerMapping(),
        extraction=ExtractionSettings(),
        scale=ScaleSettings(),
        output=OutputSettings(),
        priority_entities=[
            "line", "circle", "arc", "text", "dimension",
            "polyline", "spline", "ellipse"
        ],
        analysis_prompt_hints=[
            "Analyze this technical drawing",
            "Identify geometric entities: lines, circles, arcs",
            "Extract text and dimensions",
            "Note any standard symbols or patterns",
        ],
    )


# ============================================================================
# Profile Management
# ============================================================================

_BUILTIN_PROFILES: Dict[ProfileType, ConversionProfile] = {}
_CUSTOM_PROFILES: Dict[str, ConversionProfile] = {}


def _initialize_builtin_profiles() -> None:
    """Initialize built-in profiles."""
    global _BUILTIN_PROFILES
    _BUILTIN_PROFILES = {
        ProfileType.ARCHITECTURAL: get_architectural_profile(),
        ProfileType.MECHANICAL: get_mechanical_profile(),
        ProfileType.ELECTRICAL: get_electrical_profile(),
        ProfileType.STRUCTURAL: get_structural_profile(),
        ProfileType.CIVIL: get_civil_profile(),
        ProfileType.HVAC: get_hvac_profile(),
        ProfileType.PLUMBING: get_plumbing_profile(),
        ProfileType.FIRE_ALARM: get_fire_alarm_profile(),
        ProfileType.GENERAL: get_general_profile(),
    }


def get_profile(profile_type: ProfileType) -> ConversionProfile:
    """
    Get a profile by type.

    Args:
        profile_type: Profile type enum

    Returns:
        ConversionProfile instance
    """
    if not _BUILTIN_PROFILES:
        _initialize_builtin_profiles()

    return _BUILTIN_PROFILES.get(profile_type, get_general_profile())


def get_profile_by_name(name: str) -> Optional[ConversionProfile]:
    """
    Get a profile by name.

    Args:
        name: Profile name (case-insensitive)

    Returns:
        ConversionProfile or None
    """
    # Check custom profiles first
    if name.lower() in _CUSTOM_PROFILES:
        return _CUSTOM_PROFILES[name.lower()]

    # Check builtin profiles
    try:
        profile_type = ProfileType(name.lower())
        return get_profile(profile_type)
    except ValueError:
        pass

    return None


def register_custom_profile(profile: ConversionProfile) -> None:
    """
    Register a custom profile.

    Args:
        profile: ConversionProfile to register
    """
    _CUSTOM_PROFILES[profile.name.lower()] = profile


def list_profiles() -> List[Dict[str, str]]:
    """
    List all available profiles.

    Returns:
        List of profile info dicts
    """
    if not _BUILTIN_PROFILES:
        _initialize_builtin_profiles()

    profiles = []

    # Built-in profiles
    for profile_type, profile in _BUILTIN_PROFILES.items():
        profiles.append({
            "name": profile.name,
            "type": profile_type.value,
            "description": profile.description,
            "builtin": True,
        })

    # Custom profiles
    for name, profile in _CUSTOM_PROFILES.items():
        profiles.append({
            "name": profile.name,
            "type": "custom",
            "description": profile.description,
            "builtin": False,
        })

    return profiles


def auto_detect_profile(
    analysis_result: Optional[Dict[str, Any]] = None,
    filename: Optional[str] = None,
) -> ConversionProfile:
    """
    Auto-detect appropriate profile from drawing analysis or filename.

    Args:
        analysis_result: Gemini analysis result dict
        filename: Original filename

    Returns:
        Best matching ConversionProfile
    """
    if not _BUILTIN_PROFILES:
        _initialize_builtin_profiles()

    # Keywords for each profile type
    keywords = {
        ProfileType.ARCHITECTURAL: [
            "floor plan", "elevation", "section", "architectural",
            "room", "door", "window", "wall", "corridor", "restroom",
        ],
        ProfileType.MECHANICAL: [
            "mechanical", "machine", "part", "assembly", "shaft",
            "bearing", "thread", "tolerance", "detail",
        ],
        ProfileType.ELECTRICAL: [
            "electrical", "schematic", "wiring", "circuit", "panel",
            "outlet", "switch", "transformer", "motor",
        ],
        ProfileType.STRUCTURAL: [
            "structural", "beam", "column", "footing", "foundation",
            "rebar", "concrete", "steel", "framing",
        ],
        ProfileType.CIVIL: [
            "civil", "site", "grading", "contour", "property",
            "easement", "survey", "topographic",
        ],
        ProfileType.HVAC: [
            "hvac", "duct", "mechanical", "air handling", "diffuser",
            "vav", "ahu", "cfm", "ventilation",
        ],
        ProfileType.PLUMBING: [
            "plumbing", "pipe", "sanitary", "drainage", "fixture",
            "water", "riser", "isometric",
        ],
        ProfileType.FIRE_ALARM: [
            "fire alarm", "smoke detector", "pull station", "facp",
            "notification", "horn", "strobe", "fa",
        ],
    }

    scores = {pt: 0 for pt in ProfileType}

    # Check filename
    if filename:
        filename_lower = filename.lower()
        for profile_type, kw_list in keywords.items():
            for kw in kw_list:
                if kw in filename_lower:
                    scores[profile_type] += 1

    # Check analysis result
    if analysis_result:
        result_str = str(analysis_result).lower()
        for profile_type, kw_list in keywords.items():
            for kw in kw_list:
                if kw in result_str:
                    scores[profile_type] += 1

        # Check drawing type field specifically
        drawing_type = analysis_result.get("drawing_type", "").lower()
        for profile_type, kw_list in keywords.items():
            for kw in kw_list:
                if kw in drawing_type:
                    scores[profile_type] += 3  # Higher weight

    # Find best match
    best_profile = ProfileType.GENERAL
    best_score = 0

    for profile_type, score in scores.items():
        if score > best_score:
            best_score = score
            best_profile = profile_type

    return get_profile(best_profile)


def create_custom_profile(
    name: str,
    base_profile: ProfileType = ProfileType.GENERAL,
    **overrides,
) -> ConversionProfile:
    """
    Create a custom profile based on an existing profile.

    Args:
        name: Name for the custom profile
        base_profile: Base profile to derive from
        **overrides: Settings to override

    Returns:
        New ConversionProfile

    Example:
        >>> profile = create_custom_profile(
        ...     name="My Architectural",
        ...     base_profile=ProfileType.ARCHITECTURAL,
        ...     input_dpi=400,
        ...     export_dxf=False,
        ... )
    """
    base = get_profile(base_profile)

    # Create new profile with overrides
    profile = ConversionProfile(
        name=name,
        profile_type=ProfileType.CUSTOM,
        description=overrides.get("description", f"Custom profile based on {base.name}"),
        layers=base.layers,
        extraction=base.extraction,
        scale=base.scale,
        output=base.output,
        priority_entities=base.priority_entities.copy(),
        analysis_prompt_hints=base.analysis_prompt_hints.copy(),
    )

    # Apply overrides
    if "input_dpi" in overrides:
        profile.scale.input_dpi = overrides["input_dpi"]
    if "drawing_units" in overrides:
        profile.scale.drawing_units = overrides["drawing_units"]
    if "export_dxf" in overrides:
        profile.output.export_dxf = overrides["export_dxf"]
    if "generate_preview" in overrides:
        profile.output.generate_preview = overrides["generate_preview"]
    if "layer_prefix" in overrides:
        profile.layers.prefix = overrides["layer_prefix"]

    return profile
