"""
Knowledge Query Module for CAD Standards.

Provides lookup functionality for CAD standards including layers, blocks,
colors, linetypes, and attributes based on element type and system.
This is Phase F of the Semantic Intelligence Pipeline.

Standards Hierarchy:
1. Project-specific standards (custom overrides)
2. Company standards (org-wide defaults)
3. Default AEC standards (NCS-based)

Usage:
    >>> from aec_agent.mcp.tools.knowledge_query import query_cad_standards
    >>> standards = await query_cad_standards(
    ...     element_type="valve",
    ...     subtype="gate",
    ...     system="domestic_cold_water",
    ... )
    >>> print(f"Layer: {standards.layer}, Block: {standards.block_name}")
"""

import os
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

import structlog

logger = structlog.get_logger(__name__)


class Discipline(str, Enum):
    """AEC discipline categories."""
    ARCHITECTURAL = "architectural"
    MECHANICAL = "mechanical"
    ELECTRICAL = "electrical"
    PLUMBING = "plumbing"
    FIRE = "fire"
    LOW_VOLTAGE = "low_voltage"
    CIVIL = "civil"
    STRUCTURAL = "structural"
    UNKNOWN = "unknown"


class SystemType(str, Enum):
    """MEP system types."""
    # HVAC Systems
    SUPPLY_AIR = "supply_air"
    RETURN_AIR = "return_air"
    EXHAUST_AIR = "exhaust_air"
    OUTSIDE_AIR = "outside_air"
    CHILLED_WATER = "chilled_water"
    HOT_WATER_HEATING = "hot_water_heating"
    STEAM = "steam"
    CONDENSATE = "condensate"
    REFRIGERANT = "refrigerant"

    # Plumbing Systems
    DOMESTIC_COLD_WATER = "domestic_cold_water"
    DOMESTIC_HOT_WATER = "domestic_hot_water"
    SANITARY = "sanitary"
    VENT = "vent"
    STORM_DRAIN = "storm_drain"
    NATURAL_GAS = "natural_gas"

    # Electrical Systems
    POWER = "power"
    LIGHTING = "lighting"
    EMERGENCY_POWER = "emergency_power"
    GROUNDING = "grounding"

    # Fire Systems
    FIRE_ALARM = "fire_alarm"
    FIRE_SUPPRESSION = "fire_suppression"
    FIRE_STANDPIPE = "fire_standpipe"

    # Low Voltage Systems
    DATA = "data"
    VOICE = "voice"
    SECURITY = "security"
    AV = "audio_visual"

    UNKNOWN = "unknown"


@dataclass
class CADStandards:
    """CAD standards for an element type."""
    layer: str
    color: int = 7  # White/Black (depending on background)
    linetype: str = "CONTINUOUS"
    lineweight: float = 0.25  # mm
    block_name: str | None = None
    attributes: dict[str, str] = field(default_factory=dict)
    discipline: Discipline = Discipline.UNKNOWN
    system: SystemType = SystemType.UNKNOWN
    description: str = ""
    source: str = "default"  # "project", "company", or "default"


@dataclass
class GroundedElement:
    """An element with CAD standards applied."""
    element_id: str
    element_type: str
    subtype: str | None = None
    position: tuple[float, float] | None = None
    standards: CADStandards | None = None
    original_data: dict[str, Any] = field(default_factory=dict)


@dataclass
class KnowledgeGroundingResult:
    """Result from knowledge grounding."""
    grounded_elements: list[GroundedElement] = field(default_factory=list)
    standards_applied: int = 0
    standards_missing: int = 0
    statistics: dict[str, Any] = field(default_factory=dict)


# =============================================================================
# Standards Database
# =============================================================================

# NCS-based layer naming convention
# Format: {DISCIPLINE}-{MAJOR}-{MINOR}-{STATUS}
# Example: M-HVAC-DUCT, P-DOMW-PIPE, E-POWR-WIRE

# Color codes (AutoCAD ACI):
# 1=Red, 2=Yellow, 3=Green, 4=Cyan, 5=Blue, 6=Magenta, 7=White/Black, 8=Gray

_DISCIPLINE_PREFIXES = {
    Discipline.ARCHITECTURAL: "A",
    Discipline.MECHANICAL: "M",
    Discipline.ELECTRICAL: "E",
    Discipline.PLUMBING: "P",
    Discipline.FIRE: "F",
    Discipline.LOW_VOLTAGE: "T",  # Telecommunications
    Discipline.CIVIL: "C",
    Discipline.STRUCTURAL: "S",
}

_SYSTEM_LAYER_MAPPING = {
    # HVAC
    SystemType.SUPPLY_AIR: ("M", "HVAC", "SPLY", 5),  # Blue
    SystemType.RETURN_AIR: ("M", "HVAC", "RETN", 4),  # Cyan
    SystemType.EXHAUST_AIR: ("M", "HVAC", "EXHA", 6),  # Magenta
    SystemType.OUTSIDE_AIR: ("M", "HVAC", "OTSD", 3),  # Green
    SystemType.CHILLED_WATER: ("M", "HVAC", "CHWT", 5),  # Blue
    SystemType.HOT_WATER_HEATING: ("M", "HVAC", "HWTH", 1),  # Red
    SystemType.STEAM: ("M", "HVAC", "STEM", 1),  # Red
    SystemType.CONDENSATE: ("M", "HVAC", "COND", 2),  # Yellow
    SystemType.REFRIGERANT: ("M", "HVAC", "REFR", 4),  # Cyan

    # Plumbing
    SystemType.DOMESTIC_COLD_WATER: ("P", "DOMW", "COLD", 5),  # Blue
    SystemType.DOMESTIC_HOT_WATER: ("P", "DOMW", "HOT", 1),  # Red
    SystemType.SANITARY: ("P", "SNTY", "PIPE", 3),  # Green
    SystemType.VENT: ("P", "VENT", "PIPE", 6),  # Magenta
    SystemType.STORM_DRAIN: ("P", "STRM", "PIPE", 4),  # Cyan
    SystemType.NATURAL_GAS: ("P", "NGAS", "PIPE", 2),  # Yellow

    # Electrical
    SystemType.POWER: ("E", "POWR", "WIRE", 1),  # Red
    SystemType.LIGHTING: ("E", "LITE", "WIRE", 2),  # Yellow
    SystemType.EMERGENCY_POWER: ("E", "EMER", "WIRE", 1),  # Red
    SystemType.GROUNDING: ("E", "GRND", "WIRE", 3),  # Green

    # Fire
    SystemType.FIRE_ALARM: ("F", "ALRM", "WIRE", 1),  # Red
    SystemType.FIRE_SUPPRESSION: ("F", "SPKL", "PIPE", 1),  # Red
    SystemType.FIRE_STANDPIPE: ("F", "STPF", "PIPE", 1),  # Red

    # Low Voltage
    SystemType.DATA: ("T", "DATA", "WIRE", 5),  # Blue
    SystemType.VOICE: ("T", "VOIC", "WIRE", 3),  # Green
    SystemType.SECURITY: ("T", "SECR", "WIRE", 1),  # Red
    SystemType.AV: ("T", "AUDV", "WIRE", 6),  # Magenta
}

# Element type to layer suffix mapping
_ELEMENT_LAYER_SUFFIXES = {
    # Mechanical
    "duct": "DUCT",
    "diffuser": "DIFF",
    "grille": "GRIL",
    "register": "REGI",
    "damper": "DAMP",
    "ahu": "EQPM",
    "fan": "EQPM",
    "vav": "EQPM",
    "fcu": "EQPM",
    "coil": "EQPM",
    "pump": "EQPM",

    # Plumbing
    "pipe": "PIPE",
    "valve": "VALV",
    "fitting": "FTTG",
    "fixture": "FIXT",
    "floor_drain": "DRAN",
    "cleanout": "CLEN",
    "water_heater": "EQPM",

    # Electrical
    "outlet": "OUTL",
    "switch": "SWTC",
    "panel": "PANL",
    "transformer": "XFMR",
    "motor": "MOTR",
    "light": "FIXT",
    "receptacle": "RCPT",
    "junction_box": "JBOX",
    "conduit": "COND",
    "wire": "WIRE",

    # Fire
    "smoke_detector": "DETC",
    "heat_detector": "DETC",
    "pull_station": "ANUN",
    "horn_strobe": "ANUN",
    "sprinkler": "SPKL",
    "fire_extinguisher": "EXTN",
    "fdc": "CONN",  # Fire department connection

    # Low Voltage
    "data_outlet": "OUTL",
    "camera": "SECU",
    "card_reader": "SECU",
    "speaker": "AUDV",
}

# Block name patterns by element type and subtype
_BLOCK_NAME_PATTERNS = {
    # Valves
    ("valve", "gate"): "P-VALV-GATE",
    ("valve", "ball"): "P-VALV-BALL",
    ("valve", "butterfly"): "P-VALV-BTRF",
    ("valve", "globe"): "P-VALV-GLOB",
    ("valve", "check"): "P-VALV-CHEK",
    ("valve", "pressure_reducing"): "P-VALV-PRV",
    ("valve", "pressure_relief"): "P-VALV-PRF",
    ("valve", "balancing"): "P-VALV-BAL",
    ("valve", "control"): "P-VALV-CTRL",
    ("valve", "solenoid"): "P-VALV-SOL",
    ("valve", None): "P-VALV",

    # Diffusers
    ("diffuser", "square"): "M-DIFF-SQ",
    ("diffuser", "round"): "M-DIFF-RD",
    ("diffuser", "linear"): "M-DIFF-LN",
    ("diffuser", "slot"): "M-DIFF-SLOT",
    ("diffuser", "perforated"): "M-DIFF-PERF",
    ("diffuser", None): "M-DIFF",

    # Grilles
    ("grille", "return"): "M-GRIL-RTN",
    ("grille", "transfer"): "M-GRIL-XFR",
    ("grille", "exhaust"): "M-GRIL-EXH",
    ("grille", None): "M-GRIL",

    # Dampers
    ("damper", "fire"): "M-DAMP-FIRE",
    ("damper", "smoke"): "M-DAMP-SMOK",
    ("damper", "volume"): "M-DAMP-VOL",
    ("damper", "backdraft"): "M-DAMP-BD",
    ("damper", None): "M-DAMP",

    # Electrical outlets
    ("outlet", "duplex"): "E-OUTL-DUP",
    ("outlet", "quad"): "E-OUTL-QUAD",
    ("outlet", "gfci"): "E-OUTL-GFCI",
    ("outlet", "dedicated"): "E-OUTL-DED",
    ("outlet", "floor"): "E-OUTL-FLR",
    ("outlet", None): "E-OUTL",

    # Switches
    ("switch", "single"): "E-SWTC-1P",
    ("switch", "3way"): "E-SWTC-3W",
    ("switch", "4way"): "E-SWTC-4W",
    ("switch", "dimmer"): "E-SWTC-DIM",
    ("switch", "occupancy"): "E-SWTC-OCC",
    ("switch", None): "E-SWTC",

    # Lights
    ("light", "recessed"): "E-LITE-REC",
    ("light", "pendant"): "E-LITE-PND",
    ("light", "surface"): "E-LITE-SRF",
    ("light", "linear"): "E-LITE-LIN",
    ("light", "emergency"): "E-LITE-EMR",
    ("light", "exit"): "E-LITE-EXIT",
    ("light", None): "E-LITE",

    # Fire detection
    ("smoke_detector", "photoelectric"): "F-DETC-SMOK-P",
    ("smoke_detector", "ionization"): "F-DETC-SMOK-I",
    ("smoke_detector", "duct"): "F-DETC-SMOK-D",
    ("smoke_detector", None): "F-DETC-SMOK",
    ("heat_detector", "fixed_temp"): "F-DETC-HEAT-FT",
    ("heat_detector", "rate_of_rise"): "F-DETC-HEAT-RR",
    ("heat_detector", None): "F-DETC-HEAT",

    # Fire notification
    ("horn_strobe", "wall"): "F-ANUN-HS-W",
    ("horn_strobe", "ceiling"): "F-ANUN-HS-C",
    ("horn_strobe", None): "F-ANUN-HS",
    ("pull_station", None): "F-ANUN-PULL",

    # Sprinklers
    ("sprinkler", "pendant"): "F-SPKL-PND",
    ("sprinkler", "upright"): "F-SPKL-UPR",
    ("sprinkler", "sidewall"): "F-SPKL-SW",
    ("sprinkler", "concealed"): "F-SPKL-CON",
    ("sprinkler", None): "F-SPKL",

    # Plumbing fixtures
    ("fixture", "water_closet"): "P-FIXT-WC",
    ("fixture", "lavatory"): "P-FIXT-LAV",
    ("fixture", "sink"): "P-FIXT-SINK",
    ("fixture", "urinal"): "P-FIXT-URN",
    ("fixture", "shower"): "P-FIXT-SHWR",
    ("fixture", "drinking_fountain"): "P-FIXT-DF",
    ("fixture", None): "P-FIXT",

    # Low voltage
    ("data_outlet", "rj45"): "T-DATA-RJ45",
    ("data_outlet", "fiber"): "T-DATA-FBR",
    ("data_outlet", None): "T-DATA",
    ("camera", "dome"): "T-SECU-CAM-D",
    ("camera", "ptz"): "T-SECU-CAM-PTZ",
    ("camera", "bullet"): "T-SECU-CAM-B",
    ("camera", None): "T-SECU-CAM",
    ("card_reader", None): "T-SECU-CRDR",
    ("speaker", None): "T-AUDV-SPKR",
}

# Default attributes by element type
_DEFAULT_ATTRIBUTES = {
    "valve": {"SIZE": "", "TYPE": "", "TAG": ""},
    "diffuser": {"SIZE": "", "CFM": "", "TAG": ""},
    "grille": {"SIZE": "", "CFM": "", "TAG": ""},
    "outlet": {"CIRCUIT": "", "VOLTAGE": "", "TAG": ""},
    "light": {"WATTS": "", "TYPE": "", "TAG": ""},
    "smoke_detector": {"ZONE": "", "ADDRESS": "", "TAG": ""},
    "sprinkler": {"SIZE": "", "K_FACTOR": "", "TAG": ""},
    "fixture": {"MODEL": "", "GPM": "", "TAG": ""},
}


# =============================================================================
# YAML Standards Loader
# =============================================================================

_YAML_CACHE: dict[str, dict] = {}


def _load_yaml_standards(standards_dir: Path | None = None) -> dict:
    """
    Load CAD standards from YAML files in the knowledge_base directory.

    Args:
        standards_dir: Path to standards directory. Defaults to
                       knowledge_base/cad_standards/ relative to project root.

    Returns:
        Dictionary of standards by discipline/system/element type.
    """
    if standards_dir is None:
        # Find project root (look for pyproject.toml or setup.py)
        current = Path(__file__).resolve()
        for parent in current.parents:
            if (parent / "pyproject.toml").exists() or (parent / "setup.py").exists():
                standards_dir = parent / "knowledge_base" / "cad_standards"
                break
        else:
            # Fallback to relative path
            standards_dir = Path(__file__).parent.parent.parent.parent.parent / "knowledge_base" / "cad_standards"

    cache_key = str(standards_dir)
    if cache_key in _YAML_CACHE:
        return _YAML_CACHE[cache_key]

    standards = {}

    if not standards_dir.exists():
        logger.debug("Standards directory not found, using built-in defaults", path=str(standards_dir))
        _YAML_CACHE[cache_key] = standards
        return standards

    try:
        import yaml
    except ImportError:
        logger.warning("PyYAML not installed, using built-in standards only")
        _YAML_CACHE[cache_key] = standards
        return standards

    for yaml_file in standards_dir.glob("*.yaml"):
        try:
            with open(yaml_file, encoding="utf-8") as f:
                data = yaml.safe_load(f)
                if data:
                    discipline = yaml_file.stem  # e.g., "plumbing" from "plumbing.yaml"
                    standards[discipline] = data
                    logger.debug("Loaded standards", file=yaml_file.name, discipline=discipline)
        except Exception as e:
            logger.warning(f"Failed to load standards file: {yaml_file}", error=str(e))

    _YAML_CACHE[cache_key] = standards
    return standards


# =============================================================================
# Standards Query Functions
# =============================================================================

async def query_cad_standards(
    element_type: str,
    subtype: str | None = None,
    system: str | SystemType | None = None,
    discipline: str | Discipline | None = None,
    size: str | None = None,
    project_standards: dict | None = None,
    company_standards: dict | None = None,
) -> CADStandards:
    """
    Query the knowledge base for CAD standards.

    Looks up layer name, color, linetype, block name, and default attributes
    for the given element type and system.

    Args:
        element_type: Type of element (e.g., "valve", "diffuser", "outlet")
        subtype: Specific subtype (e.g., "gate", "ball", "duplex")
        system: MEP system (e.g., "domestic_cold_water", SystemType.SUPPLY_AIR)
        discipline: AEC discipline (e.g., "plumbing", Discipline.MECHANICAL)
        size: Element size (e.g., '3/4"', "24x24")
        project_standards: Optional project-specific overrides
        company_standards: Optional company-wide overrides

    Returns:
        CADStandards with layer, color, linetype, block_name, and attributes.

    Example:
        >>> standards = await query_cad_standards(
        ...     element_type="valve",
        ...     subtype="gate",
        ...     system="domestic_cold_water",
        ...     size='3/4"',
        ... )
        >>> print(f"Layer: {standards.layer}")  # "P-DOMW-VALV"
        >>> print(f"Block: {standards.block_name}")  # "P-VALV-GATE"
    """
    # Normalize inputs
    element_type = element_type.lower().replace("-", "_").replace(" ", "_") if element_type else "unknown"
    subtype = subtype.lower().replace("-", "_").replace(" ", "_") if subtype else None

    # Convert string system to enum
    if isinstance(system, str):
        system = _parse_system_type(system)
    elif system is None:
        system = _infer_system_from_element(element_type, subtype)

    # Convert string discipline to enum
    if isinstance(discipline, str):
        discipline = _parse_discipline(discipline)
    elif discipline is None:
        discipline = _infer_discipline_from_element(element_type, system)

    logger.debug(
        "Querying CAD standards",
        element_type=element_type,
        subtype=subtype,
        system=system.value if system else None,
        discipline=discipline.value if discipline else None,
    )

    # Priority 1: Project-specific standards
    if project_standards:
        result = _lookup_in_standards(project_standards, element_type, subtype, system, discipline)
        if result:
            result.source = "project"
            return result

    # Priority 2: Company standards
    if company_standards:
        result = _lookup_in_standards(company_standards, element_type, subtype, system, discipline)
        if result:
            result.source = "company"
            return result

    # Priority 3: YAML file standards
    yaml_standards = _load_yaml_standards()
    if yaml_standards:
        result = _lookup_in_yaml_standards(yaml_standards, element_type, subtype, system, discipline)
        if result:
            result.source = "yaml"
            return result

    # Priority 4: Built-in defaults
    return _build_default_standards(element_type, subtype, system, discipline, size)


def _parse_system_type(system_str: str) -> SystemType:
    """Parse a system string to SystemType enum."""
    normalized = system_str.lower().replace("-", "_").replace(" ", "_")

    # Direct match
    try:
        return SystemType(normalized)
    except ValueError:
        pass

    # Common aliases
    aliases = {
        "dcw": SystemType.DOMESTIC_COLD_WATER,
        "dhw": SystemType.DOMESTIC_HOT_WATER,
        "chw": SystemType.CHILLED_WATER,
        "hw": SystemType.HOT_WATER_HEATING,
        "sa": SystemType.SUPPLY_AIR,
        "ra": SystemType.RETURN_AIR,
        "ea": SystemType.EXHAUST_AIR,
        "oa": SystemType.OUTSIDE_AIR,
        "cw": SystemType.CHILLED_WATER,
        "hw": SystemType.HOT_WATER_HEATING,
        "fa": SystemType.FIRE_ALARM,
        "cold": SystemType.DOMESTIC_COLD_WATER,
        "hot": SystemType.DOMESTIC_HOT_WATER,
        "supply": SystemType.SUPPLY_AIR,
        "return": SystemType.RETURN_AIR,
        "exhaust": SystemType.EXHAUST_AIR,
        "gas": SystemType.NATURAL_GAS,
        "sprinkler": SystemType.FIRE_SUPPRESSION,
    }

    if normalized in aliases:
        return aliases[normalized]

    return SystemType.UNKNOWN


def _parse_discipline(discipline_str: str) -> Discipline:
    """Parse a discipline string to Discipline enum."""
    normalized = discipline_str.lower().replace("-", "_").replace(" ", "_")

    try:
        return Discipline(normalized)
    except ValueError:
        pass

    aliases = {
        "hvac": Discipline.MECHANICAL,
        "mech": Discipline.MECHANICAL,
        "elec": Discipline.ELECTRICAL,
        "plmb": Discipline.PLUMBING,
        "fire_alarm": Discipline.FIRE,
        "fire_protection": Discipline.FIRE,
        "fp": Discipline.FIRE,
        "telecom": Discipline.LOW_VOLTAGE,
        "tel": Discipline.LOW_VOLTAGE,
        "data": Discipline.LOW_VOLTAGE,
        "security": Discipline.LOW_VOLTAGE,
        "arch": Discipline.ARCHITECTURAL,
        "struct": Discipline.STRUCTURAL,
    }

    if normalized in aliases:
        return aliases[normalized]

    return Discipline.UNKNOWN


def _infer_system_from_element(element_type: str, subtype: str | None) -> SystemType:
    """Infer the system type from element type."""
    type_to_system = {
        # Valves/pipes default to plumbing cold water
        "valve": SystemType.DOMESTIC_COLD_WATER,
        "pipe": SystemType.DOMESTIC_COLD_WATER,
        "fitting": SystemType.DOMESTIC_COLD_WATER,

        # HVAC elements
        "duct": SystemType.SUPPLY_AIR,
        "diffuser": SystemType.SUPPLY_AIR,
        "grille": SystemType.RETURN_AIR,
        "register": SystemType.SUPPLY_AIR,
        "damper": SystemType.SUPPLY_AIR,
        "ahu": SystemType.SUPPLY_AIR,
        "vav": SystemType.SUPPLY_AIR,
        "fcu": SystemType.SUPPLY_AIR,

        # Electrical
        "outlet": SystemType.POWER,
        "receptacle": SystemType.POWER,
        "switch": SystemType.LIGHTING,
        "light": SystemType.LIGHTING,
        "panel": SystemType.POWER,
        "transformer": SystemType.POWER,

        # Fire
        "smoke_detector": SystemType.FIRE_ALARM,
        "heat_detector": SystemType.FIRE_ALARM,
        "pull_station": SystemType.FIRE_ALARM,
        "horn_strobe": SystemType.FIRE_ALARM,
        "sprinkler": SystemType.FIRE_SUPPRESSION,

        # Plumbing fixtures
        "fixture": SystemType.SANITARY,
        "floor_drain": SystemType.SANITARY,
        "cleanout": SystemType.SANITARY,

        # Low voltage
        "data_outlet": SystemType.DATA,
        "camera": SystemType.SECURITY,
        "card_reader": SystemType.SECURITY,
        "speaker": SystemType.AV,
    }

    return type_to_system.get(element_type, SystemType.UNKNOWN)


def _infer_discipline_from_element(element_type: str, system: SystemType | None) -> Discipline:
    """Infer the discipline from element type or system."""
    type_to_discipline = {
        # Mechanical
        "duct": Discipline.MECHANICAL,
        "diffuser": Discipline.MECHANICAL,
        "grille": Discipline.MECHANICAL,
        "damper": Discipline.MECHANICAL,
        "ahu": Discipline.MECHANICAL,
        "vav": Discipline.MECHANICAL,
        "fcu": Discipline.MECHANICAL,

        # Electrical
        "outlet": Discipline.ELECTRICAL,
        "receptacle": Discipline.ELECTRICAL,
        "switch": Discipline.ELECTRICAL,
        "light": Discipline.ELECTRICAL,
        "panel": Discipline.ELECTRICAL,
        "conduit": Discipline.ELECTRICAL,

        # Plumbing
        "valve": Discipline.PLUMBING,
        "pipe": Discipline.PLUMBING,
        "fixture": Discipline.PLUMBING,
        "floor_drain": Discipline.PLUMBING,

        # Fire
        "smoke_detector": Discipline.FIRE,
        "heat_detector": Discipline.FIRE,
        "sprinkler": Discipline.FIRE,
        "pull_station": Discipline.FIRE,
        "horn_strobe": Discipline.FIRE,

        # Low voltage
        "data_outlet": Discipline.LOW_VOLTAGE,
        "camera": Discipline.LOW_VOLTAGE,
        "card_reader": Discipline.LOW_VOLTAGE,
        "speaker": Discipline.LOW_VOLTAGE,
    }

    if element_type in type_to_discipline:
        return type_to_discipline[element_type]

    # Infer from system
    if system:
        system_to_discipline = {
            SystemType.SUPPLY_AIR: Discipline.MECHANICAL,
            SystemType.RETURN_AIR: Discipline.MECHANICAL,
            SystemType.EXHAUST_AIR: Discipline.MECHANICAL,
            SystemType.CHILLED_WATER: Discipline.MECHANICAL,
            SystemType.HOT_WATER_HEATING: Discipline.MECHANICAL,
            SystemType.DOMESTIC_COLD_WATER: Discipline.PLUMBING,
            SystemType.DOMESTIC_HOT_WATER: Discipline.PLUMBING,
            SystemType.SANITARY: Discipline.PLUMBING,
            SystemType.POWER: Discipline.ELECTRICAL,
            SystemType.LIGHTING: Discipline.ELECTRICAL,
            SystemType.FIRE_ALARM: Discipline.FIRE,
            SystemType.FIRE_SUPPRESSION: Discipline.FIRE,
            SystemType.DATA: Discipline.LOW_VOLTAGE,
            SystemType.SECURITY: Discipline.LOW_VOLTAGE,
        }
        if system in system_to_discipline:
            return system_to_discipline[system]

    return Discipline.UNKNOWN


def _lookup_in_standards(
    standards: dict,
    element_type: str,
    subtype: str | None,
    system: SystemType | None,
    discipline: Discipline | None,
) -> CADStandards | None:
    """Look up standards in a dictionary structure."""
    # Try discipline.element_type.subtype path
    if discipline and discipline != Discipline.UNKNOWN:
        disc_key = discipline.value
        if disc_key in standards:
            disc_standards = standards[disc_key]
            if element_type in disc_standards:
                elem_standards = disc_standards[element_type]
                if subtype and subtype in elem_standards:
                    return _standards_from_dict(elem_standards[subtype], system, discipline)
                elif "default" in elem_standards:
                    return _standards_from_dict(elem_standards["default"], system, discipline)
                elif isinstance(elem_standards, dict) and "layer" in elem_standards:
                    return _standards_from_dict(elem_standards, system, discipline)

    return None


def _lookup_in_yaml_standards(
    yaml_standards: dict,
    element_type: str,
    subtype: str | None,
    system: SystemType | None,
    discipline: Discipline | None,
) -> CADStandards | None:
    """Look up standards in YAML-loaded dictionary."""
    # The YAML structure is: {discipline: {systems: {...}, elements: {...}}}
    if discipline and discipline.value in yaml_standards:
        disc_data = yaml_standards[discipline.value]

        # Check elements section
        if "elements" in disc_data and element_type in disc_data["elements"]:
            elem_data = disc_data["elements"][element_type]

            if subtype and subtype in elem_data:
                return _standards_from_yaml_element(elem_data[subtype], system, discipline, disc_data)
            elif isinstance(elem_data, dict):
                return _standards_from_yaml_element(elem_data, system, discipline, disc_data)

    return None


def _standards_from_dict(data: dict, system: SystemType | None, discipline: Discipline | None) -> CADStandards:
    """Create CADStandards from a dictionary."""
    return CADStandards(
        layer=data.get("layer", "0"),
        color=data.get("color", 7),
        linetype=data.get("linetype", "CONTINUOUS"),
        lineweight=data.get("lineweight", 0.25),
        block_name=data.get("block_name"),
        attributes=data.get("attributes", {}),
        discipline=discipline or Discipline.UNKNOWN,
        system=system or SystemType.UNKNOWN,
        description=data.get("description", ""),
    )


def _standards_from_yaml_element(
    elem_data: dict,
    system: SystemType | None,
    discipline: Discipline | None,
    disc_data: dict,
) -> CADStandards:
    """Create CADStandards from YAML element data."""
    # Build layer from system and suffix
    layer_suffix = elem_data.get("layer_suffix", "")
    layer_prefix = ""
    color = 7

    if system and system in _SYSTEM_LAYER_MAPPING:
        prefix, major, minor, sys_color = _SYSTEM_LAYER_MAPPING[system]
        layer_prefix = f"{prefix}-{major}-{minor}"
        color = sys_color
    elif discipline and discipline in _DISCIPLINE_PREFIXES:
        layer_prefix = _DISCIPLINE_PREFIXES[discipline]

    layer = layer_prefix + layer_suffix if layer_suffix else layer_prefix or "0"

    return CADStandards(
        layer=elem_data.get("layer", layer),
        color=elem_data.get("color", color),
        linetype=elem_data.get("linetype", "CONTINUOUS"),
        lineweight=elem_data.get("lineweight", 0.25),
        block_name=elem_data.get("block_name"),
        attributes=elem_data.get("attributes", {}),
        discipline=discipline or Discipline.UNKNOWN,
        system=system or SystemType.UNKNOWN,
        description=elem_data.get("description", ""),
    )


def _build_default_standards(
    element_type: str,
    subtype: str | None,
    system: SystemType | None,
    discipline: Discipline | None,
    size: str | None,
) -> CADStandards:
    """Build CAD standards using built-in defaults."""
    # Determine layer name
    layer_prefix = ""
    layer_suffix = ""
    color = 7

    if system and system in _SYSTEM_LAYER_MAPPING:
        prefix, major, minor, sys_color = _SYSTEM_LAYER_MAPPING[system]
        layer_prefix = f"{prefix}-{major}"
        color = sys_color
    elif discipline and discipline in _DISCIPLINE_PREFIXES:
        layer_prefix = _DISCIPLINE_PREFIXES[discipline]
    else:
        layer_prefix = "0"

    # Get element-specific suffix
    layer_suffix = _ELEMENT_LAYER_SUFFIXES.get(element_type, "")

    # Build full layer name
    if layer_suffix:
        layer = f"{layer_prefix}-{layer_suffix}"
    else:
        layer = layer_prefix

    # Get block name
    block_name = None
    if (element_type, subtype) in _BLOCK_NAME_PATTERNS:
        block_name = _BLOCK_NAME_PATTERNS[(element_type, subtype)]
    elif (element_type, None) in _BLOCK_NAME_PATTERNS:
        block_name = _BLOCK_NAME_PATTERNS[(element_type, None)]

    # Get default attributes
    attributes = dict(_DEFAULT_ATTRIBUTES.get(element_type, {}))
    if size and "SIZE" in attributes:
        attributes["SIZE"] = size

    return CADStandards(
        layer=layer,
        color=color,
        linetype="CONTINUOUS",
        lineweight=0.25,
        block_name=block_name,
        attributes=attributes,
        discipline=discipline or Discipline.UNKNOWN,
        system=system or SystemType.UNKNOWN,
        description=f"Default standards for {element_type}",
        source="default",
    )


# =============================================================================
# Element Grounding Functions
# =============================================================================

async def ground_element(
    element: Any,
    project_standards: dict | None = None,
    company_standards: dict | None = None,
) -> GroundedElement:
    """
    Apply CAD standards to a single element.

    Args:
        element: Element object with type, subtype, position, etc.
        project_standards: Optional project-specific standards
        company_standards: Optional company standards

    Returns:
        GroundedElement with standards applied.
    """
    # Extract element properties
    element_type = getattr(element, "type", None) or getattr(element, "element_type", "unknown")
    subtype = getattr(element, "subtype", None)
    system = getattr(element, "system", None)
    discipline = getattr(element, "discipline", None)
    size = getattr(element, "size", None)
    position = getattr(element, "position", None) or getattr(element, "center", None)
    element_id = getattr(element, "id", None) or str(id(element))

    # Query standards
    standards = await query_cad_standards(
        element_type=element_type,
        subtype=subtype,
        system=system,
        discipline=discipline,
        size=size,
        project_standards=project_standards,
        company_standards=company_standards,
    )

    # Create grounded element
    return GroundedElement(
        element_id=element_id,
        element_type=element_type,
        subtype=subtype,
        position=position if isinstance(position, tuple) else None,
        standards=standards,
        original_data={
            "system": system,
            "discipline": discipline,
            "size": size,
        },
    )


async def ground_elements(
    elements: list[Any],
    project_standards: dict | None = None,
    company_standards: dict | None = None,
) -> KnowledgeGroundingResult:
    """
    Apply CAD standards to multiple elements.

    Args:
        elements: List of elements to ground
        project_standards: Optional project-specific standards
        company_standards: Optional company standards

    Returns:
        KnowledgeGroundingResult with all grounded elements.

    Example:
        >>> result = await ground_elements(smart_symbols)
        >>> for elem in result.grounded_elements:
        ...     print(f"{elem.element_type} -> Layer: {elem.standards.layer}")
    """
    result = KnowledgeGroundingResult()

    logger.info("Grounding elements with CAD standards", count=len(elements))

    standards_by_source: dict[str, int] = {}

    for element in elements:
        try:
            grounded = await ground_element(
                element,
                project_standards=project_standards,
                company_standards=company_standards,
            )
            result.grounded_elements.append(grounded)

            if grounded.standards:
                result.standards_applied += 1
                source = grounded.standards.source
                standards_by_source[source] = standards_by_source.get(source, 0) + 1
            else:
                result.standards_missing += 1

        except Exception as e:
            logger.warning(f"Failed to ground element: {e}")
            result.standards_missing += 1

    # Compute statistics
    result.statistics = {
        "total_elements": len(elements),
        "standards_applied": result.standards_applied,
        "standards_missing": result.standards_missing,
        "by_source": standards_by_source,
        "coverage_percent": (result.standards_applied / len(elements) * 100) if elements else 0,
    }

    logger.info(
        "Knowledge grounding complete",
        applied=result.standards_applied,
        missing=result.standards_missing,
        coverage=f"{result.statistics['coverage_percent']:.1f}%",
    )

    return result


# =============================================================================
# Convenience Functions
# =============================================================================

def get_layer_for_element(
    element_type: str,
    system: str | SystemType | None = None,
) -> str:
    """
    Synchronous helper to get layer name for an element type.

    Args:
        element_type: Type of element
        system: Optional system type

    Returns:
        Layer name string.
    """
    if isinstance(system, str):
        system = _parse_system_type(system)

    if system is None:
        system = _infer_system_from_element(element_type, None)

    layer_prefix = ""
    layer_suffix = ""

    if system and system in _SYSTEM_LAYER_MAPPING:
        prefix, major, minor, _ = _SYSTEM_LAYER_MAPPING[system]
        layer_prefix = f"{prefix}-{major}"

    layer_suffix = _ELEMENT_LAYER_SUFFIXES.get(element_type, "")

    if layer_suffix:
        return f"{layer_prefix}-{layer_suffix}"
    return layer_prefix or "0"


def get_block_name(
    element_type: str,
    subtype: str | None = None,
) -> str | None:
    """
    Synchronous helper to get block name for an element.

    Args:
        element_type: Type of element
        subtype: Optional subtype

    Returns:
        Block name or None if not found.
    """
    element_type = element_type.lower().replace("-", "_").replace(" ", "_")
    subtype = subtype.lower().replace("-", "_").replace(" ", "_") if subtype else None

    if (element_type, subtype) in _BLOCK_NAME_PATTERNS:
        return _BLOCK_NAME_PATTERNS[(element_type, subtype)]
    if (element_type, None) in _BLOCK_NAME_PATTERNS:
        return _BLOCK_NAME_PATTERNS[(element_type, None)]
    return None


def get_color_for_system(system: str | SystemType) -> int:
    """
    Get the standard color for a system type.

    Args:
        system: System type string or enum

    Returns:
        AutoCAD ACI color number.
    """
    if isinstance(system, str):
        system = _parse_system_type(system)

    if system and system in _SYSTEM_LAYER_MAPPING:
        _, _, _, color = _SYSTEM_LAYER_MAPPING[system]
        return color

    return 7  # White/Black default


# =============================================================================
# Gemini LLM Integration for Advanced Standards Query
# =============================================================================

@dataclass
class LLMQueryResult:
    """Result from LLM-based standards query."""
    answer: str
    element_type: str | None = None
    subtype: str | None = None
    system: str | None = None
    discipline: str | None = None
    recommended_layer: str | None = None
    recommended_block: str | None = None
    recommended_color: int | None = None
    attributes: dict[str, str] = field(default_factory=dict)
    code_references: list[str] = field(default_factory=list)
    confidence: float = 0.0
    provider: str = ""
    raw_response: str = ""


class KnowledgeLLM:
    """
    LLM-powered knowledge query for CAD standards and building codes.

    Uses Google Gemini (or fallback providers) for intelligent standards
    lookup, element classification recommendations, and code compliance hints.

    Supports:
    - Natural language queries about CAD standards
    - Element classification from text descriptions
    - Attribute and specification extraction
    - Building code reference suggestions

    Example:
        >>> llm = KnowledgeLLM()
        >>> result = await llm.query("What layer should a 3/4 inch gate valve be on?")
        >>> print(result.recommended_layer)  # "P-DOMW-VALV"
    """

    def __init__(
        self,
        provider: str = "auto",
        temperature: float = 0.2,
        max_tokens: int = 1000,
    ):
        """
        Initialize KnowledgeLLM.

        Args:
            provider: LLM provider ("gemini", "openai", "anthropic", "auto")
            temperature: Response temperature (lower = more deterministic)
            max_tokens: Maximum response tokens
        """
        self.provider = provider.lower()
        self.temperature = temperature
        self.max_tokens = max_tokens

        # Lazy-loaded clients
        self._gemini_model = None
        self._openai_client = None
        self._anthropic_client = None

        # Load settings
        try:
            from aec_agent.config.settings import get_settings
            self._settings = get_settings()
        except ImportError:
            self._settings = None

    def _get_api_key(self, provider: str) -> str | None:
        """Get API key for a provider."""
        if not self._settings:
            # Fallback to environment variables
            import os
            if provider == "gemini":
                return os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
            elif provider == "openai":
                return os.environ.get("OPENAI_API_KEY")
            elif provider == "anthropic":
                return os.environ.get("ANTHROPIC_API_KEY")
            return None

        if provider == "gemini":
            return self._settings.gemini_api_key
        elif provider == "openai":
            return self._settings.openai_api_key
        elif provider == "anthropic":
            return self._settings.anthropic_api_key
        return None

    def _get_available_provider(self) -> str | None:
        """Get the first available provider based on API keys."""
        if self.provider != "auto":
            if self._get_api_key(self.provider):
                return self.provider
            return None

        # Try providers in order of preference (cost-effective first)
        for provider in ["gemini", "openai", "anthropic"]:
            if self._get_api_key(provider):
                return provider
        return None

    async def _init_gemini(self):
        """Initialize Gemini client."""
        if self._gemini_model is None:
            try:
                import google.generativeai as genai

                api_key = self._get_api_key("gemini")
                if not api_key:
                    raise ValueError("GEMINI_API_KEY not configured")

                genai.configure(api_key=api_key)
                self._gemini_model = genai.GenerativeModel(
                    "gemini-2.0-flash",  # Flash has free tier, Pro does not
                    generation_config=genai.GenerationConfig(
                        temperature=self.temperature,
                        max_output_tokens=self.max_tokens,
                    ),
                )
                logger.debug("Gemini model initialized for knowledge query")
            except ImportError:
                raise ImportError(
                    "google-generativeai not installed. "
                    "Install with: pip install google-generativeai"
                )
        return self._gemini_model

    async def _init_openai(self):
        """Initialize OpenAI client."""
        if self._openai_client is None:
            try:
                from openai import AsyncOpenAI

                api_key = self._get_api_key("openai")
                if not api_key:
                    raise ValueError("OPENAI_API_KEY not configured")

                self._openai_client = AsyncOpenAI(api_key=api_key)
                logger.debug("OpenAI client initialized for knowledge query")
            except ImportError:
                raise ImportError(
                    "openai not installed. Install with: pip install openai"
                )
        return self._openai_client

    async def _init_anthropic(self):
        """Initialize Anthropic client."""
        if self._anthropic_client is None:
            try:
                import anthropic

                api_key = self._get_api_key("anthropic")
                if not api_key:
                    raise ValueError("ANTHROPIC_API_KEY not configured")

                self._anthropic_client = anthropic.AsyncAnthropic(api_key=api_key)
                logger.debug("Anthropic client initialized for knowledge query")
            except ImportError:
                raise ImportError(
                    "anthropic not installed. Install with: pip install anthropic"
                )
        return self._anthropic_client

    def _build_system_prompt(self) -> str:
        """Build the system prompt for CAD standards queries."""
        return """AEC CAD standards expert. Knowledge: NCS layers, MEP symbols, CA codes (CMC/CEC/CPC/CFC), NFPA 72.

Layer format: {Discipline}-{Major}-{Minor} (A/M/E/P/F/T - HVAC/DOMW/POWR/ALRM/DATA - VALV/DIFF/OUTL/DETC)
Colors: Blue(5)=cold/supply/data, Red(1)=hot/power/fire, Cyan(4)=return/storm, Green(3)=sanitary/voice, Magenta(6)=vent/exhaust, Yellow(2)=gas/lighting

Respond JSON only:
{"answer": "explanation", "element_type": "valve|diffuser|outlet|detector|etc", "subtype": "gate|ball|square|duplex|etc", "system": "domestic_cold_water|supply_air|power|fire_alarm|etc", "discipline": "plumbing|mechanical|electrical|fire|low_voltage", "recommended_layer": "P-DOMW-VALV", "recommended_block": "P-VALV-GATE", "recommended_color": 5, "attributes": {"SIZE": "", "TAG": ""}, "code_references": ["CPC 604.1"], "confidence": 0.95}"""

    async def query(self, question: str) -> LLMQueryResult:
        """
        Query the LLM for CAD standards information.

        Args:
            question: Natural language question about CAD standards

        Returns:
            LLMQueryResult with answer and recommendations

        Example:
            >>> result = await llm.query("What layer for a smoke detector?")
            >>> print(result.recommended_layer)  # "F-ALRM-DETC"
        """
        provider = self._get_available_provider()
        if not provider:
            return LLMQueryResult(
                answer="No LLM provider configured. Set GEMINI_API_KEY, OPENAI_API_KEY, or ANTHROPIC_API_KEY.",
                confidence=0.0,
            )

        logger.info("Querying knowledge LLM", provider=provider, question=question[:100])

        try:
            if provider == "gemini":
                return await self._query_gemini(question)
            elif provider == "openai":
                return await self._query_openai(question)
            elif provider == "anthropic":
                return await self._query_anthropic(question)
            else:
                return LLMQueryResult(
                    answer=f"Unknown provider: {provider}",
                    confidence=0.0,
                )
        except Exception as e:
            logger.error(f"LLM query failed: {e}")
            return LLMQueryResult(
                answer=f"Error querying LLM: {str(e)}",
                confidence=0.0,
            )

    async def _query_gemini(self, question: str) -> LLMQueryResult:
        """Query Gemini for CAD standards."""
        model = await self._init_gemini()

        prompt = f"{self._build_system_prompt()}\n\nUser question: {question}"

        response = await model.generate_content_async(prompt)
        raw_text = response.text

        return self._parse_llm_response(raw_text, "gemini")

    async def _query_openai(self, question: str) -> LLMQueryResult:
        """Query OpenAI for CAD standards."""
        client = await self._init_openai()

        response = await client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": self._build_system_prompt()},
                {"role": "user", "content": question},
            ],
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )

        raw_text = response.choices[0].message.content
        return self._parse_llm_response(raw_text, "openai")

    async def _query_anthropic(self, question: str) -> LLMQueryResult:
        """Query Anthropic for CAD standards."""
        client = await self._init_anthropic()

        response = await client.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=self.max_tokens,
            system=self._build_system_prompt(),
            messages=[{"role": "user", "content": question}],
        )

        raw_text = response.content[0].text
        return self._parse_llm_response(raw_text, "anthropic")

    def _parse_llm_response(self, raw_text: str, provider: str) -> LLMQueryResult:
        """Parse LLM response JSON into LLMQueryResult."""
        import json
        import re

        # Try to extract JSON from the response
        try:
            # Look for JSON block in markdown code fence
            json_match = re.search(r'```(?:json)?\s*({\s*".*?}\s*)```', raw_text, re.DOTALL)
            if json_match:
                json_str = json_match.group(1)
            else:
                # Try to find raw JSON
                json_match = re.search(r'(\{[^{}]*"answer"[^{}]*\})', raw_text, re.DOTALL)
                if json_match:
                    json_str = json_match.group(1)
                else:
                    # Fallback: try parsing the whole response
                    json_str = raw_text.strip()

            data = json.loads(json_str)

            return LLMQueryResult(
                answer=data.get("answer", raw_text),
                element_type=data.get("element_type"),
                subtype=data.get("subtype"),
                system=data.get("system"),
                discipline=data.get("discipline"),
                recommended_layer=data.get("recommended_layer"),
                recommended_block=data.get("recommended_block"),
                recommended_color=data.get("recommended_color"),
                attributes=data.get("attributes", {}),
                code_references=data.get("code_references", []),
                confidence=data.get("confidence", 0.8),
                provider=provider,
                raw_response=raw_text,
            )

        except (json.JSONDecodeError, AttributeError):
            # If JSON parsing fails, return the raw text as the answer
            return LLMQueryResult(
                answer=raw_text,
                confidence=0.5,
                provider=provider,
                raw_response=raw_text,
            )

    async def classify_element(self, description: str) -> LLMQueryResult:
        """
        Classify an element from a text description.

        Args:
            description: Text description of the element (e.g., "3/4 inch gate valve on cold water")

        Returns:
            LLMQueryResult with element classification and standards

        Example:
            >>> result = await llm.classify_element("24x24 supply air diffuser 200 CFM")
            >>> print(result.element_type)  # "diffuser"
            >>> print(result.subtype)  # "square"
            >>> print(result.recommended_block)  # "M-DIFF-SQ"
        """
        question = f"""Classify this MEP element and provide CAD standards:

Description: {description}

Identify the element type, subtype, system, and provide the correct layer, block name, color, and any attributes that should be set."""

        return await self.query(question)

    async def get_code_reference(
        self,
        element_type: str,
        context: str | None = None,
    ) -> LLMQueryResult:
        """
        Get relevant building code references for an element.

        Args:
            element_type: Type of element (e.g., "smoke detector", "sprinkler")
            context: Optional context (e.g., "corridor", "high-rise", "assembly")

        Returns:
            LLMQueryResult with code references

        Example:
            >>> result = await llm.get_code_reference("smoke detector", "corridor")
            >>> print(result.code_references)  # ["NFPA 72 17.7.3.2", "CFC 907.2"]
        """
        question = f"""What are the relevant California building code requirements for a {element_type}?
{f'Context: {context}' if context else ''}

Focus on:
- California Fire Code (CFC)
- NFPA 72 (if fire-related)
- California Mechanical Code (CMC) (if HVAC)
- California Plumbing Code (CPC) (if plumbing)
- California Electrical Code (CEC) (if electrical)
- Title 24 energy requirements (if applicable)

Provide specific code section references."""

        return await self.query(question)

    async def suggest_attributes(
        self,
        element_type: str,
        subtype: str | None = None,
        system: str | None = None,
    ) -> LLMQueryResult:
        """
        Suggest attributes for an element block.

        Args:
            element_type: Type of element
            subtype: Optional subtype
            system: Optional system type

        Returns:
            LLMQueryResult with suggested attributes

        Example:
            >>> result = await llm.suggest_attributes("diffuser", "square", "supply_air")
            >>> print(result.attributes)  # {"SIZE": "24x24", "CFM": "", "NC": "", "TAG": ""}
        """
        question = f"""What attributes should be included in a CAD block for:
Element type: {element_type}
{f'Subtype: {subtype}' if subtype else ''}
{f'System: {system}' if system else ''}

List the standard attributes with their typical formats and units."""

        return await self.query(question)


# Convenience function for quick queries
async def query_with_llm(
    question: str,
    provider: str = "auto",
) -> LLMQueryResult:
    """
    Quick query function for CAD standards questions.

    Args:
        question: Natural language question
        provider: LLM provider to use

    Returns:
        LLMQueryResult with answer and recommendations

    Example:
        >>> result = await query_with_llm("What color for hot water pipes?")
        >>> print(result.answer)  # "Hot water pipes use color 1 (Red)..."
    """
    llm = KnowledgeLLM(provider=provider)
    return await llm.query(question)
