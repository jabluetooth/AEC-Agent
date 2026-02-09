"""
Semantic OCR for AEC drawings.

Transforms raw OCR text into structured, meaningful data.
This is Layer 3 (Text Stream) of the Semantic Intelligence Pipeline.

Raw OCR outputs strings like "24x24 SA 200 CFM" - this module
parses them into structured data: {type: "supply_air_diffuser",
width: 24, height: 24, cfm: 200}.

Key capabilities:
1. Pattern-based parsing for common AEC annotations
2. LLM-based parsing for complex/ambiguous text
3. Context-aware interpretation based on drawing type
"""

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import structlog

from .document_classifier import DrawingType

logger = structlog.get_logger(__name__)


class AnnotationType(str, Enum):
    """Type of text annotation in a drawing."""
    # Identifiers
    ROOM_NAME = "room_name"             # "CONFERENCE ROOM", "LOBBY"
    ROOM_NUMBER = "room_number"         # "101", "ROOM 101"
    EQUIPMENT_TAG = "equipment_tag"     # "AHU-1", "VAV-101", "P-1"

    # Measurements
    AREA = "area"                       # "245 SF", "22.8 M2"
    DIMENSION = "dimension"             # "10'-0\"", "3048mm"
    ELEVATION = "elevation"             # "EL. 10'-6\"", "+10.50"

    # Equipment specs
    EQUIPMENT_SPEC = "equipment_spec"   # "24x24 SA 200 CFM"
    SIZE = "size"                       # "3/4\"", "100mm", "24x24"
    FLOW_RATE = "flow_rate"             # "200 CFM", "100 L/s"
    POWER = "power"                     # "5 HP", "3.7 kW"
    CAPACITY = "capacity"               # "10 TON", "35 kW"

    # References
    REFERENCE = "reference"             # "SEE DETAIL A", "TYP."
    QUANTITY = "quantity"               # "(3)", "TYP. (5)"
    SHEET_REF = "sheet_ref"             # "A1.01", "M-101"
    SECTION_REF = "section_ref"         # "1/A1.01", "SECTION A"

    # Notes
    NOTE = "note"                       # General notes
    SPECIFICATION = "specification"     # Spec references

    # System
    CIRCUIT_ID = "circuit_id"           # "CKT-1A", "20A/1P"
    PANEL_NAME = "panel_name"           # "PANEL LP-1", "MDP"

    # Unknown
    UNKNOWN = "unknown"


@dataclass
class ParsedAnnotation:
    """
    A parsed text annotation with structured data.

    Attributes:
        raw_text: Original OCR text
        annotation_type: Classified type of annotation
        structured_data: Parsed data as dictionary
        confidence: Confidence in the parsing (0.0-1.0)
        position: (x, y) position in the drawing
        bounding_box: (x, y, width, height) if available
    """
    raw_text: str
    annotation_type: AnnotationType
    structured_data: dict[str, Any]
    confidence: float
    position: tuple[float, float] = (0.0, 0.0)
    bounding_box: tuple[float, float, float, float] | None = None
    source: str = "pattern"  # "pattern" or "llm"
    metadata: dict[str, Any] = field(default_factory=dict)


# Pattern definitions for common AEC annotations
# Format: (regex_pattern, annotation_type, extraction_function)
ANNOTATION_PATTERNS: list[tuple[str, AnnotationType, str]] = [
    # Room identifiers
    (r'^ROOM\s*#?\s*(\d+[A-Z]?)$', AnnotationType.ROOM_NUMBER, 'parse_room_number'),
    (r'^(\d{2,4}[A-Z]?)$', AnnotationType.ROOM_NUMBER, 'parse_room_number'),  # "101", "101A"

    # Areas
    (r'^(\d+(?:\.\d+)?)\s*(?:SF|SQ\.?\s*FT\.?|FT2?)$', AnnotationType.AREA, 'parse_area_sf'),
    (r'^(\d+(?:\.\d+)?)\s*(?:M2|SQ\.?\s*M\.?)$', AnnotationType.AREA, 'parse_area_m2'),

    # Dimensions (imperial)
    (r"^(\d+)['′]\s*[-]?\s*(\d+(?:\d+/\d+)?)[\"″]?$", AnnotationType.DIMENSION, 'parse_dimension_imperial'),
    (r'^(\d+(?:\d+/\d+)?)[\"″]$', AnnotationType.DIMENSION, 'parse_dimension_inches'),

    # Dimensions (metric)
    (r'^(\d+(?:\.\d+)?)\s*(?:mm|MM)$', AnnotationType.DIMENSION, 'parse_dimension_mm'),
    (r'^(\d+(?:\.\d+)?)\s*(?:m|M)$', AnnotationType.DIMENSION, 'parse_dimension_m'),

    # Elevations
    (r'^(?:EL\.?|ELEV\.?)\s*(\d+)[\'′]\s*[-]?\s*(\d+(?:\d+/\d+)?)[\"″]?$', AnnotationType.ELEVATION, 'parse_elevation_imperial'),
    (r'^(?:EL\.?|ELEV\.?)\s*[+]?(\d+(?:\.\d+)?)$', AnnotationType.ELEVATION, 'parse_elevation_decimal'),
    (r'^[+](\d+(?:\.\d+)?)$', AnnotationType.ELEVATION, 'parse_elevation_decimal'),

    # Circuit IDs (must come before equipment tags to avoid false matches)
    (r'^CKT[-\s]?([A-Z]?\d+[A-Z]?)$', AnnotationType.CIRCUIT_ID, 'parse_circuit_id'),
    (r'^(\d+)[Aa]/([13])P$', AnnotationType.CIRCUIT_ID, 'parse_circuit_breaker'),

    # Equipment tags (common patterns)
    (r'^(AHU|VAV|FCU|RTU|ERU|MAU|EF|SF|RF|HWP|CWP|CHW|HW|P|FD|SD|BD|MD)[-_]?(\d+[A-Z]?)$',
     AnnotationType.EQUIPMENT_TAG, 'parse_equipment_tag'),
    (r'^([A-Z]{2,4})[-_]?(\d{1,4}[A-Z]?)$', AnnotationType.EQUIPMENT_TAG, 'parse_equipment_tag'),

    # Diffuser/grille specs: "24x24 SA 200 CFM"
    (r'^(\d+)\s*[xX×]\s*(\d+)\s*(SA|RA|EA|OA|EX)\s*(\d+)\s*CFM$',
     AnnotationType.EQUIPMENT_SPEC, 'parse_diffuser_spec'),

    # Simple size: "24x24"
    (r'^(\d+)\s*[xX×]\s*(\d+)$', AnnotationType.SIZE, 'parse_size_wxh'),

    # Flow rates
    (r'^(\d+(?:\.\d+)?)\s*CFM$', AnnotationType.FLOW_RATE, 'parse_cfm'),
    (r'^(\d+(?:\.\d+)?)\s*(?:L/S|LPS|L/MIN)$', AnnotationType.FLOW_RATE, 'parse_lps'),

    # Pipe/valve sizes
    (r'^(\d+(?:/\d+)?)[\"″]?\s*(GV|BV|CV|PRV|CK|CKV)$', AnnotationType.EQUIPMENT_SPEC, 'parse_valve_spec'),
    (r'^(\d+(?:/\d+)?)[\"″]$', AnnotationType.SIZE, 'parse_pipe_size'),

    # Power
    (r'^(\d+(?:\.\d+)?)\s*(?:HP|hp)$', AnnotationType.POWER, 'parse_hp'),
    (r'^(\d+(?:\.\d+)?)\s*(?:kW|KW|kw)$', AnnotationType.POWER, 'parse_kw'),

    # Capacity
    (r'^(\d+(?:\.\d+)?)\s*(?:TON|TONS?)$', AnnotationType.CAPACITY, 'parse_tons'),

    # Quantities
    (r'^\((\d+)\)$', AnnotationType.QUANTITY, 'parse_quantity'),
    (r'^TYP\.?\s*\((\d+)\)$', AnnotationType.QUANTITY, 'parse_typical_quantity'),
    (r'^TYP\.?$', AnnotationType.QUANTITY, 'parse_typical'),

    # Panel names
    (r'^(?:PANEL\s+)?([LM]?[PD]?P?[-]?\d*[A-Z]?)$', AnnotationType.PANEL_NAME, 'parse_panel_name'),

    # Sheet references
    (r'^([AEMPSFCL][-.]?\d{1,3}[.-]?\d{0,2})$', AnnotationType.SHEET_REF, 'parse_sheet_ref'),

    # Section references
    (r'^(\d+)/([AEMPSFCL][-.]?\d{1,3}[.-]?\d{0,2})$', AnnotationType.SECTION_REF, 'parse_section_ref'),
    (r'^(?:SECTION|SEC\.?|DETAIL|DET\.?)\s+([A-Z0-9]+)$', AnnotationType.SECTION_REF, 'parse_section_letter'),
]


def parse_annotation_by_patterns(
    text: str,
    drawing_type: DrawingType | None = None,
) -> ParsedAnnotation | None:
    """
    Parse annotation using pattern matching (fast, no LLM cost).

    Args:
        text: Raw OCR text to parse
        drawing_type: Optional drawing type for context

    Returns:
        ParsedAnnotation if a pattern matches, None otherwise
    """
    text = text.strip().upper()

    if not text:
        return None

    for pattern, anno_type, parse_func in ANNOTATION_PATTERNS:
        match = re.match(pattern, text, re.IGNORECASE)
        if match:
            # Get the parsing function
            parser = globals().get(parse_func)
            if parser:
                try:
                    structured_data = parser(match)
                    return ParsedAnnotation(
                        raw_text=text,
                        annotation_type=anno_type,
                        structured_data=structured_data,
                        confidence=0.9,  # High confidence for pattern match
                        source="pattern",
                    )
                except Exception as e:
                    logger.debug(f"Pattern parse failed: {parse_func} - {e}")
                    continue

    return None


# Parsing functions for each pattern type
def parse_room_number(match: re.Match) -> dict:
    """Parse room number."""
    room = match.group(1) if match.lastindex >= 1 else match.group(0)
    return {"room_number": room.strip()}


def parse_area_sf(match: re.Match) -> dict:
    """Parse area in square feet."""
    return {"area_sf": float(match.group(1)), "unit": "sf"}


def parse_area_m2(match: re.Match) -> dict:
    """Parse area in square meters."""
    return {"area_m2": float(match.group(1)), "unit": "m2"}


def parse_dimension_imperial(match: re.Match) -> dict:
    """Parse imperial dimension (feet-inches)."""
    feet = int(match.group(1))
    inches_str = match.group(2)

    # Handle fractions
    if '/' in inches_str:
        parts = inches_str.split('/')
        if len(parts) == 2:
            inches = float(parts[0]) / float(parts[1])
        else:
            inches = float(inches_str.replace('/', '.'))
    else:
        inches = float(inches_str) if inches_str else 0

    total_inches = feet * 12 + inches
    return {
        "feet": feet,
        "inches": inches,
        "total_inches": total_inches,
        "unit": "imperial",
    }


def parse_dimension_inches(match: re.Match) -> dict:
    """Parse dimension in inches only."""
    inches_str = match.group(1)
    if '/' in inches_str:
        parts = inches_str.split('/')
        inches = float(parts[0]) / float(parts[1])
    else:
        inches = float(inches_str)
    return {"inches": inches, "unit": "imperial"}


def parse_dimension_mm(match: re.Match) -> dict:
    """Parse dimension in millimeters."""
    return {"mm": float(match.group(1)), "unit": "metric"}


def parse_dimension_m(match: re.Match) -> dict:
    """Parse dimension in meters."""
    return {"m": float(match.group(1)), "unit": "metric"}


def parse_elevation_imperial(match: re.Match) -> dict:
    """Parse elevation in imperial units."""
    feet = int(match.group(1))
    inches_str = match.group(2)
    if '/' in inches_str:
        parts = inches_str.split('/')
        inches = float(parts[0]) / float(parts[1])
    else:
        inches = float(inches_str) if inches_str else 0
    return {"elevation_ft": feet + inches / 12, "unit": "imperial"}


def parse_elevation_decimal(match: re.Match) -> dict:
    """Parse elevation as decimal."""
    return {"elevation": float(match.group(1))}


def parse_equipment_tag(match: re.Match) -> dict:
    """Parse equipment tag (prefix + number)."""
    prefix = match.group(1).upper()
    number = match.group(2)

    # Determine equipment category from prefix
    category = _get_equipment_category(prefix)

    return {
        "tag": f"{prefix}-{number}",
        "prefix": prefix,
        "number": number,
        "category": category,
    }


def _get_equipment_category(prefix: str) -> str:
    """Map equipment prefix to category."""
    categories = {
        'AHU': 'air_handling_unit',
        'VAV': 'vav_box',
        'FCU': 'fan_coil_unit',
        'RTU': 'rooftop_unit',
        'ERU': 'energy_recovery_unit',
        'MAU': 'makeup_air_unit',
        'EF': 'exhaust_fan',
        'SF': 'supply_fan',
        'RF': 'return_fan',
        'HWP': 'hot_water_pump',
        'CWP': 'chilled_water_pump',
        'P': 'pump',
        'FD': 'fire_damper',
        'SD': 'smoke_damper',
        'BD': 'backdraft_damper',
        'MD': 'motorized_damper',
    }
    return categories.get(prefix, 'equipment')


def parse_diffuser_spec(match: re.Match) -> dict:
    """Parse diffuser specification (WxH TYPE CFM)."""
    width = int(match.group(1))
    height = int(match.group(2))
    air_type = match.group(3).upper()
    cfm = int(match.group(4))

    type_map = {
        'SA': 'supply_air',
        'RA': 'return_air',
        'EA': 'exhaust_air',
        'OA': 'outside_air',
        'EX': 'exhaust_air',
    }

    return {
        "width_inches": width,
        "height_inches": height,
        "air_type": type_map.get(air_type, air_type),
        "cfm": cfm,
        "equipment_type": "diffuser",
    }


def parse_size_wxh(match: re.Match) -> dict:
    """Parse WxH size."""
    return {
        "width": int(match.group(1)),
        "height": int(match.group(2)),
    }


def parse_cfm(match: re.Match) -> dict:
    """Parse CFM flow rate."""
    return {"cfm": float(match.group(1)), "unit": "cfm"}


def parse_lps(match: re.Match) -> dict:
    """Parse L/s flow rate."""
    return {"lps": float(match.group(1)), "unit": "lps"}


def parse_valve_spec(match: re.Match) -> dict:
    """Parse valve specification (size + type)."""
    size_str = match.group(1)
    valve_type = match.group(2).upper()

    # Parse size (may include fraction)
    if '/' in size_str:
        parts = size_str.split('/')
        size_inches = float(parts[0]) / float(parts[1])
    else:
        size_inches = float(size_str)

    type_map = {
        'GV': 'gate_valve',
        'BV': 'ball_valve',
        'CV': 'check_valve',
        'PRV': 'pressure_reducing_valve',
        'CK': 'check_valve',
        'CKV': 'check_valve',
    }

    return {
        "size_inches": size_inches,
        "valve_type": type_map.get(valve_type, valve_type),
        "equipment_type": "valve",
    }


def parse_pipe_size(match: re.Match) -> dict:
    """Parse pipe size."""
    size_str = match.group(1)
    if '/' in size_str:
        parts = size_str.split('/')
        size_inches = float(parts[0]) / float(parts[1])
    else:
        size_inches = float(size_str)
    return {"size_inches": size_inches, "type": "pipe"}


def parse_hp(match: re.Match) -> dict:
    """Parse horsepower."""
    return {"hp": float(match.group(1)), "unit": "hp"}


def parse_kw(match: re.Match) -> dict:
    """Parse kilowatts."""
    return {"kw": float(match.group(1)), "unit": "kw"}


def parse_tons(match: re.Match) -> dict:
    """Parse cooling capacity in tons."""
    return {"tons": float(match.group(1)), "unit": "ton"}


def parse_quantity(match: re.Match) -> dict:
    """Parse quantity."""
    return {"quantity": int(match.group(1)), "typical": False}


def parse_typical_quantity(match: re.Match) -> dict:
    """Parse typical quantity."""
    return {"quantity": int(match.group(1)), "typical": True}


def parse_typical(match: re.Match) -> dict:
    """Parse TYP. marker."""
    return {"typical": True}


def parse_circuit_id(match: re.Match) -> dict:
    """Parse circuit ID."""
    return {"circuit_id": match.group(1)}


def parse_circuit_breaker(match: re.Match) -> dict:
    """Parse circuit breaker spec (amps/poles)."""
    return {
        "amps": int(match.group(1)),
        "poles": int(match.group(2)),
    }


def parse_panel_name(match: re.Match) -> dict:
    """Parse panel name."""
    return {"panel_name": match.group(1)}


def parse_sheet_ref(match: re.Match) -> dict:
    """Parse sheet reference."""
    return {"sheet": match.group(1)}


def parse_section_ref(match: re.Match) -> dict:
    """Parse section reference (number/sheet)."""
    return {
        "detail_number": match.group(1),
        "sheet": match.group(2),
    }


def parse_section_letter(match: re.Match) -> dict:
    """Parse section/detail letter reference."""
    return {"section": match.group(1)}


async def parse_annotation_with_llm(
    text: str,
    drawing_type: DrawingType,
    nearby_context: list[str] | None = None,
    llm_provider: str = "groq",
) -> ParsedAnnotation:
    """
    Parse annotation using LLM for complex/ambiguous text.

    Args:
        text: Raw OCR text to parse
        drawing_type: Type of drawing for context
        nearby_context: Nearby text for additional context
        llm_provider: LLM provider to use

    Returns:
        ParsedAnnotation with structured data
    """
    from .document_classifier import _call_llm
    from ..prompts.annotation_parsing import get_annotation_parsing_prompt

    prompt = get_annotation_parsing_prompt(text, drawing_type, nearby_context)

    try:
        response = await _call_llm(prompt, llm_provider)

        # Parse JSON response
        import json

        # Handle markdown code blocks
        if "```" in response:
            start = response.find("{")
            end = response.rfind("}") + 1
            if start >= 0 and end > start:
                response = response[start:end]

        result = json.loads(response)

        # Map to AnnotationType
        type_str = result.get("annotation_type", "unknown")
        try:
            anno_type = AnnotationType(type_str)
        except ValueError:
            anno_type = AnnotationType.UNKNOWN

        return ParsedAnnotation(
            raw_text=text,
            annotation_type=anno_type,
            structured_data=result.get("structured_data", {}),
            confidence=float(result.get("confidence", 0.5)),
            source="llm",
            metadata={"reasoning": result.get("reasoning", "")},
        )

    except Exception as e:
        logger.warning(f"LLM annotation parsing failed: {e}")
        return ParsedAnnotation(
            raw_text=text,
            annotation_type=AnnotationType.UNKNOWN,
            structured_data={},
            confidence=0.0,
            source="llm_failed",
        )


async def parse_annotations(
    texts: list[dict[str, Any]],
    drawing_type: DrawingType,
    use_llm: bool = True,
    llm_provider: str = "groq",
    llm_threshold: float = 0.5,
) -> list[ParsedAnnotation]:
    """
    Parse a batch of OCR text annotations.

    Uses pattern matching first (fast, free), then falls back to
    LLM for text that doesn't match any pattern.

    Args:
        texts: List of dicts with 'text' and optional 'position', 'bbox'
        drawing_type: Type of drawing for context
        use_llm: Whether to use LLM for unmatched text
        llm_provider: LLM provider to use
        llm_threshold: Minimum text length to send to LLM

    Returns:
        List of ParsedAnnotation objects
    """
    results: list[ParsedAnnotation] = []
    llm_queue: list[tuple[int, dict]] = []

    for i, text_data in enumerate(texts):
        text = text_data.get("text", "").strip()
        position = text_data.get("position", (0.0, 0.0))
        bbox = text_data.get("bbox")

        if not text:
            continue

        # Try pattern matching first
        parsed = parse_annotation_by_patterns(text, drawing_type)

        if parsed:
            parsed.position = position
            parsed.bounding_box = bbox
            results.append(parsed)
        elif use_llm and len(text) >= 3:
            # Queue for LLM processing
            llm_queue.append((i, text_data))
        else:
            # Unknown, no LLM
            results.append(ParsedAnnotation(
                raw_text=text,
                annotation_type=AnnotationType.UNKNOWN,
                structured_data={},
                confidence=0.0,
                position=position,
                bounding_box=bbox,
                source="no_match",
            ))

    # Process LLM queue (could batch for efficiency)
    for idx, text_data in llm_queue:
        text = text_data.get("text", "")
        position = text_data.get("position", (0.0, 0.0))
        bbox = text_data.get("bbox")

        # Get nearby context
        nearby = []
        for other in texts:
            if other != text_data and other.get("text"):
                nearby.append(other["text"])
                if len(nearby) >= 3:
                    break

        parsed = await parse_annotation_with_llm(
            text, drawing_type, nearby, llm_provider
        )
        parsed.position = position
        parsed.bounding_box = bbox
        results.append(parsed)

    logger.info(
        "Parsed annotations",
        total=len(texts),
        pattern_matched=len([r for r in results if r.source == "pattern"]),
        llm_parsed=len([r for r in results if r.source == "llm"]),
        unknown=len([r for r in results if r.annotation_type == AnnotationType.UNKNOWN]),
    )

    return results


def classify_room_name(text: str) -> bool:
    """
    Check if text looks like a room name.

    Room names are typically:
    - All caps or title case
    - Common room types (OFFICE, CONFERENCE, LOBBY, etc.)
    - Not numbers or dimensions

    Args:
        text: Text to check

    Returns:
        True if text appears to be a room name
    """
    text = text.strip().upper()

    # Common room name patterns
    room_keywords = [
        'OFFICE', 'CONFERENCE', 'MEETING', 'LOBBY', 'CORRIDOR', 'HALLWAY',
        'RESTROOM', 'BATHROOM', 'TOILET', 'KITCHEN', 'BREAK', 'LOUNGE',
        'STORAGE', 'CLOSET', 'MECHANICAL', 'ELECTRICAL', 'TELECOM', 'IDF',
        'MDF', 'JANITOR', 'COPY', 'MAIL', 'RECEPTION', 'WAITING',
        'ELEVATOR', 'STAIR', 'VESTIBULE', 'ATRIUM', 'COMMONS',
        'OPEN OFFICE', 'WORKSTATION', 'CUBICLE', 'PRIVATE',
        'CLASSROOM', 'LAB', 'LABORATORY', 'EXAM', 'PATIENT',
        'NURSE', 'DOCTOR', 'TREATMENT', 'RECOVERY', 'SURGERY',
    ]

    for keyword in room_keywords:
        if keyword in text:
            return True

    # Check for "ROOM" suffix without number
    if text.endswith(' ROOM') or text.startswith('ROOM '):
        if not re.search(r'\d', text):
            return True

    return False


def get_annotation_category(anno_type: AnnotationType) -> str:
    """
    Get the general category for an annotation type.

    Categories help group related annotations for processing.

    Args:
        anno_type: The annotation type

    Returns:
        Category string
    """
    categories = {
        # Identifiers
        AnnotationType.ROOM_NAME: "identifier",
        AnnotationType.ROOM_NUMBER: "identifier",
        AnnotationType.EQUIPMENT_TAG: "identifier",

        # Measurements
        AnnotationType.AREA: "measurement",
        AnnotationType.DIMENSION: "measurement",
        AnnotationType.ELEVATION: "measurement",

        # Equipment
        AnnotationType.EQUIPMENT_SPEC: "equipment",
        AnnotationType.SIZE: "equipment",
        AnnotationType.FLOW_RATE: "equipment",
        AnnotationType.POWER: "equipment",
        AnnotationType.CAPACITY: "equipment",

        # References
        AnnotationType.REFERENCE: "reference",
        AnnotationType.QUANTITY: "reference",
        AnnotationType.SHEET_REF: "reference",
        AnnotationType.SECTION_REF: "reference",

        # Electrical
        AnnotationType.CIRCUIT_ID: "electrical",
        AnnotationType.PANEL_NAME: "electrical",

        # Notes
        AnnotationType.NOTE: "text",
        AnnotationType.SPECIFICATION: "text",

        AnnotationType.UNKNOWN: "unknown",
    }

    return categories.get(anno_type, "unknown")
