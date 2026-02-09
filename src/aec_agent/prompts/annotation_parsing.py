"""
Prompt templates for annotation parsing.

These prompts are used by semantic_ocr.py to parse raw OCR text
into structured data using LLMs.

The prompts are context-aware based on:
- Drawing type (electrical, HVAC, plumbing, etc.)
- Nearby text annotations
- Common AEC annotation patterns
"""

from ..mcp.tools.document_classifier import DrawingType


def get_annotation_parsing_prompt(
    text: str,
    drawing_type: DrawingType,
    nearby_context: list[str] | None = None,
) -> str:
    """
    Generate a prompt for parsing a single annotation.

    Args:
        text: Raw OCR text to parse
        drawing_type: Type of drawing for context
        nearby_context: Nearby text for additional context

    Returns:
        Prompt string for LLM
    """
    context_str = ""
    if nearby_context:
        context_str = f"Nearby text: {', '.join(nearby_context[:3])}\n"

    discipline_hints = _get_discipline_hints(drawing_type)

    return f"""Parse this annotation from a {drawing_type.value} engineering drawing.

Annotation text: "{text}"
{context_str}
{discipline_hints}

Identify what type of annotation this is and extract structured data.

Annotation types:
- room_name: Room names like "CONFERENCE ROOM", "LOBBY"
- room_number: Room numbers like "101", "ROOM 205A"
- equipment_tag: Equipment tags like "AHU-1", "VAV-101"
- area: Area measurements like "245 SF", "22.8 M2"
- dimension: Dimensions like "10'-0\"", "3048mm"
- elevation: Elevations like "EL. 10'-6\"", "+10.50"
- equipment_spec: Equipment specs like "24x24 SA 200 CFM", "3/4\" GV"
- size: Size values like "24x24", "3/4\""
- flow_rate: Flow rates like "200 CFM", "100 L/s"
- power: Power specs like "5 HP", "3.7 kW"
- capacity: Capacity specs like "10 TON"
- reference: References like "SEE DETAIL A", "TYP."
- quantity: Quantities like "(3)", "TYP. (5)"
- sheet_ref: Sheet references like "A1.01", "M-101"
- section_ref: Section references like "1/A1.01"
- circuit_id: Circuit IDs like "CKT-1A", "20A/1P"
- panel_name: Panel names like "PANEL LP-1"
- note: General notes
- specification: Spec references
- unknown: Cannot be classified

Return a JSON object:
{{
    "annotation_type": "<type from list above>",
    "structured_data": {{
        // Parsed fields relevant to the type
        // e.g., {{"cfm": 200, "size": "24x24", "air_type": "supply"}}
    }},
    "confidence": 0.0-1.0,
    "reasoning": "Brief explanation"
}}

Return ONLY valid JSON, no markdown or explanation."""


def _get_discipline_hints(drawing_type: DrawingType) -> str:
    """Get discipline-specific hints for the prompt."""
    hints = {
        DrawingType.ELECTRICAL_PLAN: """
Common electrical annotations:
- Circuit IDs: "CKT-1A", "20A/1P", "2-#12 + #12G"
- Panel names: "LP-1", "MDP", "PANEL RP-1A"
- Wattages: "1000W", "5kW"
- Outlet types: "DUPLEX", "QUAD", "GFI"
- Equipment: "TRANSFORMER", "DISCONNECT", "PANEL"
""",
        DrawingType.HVAC_PLAN: """
Common HVAC annotations:
- Air flow: "200 CFM", "1500 CFM"
- Diffuser specs: "24x24 SA 200 CFM" (size + type + flow)
- Equipment tags: "AHU-1", "VAV-101", "FCU-1A"
- Duct sizes: "24x12", "14\" DIA"
- Air types: SA=Supply Air, RA=Return Air, EA=Exhaust Air, OA=Outside Air
""",
        DrawingType.PLUMBING_PLAN: """
Common plumbing annotations:
- Pipe sizes: "3/4\"", "2\"", "4\" DWV"
- Valve types: "GV"=Gate Valve, "BV"=Ball Valve, "CV"=Check Valve
- Fixture counts: "LAV", "WC", "SINK"
- Flow rates: "5 GPM", "10 GPM"
- System types: DCW=Domestic Cold Water, DHW=Domestic Hot Water
""",
        DrawingType.FIRE_ALARM: """
Common fire alarm annotations:
- Device tags: "SD-1" (Smoke Detector), "HD-1" (Heat Detector)
- Zone IDs: "ZONE 1", "Z-1"
- Device types: "FACP", "NAC", "SLC", "HORN/STROBE"
- Circuit types: "SLC", "NAC", "INITIATING"
""",
        DrawingType.FLOOR_PLAN: """
Common architectural annotations:
- Room numbers: "101", "205A", "ROOM 300"
- Room names: "CONFERENCE", "OFFICE", "STORAGE"
- Areas: "245 SF", "100 M2"
- Door/window marks: "D1", "W-1", "HM-01"
""",
    }

    return hints.get(drawing_type, """
General AEC annotations:
- Room identifiers, areas, dimensions
- Equipment tags and specifications
- References to other sheets/details
""")


def get_batch_annotation_prompt(
    texts: list[str],
    drawing_type: DrawingType,
) -> str:
    """
    Generate a prompt for parsing multiple annotations at once.

    More efficient for large batches as it reduces API calls.

    Args:
        texts: List of raw OCR texts to parse
        drawing_type: Type of drawing for context

    Returns:
        Prompt string for LLM
    """
    discipline_hints = _get_discipline_hints(drawing_type)

    texts_formatted = "\n".join([f'{i+1}. "{t}"' for i, t in enumerate(texts[:20])])

    return f"""Parse these annotations from a {drawing_type.value} engineering drawing.

Annotations:
{texts_formatted}

{discipline_hints}

For each annotation, identify type and extract structured data.

Return a JSON array of objects:
[
    {{
        "index": 1,
        "raw_text": "original text",
        "annotation_type": "<type>",
        "structured_data": {{}},
        "confidence": 0.0-1.0
    }},
    ...
]

Annotation types: room_name, room_number, equipment_tag, area, dimension,
elevation, equipment_spec, size, flow_rate, power, capacity, reference,
quantity, sheet_ref, section_ref, circuit_id, panel_name, note, specification, unknown

Return ONLY valid JSON array, no markdown or explanation."""


def get_room_parsing_prompt(text: str, nearby_text: list[str]) -> str:
    """
    Generate a prompt specifically for parsing room-related text.

    Used when we suspect text is room name + number + area.

    Args:
        text: Main text to parse
        nearby_text: Nearby annotations

    Returns:
        Prompt string for LLM
    """
    nearby_str = ", ".join(nearby_text[:5]) if nearby_text else "none"

    return f"""Parse this room-related text from an architectural drawing.

Text: "{text}"
Nearby text: {nearby_str}

Extract any of:
- room_name: The name of the room (e.g., "CONFERENCE ROOM", "OFFICE")
- room_number: The room number (e.g., "101", "205A")
- area_sf: Area in square feet
- area_m2: Area in square meters

Return JSON:
{{
    "room_name": "string or null",
    "room_number": "string or null",
    "area_sf": number or null,
    "area_m2": number or null,
    "confidence": 0.0-1.0
}}

Return ONLY valid JSON."""


def get_equipment_parsing_prompt(
    text: str,
    equipment_type_hint: str | None = None,
) -> str:
    """
    Generate a prompt specifically for parsing equipment specifications.

    Args:
        text: Equipment spec text
        equipment_type_hint: Optional hint about equipment type

    Returns:
        Prompt string for LLM
    """
    hint_str = f"This is likely a {equipment_type_hint}." if equipment_type_hint else ""

    return f"""Parse this equipment specification from an MEP drawing.

Text: "{text}"
{hint_str}

Extract equipment details. Common fields:
- equipment_type: diffuser, grille, valve, damper, etc.
- size: physical dimensions (WxH or diameter)
- cfm: airflow in CFM
- gpm: flow in GPM
- hp: horsepower
- kw: kilowatts
- tons: cooling capacity in tons
- voltage: electrical voltage
- phase: electrical phase (1P, 3P)
- tag: equipment tag/ID

Return JSON with relevant fields:
{{
    "equipment_type": "string",
    // ... other relevant fields
    "confidence": 0.0-1.0
}}

Return ONLY valid JSON."""
