"""
Prompt templates for Vision LLM symbol classification.

These prompts are used by vision_llm.py to classify symbols detected
by YOLOv8 into specific subtypes using vision-capable LLMs.

Phase C: Symbol Intelligence - Two-stage classification pipeline:
1. YOLOv8 detects symbol locations and generic classes
2. Vision LLM identifies specific subtypes for ambiguous symbols

The prompts are context-aware based on:
- Drawing type (electrical, HVAC, plumbing, etc.)
- Initial YOLO classification
- Nearby text annotations
"""

from ..mcp.tools.document_classifier import DrawingType


# =============================================================================
# Symbol Categories and Subtypes
# =============================================================================

# Comprehensive symbol subtype mappings by category
SYMBOL_SUBTYPES: dict[str, dict[str, list[str]]] = {
    "mechanical": {
        "valve": [
            "gate_valve", "ball_valve", "butterfly_valve", "check_valve",
            "globe_valve", "pressure_relief_valve", "control_valve",
            "solenoid_valve", "balancing_valve", "isolation_valve",
        ],
        "diffuser": [
            "supply_air_diffuser", "return_air_grille", "exhaust_grille",
            "linear_diffuser", "slot_diffuser", "perforated_diffuser",
            "swirl_diffuser", "jet_diffuser",
        ],
        "damper": [
            "fire_damper", "smoke_damper", "volume_damper",
            "backdraft_damper", "control_damper",
        ],
        "equipment": [
            "air_handler_unit", "fan_coil_unit", "vav_box", "cav_box",
            "exhaust_fan", "supply_fan", "pump", "chiller", "boiler",
            "cooling_tower", "heat_exchanger",
        ],
        "fitting": [
            "duct_elbow", "duct_tee", "duct_reducer", "duct_cap",
            "flex_connector", "turning_vane",
        ],
    },
    "electrical": {
        "outlet": [
            "duplex_outlet", "gfci_outlet", "quad_outlet",
            "floor_outlet", "weatherproof_outlet", "usb_outlet",
            "dedicated_outlet", "isolated_ground_outlet",
        ],
        "switch": [
            "single_pole_switch", "three_way_switch", "four_way_switch",
            "dimmer_switch", "occupancy_sensor", "timer_switch",
            "key_switch", "emergency_switch",
        ],
        "light": [
            "recessed_downlight", "surface_mount_light", "pendant_light",
            "track_light", "exit_sign", "emergency_light",
            "wall_sconce", "under_cabinet_light", "troffer",
        ],
        "panel": [
            "main_distribution_panel", "load_center", "subpanel",
            "transformer", "disconnect_switch", "transfer_switch",
            "motor_control_center",
        ],
        "device": [
            "junction_box", "pull_box", "floor_box",
            "receptacle", "motor", "generator",
        ],
    },
    "fire": {
        "detection": [
            "smoke_detector", "heat_detector", "duct_smoke_detector",
            "beam_detector", "flame_detector", "multi_sensor_detector",
        ],
        "notification": [
            "horn_strobe", "speaker_strobe", "chime",
            "visual_only", "horn_only", "speaker",
        ],
        "suppression": [
            "sprinkler_head_pendant", "sprinkler_head_upright",
            "sprinkler_head_sidewall", "sprinkler_head_concealed",
            "fire_extinguisher", "standpipe_connection",
        ],
        "control": [
            "fire_alarm_control_panel", "remote_annunciator",
            "pull_station", "flow_switch", "tamper_switch",
        ],
    },
    "plumbing": {
        "valve": [
            "gate_valve", "ball_valve", "check_valve", "globe_valve",
            "pressure_reducing_valve", "backflow_preventer",
            "mixing_valve", "shutoff_valve",
        ],
        "fixture": [
            "water_closet", "urinal", "lavatory", "sink",
            "drinking_fountain", "shower", "bathtub", "mop_sink",
            "floor_drain", "roof_drain", "cleanout",
        ],
        "equipment": [
            "water_heater", "booster_pump", "sump_pump",
            "grease_interceptor", "water_softener", "expansion_tank",
        ],
        "fitting": [
            "elbow", "tee", "reducer", "cap", "union", "coupling",
            "floor_drain", "cleanout", "backwater_valve",
        ],
    },
    "low_voltage": {
        "data": [
            "data_outlet", "fiber_outlet", "coax_outlet",
            "wireless_access_point", "network_switch",
        ],
        "security": [
            "security_camera", "card_reader", "motion_sensor",
            "door_contact", "glass_break_sensor", "intercom",
        ],
        "audio_visual": [
            "speaker", "display_connection", "projector_location",
            "microphone", "volume_control",
        ],
        "control": [
            "thermostat", "occupancy_sensor", "daylight_sensor",
            "keypad", "touch_panel",
        ],
    },
}


def get_symbol_classification_prompt(
    yolo_class: str,
    yolo_category: str,
    drawing_type: DrawingType,
    nearby_text: list[str] | None = None,
    yolo_confidence: float = 0.0,
) -> str:
    """
    Generate a Vision LLM prompt for detailed symbol classification.

    Args:
        yolo_class: Initial YOLO classification (e.g., "valve", "outlet")
        yolo_category: Category from YOLO (e.g., "mechanical", "electrical")
        drawing_type: Type of drawing for context
        nearby_text: Nearby text annotations for context
        yolo_confidence: YOLO detection confidence

    Returns:
        Prompt string for Vision LLM
    """
    nearby_str = ""
    if nearby_text:
        nearby_str = f"\nNearby text annotations: {', '.join(nearby_text[:5])}"

    # Get possible subtypes for this class
    subtypes = []
    if yolo_category in SYMBOL_SUBTYPES:
        if yolo_class in SYMBOL_SUBTYPES[yolo_category]:
            subtypes = SYMBOL_SUBTYPES[yolo_category][yolo_class]

    subtypes_hint = ""
    if subtypes:
        subtypes_hint = f"\nPossible subtypes for {yolo_class}: {', '.join(subtypes[:10])}"

    discipline_hints = _get_discipline_context(drawing_type)

    return f"""Analyze this MEP symbol from an engineering drawing and classify it precisely.

Context:
- Drawing type: {drawing_type.value}
- Initial detection: {yolo_class} (category: {yolo_category}, confidence: {yolo_confidence:.2f}){nearby_str}{subtypes_hint}

{discipline_hints}

Examine the symbol's visual characteristics:
1. Shape and geometry (circle, rectangle, diamond, etc.)
2. Internal markings (lines, arrows, fill patterns)
3. Connection points and orientation
4. Any visible text or labels within the symbol

Return a JSON object with your classification:
{{
    "category": "{yolo_category}",
    "type": "{yolo_class}",
    "subtype": "specific subtype name",
    "direction": "up|down|left|right|none",
    "size_hint": "size if visible from text or scale, null otherwise",
    "system": "system type if identifiable (e.g., domestic_cold_water, supply_air)",
    "confidence": 0.0-1.0,
    "visual_features": ["list", "of", "observed", "features"],
    "reasoning": "Brief explanation of classification"
}}

Important:
- Be specific with subtype - prefer "gate_valve" over generic "valve"
- If uncertain, set confidence < 0.7 and provide best guess
- Use nearby text to inform classification (e.g., "GV" = gate valve)
- Return ONLY valid JSON, no markdown or explanation"""


def get_unknown_symbol_prompt(
    drawing_type: DrawingType,
    nearby_text: list[str] | None = None,
) -> str:
    """
    Generate a prompt for classifying a completely unknown symbol.

    Used when YOLO doesn't recognize the symbol at all.

    Args:
        drawing_type: Type of drawing for context
        nearby_text: Nearby text annotations

    Returns:
        Prompt string for Vision LLM
    """
    nearby_str = ""
    if nearby_text:
        nearby_str = f"\nNearby text: {', '.join(nearby_text[:5])}"

    discipline_hints = _get_discipline_context(drawing_type)

    return f"""Identify this symbol from a {drawing_type.value} engineering drawing.
{nearby_str}

{discipline_hints}

Analyze the symbol's visual characteristics and identify:
1. What MEP category it belongs to (mechanical, electrical, plumbing, fire, low_voltage)
2. What type of element it represents
3. Any specific subtype based on visual features
4. Orientation and connection points

Return a JSON object:
{{
    "category": "mechanical|electrical|plumbing|fire|low_voltage|architectural|unknown",
    "type": "element type (valve, outlet, diffuser, detector, etc.)",
    "subtype": "specific subtype if identifiable",
    "direction": "up|down|left|right|none",
    "size_hint": "size if visible, null otherwise",
    "system": "system type if identifiable, null otherwise",
    "confidence": 0.0-1.0,
    "visual_features": ["list", "of", "observed", "features"],
    "reasoning": "Brief explanation"
}}

Return ONLY valid JSON, no markdown or explanation."""


def get_batch_symbol_prompt(
    symbols: list[dict],
    drawing_type: DrawingType,
) -> str:
    """
    Generate a prompt for classifying multiple symbols at once.

    More efficient for large batches as it reduces API calls.

    Args:
        symbols: List of dicts with 'yolo_class', 'yolo_category', 'nearby_text'
        drawing_type: Type of drawing for context

    Returns:
        Prompt string for Vision LLM
    """
    symbols_desc = []
    for i, sym in enumerate(symbols[:10]):  # Limit to 10 per batch
        nearby = ", ".join(sym.get("nearby_text", [])[:3]) or "none"
        symbols_desc.append(
            f"{i+1}. {sym.get('yolo_class', 'unknown')} "
            f"(category: {sym.get('yolo_category', 'unknown')}, "
            f"nearby text: {nearby})"
        )

    symbols_text = "\n".join(symbols_desc)
    discipline_hints = _get_discipline_context(drawing_type)

    return f"""Classify these symbols from a {drawing_type.value} engineering drawing.

Symbols to classify:
{symbols_text}

{discipline_hints}

For each symbol, identify the specific subtype based on visual features.

Return a JSON array:
[
    {{
        "index": 1,
        "category": "category",
        "type": "type",
        "subtype": "specific subtype",
        "direction": "up|down|left|right|none",
        "system": "system type or null",
        "confidence": 0.0-1.0,
        "reasoning": "brief explanation"
    }},
    ...
]

Return ONLY valid JSON array, no markdown or explanation."""


def _get_discipline_context(drawing_type: DrawingType) -> str:
    """Get discipline-specific context for the prompt."""
    contexts = {
        DrawingType.HVAC_PLAN: """HVAC Symbol Guide:
- Diffusers: Square/rectangular for supply (SA), louvered for return (RA)
- Dampers: Rectangle with diagonal line, fire dampers have "F" or "FD" marking
- VAV boxes: Rectangle with "VAV" or variable arrow symbol
- Valves: Various shapes (gate=bowtie, ball=circle with line, butterfly=circle with wings)
- Equipment: Rectangles with text labels (AHU, FCU, RTU, etc.)

Air flow indicators:
- SA = Supply Air (to room)
- RA = Return Air (from room)
- EA = Exhaust Air
- OA = Outside Air""",

        DrawingType.ELECTRICAL_PLAN: """Electrical Symbol Guide:
- Outlets: Circles with lines (duplex=2 parallel lines, GFCI=GFI text or wavy line)
- Switches: "S" with modifiers (S3=3-way, SD=dimmer, SK=key)
- Lights: Various shapes (circle=downlight, rectangle=troffer, X=recessed)
- Panels: Rectangle with text label, sometimes with busbar symbol
- Junction boxes: Square or octagon, often labeled "J"

Circuit indicators:
- Numbers indicate circuit and phase
- "IG" = Isolated Ground
- "WP" = Weatherproof""",

        DrawingType.PLUMBING_PLAN: """Plumbing Symbol Guide:
- Valves: Gate=bowtie, Ball=circle with line through, Check=arrow with circle
- Fixtures: Symbolic shapes (WC=toilet outline, LAV=sink bowl, UR=urinal)
- Drains: Circle with crosshairs (FD=floor drain, RD=roof drain, CO=cleanout)
- Equipment: Rectangles with labels (WH=water heater, EXP=expansion tank)

Pipe system indicators:
- DCW = Domestic Cold Water (usually blue)
- DHW = Domestic Hot Water (usually red)
- W = Waste
- V = Vent
- S = Sanitary""",

        DrawingType.FIRE_ALARM: """Fire Alarm Symbol Guide:
- Smoke detectors: Circle with "S" or dot pattern
- Heat detectors: Circle with "H" or temperature rating
- Pull stations: Rectangle with handle symbol
- Horn/strobes: Rectangle with speaker and flash symbol
- Sprinklers: Circle with spray pattern, orientation indicated

Device types:
- SD = Smoke Detector
- HD = Heat Detector
- HS = Horn/Strobe
- PS = Pull Station
- FACP = Fire Alarm Control Panel""",

        DrawingType.FLOOR_PLAN: """Architectural Symbol Guide:
- Doors: Arc showing swing direction
- Windows: Parallel lines or cross pattern
- Walls: Parallel lines with fill
- Stairs: Arrow showing up direction
- Elevators: Rectangle with X or E

Room indicators:
- Room names typically in uppercase
- Room numbers often near door
- Areas shown in SF or M2""",
    }

    return contexts.get(drawing_type, """General MEP Symbol Guide:
- Symbols are standardized per discipline
- Look for text labels near symbols for identification
- Connection lines indicate system flow
- Sizes often annotated nearby""")


def get_valve_classification_prompt(
    valve_image_description: str,
    nearby_text: list[str] | None = None,
) -> str:
    """
    Specialized prompt for valve subtype classification.

    Valves are common and have many subtypes - this prompt focuses
    specifically on distinguishing valve types.

    Args:
        valve_image_description: Description of visual features
        nearby_text: Nearby text annotations

    Returns:
        Prompt string for Vision LLM
    """
    nearby_str = ""
    if nearby_text:
        nearby_str = f"\nNearby text: {', '.join(nearby_text[:5])}"

    return f"""Classify this valve symbol from an MEP drawing.
{nearby_str}

Valve Visual Guide:
1. Gate Valve: Bowtie/hourglass shape (two triangles meeting at points)
2. Ball Valve: Circle with a line through center
3. Butterfly Valve: Circle with curved wings/lines inside
4. Check Valve: Triangle/arrow pointing in flow direction
5. Globe Valve: Circle with horizontal line and stem
6. Pressure Relief: Spring symbol with arrow
7. Control Valve: Diamond shape or valve with controller symbol
8. Solenoid Valve: Coil symbol near valve body
9. Balancing Valve: Diagonal line through valve symbol

Additional identifiers:
- "GV" = Gate Valve
- "BV" = Ball Valve
- "CV" = Check Valve or Control Valve
- "PRV" = Pressure Relief Valve

Return JSON:
{{
    "subtype": "gate_valve|ball_valve|butterfly_valve|check_valve|globe_valve|pressure_relief_valve|control_valve|solenoid_valve|balancing_valve|unknown",
    "size_inches": number or null,
    "system": "water system type if identifiable",
    "normally_open": true|false|null,
    "confidence": 0.0-1.0,
    "visual_features": ["observed", "features"],
    "reasoning": "explanation"
}}

Return ONLY valid JSON."""


def get_diffuser_classification_prompt(
    nearby_text: list[str] | None = None,
) -> str:
    """
    Specialized prompt for diffuser/grille subtype classification.

    Args:
        nearby_text: Nearby text annotations (often contain CFM, size)

    Returns:
        Prompt string for Vision LLM
    """
    nearby_str = ""
    if nearby_text:
        nearby_str = f"\nNearby text: {', '.join(nearby_text[:5])}"

    return f"""Classify this air diffuser/grille symbol from an HVAC drawing.
{nearby_str}

Diffuser Visual Guide:
1. Square/Rectangular Diffuser: For supply air (SA), often 4-way throw
2. Linear Slot Diffuser: Long narrow rectangle, for continuous air distribution
3. Return Air Grille: Louvered pattern, for return air (RA)
4. Exhaust Grille: Similar to return, marked EA or with exhaust symbol
5. Perforated Diffuser: Dotted pattern inside rectangle
6. Swirl Diffuser: Circular with spiral pattern
7. Jet Diffuser: Circular with directional indicator

Size annotations typically show:
- WxH format (e.g., "24x24")
- CFM flow rate (e.g., "200 CFM")
- Air type (SA, RA, EA, OA)

Return JSON:
{{
    "subtype": "supply_air_diffuser|return_air_grille|exhaust_grille|linear_diffuser|slot_diffuser|perforated_diffuser|swirl_diffuser|jet_diffuser",
    "width_inches": number or null,
    "height_inches": number or null,
    "cfm": number or null,
    "air_type": "supply|return|exhaust|outside|null",
    "throw_pattern": "4-way|1-way|2-way|radial|null",
    "confidence": 0.0-1.0,
    "visual_features": ["observed", "features"],
    "reasoning": "explanation"
}}

Return ONLY valid JSON."""


def get_outlet_classification_prompt(
    nearby_text: list[str] | None = None,
) -> str:
    """
    Specialized prompt for electrical outlet subtype classification.

    Args:
        nearby_text: Nearby text annotations

    Returns:
        Prompt string for Vision LLM
    """
    nearby_str = ""
    if nearby_text:
        nearby_str = f"\nNearby text: {', '.join(nearby_text[:5])}"

    return f"""Classify this electrical outlet symbol from an electrical drawing.
{nearby_str}

Outlet Visual Guide:
1. Duplex Outlet: Circle with two parallel vertical lines
2. GFCI Outlet: Circle with "GFI" or "GFCI" text, or wavy line
3. Quad Outlet: Circle with four lines or "QUAD" text
4. Floor Outlet: Circle with "F" or shown below floor line
5. Weatherproof: Circle with "WP" or rain symbol
6. Dedicated/Isolated Ground: Circle with triangle or "IG"
7. USB Outlet: Circle with USB symbol or "USB" text

Height indicators:
- AFF = Above Finished Floor
- Standard height vs. counter height

Return JSON:
{{
    "subtype": "duplex_outlet|gfci_outlet|quad_outlet|floor_outlet|weatherproof_outlet|usb_outlet|dedicated_outlet|isolated_ground_outlet",
    "voltage": 120|208|240|277|null,
    "amperage": number or null,
    "height_aff": "string or null",
    "circuit": "circuit ID if visible",
    "special_features": ["isolated_ground", "weatherproof", "floor_mounted"],
    "confidence": 0.0-1.0,
    "visual_features": ["observed", "features"],
    "reasoning": "explanation"
}}

Return ONLY valid JSON."""


def get_detector_classification_prompt(
    nearby_text: list[str] | None = None,
) -> str:
    """
    Specialized prompt for fire detection device classification.

    Args:
        nearby_text: Nearby text annotations

    Returns:
        Prompt string for Vision LLM
    """
    nearby_str = ""
    if nearby_text:
        nearby_str = f"\nNearby text: {', '.join(nearby_text[:5])}"

    return f"""Classify this fire detection device symbol from a fire alarm drawing.
{nearby_str}

Detection Device Visual Guide:
1. Smoke Detector: Circle with "S" or dots pattern
2. Heat Detector: Circle with "H" or temperature rating (e.g., "135F")
3. Duct Smoke Detector: Rectangle with "DSD" or duct symbol
4. Beam Detector: Circle with beam lines
5. Flame Detector: Circle with flame symbol
6. Multi-Sensor: Circle with "MS" or combined symbols

Notification Device Guide:
1. Horn/Strobe: Rectangle with speaker and flash symbol
2. Strobe Only: Rectangle with flash only (ADA visual)
3. Horn Only: Rectangle with speaker symbol
4. Speaker/Strobe: Rectangle with speaker, flash, and audio notes
5. Chime: Bell symbol

Control Device Guide:
1. Pull Station: Rectangle with handle, "PS" or pull symbol
2. FACP: Large rectangle with "FACP" or panel symbol
3. Remote Annunciator: Rectangle with "RA" or display symbol

Return JSON:
{{
    "category": "detection|notification|suppression|control",
    "subtype": "smoke_detector|heat_detector|duct_smoke_detector|horn_strobe|speaker_strobe|pull_station|etc.",
    "zone": "zone ID if visible",
    "slc_address": "SLC address if visible",
    "mounting": "ceiling|wall|duct|floor",
    "candela": number or null (for strobes),
    "confidence": 0.0-1.0,
    "visual_features": ["observed", "features"],
    "reasoning": "explanation"
}}

Return ONLY valid JSON."""
