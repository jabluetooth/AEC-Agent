"""
Token optimization for MCP tools.

Provides aggressive compression strategies for tool descriptions and schemas
to minimize token usage with budget LLM providers.
"""

import json
import re
from typing import Any

# =============================================================================
# Schema and Description Caching (Phase 1 optimization)
# =============================================================================
# Caches compressed schemas and descriptions to avoid regeneration
_schema_cache: dict[str, dict] = {}
_description_cache: dict[str, str] = {}


def clear_schema_cache() -> None:
    """Clear cached schemas and descriptions.

    Call this when tools are reloaded or compression settings change.
    """
    global _schema_cache, _description_cache
    _schema_cache.clear()
    _description_cache.clear()


def get_cache_stats() -> dict[str, int]:
    """Get cache statistics for monitoring."""
    return {
        "schema_cache_size": len(_schema_cache),
        "description_cache_size": len(_description_cache),
    }


# Minimal tool descriptions - under 50 words each
MINIMAL_DESCRIPTIONS = {
    # AutoCAD tools
    "autocad_list_layers": "List all layers with name, color, visibility.",
    "autocad_create_layer": "Create layer. Args: name (str), color (int 1-255).",
    "autocad_set_layer_state": "Set layer on/off/frozen. Args: name, is_on, is_frozen.",
    "autocad_draw_line": "Draw line. Args: start_x, start_y, end_x, end_y, layer.",
    "autocad_draw_circle": "Draw circle. Args: center_x, center_y, radius, layer.",
    "autocad_draw_rectangle": "Draw rectangle. Args: corner1_x/y, corner2_x/y, layer.",
    "autocad_get_drawing_info": "Get drawing name, path, statistics.",
    "autocad_get_entities": "Get entities. Args: layer, entity_type, limit (100).",
    "autocad_delete_entity": "Delete entity by handle. Args: handle (str).",

    # Revit tools
    "revit_list_levels": "List all levels with id, name, elevation (meters).",
    "revit_create_level": "Create level. Args: name (str), elevation (float meters).",
    "revit_list_walls": "List all walls with id, type, level, length.",
    "revit_create_wall": "Create wall. Args: start_x/y, end_x/y, level_name, height (3.0m).",
    "revit_list_rooms": "List rooms. Args: use_cache (bool, faster if true).",
    "revit_get_document_status": "Get document title, path, worksharing info.",
    "revit_delete_element": "Delete element by ID. Warning: irreversible.",

    # Metadata tools
    "find_elements": "Semantic search. Args: query (str), source, category, layer, limit.",
    "get_nearby_elements": "Find elements within distance. Args: element_id, distance (1m), limit.",
    "get_related_elements": "Get related elements. Args: element_id, relation_type, limit.",
    "resolve_coordinates": "Get coordinates for element reference. Args: element_reference (str).",
    "sync_metadata": "Extract elements to database. Args: source (autocad/revit).",

    # Smart tools
    "draw_line_between": "Draw line between elements. Args: from_element, to_element, layer.",
    "draw_circle_at": "Draw circle at element. Args: element (str), radius, layer.",
    "draw_rectangle_around": "Draw rectangle around element. Args: element, padding (0.5m), layer.",
    "get_distance_between": "Get distance between elements. Args: element1, element2.",

    # Vectorization tools
    # IMPORTANT: vectorize_pdf is THE primary tool for all PDF-to-CAD conversions.
    # It uses Gemini Vision + OpenCV for intelligent extraction. Other tools are legacy/low-level.
    "vectorize_pdf": (
        "PRIMARY TOOL: Convert PDF to AutoCAD vector entities using Gemini Vision + OpenCV. "
        "Use this for ALL PDF vectorization requests. Handles the full pipeline automatically. "
        "Uses AI understanding for semantic extraction + OpenCV for pixel-perfect geometry. "
        "Args: pdf_path (str, required), page (int), extraction_method ('hybrid'/'direct'/'best'), "
        "create_in_autocad (bool, default True), validate (bool)."
    ),
    "raster_pdf_to_vector_pipeline": (
        "LEGACY — use vectorize_pdf instead. Traditional raster-to-vector conversion. "
        "Only use if specifically requested for high-fidelity legacy workflows."
    ),
    "raster_auto_vectorize": (
        "LOW-LEVEL — do NOT call directly. Use vectorize_pdf instead. "
        "Runs OpenCV detection on an already-processed bitonal TIFF only."
    ),
    "raster_import_pdf": (
        "LOW-LEVEL — do NOT call directly. Use vectorize_pdf instead. "
        "Imports vector PDF via PDFIMPORT. The pipeline tool calls this automatically."
    ),
    "raster_convert_pdf": (
        "LOW-LEVEL — do NOT call directly. Use vectorize_pdf instead. "
        "Converts PDF page to bitonal TIFF. The pipeline tool calls this automatically."
    ),
    "raster_attach_image": (
        "LOW-LEVEL — do NOT call directly. Use vectorize_pdf instead. "
        "Attaches a raster image to AutoCAD. The pipeline tool calls this automatically."
    ),
    "raster_cleanup": (
        "LOW-LEVEL — do NOT call directly. Use vectorize_pdf instead. "
        "Despeckles/deskews raster image. The pipeline tool calls this automatically."
    ),
    "raster_vectorize": "LOW-LEVEL — use vectorize_pdf instead. VTools vectorization.",
    "raster_ocr_extract": "Extract text from raster images using Raster Design OCR.",
    "raster_get_status": "Get info about raster images in the current AutoCAD drawing.",
    "raster_get_entity_count": "Count entities in drawing, optionally filtered by layer.",
    "raster_fade_image": "Fade raster images to reduce opacity. Args: fade_percent (int).",
    "raster_process_image": "Apply bitonal image filter via Raster Design ibfilter.",
    "raster_create_primitive": "Create a REM primitive from raster data.",
    "raster_select_entities": "Select raster entities within a rectangular region.",
    "raster_follower": "Semi-automatic follower for tracing raster lines/contours.",
    "raster_recognize_text": "Recognize and convert raster text to AutoCAD TEXT entities.",
    "raster_store_vectorized": "Extract all entities from drawing and store in PostgreSQL.",
    "raster_topology_cleanup": (
        "Post-process vectorized geometry: merge fragmented lines and snap dangling endpoints. "
        "Args: lines_json (list), snap_tolerance (float)."
    ),

    # Common
    "ping": "Check server connectivity.",
    "get_server_status": "Get server status, lock stats, cache info.",
    "check_sidecar": "Check if AutoCAD/Revit sidecar is healthy. Args: sidecar_type.",
    "get_file_context": "Get drawing summary (entities, layers, stats). Works without PostgreSQL.",
    "get_cache_status": "Get cache status and active project info.",
}


# Minimal schema - type only, no descriptions
def create_minimal_schema(full_schema: dict[str, Any]) -> dict[str, Any]:
    """
    Create minimal schema with types only, no descriptions.

    Reduces schema from ~100 tokens to ~20 tokens per tool.
    """
    if not full_schema:
        return full_schema

    result = {"type": full_schema.get("type", "object")}

    if "properties" in full_schema:
        props = {}
        for name, prop in full_schema["properties"].items():
            minimal = {"type": prop.get("type", "string")}
            # Keep only essential constraints
            if "enum" in prop:
                minimal["enum"] = prop["enum"]
            if "default" in prop:
                minimal["default"] = prop["default"]
            props[name] = minimal
        result["properties"] = props

    if "required" in full_schema:
        result["required"] = full_schema["required"]

    return result


def compress_description_aggressive(desc: str) -> str:
    """
    Aggressively compress description to minimal form.

    - Remove all examples
    - Remove Args/Returns sections (keep in schema)
    - Keep only first sentence
    - Compress whitespace
    """
    if not desc:
        return ""

    # Remove example sections
    desc = re.sub(r'Example[s]?:.*?(?=\n\n|\Z)', '', desc, flags=re.DOTALL | re.IGNORECASE)

    # Remove Args section (schema has the info)
    desc = re.sub(r'Args:.*?(?=Returns:|Warning:|Example|\Z)', '', desc, flags=re.DOTALL)

    # Remove Returns section
    desc = re.sub(r'Returns:.*?(?=Warning:|Example|\Z)', '', desc, flags=re.DOTALL)

    # Get first sentence
    lines = [l.strip() for l in desc.split('\n') if l.strip()]
    if lines:
        first_line = lines[0]
        # Get first sentence
        match = re.match(r'^[^.!?]+[.!?]?', first_line)
        if match:
            return match.group(0).strip()
        return first_line[:100]

    return ""


def get_optimized_description(tool_name: str, full_desc: str, mode: str = "standard") -> str:
    """
    Get optimized tool description based on mode.

    Uses caching to avoid recomputation for repeated calls.

    Args:
        tool_name: Name of the tool
        full_desc: Full tool description
        mode: Optimization level
            - "full": No optimization
            - "standard": Remove examples, compress
            - "minimal": Use pre-defined minimal descriptions
            - "ultra": Tool name only
    """
    if mode == "full":
        return full_desc

    # Check cache first
    cache_key = f"{tool_name}:{mode}"
    if cache_key in _description_cache:
        return _description_cache[cache_key]

    # Compute optimized description
    if mode == "ultra":
        result = tool_name.replace("_", " ").title()
    elif mode == "minimal":
        result = MINIMAL_DESCRIPTIONS.get(tool_name, compress_description_aggressive(full_desc))
    else:
        # Standard mode
        result = compress_description_aggressive(full_desc)

    # Cache and return
    _description_cache[cache_key] = result
    return result


def get_optimized_schema(schema: dict[str, Any], mode: str = "standard") -> dict[str, Any]:
    """
    Get optimized schema based on mode.

    Uses caching based on schema hash to avoid recomputation.

    Args:
        schema: Full JSON schema
        mode: Optimization level
            - "full": No optimization
            - "standard": Compress descriptions
            - "minimal": Types only
            - "ultra": Required params only
    """
    if mode == "full":
        return schema

    # Create cache key from schema hash + mode
    try:
        schema_str = json.dumps(schema, sort_keys=True)
        cache_key = f"{hash(schema_str)}:{mode}"
    except (TypeError, ValueError):
        # If schema is not JSON serializable, skip caching
        cache_key = None

    # Check cache first
    if cache_key and cache_key in _schema_cache:
        return _schema_cache[cache_key]

    # Compute optimized schema
    if mode == "ultra":
        # Only required properties with types
        result = {"type": "object"}
        if "required" in schema and "properties" in schema:
            result["properties"] = {
                k: {"type": v.get("type", "string")}
                for k, v in schema["properties"].items()
                if k in schema["required"]
            }
            result["required"] = schema["required"]
    elif mode == "minimal":
        result = create_minimal_schema(schema)
    else:
        # Standard mode - use existing compression
        from aec_agent.frontend.mcp_client import _compress_schema
        result = _compress_schema(schema)

    # Cache and return
    if cache_key:
        _schema_cache[cache_key] = result
    return result


# Tool tiers for conditional loading
TOOL_TIERS = {
    # Essential tools - always loaded
    "essential": [
        "ping",
        "autocad_draw_line",
        "autocad_draw_circle",
        "autocad_draw_rectangle",
        "autocad_list_layers",
        "revit_create_wall",
        "revit_list_levels",
        "revit_get_document_status",
    ],

    # Standard tools - loaded by default
    "standard": [
        "autocad_create_layer",
        "autocad_set_layer_state",
        "autocad_get_drawing_info",
        "autocad_get_entities",
        "autocad_delete_entity",
        "vectorize_pdf",
        "revit_create_level",
        "revit_list_walls",
        "revit_list_rooms",
        "revit_delete_element",
    ],

    # Advanced tools - loaded when database configured
    "advanced": [
        "find_elements",
        "get_nearby_elements",
        "get_related_elements",
        "resolve_coordinates",
        "sync_metadata",
        "draw_line_between",
        "draw_circle_at",
        "draw_rectangle_around",
        "get_distance_between",
    ],
}


# MEP-specific tool tiers for domain-aware loading
# These provide more granular control when intent classification detects MEP domains
MEP_TOOL_TIERS = {
    # HVAC domain tools
    "hvac": {
        "essential": [
            "find_elements",       # Find equipment, ducts, terminals
            "get_nearby_elements", # Clearance checking
            "sync_metadata",       # Extract model data
        ],
        "standard": [
            "get_related_elements",    # System connections
            "get_distance_between",    # Spacing verification
            "draw_line_between",       # Duct routing visualization
        ],
        "advanced": [
            "get_intersecting_elements",  # Clash detection (future)
            "draw_circle_at",             # Equipment markers
            "draw_rectangle_around",      # Zone highlighting
        ],
    },

    # Electrical domain tools
    "electrical": {
        "essential": [
            "find_elements",
            "get_nearby_elements",
            "sync_metadata",
        ],
        "standard": [
            "get_related_elements",
            "get_distance_between",
        ],
        "advanced": [
            "draw_line_between",
            "get_intersecting_elements",
        ],
    },

    # Plumbing domain tools
    "plumbing": {
        "essential": [
            "find_elements",
            "get_nearby_elements",
            "sync_metadata",
        ],
        "standard": [
            "get_related_elements",
            "get_distance_between",
        ],
        "advanced": [
            "draw_line_between",
            "get_intersecting_elements",
        ],
    },

    # Fire protection domain tools
    "fire_protection": {
        "essential": [
            "find_elements",
            "get_nearby_elements",
            "sync_metadata",
        ],
        "standard": [
            "get_related_elements",
        ],
        "advanced": [
            "get_intersecting_elements",
        ],
    },

    # Low voltage domain tools (security, fire alarm, BMS, AV, data/telecom)
    "low_voltage": {
        "essential": [
            "find_elements",
            "get_nearby_elements",
            "sync_metadata",
        ],
        "standard": [
            "get_related_elements",
            "get_distance_between",
        ],
        "advanced": [
            "draw_line_between",
            "draw_circle_at",
        ],
    },
}

# Keywords to filter elements by category/type for each MEP domain
MEP_CATEGORY_FILTERS = {
    "hvac": {
        "revit_categories": [
            "Mechanical Equipment",
            "Ducts",
            "Duct Fittings",
            "Duct Accessories",
            "Air Terminals",
            "Flex Ducts",
        ],
        "autocad_layers": [
            "M-HVAC", "M-DUCT", "M-EQUIP", "MECH", "HVAC",
            "DUCT", "DIFFUSER", "AHU", "VAV",
        ],
        "entity_types": [
            "duct", "diffuser", "air terminal", "mechanical equipment",
            "vav", "ahu", "fan coil",
        ],
    },
    "electrical": {
        "revit_categories": [
            "Electrical Equipment",
            "Electrical Fixtures",
            "Conduits",
            "Cable Trays",
            "Lighting Fixtures",
        ],
        "autocad_layers": [
            "E-POWER", "E-LITE", "E-EQUIP", "ELEC", "ELECTRICAL",
            "CONDUIT", "PANEL", "RECEPTACLE",
        ],
        "entity_types": [
            "panel", "conduit", "receptacle", "switch", "light fixture",
        ],
    },
    "plumbing": {
        "revit_categories": [
            "Plumbing Fixtures",
            "Pipes",
            "Pipe Fittings",
            "Pipe Accessories",
        ],
        "autocad_layers": [
            "P-SANR", "P-DOME", "P-FIXT", "PLUMB", "PLUMBING",
            "PIPE", "DRAIN", "FIXTURE",
        ],
        "entity_types": [
            "pipe", "fixture", "sink", "toilet", "valve",
        ],
    },
    "fire_protection": {
        "revit_categories": [
            "Sprinklers",
            "Fire Alarm Devices",
        ],
        "autocad_layers": [
            "F-SPKL", "F-ALRM", "FIRE", "SPRINKLER",
        ],
        "entity_types": [
            "sprinkler", "fire alarm", "smoke detector",
        ],
    },
    "low_voltage": {
        "revit_categories": [
            "Communication Devices",
            "Data Devices",
            "Fire Alarm Devices",
            "Security Devices",
            "Telephone Devices",
            "Nurse Call Devices",
        ],
        "autocad_layers": [
            "LV-DATA", "LV-TELE", "LV-SEC", "LV-FA", "LV-AV",
            "LV-BMS", "LV-CCTV", "LV-ACC", "D-", "T-",
            "COMM", "TELECOM", "DATA", "SECURITY", "CCTV",
        ],
        "entity_types": [
            "data outlet", "camera", "card reader", "smoke detector",
            "speaker", "access point", "patch panel", "controller",
            "network switch", "wifi", "intercom", "pull station",
        ],
    },
}


def get_tools_for_tier(tier: str) -> list[str]:
    """Get tool names for a given tier and below."""
    tiers = ["essential", "standard", "advanced"]
    idx = tiers.index(tier) if tier in tiers else 1  # default standard

    tools = []
    for t in tiers[:idx + 1]:
        tools.extend(TOOL_TIERS.get(t, []))
    return tools


def get_mep_tools_for_domain(domain: str, tier: str = "standard") -> list[str]:
    """
    Get tool names for a specific MEP domain and tier.

    Args:
        domain: MEP domain (hvac, electrical, plumbing, fire_protection)
        tier: Tool tier (essential, standard, advanced)

    Returns:
        List of tool names for the domain
    """
    domain_lower = domain.lower()
    if domain_lower not in MEP_TOOL_TIERS:
        return get_tools_for_tier(tier)

    domain_tiers = MEP_TOOL_TIERS[domain_lower]
    tiers = ["essential", "standard", "advanced"]
    idx = tiers.index(tier) if tier in tiers else 1

    tools = []
    for t in tiers[:idx + 1]:
        tools.extend(domain_tiers.get(t, []))
    return tools


def get_mep_category_filter(domain: str) -> dict:
    """
    Get category filter criteria for an MEP domain.

    Returns dict with:
        - revit_categories: List of Revit categories to include
        - autocad_layers: List of AutoCAD layer patterns to include
        - entity_types: List of entity type keywords to include
    """
    return MEP_CATEGORY_FILTERS.get(domain.lower(), {})


# =============================================================================
# Domain Priority Tools (Phase 2 optimization)
# =============================================================================
# More granular control over which tools are loaded for specific domains

DOMAIN_PRIORITY_TOOLS = {
    "hvac": {
        "must_have": ["find_elements", "get_nearby_elements", "sync_metadata"],
        "useful": ["get_related_elements", "draw_line_between", "get_distance_between"],
        "exclude": [],
    },
    "electrical": {
        "must_have": ["find_elements", "get_nearby_elements", "sync_metadata"],
        "useful": ["get_related_elements", "get_distance_between"],
        "exclude": [],
    },
    "plumbing": {
        "must_have": ["find_elements", "get_nearby_elements", "sync_metadata"],
        "useful": ["get_related_elements", "get_distance_between"],
        "exclude": [],
    },
    "fire_protection": {
        "must_have": ["find_elements", "get_nearby_elements"],
        "useful": ["sync_metadata", "get_related_elements"],
        "exclude": [],
    },
    "low_voltage": {
        "must_have": ["find_elements", "get_nearby_elements"],
        "useful": ["get_related_elements", "sync_metadata", "get_distance_between"],
        "exclude": [],
    },
}


def get_domain_priority_tools(domain: str, include_useful: bool = True) -> set[str]:
    """
    Get prioritized tools for an MEP domain.

    Args:
        domain: MEP domain (hvac, electrical, plumbing, fire_protection, low_voltage)
        include_useful: Whether to include "useful" tools (default True)

    Returns:
        Set of tool names that are prioritized for this domain
    """
    domain_lower = domain.lower()
    if domain_lower not in DOMAIN_PRIORITY_TOOLS:
        return set()

    config = DOMAIN_PRIORITY_TOOLS[domain_lower]
    tools = set(config["must_have"])
    if include_useful:
        tools.update(config["useful"])
    return tools


def get_domain_excluded_tools(domain: str) -> set[str]:
    """
    Get tools that should be excluded for an MEP domain.

    Args:
        domain: MEP domain

    Returns:
        Set of tool names to exclude
    """
    domain_lower = domain.lower()
    if domain_lower not in DOMAIN_PRIORITY_TOOLS:
        return set()
    return set(DOMAIN_PRIORITY_TOOLS[domain_lower].get("exclude", []))


def estimate_token_count(text: str) -> int:
    """Rough token estimate (GPT-style: ~4 chars per token)."""
    return len(text) // 4


def estimate_tool_tokens(name: str, description: str, schema: dict) -> dict:
    """Estimate tokens for a tool in different modes."""
    import json

    return {
        "full": estimate_token_count(
            name + description + json.dumps(schema)
        ),
        "standard": estimate_token_count(
            name + compress_description_aggressive(description) +
            json.dumps(get_optimized_schema(schema, "standard"))
        ),
        "minimal": estimate_token_count(
            name + get_optimized_description(name, description, "minimal") +
            json.dumps(get_optimized_schema(schema, "minimal"))
        ),
        "ultra": estimate_token_count(
            name + get_optimized_description(name, description, "ultra") +
            json.dumps(get_optimized_schema(schema, "ultra"))
        ),
    }
