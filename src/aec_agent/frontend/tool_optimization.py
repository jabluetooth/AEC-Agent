"""
Token optimization for MCP tools.

Provides aggressive compression strategies for tool descriptions and schemas
to minimize token usage with budget LLM providers.
"""

from typing import Any
import re


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

    # Common
    "ping": "Check server connectivity.",
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

    if mode == "ultra":
        return tool_name.replace("_", " ").title()

    if mode == "minimal":
        return MINIMAL_DESCRIPTIONS.get(tool_name, compress_description_aggressive(full_desc))

    # Standard mode
    return compress_description_aggressive(full_desc)


def get_optimized_schema(schema: dict[str, Any], mode: str = "standard") -> dict[str, Any]:
    """
    Get optimized schema based on mode.

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
        return result

    if mode == "minimal":
        return create_minimal_schema(schema)

    # Standard mode - use existing compression
    from aec_agent.frontend.mcp_client import _compress_schema
    return _compress_schema(schema)


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
