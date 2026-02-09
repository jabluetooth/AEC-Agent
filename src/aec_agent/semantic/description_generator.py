"""
Description generator for CAD elements.

Generates human-readable descriptions from element properties
for use in semantic search embeddings.
"""

from typing import Any


def generate_description(element: dict[str, Any], source: str = "autocad") -> str:
    """
    Generate searchable description from element properties.

    Args:
        element: Element dict with properties
        source: 'autocad' or 'revit'

    Returns:
        Human-readable description
    """
    if source == "autocad":
        return generate_autocad_description(element)
    else:
        return generate_revit_description(element)


def generate_autocad_description(element: dict[str, Any]) -> str:
    """
    Generate description for AutoCAD entity.

    Examples:
    - "LINE on layer WALLS, length 10.5m"
    - "CIRCLE on layer ELECTRICAL, radius 0.5m at (100, 200)"
    - "BLOCK 'Chair-01' on layer FURNITURE at (50, 75)"
    - "TEXT 'Room 101' on layer ANNOTATION"

    Args:
        element: Element dict

    Returns:
        Description string
    """
    parts = []

    entity_type = element.get("entity_type", "Unknown")
    layer = element.get("layer", "0")

    # Start with type and layer
    parts.append(f"{entity_type} on layer {layer}")

    properties = element.get("properties", {})

    # Add dimension info based on type
    if entity_type in ("LINE", "POLYLINE", "LWPOLYLINE"):
        length = properties.get("length") or properties.get("Length")
        if length:
            parts.append(f"length {_format_length(length)}")

    elif entity_type == "CIRCLE":
        radius = properties.get("radius") or properties.get("Radius")
        if radius:
            parts.append(f"radius {_format_length(radius)}")

    elif entity_type == "ARC":
        radius = properties.get("radius") or properties.get("Radius")
        if radius:
            parts.append(f"radius {_format_length(radius)}")

    elif entity_type in ("TEXT", "MTEXT"):
        text_content = properties.get("content") or properties.get("text_string")
        if text_content:
            # Truncate long text
            if len(text_content) > 50:
                text_content = text_content[:47] + "..."
            parts.append(f"'{text_content}'")

    elif entity_type in ("INSERT", "BLOCKREFERENCE"):
        block_name = properties.get("block_name") or properties.get("name")
        if block_name:
            parts.append(f"block '{block_name}'")

    elif entity_type == "HATCH":
        pattern = properties.get("pattern_name") or properties.get("pattern")
        if pattern:
            parts.append(f"pattern '{pattern}'")

    # Add position if available
    centroid = element.get("centroid")
    if centroid:
        x = centroid.get("x", 0)
        y = centroid.get("y", 0)
        parts.append(f"at ({x:.1f}, {y:.1f})")

    # Add color if non-default
    color = properties.get("color")
    if color and color not in (7, 256):  # Not white/bylayer
        parts.append(f"color {color}")

    return ", ".join(parts)


def generate_revit_description(element: dict[str, Any]) -> str:
    """
    Generate description for Revit element.

    Examples:
    - "Wall: Basic Wall - Generic 200mm, height 3.0m, fire rating: 1HR"
    - "Door: Single-Flush 0915x2134mm in Wall ID:12345"
    - "Column: M_Rectangular Column 450x600mm at Level 1"
    - "Room: Office 101, area 25.5 sqm on Level 1"

    Args:
        element: Element dict

    Returns:
        Description string
    """
    parts = []

    category = element.get("category", "Unknown")
    family = element.get("family", "")
    type_name = element.get("type_name", "")

    # Start with category and family/type
    if family and type_name:
        parts.append(f"{category}: {family} - {type_name}")
    elif family:
        parts.append(f"{category}: {family}")
    elif type_name:
        parts.append(f"{category}: {type_name}")
    else:
        parts.append(category)

    properties = element.get("properties", {})

    # Add dimension info based on category
    if category in ("Walls", "Wall"):
        height = _get_param(properties, "Height", "Unconnected Height")
        if height:
            parts.append(f"height {_format_length(height)}")

        width = _get_param(properties, "Width", "Thickness")
        if width:
            parts.append(f"width {_format_length(width)}")

        fire_rating = _get_param(properties, "Fire Rating")
        if fire_rating:
            parts.append(f"fire rating: {fire_rating}")

    elif category in ("Doors", "Door"):
        width = _get_param(properties, "Width", "Rough Width")
        height = _get_param(properties, "Height", "Rough Height")
        if width and height:
            parts.append(f"{_format_length(width)} x {_format_length(height)}")

    elif category in ("Windows", "Window"):
        width = _get_param(properties, "Width", "Rough Width")
        height = _get_param(properties, "Height", "Rough Height")
        if width and height:
            parts.append(f"{_format_length(width)} x {_format_length(height)}")

    elif category in ("Rooms", "Room"):
        name = _get_param(properties, "Name")
        number = _get_param(properties, "Number")
        area = _get_param(properties, "Area")

        if name:
            parts.append(f"'{name}'")
        if number:
            parts.append(f"number {number}")
        if area:
            parts.append(f"area {_format_area(area)}")

    elif category in ("Columns", "Column", "Structural Columns"):
        dims = _get_param(properties, "b", "Width")
        depth = _get_param(properties, "h", "Depth")
        if dims and depth:
            parts.append(f"{_format_length(dims)} x {_format_length(depth)}")

    elif category in ("Floors", "Floor"):
        area = _get_param(properties, "Area")
        thickness = _get_param(properties, "Thickness", "Default Thickness")
        if area:
            parts.append(f"area {_format_area(area)}")
        if thickness:
            parts.append(f"thickness {_format_length(thickness)}")

    # Add level
    level = properties.get("level") or _get_param(properties, "Level")
    if level:
        parts.append(f"on {level}")

    # Add mark/tag if present
    mark = _get_param(properties, "Mark", "Tag")
    if mark:
        parts.append(f"mark: {mark}")

    return ", ".join(parts)


def _get_param(properties: dict[str, Any], *names: str) -> Any | None:
    """Get parameter value by trying multiple names."""
    for name in names:
        if name in properties:
            return properties[name]
        # Try case variations
        if name.lower() in properties:
            return properties[name.lower()]
        if name.upper() in properties:
            return properties[name.upper()]
    return None


def _format_length(value: Any) -> str:
    """Format length value with units."""
    if value is None:
        return ""
    try:
        num = float(value)
        if num < 0.01:
            return f"{num * 1000:.1f}mm"
        elif num < 1:
            return f"{num * 100:.1f}cm"
        else:
            return f"{num:.2f}m"
    except (ValueError, TypeError):
        return str(value)


def _format_area(value: Any) -> str:
    """Format area value with units."""
    if value is None:
        return ""
    try:
        num = float(value)
        return f"{num:.1f}sqm"
    except (ValueError, TypeError):
        return str(value)
