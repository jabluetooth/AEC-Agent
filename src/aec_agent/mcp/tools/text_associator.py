"""
Text-to-element association for AEC drawings.

Links parsed text annotations to nearby symbols and geometry.
This is Layer 3 (Text Stream) of the Semantic Intelligence Pipeline.

Key associations:
- Equipment specs → nearest equipment symbol
- Room names/numbers → enclosing room boundary
- Dimensions → associated line segment
- Flow rates → nearby diffuser/duct symbols
"""

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import structlog

from .semantic_ocr import AnnotationType, ParsedAnnotation

logger = structlog.get_logger(__name__)


class AssociationType(str, Enum):
    """Type of text-to-element association."""
    LABELS = "labels"           # Text labels an element (equipment tag)
    SPECIFIES = "specifies"     # Text specifies element properties (CFM, size)
    DESCRIBES = "describes"     # Text describes element (notes)
    DIMENSIONS = "dimensions"   # Text provides dimensions
    CONTAINED_IN = "contained_in"  # Text is contained in element (room)
    REFERENCES = "references"   # Text references element (detail callout)


@dataclass
class DetectedElement:
    """
    A detected element (symbol or geometry) that can be associated with text.

    This is a simplified representation that can hold different element types.
    """
    element_id: str
    element_type: str  # "symbol", "geometry", "room_boundary"
    position: tuple[float, float]  # Center position
    bounds: tuple[float, float, float, float]  # x, y, width, height
    category: str = ""  # "mechanical", "electrical", "plumbing", etc.
    subtype: str = ""  # "diffuser", "valve", "outlet", etc.
    confidence: float = 1.0
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def center(self) -> tuple[float, float]:
        """Get center point of the element."""
        return self.position

    @property
    def area(self) -> float:
        """Get area of bounding box."""
        return self.bounds[2] * self.bounds[3]

    def contains_point(self, x: float, y: float) -> bool:
        """Check if a point is inside this element's bounds."""
        bx, by, bw, bh = self.bounds
        return bx <= x <= bx + bw and by <= y <= by + bh

    def distance_to_point(self, x: float, y: float) -> float:
        """Calculate distance from element center to a point."""
        cx, cy = self.position
        return math.sqrt((cx - x) ** 2 + (cy - y) ** 2)

    def distance_to_edge(self, x: float, y: float) -> float:
        """Calculate minimum distance from point to element edge."""
        bx, by, bw, bh = self.bounds

        # If inside, return 0
        if self.contains_point(x, y):
            return 0.0

        # Find closest point on rectangle edge
        closest_x = max(bx, min(x, bx + bw))
        closest_y = max(by, min(y, by + bh))

        return math.sqrt((x - closest_x) ** 2 + (y - closest_y) ** 2)


@dataclass
class TextAssociation:
    """
    An association between a text annotation and an element.

    Attributes:
        annotation: The parsed text annotation
        element: The associated element
        association_type: How the text relates to the element
        confidence: Confidence in the association (0.0-1.0)
        distance: Distance between text and element
    """
    annotation: ParsedAnnotation
    element: DetectedElement
    association_type: AssociationType
    confidence: float
    distance: float
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ElementWithAnnotations:
    """
    An element with its associated text annotations.

    Collects all annotations that relate to a single element.
    """
    element: DetectedElement
    annotations: list[TextAssociation]

    @property
    def tags(self) -> list[str]:
        """Get equipment tags associated with this element."""
        return [
            a.annotation.raw_text
            for a in self.annotations
            if a.annotation.annotation_type == AnnotationType.EQUIPMENT_TAG
        ]

    @property
    def specs(self) -> dict[str, Any]:
        """Get combined specifications from all spec annotations."""
        specs = {}
        for a in self.annotations:
            if a.annotation.annotation_type in (
                AnnotationType.EQUIPMENT_SPEC,
                AnnotationType.SIZE,
                AnnotationType.FLOW_RATE,
                AnnotationType.POWER,
                AnnotationType.CAPACITY,
            ):
                specs.update(a.annotation.structured_data)
        return specs


# Association rules by annotation type
ASSOCIATION_RULES: dict[AnnotationType, dict[str, Any]] = {
    AnnotationType.EQUIPMENT_TAG: {
        "target_types": ["symbol"],
        "max_distance_factor": 2.0,  # 2x element size
        "association_type": AssociationType.LABELS,
        "prefer_direction": "any",
    },
    AnnotationType.EQUIPMENT_SPEC: {
        "target_types": ["symbol"],
        "max_distance_factor": 3.0,
        "association_type": AssociationType.SPECIFIES,
        "prefer_direction": "below",  # Specs often below symbol
    },
    AnnotationType.FLOW_RATE: {
        "target_types": ["symbol"],
        "target_categories": ["mechanical"],
        "max_distance_factor": 2.5,
        "association_type": AssociationType.SPECIFIES,
    },
    AnnotationType.SIZE: {
        "target_types": ["symbol", "geometry"],
        "max_distance_factor": 2.0,
        "association_type": AssociationType.SPECIFIES,
    },
    AnnotationType.ROOM_NAME: {
        "target_types": ["room_boundary"],
        "max_distance_factor": None,  # Must be inside
        "association_type": AssociationType.LABELS,
        "require_containment": True,
    },
    AnnotationType.ROOM_NUMBER: {
        "target_types": ["room_boundary"],
        "max_distance_factor": None,
        "association_type": AssociationType.LABELS,
        "require_containment": True,
    },
    AnnotationType.AREA: {
        "target_types": ["room_boundary"],
        "max_distance_factor": None,
        "association_type": AssociationType.SPECIFIES,
        "require_containment": True,
    },
    AnnotationType.DIMENSION: {
        "target_types": ["geometry"],
        "max_distance_factor": 1.5,
        "association_type": AssociationType.DIMENSIONS,
    },
    AnnotationType.CIRCUIT_ID: {
        "target_types": ["symbol"],
        "target_categories": ["electrical"],
        "max_distance_factor": 2.0,
        "association_type": AssociationType.LABELS,
    },
    AnnotationType.PANEL_NAME: {
        "target_types": ["symbol"],
        "target_categories": ["electrical"],
        "max_distance_factor": 3.0,
        "association_type": AssociationType.LABELS,
    },
}


def associate_text_to_elements(
    annotations: list[ParsedAnnotation],
    elements: list[DetectedElement],
    max_distance: float = 100.0,
    use_rules: bool = True,
) -> list[TextAssociation]:
    """
    Associate parsed text annotations with detected elements.

    Uses distance-based matching with type-specific rules.

    Args:
        annotations: Parsed text annotations with positions
        elements: Detected elements (symbols, geometry, rooms)
        max_distance: Maximum distance for association (pixels)
        use_rules: Whether to use annotation-type-specific rules

    Returns:
        List of TextAssociation objects
    """
    associations: list[TextAssociation] = []

    for annotation in annotations:
        if annotation.annotation_type == AnnotationType.UNKNOWN:
            continue

        # Get position (handle both tuple and missing positions)
        text_pos = annotation.position
        if not text_pos or text_pos == (0.0, 0.0):
            # Try to get from bounding box
            if annotation.bounding_box:
                bbox = annotation.bounding_box
                text_pos = (bbox[0] + bbox[2] / 2, bbox[1] + bbox[3] / 2)
            else:
                continue

        text_x, text_y = text_pos

        # Get association rules for this annotation type
        rules = ASSOCIATION_RULES.get(annotation.annotation_type, {})

        # Find candidate elements
        candidates: list[tuple[DetectedElement, float, float]] = []

        for element in elements:
            # Check type filter
            target_types = rules.get("target_types", ["symbol", "geometry"])
            if element.element_type not in target_types:
                continue

            # Check category filter
            target_cats = rules.get("target_categories")
            if target_cats and element.category not in target_cats:
                continue

            # Check containment requirement
            if rules.get("require_containment"):
                if not element.contains_point(text_x, text_y):
                    continue
                distance = 0.0
            else:
                distance = element.distance_to_edge(text_x, text_y)

                # Check max distance
                max_dist_factor = rules.get("max_distance_factor", 2.0)
                if max_dist_factor:
                    element_size = max(element.bounds[2], element.bounds[3])
                    effective_max = max(max_dist_factor * element_size, max_distance)
                    if distance > effective_max:
                        continue

            # Calculate confidence based on distance
            if distance == 0:
                confidence = 1.0
            else:
                element_size = max(element.bounds[2], element.bounds[3], 1)
                confidence = max(0.0, 1.0 - (distance / (3 * element_size)))

            candidates.append((element, distance, confidence))

        # Select best candidate(s)
        if candidates:
            # Sort by distance (closest first)
            candidates.sort(key=lambda x: x[1])

            # Take the closest match
            best_element, best_distance, best_confidence = candidates[0]

            association = TextAssociation(
                annotation=annotation,
                element=best_element,
                association_type=rules.get("association_type", AssociationType.DESCRIBES),
                confidence=best_confidence,
                distance=best_distance,
            )
            associations.append(association)

    logger.info(
        "Text-to-element association complete",
        annotations=len(annotations),
        elements=len(elements),
        associations=len(associations),
    )

    return associations


def group_annotations_by_element(
    associations: list[TextAssociation],
) -> list[ElementWithAnnotations]:
    """
    Group text associations by their target element.

    Args:
        associations: List of text-to-element associations

    Returns:
        List of elements with their associated annotations
    """
    element_map: dict[str, ElementWithAnnotations] = {}

    for assoc in associations:
        element_id = assoc.element.element_id

        if element_id not in element_map:
            element_map[element_id] = ElementWithAnnotations(
                element=assoc.element,
                annotations=[],
            )

        element_map[element_id].annotations.append(assoc)

    result = list(element_map.values())

    logger.info(
        "Grouped annotations by element",
        total_associations=len(associations),
        elements_with_annotations=len(result),
    )

    return result


def find_room_annotations(
    annotations: list[ParsedAnnotation],
    room_boundaries: list[DetectedElement],
) -> dict[str, dict[str, Any]]:
    """
    Find room-related annotations and associate them with room boundaries.

    Specifically handles:
    - Room names
    - Room numbers
    - Areas

    Args:
        annotations: Parsed text annotations
        room_boundaries: Detected room boundary elements

    Returns:
        Dictionary mapping room_id to room info
    """
    room_info: dict[str, dict[str, Any]] = {}

    # Initialize rooms
    for room in room_boundaries:
        room_info[room.element_id] = {
            "element": room,
            "name": None,
            "number": None,
            "area_sf": None,
            "area_m2": None,
            "annotations": [],
        }

    # Associate annotations
    for annotation in annotations:
        if annotation.annotation_type not in (
            AnnotationType.ROOM_NAME,
            AnnotationType.ROOM_NUMBER,
            AnnotationType.AREA,
        ):
            continue

        text_pos = annotation.position
        if not text_pos or text_pos == (0.0, 0.0):
            if annotation.bounding_box:
                bbox = annotation.bounding_box
                text_pos = (bbox[0] + bbox[2] / 2, bbox[1] + bbox[3] / 2)
            else:
                continue

        text_x, text_y = text_pos

        # Find containing room
        for room in room_boundaries:
            if room.contains_point(text_x, text_y):
                info = room_info[room.element_id]
                info["annotations"].append(annotation)

                if annotation.annotation_type == AnnotationType.ROOM_NAME:
                    # Use raw text as room name
                    info["name"] = annotation.raw_text

                elif annotation.annotation_type == AnnotationType.ROOM_NUMBER:
                    data = annotation.structured_data
                    info["number"] = data.get("room_number", annotation.raw_text)

                elif annotation.annotation_type == AnnotationType.AREA:
                    data = annotation.structured_data
                    if "area_sf" in data:
                        info["area_sf"] = data["area_sf"]
                    if "area_m2" in data:
                        info["area_m2"] = data["area_m2"]

                break  # Only associate with first containing room

    return room_info


def find_leader_targets(
    leader_lines: list[DetectedElement],
    symbols: list[DetectedElement],
    annotations: list[ParsedAnnotation],
    max_distance: float = 20.0,
) -> list[tuple[ParsedAnnotation, DetectedElement]]:
    """
    Find text annotations connected to elements via leader lines.

    Leader lines are lines that connect a text annotation to the
    element it labels (arrow pointing to element).

    Args:
        leader_lines: Detected leader line geometry
        symbols: Detected symbol elements
        annotations: Parsed text annotations
        max_distance: Max distance to consider connected

    Returns:
        List of (annotation, target_element) pairs
    """
    connections: list[tuple[ParsedAnnotation, DetectedElement]] = []

    for leader in leader_lines:
        # Leader line has two endpoints
        endpoints = leader.metadata.get("endpoints", [])
        if len(endpoints) < 2:
            continue

        start_pt, end_pt = endpoints[0], endpoints[1]

        # Find text near one endpoint
        text_match = None
        text_dist = float('inf')

        for annotation in annotations:
            pos = annotation.position
            if not pos or pos == (0.0, 0.0):
                continue

            # Check distance to both endpoints
            d_start = math.sqrt((pos[0] - start_pt[0])**2 + (pos[1] - start_pt[1])**2)
            d_end = math.sqrt((pos[0] - end_pt[0])**2 + (pos[1] - end_pt[1])**2)
            d = min(d_start, d_end)

            if d < max_distance and d < text_dist:
                text_match = annotation
                text_dist = d
                # Determine which endpoint is text end
                text_endpoint = start_pt if d_start < d_end else end_pt
                symbol_endpoint = end_pt if d_start < d_end else start_pt

        if not text_match:
            continue

        # Find symbol near the other endpoint
        symbol_match = None
        symbol_dist = float('inf')

        for symbol in symbols:
            d = symbol.distance_to_point(symbol_endpoint[0], symbol_endpoint[1])
            if d < max_distance and d < symbol_dist:
                symbol_match = symbol
                symbol_dist = d

        if text_match and symbol_match:
            connections.append((text_match, symbol_match))

    logger.info(
        "Found leader line connections",
        leaders=len(leader_lines),
        connections=len(connections),
    )

    return connections


def enrich_elements_with_text(
    elements: list[DetectedElement],
    annotations: list[ParsedAnnotation],
    max_distance: float = 100.0,
) -> list[dict[str, Any]]:
    """
    Enrich elements with associated text data.

    Creates enriched element dictionaries that include:
    - Original element data
    - Associated tags
    - Associated specifications
    - Associated notes

    Args:
        elements: Detected elements
        annotations: Parsed annotations
        max_distance: Maximum association distance

    Returns:
        List of enriched element dictionaries
    """
    # Get associations
    associations = associate_text_to_elements(
        annotations, elements, max_distance
    )

    # Group by element
    grouped = group_annotations_by_element(associations)

    # Create enriched output
    enriched: list[dict[str, Any]] = []

    for elem_with_annos in grouped:
        elem = elem_with_annos.element

        enriched_elem = {
            "element_id": elem.element_id,
            "element_type": elem.element_type,
            "position": elem.position,
            "bounds": elem.bounds,
            "category": elem.category,
            "subtype": elem.subtype,
            "confidence": elem.confidence,
            # Text-derived data
            "tags": elem_with_annos.tags,
            "specs": elem_with_annos.specs,
            "annotations": [
                {
                    "text": a.annotation.raw_text,
                    "type": a.annotation.annotation_type.value,
                    "association": a.association_type.value,
                    "confidence": a.confidence,
                }
                for a in elem_with_annos.annotations
            ],
        }
        enriched.append(enriched_elem)

    # Add elements without annotations
    associated_ids = {e["element_id"] for e in enriched}
    for elem in elements:
        if elem.element_id not in associated_ids:
            enriched.append({
                "element_id": elem.element_id,
                "element_type": elem.element_type,
                "position": elem.position,
                "bounds": elem.bounds,
                "category": elem.category,
                "subtype": elem.subtype,
                "confidence": elem.confidence,
                "tags": [],
                "specs": {},
                "annotations": [],
            })

    return enriched
