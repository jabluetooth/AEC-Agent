"""Unit tests for text_associator.py (Phase B)."""

import pytest

from aec_agent.mcp.tools.semantic_ocr import AnnotationType, ParsedAnnotation
from aec_agent.mcp.tools.text_associator import (
    AssociationType,
    DetectedElement,
    ElementWithAnnotations,
    TextAssociation,
    associate_text_to_elements,
    enrich_elements_with_text,
    find_room_annotations,
    group_annotations_by_element,
)


class TestAssociationType:
    """Tests for AssociationType enum."""

    def test_all_expected_types_exist(self):
        """All expected association types should be defined."""
        expected = [
            "LABELS", "SPECIFIES", "DESCRIBES",
            "DIMENSIONS", "CONTAINED_IN", "REFERENCES",
        ]
        for name in expected:
            assert hasattr(AssociationType, name), f"Missing AssociationType.{name}"


class TestDetectedElement:
    """Tests for DetectedElement dataclass."""

    def test_basic_creation(self):
        """Should create element with required fields."""
        elem = DetectedElement(
            element_id="sym-001",
            element_type="symbol",
            position=(100.0, 200.0),
            bounds=(80.0, 180.0, 40.0, 40.0),
            category="mechanical",
            subtype="diffuser",
        )
        assert elem.element_id == "sym-001"
        assert elem.element_type == "symbol"
        assert elem.position == (100.0, 200.0)
        assert elem.category == "mechanical"

    def test_center_property(self):
        """Center should return position."""
        elem = DetectedElement(
            element_id="test",
            element_type="symbol",
            position=(50.0, 75.0),
            bounds=(0, 0, 100, 150),
        )
        assert elem.center == (50.0, 75.0)

    def test_area_property(self):
        """Area should be width * height."""
        elem = DetectedElement(
            element_id="test",
            element_type="symbol",
            position=(0, 0),
            bounds=(0, 0, 100, 50),
        )
        assert elem.area == 5000.0

    def test_contains_point_inside(self):
        """contains_point should return True for points inside bounds."""
        elem = DetectedElement(
            element_id="test",
            element_type="room_boundary",
            position=(150.0, 150.0),
            bounds=(100.0, 100.0, 100.0, 100.0),
        )
        # Point inside
        assert elem.contains_point(150.0, 150.0) is True
        # Point on edge (inclusive)
        assert elem.contains_point(100.0, 100.0) is True
        assert elem.contains_point(200.0, 200.0) is True

    def test_contains_point_outside(self):
        """contains_point should return False for points outside bounds."""
        elem = DetectedElement(
            element_id="test",
            element_type="room_boundary",
            position=(150.0, 150.0),
            bounds=(100.0, 100.0, 100.0, 100.0),
        )
        # Point outside
        assert elem.contains_point(50.0, 50.0) is False
        assert elem.contains_point(250.0, 250.0) is False

    def test_distance_to_point(self):
        """distance_to_point should calculate correct distance."""
        elem = DetectedElement(
            element_id="test",
            element_type="symbol",
            position=(100.0, 100.0),
            bounds=(80.0, 80.0, 40.0, 40.0),
        )
        # Same point = 0
        assert elem.distance_to_point(100.0, 100.0) == 0.0
        # 3-4-5 triangle
        assert elem.distance_to_point(103.0, 104.0) == 5.0

    def test_distance_to_edge_inside(self):
        """distance_to_edge should return 0 for points inside."""
        elem = DetectedElement(
            element_id="test",
            element_type="room_boundary",
            position=(150.0, 150.0),
            bounds=(100.0, 100.0, 100.0, 100.0),
        )
        assert elem.distance_to_edge(150.0, 150.0) == 0.0

    def test_distance_to_edge_outside(self):
        """distance_to_edge should calculate distance from edge."""
        elem = DetectedElement(
            element_id="test",
            element_type="symbol",
            position=(50.0, 50.0),
            bounds=(0.0, 0.0, 100.0, 100.0),
        )
        # Point directly to the right of box
        assert elem.distance_to_edge(110.0, 50.0) == 10.0
        # Point diagonally outside (corner case)
        dist = elem.distance_to_edge(110.0, 110.0)
        assert abs(dist - 14.142) < 0.01  # sqrt(10^2 + 10^2)


class TestTextAssociation:
    """Tests for TextAssociation dataclass."""

    def test_basic_creation(self):
        """Should create association with required fields."""
        anno = ParsedAnnotation(
            raw_text="200 CFM",
            annotation_type=AnnotationType.FLOW_RATE,
            structured_data={"cfm": 200},
            confidence=0.9,
            position=(100.0, 150.0),
        )
        elem = DetectedElement(
            element_id="sym-001",
            element_type="symbol",
            position=(100.0, 100.0),
            bounds=(80.0, 80.0, 40.0, 40.0),
        )
        assoc = TextAssociation(
            annotation=anno,
            element=elem,
            association_type=AssociationType.SPECIFIES,
            confidence=0.85,
            distance=30.0,
        )
        assert assoc.annotation.raw_text == "200 CFM"
        assert assoc.element.element_id == "sym-001"
        assert assoc.association_type == AssociationType.SPECIFIES


class TestElementWithAnnotations:
    """Tests for ElementWithAnnotations dataclass."""

    def test_tags_property(self):
        """tags should return equipment tags."""
        elem = DetectedElement(
            element_id="sym-001",
            element_type="symbol",
            position=(0, 0),
            bounds=(0, 0, 10, 10),
        )
        tag_anno = ParsedAnnotation(
            raw_text="AHU-1",
            annotation_type=AnnotationType.EQUIPMENT_TAG,
            structured_data={"tag": "AHU-1"},
            confidence=0.9,
        )
        assoc = TextAssociation(
            annotation=tag_anno,
            element=elem,
            association_type=AssociationType.LABELS,
            confidence=0.9,
            distance=5.0,
        )
        elem_with_annos = ElementWithAnnotations(
            element=elem,
            annotations=[assoc],
        )
        assert "AHU-1" in elem_with_annos.tags

    def test_specs_property(self):
        """specs should combine specification data."""
        elem = DetectedElement(
            element_id="sym-001",
            element_type="symbol",
            position=(0, 0),
            bounds=(0, 0, 10, 10),
        )
        cfm_anno = ParsedAnnotation(
            raw_text="200 CFM",
            annotation_type=AnnotationType.FLOW_RATE,
            structured_data={"cfm": 200},
            confidence=0.9,
        )
        size_anno = ParsedAnnotation(
            raw_text="24x24",
            annotation_type=AnnotationType.SIZE,
            structured_data={"width": 24, "height": 24},
            confidence=0.9,
        )
        assoc1 = TextAssociation(
            annotation=cfm_anno,
            element=elem,
            association_type=AssociationType.SPECIFIES,
            confidence=0.9,
            distance=5.0,
        )
        assoc2 = TextAssociation(
            annotation=size_anno,
            element=elem,
            association_type=AssociationType.SPECIFIES,
            confidence=0.9,
            distance=5.0,
        )
        elem_with_annos = ElementWithAnnotations(
            element=elem,
            annotations=[assoc1, assoc2],
        )
        specs = elem_with_annos.specs
        assert specs.get("cfm") == 200
        assert specs.get("width") == 24
        assert specs.get("height") == 24


class TestAssociateTextToElements:
    """Tests for associate_text_to_elements function."""

    def test_associate_nearby_text(self):
        """Should associate text with nearby elements."""
        annotations = [
            ParsedAnnotation(
                raw_text="VAV-101",
                annotation_type=AnnotationType.EQUIPMENT_TAG,
                structured_data={"tag": "VAV-101"},
                confidence=0.9,
                position=(105.0, 95.0),
            ),
        ]
        elements = [
            DetectedElement(
                element_id="sym-001",
                element_type="symbol",
                position=(100.0, 100.0),
                bounds=(80.0, 80.0, 40.0, 40.0),
                category="mechanical",
            ),
        ]
        associations = associate_text_to_elements(annotations, elements)
        assert len(associations) == 1
        assert associations[0].element.element_id == "sym-001"

    def test_no_association_for_distant_text(self):
        """Should not associate text that is too far."""
        annotations = [
            ParsedAnnotation(
                raw_text="VAV-101",
                annotation_type=AnnotationType.EQUIPMENT_TAG,
                structured_data={"tag": "VAV-101"},
                confidence=0.9,
                position=(500.0, 500.0),  # Far away
            ),
        ]
        elements = [
            DetectedElement(
                element_id="sym-001",
                element_type="symbol",
                position=(100.0, 100.0),
                bounds=(80.0, 80.0, 40.0, 40.0),
            ),
        ]
        associations = associate_text_to_elements(
            annotations, elements, max_distance=50.0
        )
        assert len(associations) == 0

    def test_skip_unknown_annotations(self):
        """Should skip UNKNOWN annotation types."""
        annotations = [
            ParsedAnnotation(
                raw_text="gibberish",
                annotation_type=AnnotationType.UNKNOWN,
                structured_data={},
                confidence=0.0,
                position=(100.0, 100.0),
            ),
        ]
        elements = [
            DetectedElement(
                element_id="sym-001",
                element_type="symbol",
                position=(100.0, 100.0),
                bounds=(80.0, 80.0, 40.0, 40.0),
            ),
        ]
        associations = associate_text_to_elements(annotations, elements)
        assert len(associations) == 0

    def test_room_containment(self):
        """Should associate room text only when contained."""
        annotations = [
            ParsedAnnotation(
                raw_text="LOBBY",
                annotation_type=AnnotationType.ROOM_NAME,
                structured_data={},
                confidence=0.9,
                position=(150.0, 150.0),  # Inside room
            ),
        ]
        elements = [
            DetectedElement(
                element_id="room-001",
                element_type="room_boundary",
                position=(150.0, 150.0),
                bounds=(100.0, 100.0, 100.0, 100.0),  # 100x100 room
            ),
        ]
        associations = associate_text_to_elements(annotations, elements)
        assert len(associations) == 1
        assert associations[0].association_type == AssociationType.LABELS

    def test_room_outside_not_associated(self):
        """Should not associate room text outside boundary."""
        annotations = [
            ParsedAnnotation(
                raw_text="LOBBY",
                annotation_type=AnnotationType.ROOM_NAME,
                structured_data={},
                confidence=0.9,
                position=(50.0, 50.0),  # Outside room
            ),
        ]
        elements = [
            DetectedElement(
                element_id="room-001",
                element_type="room_boundary",
                position=(150.0, 150.0),
                bounds=(100.0, 100.0, 100.0, 100.0),
            ),
        ]
        associations = associate_text_to_elements(annotations, elements)
        assert len(associations) == 0

    def test_closest_element_wins(self):
        """Should associate with closest element."""
        annotations = [
            ParsedAnnotation(
                raw_text="200 CFM",
                annotation_type=AnnotationType.FLOW_RATE,
                structured_data={"cfm": 200},
                confidence=0.9,
                position=(110.0, 100.0),
            ),
        ]
        elements = [
            DetectedElement(
                element_id="sym-001",
                element_type="symbol",
                position=(100.0, 100.0),
                bounds=(80.0, 80.0, 40.0, 40.0),
                category="mechanical",
            ),
            DetectedElement(
                element_id="sym-002",
                element_type="symbol",
                position=(200.0, 100.0),
                bounds=(180.0, 80.0, 40.0, 40.0),
                category="mechanical",
            ),
        ]
        associations = associate_text_to_elements(annotations, elements)
        assert len(associations) == 1
        assert associations[0].element.element_id == "sym-001"  # Closer


class TestGroupAnnotationsByElement:
    """Tests for group_annotations_by_element function."""

    def test_group_multiple_annotations(self):
        """Should group multiple annotations for same element."""
        elem = DetectedElement(
            element_id="sym-001",
            element_type="symbol",
            position=(100.0, 100.0),
            bounds=(80.0, 80.0, 40.0, 40.0),
        )
        anno1 = ParsedAnnotation(
            raw_text="AHU-1",
            annotation_type=AnnotationType.EQUIPMENT_TAG,
            structured_data={"tag": "AHU-1"},
            confidence=0.9,
        )
        anno2 = ParsedAnnotation(
            raw_text="5 HP",
            annotation_type=AnnotationType.POWER,
            structured_data={"hp": 5},
            confidence=0.9,
        )
        assoc1 = TextAssociation(
            annotation=anno1,
            element=elem,
            association_type=AssociationType.LABELS,
            confidence=0.9,
            distance=5.0,
        )
        assoc2 = TextAssociation(
            annotation=anno2,
            element=elem,
            association_type=AssociationType.SPECIFIES,
            confidence=0.9,
            distance=10.0,
        )
        grouped = group_annotations_by_element([assoc1, assoc2])
        assert len(grouped) == 1
        assert len(grouped[0].annotations) == 2

    def test_separate_different_elements(self):
        """Should keep annotations for different elements separate."""
        elem1 = DetectedElement(
            element_id="sym-001",
            element_type="symbol",
            position=(100.0, 100.0),
            bounds=(0, 0, 10, 10),
        )
        elem2 = DetectedElement(
            element_id="sym-002",
            element_type="symbol",
            position=(200.0, 200.0),
            bounds=(0, 0, 10, 10),
        )
        anno1 = ParsedAnnotation(
            raw_text="AHU-1",
            annotation_type=AnnotationType.EQUIPMENT_TAG,
            structured_data={},
            confidence=0.9,
        )
        anno2 = ParsedAnnotation(
            raw_text="AHU-2",
            annotation_type=AnnotationType.EQUIPMENT_TAG,
            structured_data={},
            confidence=0.9,
        )
        assoc1 = TextAssociation(
            annotation=anno1,
            element=elem1,
            association_type=AssociationType.LABELS,
            confidence=0.9,
            distance=5.0,
        )
        assoc2 = TextAssociation(
            annotation=anno2,
            element=elem2,
            association_type=AssociationType.LABELS,
            confidence=0.9,
            distance=5.0,
        )
        grouped = group_annotations_by_element([assoc1, assoc2])
        assert len(grouped) == 2


class TestFindRoomAnnotations:
    """Tests for find_room_annotations function."""

    def test_find_room_name_and_number(self):
        """Should find room name and number inside boundary."""
        annotations = [
            ParsedAnnotation(
                raw_text="CONFERENCE",
                annotation_type=AnnotationType.ROOM_NAME,
                structured_data={},
                confidence=0.9,
                position=(150.0, 140.0),
            ),
            ParsedAnnotation(
                raw_text="101",
                annotation_type=AnnotationType.ROOM_NUMBER,
                structured_data={"room_number": "101"},
                confidence=0.9,
                position=(150.0, 160.0),
            ),
        ]
        rooms = [
            DetectedElement(
                element_id="room-001",
                element_type="room_boundary",
                position=(150.0, 150.0),
                bounds=(100.0, 100.0, 100.0, 100.0),
            ),
        ]
        room_info = find_room_annotations(annotations, rooms)
        assert "room-001" in room_info
        info = room_info["room-001"]
        assert info["name"] == "CONFERENCE"
        assert info["number"] == "101"

    def test_find_room_area(self):
        """Should find room area inside boundary."""
        annotations = [
            ParsedAnnotation(
                raw_text="245 SF",
                annotation_type=AnnotationType.AREA,
                structured_data={"area_sf": 245.0},
                confidence=0.9,
                position=(150.0, 150.0),
            ),
        ]
        rooms = [
            DetectedElement(
                element_id="room-001",
                element_type="room_boundary",
                position=(150.0, 150.0),
                bounds=(100.0, 100.0, 100.0, 100.0),
            ),
        ]
        room_info = find_room_annotations(annotations, rooms)
        assert room_info["room-001"]["area_sf"] == 245.0


class TestEnrichElementsWithText:
    """Tests for enrich_elements_with_text function."""

    def test_enrich_element(self):
        """Should enrich element with associated text data."""
        elements = [
            DetectedElement(
                element_id="sym-001",
                element_type="symbol",
                position=(100.0, 100.0),
                bounds=(80.0, 80.0, 40.0, 40.0),
                category="mechanical",
                subtype="diffuser",
            ),
        ]
        annotations = [
            ParsedAnnotation(
                raw_text="200 CFM",
                annotation_type=AnnotationType.FLOW_RATE,
                structured_data={"cfm": 200},
                confidence=0.9,
                position=(100.0, 130.0),
            ),
        ]
        enriched = enrich_elements_with_text(elements, annotations)
        assert len(enriched) == 1
        assert enriched[0]["specs"].get("cfm") == 200

    def test_include_elements_without_annotations(self):
        """Should include elements that have no annotations."""
        elements = [
            DetectedElement(
                element_id="sym-001",
                element_type="symbol",
                position=(100.0, 100.0),
                bounds=(80.0, 80.0, 40.0, 40.0),
            ),
        ]
        annotations = []  # No annotations
        enriched = enrich_elements_with_text(elements, annotations)
        assert len(enriched) == 1
        assert enriched[0]["element_id"] == "sym-001"
        assert enriched[0]["tags"] == []
        assert enriched[0]["specs"] == {}


class TestCategoryFiltering:
    """Tests for category-based filtering in associations."""

    def test_flow_rate_prefers_mechanical(self):
        """Flow rate should prefer mechanical category elements."""
        annotations = [
            ParsedAnnotation(
                raw_text="200 CFM",
                annotation_type=AnnotationType.FLOW_RATE,
                structured_data={"cfm": 200},
                confidence=0.9,
                position=(100.0, 100.0),
            ),
        ]
        elements = [
            DetectedElement(
                element_id="elec-001",
                element_type="symbol",
                position=(100.0, 100.0),
                bounds=(80.0, 80.0, 40.0, 40.0),
                category="electrical",  # Wrong category
            ),
            DetectedElement(
                element_id="mech-001",
                element_type="symbol",
                position=(110.0, 100.0),
                bounds=(90.0, 80.0, 40.0, 40.0),
                category="mechanical",  # Right category
            ),
        ]
        associations = associate_text_to_elements(annotations, elements)
        # Should associate with mechanical element
        if associations:
            assert associations[0].element.element_id == "mech-001"
