"""Unit tests for semantic_ocr.py (Phase B)."""

import pytest

from aec_agent.mcp.tools.document_classifier import DrawingType
from aec_agent.mcp.tools.semantic_ocr import (
    AnnotationType,
    ParsedAnnotation,
    classify_room_name,
    get_annotation_category,
    parse_annotation_by_patterns,
)


class TestAnnotationType:
    """Tests for AnnotationType enum."""

    def test_all_expected_types_exist(self):
        """All expected annotation types should be defined."""
        expected = [
            "ROOM_NAME", "ROOM_NUMBER", "EQUIPMENT_TAG",
            "AREA", "DIMENSION", "ELEVATION",
            "EQUIPMENT_SPEC", "SIZE", "FLOW_RATE", "POWER", "CAPACITY",
            "REFERENCE", "QUANTITY", "SHEET_REF", "SECTION_REF",
            "CIRCUIT_ID", "PANEL_NAME",
            "NOTE", "SPECIFICATION",
            "UNKNOWN",
        ]
        for name in expected:
            assert hasattr(AnnotationType, name), f"Missing AnnotationType.{name}"

    def test_annotation_type_values_are_strings(self):
        """Annotation type values should be lowercase strings."""
        for atype in AnnotationType:
            assert isinstance(atype.value, str)
            assert atype.value == atype.value.lower()


class TestParsedAnnotation:
    """Tests for ParsedAnnotation dataclass."""

    def test_basic_creation(self):
        """Should create annotation with required fields."""
        anno = ParsedAnnotation(
            raw_text="200 CFM",
            annotation_type=AnnotationType.FLOW_RATE,
            structured_data={"cfm": 200},
            confidence=0.9,
        )
        assert anno.raw_text == "200 CFM"
        assert anno.annotation_type == AnnotationType.FLOW_RATE
        assert anno.structured_data == {"cfm": 200}
        assert anno.confidence == 0.9

    def test_default_values(self):
        """Should have sensible defaults."""
        anno = ParsedAnnotation(
            raw_text="TEST",
            annotation_type=AnnotationType.UNKNOWN,
            structured_data={},
            confidence=0.0,
        )
        assert anno.position == (0.0, 0.0)
        assert anno.bounding_box is None
        assert anno.source == "pattern"
        assert anno.metadata == {}


class TestPatternParsing:
    """Tests for pattern-based annotation parsing."""

    def test_parse_room_number_simple(self):
        """Should parse simple room numbers."""
        result = parse_annotation_by_patterns("101")
        assert result is not None
        assert result.annotation_type == AnnotationType.ROOM_NUMBER
        assert result.structured_data.get("room_number") == "101"

    def test_parse_room_number_with_letter(self):
        """Should parse room numbers with letters."""
        result = parse_annotation_by_patterns("205A")
        assert result is not None
        assert result.annotation_type == AnnotationType.ROOM_NUMBER
        assert result.structured_data.get("room_number") == "205A"

    def test_parse_area_sf(self):
        """Should parse area in square feet."""
        result = parse_annotation_by_patterns("245 SF")
        assert result is not None
        assert result.annotation_type == AnnotationType.AREA
        assert result.structured_data.get("area_sf") == 245.0
        assert result.structured_data.get("unit") == "sf"

    def test_parse_area_sf_decimal(self):
        """Should parse decimal area."""
        result = parse_annotation_by_patterns("123.5 SF")
        assert result is not None
        assert result.structured_data.get("area_sf") == 123.5

    def test_parse_cfm(self):
        """Should parse CFM flow rate."""
        result = parse_annotation_by_patterns("200 CFM")
        assert result is not None
        assert result.annotation_type == AnnotationType.FLOW_RATE
        assert result.structured_data.get("cfm") == 200.0

    def test_parse_diffuser_spec(self):
        """Should parse diffuser specification."""
        result = parse_annotation_by_patterns("24x24 SA 200 CFM")
        assert result is not None
        assert result.annotation_type == AnnotationType.EQUIPMENT_SPEC
        assert result.structured_data.get("width_inches") == 24
        assert result.structured_data.get("height_inches") == 24
        assert result.structured_data.get("cfm") == 200
        assert result.structured_data.get("air_type") == "supply_air"

    def test_parse_diffuser_return_air(self):
        """Should parse return air diffuser."""
        result = parse_annotation_by_patterns("18x18 RA 150 CFM")
        assert result is not None
        assert result.structured_data.get("air_type") == "return_air"

    def test_parse_equipment_tag_ahu(self):
        """Should parse AHU equipment tag."""
        result = parse_annotation_by_patterns("AHU-1")
        assert result is not None
        assert result.annotation_type == AnnotationType.EQUIPMENT_TAG
        assert result.structured_data.get("prefix") == "AHU"
        assert result.structured_data.get("category") == "air_handling_unit"

    def test_parse_equipment_tag_vav(self):
        """Should parse VAV equipment tag."""
        result = parse_annotation_by_patterns("VAV-101")
        assert result is not None
        assert result.structured_data.get("prefix") == "VAV"
        assert result.structured_data.get("category") == "vav_box"

    def test_parse_valve_spec(self):
        """Should parse valve specification."""
        result = parse_annotation_by_patterns("3/4 GV")
        assert result is not None
        assert result.annotation_type == AnnotationType.EQUIPMENT_SPEC
        assert result.structured_data.get("size_inches") == 0.75
        assert result.structured_data.get("valve_type") == "gate_valve"

    def test_parse_ball_valve(self):
        """Should parse ball valve."""
        result = parse_annotation_by_patterns("1 BV")
        assert result is not None
        assert result.structured_data.get("valve_type") == "ball_valve"

    def test_parse_size_wxh(self):
        """Should parse WxH size."""
        result = parse_annotation_by_patterns("24x12")
        assert result is not None
        assert result.annotation_type == AnnotationType.SIZE
        assert result.structured_data.get("width") == 24
        assert result.structured_data.get("height") == 12

    def test_parse_quantity_parentheses(self):
        """Should parse quantity in parentheses."""
        result = parse_annotation_by_patterns("(3)")
        assert result is not None
        assert result.annotation_type == AnnotationType.QUANTITY
        assert result.structured_data.get("quantity") == 3
        assert result.structured_data.get("typical") is False

    def test_parse_typical_quantity(self):
        """Should parse typical quantity."""
        result = parse_annotation_by_patterns("TYP. (5)")
        assert result is not None
        assert result.structured_data.get("quantity") == 5
        assert result.structured_data.get("typical") is True

    def test_parse_typical_alone(self):
        """Should parse TYP. alone."""
        result = parse_annotation_by_patterns("TYP.")
        assert result is not None
        assert result.annotation_type == AnnotationType.QUANTITY
        assert result.structured_data.get("typical") is True

    def test_parse_power_hp(self):
        """Should parse horsepower."""
        result = parse_annotation_by_patterns("5 HP")
        assert result is not None
        assert result.annotation_type == AnnotationType.POWER
        assert result.structured_data.get("hp") == 5.0

    def test_parse_power_kw(self):
        """Should parse kilowatts."""
        result = parse_annotation_by_patterns("3.7 kW")
        assert result is not None
        assert result.structured_data.get("kw") == 3.7

    def test_parse_capacity_tons(self):
        """Should parse cooling capacity."""
        result = parse_annotation_by_patterns("10 TON")
        assert result is not None
        assert result.annotation_type == AnnotationType.CAPACITY
        assert result.structured_data.get("tons") == 10.0

    def test_parse_circuit_id(self):
        """Should parse circuit ID."""
        result = parse_annotation_by_patterns("CKT-1A")
        assert result is not None
        assert result.annotation_type == AnnotationType.CIRCUIT_ID
        assert result.structured_data.get("circuit_id") == "1A"

    def test_parse_circuit_breaker(self):
        """Should parse circuit breaker spec."""
        result = parse_annotation_by_patterns("20A/1P")
        assert result is not None
        assert result.structured_data.get("amps") == 20
        assert result.structured_data.get("poles") == 1

    def test_parse_sheet_reference(self):
        """Should parse sheet reference."""
        result = parse_annotation_by_patterns("E1.01")
        assert result is not None
        assert result.annotation_type == AnnotationType.SHEET_REF
        assert result.structured_data.get("sheet") == "E1.01"

    def test_parse_section_reference(self):
        """Should parse section reference."""
        result = parse_annotation_by_patterns("1/A1.01")
        assert result is not None
        assert result.annotation_type == AnnotationType.SECTION_REF
        assert result.structured_data.get("detail_number") == "1"
        assert result.structured_data.get("sheet") == "A1.01"

    def test_empty_text_returns_none(self):
        """Empty text should return None."""
        result = parse_annotation_by_patterns("")
        assert result is None

    def test_whitespace_only_returns_none(self):
        """Whitespace-only text should return None."""
        result = parse_annotation_by_patterns("   ")
        assert result is None

    def test_unrecognized_text_returns_none(self):
        """Unrecognized text should return None."""
        result = parse_annotation_by_patterns("random gibberish xyz")
        assert result is None

    def test_case_insensitive(self):
        """Parsing should be case-insensitive."""
        result1 = parse_annotation_by_patterns("200 cfm")
        result2 = parse_annotation_by_patterns("200 CFM")
        assert result1 is not None
        assert result2 is not None
        assert result1.structured_data == result2.structured_data


class TestRoomNameClassification:
    """Tests for room name classification."""

    def test_common_room_names(self):
        """Should recognize common room names."""
        room_names = [
            "CONFERENCE ROOM",
            "OFFICE",
            "LOBBY",
            "CORRIDOR",
            "RESTROOM",
            "STORAGE",
            "MECHANICAL",
            "ELECTRICAL",
        ]
        for name in room_names:
            assert classify_room_name(name) is True, f"Should recognize: {name}"

    def test_non_room_names(self):
        """Should not classify non-room text as room names."""
        non_rooms = [
            "200 CFM",
            "AHU-1",
            "24x24",
            "E1.01",
        ]
        for text in non_rooms:
            assert classify_room_name(text) is False, f"Should NOT recognize: {text}"

    def test_case_insensitive(self):
        """Room name check should be case-insensitive."""
        assert classify_room_name("conference room") is True
        assert classify_room_name("CONFERENCE ROOM") is True
        assert classify_room_name("Conference Room") is True


class TestAnnotationCategories:
    """Tests for annotation category mapping."""

    def test_identifier_categories(self):
        """Identifier annotations should map to 'identifier' category."""
        assert get_annotation_category(AnnotationType.ROOM_NAME) == "identifier"
        assert get_annotation_category(AnnotationType.ROOM_NUMBER) == "identifier"
        assert get_annotation_category(AnnotationType.EQUIPMENT_TAG) == "identifier"

    def test_measurement_categories(self):
        """Measurement annotations should map to 'measurement' category."""
        assert get_annotation_category(AnnotationType.AREA) == "measurement"
        assert get_annotation_category(AnnotationType.DIMENSION) == "measurement"
        assert get_annotation_category(AnnotationType.ELEVATION) == "measurement"

    def test_equipment_categories(self):
        """Equipment annotations should map to 'equipment' category."""
        assert get_annotation_category(AnnotationType.EQUIPMENT_SPEC) == "equipment"
        assert get_annotation_category(AnnotationType.SIZE) == "equipment"
        assert get_annotation_category(AnnotationType.FLOW_RATE) == "equipment"

    def test_electrical_categories(self):
        """Electrical annotations should map to 'electrical' category."""
        assert get_annotation_category(AnnotationType.CIRCUIT_ID) == "electrical"
        assert get_annotation_category(AnnotationType.PANEL_NAME) == "electrical"

    def test_unknown_category(self):
        """Unknown annotation should map to 'unknown' category."""
        assert get_annotation_category(AnnotationType.UNKNOWN) == "unknown"


class TestDimensionParsing:
    """Tests for dimension parsing."""

    def test_parse_inches(self):
        """Should parse inches-only dimension."""
        result = parse_annotation_by_patterns('6"')
        assert result is not None
        assert result.annotation_type == AnnotationType.DIMENSION
        assert result.structured_data.get("inches") == 6.0

    def test_parse_millimeters(self):
        """Should parse millimeter dimension."""
        result = parse_annotation_by_patterns("3048mm")
        assert result is not None
        assert result.structured_data.get("mm") == 3048.0
        assert result.structured_data.get("unit") == "metric"


class TestElevationParsing:
    """Tests for elevation parsing."""

    def test_parse_elevation_decimal(self):
        """Should parse decimal elevation."""
        result = parse_annotation_by_patterns("EL. 10.5")
        assert result is not None
        assert result.annotation_type == AnnotationType.ELEVATION
        assert result.structured_data.get("elevation") == 10.5

    def test_parse_elevation_plus(self):
        """Should parse elevation with plus sign."""
        result = parse_annotation_by_patterns("+12.0")
        assert result is not None
        assert result.structured_data.get("elevation") == 12.0


class TestHighConfidencePatterns:
    """Tests verifying high confidence for pattern matches."""

    def test_pattern_match_confidence(self):
        """Pattern matches should have high confidence."""
        result = parse_annotation_by_patterns("200 CFM")
        assert result is not None
        assert result.confidence >= 0.8

    def test_pattern_source(self):
        """Pattern matches should have 'pattern' source."""
        result = parse_annotation_by_patterns("AHU-1")
        assert result is not None
        assert result.source == "pattern"
