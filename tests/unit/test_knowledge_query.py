"""
Unit tests for knowledge_query.py - Phase F Knowledge Grounding.

Tests CAD standards lookup, element grounding, and YAML loading functionality.
"""

import pytest
from dataclasses import dataclass
from typing import Any
from unittest.mock import patch, MagicMock, AsyncMock

from aec_agent.mcp.tools.knowledge_query import (
    CADStandards,
    Discipline,
    SystemType,
    GroundedElement,
    KnowledgeGroundingResult,
    LLMQueryResult,
    KnowledgeLLM,
    query_cad_standards,
    ground_element,
    ground_elements,
    get_layer_for_element,
    get_block_name,
    get_color_for_system,
    query_with_llm,
    _parse_system_type,
    _parse_discipline,
    _infer_system_from_element,
    _infer_discipline_from_element,
    _build_default_standards,
    _load_yaml_standards,
)


# =============================================================================
# Test Fixtures
# =============================================================================

@dataclass
class MockSymbol:
    """Mock symbol for testing."""
    id: str
    type: str
    subtype: str | None = None
    system: str | None = None
    discipline: str | None = None
    size: str | None = None
    position: tuple[float, float] | None = None


@pytest.fixture
def valve_symbol():
    """Create a mock valve symbol."""
    return MockSymbol(
        id="valve-001",
        type="valve",
        subtype="gate",
        system="domestic_cold_water",
        position=(100.0, 200.0),
        size='3/4"',
    )


@pytest.fixture
def diffuser_symbol():
    """Create a mock diffuser symbol."""
    return MockSymbol(
        id="diff-001",
        type="diffuser",
        subtype="square",
        system="supply_air",
        position=(150.0, 250.0),
        size="24x24",
    )


@pytest.fixture
def outlet_symbol():
    """Create a mock electrical outlet symbol."""
    return MockSymbol(
        id="outlet-001",
        type="outlet",
        subtype="duplex",
        system="power",
        position=(300.0, 150.0),
    )


@pytest.fixture
def smoke_detector_symbol():
    """Create a mock smoke detector symbol."""
    return MockSymbol(
        id="smoke-001",
        type="smoke_detector",
        subtype="photoelectric",
        system="fire_alarm",
        position=(200.0, 300.0),
    )


@pytest.fixture
def camera_symbol():
    """Create a mock camera symbol."""
    return MockSymbol(
        id="cam-001",
        type="camera",
        subtype="dome",
        system="security",
        position=(400.0, 100.0),
    )


@pytest.fixture
def mixed_symbols(valve_symbol, diffuser_symbol, outlet_symbol, smoke_detector_symbol, camera_symbol):
    """Create a list of mixed symbols."""
    return [valve_symbol, diffuser_symbol, outlet_symbol, smoke_detector_symbol, camera_symbol]


# =============================================================================
# Test SystemType Parsing
# =============================================================================

class TestSystemTypeParsing:
    """Tests for system type string parsing."""

    def test_parse_direct_match(self):
        """Test direct enum value matching."""
        assert _parse_system_type("domestic_cold_water") == SystemType.DOMESTIC_COLD_WATER
        assert _parse_system_type("supply_air") == SystemType.SUPPLY_AIR
        assert _parse_system_type("fire_alarm") == SystemType.FIRE_ALARM

    def test_parse_with_dashes(self):
        """Test parsing with dashes instead of underscores."""
        assert _parse_system_type("domestic-cold-water") == SystemType.DOMESTIC_COLD_WATER
        assert _parse_system_type("supply-air") == SystemType.SUPPLY_AIR

    def test_parse_with_spaces(self):
        """Test parsing with spaces."""
        assert _parse_system_type("domestic cold water") == SystemType.DOMESTIC_COLD_WATER

    def test_parse_aliases(self):
        """Test common abbreviation aliases."""
        assert _parse_system_type("dcw") == SystemType.DOMESTIC_COLD_WATER
        assert _parse_system_type("dhw") == SystemType.DOMESTIC_HOT_WATER
        assert _parse_system_type("sa") == SystemType.SUPPLY_AIR
        assert _parse_system_type("ra") == SystemType.RETURN_AIR
        assert _parse_system_type("fa") == SystemType.FIRE_ALARM
        assert _parse_system_type("chw") == SystemType.CHILLED_WATER

    def test_parse_simple_names(self):
        """Test simple name aliases."""
        assert _parse_system_type("cold") == SystemType.DOMESTIC_COLD_WATER
        assert _parse_system_type("hot") == SystemType.DOMESTIC_HOT_WATER
        assert _parse_system_type("supply") == SystemType.SUPPLY_AIR
        assert _parse_system_type("return") == SystemType.RETURN_AIR
        assert _parse_system_type("exhaust") == SystemType.EXHAUST_AIR
        assert _parse_system_type("gas") == SystemType.NATURAL_GAS
        assert _parse_system_type("sprinkler") == SystemType.FIRE_SUPPRESSION

    def test_parse_unknown(self):
        """Test unknown system type returns UNKNOWN."""
        assert _parse_system_type("foobar") == SystemType.UNKNOWN
        assert _parse_system_type("") == SystemType.UNKNOWN


# =============================================================================
# Test Discipline Parsing
# =============================================================================

class TestDisciplineParsing:
    """Tests for discipline string parsing."""

    def test_parse_direct_match(self):
        """Test direct enum value matching."""
        assert _parse_discipline("plumbing") == Discipline.PLUMBING
        assert _parse_discipline("mechanical") == Discipline.MECHANICAL
        assert _parse_discipline("electrical") == Discipline.ELECTRICAL
        assert _parse_discipline("fire") == Discipline.FIRE

    def test_parse_aliases(self):
        """Test discipline aliases."""
        assert _parse_discipline("hvac") == Discipline.MECHANICAL
        assert _parse_discipline("mech") == Discipline.MECHANICAL
        assert _parse_discipline("elec") == Discipline.ELECTRICAL
        assert _parse_discipline("plmb") == Discipline.PLUMBING
        assert _parse_discipline("fp") == Discipline.FIRE
        assert _parse_discipline("telecom") == Discipline.LOW_VOLTAGE
        assert _parse_discipline("data") == Discipline.LOW_VOLTAGE
        assert _parse_discipline("security") == Discipline.LOW_VOLTAGE

    def test_parse_unknown(self):
        """Test unknown discipline returns UNKNOWN."""
        assert _parse_discipline("unknown_discipline") == Discipline.UNKNOWN


# =============================================================================
# Test System Inference
# =============================================================================

class TestSystemInference:
    """Tests for inferring system from element type."""

    def test_infer_plumbing_systems(self):
        """Test inferring plumbing systems."""
        assert _infer_system_from_element("valve", None) == SystemType.DOMESTIC_COLD_WATER
        assert _infer_system_from_element("pipe", None) == SystemType.DOMESTIC_COLD_WATER
        assert _infer_system_from_element("fitting", None) == SystemType.DOMESTIC_COLD_WATER
        assert _infer_system_from_element("fixture", None) == SystemType.SANITARY

    def test_infer_hvac_systems(self):
        """Test inferring HVAC systems."""
        assert _infer_system_from_element("duct", None) == SystemType.SUPPLY_AIR
        assert _infer_system_from_element("diffuser", None) == SystemType.SUPPLY_AIR
        assert _infer_system_from_element("grille", None) == SystemType.RETURN_AIR
        assert _infer_system_from_element("vav", None) == SystemType.SUPPLY_AIR

    def test_infer_electrical_systems(self):
        """Test inferring electrical systems."""
        assert _infer_system_from_element("outlet", None) == SystemType.POWER
        assert _infer_system_from_element("switch", None) == SystemType.LIGHTING
        assert _infer_system_from_element("light", None) == SystemType.LIGHTING
        assert _infer_system_from_element("panel", None) == SystemType.POWER

    def test_infer_fire_systems(self):
        """Test inferring fire systems."""
        assert _infer_system_from_element("smoke_detector", None) == SystemType.FIRE_ALARM
        assert _infer_system_from_element("heat_detector", None) == SystemType.FIRE_ALARM
        assert _infer_system_from_element("sprinkler", None) == SystemType.FIRE_SUPPRESSION
        assert _infer_system_from_element("horn_strobe", None) == SystemType.FIRE_ALARM

    def test_infer_low_voltage_systems(self):
        """Test inferring low voltage systems."""
        assert _infer_system_from_element("data_outlet", None) == SystemType.DATA
        assert _infer_system_from_element("camera", None) == SystemType.SECURITY
        assert _infer_system_from_element("card_reader", None) == SystemType.SECURITY
        assert _infer_system_from_element("speaker", None) == SystemType.AV

    def test_infer_unknown(self):
        """Test unknown element returns UNKNOWN system."""
        assert _infer_system_from_element("unknown_element", None) == SystemType.UNKNOWN


# =============================================================================
# Test Discipline Inference
# =============================================================================

class TestDisciplineInference:
    """Tests for inferring discipline from element type."""

    def test_infer_from_element_type(self):
        """Test inferring discipline from element type."""
        assert _infer_discipline_from_element("valve", None) == Discipline.PLUMBING
        assert _infer_discipline_from_element("duct", None) == Discipline.MECHANICAL
        assert _infer_discipline_from_element("outlet", None) == Discipline.ELECTRICAL
        assert _infer_discipline_from_element("smoke_detector", None) == Discipline.FIRE
        assert _infer_discipline_from_element("camera", None) == Discipline.LOW_VOLTAGE

    def test_infer_from_system_type(self):
        """Test inferring discipline from system type."""
        assert _infer_discipline_from_element("unknown", SystemType.SUPPLY_AIR) == Discipline.MECHANICAL
        assert _infer_discipline_from_element("unknown", SystemType.DOMESTIC_COLD_WATER) == Discipline.PLUMBING
        assert _infer_discipline_from_element("unknown", SystemType.POWER) == Discipline.ELECTRICAL
        assert _infer_discipline_from_element("unknown", SystemType.FIRE_ALARM) == Discipline.FIRE
        assert _infer_discipline_from_element("unknown", SystemType.DATA) == Discipline.LOW_VOLTAGE


# =============================================================================
# Test Default Standards Building
# =============================================================================

class TestDefaultStandards:
    """Tests for building default CAD standards."""

    def test_valve_standards(self):
        """Test default standards for valves."""
        standards = _build_default_standards(
            element_type="valve",
            subtype="gate",
            system=SystemType.DOMESTIC_COLD_WATER,
            discipline=Discipline.PLUMBING,
            size='3/4"',
        )

        assert standards.layer == "P-DOMW-VALV"
        assert standards.color == 5  # Blue for cold water
        assert standards.block_name == "P-VALV-GATE"
        assert standards.linetype == "CONTINUOUS"
        assert standards.discipline == Discipline.PLUMBING
        assert standards.system == SystemType.DOMESTIC_COLD_WATER
        assert standards.attributes.get("SIZE") == '3/4"'
        assert "TYPE" in standards.attributes
        assert "TAG" in standards.attributes

    def test_diffuser_standards(self):
        """Test default standards for diffusers."""
        standards = _build_default_standards(
            element_type="diffuser",
            subtype="square",
            system=SystemType.SUPPLY_AIR,
            discipline=Discipline.MECHANICAL,
            size="24x24",
        )

        assert standards.layer == "M-HVAC-DIFF"
        assert standards.color == 5  # Blue for supply air
        assert standards.block_name == "M-DIFF-SQ"
        assert standards.discipline == Discipline.MECHANICAL
        assert standards.system == SystemType.SUPPLY_AIR

    def test_outlet_standards(self):
        """Test default standards for outlets."""
        standards = _build_default_standards(
            element_type="outlet",
            subtype="duplex",
            system=SystemType.POWER,
            discipline=Discipline.ELECTRICAL,
            size=None,
        )

        assert standards.layer == "E-POWR-OUTL"
        assert standards.color == 1  # Red for power
        assert standards.block_name == "E-OUTL-DUP"
        assert standards.discipline == Discipline.ELECTRICAL
        assert standards.system == SystemType.POWER

    def test_smoke_detector_standards(self):
        """Test default standards for smoke detectors."""
        standards = _build_default_standards(
            element_type="smoke_detector",
            subtype="photoelectric",
            system=SystemType.FIRE_ALARM,
            discipline=Discipline.FIRE,
            size=None,
        )

        assert standards.layer == "F-ALRM-DETC"
        assert standards.color == 1  # Red for fire alarm
        assert standards.block_name == "F-DETC-SMOK-P"
        assert standards.discipline == Discipline.FIRE
        assert standards.system == SystemType.FIRE_ALARM

    def test_camera_standards(self):
        """Test default standards for cameras."""
        standards = _build_default_standards(
            element_type="camera",
            subtype="dome",
            system=SystemType.SECURITY,
            discipline=Discipline.LOW_VOLTAGE,
            size=None,
        )

        assert "SECU" in standards.layer or "T-" in standards.layer
        assert standards.color == 1  # Red for security
        assert standards.block_name == "T-SECU-CAM-D"
        assert standards.discipline == Discipline.LOW_VOLTAGE
        assert standards.system == SystemType.SECURITY

    def test_unknown_element_standards(self):
        """Test standards for unknown element type."""
        standards = _build_default_standards(
            element_type="unknown_widget",
            subtype=None,
            system=None,
            discipline=None,
            size=None,
        )

        assert standards.layer == "0"  # Default layer
        assert standards.block_name is None
        assert standards.discipline == Discipline.UNKNOWN
        assert standards.system == SystemType.UNKNOWN
        assert standards.source == "default"


# =============================================================================
# Test Query CAD Standards
# =============================================================================

class TestQueryCADStandards:
    """Tests for querying CAD standards."""

    @pytest.mark.asyncio
    async def test_query_valve_standards(self):
        """Test querying standards for a valve."""
        standards = await query_cad_standards(
            element_type="valve",
            subtype="gate",
            system="domestic_cold_water",
        )

        assert standards.layer == "P-DOMW-VALV"
        assert standards.block_name == "P-VALV-GATE"
        assert standards.color == 5
        assert standards.discipline == Discipline.PLUMBING

    @pytest.mark.asyncio
    async def test_query_diffuser_standards(self):
        """Test querying standards for a diffuser."""
        standards = await query_cad_standards(
            element_type="diffuser",
            subtype="round",
            system="supply_air",
        )

        assert "DIFF" in standards.layer
        assert standards.block_name == "M-DIFF-RD"
        assert standards.discipline == Discipline.MECHANICAL

    @pytest.mark.asyncio
    async def test_query_with_system_enum(self):
        """Test querying with SystemType enum."""
        standards = await query_cad_standards(
            element_type="valve",
            subtype="ball",
            system=SystemType.DOMESTIC_HOT_WATER,
        )

        assert "DOMW" in standards.layer or "HOT" in standards.layer
        assert standards.color == 1  # Red for hot water
        assert standards.block_name == "P-VALV-BALL"

    @pytest.mark.asyncio
    async def test_query_with_discipline_enum(self):
        """Test querying with Discipline enum."""
        standards = await query_cad_standards(
            element_type="outlet",
            discipline=Discipline.ELECTRICAL,
        )

        assert standards.discipline == Discipline.ELECTRICAL
        assert "E-" in standards.layer or "OUTL" in standards.layer

    @pytest.mark.asyncio
    async def test_query_with_size(self):
        """Test querying with size specification."""
        standards = await query_cad_standards(
            element_type="valve",
            subtype="gate",
            system="domestic_cold_water",
            size='2"',
        )

        assert standards.attributes.get("SIZE") == '2"'

    @pytest.mark.asyncio
    async def test_query_normalizes_input(self):
        """Test that input strings are normalized."""
        # With dashes
        standards1 = await query_cad_standards(
            element_type="smoke-detector",
            subtype="photo-electric",
        )

        # With underscores
        standards2 = await query_cad_standards(
            element_type="smoke_detector",
            subtype="photoelectric",
        )

        # Should get same layer
        assert standards1.layer == standards2.layer

    @pytest.mark.asyncio
    async def test_query_with_project_standards(self):
        """Test querying with project-specific standards."""
        project_standards = {
            "plumbing": {
                "valve": {
                    "gate": {
                        "layer": "PROJ-VALVE-GATE",
                        "color": 30,
                        "block_name": "PROJ-VALVE",
                    }
                }
            }
        }

        standards = await query_cad_standards(
            element_type="valve",
            subtype="gate",
            system="domestic_cold_water",
            discipline="plumbing",
            project_standards=project_standards,
        )

        assert standards.layer == "PROJ-VALVE-GATE"
        assert standards.color == 30
        assert standards.source == "project"


# =============================================================================
# Test Ground Element
# =============================================================================

class TestGroundElement:
    """Tests for grounding individual elements."""

    @pytest.mark.asyncio
    async def test_ground_valve(self, valve_symbol):
        """Test grounding a valve symbol."""
        grounded = await ground_element(valve_symbol)

        assert grounded.element_id == "valve-001"
        assert grounded.element_type == "valve"
        assert grounded.subtype == "gate"
        assert grounded.position == (100.0, 200.0)
        assert grounded.standards is not None
        assert grounded.standards.block_name == "P-VALV-GATE"

    @pytest.mark.asyncio
    async def test_ground_diffuser(self, diffuser_symbol):
        """Test grounding a diffuser symbol."""
        grounded = await ground_element(diffuser_symbol)

        assert grounded.element_type == "diffuser"
        assert grounded.standards.block_name == "M-DIFF-SQ"
        assert grounded.standards.discipline == Discipline.MECHANICAL

    @pytest.mark.asyncio
    async def test_ground_outlet(self, outlet_symbol):
        """Test grounding an electrical outlet."""
        grounded = await ground_element(outlet_symbol)

        assert grounded.element_type == "outlet"
        assert grounded.standards.block_name == "E-OUTL-DUP"
        assert grounded.standards.discipline == Discipline.ELECTRICAL

    @pytest.mark.asyncio
    async def test_ground_with_project_standards(self, valve_symbol):
        """Test grounding with project-specific standards."""
        project_standards = {
            "plumbing": {
                "valve": {
                    "gate": {
                        "layer": "CUSTOM-VALVE",
                        "block_name": "CUSTOM-GATE",
                    }
                }
            }
        }

        grounded = await ground_element(
            valve_symbol,
            project_standards=project_standards,
        )

        assert grounded.standards.source == "project"


# =============================================================================
# Test Ground Elements (Batch)
# =============================================================================

class TestGroundElements:
    """Tests for grounding multiple elements."""

    @pytest.mark.asyncio
    async def test_ground_mixed_elements(self, mixed_symbols):
        """Test grounding a mixed list of symbols."""
        result = await ground_elements(mixed_symbols)

        assert isinstance(result, KnowledgeGroundingResult)
        assert len(result.grounded_elements) == 5
        assert result.standards_applied == 5
        assert result.standards_missing == 0
        assert result.statistics["coverage_percent"] == 100.0

    @pytest.mark.asyncio
    async def test_ground_elements_statistics(self, mixed_symbols):
        """Test statistics from grounding."""
        result = await ground_elements(mixed_symbols)

        assert "total_elements" in result.statistics
        assert result.statistics["total_elements"] == 5
        assert "by_source" in result.statistics
        assert "default" in result.statistics["by_source"]

    @pytest.mark.asyncio
    async def test_ground_empty_list(self):
        """Test grounding an empty list."""
        result = await ground_elements([])

        assert len(result.grounded_elements) == 0
        assert result.standards_applied == 0
        assert result.statistics["coverage_percent"] == 0

    @pytest.mark.asyncio
    async def test_ground_elements_preserves_order(self, mixed_symbols):
        """Test that element order is preserved."""
        result = await ground_elements(mixed_symbols)

        assert result.grounded_elements[0].element_type == "valve"
        assert result.grounded_elements[1].element_type == "diffuser"
        assert result.grounded_elements[2].element_type == "outlet"
        assert result.grounded_elements[3].element_type == "smoke_detector"
        assert result.grounded_elements[4].element_type == "camera"


# =============================================================================
# Test Convenience Functions
# =============================================================================

class TestConvenienceFunctions:
    """Tests for convenience/helper functions."""

    def test_get_layer_for_element(self):
        """Test get_layer_for_element function."""
        assert get_layer_for_element("valve", "domestic_cold_water") == "P-DOMW-VALV"
        assert get_layer_for_element("diffuser", "supply_air") == "M-HVAC-DIFF"
        assert get_layer_for_element("outlet", "power") == "E-POWR-OUTL"
        assert get_layer_for_element("smoke_detector", "fire_alarm") == "F-ALRM-DETC"

    def test_get_layer_with_system_enum(self):
        """Test get_layer_for_element with SystemType enum."""
        layer = get_layer_for_element("valve", SystemType.DOMESTIC_HOT_WATER)
        assert "DOMW" in layer or "HOT" in layer

    def test_get_layer_infers_system(self):
        """Test that system is inferred if not provided."""
        layer = get_layer_for_element("valve")
        assert "VALV" in layer

    def test_get_block_name(self):
        """Test get_block_name function."""
        assert get_block_name("valve", "gate") == "P-VALV-GATE"
        assert get_block_name("valve", "ball") == "P-VALV-BALL"
        assert get_block_name("diffuser", "square") == "M-DIFF-SQ"
        assert get_block_name("outlet", "duplex") == "E-OUTL-DUP"
        assert get_block_name("smoke_detector", "photoelectric") == "F-DETC-SMOK-P"

    def test_get_block_name_fallback(self):
        """Test fallback to generic block name."""
        assert get_block_name("valve", None) == "P-VALV"
        assert get_block_name("diffuser", None) == "M-DIFF"
        assert get_block_name("outlet", None) == "E-OUTL"

    def test_get_block_name_unknown(self):
        """Test unknown element returns None."""
        assert get_block_name("unknown_element", "unknown_subtype") is None

    def test_get_color_for_system(self):
        """Test get_color_for_system function."""
        assert get_color_for_system("domestic_cold_water") == 5  # Blue
        assert get_color_for_system("domestic_hot_water") == 1  # Red
        assert get_color_for_system("supply_air") == 5  # Blue
        assert get_color_for_system("return_air") == 4  # Cyan
        assert get_color_for_system("fire_alarm") == 1  # Red
        assert get_color_for_system("power") == 1  # Red

    def test_get_color_for_system_enum(self):
        """Test get_color_for_system with SystemType enum."""
        assert get_color_for_system(SystemType.DOMESTIC_COLD_WATER) == 5
        assert get_color_for_system(SystemType.FIRE_ALARM) == 1

    def test_get_color_unknown_system(self):
        """Test unknown system returns white/black default."""
        assert get_color_for_system("unknown_system") == 7
        assert get_color_for_system(SystemType.UNKNOWN) == 7


# =============================================================================
# Test YAML Loading
# =============================================================================

class TestYAMLLoading:
    """Tests for YAML standards loading."""

    def test_load_yaml_standards_missing_dir(self, tmp_path):
        """Test loading from non-existent directory."""
        missing_dir = tmp_path / "nonexistent"
        standards = _load_yaml_standards(missing_dir)
        assert standards == {}

    def test_load_yaml_standards_empty_dir(self, tmp_path):
        """Test loading from empty directory."""
        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()
        standards = _load_yaml_standards(empty_dir)
        assert standards == {}

    def test_load_yaml_standards_caches(self, tmp_path):
        """Test that YAML loading is cached."""
        from aec_agent.mcp.tools.knowledge_query import _YAML_CACHE

        test_dir = tmp_path / "test_cache"
        test_dir.mkdir()

        # First call
        _load_yaml_standards(test_dir)
        assert str(test_dir) in _YAML_CACHE

        # Clear cache for other tests
        _YAML_CACHE.clear()


# =============================================================================
# Test Block Name Patterns
# =============================================================================

class TestBlockNamePatterns:
    """Tests for block name pattern coverage."""

    @pytest.mark.parametrize("element_type,subtype,expected", [
        # Valves
        ("valve", "gate", "P-VALV-GATE"),
        ("valve", "ball", "P-VALV-BALL"),
        ("valve", "butterfly", "P-VALV-BTRF"),
        ("valve", "check", "P-VALV-CHEK"),
        ("valve", "pressure_reducing", "P-VALV-PRV"),
        # Diffusers
        ("diffuser", "square", "M-DIFF-SQ"),
        ("diffuser", "round", "M-DIFF-RD"),
        ("diffuser", "linear", "M-DIFF-LN"),
        # Grilles
        ("grille", "return", "M-GRIL-RTN"),
        ("grille", "transfer", "M-GRIL-XFR"),
        # Dampers
        ("damper", "fire", "M-DAMP-FIRE"),
        ("damper", "smoke", "M-DAMP-SMOK"),
        ("damper", "volume", "M-DAMP-VOL"),
        # Outlets
        ("outlet", "duplex", "E-OUTL-DUP"),
        ("outlet", "quad", "E-OUTL-QUAD"),
        ("outlet", "gfci", "E-OUTL-GFCI"),
        # Switches
        ("switch", "single", "E-SWTC-1P"),
        ("switch", "3way", "E-SWTC-3W"),
        ("switch", "dimmer", "E-SWTC-DIM"),
        # Lights
        ("light", "recessed", "E-LITE-REC"),
        ("light", "pendant", "E-LITE-PND"),
        ("light", "emergency", "E-LITE-EMR"),
        ("light", "exit", "E-LITE-EXIT"),
        # Fire detectors
        ("smoke_detector", "photoelectric", "F-DETC-SMOK-P"),
        ("smoke_detector", "ionization", "F-DETC-SMOK-I"),
        ("smoke_detector", "duct", "F-DETC-SMOK-D"),
        ("heat_detector", "fixed_temp", "F-DETC-HEAT-FT"),
        ("heat_detector", "rate_of_rise", "F-DETC-HEAT-RR"),
        # Fire notification
        ("horn_strobe", "wall", "F-ANUN-HS-W"),
        ("horn_strobe", "ceiling", "F-ANUN-HS-C"),
        ("pull_station", None, "F-ANUN-PULL"),
        # Sprinklers
        ("sprinkler", "pendant", "F-SPKL-PND"),
        ("sprinkler", "upright", "F-SPKL-UPR"),
        ("sprinkler", "sidewall", "F-SPKL-SW"),
        # Fixtures
        ("fixture", "water_closet", "P-FIXT-WC"),
        ("fixture", "lavatory", "P-FIXT-LAV"),
        ("fixture", "sink", "P-FIXT-SINK"),
        # Low voltage
        ("data_outlet", "rj45", "T-DATA-RJ45"),
        ("data_outlet", "fiber", "T-DATA-FBR"),
        ("camera", "dome", "T-SECU-CAM-D"),
        ("camera", "ptz", "T-SECU-CAM-PTZ"),
        ("card_reader", None, "T-SECU-CRDR"),
    ])
    def test_block_name_patterns(self, element_type, subtype, expected):
        """Test that block name patterns return expected values."""
        assert get_block_name(element_type, subtype) == expected


# =============================================================================
# Test System Layer Mapping
# =============================================================================

class TestSystemLayerMapping:
    """Tests for system-to-layer mapping."""

    @pytest.mark.parametrize("system,expected_prefix,expected_color", [
        # HVAC
        (SystemType.SUPPLY_AIR, "M-HVAC-SPLY", 5),
        (SystemType.RETURN_AIR, "M-HVAC-RETN", 4),
        (SystemType.EXHAUST_AIR, "M-HVAC-EXHA", 6),
        (SystemType.CHILLED_WATER, "M-HVAC-CHWT", 5),
        (SystemType.HOT_WATER_HEATING, "M-HVAC-HWTH", 1),
        # Plumbing
        (SystemType.DOMESTIC_COLD_WATER, "P-DOMW-COLD", 5),
        (SystemType.DOMESTIC_HOT_WATER, "P-DOMW-HOT", 1),
        (SystemType.SANITARY, "P-SNTY-PIPE", 3),
        (SystemType.VENT, "P-VENT-PIPE", 6),
        (SystemType.NATURAL_GAS, "P-NGAS-PIPE", 2),
        # Electrical
        (SystemType.POWER, "E-POWR-WIRE", 1),
        (SystemType.LIGHTING, "E-LITE-WIRE", 2),
        # Fire
        (SystemType.FIRE_ALARM, "F-ALRM-WIRE", 1),
        (SystemType.FIRE_SUPPRESSION, "F-SPKL-PIPE", 1),
        # Low Voltage
        (SystemType.DATA, "T-DATA-WIRE", 5),
        (SystemType.SECURITY, "T-SECR-WIRE", 1),
    ])
    def test_system_layer_mapping(self, system, expected_prefix, expected_color):
        """Test system-to-layer mapping returns expected values."""
        from aec_agent.mcp.tools.knowledge_query import _SYSTEM_LAYER_MAPPING

        assert system in _SYSTEM_LAYER_MAPPING
        prefix, major, minor, color = _SYSTEM_LAYER_MAPPING[system]
        full_prefix = f"{prefix}-{major}-{minor}"
        assert full_prefix == expected_prefix
        assert color == expected_color


# =============================================================================
# Test CADStandards Dataclass
# =============================================================================

class TestCADStandardsDataclass:
    """Tests for CADStandards dataclass."""

    def test_default_values(self):
        """Test default values."""
        standards = CADStandards(layer="TEST")

        assert standards.layer == "TEST"
        assert standards.color == 7
        assert standards.linetype == "CONTINUOUS"
        assert standards.lineweight == 0.25
        assert standards.block_name is None
        assert standards.attributes == {}
        assert standards.discipline == Discipline.UNKNOWN
        assert standards.system == SystemType.UNKNOWN
        assert standards.description == ""
        assert standards.source == "default"

    def test_full_initialization(self):
        """Test full initialization."""
        standards = CADStandards(
            layer="P-DOMW-VALV",
            color=5,
            linetype="CONTINUOUS",
            lineweight=0.35,
            block_name="P-VALV-GATE",
            attributes={"SIZE": '3/4"', "TAG": "V-1"},
            discipline=Discipline.PLUMBING,
            system=SystemType.DOMESTIC_COLD_WATER,
            description="Gate valve",
            source="yaml",
        )

        assert standards.layer == "P-DOMW-VALV"
        assert standards.color == 5
        assert standards.block_name == "P-VALV-GATE"
        assert standards.attributes["SIZE"] == '3/4"'
        assert standards.source == "yaml"


# =============================================================================
# Test GroundedElement Dataclass
# =============================================================================

class TestGroundedElementDataclass:
    """Tests for GroundedElement dataclass."""

    def test_default_values(self):
        """Test default values."""
        elem = GroundedElement(element_id="test", element_type="valve")

        assert elem.element_id == "test"
        assert elem.element_type == "valve"
        assert elem.subtype is None
        assert elem.position is None
        assert elem.standards is None
        assert elem.original_data == {}

    def test_full_initialization(self):
        """Test full initialization."""
        standards = CADStandards(layer="P-VALV")
        elem = GroundedElement(
            element_id="valve-001",
            element_type="valve",
            subtype="gate",
            position=(100.0, 200.0),
            standards=standards,
            original_data={"size": '3/4"'},
        )

        assert elem.element_id == "valve-001"
        assert elem.subtype == "gate"
        assert elem.position == (100.0, 200.0)
        assert elem.standards.layer == "P-VALV"
        assert elem.original_data["size"] == '3/4"'


# =============================================================================
# Test LLMQueryResult Dataclass
# =============================================================================

class TestLLMQueryResultDataclass:
    """Tests for LLMQueryResult dataclass."""

    def test_default_values(self):
        """Test default values."""
        result = LLMQueryResult(answer="Test answer")

        assert result.answer == "Test answer"
        assert result.element_type is None
        assert result.subtype is None
        assert result.system is None
        assert result.discipline is None
        assert result.recommended_layer is None
        assert result.recommended_block is None
        assert result.recommended_color is None
        assert result.attributes == {}
        assert result.code_references == []
        assert result.confidence == 0.0
        assert result.provider == ""
        assert result.raw_response == ""

    def test_full_initialization(self):
        """Test full initialization."""
        result = LLMQueryResult(
            answer="Gate valve on cold water uses layer P-DOMW-VALV",
            element_type="valve",
            subtype="gate",
            system="domestic_cold_water",
            discipline="plumbing",
            recommended_layer="P-DOMW-VALV",
            recommended_block="P-VALV-GATE",
            recommended_color=5,
            attributes={"SIZE": '3/4"', "TAG": "V-1"},
            code_references=["CPC 604.1"],
            confidence=0.95,
            provider="gemini",
            raw_response='{"answer": "..."}',
        )

        assert result.answer == "Gate valve on cold water uses layer P-DOMW-VALV"
        assert result.element_type == "valve"
        assert result.subtype == "gate"
        assert result.system == "domestic_cold_water"
        assert result.discipline == "plumbing"
        assert result.recommended_layer == "P-DOMW-VALV"
        assert result.recommended_block == "P-VALV-GATE"
        assert result.recommended_color == 5
        assert result.attributes["SIZE"] == '3/4"'
        assert "CPC 604.1" in result.code_references
        assert result.confidence == 0.95
        assert result.provider == "gemini"


# =============================================================================
# Test KnowledgeLLM Class
# =============================================================================

class TestKnowledgeLLM:
    """Tests for KnowledgeLLM class."""

    def test_initialization_default(self):
        """Test default initialization."""
        llm = KnowledgeLLM()

        assert llm.provider == "auto"
        assert llm.temperature == 0.2
        assert llm.max_tokens == 1000
        assert llm._gemini_model is None
        assert llm._openai_client is None
        assert llm._anthropic_client is None

    def test_initialization_with_params(self):
        """Test initialization with custom parameters."""
        llm = KnowledgeLLM(
            provider="gemini",
            temperature=0.5,
            max_tokens=2000,
        )

        assert llm.provider == "gemini"
        assert llm.temperature == 0.5
        assert llm.max_tokens == 2000

    def test_get_api_key_from_env(self):
        """Test getting API key from environment variables."""
        llm = KnowledgeLLM()
        llm._settings = None  # Force fallback to env vars

        with patch.dict("os.environ", {"GEMINI_API_KEY": "test-key"}):
            key = llm._get_api_key("gemini")
            assert key == "test-key"

    def test_get_api_key_gemini_fallback(self):
        """Test Gemini API key falls back to GOOGLE_API_KEY."""
        llm = KnowledgeLLM()
        llm._settings = None

        with patch.dict("os.environ", {"GOOGLE_API_KEY": "google-key"}, clear=True):
            key = llm._get_api_key("gemini")
            assert key == "google-key"

    def test_get_available_provider_auto(self):
        """Test auto provider selection."""
        llm = KnowledgeLLM(provider="auto")
        llm._settings = None

        with patch.dict("os.environ", {"GEMINI_API_KEY": "test-key"}):
            provider = llm._get_available_provider()
            assert provider == "gemini"

    def test_get_available_provider_specific(self):
        """Test specific provider selection."""
        llm = KnowledgeLLM(provider="openai")
        llm._settings = None

        with patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"}):
            provider = llm._get_available_provider()
            assert provider == "openai"

    def test_get_available_provider_none(self):
        """Test no provider available."""
        llm = KnowledgeLLM(provider="auto")
        llm._settings = None

        with patch.dict("os.environ", {}, clear=True):
            provider = llm._get_available_provider()
            assert provider is None

    def test_build_system_prompt(self):
        """Test system prompt building."""
        llm = KnowledgeLLM()
        prompt = llm._build_system_prompt()

        assert "AEC" in prompt
        assert "CAD" in prompt
        assert "NCS" in prompt
        assert "layer" in prompt.lower()
        assert "JSON" in prompt

    def test_parse_llm_response_json(self):
        """Test parsing JSON response."""
        llm = KnowledgeLLM()

        raw_json = '''```json
{
    "answer": "Use layer P-DOMW-VALV for valves",
    "element_type": "valve",
    "subtype": "gate",
    "system": "domestic_cold_water",
    "discipline": "plumbing",
    "recommended_layer": "P-DOMW-VALV",
    "recommended_block": "P-VALV-GATE",
    "recommended_color": 5,
    "attributes": {"SIZE": "", "TAG": ""},
    "code_references": ["CPC 604.1"],
    "confidence": 0.95
}
```'''

        result = llm._parse_llm_response(raw_json, "gemini")

        assert result.answer == "Use layer P-DOMW-VALV for valves"
        assert result.element_type == "valve"
        assert result.subtype == "gate"
        assert result.recommended_layer == "P-DOMW-VALV"
        assert result.recommended_block == "P-VALV-GATE"
        assert result.recommended_color == 5
        assert result.confidence == 0.95
        assert result.provider == "gemini"

    def test_parse_llm_response_raw_json(self):
        """Test parsing raw JSON response without code fence."""
        llm = KnowledgeLLM()

        raw_json = '{"answer": "Test answer", "element_type": "valve", "confidence": 0.8}'

        result = llm._parse_llm_response(raw_json, "openai")

        assert result.answer == "Test answer"
        assert result.element_type == "valve"
        assert result.confidence == 0.8
        assert result.provider == "openai"

    def test_parse_llm_response_invalid_json(self):
        """Test parsing invalid JSON falls back to raw text."""
        llm = KnowledgeLLM()

        raw_text = "This is not valid JSON, but a helpful answer about valves."

        result = llm._parse_llm_response(raw_text, "anthropic")

        assert result.answer == raw_text
        assert result.confidence == 0.5
        assert result.provider == "anthropic"

    @pytest.mark.asyncio
    async def test_query_no_provider(self):
        """Test query with no provider configured."""
        llm = KnowledgeLLM()
        llm._settings = None

        with patch.dict("os.environ", {}, clear=True):
            result = await llm.query("What layer for a valve?")

            assert "No LLM provider configured" in result.answer
            assert result.confidence == 0.0

    @pytest.mark.asyncio
    async def test_query_gemini_mock(self):
        """Test query with mocked Gemini response."""
        llm = KnowledgeLLM(provider="gemini")
        llm._settings = None

        mock_response = MagicMock()
        mock_response.text = '''```json
{
    "answer": "Valves use P-DOMW-VALV",
    "element_type": "valve",
    "recommended_layer": "P-DOMW-VALV",
    "confidence": 0.9
}
```'''

        mock_model = MagicMock()
        mock_model.generate_content_async = AsyncMock(return_value=mock_response)

        with patch.dict("os.environ", {"GEMINI_API_KEY": "test-key"}):
            with patch.object(llm, "_init_gemini", new_callable=AsyncMock, return_value=mock_model):
                result = await llm.query("What layer for a valve?")

                assert result.recommended_layer == "P-DOMW-VALV"
                assert result.element_type == "valve"
                assert result.provider == "gemini"

    @pytest.mark.asyncio
    async def test_classify_element(self):
        """Test element classification."""
        llm = KnowledgeLLM(provider="gemini")
        llm._settings = None

        mock_response = MagicMock()
        mock_response.text = '''```json
{
    "answer": "This is a 24x24 supply air diffuser",
    "element_type": "diffuser",
    "subtype": "square",
    "system": "supply_air",
    "recommended_layer": "M-HVAC-DIFF",
    "recommended_block": "M-DIFF-SQ",
    "attributes": {"SIZE": "24x24", "CFM": "200"},
    "confidence": 0.95
}
```'''

        mock_model = MagicMock()
        mock_model.generate_content_async = AsyncMock(return_value=mock_response)

        with patch.dict("os.environ", {"GEMINI_API_KEY": "test-key"}):
            with patch.object(llm, "_init_gemini", new_callable=AsyncMock, return_value=mock_model):
                result = await llm.classify_element("24x24 supply air diffuser 200 CFM")

                assert result.element_type == "diffuser"
                assert result.subtype == "square"
                assert result.recommended_block == "M-DIFF-SQ"

    @pytest.mark.asyncio
    async def test_get_code_reference(self):
        """Test code reference lookup."""
        llm = KnowledgeLLM(provider="gemini")
        llm._settings = None

        mock_response = MagicMock()
        mock_response.text = '''```json
{
    "answer": "Smoke detectors in corridors must comply with NFPA 72 and CFC",
    "element_type": "smoke_detector",
    "code_references": ["NFPA 72 17.7.3.2", "CFC 907.2.10"],
    "confidence": 0.9
}
```'''

        mock_model = MagicMock()
        mock_model.generate_content_async = AsyncMock(return_value=mock_response)

        with patch.dict("os.environ", {"GEMINI_API_KEY": "test-key"}):
            with patch.object(llm, "_init_gemini", new_callable=AsyncMock, return_value=mock_model):
                result = await llm.get_code_reference("smoke detector", "corridor")

                assert "NFPA 72" in result.code_references[0]
                assert len(result.code_references) == 2

    @pytest.mark.asyncio
    async def test_suggest_attributes(self):
        """Test attribute suggestions."""
        llm = KnowledgeLLM(provider="gemini")
        llm._settings = None

        mock_response = MagicMock()
        mock_response.text = '''```json
{
    "answer": "Diffusers should have SIZE, CFM, NC, and TAG attributes",
    "element_type": "diffuser",
    "attributes": {"SIZE": "24x24", "CFM": "", "NC": "", "TAG": ""},
    "confidence": 0.9
}
```'''

        mock_model = MagicMock()
        mock_model.generate_content_async = AsyncMock(return_value=mock_response)

        with patch.dict("os.environ", {"GEMINI_API_KEY": "test-key"}):
            with patch.object(llm, "_init_gemini", new_callable=AsyncMock, return_value=mock_model):
                result = await llm.suggest_attributes("diffuser", "square", "supply_air")

                assert "CFM" in result.attributes
                assert "SIZE" in result.attributes


# =============================================================================
# Test query_with_llm Convenience Function
# =============================================================================

class TestQueryWithLLM:
    """Tests for query_with_llm convenience function."""

    @pytest.mark.asyncio
    async def test_query_with_llm_no_provider(self):
        """Test convenience function with no provider configured."""
        with patch.dict("os.environ", {}, clear=True):
            # Need to also patch settings to return None
            with patch("aec_agent.mcp.tools.knowledge_query.KnowledgeLLM._get_api_key", return_value=None):
                result = await query_with_llm("What layer for a valve?")

                assert "No LLM provider configured" in result.answer
                assert result.confidence == 0.0

    @pytest.mark.asyncio
    async def test_query_with_llm_mock(self):
        """Test convenience function with mocked response."""
        mock_response = MagicMock()
        mock_response.text = '{"answer": "Test", "confidence": 0.8}'

        mock_model = MagicMock()
        mock_model.generate_content_async = AsyncMock(return_value=mock_response)

        with patch.dict("os.environ", {"GEMINI_API_KEY": "test-key"}):
            with patch("google.generativeai.configure"):
                with patch("google.generativeai.GenerativeModel", return_value=mock_model):
                    # Create the llm and set the mock directly
                    llm = KnowledgeLLM(provider="gemini")
                    llm._settings = None
                    llm._gemini_model = mock_model
                    result = await llm.query("Test question")

                    assert result.answer == "Test"
                    assert result.confidence == 0.8
                    assert result.provider == "gemini"
