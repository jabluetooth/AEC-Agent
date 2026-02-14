"""
Unit tests for Phase 4: Adaptive Extraction.

Tests the gemini_first.adaptive_extraction module which extracts AutoCAD
entities from Gemini's drawing analysis using direct, guided, and OpenCV strategies.
"""

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# Import module under test
from aec_agent.mcp.tools.gemini_first.adaptive_extraction import (
    ExtractionSource,
    EntityType,
    EntityToCreate,
    RasterCommand,
    ExtractionResult,
    extract_all,
    extract_direct_only,
    direct_extraction,
    guided_rasterization,
    selective_opencv,
    extract_lines_direct,
    extract_arcs_direct,
    extract_circles_direct,
    extract_text_direct,
    extract_symbols_direct,
    extract_dimensions_direct,
    get_layer_for_element_type,
    get_layer_for_text,
    get_layer_for_symbol,
    get_block_name,
    get_vtool_for_path_type,
    get_entities_by_type,
    get_entities_by_layer,
    get_required_layers,
    get_required_blocks,
    ELEMENT_TYPE_TO_LAYER,
    TEXT_TYPE_TO_LAYER,
    SYMBOL_TYPE_TO_LAYER,
    SYMBOL_TO_BLOCK,
    VTOOL_MAPPING,
)
from aec_agent.mcp.tools.gemini_first.gemini_understanding import (
    DrawingAnalysis,
    DrawingElements,
    DetectedLine,
    DetectedArc,
    DetectedCircle,
    DetectedText,
    DetectedSymbol,
    DetectedDimension,
    SpecialRegion,
    ExtractionStrategyConfig,
)
from aec_agent.mcp.tools.gemini_first.coordinate_calibration import (
    ScaleCalibration,
)


class TestEntityToCreate:
    """Tests for EntityToCreate dataclass."""

    def test_basic_creation(self):
        """Test creating an EntityToCreate."""
        entity = EntityToCreate(
            entity_type=EntityType.LINE,
            layer="A-WALL",
            properties={
                "start": (0.0, 0.0),
                "end": (100.0, 0.0),
                "linetype": "Continuous",
            },
        )

        assert entity.entity_type == EntityType.LINE
        assert entity.layer == "A-WALL"
        assert entity.properties["start"] == (0.0, 0.0)
        assert entity.source == ExtractionSource.DIRECT

    def test_to_dict(self):
        """Test serialization to dictionary."""
        entity = EntityToCreate(
            entity_type=EntityType.CIRCLE,
            layer="M-EQPM",
            properties={"center": (50.0, 50.0), "radius": 10.0},
            source=ExtractionSource.SELECTIVE_OPENCV,
            confidence=0.9,
        )

        d = entity.to_dict()
        assert d["entity_type"] == "circle"
        assert d["layer"] == "M-EQPM"
        assert d["source"] == "selective_opencv"
        assert d["confidence"] == 0.9

    def test_from_dict(self):
        """Test deserialization from dictionary."""
        data = {
            "entity_type": "line",
            "layer": "E-POWR",
            "properties": {"start": [0, 0], "end": [100, 100]},
            "source": "direct",
            "confidence": 1.0,
        }

        entity = EntityToCreate.from_dict(data)
        assert entity.entity_type == "line"
        assert entity.layer == "E-POWR"
        assert entity.confidence == 1.0


class TestRasterCommand:
    """Tests for RasterCommand dataclass."""

    def test_basic_creation(self):
        """Test creating a RasterCommand."""
        cmd = RasterCommand(
            tool="VFPLINE",
            start_point=(100.0, 200.0),
            layer="M-DUCT",
            options={"gap_jump": 3},
        )

        assert cmd.tool == "VFPLINE"
        assert cmd.start_point == (100.0, 200.0)
        assert cmd.layer == "M-DUCT"

    def test_to_dict(self):
        """Test serialization to dictionary."""
        cmd = RasterCommand(
            tool="VARC",
            start_point=(50.0, 50.0),
            layer="A-DOOR",
            expected_end=(150.0, 50.0),
        )

        d = cmd.to_dict()
        assert d["tool"] == "VARC"
        assert d["start_point"] == [50.0, 50.0]
        assert d["expected_end"] == [150.0, 50.0]

    def test_to_command_string(self):
        """Test generating AutoCAD command string."""
        cmd = RasterCommand(
            tool="VLINE",
            start_point=(100.5, 200.25),
            layer="0",
        )

        cmd_str = cmd.to_command_string()
        assert cmd_str.startswith("VLINE")
        assert "100.5" in cmd_str
        assert "200.25" in cmd_str


class TestExtractionResult:
    """Tests for ExtractionResult dataclass."""

    def test_basic_creation(self):
        """Test creating an ExtractionResult."""
        result = ExtractionResult()

        assert result.total_entities == 0
        assert result.total_commands == 0
        assert result.primary_strategy == "direct"

    def test_with_entities(self):
        """Test ExtractionResult with entities."""
        entities = [
            EntityToCreate(EntityType.LINE, "A-WALL", {"start": (0, 0), "end": (10, 10)}),
            EntityToCreate(EntityType.CIRCLE, "M-EQPM", {"center": (5, 5), "radius": 2}),
        ]

        result = ExtractionResult(
            entities=entities,
            direct_count=2,
            primary_strategy="direct",
        )

        assert result.total_entities == 2
        assert result.direct_count == 2

    def test_to_dict(self):
        """Test serialization to dictionary."""
        result = ExtractionResult(
            direct_count=5,
            opencv_count=2,
            drawing_type="floor_plan",
            calibration_confidence=0.85,
        )

        d = result.to_dict()
        assert d["statistics"]["direct_count"] == 5
        assert d["statistics"]["opencv_count"] == 2
        assert d["metadata"]["drawing_type"] == "floor_plan"


class TestLayerMapping:
    """Tests for layer mapping functions."""

    def test_get_layer_for_wall(self):
        """Test layer for wall element."""
        layer = get_layer_for_element_type("wall")
        assert layer == "A-WALL"

    def test_get_layer_for_duct(self):
        """Test layer for duct element."""
        layer = get_layer_for_element_type("duct")
        assert layer == "M-DUCT"

    def test_get_layer_for_outlet(self):
        """Test layer for outlet element."""
        layer = get_layer_for_element_type("outlet")
        assert layer == "E-POWR-OUTL"

    def test_get_layer_for_diffuser(self):
        """Test layer for diffuser element."""
        layer = get_layer_for_element_type("diffuser")
        assert layer == "M-DIFF"

    def test_get_layer_for_detector(self):
        """Test layer for detector element."""
        layer = get_layer_for_element_type("detector")
        assert layer == "F-ALRM-DETC"

    def test_get_layer_with_subtype(self):
        """Test layer with subtype."""
        layer = get_layer_for_element_type("supply", "diffuser")
        # Falls back to base type or default
        assert layer in ("M-DIFF-SUPP", "0")

    def test_get_layer_unknown(self):
        """Test unknown element returns default layer."""
        layer = get_layer_for_element_type("unknown_element")
        assert layer == "0"

    def test_get_layer_case_insensitive(self):
        """Test that layer lookup is case insensitive."""
        layer = get_layer_for_element_type("WALL")
        assert layer == "A-WALL"

    def test_get_layer_with_dashes(self):
        """Test element type with dashes."""
        layer = get_layer_for_element_type("door-swing")
        assert layer == "A-DOOR"


class TestTextLayerMapping:
    """Tests for text layer mapping."""

    def test_get_layer_for_room_name(self):
        """Test layer for room name text."""
        layer = get_layer_for_text("room_name")
        assert layer == "A-AREA-IDEN"

    def test_get_layer_for_dimension(self):
        """Test layer for dimension text."""
        layer = get_layer_for_text("dimension")
        assert layer == "G-ANNO-DIMS"

    def test_get_layer_for_note(self):
        """Test layer for note text."""
        layer = get_layer_for_text("note")
        assert layer == "G-ANNO-NOTE"

    def test_get_layer_for_equipment_tag(self):
        """Test layer for equipment tag."""
        layer = get_layer_for_text("equipment_tag")
        assert layer == "G-ANNO-TAGS"

    def test_get_layer_unknown_text(self):
        """Test unknown text type returns default."""
        layer = get_layer_for_text("unknown")
        assert layer == "G-ANNO-TEXT"


class TestSymbolLayerMapping:
    """Tests for symbol layer mapping."""

    def test_get_layer_for_diffuser_symbol(self):
        """Test layer for diffuser symbol."""
        layer = get_layer_for_symbol("diffuser")
        assert layer == "M-DIFF"

    def test_get_layer_for_outlet_symbol(self):
        """Test layer for outlet symbol."""
        layer = get_layer_for_symbol("outlet")
        assert layer == "E-POWR-OUTL"

    def test_get_layer_for_detector_symbol(self):
        """Test layer for detector symbol."""
        layer = get_layer_for_symbol("detector")
        assert layer == "F-ALRM-DETC"

    def test_get_layer_unknown_symbol(self):
        """Test unknown symbol returns default."""
        layer = get_layer_for_symbol("custom_symbol")
        assert layer == "0"


class TestBlockNameMapping:
    """Tests for block name mapping."""

    def test_get_block_for_supply_diffuser(self):
        """Test block for supply square diffuser."""
        block = get_block_name("diffuser", "supply_square")
        assert block == "M-DIFF-SQ-S"

    def test_get_block_for_duplex_outlet(self):
        """Test block for duplex outlet."""
        block = get_block_name("outlet", "duplex")
        assert block == "E-OUTL-DUP"

    def test_get_block_for_gate_valve(self):
        """Test block for gate valve."""
        block = get_block_name("valve", "gate")
        assert block == "P-VALV-GATE"

    def test_get_block_for_smoke_detector(self):
        """Test block for smoke detector."""
        block = get_block_name("detector", "smoke")
        assert block == "F-DETC-SMOK"

    def test_get_block_without_subtype(self):
        """Test block without subtype."""
        block = get_block_name("diffuser")
        assert block == "M-DIFF-GEN"

    def test_get_block_unknown_generates_name(self):
        """Test unknown symbol generates default name."""
        block = get_block_name("custom_widget", "special")
        assert block == "CUSTOM_WIDGET-SPECIAL"

    def test_get_block_case_handling(self):
        """Test case handling in block lookup."""
        block = get_block_name("DIFFUSER", "SUPPLY_SQUARE")
        assert block == "M-DIFF-SQ-S"


class TestVToolMapping:
    """Tests for VTool mapping."""

    def test_get_vtool_for_polyline(self):
        """Test VTool for polyline."""
        tool = get_vtool_for_path_type("polyline")
        assert tool == "VFPLINE"

    def test_get_vtool_for_arc(self):
        """Test VTool for arc."""
        tool = get_vtool_for_path_type("arc")
        assert tool == "VARC"

    def test_get_vtool_for_circle(self):
        """Test VTool for circle."""
        tool = get_vtool_for_path_type("circle")
        assert tool == "VCIRCLE"

    def test_get_vtool_for_contour(self):
        """Test VTool for contour."""
        tool = get_vtool_for_path_type("contour")
        assert tool == "VFCONTOUR"

    def test_get_vtool_unknown_defaults(self):
        """Test unknown path type defaults to VFPLINE."""
        tool = get_vtool_for_path_type("unknown")
        assert tool == "VFPLINE"


class TestDirectExtraction:
    """Tests for direct extraction functions."""

    @pytest.fixture
    def calibration(self):
        """Create a test calibration."""
        return ScaleCalibration(
            scale_factor=0.1,  # 1 pixel = 0.1 DWG units
            units="inches",
            confidence=0.9,
            method="test",
            image_width=1000,
            image_height=1000,
        )

    @pytest.fixture
    def elements(self):
        """Create test drawing elements."""
        return DrawingElements(
            lines=[
                DetectedLine(
                    start=(100, 200),
                    end=(300, 200),
                    line_type="wall",
                    linetype="continuous",
                ),
                DetectedLine(
                    start=(100, 400),
                    end=(300, 400),
                    line_type="duct",
                    linetype="dashed",
                ),
            ],
            arcs=[
                DetectedArc(
                    center=(150, 300),
                    radius=50,
                    start_angle=0,
                    end_angle=90,
                    arc_type="door_swing",
                ),
            ],
            circles=[
                DetectedCircle(
                    center=(500, 500),
                    radius=25,
                    circle_type="column",
                ),
            ],
            text=[
                DetectedText(
                    content="OFFICE 101",
                    position=(200, 300),
                    height_px=20,
                    text_type="room_name",
                ),
            ],
            symbols=[
                DetectedSymbol(
                    symbol_type="diffuser",
                    subtype="supply_square",
                    position=(400, 400),
                    rotation=45,
                    tag="D-1",
                ),
            ],
            dimensions=[
                DetectedDimension(
                    value="20'-0\"",
                    numeric_value=240.0,
                    unit="inches",
                    start=(100, 200),
                    end=(300, 200),
                    text_position=(200, 180),
                ),
            ],
        )

    def test_extract_lines(self, elements, calibration):
        """Test extracting lines."""
        entities = extract_lines_direct(elements, calibration)

        assert len(entities) == 2
        assert entities[0].entity_type == EntityType.LINE
        assert entities[0].layer == "A-WALL"
        assert entities[0].properties["linetype"] == "Continuous"

        # Check coordinate conversion
        start = entities[0].properties["start"]
        end = entities[0].properties["end"]
        # x: 100 * 0.1 = 10, y: (1000 - 200) * 0.1 = 80
        assert abs(start[0] - 10.0) < 0.01
        assert abs(start[1] - 80.0) < 0.01

    def test_extract_arcs(self, elements, calibration):
        """Test extracting arcs."""
        entities = extract_arcs_direct(elements, calibration)

        assert len(entities) == 1
        assert entities[0].entity_type == EntityType.ARC
        assert entities[0].layer == "A-DOOR"
        assert entities[0].properties["radius"] == 5.0  # 50 * 0.1

    def test_extract_circles(self, elements, calibration):
        """Test extracting circles."""
        entities = extract_circles_direct(elements, calibration)

        assert len(entities) == 1
        assert entities[0].entity_type == EntityType.CIRCLE
        assert entities[0].layer == "A-COLS"
        assert entities[0].properties["radius"] == 2.5  # 25 * 0.1

    def test_extract_text(self, elements, calibration):
        """Test extracting text."""
        entities = extract_text_direct(elements, calibration)

        assert len(entities) == 1
        assert entities[0].entity_type == EntityType.MTEXT
        assert entities[0].layer == "A-AREA-IDEN"
        assert entities[0].properties["content"] == "OFFICE 101"
        assert entities[0].properties["height"] == 2.0  # 20 * 0.1

    def test_extract_symbols(self, elements, calibration):
        """Test extracting symbols as blocks."""
        entities = extract_symbols_direct(elements, calibration)

        assert len(entities) == 1
        assert entities[0].entity_type == EntityType.BLOCK
        assert entities[0].layer == "M-DIFF"
        assert entities[0].properties["block_name"] == "M-DIFF-SQ-S"
        assert entities[0].properties["rotation"] == 45
        assert entities[0].properties["attributes"]["TAG"] == "D-1"

    def test_extract_dimensions(self, elements, calibration):
        """Test extracting dimensions."""
        entities = extract_dimensions_direct(elements, calibration)

        assert len(entities) == 1
        assert entities[0].entity_type == EntityType.DIMENSION
        assert entities[0].layer == "G-ANNO-DIMS"
        assert entities[0].properties["text"] == "20'-0\""


class TestExtractAll:
    """Tests for the extract_all coordinator function."""

    @pytest.fixture
    def mock_analysis(self):
        """Create a mock DrawingAnalysis."""
        return DrawingAnalysis(
            drawing_type="floor_plan",
            scale="1/4\" = 1'-0\"",
            elements=DrawingElements(
                lines=[
                    DetectedLine(start=(100, 100), end=(200, 100), line_type="wall"),
                ],
                text=[
                    DetectedText(content="TEST", position=(150, 150), text_type="note"),
                ],
            ),
            extraction_strategy=ExtractionStrategyConfig(
                primary_strategy="direct",
                special_regions=[],
            ),
        )

    @pytest.fixture
    def mock_calibration(self):
        """Create a mock ScaleCalibration."""
        return ScaleCalibration(
            scale_factor=0.16,
            units="inches",
            confidence=0.85,
            method="scale_notation",
            image_width=3600,
            image_height=2400,
        )

    @pytest.mark.asyncio
    async def test_extract_all_direct_strategy(self, mock_analysis, mock_calibration):
        """Test extract_all with direct strategy."""
        result = await extract_all(mock_analysis, mock_calibration)

        assert isinstance(result, ExtractionResult)
        assert result.total_entities == 2  # 1 line + 1 text
        assert result.direct_count == 2
        assert result.primary_strategy == "direct"

    @pytest.mark.asyncio
    async def test_extract_direct_only(self, mock_analysis, mock_calibration):
        """Test extract_direct_only function."""
        result = await extract_direct_only(mock_analysis, mock_calibration)

        assert isinstance(result, ExtractionResult)
        assert result.direct_count == 2
        assert result.opencv_count == 0
        assert result.guided_count == 0


class TestHelperFunctions:
    """Tests for helper functions."""

    def test_get_entities_by_type(self):
        """Test grouping entities by type."""
        entities = [
            EntityToCreate(EntityType.LINE, "A-WALL", {}),
            EntityToCreate(EntityType.LINE, "M-DUCT", {}),
            EntityToCreate(EntityType.CIRCLE, "A-COLS", {}),
            EntityToCreate(EntityType.MTEXT, "G-ANNO-NOTE", {}),
        ]

        result = ExtractionResult(entities=entities)
        grouped = get_entities_by_type(result)

        assert len(grouped["line"]) == 2
        assert len(grouped["circle"]) == 1
        assert len(grouped["mtext"]) == 1

    def test_get_entities_by_layer(self):
        """Test grouping entities by layer."""
        entities = [
            EntityToCreate(EntityType.LINE, "A-WALL", {}),
            EntityToCreate(EntityType.LINE, "A-WALL", {}),
            EntityToCreate(EntityType.CIRCLE, "M-EQPM", {}),
        ]

        result = ExtractionResult(entities=entities)
        grouped = get_entities_by_layer(result)

        assert len(grouped["A-WALL"]) == 2
        assert len(grouped["M-EQPM"]) == 1

    def test_get_required_layers(self):
        """Test getting required layers."""
        entities = [
            EntityToCreate(EntityType.LINE, "A-WALL", {}),
            EntityToCreate(EntityType.LINE, "A-DOOR", {}),
            EntityToCreate(EntityType.LINE, "A-WALL", {}),  # Duplicate
            EntityToCreate(EntityType.CIRCLE, "M-EQPM", {}),
        ]

        result = ExtractionResult(entities=entities)
        layers = get_required_layers(result)

        assert len(layers) == 3
        assert "A-WALL" in layers
        assert "A-DOOR" in layers
        assert "M-EQPM" in layers
        # Should be sorted
        assert layers == sorted(layers)

    def test_get_required_blocks(self):
        """Test getting required blocks."""
        entities = [
            EntityToCreate(EntityType.LINE, "A-WALL", {}),  # Not a block
            EntityToCreate(EntityType.BLOCK, "M-DIFF", {"block_name": "M-DIFF-SQ-S"}),
            EntityToCreate(EntityType.BLOCK, "E-OUTL", {"block_name": "E-OUTL-DUP"}),
            EntityToCreate(EntityType.BLOCK, "M-DIFF", {"block_name": "M-DIFF-SQ-S"}),  # Dup
        ]

        result = ExtractionResult(entities=entities)
        blocks = get_required_blocks(result)

        assert len(blocks) == 2
        assert "M-DIFF-SQ-S" in blocks
        assert "E-OUTL-DUP" in blocks


class TestConstants:
    """Tests for constants."""

    def test_element_type_to_layer_has_key_types(self):
        """Test that ELEMENT_TYPE_TO_LAYER has key element types."""
        assert "wall" in ELEMENT_TYPE_TO_LAYER
        assert "duct" in ELEMENT_TYPE_TO_LAYER
        assert "pipe" in ELEMENT_TYPE_TO_LAYER
        assert "outlet" in ELEMENT_TYPE_TO_LAYER
        assert "diffuser" in ELEMENT_TYPE_TO_LAYER

    def test_text_type_to_layer_has_key_types(self):
        """Test that TEXT_TYPE_TO_LAYER has key text types."""
        assert "room_name" in TEXT_TYPE_TO_LAYER
        assert "dimension" in TEXT_TYPE_TO_LAYER
        assert "note" in TEXT_TYPE_TO_LAYER
        assert "label" in TEXT_TYPE_TO_LAYER

    def test_symbol_to_block_has_key_symbols(self):
        """Test that SYMBOL_TO_BLOCK has key symbols."""
        assert ("diffuser", "supply_square") in SYMBOL_TO_BLOCK
        assert ("outlet", "duplex") in SYMBOL_TO_BLOCK
        assert ("valve", "gate") in SYMBOL_TO_BLOCK
        assert ("detector", "smoke") in SYMBOL_TO_BLOCK

    def test_vtool_mapping_has_key_tools(self):
        """Test that VTOOL_MAPPING has key tools."""
        assert "polyline" in VTOOL_MAPPING
        assert "arc" in VTOOL_MAPPING
        assert "circle" in VTOOL_MAPPING
        assert "line" in VTOOL_MAPPING
        assert "contour" in VTOOL_MAPPING


class TestGuidedRasterization:
    """Tests for guided rasterization."""

    @pytest.fixture
    def mock_analysis_with_regions(self):
        """Create analysis with special regions."""
        return DrawingAnalysis(
            drawing_type="mechanical",
            elements=DrawingElements(),
            extraction_strategy=ExtractionStrategyConfig(
                primary_strategy="hybrid",
                special_regions=[
                    SpecialRegion(
                        bounds=(100, 100, 500, 500),
                        strategy="guided_rasterization",
                        reason="Complex ductwork",
                        expected_pattern="polyline",
                    ),
                ],
            ),
        )

    @pytest.fixture
    def mock_calibration(self):
        """Create a mock calibration."""
        return ScaleCalibration(
            scale_factor=0.1,
            units="inches",
            confidence=0.9,
            method="test",
            image_width=1000,
            image_height=1000,
        )

    @pytest.mark.asyncio
    async def test_guided_rasterization_generates_commands(
        self, mock_analysis_with_regions, mock_calibration
    ):
        """Test that guided rasterization generates commands."""
        commands = await guided_rasterization(
            mock_analysis_with_regions,
            mock_calibration,
            Path("/fake/image.png"),
        )

        assert len(commands) == 1
        assert commands[0].tool == "VFPLINE"
        # Center of region (300, 300) converted to DWG
        # x: 300 * 0.1 = 30, y: (1000 - 300) * 0.1 = 70


class TestSelectiveOpencv:
    """Tests for selective OpenCV extraction."""

    @pytest.fixture
    def mock_analysis_with_opencv_regions(self):
        """Create analysis with OpenCV regions."""
        return DrawingAnalysis(
            drawing_type="mechanical",
            elements=DrawingElements(),
            extraction_strategy=ExtractionStrategyConfig(
                primary_strategy="hybrid",
                special_regions=[
                    SpecialRegion(
                        bounds=(100, 100, 500, 500),
                        strategy="selective_opencv",
                        reason="Hatching pattern",
                        expected_pattern="parallel_lines",
                    ),
                ],
            ),
        )

    @pytest.fixture
    def mock_calibration(self):
        """Create a mock calibration."""
        return ScaleCalibration(
            scale_factor=0.1,
            units="inches",
            confidence=0.9,
            method="test",
            image_width=1000,
            image_height=1000,
        )

    @pytest.mark.asyncio
    async def test_selective_opencv_no_regions(self, mock_calibration):
        """Test selective_opencv with no regions returns empty."""
        analysis = DrawingAnalysis(
            drawing_type="floor_plan",
            elements=DrawingElements(),
            extraction_strategy=ExtractionStrategyConfig(
                primary_strategy="direct",
                special_regions=[],
            ),
        )

        entities = await selective_opencv(
            analysis,
            mock_calibration,
            Path("/fake/image.png"),
        )

        assert entities == []

    @pytest.mark.asyncio
    async def test_selective_opencv_missing_image(
        self, mock_analysis_with_opencv_regions, mock_calibration
    ):
        """Test selective_opencv handles missing image gracefully."""
        entities = await selective_opencv(
            mock_analysis_with_opencv_regions,
            mock_calibration,
            Path("/nonexistent/image.png"),
        )

        # Should return empty list, not raise
        assert entities == []
