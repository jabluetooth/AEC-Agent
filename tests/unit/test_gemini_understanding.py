"""
Unit tests for Phase 2: Gemini Understanding.

Tests the gemini_first.gemini_understanding module which analyzes
drawings using Gemini Vision AI.
"""

import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# Import module under test
from aec_agent.mcp.tools.gemini_first.gemini_understanding import (
    CalibrationHint,
    DetectedArc,
    DetectedCircle,
    DetectedDimension,
    DetectedLine,
    DetectedSymbol,
    DetectedText,
    DrawingAnalysis,
    DrawingAnalyzer,
    DrawingElements,
    DrawingType,
    ExtractionStrategy,
    ExtractionStrategyConfig,
    RegionBounds,
    SpecialRegion,
    analyze_drawing,
)


class TestRegionBounds:
    """Tests for RegionBounds dataclass."""

    def test_basic_creation(self):
        """Test creating a RegionBounds."""
        bounds = RegionBounds(x1=100, y1=200, x2=500, y2=600, content="Title Block")

        assert bounds.x1 == 100
        assert bounds.y1 == 200
        assert bounds.x2 == 500
        assert bounds.y2 == 600
        assert bounds.content == "Title Block"

    def test_to_dict(self):
        """Test serialization to dictionary."""
        bounds = RegionBounds(x1=0, y1=0, x2=100, y2=100)
        d = bounds.to_dict()

        assert d["bounds"] == [0, 0, 100, 100]
        assert d["content"] is None

    def test_from_dict(self):
        """Test deserialization from dictionary."""
        data = {"bounds": [10, 20, 30, 40], "content": "Test"}
        bounds = RegionBounds.from_dict(data)

        assert bounds.x1 == 10
        assert bounds.y1 == 20
        assert bounds.x2 == 30
        assert bounds.y2 == 40
        assert bounds.content == "Test"

    def test_from_dict_missing_bounds(self):
        """Test from_dict with missing bounds defaults to zeros."""
        bounds = RegionBounds.from_dict({})
        assert bounds.x1 == 0
        assert bounds.y1 == 0


class TestDetectedLine:
    """Tests for DetectedLine dataclass."""

    def test_basic_creation(self):
        """Test creating a DetectedLine."""
        line = DetectedLine(
            start=(100, 200),
            end=(300, 400),
            line_type="wall",
            linetype="continuous",
            layer_suggestion="A-WALL",
        )

        assert line.start == (100, 200)
        assert line.end == (300, 400)
        assert line.line_type == "wall"
        assert line.linetype == "continuous"
        assert line.layer_suggestion == "A-WALL"

    def test_to_dict(self):
        """Test serialization to dictionary."""
        line = DetectedLine(start=(0, 0), end=(100, 100), line_type="duct")
        d = line.to_dict()

        assert d["start"] == [0, 0]
        assert d["end"] == [100, 100]
        assert d["type"] == "duct"


class TestDetectedSymbol:
    """Tests for DetectedSymbol dataclass."""

    def test_basic_creation(self):
        """Test creating a DetectedSymbol."""
        symbol = DetectedSymbol(
            symbol_type="diffuser",
            subtype="supply_square",
            position=(500, 300),
            rotation=45.0,
            size="24x24",
            tag="SD-1",
            associated_text=["SUPPLY AIR", "400 CFM"],
        )

        assert symbol.symbol_type == "diffuser"
        assert symbol.subtype == "supply_square"
        assert symbol.position == (500, 300)
        assert symbol.rotation == 45.0
        assert symbol.size == "24x24"
        assert symbol.tag == "SD-1"
        assert len(symbol.associated_text) == 2

    def test_to_dict(self):
        """Test serialization to dictionary."""
        symbol = DetectedSymbol(symbol_type="outlet", position=(100, 100))
        d = symbol.to_dict()

        assert d["type"] == "outlet"
        assert d["position"] == [100, 100]
        assert d["rotation"] == 0
        assert d["associated_text"] == []


class TestDetectedText:
    """Tests for DetectedText dataclass."""

    def test_basic_creation(self):
        """Test creating DetectedText."""
        text = DetectedText(
            content="MECHANICAL ROOM 101",
            position=(200, 300),
            height_px=24,
            text_type="room_name",
            associated_with="room_boundary",
        )

        assert text.content == "MECHANICAL ROOM 101"
        assert text.position == (200, 300)
        assert text.height_px == 24
        assert text.text_type == "room_name"

    def test_default_values(self):
        """Test default values."""
        text = DetectedText(content="Test", position=(0, 0))

        assert text.height_px == 12
        assert text.text_type == "note"
        assert text.associated_with is None


class TestCalibrationHint:
    """Tests for CalibrationHint dataclass."""

    def test_dimension_hint(self):
        """Test creating a dimension calibration hint."""
        hint = CalibrationHint(
            hint_type="dimension",
            description="10'-0\" dimension line",
            pixel_measurement=240.0,
            real_measurement="10'-0\"",
            confidence=0.95,
        )

        assert hint.hint_type == "dimension"
        assert hint.pixel_measurement == 240.0
        assert hint.real_measurement == "10'-0\""
        assert hint.confidence == 0.95

    def test_to_dict(self):
        """Test serialization."""
        hint = CalibrationHint(
            hint_type="sheet_border",
            description="ARCH D border",
            pixel_measurement=3600.0,
            real_measurement="36\"",
        )
        d = hint.to_dict()

        assert d["type"] == "sheet_border"
        assert d["confidence"] == 0.8  # Default


class TestDrawingElements:
    """Tests for DrawingElements dataclass."""

    def test_empty_elements(self):
        """Test empty drawing elements."""
        elements = DrawingElements()

        assert elements.total_count == 0
        assert len(elements.lines) == 0
        assert len(elements.symbols) == 0

    def test_total_count(self):
        """Test total_count property."""
        elements = DrawingElements(
            lines=[DetectedLine(start=(0, 0), end=(100, 100))],
            circles=[DetectedCircle(center=(50, 50), radius=10)],
            text=[DetectedText(content="A", position=(0, 0))],
            symbols=[
                DetectedSymbol(symbol_type="valve", position=(10, 10)),
                DetectedSymbol(symbol_type="outlet", position=(20, 20)),
            ],
        )

        assert elements.total_count == 5

    def test_to_dict(self):
        """Test serialization with counts."""
        elements = DrawingElements(
            lines=[DetectedLine(start=(0, 0), end=(100, 100))] * 3,
        )
        d = elements.to_dict()

        assert d["lines"]["count"] == 3
        assert len(d["lines"]["items"]) == 3


class TestExtractionStrategyConfig:
    """Tests for ExtractionStrategyConfig dataclass."""

    def test_default_strategy(self):
        """Test default extraction strategy."""
        config = ExtractionStrategyConfig()

        assert config.primary_strategy == "direct"
        assert config.rationale == ""
        assert len(config.special_regions) == 0

    def test_hybrid_strategy(self):
        """Test hybrid extraction strategy."""
        config = ExtractionStrategyConfig(
            primary_strategy="hybrid",
            rationale="Complex ductwork requires guided rasterization",
            per_element_strategy={
                "walls": "direct",
                "ductwork": "guided",
                "text": "direct",
            },
            special_regions=[
                SpecialRegion(
                    bounds=(100, 100, 500, 500),
                    strategy="guided_rasterization",
                    reason="Dense ductwork area",
                    expected_pattern="parallel_lines",
                )
            ],
        )

        assert config.primary_strategy == "hybrid"
        assert config.per_element_strategy["ductwork"] == "guided"
        assert len(config.special_regions) == 1


class TestDrawingAnalysis:
    """Tests for DrawingAnalysis dataclass."""

    def test_basic_creation(self):
        """Test creating a DrawingAnalysis."""
        analysis = DrawingAnalysis(
            drawing_type="mechanical",
            scale="1/4\" = 1'-0\"",
            sheet_size="ARCH D (24x36)",
            units="imperial",
            complexity="complex",
            description="HVAC floor plan showing ductwork layout",
        )

        assert analysis.drawing_type == "mechanical"
        assert analysis.scale == "1/4\" = 1'-0\""
        assert analysis.units == "imperial"
        assert analysis.complexity == "complex"

    def test_total_elements_property(self):
        """Test total_elements property."""
        analysis = DrawingAnalysis(
            drawing_type="electrical",
            elements=DrawingElements(
                symbols=[DetectedSymbol(symbol_type="outlet", position=(0, 0))] * 10,
                text=[DetectedText(content="A", position=(0, 0))] * 5,
            ),
        )

        assert analysis.total_elements == 15

    def test_to_dict(self):
        """Test full serialization."""
        analysis = DrawingAnalysis(
            drawing_type="floor_plan",
            title_block=RegionBounds(x1=2000, y1=0, x2=2400, y2=400),
            elements=DrawingElements(
                lines=[DetectedLine(start=(0, 0), end=(100, 100))],
            ),
            calibration_hints=[
                CalibrationHint(
                    hint_type="dimension",
                    description="10' dim",
                    pixel_measurement=240,
                    real_measurement="10'-0\"",
                )
            ],
        )

        d = analysis.to_dict()

        assert d["drawing_analysis"]["type"] == "floor_plan"
        assert d["regions"]["title_block"] is not None
        assert d["elements"]["lines"]["count"] == 1
        assert len(d["calibration_hints"]) == 1
        assert "metadata" in d


class TestDrawingTypeEnum:
    """Tests for DrawingType enum."""

    def test_all_drawing_types(self):
        """Test all expected drawing types exist."""
        expected = [
            "floor_plan", "electrical", "mechanical", "plumbing",
            "fire_alarm", "reflected_ceiling", "site_plan",
            "detail", "section", "elevation", "schedule", "diagram", "other",
        ]

        for dtype in expected:
            assert hasattr(DrawingType, dtype.upper())


class TestExtractionStrategyEnum:
    """Tests for ExtractionStrategy enum."""

    def test_all_strategies(self):
        """Test all expected strategies exist."""
        assert ExtractionStrategy.DIRECT.value == "direct"
        assert ExtractionStrategy.GUIDED_RASTERIZATION.value == "guided_rasterization"
        assert ExtractionStrategy.SELECTIVE_OPENCV.value == "selective_opencv"
        assert ExtractionStrategy.HYBRID.value == "hybrid"
        assert ExtractionStrategy.SKIP.value == "skip"


class TestDrawingAnalyzer:
    """Tests for DrawingAnalyzer class."""

    def test_init_defaults(self):
        """Test default initialization."""
        analyzer = DrawingAnalyzer()

        assert analyzer.model_name == "gemini-1.5-pro"
        assert analyzer.temperature == 0.1
        assert analyzer.max_output_tokens == 8192

    def test_init_custom(self):
        """Test custom initialization."""
        analyzer = DrawingAnalyzer(
            model="gemini-1.5-flash",
            temperature=0.2,
            max_output_tokens=4096,
        )

        assert analyzer.model_name == "gemini-1.5-flash"
        assert analyzer.temperature == 0.2
        assert analyzer.max_output_tokens == 4096

    def test_parse_response_direct_json(self):
        """Test parsing direct JSON response."""
        analyzer = DrawingAnalyzer()

        json_str = json.dumps({
            "drawing_analysis": {"type": "floor_plan", "complexity": "simple"},
            "elements": {},
        })

        result = analyzer._parse_response(json_str)
        assert result["drawing_analysis"]["type"] == "floor_plan"

    def test_parse_response_markdown_json(self):
        """Test parsing JSON in markdown code block."""
        analyzer = DrawingAnalyzer()

        response = """Here is the analysis:

```json
{
    "drawing_analysis": {"type": "electrical"},
    "elements": {}
}
```

Let me explain..."""

        result = analyzer._parse_response(response)
        assert result["drawing_analysis"]["type"] == "electrical"

    def test_parse_response_embedded_json(self):
        """Test parsing embedded JSON object."""
        analyzer = DrawingAnalyzer()

        response = 'The analysis shows {"drawing_analysis": {"type": "mechanical"}} as expected.'

        result = analyzer._parse_response(response)
        assert result["drawing_analysis"]["type"] == "mechanical"

    def test_parse_response_invalid(self):
        """Test parsing invalid response raises error."""
        analyzer = DrawingAnalyzer()

        with pytest.raises(ValueError, match="Failed to parse"):
            analyzer._parse_response("This is not JSON at all")

    def test_parse_analysis_full(self):
        """Test parsing complete analysis JSON."""
        analyzer = DrawingAnalyzer()

        data = {
            "drawing_analysis": {
                "type": "mechanical",
                "scale": "1/4\" = 1'-0\"",
                "sheet_size": "ARCH D",
                "units": "imperial",
                "complexity": "complex",
                "description": "HVAC plan",
            },
            "regions": {
                "title_block": {"bounds": [2000, 0, 2400, 400], "content": "Sheet 1"},
                "drawing_area": {"bounds": [0, 0, 2000, 1400]},
            },
            "elements": {
                "lines": [
                    {"start": [0, 0], "end": [100, 100], "type": "duct", "linetype": "continuous"}
                ],
                "arcs": [
                    {"center": [50, 50], "radius": 20, "start_angle": 0, "end_angle": 90, "type": "door_swing"}
                ],
                "circles": [
                    {"center": [100, 100], "radius": 10, "type": "column"}
                ],
                "text": [
                    {"content": "MECH RM", "position": [200, 300], "height_px": 18, "type": "room_name"}
                ],
                "symbols": [
                    {"type": "diffuser", "subtype": "supply_square", "position": [400, 300], "rotation": 0}
                ],
                "dimensions": [
                    {"value": "10'-0\"", "numeric_value": 120, "unit": "inches", "start": [0, 0], "end": [240, 0]}
                ],
            },
            "calibration_hints": [
                {"type": "dimension", "description": "10' dim", "pixel_measurement": 240, "real_measurement": "10'-0\""}
            ],
            "extraction_strategy": {
                "primary_strategy": "hybrid",
                "rationale": "Complex ductwork",
                "per_element_strategy": {"text": "direct"},
                "special_regions": [
                    {"bounds": [100, 100, 500, 500], "strategy": "guided_rasterization", "reason": "Dense area"}
                ],
            },
        }

        analysis = analyzer._parse_analysis(data)

        assert analysis.drawing_type == "mechanical"
        assert analysis.scale == "1/4\" = 1'-0\""
        assert analysis.complexity == "complex"
        assert analysis.title_block is not None
        assert analysis.title_block.x1 == 2000
        assert len(analysis.elements.lines) == 1
        assert len(analysis.elements.arcs) == 1
        assert len(analysis.elements.circles) == 1
        assert len(analysis.elements.text) == 1
        assert len(analysis.elements.symbols) == 1
        assert len(analysis.elements.dimensions) == 1
        assert len(analysis.calibration_hints) == 1
        assert analysis.extraction_strategy.primary_strategy == "hybrid"
        assert len(analysis.extraction_strategy.special_regions) == 1

    def test_parse_analysis_minimal(self):
        """Test parsing minimal analysis JSON."""
        analyzer = DrawingAnalyzer()

        data = {
            "drawing_analysis": {"type": "other"},
        }

        analysis = analyzer._parse_analysis(data)

        assert analysis.drawing_type == "other"
        assert analysis.elements.total_count == 0
        assert analysis.title_block is None


class TestAnalyzeDrawingMocked:
    """Integration tests with mocked Gemini API."""

    @pytest.fixture
    def sample_response(self):
        """Sample Gemini API response."""
        return {
            "drawing_analysis": {
                "type": "floor_plan",
                "scale": None,
                "sheet_size": "ARCH D",
                "units": "imperial",
                "complexity": "medium",
                "description": "Architectural floor plan",
            },
            "regions": {
                "drawing_area": {"bounds": [0, 0, 2400, 1800]},
            },
            "elements": {
                "lines": [
                    {"start": [100, 100], "end": [500, 100], "type": "wall", "linetype": "continuous"}
                ],
                "arcs": [],
                "circles": [],
                "text": [
                    {"content": "OFFICE", "position": [300, 200], "type": "room_name"}
                ],
                "symbols": [],
                "dimensions": [],
            },
            "calibration_hints": [],
            "extraction_strategy": {
                "primary_strategy": "direct",
                "rationale": "Simple floor plan with clear geometry",
            },
        }

    @pytest.mark.asyncio
    async def test_analyze_with_mock(self, tmp_path, sample_response):
        """Test analyze method with mocked Gemini API."""
        # Create a dummy image
        from PIL import Image
        img = Image.new("RGB", (100, 100), color="white")
        img_path = tmp_path / "test.png"
        img.save(img_path)

        # Mock the Gemini model
        mock_model = MagicMock()
        mock_response = MagicMock()
        mock_response.text = json.dumps(sample_response)
        mock_model.generate_content_async = AsyncMock(return_value=mock_response)

        analyzer = DrawingAnalyzer()

        with patch.object(analyzer, "_get_gemini_model", return_value=mock_model):
            analysis = await analyzer.analyze(img_path)

        assert analysis.drawing_type == "floor_plan"
        assert analysis.complexity == "medium"
        assert len(analysis.elements.lines) == 1
        assert len(analysis.elements.text) == 1
        assert analysis.extraction_strategy.primary_strategy == "direct"

    @pytest.mark.asyncio
    async def test_analyze_file_not_found(self):
        """Test analyze with non-existent file."""
        analyzer = DrawingAnalyzer()

        with pytest.raises(FileNotFoundError):
            await analyzer.analyze(Path("nonexistent.png"))

    @pytest.mark.asyncio
    async def test_analyze_drawing_convenience(self, tmp_path, sample_response):
        """Test analyze_drawing convenience function."""
        # Create a dummy image
        pytest.importorskip("PIL", reason="Pillow not installed")
        from PIL import Image
        img = Image.new("RGB", (100, 100), color="white")
        img_path = tmp_path / "test.png"
        img.save(img_path)

        # Mock the entire DrawingAnalyzer class
        mock_analysis = DrawingAnalysis(
            drawing_type="electrical",
            complexity="simple",
        )

        with patch(
            "aec_agent.mcp.tools.gemini_first.gemini_understanding.DrawingAnalyzer"
        ) as MockAnalyzer:
            mock_instance = AsyncMock()
            mock_instance.analyze = AsyncMock(return_value=mock_analysis)
            MockAnalyzer.return_value = mock_instance

            analysis = await analyze_drawing(img_path)

        assert analysis.drawing_type == "electrical"


class TestSpecialRegion:
    """Tests for SpecialRegion dataclass."""

    def test_basic_creation(self):
        """Test creating a SpecialRegion."""
        region = SpecialRegion(
            bounds=(100, 100, 500, 500),
            strategy="guided_rasterization",
            reason="Complex curved ductwork",
            expected_pattern="curves",
        )

        assert region.bounds == (100, 100, 500, 500)
        assert region.strategy == "guided_rasterization"
        assert region.expected_pattern == "curves"

    def test_to_dict(self):
        """Test serialization."""
        region = SpecialRegion(
            bounds=(0, 0, 100, 100),
            strategy="selective_opencv",
            reason="Hatching pattern",
        )
        d = region.to_dict()

        assert d["bounds"] == [0, 0, 100, 100]
        assert d["strategy"] == "selective_opencv"
        assert d["expected_pattern"] is None


class TestDetectedArc:
    """Tests for DetectedArc dataclass."""

    def test_door_swing(self):
        """Test creating a door swing arc."""
        arc = DetectedArc(
            center=(200, 300),
            radius=36,
            start_angle=0,
            end_angle=90,
            arc_type="door_swing",
        )

        assert arc.center == (200, 300)
        assert arc.radius == 36
        assert arc.start_angle == 0
        assert arc.end_angle == 90
        assert arc.arc_type == "door_swing"


class TestDetectedCircle:
    """Tests for DetectedCircle dataclass."""

    def test_column(self):
        """Test creating a column circle."""
        circle = DetectedCircle(
            center=(150, 150),
            radius=12,
            circle_type="column",
        )

        assert circle.center == (150, 150)
        assert circle.radius == 12
        assert circle.circle_type == "column"


class TestDetectedDimension:
    """Tests for DetectedDimension dataclass."""

    def test_feet_inches(self):
        """Test feet and inches dimension."""
        dim = DetectedDimension(
            value="10'-6\"",
            numeric_value=126,
            unit="inches",
            start=(100, 500),
            end=(340, 500),
            text_position=(220, 490),
        )

        assert dim.value == "10'-6\""
        assert dim.numeric_value == 126
        assert dim.unit == "inches"
