"""Unit tests for geometry_classifier.py (Phase D)."""

import pytest

from aec_agent.mcp.tools.document_classifier import DrawingType
from aec_agent.mcp.tools.geometry_classifier import (
    ClassifiedLine,
    ClassifiedPolyline,
    ClassifiedCircle,
    GeometryClassification,
    GeometryClassifier,
    GeometryClassifierResult,
    GeometrySystem,
    GeometryType,
    ParallelLinePair,
    classify_geometry,
)
from aec_agent.mcp.tools.image_vectorizer import (
    DetectedCircle,
    DetectedLine,
    DetectedPolyline,
)


class TestGeometryType:
    """Tests for GeometryType enum."""

    def test_all_types_have_string_values(self):
        """All geometry types should have string values."""
        for geo_type in GeometryType:
            assert isinstance(geo_type.value, str)
            assert len(geo_type.value) > 0

    def test_wall_type_exists(self):
        """WALL type should exist."""
        assert GeometryType.WALL.value == "wall"

    def test_duct_types_exist(self):
        """Duct types should exist."""
        assert GeometryType.DUCT.value == "duct"
        assert GeometryType.DUCT_RECTANGULAR.value == "duct_rectangular"
        assert GeometryType.DUCT_ROUND.value == "duct_round"

    def test_pipe_type_exists(self):
        """PIPE type should exist."""
        assert GeometryType.PIPE.value == "pipe"

    def test_dimension_line_type_exists(self):
        """DIMENSION_LINE type should exist."""
        assert GeometryType.DIMENSION_LINE.value == "dimension"


class TestGeometrySystem:
    """Tests for GeometrySystem enum."""

    def test_hvac_systems(self):
        """HVAC systems should exist."""
        assert GeometrySystem.SUPPLY_AIR.value == "supply_air"
        assert GeometrySystem.RETURN_AIR.value == "return_air"
        assert GeometrySystem.EXHAUST_AIR.value == "exhaust_air"

    def test_plumbing_systems(self):
        """Plumbing systems should exist."""
        assert GeometrySystem.DOMESTIC_COLD.value == "domestic_cold"
        assert GeometrySystem.DOMESTIC_HOT.value == "domestic_hot"
        assert GeometrySystem.SANITARY.value == "sanitary"

    def test_electrical_systems(self):
        """Electrical systems should exist."""
        assert GeometrySystem.POWER.value == "power"
        assert GeometrySystem.LIGHTING.value == "lighting"


class TestGeometryClassification:
    """Tests for GeometryClassification dataclass."""

    def test_default_values(self):
        """Default values should be set correctly."""
        classification = GeometryClassification(GeometryType.UNKNOWN)
        assert classification.geometry_type == GeometryType.UNKNOWN
        assert classification.system == GeometrySystem.UNKNOWN
        assert classification.confidence == 0.0
        assert classification.size is None
        assert classification.layer_suggestion is None

    def test_custom_values(self):
        """Custom values should be preserved."""
        classification = GeometryClassification(
            geometry_type=GeometryType.PIPE,
            system=GeometrySystem.DOMESTIC_COLD,
            confidence=0.85,
            size='2"',
            layer_suggestion="P-PIPE-DCW",
            reasoning="Line connects valves",
        )
        assert classification.geometry_type == GeometryType.PIPE
        assert classification.system == GeometrySystem.DOMESTIC_COLD
        assert classification.confidence == 0.85
        assert classification.size == '2"'


class TestClassifiedLine:
    """Tests for ClassifiedLine dataclass."""

    def test_inherits_from_detected_line(self):
        """ClassifiedLine should have DetectedLine attributes."""
        line = ClassifiedLine(
            start=(0.0, 0.0),
            end=(100.0, 0.0),
            linetype="CONTINUOUS",
        )
        assert line.start == (0.0, 0.0)
        assert line.end == (100.0, 0.0)
        assert line.linetype == "CONTINUOUS"

    def test_has_classification(self):
        """ClassifiedLine should have classification attribute."""
        line = ClassifiedLine(
            start=(0.0, 0.0),
            end=(100.0, 0.0),
        )
        assert hasattr(line, 'classification')
        assert line.classification.geometry_type == GeometryType.UNKNOWN


class TestGeometryClassifier:
    """Tests for GeometryClassifier class."""

    @pytest.fixture
    def classifier(self):
        """Create a classifier instance for tests."""
        return GeometryClassifier()

    @pytest.fixture
    def horizontal_parallel_lines(self):
        """Create a pair of horizontal parallel lines (wall-like)."""
        return [
            DetectedLine(start=(0.0, 0.0), end=(100.0, 0.0), linetype="CONTINUOUS"),
            DetectedLine(start=(0.0, 6.0), end=(100.0, 6.0), linetype="CONTINUOUS"),
        ]

    @pytest.fixture
    def vertical_parallel_lines(self):
        """Create a pair of vertical parallel lines."""
        return [
            DetectedLine(start=(0.0, 0.0), end=(0.0, 100.0), linetype="CONTINUOUS"),
            DetectedLine(start=(8.0, 0.0), end=(8.0, 100.0), linetype="CONTINUOUS"),
        ]

    @pytest.fixture
    def dashed_line(self):
        """Create a dashed line."""
        return [
            DetectedLine(start=(0.0, 0.0), end=(50.0, 0.0), linetype="DASHED"),
        ]

    @pytest.fixture
    def center_line(self):
        """Create a centerline."""
        return [
            DetectedLine(start=(0.0, 0.0), end=(50.0, 0.0), linetype="CENTER"),
        ]

    def test_classifier_initialization(self, classifier):
        """Classifier should initialize with default values."""
        assert classifier.scale == 1.0
        assert classifier.wall_thickness_min == 3.0
        assert classifier.wall_thickness_max == 12.0
        assert classifier.parallel_tolerance == 3.0

    def test_custom_initialization(self):
        """Classifier should accept custom parameters."""
        classifier = GeometryClassifier(
            scale=0.5,
            wall_thickness_min=4.0,
            wall_thickness_max=10.0,
        )
        assert classifier.scale == 0.5
        assert classifier.wall_thickness_min == 4.0
        assert classifier.wall_thickness_max == 10.0

    @pytest.mark.asyncio
    async def test_classify_empty_input(self, classifier):
        """Empty input should return empty result."""
        result = await classifier.classify(
            lines=[],
            polylines=[],
            circles=[],
        )
        assert isinstance(result, GeometryClassifierResult)
        assert len(result.classified_lines) == 0
        assert len(result.parallel_pairs) == 0

    @pytest.mark.asyncio
    async def test_classify_dashed_line_as_hidden(self, classifier, dashed_line):
        """Dashed lines should be classified as HIDDEN_LINE."""
        result = await classifier.classify(lines=dashed_line)
        assert len(result.classified_lines) == 1
        assert result.classified_lines[0].classification.geometry_type == GeometryType.HIDDEN_LINE
        assert result.classified_lines[0].classification.confidence >= 0.9

    @pytest.mark.asyncio
    async def test_classify_center_line(self, classifier, center_line):
        """Center lines should be classified as CENTERLINE."""
        result = await classifier.classify(lines=center_line)
        assert len(result.classified_lines) == 1
        assert result.classified_lines[0].classification.geometry_type == GeometryType.CENTERLINE

    @pytest.mark.asyncio
    async def test_find_parallel_pairs(self, classifier, horizontal_parallel_lines):
        """Should detect parallel line pairs."""
        result = await classifier.classify(
            lines=horizontal_parallel_lines,
            drawing_type=DrawingType.FLOOR_PLAN,
        )
        assert len(result.parallel_pairs) >= 1
        # 6" spacing is within wall thickness range
        pair = result.parallel_pairs[0]
        assert 5.5 <= pair.spacing <= 6.5  # Allow some tolerance

    @pytest.mark.asyncio
    async def test_classify_walls_in_floor_plan(self, classifier, horizontal_parallel_lines):
        """Parallel lines with wall-like spacing should be walls in floor plans."""
        result = await classifier.classify(
            lines=horizontal_parallel_lines,
            drawing_type=DrawingType.FLOOR_PLAN,
        )
        # Should have wall centerlines
        assert len(result.wall_centerlines) >= 1

    @pytest.mark.asyncio
    async def test_classify_with_hvac_drawing_type(self, classifier):
        """Should detect ducts in HVAC plans."""
        # Create lines with 12" spacing (duct-like)
        duct_lines = [
            DetectedLine(start=(0.0, 0.0), end=(200.0, 0.0), linetype="CONTINUOUS"),
            DetectedLine(start=(0.0, 12.0), end=(200.0, 12.0), linetype="CONTINUOUS"),
        ]
        result = await classifier.classify(
            lines=duct_lines,
            drawing_type=DrawingType.HVAC_PLAN,
        )
        assert len(result.duct_boundaries) >= 1

    @pytest.mark.asyncio
    async def test_statistics_computed(self, classifier, horizontal_parallel_lines):
        """Result should include classification statistics."""
        result = await classifier.classify(lines=horizontal_parallel_lines)
        assert "total_elements" in result.statistics
        assert "classified_elements" in result.statistics
        assert "type_counts" in result.statistics
        assert result.statistics["total_elements"] == 2


class TestClassifyGeometryFunction:
    """Tests for the classify_geometry convenience function."""

    @pytest.mark.asyncio
    async def test_basic_call(self):
        """Should work with minimal arguments."""
        lines = [
            DetectedLine(start=(0.0, 0.0), end=(100.0, 0.0), linetype="CONTINUOUS"),
        ]
        result = await classify_geometry(lines=lines)
        assert isinstance(result, GeometryClassifierResult)
        assert len(result.classified_lines) == 1

    @pytest.mark.asyncio
    async def test_with_drawing_type(self):
        """Should accept drawing_type parameter."""
        lines = [
            DetectedLine(start=(0.0, 0.0), end=(100.0, 0.0), linetype="CONTINUOUS"),
        ]
        result = await classify_geometry(
            lines=lines,
            drawing_type=DrawingType.PLUMBING_PLAN,
        )
        assert isinstance(result, GeometryClassifierResult)

    @pytest.mark.asyncio
    async def test_with_classifier_kwargs(self):
        """Should pass kwargs to classifier."""
        lines = [
            DetectedLine(start=(0.0, 0.0), end=(100.0, 0.0), linetype="CONTINUOUS"),
        ]
        result = await classify_geometry(
            lines=lines,
            wall_thickness_min=5.0,
            wall_thickness_max=8.0,
        )
        assert isinstance(result, GeometryClassifierResult)


class TestLineAngleCalculation:
    """Tests for line angle calculations."""

    @pytest.fixture
    def classifier(self):
        return GeometryClassifier()

    def test_horizontal_line_angle(self, classifier):
        """Horizontal line should have angle 0 or 180."""
        line = DetectedLine(start=(0.0, 0.0), end=(100.0, 0.0))
        angle = classifier._line_angle(line)
        assert angle == pytest.approx(0.0, abs=0.1) or angle == pytest.approx(180.0, abs=0.1)

    def test_vertical_line_angle(self, classifier):
        """Vertical line should have angle 90 or 270."""
        line = DetectedLine(start=(0.0, 0.0), end=(0.0, 100.0))
        angle = classifier._line_angle(line)
        assert angle == pytest.approx(90.0, abs=0.1) or angle == pytest.approx(270.0, abs=0.1)

    def test_45_degree_line(self, classifier):
        """45-degree line should have correct angle."""
        line = DetectedLine(start=(0.0, 0.0), end=(100.0, 100.0))
        angle = classifier._line_angle(line)
        assert angle == pytest.approx(45.0, abs=0.1) or angle == pytest.approx(225.0, abs=0.1)


class TestLineLengthCalculation:
    """Tests for line length calculations."""

    @pytest.fixture
    def classifier(self):
        return GeometryClassifier()

    def test_horizontal_line_length(self, classifier):
        """Horizontal line length should be correct."""
        line = DetectedLine(start=(0.0, 0.0), end=(100.0, 0.0))
        length = classifier._line_length(line)
        assert length == pytest.approx(100.0, abs=0.01)

    def test_vertical_line_length(self, classifier):
        """Vertical line length should be correct."""
        line = DetectedLine(start=(0.0, 0.0), end=(0.0, 50.0))
        length = classifier._line_length(line)
        assert length == pytest.approx(50.0, abs=0.01)

    def test_diagonal_line_length(self, classifier):
        """Diagonal line length should be correct (3-4-5 triangle)."""
        line = DetectedLine(start=(0.0, 0.0), end=(3.0, 4.0))
        length = classifier._line_length(line)
        assert length == pytest.approx(5.0, abs=0.01)


class TestPerpendicularDistance:
    """Tests for perpendicular distance calculation."""

    @pytest.fixture
    def classifier(self):
        return GeometryClassifier()

    def test_parallel_horizontal_lines(self, classifier):
        """Perpendicular distance between parallel horizontal lines."""
        line1 = DetectedLine(start=(0.0, 0.0), end=(100.0, 0.0))
        line2 = DetectedLine(start=(0.0, 10.0), end=(100.0, 10.0))
        dist = classifier._perpendicular_distance(line1, line2)
        assert dist == pytest.approx(10.0, abs=0.1)

    def test_parallel_vertical_lines(self, classifier):
        """Perpendicular distance between parallel vertical lines."""
        line1 = DetectedLine(start=(0.0, 0.0), end=(0.0, 100.0))
        line2 = DetectedLine(start=(8.0, 0.0), end=(8.0, 100.0))
        dist = classifier._perpendicular_distance(line1, line2)
        assert dist == pytest.approx(8.0, abs=0.1)


class TestDimensionTextDetection:
    """Tests for dimension text pattern matching."""

    @pytest.fixture
    def classifier(self):
        return GeometryClassifier()

    def test_feet_inches_format(self, classifier):
        """Should recognize feet-inches format."""
        assert classifier._is_dimension_text("12'-0\"") is True
        assert classifier._is_dimension_text("6'-6\"") is True

    def test_inches_format(self, classifier):
        """Should recognize inches format."""
        assert classifier._is_dimension_text('6"') is True
        assert classifier._is_dimension_text('24"') is True

    def test_feet_format(self, classifier):
        """Should recognize feet format."""
        assert classifier._is_dimension_text("10'") is True

    def test_decimal_format(self, classifier):
        """Should recognize decimal format."""
        assert classifier._is_dimension_text("3.5") is True
        assert classifier._is_dimension_text("12.75") is True

    def test_fraction_format(self, classifier):
        """Should recognize fraction format."""
        assert classifier._is_dimension_text("1/2") is True
        assert classifier._is_dimension_text("3/4") is True

    def test_non_dimension_text(self, classifier):
        """Should reject non-dimension text."""
        assert classifier._is_dimension_text("HVAC") is False
        assert classifier._is_dimension_text("room") is False
        assert classifier._is_dimension_text("ABC123") is False


class TestPipeSizeTextDetection:
    """Tests for pipe size text pattern matching."""

    @pytest.fixture
    def classifier(self):
        return GeometryClassifier()

    def test_fraction_pipe_size(self, classifier):
        """Should recognize fractional pipe sizes."""
        assert classifier._is_pipe_size_text('3/4"') is True
        assert classifier._is_pipe_size_text('1/2"') is True

    def test_integer_pipe_size(self, classifier):
        """Should recognize integer pipe sizes."""
        assert classifier._is_pipe_size_text('2"') is True
        assert classifier._is_pipe_size_text('4"') is True

    def test_mixed_pipe_size(self, classifier):
        """Should recognize mixed fraction pipe sizes."""
        assert classifier._is_pipe_size_text('1-1/2"') is True
        assert classifier._is_pipe_size_text('2-1/2"') is True

    def test_non_pipe_size_text(self, classifier):
        """Should reject non-pipe-size text."""
        assert classifier._is_pipe_size_text("CFM") is False
        assert classifier._is_pipe_size_text("24x12") is False
