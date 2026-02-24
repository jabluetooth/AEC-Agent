"""
Unit tests for VTracer Extraction Module.

Tests the VTracer-based raster-to-vector conversion for the Gemini-First pipeline.
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch, MagicMock

# Import types conditionally since VTracer may not be installed
from aec_agent.mcp.tools.gemini_first.vtracer_extraction import (
    VTracerConfig,
    VTracerColorMode,
    VTracerMode,
    ExtractedPath,
    BezierSegment,
    LineSegment,
    VTracerExtractionResult,
    VTracerExtractor,
    is_vtracer_available,
    VTRACER_AVAILABLE,
)


class TestVTracerConfig:
    """Tests for VTracerConfig dataclass."""

    def test_default_config(self):
        """Test default configuration values."""
        config = VTracerConfig()
        assert config.colormode == VTracerColorMode.BINARY
        assert config.mode == VTracerMode.SPLINE
        assert config.filter_speckle == 4
        assert config.corner_threshold == 60
        assert config.length_threshold == 4.0
        assert config.splice_threshold == 45
        assert config.path_precision == 3

    def test_custom_config(self):
        """Test custom configuration values."""
        config = VTracerConfig(
            colormode=VTracerColorMode.COLOR,
            mode=VTracerMode.POLYGON,
            filter_speckle=8,
            corner_threshold=45,
        )
        assert config.colormode == VTracerColorMode.COLOR
        assert config.mode == VTracerMode.POLYGON
        assert config.filter_speckle == 8
        assert config.corner_threshold == 45

    def test_config_from_settings(self):
        """Test creating config from application settings."""
        with patch("aec_agent.mcp.tools.gemini_first.vtracer_extraction.get_settings") as mock_settings:
            mock_settings.return_value = Mock(
                vtracer_mode="polygon",
                vtracer_filter_speckle=6,
                vtracer_corner_threshold=50,
                vtracer_length_threshold=5.0,
                vtracer_splice_threshold=40,
                vtracer_path_precision=2,
            )
            config = VTracerConfig.from_settings()
            assert config.mode == VTracerMode.POLYGON
            assert config.filter_speckle == 6
            assert config.corner_threshold == 50


class TestLineSegment:
    """Tests for LineSegment dataclass."""

    def test_line_segment_creation(self):
        """Test creating a line segment."""
        segment = LineSegment(
            start=(0.0, 0.0),
            end=(100.0, 0.0),
        )
        assert segment.start == (0.0, 0.0)
        assert segment.end == (100.0, 0.0)

    def test_line_length(self):
        """Test line length calculation."""
        segment = LineSegment(start=(0.0, 0.0), end=(100.0, 0.0))
        assert segment.length == 100.0

        # Diagonal line
        segment2 = LineSegment(start=(0.0, 0.0), end=(3.0, 4.0))
        assert segment2.length == 5.0

    def test_line_angle(self):
        """Test line angle calculation."""
        # Horizontal line
        segment = LineSegment(start=(0.0, 0.0), end=(100.0, 0.0))
        assert segment.angle == 0.0

        # Vertical line
        segment2 = LineSegment(start=(0.0, 0.0), end=(0.0, 100.0))
        assert segment2.angle == 90.0

        # 45 degree line
        segment3 = LineSegment(start=(0.0, 0.0), end=(100.0, 100.0))
        assert segment3.angle == 45.0

    def test_line_to_dict(self):
        """Test line serialization."""
        segment = LineSegment(start=(10.0, 20.0), end=(30.0, 40.0))
        d = segment.to_dict()
        assert d["type"] == "line"
        assert d["start"] == [10.0, 20.0]
        assert d["end"] == [30.0, 40.0]
        assert "length" in d
        assert "angle" in d


class TestBezierSegment:
    """Tests for BezierSegment dataclass."""

    def test_bezier_segment_creation(self):
        """Test creating a Bézier segment."""
        segment = BezierSegment(
            start=(0.0, 0.0),
            control1=(33.0, 100.0),
            control2=(66.0, 100.0),
            end=(100.0, 0.0),
        )
        assert segment.start == (0.0, 0.0)
        assert segment.end == (100.0, 0.0)
        assert segment.control1 == (33.0, 100.0)
        assert segment.control2 == (66.0, 100.0)

    def test_bezier_approximate_length(self):
        """Test Bézier arc length approximation."""
        # Simple curve
        segment = BezierSegment(
            start=(0.0, 0.0),
            control1=(0.0, 50.0),
            control2=(100.0, 50.0),
            end=(100.0, 0.0),
        )
        length = segment.approximate_length(segments=10)
        # Length should be greater than straight line distance (100)
        assert length > 100.0
        # But less than going through control points
        assert length < 200.0

    def test_bezier_to_dict(self):
        """Test Bézier serialization."""
        segment = BezierSegment(
            start=(0.0, 0.0),
            control1=(10.0, 20.0),
            control2=(30.0, 20.0),
            end=(40.0, 0.0),
        )
        d = segment.to_dict()
        assert d["type"] == "cubic_bezier"
        assert d["start"] == [0.0, 0.0]
        assert d["control1"] == [10.0, 20.0]
        assert d["control2"] == [30.0, 20.0]
        assert d["end"] == [40.0, 0.0]


class TestExtractedPath:
    """Tests for ExtractedPath dataclass."""

    def test_empty_path(self):
        """Test empty path."""
        path = ExtractedPath()
        assert path.num_segments == 0
        assert path.total_length == 0.0
        assert path.is_closed is False

    def test_path_with_lines(self):
        """Test path with line segments."""
        path = ExtractedPath(
            segments=[
                LineSegment(start=(0.0, 0.0), end=(100.0, 0.0)),
                LineSegment(start=(100.0, 0.0), end=(100.0, 100.0)),
            ],
            is_closed=False,
        )
        assert path.num_segments == 2
        assert path.total_length == 200.0

    def test_path_bounding_box(self):
        """Test path bounding box calculation."""
        path = ExtractedPath(
            segments=[
                LineSegment(start=(10.0, 20.0), end=(50.0, 80.0)),
                LineSegment(start=(50.0, 80.0), end=(90.0, 30.0)),
            ],
        )
        bbox = path.bounding_box
        assert bbox == (10.0, 20.0, 90.0, 80.0)  # min_x, min_y, max_x, max_y

    def test_path_to_dict(self):
        """Test path serialization."""
        path = ExtractedPath(
            segments=[
                LineSegment(start=(0.0, 0.0), end=(100.0, 0.0)),
            ],
            is_closed=True,
            fill_color="#000000",
        )
        d = path.to_dict()
        assert d["is_closed"] is True
        assert d["fill_color"] == "#000000"
        assert d["num_segments"] == 1
        assert len(d["segments"]) == 1


class TestVTracerExtractor:
    """Tests for VTracerExtractor class."""

    def test_extractor_initialization(self):
        """Test extractor initialization."""
        extractor = VTracerExtractor()
        assert extractor.config is not None

    def test_is_available(self):
        """Test availability check."""
        extractor = VTracerExtractor()
        # Should return True or False based on whether vtracer is installed
        assert isinstance(extractor.is_available, bool)

    def test_set_image(self):
        """Test setting image."""
        extractor = VTracerExtractor()
        image = np.zeros((100, 100), dtype=np.uint8)
        extractor.set_image(image)
        assert extractor._image is not None

    @pytest.mark.skipif(not VTRACER_AVAILABLE, reason="VTracer not installed")
    @pytest.mark.requires_vtracer
    def test_extract_simple_image(self):
        """Test extracting from a simple binary image."""
        # Create a simple image with a white rectangle on black background
        image = np.zeros((100, 100), dtype=np.uint8)
        image[20:80, 20:80] = 255

        extractor = VTracerExtractor(image)
        result = extractor.extract()

        assert isinstance(result, VTracerExtractionResult)
        assert result.image_width == 100
        assert result.image_height == 100
        # Should have at least one path (the rectangle)
        assert result.num_paths >= 1

    def test_extract_no_vtracer(self):
        """Test extraction when VTracer is not available."""
        with patch(
            "aec_agent.mcp.tools.gemini_first.vtracer_extraction.VTRACER_AVAILABLE",
            False,
        ):
            image = np.zeros((100, 100), dtype=np.uint8)
            extractor = VTracerExtractor(image)

            # Should gracefully return empty result
            result = extractor.extract()
            assert isinstance(result, VTracerExtractionResult)
            assert result.num_paths == 0

    def test_extract_no_image(self):
        """Test extraction without providing an image."""
        extractor = VTracerExtractor()
        result = extractor.extract()
        assert result.num_paths == 0


class TestSVGPathParsing:
    """Tests for SVG path data parsing."""

    def test_parse_simple_line_path(self):
        """Test parsing a simple line path."""
        extractor = VTracerExtractor()
        path = extractor._parse_path_data("M 0 0 L 100 0")

        assert len(path.segments) == 1
        assert isinstance(path.segments[0], LineSegment)
        assert path.segments[0].start == (0.0, 0.0)
        assert path.segments[0].end == (100.0, 0.0)

    def test_parse_closed_path(self):
        """Test parsing a closed path."""
        extractor = VTracerExtractor()
        path = extractor._parse_path_data("M 0 0 L 100 0 L 100 100 L 0 100 Z")

        assert len(path.segments) == 4  # 3 lines + closing line
        assert path.is_closed is True

    def test_parse_cubic_bezier(self):
        """Test parsing a cubic Bézier curve."""
        extractor = VTracerExtractor()
        path = extractor._parse_path_data("M 0 0 C 10 20 30 20 40 0")

        assert len(path.segments) == 1
        assert isinstance(path.segments[0], BezierSegment)
        assert path.segments[0].start == (0.0, 0.0)
        assert path.segments[0].control1 == (10.0, 20.0)
        assert path.segments[0].control2 == (30.0, 20.0)
        assert path.segments[0].end == (40.0, 0.0)

    def test_parse_horizontal_vertical(self):
        """Test parsing H and V commands."""
        extractor = VTracerExtractor()
        path = extractor._parse_path_data("M 0 0 H 100 V 50")

        assert len(path.segments) == 2
        assert path.segments[0].end == (100.0, 0.0)
        assert path.segments[1].end == (100.0, 50.0)

    def test_parse_relative_commands(self):
        """Test parsing relative path commands."""
        extractor = VTracerExtractor()
        path = extractor._parse_path_data("M 10 10 l 20 0 l 0 20")

        assert len(path.segments) == 2
        assert path.segments[0].start == (10.0, 10.0)
        assert path.segments[0].end == (30.0, 10.0)
        assert path.segments[1].end == (30.0, 30.0)


class TestVTracerExtractionResult:
    """Tests for VTracerExtractionResult dataclass."""

    def test_empty_result(self):
        """Test empty result."""
        result = VTracerExtractionResult()
        assert result.num_paths == 0
        assert result.total_segments == 0
        assert result.svg_content == ""

    def test_result_with_paths(self):
        """Test result with paths."""
        paths = [
            ExtractedPath(
                segments=[
                    LineSegment(start=(0.0, 0.0), end=(100.0, 0.0)),
                    LineSegment(start=(100.0, 0.0), end=(100.0, 100.0)),
                ]
            ),
            ExtractedPath(
                segments=[
                    LineSegment(start=(200.0, 200.0), end=(300.0, 300.0)),
                ]
            ),
        ]
        result = VTracerExtractionResult(
            paths=paths,
            image_width=400,
            image_height=400,
        )
        assert result.num_paths == 2
        assert result.total_segments == 3

    def test_result_to_dict(self):
        """Test result serialization."""
        result = VTracerExtractionResult(
            paths=[
                ExtractedPath(
                    segments=[LineSegment(start=(0.0, 0.0), end=(100.0, 0.0))],
                )
            ],
            image_width=100,
            image_height=100,
        )
        d = result.to_dict()
        assert d["num_paths"] == 1
        assert d["total_segments"] == 1
        assert d["image_size"] == (100, 100)


class TestHelperFunctions:
    """Tests for module-level helper functions."""

    def test_is_vtracer_available_function(self):
        """Test the is_vtracer_available function."""
        result = is_vtracer_available()
        assert isinstance(result, bool)
        assert result == VTRACER_AVAILABLE
