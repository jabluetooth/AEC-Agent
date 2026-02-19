"""
Unit tests for Hybrid Extraction (Gemini + OpenCV + YOLO fusion).

Tests the integration of multiple extraction methods for optimal
raster PDF to vector conversion.
"""

import pytest
import numpy as np
from pathlib import Path
from unittest.mock import MagicMock, AsyncMock, patch

from aec_agent.mcp.tools.gemini_first.adaptive_extraction import (
    HybridExtractionConfig,
    HybridExtractionResult,
    hybrid_extract_all,
    hybrid_opencv_extraction,
    hybrid_yolo_extraction,
    _merge_duplicate_entities,
    _merge_lines,
    _merge_circles,
    _infer_layer_for_region,
    _enrich_symbol_attributes,
    EntityToCreate,
    EntityType,
    ExtractionSource,
)
from aec_agent.mcp.tools.gemini_first.gemini_understanding import (
    DrawingAnalysis,
    DrawingElements,
    DrawingType,
    ExtractionStrategy,
    ExtractionStrategyConfig,
    SpecialRegion,
    DetectedSymbol,
)
from aec_agent.mcp.tools.gemini_first.coordinate_calibration import ScaleCalibration


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def mock_calibration():
    """Create a mock calibration object."""
    calibration = MagicMock(spec=ScaleCalibration)
    calibration.method = "dimension"
    calibration.confidence = 0.95
    calibration.dpi = 300
    calibration.to_dwg = lambda x, y: (x / 300.0, (1000 - y) / 300.0)
    calibration.scale_length = lambda px: px / 300.0
    return calibration


@pytest.fixture
def mock_analysis():
    """Create a mock drawing analysis."""
    analysis = MagicMock(spec=DrawingAnalysis)
    analysis.drawing_type = "floor_plan"
    analysis.total_elements = 50

    # Elements
    analysis.elements = MagicMock(spec=DrawingElements)
    analysis.elements.lines = []
    analysis.elements.arcs = []
    analysis.elements.circles = []
    analysis.elements.text = []
    analysis.elements.symbols = []
    analysis.elements.dimensions = []

    # Extraction strategy
    analysis.extraction_strategy = MagicMock(spec=ExtractionStrategyConfig)
    analysis.extraction_strategy.primary_strategy = "hybrid"
    analysis.extraction_strategy.special_regions = []

    return analysis


@pytest.fixture
def mock_image():
    """Create a mock grayscale image."""
    # 1000x800 white image with some black lines
    image = np.ones((800, 1000, 3), dtype=np.uint8) * 255

    # Draw some lines (black)
    image[100:102, 100:500] = 0  # Horizontal line
    image[200:400, 300:302] = 0  # Vertical line

    # Draw a circle (approximate)
    import cv2
    cv2.circle(image, (600, 400), 50, (0, 0, 0), 2)

    return image


@pytest.fixture
def hybrid_config():
    """Create a hybrid extraction configuration."""
    return HybridExtractionConfig(
        use_opencv_for_lines=True,
        use_opencv_for_circles=True,
        use_yolo_for_symbols=False,  # Disable YOLO for unit tests
        use_gemini_for_text=True,
        opencv_line_min_length=20,
        opencv_circle_min_radius=10,
        opencv_circle_max_radius=100,
        coordinate_tolerance=5.0,
        prefer_opencv_geometry=True,
    )


# =============================================================================
# Test HybridExtractionConfig
# =============================================================================

class TestHybridExtractionConfig:
    """Tests for HybridExtractionConfig dataclass."""

    def test_default_values(self):
        """Test default configuration values."""
        config = HybridExtractionConfig()

        assert config.use_opencv_for_lines is True
        assert config.use_opencv_for_circles is True
        assert config.use_yolo_for_symbols is True
        assert config.use_gemini_for_text is True
        assert config.opencv_line_min_length == 30
        assert config.yolo_confidence_threshold == 0.5
        assert config.coordinate_tolerance == 5.0
        assert config.prefer_opencv_geometry is True
        assert config.enable_gemini_validation is True

    def test_custom_values(self):
        """Test custom configuration values."""
        config = HybridExtractionConfig(
            use_opencv_for_lines=False,
            opencv_line_min_length=50,
            yolo_confidence_threshold=0.7,
            coordinate_tolerance=10.0,
        )

        assert config.use_opencv_for_lines is False
        assert config.opencv_line_min_length == 50
        assert config.yolo_confidence_threshold == 0.7
        assert config.coordinate_tolerance == 10.0


# =============================================================================
# Test HybridExtractionResult
# =============================================================================

class TestHybridExtractionResult:
    """Tests for HybridExtractionResult dataclass."""

    def test_creation(self):
        """Test result creation."""
        result = HybridExtractionResult(
            primary_strategy="hybrid",
            drawing_type="floor_plan",
            gemini_entities=10,
            opencv_entities=50,
            yolo_entities=5,
        )

        assert result.primary_strategy == "hybrid"
        assert result.gemini_entities == 10
        assert result.opencv_entities == 50
        assert result.yolo_entities == 5

    def test_to_dict(self):
        """Test serialization to dict."""
        result = HybridExtractionResult(
            primary_strategy="hybrid",
            gemini_entities=10,
            opencv_entities=50,
            yolo_entities=5,
            duplicates_merged=3,
        )

        data = result.to_dict()

        assert "hybrid_statistics" in data
        assert data["hybrid_statistics"]["gemini_entities"] == 10
        assert data["hybrid_statistics"]["opencv_entities"] == 50
        assert data["hybrid_statistics"]["yolo_entities"] == 5
        assert data["hybrid_statistics"]["duplicates_merged"] == 3


# =============================================================================
# Test Entity Merging
# =============================================================================

class TestEntityMerging:
    """Tests for duplicate entity merging."""

    def test_merge_empty_list(self):
        """Test merging empty list."""
        merged, count = _merge_duplicate_entities([])
        assert merged == []
        assert count == 0

    def test_merge_no_duplicates(self):
        """Test merging with no duplicates."""
        entities = [
            EntityToCreate(
                entity_type=EntityType.LINE,
                layer="A-WALL",
                properties={"start": (0.0, 0.0), "end": (10.0, 0.0)},
                source=ExtractionSource.DIRECT,
            ),
            EntityToCreate(
                entity_type=EntityType.LINE,
                layer="A-WALL",
                properties={"start": (20.0, 0.0), "end": (30.0, 0.0)},
                source=ExtractionSource.SELECTIVE_OPENCV,
            ),
        ]

        merged, count = _merge_duplicate_entities(entities, tolerance=5.0)

        assert len(merged) == 2
        assert count == 0

    def test_merge_duplicate_lines(self):
        """Test merging duplicate lines."""
        entities = [
            EntityToCreate(
                entity_type=EntityType.LINE,
                layer="A-WALL",
                properties={"start": (0.0, 0.0), "end": (10.0, 0.0)},
                source=ExtractionSource.DIRECT,
            ),
            EntityToCreate(
                entity_type=EntityType.LINE,
                layer="A-WALL",
                properties={"start": (0.5, 0.5), "end": (10.5, 0.5)},  # Within tolerance
                source=ExtractionSource.SELECTIVE_OPENCV,
            ),
        ]

        merged, count = _merge_duplicate_entities(entities, tolerance=5.0, prefer_opencv=True)

        assert len(merged) == 1
        assert count == 1
        # Should prefer OpenCV
        assert merged[0].source == ExtractionSource.SELECTIVE_OPENCV

    def test_merge_duplicate_circles(self):
        """Test merging duplicate circles."""
        entities = [
            EntityToCreate(
                entity_type=EntityType.CIRCLE,
                layer="A-COLS",
                properties={"center": (100.0, 100.0), "radius": 10.0},
                source=ExtractionSource.DIRECT,
            ),
            EntityToCreate(
                entity_type=EntityType.CIRCLE,
                layer="A-COLS",
                properties={"center": (101.0, 101.0), "radius": 10.5},  # Within tolerance
                source=ExtractionSource.SELECTIVE_OPENCV,
            ),
        ]

        merged, count = _merge_duplicate_entities(entities, tolerance=5.0, prefer_opencv=True)

        assert len(merged) == 1
        assert count == 1
        assert merged[0].source == ExtractionSource.SELECTIVE_OPENCV

    def test_merge_different_types_not_merged(self):
        """Test that different entity types are not merged."""
        entities = [
            EntityToCreate(
                entity_type=EntityType.LINE,
                layer="A-WALL",
                properties={"start": (0.0, 0.0), "end": (10.0, 0.0)},
            ),
            EntityToCreate(
                entity_type=EntityType.CIRCLE,
                layer="A-COLS",
                properties={"center": (5.0, 0.0), "radius": 10.0},
            ),
        ]

        merged, count = _merge_duplicate_entities(entities, tolerance=5.0)

        assert len(merged) == 2
        assert count == 0


# =============================================================================
# Test Layer Inference
# =============================================================================

class TestLayerInference:
    """Tests for layer inference from regions."""

    def test_infer_layer_wall(self, mock_analysis):
        """Test inferring wall layer."""
        region = SpecialRegion(
            bounds=[0, 0, 100, 100],
            reason="wall elements",
            strategy="selective_opencv",
            expected_pattern="walls",
        )

        layer = _infer_layer_for_region(region, "line", mock_analysis)
        assert layer == "A-WALL"

    def test_infer_layer_duct(self, mock_analysis):
        """Test inferring duct layer."""
        region = SpecialRegion(
            bounds=[0, 0, 100, 100],
            reason="duct system",
            strategy="selective_opencv",
            expected_pattern="duct",
        )

        layer = _infer_layer_for_region(region, "line", mock_analysis)
        assert layer == "M-DUCT"

    def test_infer_layer_from_drawing_type(self, mock_analysis):
        """Test inferring layer from drawing type."""
        mock_analysis.drawing_type = "mechanical"
        region = SpecialRegion(
            bounds=[0, 0, 100, 100],
            reason="unknown",
            strategy="selective_opencv",
        )

        layer = _infer_layer_for_region(region, "line", mock_analysis)
        assert layer == "M-DUCT"

    def test_infer_layer_default(self, mock_analysis):
        """Test default layer when no context."""
        mock_analysis.drawing_type = "other"
        region = SpecialRegion(
            bounds=[0, 0, 100, 100],
            reason="",
            strategy="selective_opencv",
        )

        layer = _infer_layer_for_region(region, "line", mock_analysis)
        assert layer == "0"


# =============================================================================
# Test Symbol Attribute Enrichment
# =============================================================================

class TestSymbolEnrichment:
    """Tests for symbol attribute enrichment."""

    def test_enrich_with_matching_symbol(self, mock_analysis):
        """Test enrichment when matching Gemini symbol found."""
        # Create a mock detected block
        block = MagicMock()
        block.position = (100.0, 200.0)

        # Add a matching symbol in Gemini analysis
        gemini_symbol = DetectedSymbol(
            symbol_type="diffuser",
            position=(105, 205),  # Close to block position
            tag="SD-1",
            size="12x12",
            associated_text=["Supply Air"],
        )
        mock_analysis.elements.symbols = [gemini_symbol]

        attributes = _enrich_symbol_attributes(block, mock_analysis)

        assert attributes["TAG"] == "SD-1"
        assert attributes["SIZE"] == "12x12"
        assert attributes["TEXT1"] == "Supply Air"

    def test_enrich_no_matching_symbol(self, mock_analysis):
        """Test enrichment when no matching symbol found."""
        block = MagicMock()
        block.position = (100.0, 200.0)

        # Symbol far away from block
        gemini_symbol = DetectedSymbol(
            symbol_type="diffuser",
            position=(500, 500),  # Far from block position
            tag="SD-2",
        )
        mock_analysis.elements.symbols = [gemini_symbol]

        attributes = _enrich_symbol_attributes(block, mock_analysis)

        assert attributes == {}  # No match found


# =============================================================================
# Test OpenCV Extraction Module
# =============================================================================

class TestOpenCVExtractionModule:
    """Tests for OpenCV extraction utilities."""

    def test_opencv_available(self):
        """Test that OpenCV is available."""
        from aec_agent.mcp.tools.gemini_first.opencv_extraction import OPENCV_AVAILABLE
        assert OPENCV_AVAILABLE is True

    def test_extractor_initialization(self, mock_image):
        """Test OpenCVExtractor initialization."""
        from aec_agent.mcp.tools.gemini_first.opencv_extraction import OpenCVExtractor

        extractor = OpenCVExtractor(mock_image, dpi=300)

        assert extractor.width == 1000
        assert extractor.height == 800
        assert extractor.dpi == 300

    def test_line_extraction(self, mock_image):
        """Test line extraction from image."""
        from aec_agent.mcp.tools.gemini_first.opencv_extraction import OpenCVExtractor

        extractor = OpenCVExtractor(mock_image)
        lines = extractor.extract_lines_lsd(min_length=20)

        # Should find at least some lines
        assert isinstance(lines, list)
        # Lines may or may not be found depending on image content

    def test_circle_extraction(self, mock_image):
        """Test circle extraction from image."""
        from aec_agent.mcp.tools.gemini_first.opencv_extraction import OpenCVExtractor

        extractor = OpenCVExtractor(mock_image)
        circles = extractor.extract_circles(min_radius=30, max_radius=70)

        # Should find circles
        assert isinstance(circles, list)

    def test_extract_all(self, mock_image):
        """Test full extraction."""
        from aec_agent.mcp.tools.gemini_first.opencv_extraction import OpenCVExtractor

        extractor = OpenCVExtractor(mock_image)
        result = extractor.extract_all()

        assert hasattr(result, 'lines')
        assert hasattr(result, 'circles')
        assert hasattr(result, 'contours')
        assert hasattr(result, 'polylines')
        assert hasattr(result, 'total_elements')

    def test_roi_extraction(self, mock_image):
        """Test extraction from region of interest."""
        from aec_agent.mcp.tools.gemini_first.opencv_extraction import OpenCVExtractor

        extractor = OpenCVExtractor(mock_image)

        # Extract from specific region
        roi = (50, 50, 550, 200)
        lines = extractor.extract_lines_lsd(roi=roi, min_length=10)

        assert isinstance(lines, list)

    def test_extracted_line_properties(self, mock_image):
        """Test ExtractedLine dataclass properties."""
        from aec_agent.mcp.tools.gemini_first.opencv_extraction import ExtractedLine, LineType

        line = ExtractedLine(
            start=(0.0, 0.0),
            end=(100.0, 0.0),
            thickness=2.0,
            line_type=LineType.CONTINUOUS,
            confidence=0.95,
        )

        assert line.length == 100.0
        assert line.angle == 0.0  # Horizontal line
        assert line.thickness == 2.0

        data = line.to_dict()
        assert data["start"] == [0.0, 0.0]
        assert data["end"] == [100.0, 0.0]
        assert data["length"] == 100.0


# =============================================================================
# Test Hybrid Extraction Integration
# =============================================================================

class TestHybridExtractionIntegration:
    """Integration tests for hybrid extraction."""

    @pytest.mark.asyncio
    async def test_hybrid_extract_no_image(self, mock_analysis, mock_calibration):
        """Test hybrid extraction without image path (Gemini only)."""
        config = HybridExtractionConfig(
            use_opencv_for_lines=False,
            use_opencv_for_circles=False,
            use_yolo_for_symbols=False,
        )

        result = await hybrid_extract_all(
            analysis=mock_analysis,
            calibration=mock_calibration,
            image_path=None,
            config=config,
        )

        assert isinstance(result, HybridExtractionResult)
        assert result.primary_strategy == "hybrid"
        assert result.opencv_entities == 0
        assert result.yolo_entities == 0

    @pytest.mark.asyncio
    async def test_hybrid_opencv_extraction_with_regions(
        self, mock_image, mock_analysis, mock_calibration, hybrid_config
    ):
        """Test OpenCV extraction with Gemini-guided regions."""
        # Add a region for OpenCV processing
        region = SpecialRegion(
            bounds=[50, 50, 550, 200],
            reason="wall_lines",
            strategy="selective_opencv",
            expected_pattern="lines",
        )
        mock_analysis.extraction_strategy.special_regions = [region]

        entities = await hybrid_opencv_extraction(
            image=mock_image,
            analysis=mock_analysis,
            calibration=mock_calibration,
            config=hybrid_config,
        )

        assert isinstance(entities, list)
        # All entities should have SELECTIVE_OPENCV source
        for e in entities:
            assert e.source == ExtractionSource.SELECTIVE_OPENCV

    @pytest.mark.asyncio
    async def test_hybrid_extract_full_workflow(
        self, mock_image, mock_analysis, mock_calibration, hybrid_config, tmp_path
    ):
        """Test full hybrid extraction workflow with image file."""
        import cv2

        # Save mock image to temp file
        image_path = tmp_path / "test_drawing.png"
        cv2.imwrite(str(image_path), mock_image)

        # Add a region for processing
        region = SpecialRegion(
            bounds=[0, 0, 1000, 800],
            reason="full_image",
            strategy="selective_opencv",
            expected_pattern="lines",
        )
        mock_analysis.extraction_strategy.special_regions = [region]

        result = await hybrid_extract_all(
            analysis=mock_analysis,
            calibration=mock_calibration,
            image_path=image_path,
            config=hybrid_config,
        )

        assert isinstance(result, HybridExtractionResult)
        assert result.primary_strategy == "hybrid"
        # Should have some statistics
        assert result.gemini_entities >= 0
        assert result.opencv_entities >= 0


# =============================================================================
# Test Edge Cases
# =============================================================================

class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_merge_reversed_line(self):
        """Test merging line with reversed endpoints."""
        entities = [
            EntityToCreate(
                entity_type=EntityType.LINE,
                layer="A-WALL",
                properties={"start": (0.0, 0.0), "end": (10.0, 0.0)},
                source=ExtractionSource.DIRECT,
            ),
            EntityToCreate(
                entity_type=EntityType.LINE,
                layer="A-WALL",
                properties={"start": (10.5, 0.5), "end": (0.5, 0.5)},  # Reversed
                source=ExtractionSource.SELECTIVE_OPENCV,
            ),
        ]

        merged, count = _merge_duplicate_entities(entities, tolerance=5.0)

        assert len(merged) == 1
        assert count == 1

    def test_empty_analysis(self, mock_calibration):
        """Test with empty drawing analysis."""
        analysis = MagicMock(spec=DrawingAnalysis)
        analysis.drawing_type = ""
        analysis.total_elements = 0
        analysis.elements = MagicMock()
        analysis.elements.lines = []
        analysis.elements.arcs = []
        analysis.elements.circles = []
        analysis.elements.text = []
        analysis.elements.symbols = []
        analysis.elements.dimensions = []
        analysis.extraction_strategy = MagicMock()
        analysis.extraction_strategy.primary_strategy = "direct"
        analysis.extraction_strategy.special_regions = []

        # Should not crash with empty analysis
        result = _infer_layer_for_region(
            SpecialRegion(bounds=[0, 0, 100, 100], reason="", strategy="opencv"),
            "line",
            analysis,
        )
        assert result == "0"

    def test_invalid_roi_bounds(self, mock_image):
        """Test with invalid ROI bounds."""
        from aec_agent.mcp.tools.gemini_first.opencv_extraction import OpenCVExtractor

        extractor = OpenCVExtractor(mock_image)

        # Negative bounds should be clamped
        lines = extractor.extract_lines_lsd(roi=(-10, -10, 100, 100))
        assert isinstance(lines, list)

        # Reversed bounds
        lines = extractor.extract_lines_lsd(roi=(100, 100, 50, 50))
        assert isinstance(lines, list)  # Should return empty or handle gracefully
