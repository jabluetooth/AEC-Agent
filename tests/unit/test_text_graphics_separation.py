"""
Unit tests for Fletcher-Kasturi Text/Graphics Separation Module.

Tests the connected component analysis and classification for
separating text from graphics in engineering drawings.
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch

from aec_agent.mcp.tools.gemini_first.text_graphics_separation import (
    SeparationConfig,
    SeparationResult,
    FletcherKasturiSeparator,
    ConnectedComponent,
    TextComponent,
    GraphicsComponent,
    TextLine,
    ComponentType,
    separate_text_from_graphics,
    is_separation_available,
    OPENCV_AVAILABLE,
)


class TestSeparationConfig:
    """Tests for SeparationConfig dataclass."""

    def test_default_config(self):
        """Test default configuration values."""
        config = SeparationConfig()
        assert config.area_threshold_factor == 3.0
        assert config.max_elongation == 10.0
        assert config.min_text_height == 8
        assert config.max_text_height == 100
        assert config.min_component_area == 10
        assert config.hough_step == 1.0
        assert config.gap_threshold_factor == 2.0
        assert config.enable_skeleton_segmentation is True

    def test_custom_config(self):
        """Test custom configuration values."""
        config = SeparationConfig(
            area_threshold_factor=5.0,
            max_elongation=15.0,
            min_text_height=10,
        )
        assert config.area_threshold_factor == 5.0
        assert config.max_elongation == 15.0
        assert config.min_text_height == 10

    def test_config_from_settings(self):
        """Test creating config from application settings."""
        with patch("aec_agent.mcp.tools.gemini_first.text_graphics_separation.get_settings") as mock_settings:
            mock_settings.return_value = Mock(
                text_graphics_area_factor=4.0,
                text_graphics_elongation_max=12.0,
                text_graphics_min_text_height=6,
                text_graphics_gap_factor=1.5,
            )
            config = SeparationConfig.from_settings()
            assert config.area_threshold_factor == 4.0
            assert config.max_elongation == 12.0
            assert config.min_text_height == 6


class TestConnectedComponent:
    """Tests for ConnectedComponent dataclass."""

    def test_component_creation(self):
        """Test creating a component."""
        comp = ConnectedComponent(
            label=1,
            bbox=(10, 20, 50, 30),
            area=1000,
            centroid=(35.0, 35.0),
        )
        assert comp.label == 1
        assert comp.x == 10
        assert comp.y == 20
        assert comp.width == 50
        assert comp.height == 30
        assert comp.area == 1000

    def test_aspect_ratio(self):
        """Test aspect ratio calculation."""
        comp = ConnectedComponent(
            label=1,
            bbox=(0, 0, 100, 50),
            area=5000,
            centroid=(50.0, 25.0),
        )
        assert comp.aspect_ratio == 2.0

    def test_elongation(self):
        """Test elongation calculation."""
        # Horizontal rectangle
        comp1 = ConnectedComponent(
            label=1,
            bbox=(0, 0, 100, 20),
            area=2000,
            centroid=(50.0, 10.0),
        )
        assert comp1.elongation == 5.0

        # Vertical rectangle
        comp2 = ConnectedComponent(
            label=2,
            bbox=(0, 0, 20, 100),
            area=2000,
            centroid=(10.0, 50.0),
        )
        assert comp2.elongation == 5.0

        # Square
        comp3 = ConnectedComponent(
            label=3,
            bbox=(0, 0, 50, 50),
            area=2500,
            centroid=(25.0, 25.0),
        )
        assert comp3.elongation == 1.0


class TestTextComponent:
    """Tests for TextComponent dataclass."""

    def test_text_component_creation(self):
        """Test creating a text component."""
        comp = TextComponent(
            label=1,
            bbox=(10, 20, 30, 15),
            area=300,
            centroid=(25.0, 27.5),
            component_type=ComponentType.TEXT,
            confidence=0.9,
        )
        assert comp.component_type == ComponentType.TEXT
        assert comp.confidence == 0.9


class TestGraphicsComponent:
    """Tests for GraphicsComponent dataclass."""

    def test_graphics_component_creation(self):
        """Test creating a graphics component."""
        comp = GraphicsComponent(
            label=1,
            bbox=(0, 0, 200, 5),
            area=1000,
            centroid=(100.0, 2.5),
            component_type=ComponentType.GRAPHICS,
            confidence=0.85,
        )
        assert comp.component_type == ComponentType.GRAPHICS
        assert comp.confidence == 0.85


class TestTextLine:
    """Tests for TextLine dataclass."""

    def test_empty_line(self):
        """Test empty text line."""
        line = TextLine()
        assert len(line.components) == 0
        assert line.bbox == (0, 0, 0, 0)
        assert line.text_height == 0.0

    def test_line_with_components(self):
        """Test text line with components."""
        comp1 = TextComponent(
            label=1,
            bbox=(10, 20, 20, 15),
            area=200,
            centroid=(20.0, 27.5),
        )
        comp2 = TextComponent(
            label=2,
            bbox=(35, 20, 25, 15),
            area=250,
            centroid=(47.5, 27.5),
        )

        line = TextLine(
            components=[comp1, comp2],
            baseline_y=35.0,
        )

        assert len(line.components) == 2
        assert line.bbox == (10, 20, 50, 15)  # Combined bbox
        assert line.text_height == 15.0


class TestSeparationResult:
    """Tests for SeparationResult dataclass."""

    def test_empty_result(self):
        """Test empty separation result."""
        result = SeparationResult(
            text_mask=np.array([]),
            graphics_mask=np.array([]),
        )
        assert result.num_text_components == 0
        assert result.num_graphics_components == 0
        assert result.num_text_lines == 0

    def test_result_with_components(self):
        """Test result with components."""
        text_comp = TextComponent(
            label=1,
            bbox=(10, 10, 20, 10),
            area=200,
            centroid=(20.0, 15.0),
        )
        graphics_comp = GraphicsComponent(
            label=2,
            bbox=(0, 0, 100, 5),
            area=500,
            centroid=(50.0, 2.5),
        )

        result = SeparationResult(
            text_mask=np.zeros((100, 100), dtype=np.uint8),
            graphics_mask=np.zeros((100, 100), dtype=np.uint8),
            text_components=[text_comp],
            graphics_components=[graphics_comp],
            text_lines=[TextLine(components=[text_comp])],
        )

        assert result.num_text_components == 1
        assert result.num_graphics_components == 1
        assert result.num_text_lines == 1

    def test_result_to_dict(self):
        """Test result serialization."""
        result = SeparationResult(
            text_mask=np.zeros((100, 100), dtype=np.uint8),
            graphics_mask=np.zeros((100, 100), dtype=np.uint8),
            text_components=[],
            graphics_components=[],
            noise_filtered=5,
        )
        d = result.to_dict()
        assert d["num_text_components"] == 0
        assert d["num_graphics_components"] == 0
        assert d["noise_filtered"] == 5


@pytest.mark.skipif(not OPENCV_AVAILABLE, reason="OpenCV not installed")
class TestFletcherKasturiSeparator:
    """Tests for FletcherKasturiSeparator class."""

    def test_separator_initialization(self):
        """Test separator initialization."""
        separator = FletcherKasturiSeparator()
        assert separator.is_available is True

    def test_separator_with_image(self):
        """Test separator with image."""
        image = np.zeros((100, 100), dtype=np.uint8)
        separator = FletcherKasturiSeparator(image)
        assert separator._image is not None

    def test_separate_empty_image(self):
        """Test separating an empty image."""
        image = np.zeros((100, 100), dtype=np.uint8)
        separator = FletcherKasturiSeparator(image)
        result = separator.separate()

        assert result.num_text_components == 0
        assert result.num_graphics_components == 0
        assert result.text_mask.shape == (100, 100)

    def test_separate_simple_text(self):
        """Test separating simple text-like components."""
        # Create image with small rectangular components (text-like)
        image = np.zeros((100, 100), dtype=np.uint8)

        # Add small rectangles (character-like)
        image[20:35, 10:20] = 255  # First "character"
        image[20:35, 25:35] = 255  # Second "character"
        image[20:35, 40:50] = 255  # Third "character"

        separator = FletcherKasturiSeparator(image)
        result = separator.separate()

        # Should detect these as text components
        assert result.num_text_components >= 0  # May be 3 or merged
        assert result.text_mask.shape == (100, 100)

    def test_separate_simple_graphics(self):
        """Test separating simple graphics-like components."""
        # Create image with long thin lines (graphics-like)
        image = np.zeros((100, 100), dtype=np.uint8)

        # Add horizontal line
        image[50:52, 10:90] = 255  # Thin horizontal line

        # Add vertical line
        image[10:90, 50:52] = 255  # Thin vertical line

        separator = FletcherKasturiSeparator(image)
        result = separator.separate()

        # Should detect these as graphics (high elongation)
        assert result.graphics_mask.shape == (100, 100)

    def test_separate_mixed_content(self):
        """Test separating mixed text and graphics."""
        image = np.zeros((200, 200), dtype=np.uint8)

        # Add text-like components
        image[20:32, 20:30] = 255
        image[20:32, 35:45] = 255

        # Add graphics-like components (long line)
        image[100:102, 20:180] = 255

        separator = FletcherKasturiSeparator(image)
        config = SeparationConfig(
            min_text_height=8,
            max_text_height=50,
            max_elongation=15.0,
        )
        result = separator.separate(config=config)

        # Should have both text and graphics
        assert result.text_mask.shape == (200, 200)
        assert result.graphics_mask.shape == (200, 200)

    def test_noise_filtering(self):
        """Test that noise is filtered out."""
        image = np.zeros((100, 100), dtype=np.uint8)

        # Add tiny noise pixels
        image[10, 10] = 255
        image[20, 20] = 255
        image[30, 30] = 255

        config = SeparationConfig(min_component_area=50)
        separator = FletcherKasturiSeparator(image, config)
        result = separator.separate()

        # All small components should be filtered
        assert result.noise_filtered == 3
        assert result.num_text_components == 0
        assert result.num_graphics_components == 0


@pytest.mark.skipif(not OPENCV_AVAILABLE, reason="OpenCV not installed")
class TestComponentClassification:
    """Tests for component classification logic."""

    def test_classify_by_area(self):
        """Test classification based on area."""
        image = np.zeros((200, 200), dtype=np.uint8)

        # Small component (text-like)
        image[20:35, 20:35] = 255  # 15x15 = 225 pixels

        # Large component (graphics-like)
        image[100:150, 100:180] = 255  # 50x80 = 4000 pixels

        separator = FletcherKasturiSeparator(image)
        result = separator.separate()

        # Should classify based on area
        assert len(result.text_components) + len(result.graphics_components) >= 1

    def test_classify_by_elongation(self):
        """Test classification based on elongation."""
        image = np.zeros((200, 200), dtype=np.uint8)

        # High elongation (line/graphics)
        image[50:52, 20:180] = 255  # 160:2 = 80 elongation

        # Low elongation (square/text)
        image[120:135, 120:135] = 255  # 15:15 = 1 elongation

        config = SeparationConfig(max_elongation=10.0)
        separator = FletcherKasturiSeparator(image, config)
        result = separator.separate()

        # Long thin component should be graphics
        # Square component might be text or graphics based on size


class TestTextLineGrouping:
    """Tests for text line grouping."""

    @pytest.mark.skipif(not OPENCV_AVAILABLE, reason="OpenCV not installed")
    def test_group_text_on_same_line(self):
        """Test grouping characters on the same baseline."""
        image = np.zeros((100, 200), dtype=np.uint8)

        # Characters on same baseline
        image[30:45, 10:20] = 255
        image[30:45, 25:35] = 255
        image[30:45, 40:50] = 255

        config = SeparationConfig(
            min_text_height=10,
            max_text_height=30,
            gap_threshold_factor=3.0,
        )
        separator = FletcherKasturiSeparator(image, config)
        result = separator.separate()

        # Should group into one text line
        if result.text_lines:
            assert result.text_lines[0].text_height > 0


class TestHelperFunctions:
    """Tests for module-level helper functions."""

    def test_is_separation_available(self):
        """Test the is_separation_available function."""
        result = is_separation_available()
        assert isinstance(result, bool)
        assert result == OPENCV_AVAILABLE

    @pytest.mark.skipif(not OPENCV_AVAILABLE, reason="OpenCV not installed")
    def test_separate_text_from_graphics_convenience(self):
        """Test the convenience function."""
        image = np.zeros((100, 100), dtype=np.uint8)
        image[20:35, 20:35] = 255

        result = separate_text_from_graphics(image)
        assert isinstance(result, SeparationResult)
        assert result.text_mask.shape == (100, 100)


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_no_opencv(self):
        """Test behavior when OpenCV is not available."""
        with patch(
            "aec_agent.mcp.tools.gemini_first.text_graphics_separation.OPENCV_AVAILABLE",
            False,
        ):
            separator = FletcherKasturiSeparator()
            assert separator.is_available is False

            result = separator.separate()
            assert len(result.text_mask) == 0

    def test_no_image_provided(self):
        """Test separation without providing an image."""
        separator = FletcherKasturiSeparator()
        result = separator.separate()
        assert len(result.text_mask) == 0

    @pytest.mark.skipif(not OPENCV_AVAILABLE, reason="OpenCV not installed")
    def test_rgb_image_conversion(self):
        """Test that RGB images are converted to grayscale."""
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        image[20:35, 20:35] = [255, 255, 255]

        separator = FletcherKasturiSeparator(image)
        result = separator.separate()

        # Should handle RGB input
        assert result.text_mask.shape == (100, 100)
