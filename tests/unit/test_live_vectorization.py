"""
Unit tests for LIVE Vectorization (Phase C.3).

Tests the layer-wise image vectorization module.
"""

import pytest
import numpy as np
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import tempfile

from aec_agent.mcp.tools.gemini_first.live_vectorization import (
    LIVEConfig,
    VectorPath,
    VectorLayer,
    LIVEResult,
    is_live_available,
)


class TestLIVEConfig:
    """Tests for LIVEConfig dataclass."""

    def test_default_config(self):
        """Test default configuration values."""
        config = LIVEConfig()
        assert config.num_layers == 5
        assert config.paths_per_layer == 1
        assert config.control_points_per_path == 8
        assert config.iterations_per_layer == 500
        assert config.learning_rate == 0.01
        assert config.xing_loss_weight == 0.1
        assert config.udf_loss_weight == 1.0
        assert config.use_color is True

    def test_custom_config(self):
        """Test custom configuration values."""
        config = LIVEConfig(
            num_layers=10,
            paths_per_layer=3,
            iterations_per_layer=1000,
            gpu_id=-1,
        )
        assert config.num_layers == 10
        assert config.paths_per_layer == 3
        assert config.iterations_per_layer == 1000
        assert config.gpu_id == -1

    def test_to_dict(self):
        """Test conversion to dictionary."""
        config = LIVEConfig(
            num_layers=3,
            paths_per_layer=2,
        )
        result = config.to_dict()
        assert isinstance(result, dict)
        assert result["num_layers"] == 3
        assert result["paths_per_layer"] == 2
        assert "learning_rate" in result


class TestVectorPath:
    """Tests for VectorPath dataclass."""

    def test_path_creation(self):
        """Test creating a vector path."""
        path = VectorPath(
            control_points=[
                (10.0, 10.0),
                (50.0, 10.0),
                (50.0, 50.0),
                (10.0, 50.0),
            ],
            fill_color=(255, 128, 0),
            opacity=0.9,
        )
        assert len(path.control_points) == 4
        assert path.fill_color == (255, 128, 0)
        assert path.opacity == 0.9

    def test_default_values(self):
        """Test default path values."""
        path = VectorPath(
            control_points=[(0, 0), (100, 0), (100, 100), (0, 100)],
        )
        assert path.fill_color == (0, 0, 0)
        assert path.opacity == 1.0

    def test_svg_path_generation(self):
        """Test SVG path data generation."""
        path = VectorPath(
            control_points=[
                (0.0, 0.0),
                (25.0, 0.0),
                (50.0, 25.0),
                (50.0, 50.0),
                (25.0, 50.0),
                (0.0, 25.0),
            ],
        )
        svg_path = path.to_svg_path()
        assert "M 0.000 0.000" in svg_path
        assert "C" in svg_path  # Cubic Bezier
        assert "Z" in svg_path  # Closed path

    def test_empty_path(self):
        """Test handling of empty control points."""
        path = VectorPath(control_points=[])
        assert path.to_svg_path() == ""

    def test_too_few_points(self):
        """Test handling of too few control points."""
        path = VectorPath(control_points=[(0, 0), (10, 10)])
        # Should return empty string or minimal path
        svg_path = path.to_svg_path()
        assert svg_path == ""

    def test_to_dict(self):
        """Test conversion to dictionary."""
        path = VectorPath(
            control_points=[(0, 0), (10, 0), (10, 10), (0, 10)],
            fill_color=(100, 150, 200),
            opacity=0.75,
        )
        result = path.to_dict()
        assert "control_points" in result
        assert "fill_color" in result
        assert "opacity" in result
        assert "svg_path" in result


class TestVectorLayer:
    """Tests for VectorLayer dataclass."""

    def test_layer_creation(self):
        """Test creating a vector layer."""
        paths = [
            VectorPath(
                [(0, 0), (50, 0), (50, 50), (0, 50)],
                fill_color=(255, 0, 0),
            ),
        ]
        layer = VectorLayer(
            layer_index=0,
            paths=paths,
            fill_color=(128, 128, 128),
        )
        assert layer.layer_index == 0
        assert len(layer.paths) == 1
        assert layer.fill_color == (128, 128, 128)

    def test_empty_layer(self):
        """Test creating an empty layer."""
        layer = VectorLayer(
            layer_index=1,
            paths=[],
        )
        assert layer.layer_index == 1
        assert len(layer.paths) == 0

    def test_to_dict(self):
        """Test conversion to dictionary."""
        layer = VectorLayer(
            layer_index=2,
            paths=[
                VectorPath([(0, 0), (10, 0), (10, 10), (0, 10)]),
            ],
            fill_color=(255, 255, 255),
        )
        result = layer.to_dict()
        assert result["layer_index"] == 2
        assert result["num_paths"] == 1
        assert "paths" in result
        assert "fill_color" in result


class TestLIVEResult:
    """Tests for LIVEResult dataclass."""

    def test_result_creation(self):
        """Test creating a result object."""
        layers = [
            VectorLayer(
                layer_index=0,
                paths=[VectorPath([(0, 0), (50, 0), (50, 50), (0, 50)])],
            ),
            VectorLayer(
                layer_index=1,
                paths=[VectorPath([(10, 10), (40, 10), (40, 40), (10, 40)])],
            ),
        ]
        config = LIVEConfig()

        result = LIVEResult(
            layers=layers,
            loss_history=[[1.0, 0.5], [0.4, 0.3]],
            final_loss=0.3,
            image_size=(100, 100),
            processing_time_ms=5000.0,
            config=config,
            device="cpu",
        )

        assert len(result.layers) == 2
        assert result.final_loss == 0.3
        assert result.image_size == (100, 100)
        assert len(result.loss_history) == 2

    def test_to_svg(self):
        """Test SVG generation."""
        layers = [
            VectorLayer(
                layer_index=0,
                paths=[
                    VectorPath(
                        [(10, 10), (40, 10), (40, 40), (25, 50), (10, 40)],
                        fill_color=(200, 100, 50),
                        opacity=0.8,
                    ),
                ],
            ),
        ]
        config = LIVEConfig()

        result = LIVEResult(
            layers=layers,
            loss_history=[[0.1]],
            final_loss=0.1,
            image_size=(100, 100),
            processing_time_ms=100.0,
            config=config,
        )

        svg = result.to_svg()
        assert '<svg xmlns="http://www.w3.org/2000/svg"' in svg
        assert 'width="100"' in svg
        assert 'height="100"' in svg
        assert "<rect" in svg  # Background
        assert "<path" in svg
        assert "</svg>" in svg

    def test_to_svg_with_background(self):
        """Test SVG generation with custom background."""
        layers = []
        config = LIVEConfig()

        result = LIVEResult(
            layers=layers,
            loss_history=[],
            final_loss=0.0,
            image_size=(200, 150),
            processing_time_ms=0.0,
            config=config,
        )

        svg = result.to_svg(background_color="black")
        assert 'fill="black"' in svg

    def test_save_svg(self):
        """Test saving SVG to file."""
        layers = [
            VectorLayer(
                layer_index=0,
                paths=[
                    VectorPath(
                        [(5, 5), (45, 5), (45, 45), (5, 45)],
                        fill_color=(0, 128, 255),
                    ),
                ],
            ),
        ]
        config = LIVEConfig()

        result = LIVEResult(
            layers=layers,
            loss_history=[],
            final_loss=0.0,
            image_size=(50, 50),
            processing_time_ms=50.0,
            config=config,
        )

        with tempfile.NamedTemporaryFile(suffix=".svg", delete=False) as f:
            result.save_svg(f.name)
            svg_path = Path(f.name)

        assert svg_path.exists()
        content = svg_path.read_text()
        assert "<svg" in content
        svg_path.unlink()

    def test_to_dict(self):
        """Test conversion to dictionary."""
        config = LIVEConfig(num_layers=3)
        result = LIVEResult(
            layers=[
                VectorLayer(0, [VectorPath([(0, 0), (10, 0), (10, 10), (0, 10)])]),
                VectorLayer(1, []),
            ],
            loss_history=[[1.0], [0.5]],
            final_loss=0.5,
            image_size=(200, 200),
            processing_time_ms=2500.0,
            config=config,
            device="cuda:0",
        )

        data = result.to_dict()
        assert data["num_layers"] == 2
        assert data["total_paths"] == 1
        assert data["final_loss"] == 0.5
        assert data["device"] == "cuda:0"
        assert "config" in data
        assert "layers" in data


class TestAvailabilityCheck:
    """Tests for availability checking."""

    def test_availability_returns_bool(self):
        """Test that availability check returns a boolean."""
        result = is_live_available()
        assert isinstance(result, bool)


@pytest.mark.requires_phase_c
class TestLIVEWithTorch:
    """Tests requiring PyTorch."""

    def test_closed_bezier_path_init(self):
        """Test ClosedBezierPath initialization."""
        torch = pytest.importorskip("torch")

        from aec_agent.mcp.tools.gemini_first.live_vectorization import (
            ClosedBezierPath,
        )

        path = ClosedBezierPath(
            num_control_points=8,
            image_size=(100, 100),
            init_scale=0.3,
            device=torch.device("cpu"),
        )

        assert path.num_control_points == 8
        assert path.control_points.shape == (8, 2)

    def test_closed_bezier_path_sample_points(self):
        """Test point sampling along path."""
        torch = pytest.importorskip("torch")

        from aec_agent.mcp.tools.gemini_first.live_vectorization import (
            ClosedBezierPath,
        )

        path = ClosedBezierPath(
            num_control_points=6,
            image_size=(100, 100),
            device=torch.device("cpu"),
        )

        points = path.sample_points(50)
        assert points.shape[0] > 0
        assert points.shape[1] == 2

    def test_closed_bezier_path_render(self):
        """Test path rendering."""
        torch = pytest.importorskip("torch")

        from aec_agent.mcp.tools.gemini_first.live_vectorization import (
            ClosedBezierPath,
        )

        path = ClosedBezierPath(
            num_control_points=6,
            image_size=(50, 50),
            device=torch.device("cpu"),
        )

        rendered = path.render()
        assert rendered.shape == (50, 50)
        assert rendered.min() >= 0.0
        assert rendered.max() <= 1.0

    def test_closed_bezier_path_get_vector_path(self):
        """Test extracting VectorPath."""
        torch = pytest.importorskip("torch")

        from aec_agent.mcp.tools.gemini_first.live_vectorization import (
            ClosedBezierPath,
        )

        path = ClosedBezierPath(
            num_control_points=8,
            image_size=(100, 100),
            device=torch.device("cpu"),
        )

        vector_path = path.get_vector_path()
        assert isinstance(vector_path, VectorPath)
        assert len(vector_path.control_points) == 8

    def test_live_layer_init(self):
        """Test LIVELayer initialization."""
        torch = pytest.importorskip("torch")

        from aec_agent.mcp.tools.gemini_first.live_vectorization import (
            LIVELayer,
        )

        layer = LIVELayer(
            num_paths=3,
            control_points_per_path=6,
            image_size=(100, 100),
            device=torch.device("cpu"),
        )

        assert layer.num_paths == 3
        assert len(layer.paths) == 3

    def test_live_layer_render(self):
        """Test layer rendering."""
        torch = pytest.importorskip("torch")

        from aec_agent.mcp.tools.gemini_first.live_vectorization import (
            LIVELayer,
        )

        layer = LIVELayer(
            num_paths=2,
            control_points_per_path=6,
            image_size=(50, 50),
            device=torch.device("cpu"),
        )

        result, alpha = layer.render()
        assert result.shape == (50, 50, 3)
        assert alpha.shape == (50, 50)

    def test_live_layer_get_vector_layer(self):
        """Test extracting VectorLayer."""
        torch = pytest.importorskip("torch")

        from aec_agent.mcp.tools.gemini_first.live_vectorization import (
            LIVELayer,
        )

        layer = LIVELayer(
            num_paths=2,
            control_points_per_path=8,
            image_size=(100, 100),
            device=torch.device("cpu"),
        )

        vector_layer = layer.get_vector_layer(0)
        assert isinstance(vector_layer, VectorLayer)
        assert vector_layer.layer_index == 0
        assert len(vector_layer.paths) == 2


@pytest.mark.requires_phase_c
class TestLIVEVectorizerIntegration:
    """Integration tests for LIVE vectorizer."""

    def test_vectorizer_singleton(self):
        """Test vectorizer singleton pattern."""
        torch = pytest.importorskip("torch")

        from aec_agent.mcp.tools.gemini_first.live_vectorization import (
            LIVEVectorizer,
        )

        config = LIVEConfig(gpu_id=-1)
        vec1 = LIVEVectorizer.get_instance(config)
        vec2 = LIVEVectorizer.get_instance()

        assert vec1 is vec2

        # Reset for other tests
        LIVEVectorizer.reset_instance()

    def test_vectorizer_device_selection(self):
        """Test device selection logic."""
        torch = pytest.importorskip("torch")

        from aec_agent.mcp.tools.gemini_first.live_vectorization import (
            LIVEVectorizer,
        )

        # Force CPU
        config = LIVEConfig(gpu_id=-1)
        vectorizer = LIVEVectorizer(config)
        assert vectorizer.device == torch.device("cpu")
