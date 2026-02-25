"""
Unit tests for Bezier Splatting (Phase C.2).

Tests the differentiable curve fitting module.
"""

import pytest
import numpy as np
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import tempfile

from aec_agent.mcp.tools.gemini_first.bezier_splatting import (
    CurveType,
    BezierSplattingConfig,
    OptimizedCurve,
    BezierSplattingResult,
    is_bezier_splatting_available,
)


class TestCurveType:
    """Tests for CurveType enum."""

    def test_curve_types_exist(self):
        """Verify all expected curve types exist."""
        assert CurveType.LINEAR.value == "linear"
        assert CurveType.QUADRATIC.value == "quadratic"
        assert CurveType.CUBIC.value == "cubic"

    def test_curve_type_from_string(self):
        """Test creating CurveType from string."""
        assert CurveType("linear") == CurveType.LINEAR
        assert CurveType("quadratic") == CurveType.QUADRATIC
        assert CurveType("cubic") == CurveType.CUBIC


class TestBezierSplattingConfig:
    """Tests for BezierSplattingConfig dataclass."""

    def test_default_config(self):
        """Test default configuration values."""
        config = BezierSplattingConfig()
        assert config.num_curves == 64
        assert config.curve_type == CurveType.CUBIC
        assert config.points_per_curve == 20
        assert config.iterations == 500
        assert config.learning_rate == 0.01
        assert config.gaussian_sigma == 1.0
        assert config.stroke_width == 2.0
        assert config.adaptive_control is True

    def test_custom_config(self):
        """Test custom configuration values."""
        config = BezierSplattingConfig(
            num_curves=128,
            curve_type=CurveType.QUADRATIC,
            iterations=1000,
            learning_rate=0.001,
            gpu_id=-1,
        )
        assert config.num_curves == 128
        assert config.curve_type == CurveType.QUADRATIC
        assert config.iterations == 1000
        assert config.learning_rate == 0.001
        assert config.gpu_id == -1

    def test_to_dict(self):
        """Test conversion to dictionary."""
        config = BezierSplattingConfig(
            num_curves=32,
            iterations=200,
        )
        result = config.to_dict()
        assert isinstance(result, dict)
        assert result["num_curves"] == 32
        assert result["iterations"] == 200
        assert result["curve_type"] == "cubic"


class TestOptimizedCurve:
    """Tests for OptimizedCurve dataclass."""

    def test_linear_curve_creation(self):
        """Test creating a linear curve."""
        curve = OptimizedCurve(
            control_points=[(0.0, 0.0), (100.0, 100.0)],
            curve_type=CurveType.LINEAR,
            stroke_width=2.0,
        )
        assert len(curve.control_points) == 2
        assert curve.curve_type == CurveType.LINEAR
        assert curve.stroke_width == 2.0
        assert curve.opacity == 1.0

    def test_cubic_curve_creation(self):
        """Test creating a cubic curve."""
        curve = OptimizedCurve(
            control_points=[
                (0.0, 0.0),
                (25.0, 50.0),
                (75.0, 50.0),
                (100.0, 0.0),
            ],
            curve_type=CurveType.CUBIC,
            stroke_width=1.5,
            color=(255, 0, 0),
            opacity=0.8,
        )
        assert len(curve.control_points) == 4
        assert curve.color == (255, 0, 0)
        assert curve.opacity == 0.8

    def test_linear_svg_path(self):
        """Test SVG path generation for linear curve."""
        curve = OptimizedCurve(
            control_points=[(10.0, 20.0), (30.0, 40.0)],
            curve_type=CurveType.LINEAR,
        )
        path = curve.to_svg_path()
        assert "M 10.000 20.000" in path
        assert "L 30.000 40.000" in path

    def test_quadratic_svg_path(self):
        """Test SVG path generation for quadratic curve."""
        curve = OptimizedCurve(
            control_points=[
                (0.0, 0.0),
                (50.0, 100.0),
                (100.0, 0.0),
            ],
            curve_type=CurveType.QUADRATIC,
        )
        path = curve.to_svg_path()
        assert "M 0.000 0.000" in path
        assert "Q" in path

    def test_cubic_svg_path(self):
        """Test SVG path generation for cubic curve."""
        curve = OptimizedCurve(
            control_points=[
                (0.0, 0.0),
                (33.0, 100.0),
                (66.0, 100.0),
                (100.0, 0.0),
            ],
            curve_type=CurveType.CUBIC,
        )
        path = curve.to_svg_path()
        assert "M 0.000 0.000" in path
        assert "C" in path

    def test_empty_control_points(self):
        """Test handling of empty control points."""
        curve = OptimizedCurve(
            control_points=[],
            curve_type=CurveType.LINEAR,
        )
        assert curve.to_svg_path() == ""

    def test_to_dict(self):
        """Test conversion to dictionary."""
        curve = OptimizedCurve(
            control_points=[(0.0, 0.0), (100.0, 100.0)],
            curve_type=CurveType.LINEAR,
            stroke_width=3.0,
        )
        result = curve.to_dict()
        assert "control_points" in result
        assert "curve_type" in result
        assert "svg_path" in result
        assert result["stroke_width"] == 3.0


class TestBezierSplattingResult:
    """Tests for BezierSplattingResult dataclass."""

    def test_result_creation(self):
        """Test creating a result object."""
        curves = [
            OptimizedCurve(
                [(0, 0), (100, 100)],
                CurveType.LINEAR,
            ),
        ]
        config = BezierSplattingConfig()

        result = BezierSplattingResult(
            curves=curves,
            loss_history=[1.0, 0.5, 0.25],
            final_loss=0.25,
            image_size=(640, 480),
            processing_time_ms=1500.0,
            config=config,
            device="cpu",
        )

        assert len(result.curves) == 1
        assert result.final_loss == 0.25
        assert result.image_size == (640, 480)
        assert len(result.loss_history) == 3

    def test_to_svg(self):
        """Test SVG generation."""
        curves = [
            OptimizedCurve(
                [(10, 10), (90, 90)],
                CurveType.LINEAR,
                stroke_width=2.0,
                color=(0, 0, 0),
            ),
        ]
        config = BezierSplattingConfig()

        result = BezierSplattingResult(
            curves=curves,
            loss_history=[0.1],
            final_loss=0.1,
            image_size=(100, 100),
            processing_time_ms=100.0,
            config=config,
        )

        svg = result.to_svg()
        assert '<svg xmlns="http://www.w3.org/2000/svg"' in svg
        assert 'width="100"' in svg
        assert 'height="100"' in svg
        assert "<path" in svg
        assert "</svg>" in svg

    def test_save_svg(self):
        """Test saving SVG to file."""
        curves = [
            OptimizedCurve(
                [(0, 0), (50, 50)],
                CurveType.LINEAR,
            ),
        ]
        config = BezierSplattingConfig()

        result = BezierSplattingResult(
            curves=curves,
            loss_history=[],
            final_loss=0.0,
            image_size=(100, 100),
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
        config = BezierSplattingConfig(num_curves=16)
        result = BezierSplattingResult(
            curves=[],
            loss_history=[1.0, 0.5],
            final_loss=0.5,
            image_size=(200, 200),
            processing_time_ms=200.0,
            config=config,
            device="cuda:0",
        )

        data = result.to_dict()
        assert data["num_curves"] == 0
        assert data["final_loss"] == 0.5
        assert data["device"] == "cuda:0"
        assert "config" in data


class TestAvailabilityCheck:
    """Tests for availability checking."""

    def test_availability_returns_bool(self):
        """Test that availability check returns a boolean."""
        result = is_bezier_splatting_available()
        assert isinstance(result, bool)


@pytest.mark.requires_phase_c
class TestBezierSplattingWithTorch:
    """Tests requiring PyTorch."""

    def test_bezier_curve_module_init(self):
        """Test BezierCurveModule initialization."""
        torch = pytest.importorskip("torch")

        from aec_agent.mcp.tools.gemini_first.bezier_splatting import (
            BezierCurveModule,
        )

        module = BezierCurveModule(
            num_curves=8,
            curve_type=CurveType.CUBIC,
            image_size=(100, 100),
            points_per_curve=10,
            device=torch.device("cpu"),
        )

        assert module.num_curves == 8
        assert module.curve_type == CurveType.CUBIC
        assert module.control_points.shape == (8, 4, 2)

    def test_bezier_curve_module_evaluate(self):
        """Test curve evaluation."""
        torch = pytest.importorskip("torch")

        from aec_agent.mcp.tools.gemini_first.bezier_splatting import (
            BezierCurveModule,
        )

        module = BezierCurveModule(
            num_curves=4,
            curve_type=CurveType.CUBIC,
            image_size=(100, 100),
            points_per_curve=20,
            device=torch.device("cpu"),
        )

        points = module.evaluate_curves()
        assert points.shape == (4, 20, 2)

    def test_bezier_curve_module_render(self):
        """Test curve rendering."""
        torch = pytest.importorskip("torch")

        from aec_agent.mcp.tools.gemini_first.bezier_splatting import (
            BezierCurveModule,
        )

        module = BezierCurveModule(
            num_curves=2,
            curve_type=CurveType.LINEAR,
            image_size=(50, 50),
            points_per_curve=10,
            device=torch.device("cpu"),
        )

        rendered = module.render(sigma=1.0)
        assert rendered.shape == (50, 50)
        assert rendered.min() >= 0.0
        assert rendered.max() <= 1.0

    def test_bezier_curve_module_get_curves(self):
        """Test extracting OptimizedCurve objects."""
        torch = pytest.importorskip("torch")

        from aec_agent.mcp.tools.gemini_first.bezier_splatting import (
            BezierCurveModule,
        )

        module = BezierCurveModule(
            num_curves=4,
            curve_type=CurveType.QUADRATIC,
            image_size=(100, 100),
            device=torch.device("cpu"),
        )

        curves = module.get_curves()
        # Some curves may be pruned if opacity is too low
        assert isinstance(curves, list)
        for curve in curves:
            assert isinstance(curve, OptimizedCurve)
            assert curve.curve_type == CurveType.QUADRATIC


@pytest.mark.requires_phase_c
class TestBezierSplattingOptimizerIntegration:
    """Integration tests for optimizer."""

    def test_optimizer_singleton(self):
        """Test optimizer singleton pattern."""
        torch = pytest.importorskip("torch")

        from aec_agent.mcp.tools.gemini_first.bezier_splatting import (
            BezierSplattingOptimizer,
        )

        config = BezierSplattingConfig(gpu_id=-1)
        opt1 = BezierSplattingOptimizer.get_instance(config)
        opt2 = BezierSplattingOptimizer.get_instance()

        assert opt1 is opt2

        # Reset for other tests
        BezierSplattingOptimizer.reset_instance()

    def test_optimizer_device_selection(self):
        """Test device selection logic."""
        torch = pytest.importorskip("torch")

        from aec_agent.mcp.tools.gemini_first.bezier_splatting import (
            BezierSplattingOptimizer,
        )

        # Force CPU
        config = BezierSplattingConfig(gpu_id=-1)
        optimizer = BezierSplattingOptimizer(config)
        assert optimizer.device == torch.device("cpu")
