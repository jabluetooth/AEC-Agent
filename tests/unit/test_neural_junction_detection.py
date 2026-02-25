"""
Unit tests for Neural Junction Detection (Phase C.1).

Tests the HAWP-based junction detection module.
"""

import pytest
import numpy as np
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

from aec_agent.mcp.tools.gemini_first.neural_junction_detection import (
    JunctionType,
    JunctionDetectionConfig,
    DetectedJunction,
    DetectedWireframeLine,
    JunctionDetectionResult,
    is_junction_detection_available,
    snap_endpoints_to_junctions,
)


class TestJunctionType:
    """Tests for JunctionType enum."""

    def test_junction_types_exist(self):
        """Verify all expected junction types exist."""
        expected_types = [
            "T_JUNCTION", "L_JUNCTION", "X_JUNCTION", "Y_JUNCTION",
            "CORNER", "ENDPOINT", "CROSSING", "TANGENT",
            "PARALLEL_END", "COMPLEX", "UNKNOWN"
        ]
        for jtype in expected_types:
            assert hasattr(JunctionType, jtype), f"Missing junction type: {jtype}"

    def test_junction_type_values(self):
        """Test that junction types have string values."""
        assert JunctionType.T_JUNCTION.value == "T"
        assert JunctionType.CORNER.value == "corner"
        assert JunctionType.ENDPOINT.value == "endpoint"


class TestJunctionDetectionConfig:
    """Tests for JunctionDetectionConfig dataclass."""

    def test_default_config(self):
        """Test default configuration values."""
        config = JunctionDetectionConfig()
        assert config.model_name == "hawpv3"
        assert config.confidence_threshold == 0.5
        assert config.nms_threshold == 0.3
        assert config.max_junctions == 1000
        assert config.gpu_id is None

    def test_custom_config(self):
        """Test custom configuration values."""
        config = JunctionDetectionConfig(
            model_name="custom_model",
            confidence_threshold=0.8,
            gpu_id=-1,
        )
        assert config.model_name == "custom_model"
        assert config.confidence_threshold == 0.8
        assert config.gpu_id == -1

    def test_config_attributes(self):
        """Test config has expected attributes."""
        config = JunctionDetectionConfig(confidence_threshold=0.7)
        assert hasattr(config, "model_name")
        assert hasattr(config, "confidence_threshold")
        assert hasattr(config, "line_threshold")
        assert hasattr(config, "nms_threshold")
        assert config.confidence_threshold == 0.7


class TestDetectedJunction:
    """Tests for DetectedJunction dataclass."""

    def test_junction_creation(self):
        """Test creating a detected junction."""
        junction = DetectedJunction(
            position=(100.0, 200.0),
            junction_type=JunctionType.T_JUNCTION,
            confidence=0.95,
        )
        assert junction.position == (100.0, 200.0)
        assert junction.junction_type == JunctionType.T_JUNCTION
        assert junction.confidence == 0.95
        assert junction.connected_line_indices == []

    def test_junction_with_connected_lines(self):
        """Test junction with connected line indices."""
        junction = DetectedJunction(
            position=(50.0, 50.0),
            junction_type=JunctionType.X_JUNCTION,
            confidence=0.88,
            connected_line_indices=[0, 1, 2, 3],
        )
        assert len(junction.connected_line_indices) == 4

    def test_junction_to_dict(self):
        """Test conversion to dictionary."""
        junction = DetectedJunction(
            position=(10.0, 20.0),
            junction_type=JunctionType.CORNER,
            confidence=0.75,
        )
        result = junction.to_dict()
        assert result["position"] == [10.0, 20.0]  # Returns list, not tuple
        assert result["type"] == "corner"  # Key is "type", not "junction_type"
        assert result["confidence"] == 0.75


class TestDetectedWireframeLine:
    """Tests for DetectedWireframeLine dataclass."""

    def test_line_creation(self):
        """Test creating a wireframe line."""
        line = DetectedWireframeLine(
            start=(0.0, 0.0),
            end=(100.0, 100.0),
            confidence=0.9,
        )
        assert line.start == (0.0, 0.0)
        assert line.end == (100.0, 100.0)
        assert line.confidence == 0.9

    def test_line_length(self):
        """Test line length calculation."""
        line = DetectedWireframeLine(
            start=(0.0, 0.0),
            end=(3.0, 4.0),
            confidence=1.0,
        )
        assert abs(line.length - 5.0) < 0.001

    def test_line_to_dict(self):
        """Test conversion to dictionary."""
        line = DetectedWireframeLine(
            start=(10.0, 20.0),
            end=(30.0, 40.0),
            confidence=0.85,
        )
        result = line.to_dict()
        assert "start" in result
        assert "end" in result
        assert "length" in result


class TestJunctionDetectionResult:
    """Tests for JunctionDetectionResult dataclass."""

    def test_result_creation(self):
        """Test creating a detection result."""
        junctions = [
            DetectedJunction((10, 10), JunctionType.T_JUNCTION, 0.9),
            DetectedJunction((50, 50), JunctionType.L_JUNCTION, 0.8),
        ]
        lines = [
            DetectedWireframeLine((0, 0), (100, 100), 0.95),
        ]
        result = JunctionDetectionResult(
            junctions=junctions,
            lines=lines,
            image_width=640,
            image_height=480,
            processing_time_ms=150.0,
        )
        assert len(result.junctions) == 2
        assert len(result.lines) == 1
        assert result.image_width == 640
        assert result.image_height == 480
        assert result.processing_time_ms == 150.0

    def test_result_to_dict(self):
        """Test conversion to dictionary."""
        result = JunctionDetectionResult(
            junctions=[],
            lines=[],
            image_width=800,
            image_height=600,
            processing_time_ms=100.0,
        )
        data = result.to_dict()
        assert data["num_junctions"] == 0
        assert data["num_lines"] == 0
        assert data["image_size"] == (800, 600)  # Returned as tuple


class TestSnapEndpointsToJunctions:
    """Tests for endpoint snapping utility."""

    def test_snap_to_nearby_junction(self):
        """Test that endpoints snap to nearby junctions."""
        # Create line with endpoint near a junction
        # Lines are represented as ((start_x, start_y), (end_x, end_y))
        line = ((10.0, 10.0), (102.0, 102.0))

        # Create junction at (100, 100)
        junction = DetectedJunction(
            position=(100.0, 100.0),
            junction_type=JunctionType.T_JUNCTION,
            confidence=0.95,
        )

        # Snap with threshold of 5 pixels
        snapped = snap_endpoints_to_junctions(
            lines=[line],
            junctions=[junction],
            threshold=5.0,
        )

        # End point should be snapped to junction
        assert len(snapped) == 1
        start, end = snapped[0]
        assert abs(end[0] - 100.0) < 0.001
        assert abs(end[1] - 100.0) < 0.001

    def test_no_snap_when_too_far(self):
        """Test that endpoints don't snap when junction is too far."""
        line = ((0.0, 0.0), (50.0, 50.0))

        junction = DetectedJunction(
            position=(100.0, 100.0),
            junction_type=JunctionType.T_JUNCTION,
            confidence=0.95,
        )

        snapped = snap_endpoints_to_junctions(
            lines=[line],
            junctions=[junction],
            threshold=5.0,  # Junction is ~70 pixels away
        )

        # Should not be snapped
        start, end = snapped[0]
        assert end[0] == 50.0
        assert end[1] == 50.0


class TestAvailabilityCheck:
    """Tests for availability checking."""

    def test_availability_returns_bool(self):
        """Test that availability check returns a boolean."""
        result = is_junction_detection_available()
        assert isinstance(result, bool)


@pytest.mark.requires_phase_c
@pytest.mark.requires_gpu
class TestJunctionDetectorIntegration:
    """Integration tests requiring Phase C dependencies."""

    def test_detector_singleton(self):
        """Test detector singleton pattern."""
        pytest.importorskip("torch")

        from aec_agent.mcp.tools.gemini_first.neural_junction_detection import (
            JunctionDetector,
        )

        config = JunctionDetectionConfig(gpu_id=-1)  # Force CPU
        detector1 = JunctionDetector.get_instance(config)
        detector2 = JunctionDetector.get_instance()

        assert detector1 is detector2

        # Reset for other tests
        JunctionDetector.reset_instance()
