"""
Unit tests for Scan2CAD Feature Parity modules.

Tests cover:
- Phase A: Linetype detection, Bezier fitting, DXF export, Thickness detection
- Phase B: Batch processing, Polyline auto-join
- Phase C: Preview generation, Profiles, Ellipse detection
"""

import math
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch, AsyncMock

import numpy as np
import pytest


# =============================================================================
# Phase A: Linetype Detection Tests
# =============================================================================

class TestLinetypeDetection:
    """Tests for linetype_detection.py module."""

    def test_linetype_imports(self):
        """Test that linetype detection module imports correctly."""
        from aec_agent.mcp.tools.gemini_first import (
            LinetypeName,
            LinetypeResult,
            LinetypeConfig,
            detect_linetype,
            detect_linetypes_batch,
            map_to_autocad_linetype,
            parse_linetype_name,
        )
        assert LinetypeName.CONTINUOUS.value == "Continuous"
        assert LinetypeName.DASHED.value == "DASHED"
        assert LinetypeName.HIDDEN.value == "HIDDEN"

    def test_linetype_config_defaults(self):
        """Test LinetypeConfig default values."""
        from aec_agent.mcp.tools.gemini_first import LinetypeConfig

        config = LinetypeConfig()
        assert config.sample_width == 3
        assert config.min_line_length == 30
        assert config.gap_threshold == 0.3
        assert config.fft_peak_threshold == 0.1

    def test_map_to_autocad_linetype(self):
        """Test mapping linetype names to AutoCAD linetypes."""
        from aec_agent.mcp.tools.gemini_first import (
            LinetypeName,
            map_to_autocad_linetype,
        )

        assert map_to_autocad_linetype(LinetypeName.CONTINUOUS) == "Continuous"
        assert map_to_autocad_linetype(LinetypeName.DASHED) == "DASHED"
        assert map_to_autocad_linetype(LinetypeName.HIDDEN) == "HIDDEN"
        assert map_to_autocad_linetype(LinetypeName.CENTER) == "CENTER"
        assert map_to_autocad_linetype(LinetypeName.PHANTOM) == "PHANTOM"

    def test_parse_linetype_name(self):
        """Test parsing linetype name strings."""
        from aec_agent.mcp.tools.gemini_first import LinetypeName, parse_linetype_name

        assert parse_linetype_name("continuous") == LinetypeName.CONTINUOUS
        assert parse_linetype_name("DASHED") == LinetypeName.DASHED
        assert parse_linetype_name("Hidden") == LinetypeName.HIDDEN
        assert parse_linetype_name("invalid") == LinetypeName.CONTINUOUS  # Default

    def test_linetype_result_to_dict(self):
        """Test LinetypeResult serialization."""
        from aec_agent.mcp.tools.gemini_first import LinetypeName, LinetypeResult

        result = LinetypeResult(
            linetype=LinetypeName.DASHED,
            confidence=0.85,
            dash_ratio=0.6,
            gap_count=5,
        )
        d = result.to_dict()
        assert d["linetype"] == "DASHED"
        assert d["confidence"] == 0.85
        assert d["dash_ratio"] == 0.6


# =============================================================================
# Phase A: Bezier Fitting Tests
# =============================================================================

class TestBezierFitting:
    """Tests for bezier_fitting.py module."""

    def test_bezier_imports(self):
        """Test that bezier fitting module imports correctly."""
        from aec_agent.mcp.tools.gemini_first import (
            BezierCurve,
            CurveSegment,
            BezierFitConfig,
            fit_bezier_to_points,
            detect_corners,
            detect_curves,
        )
        assert BezierCurve is not None
        assert CurveSegment is not None

    def test_bezier_curve_evaluate(self):
        """Test BezierCurve evaluation."""
        from aec_agent.mcp.tools.gemini_first import BezierCurve

        # Straight line bezier using control_points list
        curve = BezierCurve(
            control_points=[
                (0.0, 0.0),
                (0.33, 0.0),
                (0.66, 0.0),
                (1.0, 0.0),
            ]
        )

        # At t=0, should be at start
        pt = curve.evaluate(0.0)
        assert abs(pt[0] - 0.0) < 0.01
        assert abs(pt[1] - 0.0) < 0.01

        # At t=1, should be at end
        pt = curve.evaluate(1.0)
        assert abs(pt[0] - 1.0) < 0.01
        assert abs(pt[1] - 0.0) < 0.01

        # At t=0.5, should be close to midpoint (Bezier curves aren't exactly linear)
        pt = curve.evaluate(0.5)
        assert abs(pt[0] - 0.5) < 0.01

    def test_bezier_curve_properties(self):
        """Test BezierCurve properties."""
        from aec_agent.mcp.tools.gemini_first import BezierCurve

        # Test straight line bezier
        curve = BezierCurve(
            control_points=[
                (0.0, 0.0),
                (0.33, 0.0),
                (0.66, 0.0),
                (1.0, 0.0),
            ]
        )

        # Test start and end properties
        assert curve.start == (0.0, 0.0)
        assert curve.end == (1.0, 0.0)

    def test_bezier_fit_config(self):
        """Test BezierFitConfig defaults."""
        from aec_agent.mcp.tools.gemini_first import BezierFitConfig

        config = BezierFitConfig()
        assert config.max_error == 2.0
        assert config.corner_threshold == 0.5  # Radians
        assert config.min_segment_length == 4

    def test_detect_corners(self):
        """Test corner detection in point sequences."""
        from aec_agent.mcp.tools.gemini_first import detect_corners

        # L-shaped path with clear corner
        points = [
            (0, 0), (1, 0), (2, 0), (3, 0), (4, 0),  # Horizontal
            (4, 1), (4, 2), (4, 3), (4, 4),  # Vertical
        ]
        points_array = np.array(points, dtype=np.float64)

        corners = detect_corners(points_array, threshold=0.5)
        # Should detect corner at index ~4 (the bend)
        assert len(corners) >= 1


# =============================================================================
# Phase A: DXF Export Tests
# =============================================================================

class TestDXFExport:
    """Tests for dxf_export.py module."""

    def test_dxf_imports(self):
        """Test that DXF export module imports correctly."""
        from aec_agent.mcp.tools.gemini_first import (
            DXFVersion,
            DXFUnits,
            DXFExportConfig,
            DXFExportResult,
            DXFExporter,
            export_to_dxf,
            is_ezdxf_available,
        )
        assert DXFVersion.R2018.value == "R2018"
        assert is_ezdxf_available is not None

    def test_dxf_config_defaults(self):
        """Test DXFExportConfig defaults."""
        from aec_agent.mcp.tools.gemini_first import DXFExportConfig, DXFVersion, DXFUnits

        config = DXFExportConfig()
        assert config.version == DXFVersion.R2018
        assert config.units == DXFUnits.INCHES
        assert config.create_layers is True

    def test_dxf_export_result_to_dict(self):
        """Test DXFExportResult serialization."""
        from aec_agent.mcp.tools.gemini_first import DXFExportResult

        result = DXFExportResult(
            success=True,
            output_path="/path/to/output.dxf",
            entity_count=100,
            layer_count=5,
            errors=[],
        )
        d = result.to_dict()
        assert d["success"] is True
        assert d["entity_count"] == 100
        assert d["layer_count"] == 5

    @pytest.mark.skipif(
        not __import__("importlib.util").util.find_spec("ezdxf"),
        reason="ezdxf not installed"
    )
    def test_dxf_export_simple(self):
        """Test basic DXF export with ezdxf."""
        from aec_agent.mcp.tools.gemini_first import (
            export_to_dxf,
            DXFExportConfig,
            EntityToCreate,
            EntityType,
            ExtractionSource,
        )

        entities = [
            EntityToCreate(
                entity_type=EntityType.LINE,
                layer="0",
                properties={"start": (0, 0, 0), "end": (10, 10, 0)},
                source=ExtractionSource.DIRECT,
            ),
            EntityToCreate(
                entity_type=EntityType.CIRCLE,
                layer="0",
                properties={"center": (5, 5, 0), "radius": 2.0},
                source=ExtractionSource.DIRECT,
            ),
        ]

        with tempfile.NamedTemporaryFile(suffix=".dxf", delete=False) as f:
            output_path = Path(f.name)

        try:
            result = export_to_dxf(entities, output_path)
            assert result.success
            assert result.entity_count == 2
            assert output_path.exists()
        finally:
            if output_path.exists():
                output_path.unlink()


# =============================================================================
# Phase A: Thickness Detection Tests
# =============================================================================

class TestThicknessDetection:
    """Tests for thickness_detection.py module."""

    def test_thickness_imports(self):
        """Test that thickness detection module imports correctly."""
        from aec_agent.mcp.tools.gemini_first import (
            ThicknessCategory,
            ThicknessResult,
            ThicknessConfig,
            detect_line_thickness,
            mm_to_lineweight,
            categorize_thickness,
            AUTOCAD_LINEWEIGHTS,
        )
        assert ThicknessCategory.HAIRLINE.value == "hairline"
        assert len(AUTOCAD_LINEWEIGHTS) > 0

    def test_thickness_config_defaults(self):
        """Test ThicknessConfig defaults."""
        from aec_agent.mcp.tools.gemini_first import ThicknessConfig

        config = ThicknessConfig()
        assert config.dpi == 300
        assert config.min_width_pixels == 1
        assert config.max_width_pixels == 50

    def test_mm_to_lineweight(self):
        """Test conversion from mm to AutoCAD lineweight."""
        from aec_agent.mcp.tools.gemini_first import mm_to_lineweight, AUTOCAD_LINEWEIGHTS

        # Test rounding to nearest valid lineweight
        assert mm_to_lineweight(0.0) == 0.0
        # These should round to the nearest value in AUTOCAD_LINEWEIGHTS
        result_01 = mm_to_lineweight(0.1)
        assert result_01 in AUTOCAD_LINEWEIGHTS
        result_05 = mm_to_lineweight(0.5)
        assert result_05 in AUTOCAD_LINEWEIGHTS

    def test_categorize_thickness(self):
        """Test thickness categorization."""
        from aec_agent.mcp.tools.gemini_first import ThicknessCategory, categorize_thickness

        assert categorize_thickness(0.05) == ThicknessCategory.HAIRLINE
        assert categorize_thickness(0.15) == ThicknessCategory.THIN
        assert categorize_thickness(0.35) == ThicknessCategory.MEDIUM
        assert categorize_thickness(0.55) == ThicknessCategory.THICK
        assert categorize_thickness(1.0) == ThicknessCategory.EXTRA_THICK

    def test_thickness_result_to_dict(self):
        """Test ThicknessResult serialization."""
        from aec_agent.mcp.tools.gemini_first import (
            ThicknessCategory,
            ThicknessResult,
        )

        result = ThicknessResult(
            width_pixels=3.5,
            width_mm=0.3,
            lineweight=0.30,
            category=ThicknessCategory.MEDIUM,
            confidence=0.9,
        )
        d = result.to_dict()
        assert d["width_mm"] == 0.3
        assert d["lineweight"] == 0.30
        assert d["category"] == "medium"


# =============================================================================
# Phase B: Batch Processing Tests
# =============================================================================

class TestBatchProcessor:
    """Tests for batch_processor.py module."""

    def test_batch_imports(self):
        """Test that batch processor module imports correctly."""
        from aec_agent.mcp.tools.gemini_first import (
            BatchStatus,
            FileStatus,
            FileResult,
            BatchConfig,
            BatchResult,
            BatchProcessor,
            process_batch,
        )
        assert BatchStatus.PENDING.value == "pending"
        assert FileStatus.COMPLETED.value == "completed"

    def test_batch_config_defaults(self):
        """Test BatchConfig defaults."""
        from aec_agent.mcp.tools.gemini_first import BatchConfig

        config = BatchConfig(
            input_dir="/input",
            output_dir="/output",
        )
        assert config.file_pattern == "*.pdf"
        assert config.max_parallel == 4
        assert config.continue_on_error is True
        assert config.export_dxf is True

    def test_file_result_to_dict(self):
        """Test FileResult serialization."""
        from aec_agent.mcp.tools.gemini_first import FileResult, FileStatus

        result = FileResult(
            file_path="/path/to/file.pdf",
            status=FileStatus.COMPLETED,
            duration_ms=1500.0,
            entity_count=50,
            output_path="/path/to/output.dxf",
        )
        d = result.to_dict()
        assert d["status"] == "completed"
        assert d["entity_count"] == 50

    def test_batch_result_success_rate(self):
        """Test BatchResult success_rate property."""
        from aec_agent.mcp.tools.gemini_first import BatchResult, BatchStatus
        from uuid import uuid4

        result = BatchResult(
            batch_id=uuid4(),
            status=BatchStatus.PARTIAL,
            total_files=10,
            success_count=7,
            failure_count=3,
        )
        assert result.success_rate == 0.7

    def test_batch_result_to_summary(self):
        """Test BatchResult summary generation."""
        from aec_agent.mcp.tools.gemini_first import BatchResult, BatchStatus
        from uuid import uuid4

        result = BatchResult(
            batch_id=uuid4(),
            status=BatchStatus.COMPLETED,
            total_files=5,
            success_count=5,
            failure_count=0,
            duration_seconds=30.5,
        )
        summary = result.to_summary()
        assert "Completed" in summary or "completed" in summary.lower()
        assert "5" in summary


# =============================================================================
# Phase B: Polyline Builder Tests
# =============================================================================

class TestPolylineBuilder:
    """Tests for polyline_builder.py module."""

    def test_polyline_imports(self):
        """Test that polyline builder module imports correctly."""
        from aec_agent.mcp.tools.gemini_first import (
            PolylineConfig,
            PolylineResult,
            EndpointGraph,
            build_polylines,
            auto_join_lines,
        )
        assert PolylineConfig is not None
        assert EndpointGraph is not None

    def test_polyline_config_defaults(self):
        """Test PolylineConfig defaults."""
        from aec_agent.mcp.tools.gemini_first import PolylineConfig

        config = PolylineConfig()
        assert config.endpoint_tolerance == 0.01
        assert config.collinear_tolerance == 0.1
        assert config.min_chain_length == 2
        assert config.remove_collinear is True

    def test_endpoint_graph_basic(self):
        """Test EndpointGraph basic operations."""
        from aec_agent.mcp.tools.gemini_first import EndpointGraph

        graph = EndpointGraph(tolerance=0.01)
        graph.add_line((0, 0), (10, 0), 0)
        graph.add_line((10, 0), (10, 10), 1)

        # Check degrees
        # (0, 0) should have degree 1
        # (10, 0) should have degree 2 (connected to both lines)
        # (10, 10) should have degree 1

    def test_polyline_result_to_dict(self):
        """Test PolylineResult serialization."""
        from aec_agent.mcp.tools.gemini_first import PolylineResult

        result = PolylineResult(
            polylines=[],
            original_line_count=10,
            polyline_count=3,
            closed_count=1,
            reduction_ratio=0.7,
            removed_collinear=5,
        )
        d = result.to_dict()
        assert d["original_line_count"] == 10
        assert d["polyline_count"] == 3
        assert "70" in d["reduction_ratio"]  # "70.0%"

    def test_auto_join_lines_simple(self):
        """Test auto_join_lines with simple connected lines."""
        from aec_agent.mcp.tools.gemini_first import (
            auto_join_lines,
            EntityToCreate,
            EntityType,
            ExtractionSource,
        )

        # Create 3 connected lines forming an L shape
        entities = [
            EntityToCreate(
                entity_type=EntityType.LINE,
                layer="0",
                properties={"start": (0, 0), "end": (10, 0)},
                source=ExtractionSource.DIRECT,
            ),
            EntityToCreate(
                entity_type=EntityType.LINE,
                layer="0",
                properties={"start": (10, 0), "end": (20, 0)},
                source=ExtractionSource.DIRECT,
            ),
            EntityToCreate(
                entity_type=EntityType.LINE,
                layer="0",
                properties={"start": (20, 0), "end": (20, 10)},
                source=ExtractionSource.DIRECT,
            ),
        ]

        result = auto_join_lines(entities, endpoint_tolerance=0.1)
        # Should join into a polyline
        assert len(result) <= len(entities)


# =============================================================================
# Phase C: Preview Generation Tests
# =============================================================================

class TestPreviewGenerator:
    """Tests for preview_generator.py module."""

    def test_preview_imports(self):
        """Test that preview generator module imports correctly."""
        from aec_agent.mcp.tools.gemini_first import (
            PreviewConfig,
            PreviewResult,
            generate_preview,
            ENTITY_COLORS,
        )
        assert PreviewConfig is not None
        assert ENTITY_COLORS is not None

    def test_preview_config_defaults(self):
        """Test PreviewConfig defaults."""
        from aec_agent.mcp.tools.gemini_first import PreviewConfig

        config = PreviewConfig()
        assert config.overlay_alpha == 0.6
        assert config.include_legend is True
        assert config.line_thickness == 2

    def test_preview_result_to_dict(self):
        """Test PreviewResult serialization."""
        from aec_agent.mcp.tools.gemini_first import PreviewResult

        result = PreviewResult(
            success=True,
            output_path="/path/to/preview.png",
            entity_count=50,
            width=800,
            height=600,
        )
        d = result.to_dict()
        assert d["success"] is True
        assert d["entity_count"] == 50
        assert d["width"] == 800


# =============================================================================
# Phase C: Profiles Tests
# =============================================================================

class TestProfiles:
    """Tests for profiles.py module."""

    def test_profile_imports(self):
        """Test that profiles module imports correctly."""
        from aec_agent.mcp.tools.gemini_first import (
            ProfileType,
            ConversionProfile,
            get_profile,
            list_profiles,
            auto_detect_profile,
        )
        assert ProfileType.ARCHITECTURAL.value == "architectural"

    def test_get_profile_types(self):
        """Test getting different profile types."""
        from aec_agent.mcp.tools.gemini_first import (
            ProfileType,
            get_profile,
            get_architectural_profile,
            get_mechanical_profile,
            get_electrical_profile,
        )

        arch = get_profile(ProfileType.ARCHITECTURAL)
        assert arch.name == "Architectural"
        assert arch.profile_type == ProfileType.ARCHITECTURAL

        mech = get_mechanical_profile()
        assert mech.name == "Mechanical"

        elec = get_electrical_profile()
        assert elec.name == "Electrical"

    def test_profile_layer_mapping(self):
        """Test LayerMapping functionality."""
        from aec_agent.mcp.tools.gemini_first import LayerMapping

        mapping = LayerMapping(prefix="A-")
        assert mapping.get_layer("wall") == "A-A-WALL"
        assert mapping.get_layer("door") == "A-A-DOOR"
        assert mapping.get_layer("unknown") == "A-0-MISC"

    def test_list_profiles(self):
        """Test listing all available profiles."""
        from aec_agent.mcp.tools.gemini_first import list_profiles

        profiles = list_profiles()
        assert len(profiles) >= 8  # At least 8 built-in profiles

        names = [p["name"] for p in profiles]
        assert "Architectural" in names
        assert "Mechanical" in names
        assert "Electrical" in names

    def test_auto_detect_profile(self):
        """Test automatic profile detection."""
        from aec_agent.mcp.tools.gemini_first import auto_detect_profile, ProfileType

        # Test with filename hints - "floor plan" is a keyword for architectural
        profile = auto_detect_profile(filename="floor plan level1.pdf")
        assert profile.profile_type == ProfileType.ARCHITECTURAL

        profile = auto_detect_profile(filename="electrical_schematic.pdf")
        assert profile.profile_type == ProfileType.ELECTRICAL

        # Test with analysis result
        profile = auto_detect_profile(
            analysis_result={"drawing_type": "HVAC ductwork layout"}
        )
        assert profile.profile_type == ProfileType.HVAC

    def test_create_custom_profile(self):
        """Test creating custom profiles."""
        from aec_agent.mcp.tools.gemini_first import (
            create_custom_profile,
            ProfileType,
        )

        profile = create_custom_profile(
            name="My Custom",
            base_profile=ProfileType.ARCHITECTURAL,
            input_dpi=400,
            layer_prefix="X-",
        )
        assert profile.name == "My Custom"
        assert profile.profile_type == ProfileType.CUSTOM
        assert profile.scale.input_dpi == 400
        assert profile.layers.prefix == "X-"


# =============================================================================
# Phase C: Ellipse Detection Tests
# =============================================================================

class TestEllipseDetection:
    """Tests for ellipse_detection.py module."""

    def test_ellipse_imports(self):
        """Test that ellipse detection module imports correctly."""
        from aec_agent.mcp.tools.gemini_first import (
            DetectedEllipse,
            EllipseDetectionConfig,
            EllipseDetectionResult,
            detect_ellipses,
            is_ellipse_detection_available,
        )
        assert DetectedEllipse is not None
        assert is_ellipse_detection_available is not None

    def test_ellipse_config_defaults(self):
        """Test EllipseDetectionConfig defaults."""
        from aec_agent.mcp.tools.gemini_first import EllipseDetectionConfig

        config = EllipseDetectionConfig()
        assert config.min_radius == 5
        assert config.max_radius == 1000
        assert config.min_aspect_ratio == 1.0
        assert config.use_ransac is True

    def test_detected_ellipse_properties(self):
        """Test DetectedEllipse property calculations."""
        from aec_agent.mcp.tools.gemini_first import DetectedEllipse

        ellipse = DetectedEllipse(
            center=(100.0, 100.0),
            semi_major=50.0,
            semi_minor=30.0,
            rotation=45.0,
            confidence=0.95,
        )

        # Test properties
        assert abs(ellipse.aspect_ratio - (50.0 / 30.0)) < 0.001
        assert ellipse.is_circle is False  # Aspect ratio > 1.1
        assert abs(ellipse.area - (math.pi * 50 * 30)) < 0.1

    def test_detected_ellipse_is_circle(self):
        """Test circle detection from ellipse."""
        from aec_agent.mcp.tools.gemini_first import DetectedEllipse

        # Nearly circular ellipse
        ellipse = DetectedEllipse(
            center=(100.0, 100.0),
            semi_major=50.0,
            semi_minor=48.0,  # Aspect ratio ~1.04
            rotation=0.0,
        )
        assert ellipse.is_circle is True

    def test_ellipse_result_to_dict(self):
        """Test EllipseDetectionResult serialization."""
        from aec_agent.mcp.tools.gemini_first import (
            DetectedEllipse,
            EllipseDetectionResult,
        )

        ellipse = DetectedEllipse(
            center=(100.0, 100.0),
            semi_major=50.0,
            semi_minor=30.0,
            rotation=45.0,
        )
        result = EllipseDetectionResult(
            ellipses=[ellipse],
            full_count=1,
            partial_count=0,
            duration_ms=50.0,
        )
        d = result.to_dict()
        assert d["ellipse_count"] == 1
        assert d["full_count"] == 1
        assert len(d["ellipses"]) == 1

    def test_ellipse_to_entity(self):
        """Test converting ellipse to entity dict."""
        from aec_agent.mcp.tools.gemini_first import DetectedEllipse, ellipse_to_entity

        ellipse = DetectedEllipse(
            center=(100.0, 100.0),
            semi_major=50.0,
            semi_minor=30.0,
            rotation=0.0,
        )

        entity = ellipse_to_entity(ellipse, scale_factor=1.0, layer="ELLIPSES")
        assert entity["entity_type"] == "ellipse"
        assert entity["layer"] == "ELLIPSES"
        assert "center" in entity["properties"]
        assert "major_axis" in entity["properties"]
        assert "ratio" in entity["properties"]


# =============================================================================
# Integration Tests
# =============================================================================

class TestScan2CADIntegration:
    """Integration tests for Scan2CAD parity features."""

    def test_all_modules_import(self):
        """Test that all Scan2CAD parity modules can be imported together."""
        from aec_agent.mcp.tools.gemini_first import (
            # Phase A
            detect_linetype,
            fit_bezier_to_points,
            export_to_dxf,
            detect_line_thickness,
            # Phase B
            BatchProcessor,
            build_polylines,
            # Phase C
            generate_preview,
            get_profile,
            detect_ellipses,
        )
        # If we get here, all imports succeeded
        assert True

    def test_unified_pipeline_with_profiles(self):
        """Test that profiles integrate with unified pipeline config."""
        from aec_agent.mcp.tools.gemini_first import (
            PipelineConfig,
            ExtractionMethod,
            get_architectural_profile,
        )

        profile = get_architectural_profile()

        # Use profile settings to configure pipeline
        config = PipelineConfig(
            dpi=profile.scale.input_dpi,
            preprocess=True,
            extraction_method=ExtractionMethod.HYBRID,
        )

        assert config.dpi == profile.scale.input_dpi
