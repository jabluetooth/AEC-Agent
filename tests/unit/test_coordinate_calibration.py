"""
Unit tests for Phase 3: Coordinate Calibration.

Tests the gemini_first.coordinate_calibration module which converts
pixel coordinates to DWG units.
"""

import math
from pathlib import Path
from unittest.mock import MagicMock

import pytest

# Import module under test
from aec_agent.mcp.tools.gemini_first.coordinate_calibration import (
    ScaleCalibration,
    calibrate_from_analysis,
    calibrate_from_dimension,
    calibrate_from_scale_notation,
    calibrate_from_sheet_size,
    calibrate_from_dpi_default,
    calibrate_manual,
    parse_measurement,
    parse_scale_notation,
    parse_sheet_size,
    estimate_drawing_bounds,
    SHEET_SIZES,
)
from aec_agent.mcp.tools.gemini_first.gemini_understanding import (
    CalibrationHint,
    DrawingAnalysis,
    DrawingElements,
    RegionBounds,
    ExtractionStrategyConfig,
)


class TestParseMeasurement:
    """Tests for parse_measurement function."""

    def test_feet_and_inches_standard(self):
        """Test standard feet-inches format: 20'-0"."""
        value, unit = parse_measurement("20'-0\"")
        assert value == 240.0  # 20 feet = 240 inches
        assert unit == "inches"

    def test_feet_and_inches_with_inches(self):
        """Test feet-inches with non-zero inches: 20'-6"."""
        value, unit = parse_measurement("20'-6\"")
        assert value == 246.0  # 20*12 + 6 = 246 inches
        assert unit == "inches"

    def test_feet_and_inches_space_separator(self):
        """Test feet-inches with space: 20' 6"."""
        value, unit = parse_measurement("20' 6\"")
        assert value == 246.0
        assert unit == "inches"

    def test_feet_and_inches_no_inches(self):
        """Test feet only with apostrophe: 20'."""
        value, unit = parse_measurement("20'")
        assert value == 240.0
        assert unit == "inches"

    def test_inches_with_quote(self):
        """Test inches with quote: 24"."""
        value, unit = parse_measurement("24\"")
        assert value == 24.0
        assert unit == "inches"

    def test_inches_with_in_suffix(self):
        """Test inches with 'in' suffix."""
        value, unit = parse_measurement("24 in")
        assert value == 24.0
        assert unit == "inches"

    def test_millimeters(self):
        """Test millimeters: 100mm."""
        value, unit = parse_measurement("100mm")
        assert value == 100.0
        assert unit == "mm"

    def test_millimeters_with_space(self):
        """Test millimeters with space: 100 mm."""
        value, unit = parse_measurement("100 mm")
        assert value == 100.0
        assert unit == "mm"

    def test_meters(self):
        """Test meters: 1.5m."""
        value, unit = parse_measurement("1.5m")
        assert value == 1500.0  # Converted to mm
        assert unit == "mm"

    def test_centimeters(self):
        """Test centimeters: 50cm."""
        value, unit = parse_measurement("50cm")
        assert value == 500.0  # Converted to mm
        assert unit == "mm"

    def test_plain_number(self):
        """Test plain number (assumed inches)."""
        value, unit = parse_measurement("24")
        assert value == 24.0
        assert unit == "inches"

    def test_decimal_feet_inches(self):
        """Test decimal feet: 20.5'-0"."""
        value, unit = parse_measurement("20.5'-0\"")
        assert value == 246.0  # 20.5 * 12 = 246
        assert unit == "inches"

    def test_empty_string(self):
        """Test empty string returns None."""
        value, unit = parse_measurement("")
        assert value is None
        assert unit is None

    def test_invalid_format(self):
        """Test invalid format returns None."""
        value, unit = parse_measurement("invalid")
        assert value is None
        assert unit is None


class TestParseScaleNotation:
    """Tests for parse_scale_notation function."""

    def test_quarter_inch_scale(self):
        """Test 1/4" = 1'-0" scale at 300 DPI."""
        # 1/4" = 12" means 1 paper inch = 48 real inches
        # At 300 DPI, 1 pixel = 1/300 paper inch = 48/300 = 0.16 real inches
        value, unit = parse_scale_notation('1/4" = 1\'-0"', dpi=300)
        assert value is not None
        assert unit == "inches"
        assert abs(value - 0.16) < 0.001

    def test_eighth_inch_scale(self):
        """Test 1/8" = 1'-0" scale."""
        # 1/8" = 12" means 1 paper inch = 96 real inches
        # At 300 DPI, 1 pixel = 96/300 = 0.32 real inches
        value, unit = parse_scale_notation('1/8" = 1\'-0"', dpi=300)
        assert value is not None
        assert unit == "inches"
        assert abs(value - 0.32) < 0.001

    def test_full_inch_scale(self):
        """Test 1" = 1'-0" scale."""
        # 1" = 12" means 1 paper inch = 12 real inches
        # At 300 DPI, 1 pixel = 12/300 = 0.04 real inches
        value, unit = parse_scale_notation('1" = 1\'-0"', dpi=300)
        assert value is not None
        assert unit == "inches"
        assert abs(value - 0.04) < 0.001

    def test_metric_scale_100(self):
        """Test 1:100 metric scale."""
        # At 300 DPI, 1 pixel = 25.4/300 paper mm = 0.0847 paper mm
        # 1:100 means 1 paper mm = 100 real mm
        # So 1 pixel = 0.0847 * 100 = 8.47 real mm
        value, unit = parse_scale_notation("1:100", dpi=300)
        assert value is not None
        assert unit == "mm"
        expected = (25.4 / 300) * 100
        assert abs(value - expected) < 0.01

    def test_metric_scale_50(self):
        """Test 1:50 metric scale."""
        value, unit = parse_scale_notation("1:50", dpi=300)
        assert value is not None
        assert unit == "mm"
        expected = (25.4 / 300) * 50
        assert abs(value - expected) < 0.01

    def test_different_dpi(self):
        """Test scale parsing at different DPI."""
        # Same scale at 150 DPI should give 2x the units per pixel
        value_300, _ = parse_scale_notation('1/4" = 1\'-0"', dpi=300)
        value_150, _ = parse_scale_notation('1/4" = 1\'-0"', dpi=150)
        assert value_150 is not None
        assert abs(value_150 / value_300 - 2.0) < 0.01

    def test_empty_string(self):
        """Test empty string returns None."""
        value, unit = parse_scale_notation("")
        assert value is None
        assert unit is None

    def test_invalid_format(self):
        """Test invalid format returns None."""
        value, unit = parse_scale_notation("invalid scale")
        assert value is None
        assert unit is None


class TestParseSheetSize:
    """Tests for parse_sheet_size function."""

    def test_arch_d(self):
        """Test ARCH D sheet size."""
        width, height = parse_sheet_size("ARCH D")
        assert width == 24.0
        assert height == 36.0

    def test_arch_d_case_insensitive(self):
        """Test case insensitivity."""
        width, height = parse_sheet_size("arch d")
        assert width == 24.0
        assert height == 36.0

    def test_arch_d_with_dimensions(self):
        """Test ARCH D with dimensions in parentheses."""
        width, height = parse_sheet_size("ARCH D (24x36)")
        assert width == 24.0
        assert height == 36.0

    def test_ansi_b(self):
        """Test ANSI B sheet size."""
        width, height = parse_sheet_size("ANSI B")
        assert width == 11.0
        assert height == 17.0

    def test_a1_metric(self):
        """Test A1 sheet size (ISO metric)."""
        width, height = parse_sheet_size("A1")
        assert abs(width - 23.39) < 0.01
        assert abs(height - 33.11) < 0.01

    def test_dimensions_only(self):
        """Test parsing dimensions without name: 24x36."""
        width, height = parse_sheet_size("24x36")
        assert width == 24.0
        assert height == 36.0

    def test_dimensions_with_quotes(self):
        """Test parsing dimensions with quotes: 24"x36"."""
        width, height = parse_sheet_size("24\"x36\"")
        assert width == 24.0
        assert height == 36.0

    def test_dimensions_with_spaces(self):
        """Test parsing dimensions with spaces: 24 x 36."""
        width, height = parse_sheet_size("24 x 36")
        assert width == 24.0
        assert height == 36.0

    def test_empty_string(self):
        """Test empty string returns None."""
        width, height = parse_sheet_size("")
        assert width is None
        assert height is None

    def test_unknown_size(self):
        """Test unknown size returns None."""
        width, height = parse_sheet_size("UNKNOWN")
        assert width is None
        assert height is None


class TestScaleCalibration:
    """Tests for ScaleCalibration dataclass."""

    def test_basic_creation(self):
        """Test creating a ScaleCalibration."""
        cal = ScaleCalibration(
            scale_factor=0.16,
            units="inches",
            confidence=0.9,
            method="dimension",
            image_width=3600,
            image_height=2400,
        )

        assert cal.scale_factor == 0.16
        assert cal.units == "inches"
        assert cal.confidence == 0.9
        assert cal.method == "dimension"
        assert cal.image_width == 3600
        assert cal.image_height == 2400

    def test_to_dwg_basic(self):
        """Test basic coordinate conversion."""
        cal = ScaleCalibration(
            scale_factor=0.1,  # 1 pixel = 0.1 units
            units="inches",
            confidence=0.9,
            method="test",
            image_width=1000,
            image_height=1000,
        )

        # Point at (100, 200) pixels
        # x_dwg = 100 * 0.1 = 10
        # y_dwg = (1000 - 200) * 0.1 = 80 (Y flipped)
        dwg_x, dwg_y = cal.to_dwg(100, 200)
        assert abs(dwg_x - 10.0) < 0.001
        assert abs(dwg_y - 80.0) < 0.001

    def test_to_dwg_origin_at_top(self):
        """Test that image top (y=0) maps to DWG max Y."""
        cal = ScaleCalibration(
            scale_factor=0.1,
            units="inches",
            confidence=0.9,
            method="test",
            image_width=1000,
            image_height=1000,
        )

        # Top-left corner (0, 0) in image -> (0, 100) in DWG
        dwg_x, dwg_y = cal.to_dwg(0, 0)
        assert dwg_x == 0.0
        assert dwg_y == 100.0  # height * scale_factor

    def test_to_dwg_origin_at_bottom(self):
        """Test that image bottom (y=max) maps to DWG y=0."""
        cal = ScaleCalibration(
            scale_factor=0.1,
            units="inches",
            confidence=0.9,
            method="test",
            image_width=1000,
            image_height=1000,
        )

        # Bottom-left corner (0, 1000) in image -> (0, 0) in DWG
        dwg_x, dwg_y = cal.to_dwg(0, 1000)
        assert dwg_x == 0.0
        assert dwg_y == 0.0

    def test_to_dwg_with_origin_offset(self):
        """Test coordinate conversion with origin offset."""
        cal = ScaleCalibration(
            scale_factor=0.1,
            units="inches",
            confidence=0.9,
            method="test",
            image_width=1000,
            image_height=1000,
            origin_offset=(100.0, 50.0),  # DWG origin offset
        )

        # Point at (0, 1000) in image -> (100, 50) in DWG (bottom-left + offset)
        dwg_x, dwg_y = cal.to_dwg(0, 1000)
        assert dwg_x == 100.0
        assert dwg_y == 50.0

    def test_to_pixels_inverse(self):
        """Test that to_pixels is inverse of to_dwg."""
        cal = ScaleCalibration(
            scale_factor=0.16,
            units="inches",
            confidence=0.9,
            method="test",
            image_width=3600,
            image_height=2400,
        )

        # Convert pixel to DWG and back
        original_px = (500.0, 300.0)
        dwg_coords = cal.to_dwg(*original_px)
        recovered_px = cal.to_pixels(*dwg_coords)

        assert abs(recovered_px[0] - original_px[0]) < 0.001
        assert abs(recovered_px[1] - original_px[1]) < 0.001

    def test_scale_length(self):
        """Test scaling a length (distance)."""
        cal = ScaleCalibration(
            scale_factor=0.16,
            units="inches",
            confidence=0.9,
            method="test",
            image_width=3600,
            image_height=2400,
        )

        # 100 pixels * 0.16 = 16 inches
        length = cal.scale_length(100)
        assert abs(length - 16.0) < 0.001

    def test_convert_units_inches_to_feet(self):
        """Test unit conversion from inches to feet."""
        cal = ScaleCalibration(
            scale_factor=0.16,  # 0.16 inches/pixel
            units="inches",
            confidence=0.9,
            method="test",
            image_width=3600,
            image_height=2400,
        )

        cal_feet = cal.convert_units_to("feet")
        assert cal_feet.units == "feet"
        assert abs(cal_feet.scale_factor - 0.16 / 12) < 0.0001

    def test_convert_units_inches_to_mm(self):
        """Test unit conversion from inches to mm."""
        cal = ScaleCalibration(
            scale_factor=0.16,  # 0.16 inches/pixel
            units="inches",
            confidence=0.9,
            method="test",
            image_width=3600,
            image_height=2400,
        )

        cal_mm = cal.convert_units_to("mm")
        assert cal_mm.units == "mm"
        assert abs(cal_mm.scale_factor - 0.16 * 25.4) < 0.001

    def test_convert_units_invalid(self):
        """Test invalid unit conversion raises ValueError."""
        cal = ScaleCalibration(
            scale_factor=0.16,
            units="inches",
            confidence=0.9,
            method="test",
            image_width=3600,
            image_height=2400,
        )

        with pytest.raises(ValueError):
            cal.convert_units_to("cubits")

    def test_to_dict(self):
        """Test serialization to dictionary."""
        cal = ScaleCalibration(
            scale_factor=0.16,
            units="inches",
            confidence=0.9,
            method="dimension",
            image_width=3600,
            image_height=2400,
            source_description="Test calibration",
        )

        d = cal.to_dict()
        assert d["scale_factor"] == 0.16
        assert d["units"] == "inches"
        assert d["confidence"] == 0.9
        assert d["method"] == "dimension"
        assert d["image_width"] == 3600
        assert d["image_height"] == 2400
        assert d["source_description"] == "Test calibration"
        assert "units_per_pixel" in d
        assert "pixels_per_unit" in d

    def test_from_dict(self):
        """Test deserialization from dictionary."""
        data = {
            "scale_factor": 0.16,
            "units": "inches",
            "confidence": 0.9,
            "method": "dimension",
            "image_width": 3600,
            "image_height": 2400,
            "origin_offset": [10.0, 20.0],
            "source_description": "Test",
        }

        cal = ScaleCalibration.from_dict(data)
        assert cal.scale_factor == 0.16
        assert cal.units == "inches"
        assert cal.origin_offset == (10.0, 20.0)


class TestCalibrateFromDimension:
    """Tests for calibrate_from_dimension function."""

    def test_basic_calibration(self):
        """Test calibration from a dimension hint."""
        hint = CalibrationHint(
            hint_type="dimension",
            description="Wall length",
            pixel_measurement=150.0,  # 150 pixels
            real_measurement="24\"",  # 24 inches
            confidence=0.9,
        )

        cal = calibrate_from_dimension(hint, image_width=3600, image_height=2400)

        assert cal is not None
        assert cal.method == "dimension"
        assert abs(cal.scale_factor - 24.0 / 150.0) < 0.001  # 0.16 inches/pixel
        assert cal.units == "inches"

    def test_feet_inches_measurement(self):
        """Test calibration with feet-inches measurement."""
        hint = CalibrationHint(
            hint_type="dimension",
            description="Room width",
            pixel_measurement=750.0,
            real_measurement="10'-0\"",  # 120 inches
            confidence=0.85,
        )

        cal = calibrate_from_dimension(hint, image_width=3600, image_height=2400)

        assert cal is not None
        assert abs(cal.scale_factor - 120.0 / 750.0) < 0.001
        assert cal.units == "inches"

    def test_metric_measurement(self):
        """Test calibration with metric measurement."""
        hint = CalibrationHint(
            hint_type="dimension",
            description="Door width",
            pixel_measurement=100.0,
            real_measurement="900mm",
            confidence=0.9,
        )

        cal = calibrate_from_dimension(hint, image_width=3600, image_height=2400)

        assert cal is not None
        assert abs(cal.scale_factor - 9.0) < 0.001  # 9 mm/pixel
        assert cal.units == "mm"

    def test_zero_pixel_measurement(self):
        """Test that zero pixel measurement returns None."""
        hint = CalibrationHint(
            hint_type="dimension",
            description="Test",
            pixel_measurement=0,
            real_measurement="24\"",
            confidence=0.9,
        )

        cal = calibrate_from_dimension(hint, image_width=3600, image_height=2400)
        assert cal is None

    def test_invalid_measurement(self):
        """Test that invalid measurement returns None."""
        hint = CalibrationHint(
            hint_type="dimension",
            description="Test",
            pixel_measurement=100,
            real_measurement="invalid",
            confidence=0.9,
        )

        cal = calibrate_from_dimension(hint, image_width=3600, image_height=2400)
        assert cal is None


class TestCalibrateFromScaleNotation:
    """Tests for calibrate_from_scale_notation function."""

    def test_quarter_inch_scale(self):
        """Test calibration from 1/4" = 1'-0" scale."""
        cal = calibrate_from_scale_notation(
            '1/4" = 1\'-0"',
            dpi=300,
            image_width=3600,
            image_height=2400,
        )

        assert cal is not None
        assert cal.method == "scale_notation"
        assert cal.confidence == 0.85
        assert abs(cal.scale_factor - 0.16) < 0.001

    def test_metric_scale(self):
        """Test calibration from 1:100 scale."""
        cal = calibrate_from_scale_notation(
            "1:100",
            dpi=300,
            image_width=3600,
            image_height=2400,
        )

        assert cal is not None
        assert cal.units == "mm"
        expected = (25.4 / 300) * 100
        assert abs(cal.scale_factor - expected) < 0.01

    def test_invalid_scale(self):
        """Test that invalid scale returns None."""
        cal = calibrate_from_scale_notation(
            "not a scale",
            dpi=300,
            image_width=3600,
            image_height=2400,
        )

        assert cal is None


class TestCalibrateFromSheetSize:
    """Tests for calibrate_from_sheet_size function."""

    def test_arch_d_sheet(self):
        """Test calibration from ARCH D sheet size."""
        # ARCH D is 24x36 inches
        # If image is 3600x5400 pixels, scale = 24/3600 = 0.00667 inches/pixel
        cal = calibrate_from_sheet_size(
            "ARCH D",
            image_width=3600,
            image_height=5400,
        )

        assert cal is not None
        assert cal.method == "sheet_size"
        assert cal.units == "inches"
        # Geometric mean of 24/3600 and 36/5400
        expected = math.sqrt((24.0 / 3600) * (36.0 / 5400))
        assert abs(cal.scale_factor - expected) < 0.0001

    def test_aspect_ratio_affects_confidence(self):
        """Test that mismatched aspect ratio reduces confidence."""
        # ARCH D is 24x36 (ratio 0.667)
        # Image 3600x2400 has ratio 1.5 - significantly different
        cal = calibrate_from_sheet_size(
            "ARCH D",
            image_width=3600,
            image_height=2400,
        )

        assert cal is not None
        assert cal.confidence < 0.70  # Reduced due to aspect ratio mismatch

    def test_matching_aspect_ratio(self):
        """Test that matching aspect ratio gives higher confidence."""
        # ARCH D is 24x36 (ratio 0.667)
        # Image 2400x3600 has ratio 0.667 - matching
        cal = calibrate_from_sheet_size(
            "ARCH D",
            image_width=2400,
            image_height=3600,
        )

        assert cal is not None
        assert cal.confidence >= 0.65  # Good confidence

    def test_unknown_sheet(self):
        """Test that unknown sheet returns None."""
        cal = calibrate_from_sheet_size(
            "UNKNOWN SIZE",
            image_width=3600,
            image_height=2400,
        )

        assert cal is None


class TestCalibrateFromDpiDefault:
    """Tests for calibrate_from_dpi_default function."""

    def test_300_dpi(self):
        """Test default calibration at 300 DPI."""
        cal = calibrate_from_dpi_default(
            dpi=300,
            image_width=3600,
            image_height=2400,
        )

        assert cal is not None
        assert cal.method == "dpi_default"
        assert cal.confidence == 0.30  # Low confidence
        assert abs(cal.scale_factor - 1.0 / 300) < 0.00001

    def test_150_dpi(self):
        """Test default calibration at 150 DPI."""
        cal = calibrate_from_dpi_default(
            dpi=150,
            image_width=1800,
            image_height=1200,
        )

        assert abs(cal.scale_factor - 1.0 / 150) < 0.00001


class TestCalibrateManual:
    """Tests for calibrate_manual function."""

    def test_basic_manual(self):
        """Test creating manual calibration."""
        cal = calibrate_manual(
            scale_factor=0.16,
            units="inches",
            image_width=3600,
            image_height=2400,
        )

        assert cal.scale_factor == 0.16
        assert cal.units == "inches"
        assert cal.method == "manual"
        assert cal.confidence == 1.0  # Full confidence for manual

    def test_with_origin_offset(self):
        """Test manual calibration with origin offset."""
        cal = calibrate_manual(
            scale_factor=0.16,
            units="inches",
            image_width=3600,
            image_height=2400,
            origin_offset=(100.0, 50.0),
        )

        assert cal.origin_offset == (100.0, 50.0)


class TestCalibrateFromAnalysis:
    """Tests for calibrate_from_analysis function."""

    def _create_analysis(
        self,
        calibration_hints=None,
        scale=None,
        sheet_size=None,
    ) -> DrawingAnalysis:
        """Helper to create DrawingAnalysis with specified parameters."""
        return DrawingAnalysis(
            drawing_type="floor_plan",
            scale=scale,
            sheet_size=sheet_size,
            units="imperial",
            complexity="medium",
            calibration_hints=calibration_hints or [],
            elements=DrawingElements(),
            extraction_strategy=ExtractionStrategyConfig(),
        )

    def test_dimension_highest_priority(self):
        """Test that dimension hint takes priority over other methods."""
        hints = [
            CalibrationHint(
                hint_type="dimension",
                description="Wall",
                pixel_measurement=150.0,
                real_measurement="24\"",
                confidence=0.9,
            )
        ]
        analysis = self._create_analysis(
            calibration_hints=hints,
            scale='1/4" = 1\'-0"',  # Would give different result
            sheet_size="ARCH D",  # Would give different result
        )

        cal = calibrate_from_analysis(
            analysis,
            image_width=3600,
            image_height=2400,
            image_dpi=300,
        )

        assert cal.method == "dimension"
        assert abs(cal.scale_factor - 24.0 / 150.0) < 0.001

    def test_scale_notation_fallback(self):
        """Test that scale notation is used when no dimension available."""
        analysis = self._create_analysis(
            scale='1/4" = 1\'-0"',
            sheet_size="ARCH D",
        )

        cal = calibrate_from_analysis(
            analysis,
            image_width=3600,
            image_height=2400,
            image_dpi=300,
        )

        assert cal.method == "scale_notation"

    def test_sheet_size_fallback(self):
        """Test that sheet size is used when no scale notation available."""
        analysis = self._create_analysis(
            sheet_size="ARCH D",
        )

        cal = calibrate_from_analysis(
            analysis,
            image_width=3600,
            image_height=2400,
            image_dpi=300,
        )

        assert cal.method == "sheet_size"

    def test_dpi_default_fallback(self):
        """Test DPI default when no other info available."""
        analysis = self._create_analysis()

        cal = calibrate_from_analysis(
            analysis,
            image_width=3600,
            image_height=2400,
            image_dpi=300,
        )

        assert cal.method == "dpi_default"
        assert cal.confidence == 0.30

    def test_prefer_units_conversion(self):
        """Test that prefer_units converts output units."""
        hints = [
            CalibrationHint(
                hint_type="dimension",
                description="Wall",
                pixel_measurement=150.0,
                real_measurement="24\"",
                confidence=0.9,
            )
        ]
        analysis = self._create_analysis(calibration_hints=hints)

        cal = calibrate_from_analysis(
            analysis,
            image_width=3600,
            image_height=2400,
            image_dpi=300,
            prefer_units="mm",  # Convert to mm
        )

        assert cal.units == "mm"
        # 24/150 inches/pixel * 25.4 mm/inch
        expected_mm = (24.0 / 150.0) * 25.4
        assert abs(cal.scale_factor - expected_mm) < 0.01


class TestEstimateDrawingBounds:
    """Tests for estimate_drawing_bounds function."""

    def test_with_drawing_area(self):
        """Test bounds estimation with drawing area defined."""
        analysis = DrawingAnalysis(
            drawing_type="floor_plan",
            drawing_area=RegionBounds(x1=100, y1=100, x2=3500, y2=2300),
            elements=DrawingElements(),
            extraction_strategy=ExtractionStrategyConfig(),
        )

        cal = ScaleCalibration(
            scale_factor=0.1,
            units="inches",
            confidence=0.9,
            method="test",
            image_width=3600,
            image_height=2400,
        )

        min_pt, max_pt = estimate_drawing_bounds(analysis, cal)

        # x1=100, y2=2300 -> bottom-left in DWG
        # x2=3500, y1=100 -> top-right in DWG
        assert abs(min_pt[0] - 100 * 0.1) < 0.001
        assert abs(min_pt[1] - (2400 - 2300) * 0.1) < 0.001
        assert abs(max_pt[0] - 3500 * 0.1) < 0.001
        assert abs(max_pt[1] - (2400 - 100) * 0.1) < 0.001

    def test_without_drawing_area(self):
        """Test bounds estimation without drawing area (uses full image)."""
        analysis = DrawingAnalysis(
            drawing_type="floor_plan",
            elements=DrawingElements(),
            extraction_strategy=ExtractionStrategyConfig(),
        )

        cal = ScaleCalibration(
            scale_factor=0.1,
            units="inches",
            confidence=0.9,
            method="test",
            image_width=3600,
            image_height=2400,
        )

        min_pt, max_pt = estimate_drawing_bounds(analysis, cal)

        # Full image: (0,0) to (3600, 2400) pixels
        # In DWG: (0, 0) to (360, 240) inches
        assert abs(min_pt[0]) < 0.001
        assert abs(min_pt[1]) < 0.001
        assert abs(max_pt[0] - 360.0) < 0.001
        assert abs(max_pt[1] - 240.0) < 0.001


class TestSheetSizesConstant:
    """Tests for SHEET_SIZES constant."""

    def test_arch_sizes_present(self):
        """Test that ARCH sizes are defined."""
        assert "ARCH A" in SHEET_SIZES
        assert "ARCH B" in SHEET_SIZES
        assert "ARCH C" in SHEET_SIZES
        assert "ARCH D" in SHEET_SIZES
        assert "ARCH E" in SHEET_SIZES

    def test_ansi_sizes_present(self):
        """Test that ANSI sizes are defined."""
        assert "ANSI A" in SHEET_SIZES
        assert "ANSI B" in SHEET_SIZES
        assert "ANSI C" in SHEET_SIZES
        assert "ANSI D" in SHEET_SIZES
        assert "ANSI E" in SHEET_SIZES

    def test_iso_sizes_present(self):
        """Test that ISO A sizes are defined."""
        assert "A0" in SHEET_SIZES
        assert "A1" in SHEET_SIZES
        assert "A2" in SHEET_SIZES
        assert "A3" in SHEET_SIZES
        assert "A4" in SHEET_SIZES

    def test_sizes_are_tuples(self):
        """Test that all sizes are (width, height) tuples."""
        for name, size in SHEET_SIZES.items():
            assert isinstance(size, tuple)
            assert len(size) == 2
            assert size[0] > 0
            assert size[1] > 0
