"""
Phase 3: Coordinate Calibration - Map Pixels to DWG Units.

This module converts Gemini's pixel coordinates to AutoCAD DWG units using
calibration hints from Phase 2's drawing analysis.

Key Features:
- Multiple calibration methods (dimension, scale notation, sheet size, DPI default)
- Confidence-based method selection
- Y-axis flip (PDF top-down -> AutoCAD bottom-up)
- Imperial and metric unit support

Usage:
    >>> calibration = calibrate_from_analysis(analysis, width_px=3600, height_px=2400)
    >>> dwg_x, dwg_y = calibration.to_dwg(100, 200)  # Convert pixel to DWG coords
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Optional, Tuple, List

import structlog

from .gemini_understanding import DrawingAnalysis, CalibrationHint

logger = structlog.get_logger(__name__)


# Standard sheet sizes in inches (width, height)
SHEET_SIZES = {
    # ARCH sizes (common in US architecture)
    "ARCH A": (9.0, 12.0),
    "ARCH B": (12.0, 18.0),
    "ARCH C": (18.0, 24.0),
    "ARCH D": (24.0, 36.0),
    "ARCH E": (36.0, 48.0),
    "ARCH E1": (30.0, 42.0),
    # ANSI sizes (engineering)
    "ANSI A": (8.5, 11.0),
    "ANSI B": (11.0, 17.0),
    "ANSI C": (17.0, 22.0),
    "ANSI D": (22.0, 34.0),
    "ANSI E": (34.0, 44.0),
    # ISO A sizes (international, in inches)
    "A0": (33.11, 46.81),
    "A1": (23.39, 33.11),
    "A2": (16.54, 23.39),
    "A3": (11.69, 16.54),
    "A4": (8.27, 11.69),
    # Common custom sizes
    "24X36": (24.0, 36.0),
    "30X42": (30.0, 42.0),
    "36X48": (36.0, 48.0),
}


@dataclass
class ScaleCalibration:
    """
    Scale calibration result for converting pixel coordinates to DWG units.

    Attributes:
        scale_factor: DWG units per pixel
        units: Unit system ("inches", "feet", "mm", "m")
        confidence: Confidence level (0-1)
        method: Calibration method used
        image_width: Original image width in pixels
        image_height: Original image height in pixels (for Y-axis flip)
        origin_offset: Optional origin offset (dwg_x, dwg_y) to add after transform
    """

    scale_factor: float  # DWG units per pixel
    units: str  # "inches", "feet", "mm", "m"
    confidence: float  # 0-1
    method: str  # "dimension", "scale_notation", "sheet_size", "dpi_default", "manual"

    # Image dimensions for coordinate transform
    image_width: int
    image_height: int

    # Optional origin offset
    origin_offset: Tuple[float, float] = (0.0, 0.0)

    # Source details for debugging
    source_description: str = ""

    def to_dwg(self, pixel_x: float, pixel_y: float) -> Tuple[float, float]:
        """
        Convert pixel coordinates to DWG coordinates.

        The Y-axis is flipped because:
        - Image coordinates: origin at top-left, Y increases downward
        - AutoCAD coordinates: origin at bottom-left, Y increases upward

        Args:
            pixel_x: X coordinate in pixels
            pixel_y: Y coordinate in pixels

        Returns:
            Tuple of (dwg_x, dwg_y) in DWG units
        """
        dwg_x = pixel_x * self.scale_factor + self.origin_offset[0]
        dwg_y = (self.image_height - pixel_y) * self.scale_factor + self.origin_offset[1]
        return (dwg_x, dwg_y)

    def to_pixels(self, dwg_x: float, dwg_y: float) -> Tuple[float, float]:
        """
        Convert DWG coordinates back to pixel coordinates.

        Args:
            dwg_x: X coordinate in DWG units
            dwg_y: Y coordinate in DWG units

        Returns:
            Tuple of (pixel_x, pixel_y)
        """
        pixel_x = (dwg_x - self.origin_offset[0]) / self.scale_factor
        pixel_y = self.image_height - (dwg_y - self.origin_offset[1]) / self.scale_factor
        return (pixel_x, pixel_y)

    def scale_length(self, pixel_length: float) -> float:
        """
        Scale a length (distance) from pixels to DWG units.

        Unlike to_dwg(), this does not flip axes or add offsets.
        Use for radii, dimensions, heights, etc.

        Args:
            pixel_length: Length in pixels

        Returns:
            Length in DWG units
        """
        return pixel_length * self.scale_factor

    def convert_units_to(self, target_units: str) -> "ScaleCalibration":
        """
        Create a new ScaleCalibration with different units.

        Args:
            target_units: Target unit system ("inches", "feet", "mm", "m")

        Returns:
            New ScaleCalibration with converted scale factor
        """
        conversion_factors = {
            # From inches to target
            ("inches", "feet"): 1.0 / 12.0,
            ("inches", "mm"): 25.4,
            ("inches", "m"): 0.0254,
            ("inches", "inches"): 1.0,
            # From feet to target
            ("feet", "inches"): 12.0,
            ("feet", "mm"): 304.8,
            ("feet", "m"): 0.3048,
            ("feet", "feet"): 1.0,
            # From mm to target
            ("mm", "inches"): 1.0 / 25.4,
            ("mm", "feet"): 1.0 / 304.8,
            ("mm", "m"): 0.001,
            ("mm", "mm"): 1.0,
            # From m to target
            ("m", "inches"): 39.3701,
            ("m", "feet"): 3.28084,
            ("m", "mm"): 1000.0,
            ("m", "m"): 1.0,
        }

        key = (self.units, target_units)
        if key not in conversion_factors:
            raise ValueError(f"Unknown unit conversion: {self.units} -> {target_units}")

        factor = conversion_factors[key]

        return ScaleCalibration(
            scale_factor=self.scale_factor * factor,
            units=target_units,
            confidence=self.confidence,
            method=self.method,
            image_width=self.image_width,
            image_height=self.image_height,
            origin_offset=(
                self.origin_offset[0] * factor,
                self.origin_offset[1] * factor,
            ),
            source_description=self.source_description,
        )

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "scale_factor": self.scale_factor,
            "units": self.units,
            "confidence": self.confidence,
            "method": self.method,
            "image_width": self.image_width,
            "image_height": self.image_height,
            "origin_offset": list(self.origin_offset),
            "source_description": self.source_description,
            # Computed values for convenience
            "units_per_pixel": self.scale_factor,
            "pixels_per_unit": 1.0 / self.scale_factor if self.scale_factor > 0 else 0,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "ScaleCalibration":
        """Create from dictionary."""
        return cls(
            scale_factor=data["scale_factor"],
            units=data["units"],
            confidence=data["confidence"],
            method=data["method"],
            image_width=data["image_width"],
            image_height=data["image_height"],
            origin_offset=tuple(data.get("origin_offset", [0.0, 0.0])),
            source_description=data.get("source_description", ""),
        )


def parse_measurement(measurement_str: str) -> Tuple[Optional[float], Optional[str]]:
    """
    Parse a measurement string into (value, unit).

    Supported formats:
    - Feet and inches: 20'-0", 20' 0", 20'0", 20'-6"
    - Inches: 24", 24 in, 24"
    - Millimeters: 100mm, 100 mm
    - Meters: 1.5m, 1.5 m (NOT mm)
    - Centimeters: 50cm, 50 cm
    - Plain numbers: 24 (assumes inches for imperial context)

    Args:
        measurement_str: The measurement string to parse

    Returns:
        Tuple of (numeric_value, unit_name) or (None, None) if parsing fails

    Examples:
        >>> parse_measurement("20'-0\"")
        (240.0, 'inches')
        >>> parse_measurement("24\"")
        (24.0, 'inches')
        >>> parse_measurement("100mm")
        (100.0, 'mm')
    """
    if not measurement_str:
        return (None, None)

    measurement_str = measurement_str.strip()

    # Feet and inches: 20'-0", 20' 0", 20'0", 20'-6", 20'6"
    # Pattern: digits + apostrophe + optional dash/space + optional digits + optional quote
    feet_inches_pattern = r"(\d+(?:\.\d+)?)['\u2032]\s*[-\s]?\s*(\d+(?:\.\d+)?)?[\"\u2033]?"
    match = re.match(feet_inches_pattern, measurement_str)
    if match:
        feet = float(match.group(1))
        inches = float(match.group(2)) if match.group(2) else 0
        return (feet * 12.0 + inches, "inches")

    # Plain inches: 24", 24 in, 24in, 24 inches
    inches_pattern = r"(\d+(?:\.\d+)?)\s*(?:[\"\u2033]|in(?:ches?)?)"
    match = re.match(inches_pattern, measurement_str, re.IGNORECASE)
    if match:
        return (float(match.group(1)), "inches")

    # Feet only: 20', 20 ft, 20ft, 20 feet
    feet_pattern = r"(\d+(?:\.\d+)?)\s*(?:['\u2032]|ft|feet)"
    match = re.match(feet_pattern, measurement_str, re.IGNORECASE)
    if match:
        return (float(match.group(1)) * 12.0, "inches")  # Convert to inches

    # Millimeters: 100mm, 100 mm, 100 millimeters
    mm_pattern = r"(\d+(?:\.\d+)?)\s*mm(?:illimeters?)?"
    match = re.match(mm_pattern, measurement_str, re.IGNORECASE)
    if match:
        return (float(match.group(1)), "mm")

    # Centimeters: 50cm, 50 cm
    cm_pattern = r"(\d+(?:\.\d+)?)\s*cm"
    match = re.match(cm_pattern, measurement_str, re.IGNORECASE)
    if match:
        return (float(match.group(1)) * 10.0, "mm")  # Convert to mm

    # Meters: 1.5m, 1.5 m (but NOT mm)
    # Use negative lookbehind to exclude mm
    m_pattern = r"(\d+(?:\.\d+)?)\s*m(?!m)(?:eters?)?"
    match = re.match(m_pattern, measurement_str, re.IGNORECASE)
    if match:
        return (float(match.group(1)) * 1000.0, "mm")  # Convert to mm

    # Plain number (assume inches in imperial context)
    plain_pattern = r"^(\d+(?:\.\d+)?)$"
    match = re.match(plain_pattern, measurement_str)
    if match:
        return (float(match.group(1)), "inches")

    return (None, None)


def parse_scale_notation(
    scale_str: str,
    dpi: int = 300,
) -> Tuple[Optional[float], Optional[str]]:
    """
    Parse architectural scale notation to get units per pixel.

    Supported formats:
    - Imperial: 1/4" = 1'-0", 1/8" = 1'-0", 1" = 1'-0"
    - Metric: 1:100, 1:50, 1:200
    - Hybrid: 1/4" = 1'

    Args:
        scale_str: The scale notation string
        dpi: DPI of the rendered image (default 300)

    Returns:
        Tuple of (units_per_pixel, unit_name) or (None, None) if parsing fails

    Examples:
        >>> parse_scale_notation("1/4\" = 1'-0\"", dpi=300)
        (0.16, 'inches')  # 1 pixel = 0.16 real inches at 1/4" scale
    """
    if not scale_str:
        return (None, None)

    scale_str = scale_str.strip()

    # Imperial scale: 1/4" = 1'-0", 1/8" = 1', etc.
    # Paper measurement (fraction or decimal) = Real measurement
    imperial_pattern = r"(\d+(?:\.\d+)?)\s*/\s*(\d+)\s*[\"\u2033]\s*=\s*(\d+)\s*['\u2032]\s*[-\s]?\s*(\d+)?\s*[\"\u2033]?"
    match = re.match(imperial_pattern, scale_str)
    if match:
        paper_numerator = float(match.group(1))
        paper_denominator = float(match.group(2))
        paper_inches = paper_numerator / paper_denominator

        real_feet = float(match.group(3))
        real_inches = float(match.group(4)) if match.group(4) else 0
        real_total_inches = real_feet * 12.0 + real_inches

        # paper_inches on paper = real_total_inches in reality
        # So 1 paper inch = real_total_inches / paper_inches real inches
        inches_per_paper_inch = real_total_inches / paper_inches

        # At given DPI, 1 pixel = 1/dpi paper inches
        # So 1 pixel = (1/dpi) * inches_per_paper_inch real inches
        inches_per_pixel = (1.0 / dpi) * inches_per_paper_inch

        return (inches_per_pixel, "inches")

    # Simplified imperial: 1" = 1'-0"
    simple_imperial = r"(\d+(?:\.\d+)?)\s*[\"\u2033]\s*=\s*(\d+)\s*['\u2032](?:\s*[-\s]?\s*(\d+)\s*[\"\u2033]?)?"
    match = re.match(simple_imperial, scale_str)
    if match:
        paper_inches = float(match.group(1))
        real_feet = float(match.group(2))
        real_inches = float(match.group(3)) if match.group(3) else 0
        real_total_inches = real_feet * 12.0 + real_inches

        inches_per_paper_inch = real_total_inches / paper_inches
        inches_per_pixel = (1.0 / dpi) * inches_per_paper_inch

        return (inches_per_pixel, "inches")

    # Metric ratio: 1:100, 1:50, 1 : 200
    metric_pattern = r"1\s*:\s*(\d+)"
    match = re.match(metric_pattern, scale_str)
    if match:
        scale_ratio = float(match.group(1))
        # 1 paper unit = scale_ratio real units
        # At given DPI, 1 pixel = 1/dpi paper inches = 25.4/dpi paper mm
        mm_per_pixel = (25.4 / dpi) * scale_ratio

        return (mm_per_pixel, "mm")

    return (None, None)


def parse_sheet_size(sheet_str: str) -> Tuple[Optional[float], Optional[float]]:
    """
    Parse sheet size to (width_inches, height_inches).

    Supports:
    - Standard names: "ARCH D", "ANSI B", "A1"
    - Dimensions: "24x36", "24\" x 36\"", "24 x 36"
    - With descriptive text: "ARCH D (24x36)"

    Args:
        sheet_str: Sheet size description

    Returns:
        Tuple of (width_inches, height_inches) or (None, None) if parsing fails

    Examples:
        >>> parse_sheet_size("ARCH D")
        (24.0, 36.0)
        >>> parse_sheet_size("24x36")
        (24.0, 36.0)
    """
    if not sheet_str:
        return (None, None)

    sheet_str = sheet_str.upper().strip()

    # Try to match known sheet sizes
    for name, (width, height) in SHEET_SIZES.items():
        if name.upper() in sheet_str:
            return (width, height)

    # Try to parse dimensions: 24x36, 24"x36", 24 x 36
    dim_pattern = r"(\d+(?:\.\d+)?)\s*[\"\u2033]?\s*[xX\u00d7]\s*(\d+(?:\.\d+)?)\s*[\"\u2033]?"
    match = re.search(dim_pattern, sheet_str)
    if match:
        width = float(match.group(1))
        height = float(match.group(2))
        return (width, height)

    return (None, None)


def calibrate_from_dimension(
    hint: CalibrationHint,
    image_width: int,
    image_height: int,
) -> Optional[ScaleCalibration]:
    """
    Calibrate scale using a known dimension measurement.

    This is the highest-confidence calibration method because it uses
    an actual measured dimension from the drawing.

    Args:
        hint: CalibrationHint with pixel and real measurements
        image_width: Image width in pixels
        image_height: Image height in pixels

    Returns:
        ScaleCalibration or None if calibration fails
    """
    if hint.pixel_measurement <= 0:
        return None

    real_value, real_unit = parse_measurement(hint.real_measurement)
    if real_value is None or real_value <= 0:
        return None

    # Scale factor = real units / pixels
    scale_factor = real_value / hint.pixel_measurement

    logger.debug(
        "calibrated_from_dimension",
        pixel_measurement=hint.pixel_measurement,
        real_measurement=hint.real_measurement,
        real_value=real_value,
        real_unit=real_unit,
        scale_factor=scale_factor,
    )

    return ScaleCalibration(
        scale_factor=scale_factor,
        units=real_unit,
        confidence=min(hint.confidence, 0.95),  # Cap at 0.95
        method="dimension",
        image_width=image_width,
        image_height=image_height,
        source_description=f"Dimension: {hint.real_measurement} = {hint.pixel_measurement}px ({hint.description})",
    )


def calibrate_from_scale_notation(
    scale_str: str,
    dpi: int,
    image_width: int,
    image_height: int,
) -> Optional[ScaleCalibration]:
    """
    Calibrate scale using scale notation from title block.

    Args:
        scale_str: Scale notation string (e.g., "1/4\" = 1'-0\"")
        dpi: DPI of the rendered image
        image_width: Image width in pixels
        image_height: Image height in pixels

    Returns:
        ScaleCalibration or None if calibration fails
    """
    units_per_pixel, units = parse_scale_notation(scale_str, dpi)
    if units_per_pixel is None:
        return None

    logger.debug(
        "calibrated_from_scale_notation",
        scale_str=scale_str,
        dpi=dpi,
        units_per_pixel=units_per_pixel,
        units=units,
    )

    return ScaleCalibration(
        scale_factor=units_per_pixel,
        units=units,
        confidence=0.85,
        method="scale_notation",
        image_width=image_width,
        image_height=image_height,
        source_description=f"Scale notation: {scale_str}",
    )


def calibrate_from_sheet_size(
    sheet_str: str,
    image_width: int,
    image_height: int,
) -> Optional[ScaleCalibration]:
    """
    Calibrate scale using sheet size.

    This assumes the image represents the full sheet at actual size.
    Lower confidence because we don't know the drawing scale.

    Args:
        sheet_str: Sheet size description
        image_width: Image width in pixels
        image_height: Image height in pixels

    Returns:
        ScaleCalibration or None if calibration fails
    """
    sheet_width, sheet_height = parse_sheet_size(sheet_str)
    if sheet_width is None:
        return None

    # Calculate scale factor using both dimensions and average
    factor_x = sheet_width / image_width
    factor_y = sheet_height / image_height

    # Use geometric mean for better accuracy
    scale_factor = math.sqrt(factor_x * factor_y)

    # Check aspect ratio consistency
    aspect_ratio_sheet = sheet_width / sheet_height
    aspect_ratio_image = image_width / image_height
    aspect_ratio_diff = abs(aspect_ratio_sheet - aspect_ratio_image) / aspect_ratio_sheet

    # Adjust confidence based on aspect ratio match
    confidence = 0.70 - (aspect_ratio_diff * 0.3)  # Reduce confidence if aspect ratios differ
    confidence = max(0.4, min(0.75, confidence))

    logger.debug(
        "calibrated_from_sheet_size",
        sheet_str=sheet_str,
        sheet_width=sheet_width,
        sheet_height=sheet_height,
        scale_factor=scale_factor,
        confidence=confidence,
    )

    return ScaleCalibration(
        scale_factor=scale_factor,
        units="inches",
        confidence=confidence,
        method="sheet_size",
        image_width=image_width,
        image_height=image_height,
        source_description=f"Sheet size: {sheet_str} ({sheet_width}x{sheet_height} inches)",
    )


def calibrate_from_dpi_default(
    dpi: int,
    image_width: int,
    image_height: int,
) -> ScaleCalibration:
    """
    Create a default calibration based on DPI.

    This is the fallback when no other calibration method works.
    Assumes 1 pixel = 1/DPI inches (paper space).

    Args:
        dpi: DPI of the rendered image
        image_width: Image width in pixels
        image_height: Image height in pixels

    Returns:
        ScaleCalibration (always succeeds)
    """
    # 1 pixel = 1/DPI paper inches
    # This is effectively 1:1 paper scale
    scale_factor = 1.0 / dpi

    logger.debug(
        "calibrated_from_dpi_default",
        dpi=dpi,
        scale_factor=scale_factor,
    )

    return ScaleCalibration(
        scale_factor=scale_factor,
        units="inches",
        confidence=0.30,  # Low confidence - just a guess
        method="dpi_default",
        image_width=image_width,
        image_height=image_height,
        source_description=f"DPI default: 1px = 1/{dpi} inches (paper space)",
    )


def calibrate_from_analysis(
    analysis: DrawingAnalysis,
    image_width: int,
    image_height: int,
    image_dpi: int = 300,
    prefer_units: Optional[str] = None,
) -> ScaleCalibration:
    """
    Determine scale factor using multiple methods, ranked by confidence.

    This is the MAIN calibration function for Phase 3. It tries multiple
    calibration methods in order of confidence and returns the best one.

    Calibration method priority:
    1. Known dimension (highest confidence ~90%)
    2. Scale notation (high confidence ~85%)
    3. Sheet size (medium confidence ~70%)
    4. DPI default (low confidence ~30%)

    Args:
        analysis: DrawingAnalysis from Phase 2
        image_width: Image width in pixels
        image_height: Image height in pixels
        image_dpi: DPI used for rendering (default 300)
        prefer_units: Preferred output units (will convert if different)

    Returns:
        ScaleCalibration with the highest confidence method

    Example:
        >>> analysis = await analyze_drawing("floor_plan.png")
        >>> calibration = calibrate_from_analysis(
        ...     analysis=analysis,
        ...     image_width=3600,
        ...     image_height=2400,
        ...     image_dpi=300,
        ... )
        >>> print(f"Scale: {calibration.scale_factor:.6f} {calibration.units}/px")
        >>> print(f"Method: {calibration.method} (confidence: {calibration.confidence:.0%})")
        >>>
        >>> # Convert coordinates
        >>> dwg_x, dwg_y = calibration.to_dwg(100, 200)
    """
    calibrations: List[ScaleCalibration] = []

    logger.info(
        "calibrating_coordinates",
        image_width=image_width,
        image_height=image_height,
        image_dpi=image_dpi,
        calibration_hints=len(analysis.calibration_hints),
        scale=analysis.scale,
        sheet_size=analysis.sheet_size,
    )

    # Method 1: Known dimensions (highest confidence)
    for hint in analysis.calibration_hints:
        if hint.hint_type == "dimension":
            cal = calibrate_from_dimension(hint, image_width, image_height)
            if cal:
                calibrations.append(cal)

    # Method 2: Scale notation (high confidence)
    if analysis.scale:
        cal = calibrate_from_scale_notation(
            analysis.scale,
            image_dpi,
            image_width,
            image_height,
        )
        if cal:
            calibrations.append(cal)

    # Method 3: Sheet size (medium confidence)
    if analysis.sheet_size:
        cal = calibrate_from_sheet_size(
            analysis.sheet_size,
            image_width,
            image_height,
        )
        if cal:
            calibrations.append(cal)

    # Method 4: DPI default (always available, low confidence)
    calibrations.append(
        calibrate_from_dpi_default(image_dpi, image_width, image_height)
    )

    # Sort by confidence (descending) and select best
    calibrations.sort(key=lambda c: c.confidence, reverse=True)
    best = calibrations[0]

    logger.info(
        "calibration_selected",
        method=best.method,
        confidence=best.confidence,
        scale_factor=best.scale_factor,
        units=best.units,
        alternatives=len(calibrations) - 1,
    )

    # Convert units if preferred
    if prefer_units and prefer_units != best.units:
        try:
            best = best.convert_units_to(prefer_units)
            logger.debug(
                "calibration_units_converted",
                from_units=calibrations[0].units,
                to_units=prefer_units,
                new_scale_factor=best.scale_factor,
            )
        except ValueError as e:
            logger.warning("unit_conversion_failed", error=str(e))

    return best


def calibrate_manual(
    scale_factor: float,
    units: str,
    image_width: int,
    image_height: int,
    origin_offset: Tuple[float, float] = (0.0, 0.0),
) -> ScaleCalibration:
    """
    Create a manual calibration with known scale factor.

    Use this when you know the exact scale factor and don't need
    automatic detection from the drawing analysis.

    Args:
        scale_factor: DWG units per pixel
        units: Unit system ("inches", "feet", "mm", "m")
        image_width: Image width in pixels
        image_height: Image height in pixels
        origin_offset: Optional origin offset (dwg_x, dwg_y)

    Returns:
        ScaleCalibration with manual settings

    Example:
        >>> calibration = calibrate_manual(
        ...     scale_factor=0.16,  # 1 pixel = 0.16 inches
        ...     units="inches",
        ...     image_width=3600,
        ...     image_height=2400,
        ... )
    """
    return ScaleCalibration(
        scale_factor=scale_factor,
        units=units,
        confidence=1.0,  # Manual = full confidence
        method="manual",
        image_width=image_width,
        image_height=image_height,
        origin_offset=origin_offset,
        source_description=f"Manual: {scale_factor} {units}/pixel",
    )


def estimate_drawing_bounds(
    analysis: DrawingAnalysis,
    calibration: ScaleCalibration,
) -> Tuple[Tuple[float, float], Tuple[float, float]]:
    """
    Estimate the drawing bounds in DWG coordinates.

    Uses the drawing area region from analysis if available,
    otherwise uses the full image bounds.

    Args:
        analysis: DrawingAnalysis from Phase 2
        calibration: ScaleCalibration from Phase 3

    Returns:
        Tuple of ((min_x, min_y), (max_x, max_y)) in DWG coordinates
    """
    if analysis.drawing_area:
        # Use drawing area bounds
        x1, y1, x2, y2 = (
            analysis.drawing_area.x1,
            analysis.drawing_area.y1,
            analysis.drawing_area.x2,
            analysis.drawing_area.y2,
        )
    else:
        # Use full image
        x1, y1 = 0, 0
        x2, y2 = calibration.image_width, calibration.image_height

    # Convert corners to DWG coordinates
    # Note: y1 in image becomes the max y in DWG (top of image)
    min_dwg = calibration.to_dwg(x1, y2)  # Bottom-left in DWG
    max_dwg = calibration.to_dwg(x2, y1)  # Top-right in DWG

    return (min_dwg, max_dwg)
