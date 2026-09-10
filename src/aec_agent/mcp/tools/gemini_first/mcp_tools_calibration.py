"""
MCP Tools for Gemini-First PDF to AutoCAD Pipeline.

This module exposes the Gemini-First pipeline tools as MCP tools:
- Phase 1: PDF Intake & Rendering
- Phase 2: Gemini Understanding (Drawing Analysis)
- Phase 3: Coordinate Calibration (Map Pixels to DWG Units)
- Phase 4: Adaptive Extraction (Direct / Guided / Selective)
- Phase 5: AutoCAD Entity Creation (Draw in DWG)
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional, Tuple

import structlog

from aec_agent.mcp.server import mcp
from ..base import ErrorCode, error_result, success_result
from .pdf_intake import (
    PDFRenderResult,
    get_pdf_info,
    render_all_pages,
    render_pdf_high_quality,
    render_pdf_high_quality_async,
)
from .gemini_understanding import (
    DrawingAnalysis,
    DrawingAnalyzer,
    analyze_drawing,
)
from .coordinate_calibration import (
    ScaleCalibration,
    calibrate_from_analysis,
    calibrate_manual,
    parse_measurement,
    parse_scale_notation,
    parse_sheet_size,
    estimate_drawing_bounds,
)
from .adaptive_extraction import (
    ExtractionResult,
    EntityToCreate,
    RasterCommand,
    extract_all,
    extract_direct_only,
    get_layer_for_element_type,
    get_block_name,
    get_entities_by_type,
    get_entities_by_layer,
    get_required_layers,
    get_required_blocks,
    # Hybrid extraction (Gemini + OpenCV + YOLO fusion)
    HybridExtractionConfig,
    HybridExtractionResult,
    hybrid_extract_all,
)
from .autocad_creation import (
    AutoCADCreationResult,
    CreationStatistics,
    EntityCreationResult,
    LayerCreationResult,
    create_entities_in_autocad,
    create_entities_batch,
    get_entity_type_stats,
    get_failed_by_type,
)
from .validation import (
    ValidationStatus,
    IssueType,
    IssueSeverity,
    CorrectionAction,
    ValidationIssue,
    Correction,
    CorrectionResult,
    ValidationResult,
    validate_extraction,
    apply_corrections,
    validate_with_gemini,
    get_critical_issues,
    get_issues_by_type,
    summarize_validation,
)


from .mcp_tools_helpers import logger

# ==============================================================================
# PHASE 3: Coordinate Calibration (Map Pixels to DWG Units)
# ==============================================================================


@mcp.tool()
async def gemini_calibrate_coordinates(
    image_path: str,
    image_width: int,
    image_height: int,
    image_dpi: int = 300,
    model: str = "gemini-1.5-flash",
    prefer_units: Optional[str] = None,
) -> dict[str, Any]:
    """
    Calibrate pixel coordinates to DWG units using Gemini analysis.

    This is Phase 3 of the Gemini-First pipeline. It analyzes a drawing image
    to determine the scale factor for converting pixel coordinates to AutoCAD
    drawing units.

    The calibration uses multiple methods in order of confidence:
    1. Known dimensions (90% confidence) - Uses dimension lines with measurements
    2. Scale notation (85% confidence) - Uses scale like "1/4" = 1'-0""
    3. Sheet size (70% confidence) - Uses sheet size like "ARCH D"
    4. DPI default (30% confidence) - Falls back to 1px = 1/DPI inches

    Args:
        image_path: Path to the drawing image
        image_width: Image width in pixels
        image_height: Image height in pixels
        image_dpi: DPI used for rendering (default 300)
        model: Gemini model for analysis (flash recommended for speed)
        prefer_units: Preferred output units (inches, feet, mm, m)

    Returns:
        Success result with calibration data including scale factor, units,
        confidence, and method used.

    Example:
        >>> result = await gemini_calibrate_coordinates(
        ...     "floor_plan.png",
        ...     image_width=3600,
        ...     image_height=2400,
        ...     image_dpi=300
        ... )
        >>> if result["success"]:
        ...     cal = result["data"]
        ...     print(f"Scale: {cal['scale_factor']:.6f} {cal['units']}/pixel")
        ...     print(f"Method: {cal['method']} ({cal['confidence']:.0%} confidence)")
    """
    logger.info(
        "gemini_calibrate_coordinates_called",
        image_path=image_path,
        image_width=image_width,
        image_height=image_height,
        image_dpi=image_dpi,
    )

    # Validate inputs
    img_path = Path(image_path)
    if not img_path.exists():
        return error_result(
            ErrorCode.ELEMENT_NOT_FOUND,
            f"Image file not found: {image_path}",
        )

    if image_width <= 0 or image_height <= 0:
        return error_result(
            ErrorCode.INVALID_PARAMS,
            f"Invalid image dimensions: {image_width}x{image_height}",
        )

    try:
        # Phase 2: Analyze drawing (needed for calibration hints)
        analyzer = DrawingAnalyzer(model=model)
        analysis = await analyzer.analyze(img_path)

        # Phase 3: Calibrate coordinates
        calibration = calibrate_from_analysis(
            analysis=analysis,
            image_width=image_width,
            image_height=image_height,
            image_dpi=image_dpi,
            prefer_units=prefer_units,
        )

        logger.info(
            "gemini_calibrate_coordinates_success",
            method=calibration.method,
            confidence=calibration.confidence,
            scale_factor=calibration.scale_factor,
            units=calibration.units,
        )

        return success_result(
            data=calibration.to_dict(),
            message=f"Calibrated: {calibration.scale_factor:.6f} {calibration.units}/pixel "
                    f"({calibration.method}, {calibration.confidence:.0%} confidence)",
        )

    except FileNotFoundError as e:
        return error_result(ErrorCode.ELEMENT_NOT_FOUND, str(e))
    except ValueError as e:
        return error_result(ErrorCode.INVALID_PARAMS, str(e))
    except Exception as e:
        logger.exception("gemini_calibrate_coordinates_failed", error=str(e))
        return error_result(ErrorCode.INTERNAL_ERROR, f"Failed to calibrate: {e}")


@mcp.tool()
async def gemini_calibrate_manual(
    scale_factor: float,
    units: str,
    image_width: int,
    image_height: int,
    origin_x: float = 0.0,
    origin_y: float = 0.0,
) -> dict[str, Any]:
    """
    Create a manual calibration with known scale factor.

    Use this when you already know the exact scale factor and don't need
    automatic detection. This is useful for:
    - Drawings with known fixed scales
    - Correcting a previous calibration
    - Testing coordinate transforms

    Args:
        scale_factor: DWG units per pixel
        units: Unit system (inches, feet, mm, m)
        image_width: Image width in pixels
        image_height: Image height in pixels
        origin_x: X offset for DWG origin (default 0)
        origin_y: Y offset for DWG origin (default 0)

    Returns:
        Success result with calibration data.

    Example:
        >>> result = await gemini_calibrate_manual(
        ...     scale_factor=0.16,  # 1/4" scale at 300 DPI
        ...     units="inches",
        ...     image_width=3600,
        ...     image_height=2400
        ... )
    """
    logger.info(
        "gemini_calibrate_manual_called",
        scale_factor=scale_factor,
        units=units,
        image_width=image_width,
        image_height=image_height,
    )

    # Validate inputs
    if scale_factor <= 0:
        return error_result(
            ErrorCode.INVALID_PARAMS,
            f"Scale factor must be positive, got {scale_factor}",
        )

    valid_units = {"inches", "feet", "mm", "m"}
    if units not in valid_units:
        return error_result(
            ErrorCode.INVALID_PARAMS,
            f"Invalid units '{units}'. Valid options: {valid_units}",
        )

    if image_width <= 0 or image_height <= 0:
        return error_result(
            ErrorCode.INVALID_PARAMS,
            f"Invalid image dimensions: {image_width}x{image_height}",
        )

    try:
        calibration = calibrate_manual(
            scale_factor=scale_factor,
            units=units,
            image_width=image_width,
            image_height=image_height,
            origin_offset=(origin_x, origin_y),
        )

        logger.info(
            "gemini_calibrate_manual_success",
            scale_factor=scale_factor,
            units=units,
        )

        return success_result(
            data=calibration.to_dict(),
            message=f"Manual calibration: {scale_factor} {units}/pixel",
        )

    except Exception as e:
        logger.exception("gemini_calibrate_manual_failed", error=str(e))
        return error_result(ErrorCode.INTERNAL_ERROR, f"Failed to create calibration: {e}")


@mcp.tool()
async def gemini_convert_coordinates(
    pixel_x: float,
    pixel_y: float,
    scale_factor: float,
    units: str,
    image_height: int,
    origin_x: float = 0.0,
    origin_y: float = 0.0,
) -> dict[str, Any]:
    """
    Convert pixel coordinates to DWG coordinates using a known calibration.

    This is a utility tool for converting individual points or testing
    coordinate transforms without creating a full calibration object.

    Args:
        pixel_x: X coordinate in pixels
        pixel_y: Y coordinate in pixels
        scale_factor: DWG units per pixel
        units: Unit system (inches, feet, mm, m)
        image_height: Image height in pixels (needed for Y-axis flip)
        origin_x: X offset for DWG origin (default 0)
        origin_y: Y offset for DWG origin (default 0)

    Returns:
        Success result with converted DWG coordinates.

    Example:
        >>> result = await gemini_convert_coordinates(
        ...     pixel_x=100,
        ...     pixel_y=200,
        ...     scale_factor=0.16,
        ...     units="inches",
        ...     image_height=2400
        ... )
        >>> if result["success"]:
        ...     dwg = result["data"]
        ...     print(f"DWG: ({dwg['dwg_x']:.2f}, {dwg['dwg_y']:.2f})")
    """
    # Validate inputs
    if scale_factor <= 0:
        return error_result(
            ErrorCode.INVALID_PARAMS,
            f"Scale factor must be positive, got {scale_factor}",
        )

    if image_height <= 0:
        return error_result(
            ErrorCode.INVALID_PARAMS,
            f"Image height must be positive, got {image_height}",
        )

    try:
        # Create temporary calibration for conversion
        calibration = calibrate_manual(
            scale_factor=scale_factor,
            units=units,
            image_width=1,  # Not needed for conversion
            image_height=image_height,
            origin_offset=(origin_x, origin_y),
        )

        # Convert coordinates
        dwg_x, dwg_y = calibration.to_dwg(pixel_x, pixel_y)

        return success_result(
            data={
                "pixel_x": pixel_x,
                "pixel_y": pixel_y,
                "dwg_x": dwg_x,
                "dwg_y": dwg_y,
                "units": units,
                "scale_factor": scale_factor,
            },
            message=f"Converted ({pixel_x:.1f}, {pixel_y:.1f}) px -> ({dwg_x:.2f}, {dwg_y:.2f}) {units}",
        )

    except Exception as e:
        logger.exception("gemini_convert_coordinates_failed", error=str(e))
        return error_result(ErrorCode.INTERNAL_ERROR, f"Failed to convert: {e}")


@mcp.tool()
async def gemini_parse_scale(
    scale_notation: str,
    dpi: int = 300,
) -> dict[str, Any]:
    """
    Parse an architectural scale notation string.

    Useful for understanding scale notations found in title blocks
    or for validating scale settings.

    Supported formats:
    - Imperial: 1/4" = 1'-0", 1/8" = 1'-0", 1" = 1'-0"
    - Metric: 1:100, 1:50, 1:200

    Args:
        scale_notation: The scale notation string to parse
        dpi: DPI for calculating pixels per unit (default 300)

    Returns:
        Success result with parsed scale information.

    Example:
        >>> result = await gemini_parse_scale("1/4\" = 1'-0\"")
        >>> if result["success"]:
        ...     data = result["data"]
        ...     print(f"1 pixel = {data['units_per_pixel']:.6f} {data['units']}")
    """
    if not scale_notation:
        return error_result(
            ErrorCode.INVALID_PARAMS,
            "Scale notation cannot be empty",
        )

    units_per_pixel, units = parse_scale_notation(scale_notation, dpi)

    if units_per_pixel is None:
        return error_result(
            ErrorCode.INVALID_PARAMS,
            f"Could not parse scale notation: '{scale_notation}'. "
            "Supported formats: 1/4\" = 1'-0\", 1:100, etc.",
        )

    return success_result(
        data={
            "scale_notation": scale_notation,
            "units_per_pixel": units_per_pixel,
            "units": units,
            "dpi": dpi,
            "pixels_per_unit": 1.0 / units_per_pixel if units_per_pixel > 0 else 0,
        },
        message=f"Parsed: 1 pixel = {units_per_pixel:.6f} {units}",
    )


@mcp.tool()
async def gemini_parse_measurement(
    measurement: str,
) -> dict[str, Any]:
    """
    Parse a measurement string into numeric value and units.

    Useful for understanding dimension values or converting between formats.

    Supported formats:
    - Feet and inches: 20'-0", 20' 6", 20'-6"
    - Inches: 24", 24 in
    - Millimeters: 100mm, 100 mm
    - Meters: 1.5m
    - Centimeters: 50cm

    Args:
        measurement: The measurement string to parse

    Returns:
        Success result with parsed measurement value and units.

    Example:
        >>> result = await gemini_parse_measurement("20'-6\"")
        >>> if result["success"]:
        ...     print(f"Value: {result['data']['value']} {result['data']['units']}")
        ...     # Output: Value: 246.0 inches
    """
    if not measurement:
        return error_result(
            ErrorCode.INVALID_PARAMS,
            "Measurement string cannot be empty",
        )

    value, units = parse_measurement(measurement)

    if value is None:
        return error_result(
            ErrorCode.INVALID_PARAMS,
            f"Could not parse measurement: '{measurement}'. "
            "Supported formats: 20'-6\", 24\", 100mm, 1.5m, etc.",
        )

    return success_result(
        data={
            "original": measurement,
            "value": value,
            "units": units,
        },
        message=f"Parsed: {measurement} = {value} {units}",
    )
