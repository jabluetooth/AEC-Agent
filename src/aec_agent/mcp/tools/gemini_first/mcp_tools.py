"""
MCP Tools for Gemini-First PDF to AutoCAD Pipeline.

This module exposes the Gemini-First pipeline tools as MCP tools:
- Phase 1: PDF Intake & Rendering
- Phase 2: Gemini Understanding (Drawing Analysis)
- Phase 3: Coordinate Calibration (Map Pixels to DWG Units)
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
)

logger = structlog.get_logger(__name__)


@mcp.tool()
async def gemini_render_pdf(
    file_path: str,
    page: int = 1,
    dpi: int = 300,
    output_dir: Optional[str] = None,
    convert_grayscale: bool = True,
) -> dict[str, Any]:
    """
    Render a PDF page to a high-quality image for Gemini Vision analysis.

    This is Phase 1 of the Gemini-First pipeline. Unlike bitonal conversion,
    this preserves full grayscale/color information for better AI analysis.

    Args:
        file_path: Path to the PDF file
        page: Page number to render (1-indexed)
        dpi: Resolution (72-1200, default 300)
        output_dir: Output directory (optional)
        convert_grayscale: Convert to grayscale if no color (default True)

    Returns:
        Success result with image path and metadata, or error result.

    Example:
        >>> result = await gemini_render_pdf("drawing.pdf", page=1, dpi=300)
        >>> if result["success"]:
        ...     print(f"Image: {result['data']['image_path']}")
        ...     print(f"Size: {result['data']['width_px']}x{result['data']['height_px']}")
    """
    logger.info(
        "gemini_render_pdf_called",
        file_path=file_path,
        page=page,
        dpi=dpi,
    )

    # Validate inputs
    pdf_path = Path(file_path)
    if not pdf_path.exists():
        return error_result(
            ErrorCode.ELEMENT_NOT_FOUND,
            f"PDF file not found: {file_path}",
        )

    if not pdf_path.suffix.lower() == ".pdf":
        return error_result(
            ErrorCode.INVALID_PARAMS,
            f"File is not a PDF: {file_path}",
        )

    if not 72 <= dpi <= 1200:
        return error_result(
            ErrorCode.INVALID_PARAMS,
            f"DPI must be between 72 and 1200, got {dpi}",
        )

    if page < 1:
        return error_result(
            ErrorCode.INVALID_PARAMS,
            f"Page number must be >= 1, got {page}",
        )

    try:
        # Convert 1-indexed page to 0-indexed
        page_number = page - 1

        # Render the PDF
        result = await render_pdf_high_quality_async(
            pdf_path=pdf_path,
            page_number=page_number,
            dpi=dpi,
            output_dir=Path(output_dir) if output_dir else None,
            convert_grayscale=convert_grayscale,
        )

        logger.info(
            "gemini_render_pdf_success",
            image_path=str(result.image_path),
            width_px=result.width_px,
            height_px=result.height_px,
            color_mode=result.color_mode,
        )

        return success_result(
            data=result.to_dict(),
            message=f"Rendered page {page} to {result.image_path.name} "
                    f"({result.width_px}x{result.height_px}, {result.color_mode})",
        )

    except FileNotFoundError as e:
        return error_result(ErrorCode.ELEMENT_NOT_FOUND, str(e))
    except ValueError as e:
        return error_result(ErrorCode.INVALID_PARAMS, str(e))
    except Exception as e:
        logger.exception("gemini_render_pdf_failed", error=str(e))
        return error_result(ErrorCode.INTERNAL_ERROR, f"Failed to render PDF: {e}")


@mcp.tool()
async def gemini_get_pdf_info(file_path: str) -> dict[str, Any]:
    """
    Get information about a PDF file without rendering.

    Useful for determining page count and dimensions before rendering.

    Args:
        file_path: Path to the PDF file

    Returns:
        Success result with PDF metadata, or error result.

    Example:
        >>> result = await gemini_get_pdf_info("drawing.pdf")
        >>> if result["success"]:
        ...     print(f"Pages: {result['data']['page_count']}")
        ...     for page in result['data']['pages']:
        ...         print(f"  Page {page['page_number']}: {page['width_inches']}x{page['height_inches']} inches")
    """
    logger.info("gemini_get_pdf_info_called", file_path=file_path)

    pdf_path = Path(file_path)
    if not pdf_path.exists():
        return error_result(
            ErrorCode.ELEMENT_NOT_FOUND,
            f"PDF file not found: {file_path}",
        )

    try:
        info = get_pdf_info(pdf_path)

        return success_result(
            data=info.to_dict(),
            message=f"PDF has {info.page_count} page(s)",
        )

    except Exception as e:
        logger.exception("gemini_get_pdf_info_failed", error=str(e))
        return error_result(ErrorCode.INTERNAL_ERROR, f"Failed to get PDF info: {e}")


@mcp.tool()
async def gemini_render_all_pages(
    file_path: str,
    dpi: int = 300,
    output_dir: Optional[str] = None,
    convert_grayscale: bool = True,
) -> dict[str, Any]:
    """
    Render all pages of a PDF to high-quality images.

    Args:
        file_path: Path to the PDF file
        dpi: Resolution (72-1200, default 300)
        output_dir: Output directory (optional)
        convert_grayscale: Convert to grayscale if no color (default True)

    Returns:
        Success result with list of image paths and metadata, or error result.

    Example:
        >>> result = await gemini_render_all_pages("multipage.pdf")
        >>> if result["success"]:
        ...     for page_result in result['data']['pages']:
        ...         print(f"Page {page_result['page_number']}: {page_result['image_path']}")
    """
    logger.info(
        "gemini_render_all_pages_called",
        file_path=file_path,
        dpi=dpi,
    )

    pdf_path = Path(file_path)
    if not pdf_path.exists():
        return error_result(
            ErrorCode.ELEMENT_NOT_FOUND,
            f"PDF file not found: {file_path}",
        )

    try:
        results = render_all_pages(
            pdf_path=pdf_path,
            dpi=dpi,
            output_dir=Path(output_dir) if output_dir else None,
            convert_grayscale=convert_grayscale,
        )

        pages_data = [r.to_dict() for r in results]

        logger.info(
            "gemini_render_all_pages_success",
            page_count=len(results),
        )

        return success_result(
            data={
                "page_count": len(results),
                "pages": pages_data,
            },
            message=f"Rendered {len(results)} page(s)",
        )

    except Exception as e:
        logger.exception("gemini_render_all_pages_failed", error=str(e))
        return error_result(ErrorCode.INTERNAL_ERROR, f"Failed to render PDF pages: {e}")


@mcp.tool()
async def gemini_compare_rendering_quality(
    file_path: str,
    page: int = 1,
    dpi: int = 300,
) -> dict[str, Any]:
    """
    Compare high-quality rendering with bitonal conversion.

    This demonstrates why Gemini-First (high-quality) is better than
    the existing bitonal approach by showing information metrics.

    Args:
        file_path: Path to the PDF file
        page: Page number to compare (1-indexed)
        dpi: Resolution for comparison

    Returns:
        Success result with comparison metrics, or error result.
    """
    logger.info(
        "gemini_compare_rendering_quality_called",
        file_path=file_path,
        page=page,
    )

    pdf_path = Path(file_path)
    if not pdf_path.exists():
        return error_result(
            ErrorCode.ELEMENT_NOT_FOUND,
            f"PDF file not found: {file_path}",
        )

    try:
        from .pdf_intake import compare_with_bitonal

        comparison = compare_with_bitonal(
            pdf_path=pdf_path,
            page_number=page - 1,  # Convert to 0-indexed
            dpi=dpi,
        )

        return success_result(
            data=comparison,
            message=f"High-quality preserves {comparison['information_preserved']} more information",
        )

    except Exception as e:
        logger.exception("gemini_compare_rendering_quality_failed", error=str(e))
        return error_result(ErrorCode.INTERNAL_ERROR, f"Failed to compare: {e}")


# ==============================================================================
# PHASE 2: Gemini Understanding (Drawing Analysis)
# ==============================================================================


@mcp.tool()
async def gemini_analyze_drawing(
    image_path: str,
    model: str = "gemini-1.5-pro",
    context: Optional[str] = None,
) -> dict[str, Any]:
    """
    Analyze a drawing image using Gemini Vision AI.

    This is Phase 2 of the Gemini-First pipeline. It uses Gemini Vision to
    fully understand the drawing BEFORE any extraction, identifying:
    - Drawing type (floor_plan, electrical, mechanical, etc.)
    - All elements with pixel locations (lines, arcs, circles, text, symbols)
    - Scale and calibration hints
    - Recommended extraction strategy

    Args:
        image_path: Path to the drawing image (PNG, JPG, etc.)
        model: Gemini model to use ("gemini-1.5-pro" or "gemini-1.5-flash")
        context: Optional context/hints about the drawing

    Returns:
        Success result with complete drawing analysis, or error result.

    Example:
        >>> result = await gemini_analyze_drawing("floor_plan.png")
        >>> if result["success"]:
        ...     data = result["data"]
        ...     print(f"Type: {data['drawing_analysis']['type']}")
        ...     print(f"Complexity: {data['drawing_analysis']['complexity']}")
        ...     print(f"Total elements: {data['metadata']['total_elements']}")
    """
    logger.info(
        "gemini_analyze_drawing_called",
        image_path=image_path,
        model=model,
    )

    # Validate inputs
    img_path = Path(image_path)
    if not img_path.exists():
        return error_result(
            ErrorCode.ELEMENT_NOT_FOUND,
            f"Image file not found: {image_path}",
        )

    valid_extensions = {".png", ".jpg", ".jpeg", ".tiff", ".tif", ".bmp", ".gif"}
    if img_path.suffix.lower() not in valid_extensions:
        return error_result(
            ErrorCode.INVALID_PARAMS,
            f"Invalid image format: {img_path.suffix}. Supported: {valid_extensions}",
        )

    try:
        # Analyze the drawing
        analyzer = DrawingAnalyzer(model=model)
        analysis = await analyzer.analyze(img_path, context=context)

        logger.info(
            "gemini_analyze_drawing_success",
            drawing_type=analysis.drawing_type,
            complexity=analysis.complexity,
            total_elements=analysis.total_elements,
        )

        return success_result(
            data=analysis.to_dict(),
            message=f"Analyzed {analysis.drawing_type} drawing "
                    f"({analysis.complexity} complexity, {analysis.total_elements} elements)",
        )

    except FileNotFoundError as e:
        return error_result(ErrorCode.ELEMENT_NOT_FOUND, str(e))
    except ValueError as e:
        return error_result(ErrorCode.INVALID_PARAMS, str(e))
    except Exception as e:
        logger.exception("gemini_analyze_drawing_failed", error=str(e))
        return error_result(ErrorCode.INTERNAL_ERROR, f"Failed to analyze drawing: {e}")


@mcp.tool()
async def gemini_analyze_pdf(
    file_path: str,
    page: int = 1,
    dpi: int = 300,
    model: str = "gemini-1.5-pro",
    context: Optional[str] = None,
) -> dict[str, Any]:
    """
    Render a PDF page and analyze it with Gemini Vision in one step.

    This combines Phase 1 (PDF Intake) and Phase 2 (Gemini Understanding)
    for convenience. Use this when you want to go directly from PDF to
    complete drawing analysis.

    Args:
        file_path: Path to the PDF file
        page: Page number to analyze (1-indexed)
        dpi: Resolution for rendering (72-1200, default 300)
        model: Gemini model to use
        context: Optional context/hints about the drawing

    Returns:
        Success result with rendering info and drawing analysis, or error result.

    Example:
        >>> result = await gemini_analyze_pdf("drawings.pdf", page=1)
        >>> if result["success"]:
        ...     data = result["data"]
        ...     print(f"Image: {data['render']['image_path']}")
        ...     print(f"Type: {data['analysis']['drawing_analysis']['type']}")
    """
    logger.info(
        "gemini_analyze_pdf_called",
        file_path=file_path,
        page=page,
        dpi=dpi,
        model=model,
    )

    # Validate PDF path
    pdf_path = Path(file_path)
    if not pdf_path.exists():
        return error_result(
            ErrorCode.ELEMENT_NOT_FOUND,
            f"PDF file not found: {file_path}",
        )

    if not pdf_path.suffix.lower() == ".pdf":
        return error_result(
            ErrorCode.INVALID_PARAMS,
            f"File is not a PDF: {file_path}",
        )

    try:
        # Phase 1: Render PDF
        render_result = await render_pdf_high_quality_async(
            pdf_path=pdf_path,
            page_number=page - 1,  # Convert to 0-indexed
            dpi=dpi,
            convert_grayscale=True,
        )

        logger.info(
            "gemini_analyze_pdf_rendered",
            image_path=str(render_result.image_path),
        )

        # Phase 2: Analyze with Gemini
        analyzer = DrawingAnalyzer(model=model)
        analysis = await analyzer.analyze(render_result.image_path, context=context)

        logger.info(
            "gemini_analyze_pdf_success",
            drawing_type=analysis.drawing_type,
            total_elements=analysis.total_elements,
        )

        return success_result(
            data={
                "render": render_result.to_dict(),
                "analysis": analysis.to_dict(),
            },
            message=f"Rendered page {page} and analyzed as {analysis.drawing_type} "
                    f"({analysis.total_elements} elements)",
        )

    except FileNotFoundError as e:
        return error_result(ErrorCode.ELEMENT_NOT_FOUND, str(e))
    except ValueError as e:
        return error_result(ErrorCode.INVALID_PARAMS, str(e))
    except Exception as e:
        logger.exception("gemini_analyze_pdf_failed", error=str(e))
        return error_result(ErrorCode.INTERNAL_ERROR, f"Failed to analyze PDF: {e}")


@mcp.tool()
async def gemini_get_extraction_strategy(
    image_path: str,
    model: str = "gemini-1.5-flash",
) -> dict[str, Any]:
    """
    Get recommended extraction strategy for a drawing without full analysis.

    This is a lighter-weight version of gemini_analyze_drawing that focuses
    specifically on determining the best extraction approach.

    Args:
        image_path: Path to the drawing image
        model: Gemini model to use (flash recommended for speed)

    Returns:
        Success result with extraction strategy recommendation, or error result.

    Example:
        >>> result = await gemini_get_extraction_strategy("complex_hvac.png")
        >>> if result["success"]:
        ...     strategy = result["data"]["extraction_strategy"]
        ...     print(f"Primary: {strategy['primary_strategy']}")
        ...     print(f"Rationale: {strategy['rationale']}")
    """
    logger.info(
        "gemini_get_extraction_strategy_called",
        image_path=image_path,
        model=model,
    )

    img_path = Path(image_path)
    if not img_path.exists():
        return error_result(
            ErrorCode.ELEMENT_NOT_FOUND,
            f"Image file not found: {image_path}",
        )

    try:
        # Use fast model for quick strategy assessment
        analyzer = DrawingAnalyzer(model=model)
        analysis = await analyzer.analyze(img_path)

        strategy = analysis.extraction_strategy.to_dict()

        logger.info(
            "gemini_get_extraction_strategy_success",
            primary_strategy=strategy["primary_strategy"],
        )

        return success_result(
            data={
                "drawing_type": analysis.drawing_type,
                "complexity": analysis.complexity,
                "extraction_strategy": strategy,
            },
            message=f"Recommended strategy: {strategy['primary_strategy']} "
                    f"({strategy['rationale'][:50]}...)" if strategy.get('rationale') else
                    f"Recommended strategy: {strategy['primary_strategy']}",
        )

    except Exception as e:
        logger.exception("gemini_get_extraction_strategy_failed", error=str(e))
        return error_result(ErrorCode.INTERNAL_ERROR, f"Failed to get strategy: {e}")


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


@mcp.tool()
async def gemini_analyze_pdf_calibrated(
    file_path: str,
    page: int = 1,
    dpi: int = 300,
    model: str = "gemini-1.5-pro",
    prefer_units: Optional[str] = None,
) -> dict[str, Any]:
    """
    Render, analyze, and calibrate a PDF page in one step.

    This combines Phase 1 (PDF Intake), Phase 2 (Gemini Understanding),
    and Phase 3 (Coordinate Calibration) for convenience. Use this when
    you want to go directly from PDF to calibrated drawing analysis.

    Args:
        file_path: Path to the PDF file
        page: Page number to analyze (1-indexed)
        dpi: Resolution for rendering (72-1200, default 300)
        model: Gemini model to use
        prefer_units: Preferred output units (inches, feet, mm, m)

    Returns:
        Success result with rendering info, drawing analysis, and calibration.

    Example:
        >>> result = await gemini_analyze_pdf_calibrated("floor_plan.pdf")
        >>> if result["success"]:
        ...     data = result["data"]
        ...     print(f"Image: {data['render']['image_path']}")
        ...     print(f"Type: {data['analysis']['drawing_analysis']['type']}")
        ...     print(f"Scale: {data['calibration']['scale_factor']} {data['calibration']['units']}/px")
    """
    logger.info(
        "gemini_analyze_pdf_calibrated_called",
        file_path=file_path,
        page=page,
        dpi=dpi,
        model=model,
    )

    # Validate PDF path
    pdf_path = Path(file_path)
    if not pdf_path.exists():
        return error_result(
            ErrorCode.ELEMENT_NOT_FOUND,
            f"PDF file not found: {file_path}",
        )

    if not pdf_path.suffix.lower() == ".pdf":
        return error_result(
            ErrorCode.INVALID_PARAMS,
            f"File is not a PDF: {file_path}",
        )

    try:
        # Phase 1: Render PDF
        render_result = await render_pdf_high_quality_async(
            pdf_path=pdf_path,
            page_number=page - 1,  # Convert to 0-indexed
            dpi=dpi,
            convert_grayscale=True,
        )

        logger.info(
            "gemini_analyze_pdf_calibrated_rendered",
            image_path=str(render_result.image_path),
        )

        # Phase 2: Analyze with Gemini
        analyzer = DrawingAnalyzer(model=model)
        analysis = await analyzer.analyze(render_result.image_path)

        # Phase 3: Calibrate coordinates
        calibration = calibrate_from_analysis(
            analysis=analysis,
            image_width=render_result.width_px,
            image_height=render_result.height_px,
            image_dpi=dpi,
            prefer_units=prefer_units,
        )

        # Estimate drawing bounds
        bounds = estimate_drawing_bounds(analysis, calibration)

        logger.info(
            "gemini_analyze_pdf_calibrated_success",
            drawing_type=analysis.drawing_type,
            total_elements=analysis.total_elements,
            calibration_method=calibration.method,
            calibration_confidence=calibration.confidence,
        )

        return success_result(
            data={
                "render": render_result.to_dict(),
                "analysis": analysis.to_dict(),
                "calibration": calibration.to_dict(),
                "drawing_bounds": {
                    "min": {"x": bounds[0][0], "y": bounds[0][1]},
                    "max": {"x": bounds[1][0], "y": bounds[1][1]},
                    "units": calibration.units,
                },
            },
            message=f"Rendered page {page}, analyzed as {analysis.drawing_type}, "
                    f"calibrated at {calibration.scale_factor:.6f} {calibration.units}/px "
                    f"({calibration.method}, {calibration.confidence:.0%})",
        )

    except FileNotFoundError as e:
        return error_result(ErrorCode.ELEMENT_NOT_FOUND, str(e))
    except ValueError as e:
        return error_result(ErrorCode.INVALID_PARAMS, str(e))
    except Exception as e:
        logger.exception("gemini_analyze_pdf_calibrated_failed", error=str(e))
        return error_result(ErrorCode.INTERNAL_ERROR, f"Failed to analyze PDF: {e}")


# ==============================================================================
# PHASE 4: Adaptive Extraction (Direct / Guided / Selective)
# ==============================================================================


@mcp.tool()
async def gemini_extract_entities(
    image_path: str,
    image_width: int,
    image_height: int,
    image_dpi: int = 300,
    model: str = "gemini-1.5-pro",
    include_opencv: bool = True,
) -> dict[str, Any]:
    """
    Extract AutoCAD entities from a drawing image using Gemini-First pipeline.

    This is Phase 4 of the Gemini-First pipeline. It combines Phases 2-4:
    1. Analyzes the drawing with Gemini Vision
    2. Calibrates pixel coordinates to DWG units
    3. Extracts entities using the optimal strategy for each element

    Returns entities ready for AutoCAD creation, including:
    - Lines, arcs, circles
    - Text (MTEXT)
    - Block insertions (symbols)
    - Dimensions

    Args:
        image_path: Path to the drawing image
        image_width: Image width in pixels
        image_height: Image height in pixels
        image_dpi: DPI used for rendering (default 300)
        model: Gemini model for analysis
        include_opencv: Whether to use OpenCV for special regions

    Returns:
        Success result with extracted entities and statistics.

    Example:
        >>> result = await gemini_extract_entities("floor_plan.png", 3600, 2400)
        >>> if result["success"]:
        ...     data = result["data"]
        ...     print(f"Entities: {data['statistics']['total_entities']}")
        ...     for entity in data["entities"][:5]:
        ...         print(f"  {entity['entity_type']}: {entity['layer']}")
    """
    logger.info(
        "gemini_extract_entities_called",
        image_path=image_path,
        image_width=image_width,
        image_height=image_height,
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
        # Phase 2: Analyze drawing
        analyzer = DrawingAnalyzer(model=model)
        analysis = await analyzer.analyze(img_path)

        # Phase 3: Calibrate coordinates
        calibration = calibrate_from_analysis(
            analysis=analysis,
            image_width=image_width,
            image_height=image_height,
            image_dpi=image_dpi,
        )

        # Phase 4: Extract entities
        if include_opencv:
            result = await extract_all(analysis, calibration, img_path)
        else:
            result = await extract_direct_only(analysis, calibration)

        logger.info(
            "gemini_extract_entities_success",
            total_entities=result.total_entities,
            direct_count=result.direct_count,
            opencv_count=result.opencv_count,
        )

        return success_result(
            data=result.to_dict(),
            message=f"Extracted {result.total_entities} entities "
                    f"({result.direct_count} direct, {result.opencv_count} OpenCV)",
        )

    except FileNotFoundError as e:
        return error_result(ErrorCode.ELEMENT_NOT_FOUND, str(e))
    except ValueError as e:
        return error_result(ErrorCode.INVALID_PARAMS, str(e))
    except Exception as e:
        logger.exception("gemini_extract_entities_failed", error=str(e))
        return error_result(ErrorCode.INTERNAL_ERROR, f"Failed to extract entities: {e}")


@mcp.tool()
async def gemini_extract_pdf_entities(
    file_path: str,
    page: int = 1,
    dpi: int = 300,
    model: str = "gemini-1.5-pro",
    include_opencv: bool = True,
) -> dict[str, Any]:
    """
    Complete PDF-to-entities pipeline: render, analyze, calibrate, and extract.

    This combines all four phases of the Gemini-First pipeline:
    1. Phase 1: Render PDF to high-quality image
    2. Phase 2: Analyze with Gemini Vision
    3. Phase 3: Calibrate coordinates
    4. Phase 4: Extract entities

    Args:
        file_path: Path to the PDF file
        page: Page number to process (1-indexed)
        dpi: Resolution for rendering (default 300)
        model: Gemini model for analysis
        include_opencv: Whether to use OpenCV for special regions

    Returns:
        Success result with rendered image info, analysis, calibration,
        and extracted entities.

    Example:
        >>> result = await gemini_extract_pdf_entities("drawings.pdf", page=1)
        >>> if result["success"]:
        ...     data = result["data"]
        ...     print(f"Drawing type: {data['analysis']['drawing_analysis']['type']}")
        ...     print(f"Entities: {data['extraction']['statistics']['total_entities']}")
    """
    logger.info(
        "gemini_extract_pdf_entities_called",
        file_path=file_path,
        page=page,
        dpi=dpi,
    )

    # Validate PDF path
    pdf_path = Path(file_path)
    if not pdf_path.exists():
        return error_result(
            ErrorCode.ELEMENT_NOT_FOUND,
            f"PDF file not found: {file_path}",
        )

    if not pdf_path.suffix.lower() == ".pdf":
        return error_result(
            ErrorCode.INVALID_PARAMS,
            f"File is not a PDF: {file_path}",
        )

    try:
        # Phase 1: Render PDF
        render_result = await render_pdf_high_quality_async(
            pdf_path=pdf_path,
            page_number=page - 1,
            dpi=dpi,
            convert_grayscale=True,
        )

        # Phase 2: Analyze with Gemini
        analyzer = DrawingAnalyzer(model=model)
        analysis = await analyzer.analyze(render_result.image_path)

        # Phase 3: Calibrate coordinates
        calibration = calibrate_from_analysis(
            analysis=analysis,
            image_width=render_result.width_px,
            image_height=render_result.height_px,
            image_dpi=dpi,
        )

        # Phase 4: Extract entities
        if include_opencv:
            extraction = await extract_all(
                analysis, calibration, render_result.image_path
            )
        else:
            extraction = await extract_direct_only(analysis, calibration)

        # Get drawing bounds
        bounds = estimate_drawing_bounds(analysis, calibration)

        logger.info(
            "gemini_extract_pdf_entities_success",
            drawing_type=analysis.drawing_type,
            total_entities=extraction.total_entities,
        )

        return success_result(
            data={
                "render": render_result.to_dict(),
                "analysis": analysis.to_dict(),
                "calibration": calibration.to_dict(),
                "extraction": extraction.to_dict(),
                "drawing_bounds": {
                    "min": {"x": bounds[0][0], "y": bounds[0][1]},
                    "max": {"x": bounds[1][0], "y": bounds[1][1]},
                    "units": calibration.units,
                },
            },
            message=f"Processed page {page}: {analysis.drawing_type} drawing, "
                    f"{extraction.total_entities} entities extracted",
        )

    except FileNotFoundError as e:
        return error_result(ErrorCode.ELEMENT_NOT_FOUND, str(e))
    except ValueError as e:
        return error_result(ErrorCode.INVALID_PARAMS, str(e))
    except Exception as e:
        logger.exception("gemini_extract_pdf_entities_failed", error=str(e))
        return error_result(ErrorCode.INTERNAL_ERROR, f"Failed to process PDF: {e}")


@mcp.tool()
async def gemini_get_layer_mapping(
    element_type: str,
    subtype: Optional[str] = None,
) -> dict[str, Any]:
    """
    Get the NCS-compliant layer name for an element type.

    Useful for understanding what layer an element will be placed on
    or for manual entity creation.

    Args:
        element_type: Type of element (wall, duct, outlet, diffuser, etc.)
        subtype: Optional subtype for more specific mapping

    Returns:
        Success result with layer name and mapping details.

    Example:
        >>> result = await gemini_get_layer_mapping("diffuser", "supply_square")
        >>> if result["success"]:
        ...     print(f"Layer: {result['data']['layer']}")
        ...     # Output: Layer: M-DIFF-SUPP
    """
    layer = get_layer_for_element_type(element_type, subtype)

    return success_result(
        data={
            "element_type": element_type,
            "subtype": subtype,
            "layer": layer,
        },
        message=f"Layer for {element_type}: {layer}",
    )


@mcp.tool()
async def gemini_get_block_mapping(
    symbol_type: str,
    subtype: Optional[str] = None,
) -> dict[str, Any]:
    """
    Get the block name for a symbol type.

    Useful for understanding what block will be inserted for a symbol
    or for manual block insertion.

    Args:
        symbol_type: Type of symbol (diffuser, outlet, valve, etc.)
        subtype: Optional subtype for more specific mapping

    Returns:
        Success result with block name.

    Example:
        >>> result = await gemini_get_block_mapping("valve", "gate")
        >>> if result["success"]:
        ...     print(f"Block: {result['data']['block_name']}")
        ...     # Output: Block: P-VALV-GATE
    """
    block_name = get_block_name(symbol_type, subtype)

    return success_result(
        data={
            "symbol_type": symbol_type,
            "subtype": subtype,
            "block_name": block_name,
        },
        message=f"Block for {symbol_type}: {block_name}",
    )


@mcp.tool()
async def gemini_get_required_layers(
    file_path: str,
    page: int = 1,
    dpi: int = 300,
    model: str = "gemini-1.5-flash",
) -> dict[str, Any]:
    """
    Get list of layers required for a PDF drawing.

    Analyzes the drawing and determines which layers need to be created
    before entity insertion.

    Args:
        file_path: Path to the PDF file
        page: Page number to analyze (1-indexed)
        dpi: Resolution for rendering
        model: Gemini model for analysis

    Returns:
        Success result with list of required layers.

    Example:
        >>> result = await gemini_get_required_layers("mechanical.pdf")
        >>> if result["success"]:
        ...     for layer in result["data"]["layers"]:
        ...         print(f"  {layer}")
    """
    pdf_path = Path(file_path)
    if not pdf_path.exists():
        return error_result(
            ErrorCode.ELEMENT_NOT_FOUND,
            f"PDF file not found: {file_path}",
        )

    try:
        # Phase 1: Render PDF
        render_result = await render_pdf_high_quality_async(
            pdf_path=pdf_path,
            page_number=page - 1,
            dpi=dpi,
            convert_grayscale=True,
        )

        # Phase 2: Analyze
        analyzer = DrawingAnalyzer(model=model)
        analysis = await analyzer.analyze(render_result.image_path)

        # Phase 3: Calibrate (minimal)
        calibration = calibrate_from_analysis(
            analysis=analysis,
            image_width=render_result.width_px,
            image_height=render_result.height_px,
            image_dpi=dpi,
        )

        # Phase 4: Extract (direct only for speed)
        extraction = await extract_direct_only(analysis, calibration)

        # Get required layers
        layers = get_required_layers(extraction)

        return success_result(
            data={
                "layers": layers,
                "count": len(layers),
                "drawing_type": analysis.drawing_type,
            },
            message=f"Found {len(layers)} required layers for {analysis.drawing_type}",
        )

    except Exception as e:
        logger.exception("gemini_get_required_layers_failed", error=str(e))
        return error_result(ErrorCode.INTERNAL_ERROR, f"Failed to analyze: {e}")


@mcp.tool()
async def gemini_get_required_blocks(
    file_path: str,
    page: int = 1,
    dpi: int = 300,
    model: str = "gemini-1.5-flash",
) -> dict[str, Any]:
    """
    Get list of block definitions required for a PDF drawing.

    Analyzes the drawing and determines which blocks need to be
    defined or loaded before entity insertion.

    Args:
        file_path: Path to the PDF file
        page: Page number to analyze (1-indexed)
        dpi: Resolution for rendering
        model: Gemini model for analysis

    Returns:
        Success result with list of required block names.

    Example:
        >>> result = await gemini_get_required_blocks("electrical.pdf")
        >>> if result["success"]:
        ...     for block in result["data"]["blocks"]:
        ...         print(f"  {block}")
    """
    pdf_path = Path(file_path)
    if not pdf_path.exists():
        return error_result(
            ErrorCode.ELEMENT_NOT_FOUND,
            f"PDF file not found: {file_path}",
        )

    try:
        # Phase 1: Render PDF
        render_result = await render_pdf_high_quality_async(
            pdf_path=pdf_path,
            page_number=page - 1,
            dpi=dpi,
            convert_grayscale=True,
        )

        # Phase 2: Analyze
        analyzer = DrawingAnalyzer(model=model)
        analysis = await analyzer.analyze(render_result.image_path)

        # Phase 3: Calibrate (minimal)
        calibration = calibrate_from_analysis(
            analysis=analysis,
            image_width=render_result.width_px,
            image_height=render_result.height_px,
            image_dpi=dpi,
        )

        # Phase 4: Extract (direct only for speed)
        extraction = await extract_direct_only(analysis, calibration)

        # Get required blocks
        blocks = get_required_blocks(extraction)

        return success_result(
            data={
                "blocks": blocks,
                "count": len(blocks),
                "drawing_type": analysis.drawing_type,
            },
            message=f"Found {len(blocks)} required blocks for {analysis.drawing_type}",
        )

    except Exception as e:
        logger.exception("gemini_get_required_blocks_failed", error=str(e))
        return error_result(ErrorCode.INTERNAL_ERROR, f"Failed to analyze: {e}")
