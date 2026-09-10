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


from .mcp_tools_helpers import logger, _summarize_analysis, _summarize_extraction


@mcp.tool()
async def gemini_analyze_pdf_calibrated(
    file_path: str,
    page: int = 1,
    dpi: int = 300,
    model: str = "gemini-1.5-flash",
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
    model: str = "gemini-1.5-flash",
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

        # Return summarized result to reduce token usage
        return success_result(
            data=_summarize_extraction(result),
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
    model: str = "gemini-1.5-flash",
    include_opencv: bool = True,
) -> dict[str, Any]:
    """
    DEPRECATED: Use `vectorize_pdf` instead.

    Complete PDF-to-entities pipeline: render, analyze, calibrate, and extract.

    MIGRATION: Replace with:
        >>> result = await vectorize_pdf(
        ...     pdf_path="drawings.pdf",
        ...     extraction_method="hybrid" if include_opencv else "direct",
        ...     create_in_autocad=False,
        ... )

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

        # Return summarized result to reduce token usage
        return success_result(
            data={
                "render": {"image_path": str(render_result.image_path)},
                "analysis": _summarize_analysis(analysis),
                "calibration": {
                    "scale_factor": calibration.scale_factor,
                    "units": calibration.units,
                    "method": calibration.method,
                    "confidence": f"{calibration.confidence:.0%}",
                },
                "extraction": _summarize_extraction(extraction),
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
