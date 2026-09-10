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


from .mcp_tools_helpers import logger, _summarize_analysis

# ==============================================================================
# PHASE 2: Gemini Understanding (Drawing Analysis)
# ==============================================================================


@mcp.tool()
async def gemini_analyze_drawing(
    image_path: str,
    model: str = "gemini-1.5-flash",
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
        model: Gemini model to use ("gemini-1.5-flash" recommended - has free tier)
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

        # Return summarized result to reduce token usage
        return success_result(
            data=_summarize_analysis(analysis),
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
    model: str = "gemini-1.5-flash",
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

        # Return summarized result to reduce token usage
        return success_result(
            data={
                "render": {
                    "image_path": str(render_result.image_path),
                    "width_px": render_result.width_px,
                    "height_px": render_result.height_px,
                },
                "analysis": _summarize_analysis(analysis),
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
