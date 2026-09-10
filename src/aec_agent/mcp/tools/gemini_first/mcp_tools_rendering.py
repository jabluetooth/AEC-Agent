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
