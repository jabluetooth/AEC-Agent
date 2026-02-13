"""
MCP Tools for Gemini-First PDF to AutoCAD Pipeline.

This module exposes the Gemini-First pipeline tools as MCP tools:
- Phase 1: PDF Intake & Rendering
- Phase 2: Gemini Understanding (Drawing Analysis)
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

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
