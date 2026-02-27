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

logger = structlog.get_logger(__name__)

# Maximum elements to include in tool results (reduces token usage)
MAX_ELEMENTS_IN_RESULT = 10


async def _store_extraction_to_database(
    extraction: HybridExtractionResult,
    pdf_path: Path,
    calibration: "ScaleCalibration",
) -> dict:
    """
    Store extracted entities to PostgreSQL for semantic search and spatial queries.

    Converts EntityToCreate objects to Element models and stores them in the database.

    Args:
        extraction: Hybrid extraction result with entities
        pdf_path: Source PDF file path
        calibration: Scale calibration for coordinate conversion

    Returns:
        Storage summary dict with counts
    """
    from aec_agent.db.connection import get_database_pool
    from aec_agent.db.repository import ElementRepository
    from aec_agent.db.models import Element, Project, CentroidInfo
    from uuid import uuid4
    import hashlib

    try:
        pool = await get_database_pool()
        if pool is None:
            logger.debug("Database not configured, skipping storage")
            return {"stored": 0, "skipped": True, "reason": "database_not_configured"}

        repo = ElementRepository(pool)

        # Create or get project for this PDF
        file_hash = hashlib.md5(str(pdf_path).encode()).hexdigest()
        existing_project = await repo.get_project_by_file_hash(file_hash)

        if existing_project:
            project_id = existing_project.id
            # Clear existing elements for re-extraction
            await repo.delete_project_elements(project_id)
        else:
            project = Project(
                id=uuid4(),
                name=pdf_path.stem,
                source="autocad",  # Gemini extraction targets AutoCAD
                file_path=str(pdf_path),
                file_hash=file_hash,
                metadata={
                    "extraction_method": "gemini_hybrid",
                    "scale_factor": calibration.scale_factor,
                    "units": calibration.units,
                },
            )
            project_id = await repo.create_project(project)

        # Convert and store entities
        elements = []
        for i, entity in enumerate(extraction.entities):
            # Build geometry WKT based on entity type
            geom_wkt = None
            centroid = None

            props = entity.properties or {}
            entity_type = entity.entity_type.value if hasattr(entity.entity_type, "value") else str(entity.entity_type)

            if entity_type == "LINE":
                start = props.get("start", (0, 0))
                end = props.get("end", (0, 0))
                geom_wkt = f"LINESTRING({start[0]} {start[1]}, {end[0]} {end[1]})"
                centroid = CentroidInfo(
                    x=(start[0] + end[0]) / 2,
                    y=(start[1] + end[1]) / 2,
                )
            elif entity_type == "CIRCLE":
                center = props.get("center", (0, 0))
                radius = props.get("radius", 1)
                # Store circle as a point with radius in properties
                geom_wkt = f"POINT({center[0]} {center[1]})"
                centroid = CentroidInfo(x=center[0], y=center[1])
            elif entity_type == "ARC":
                center = props.get("center", (0, 0))
                geom_wkt = f"POINT({center[0]} {center[1]})"
                centroid = CentroidInfo(x=center[0], y=center[1])
            elif entity_type == "TEXT" or entity_type == "MTEXT":
                position = props.get("position", (0, 0))
                geom_wkt = f"POINT({position[0]} {position[1]})"
                centroid = CentroidInfo(x=position[0], y=position[1])
            elif "position" in props:
                pos = props["position"]
                if isinstance(pos, (list, tuple)) and len(pos) >= 2:
                    geom_wkt = f"POINT({pos[0]} {pos[1]})"
                    centroid = CentroidInfo(x=pos[0], y=pos[1])

            # Build description for semantic search
            description_parts = [entity_type]
            if entity.layer:
                description_parts.append(f"on layer {entity.layer}")
            if props.get("text"):
                description_parts.append(f"text: {props['text'][:50]}")

            element = Element(
                id=uuid4(),
                project_id=project_id,
                source_id=f"gemini_{i}",
                source="autocad",
                entity_type=entity_type,
                layer=entity.layer,
                geom_wkt=geom_wkt,
                centroid=centroid,
                properties=props,
                description=" ".join(description_parts),
            )
            elements.append(element)

        # Batch insert
        if elements:
            count = await repo.upsert_elements_batch(elements)
            logger.info(
                "Stored extraction to database",
                project_id=str(project_id),
                entities_stored=count,
            )
            return {
                "stored": count,
                "project_id": str(project_id),
                "skipped": False,
            }

        return {"stored": 0, "skipped": False, "reason": "no_entities"}

    except Exception as e:
        logger.warning("Failed to store extraction to database", error=str(e))
        return {"stored": 0, "skipped": True, "reason": str(e)}


def _summarize_analysis(analysis: DrawingAnalysis) -> dict:
    """Summarize analysis for compact tool results."""
    elements = analysis.elements
    return {
        "drawing_type": analysis.drawing_type,
        "complexity": analysis.complexity,
        "scale": analysis.scale,
        "units": analysis.units,
        "total_elements": analysis.total_elements,
        "element_counts": {
            "lines": len(elements.lines),
            "arcs": len(elements.arcs),
            "circles": len(elements.circles),
            "text": len(elements.text),
            "symbols": len(elements.symbols),
            "dimensions": len(elements.dimensions),
        },
        "calibration_hints": len(analysis.calibration_hints),
        "extraction_strategy": analysis.extraction_strategy.primary_strategy if analysis.extraction_strategy else None,
    }


def _summarize_extraction(extraction: ExtractionResult) -> dict:
    """Summarize extraction for compact tool results."""
    by_type = {}
    for entity in extraction.entities:
        t = entity.entity_type.value
        by_type[t] = by_type.get(t, 0) + 1

    return {
        "total_entities": extraction.total_entities,
        "direct_count": extraction.direct_count,
        "guided_count": extraction.guided_count,
        "opencv_count": extraction.opencv_count,
        "by_type": by_type,
        "layers_used": list(set(e.layer for e in extraction.entities))[:10],
    }


def _summarize_creation(creation: AutoCADCreationResult) -> dict:
    """Summarize creation result for compact tool results."""
    stats_dict = creation.statistics.to_dict()
    return {
        "total_entities": creation.statistics.total_entities,
        "success_count": creation.statistics.success_count,
        "failure_count": creation.statistics.failure_count,
        "success_rate": f"{creation.success_rate:.0%}",
        "layers_created": len(creation.layer_results),
        "by_type": stats_dict.get("by_type", {}),
    }


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
    model: str = "gemini-2.0-flash",
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
        model: Gemini model to use ("gemini-2.0-flash" recommended - has free tier)
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
    model: str = "gemini-2.0-flash",
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
    model: str = "gemini-2.0-flash",
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
    model: str = "gemini-2.0-flash",
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
    model: str = "gemini-2.0-flash",
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
    model: str = "gemini-2.0-flash",
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
    model: str = "gemini-2.0-flash",
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
    model: str = "gemini-2.0-flash",
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
    model: str = "gemini-2.0-flash",
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


# ==============================================================================
# PHASE 5: AutoCAD Entity Creation (Draw in DWG)
# ==============================================================================


@mcp.tool()
async def gemini_create_entities(
    file_path: str,
    page: int = 1,
    dpi: int = 300,
    model: str = "gemini-2.0-flash",
    include_opencv: bool = True,
    create_layers: bool = True,
) -> dict[str, Any]:
    """
    Extract entities from a PDF and create them in AutoCAD.

    This combines Phases 1-5 of the Gemini-First pipeline:
    1. Phase 1: Render PDF to high-quality image
    2. Phase 2: Analyze with Gemini Vision
    3. Phase 3: Calibrate coordinates
    4. Phase 4: Extract entities
    5. Phase 5: Create entities in AutoCAD

    Args:
        file_path: Path to the PDF file
        page: Page number to process (1-indexed)
        dpi: Resolution for rendering (default 300)
        model: Gemini model for analysis
        include_opencv: Whether to use OpenCV for special regions
        create_layers: Whether to create layers that don't exist (default True)

    Returns:
        Success result with extraction and creation results.

    Example:
        >>> result = await gemini_create_entities("floor_plan.pdf")
        >>> if result["success"]:
        ...     data = result["data"]
        ...     print(f"Created: {data['creation']['statistics']['success_count']}")
        ...     print(f"Failed: {data['creation']['statistics']['failure_count']}")
    """
    logger.info(
        "gemini_create_entities_called",
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

        logger.info(
            "gemini_create_entities_rendered",
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
        )

        # Phase 4: Extract entities
        if include_opencv:
            extraction = await extract_all(
                analysis, calibration, render_result.image_path
            )
        else:
            extraction = await extract_direct_only(analysis, calibration)

        logger.info(
            "gemini_create_entities_extracted",
            total_entities=extraction.total_entities,
        )

        # Phase 5: Create entities in AutoCAD
        creation = await create_entities_in_autocad(
            extraction,
            create_layers=create_layers,
        )

        # Get drawing bounds for reference
        bounds = estimate_drawing_bounds(analysis, calibration)

        logger.info(
            "gemini_create_entities_success",
            drawing_type=analysis.drawing_type,
            extracted=extraction.total_entities,
            created=creation.success_count,
            failed=creation.failure_count,
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
                },
                "extraction": _summarize_extraction(extraction),
                "creation": _summarize_creation(creation),
                "drawing_bounds": {
                    "min": {"x": bounds[0][0], "y": bounds[0][1]},
                    "max": {"x": bounds[1][0], "y": bounds[1][1]},
                    "units": calibration.units,
                },
            },
            message=f"Created {creation.success_count} of {extraction.total_entities} entities "
                    f"({creation.success_rate:.0%} success rate)",
        )

    except FileNotFoundError as e:
        return error_result(ErrorCode.ELEMENT_NOT_FOUND, str(e))
    except ValueError as e:
        return error_result(ErrorCode.INVALID_PARAMS, str(e))
    except Exception as e:
        logger.exception("gemini_create_entities_failed", error=str(e))
        return error_result(ErrorCode.INTERNAL_ERROR, f"Failed to process PDF: {e}")


@mcp.tool()
async def gemini_vectorize_pdf(
    file_path: str,
    page: int = 1,
    dpi: int = 300,
    model: str = "gemini-2.0-flash",
    include_opencv: bool = True,
    create_layers: bool = True,
) -> dict[str, Any]:
    """
    Complete PDF-to-AutoCAD vectorization using Gemini-First pipeline.

    This is the main entry point for converting a PDF drawing to AutoCAD
    entities. It runs the complete Gemini-First pipeline (Phases 1-5).

    This tool:
    1. Renders the PDF at high quality (preserving grayscale)
    2. Uses Gemini Vision to understand the drawing
    3. Calibrates coordinates using scale/dimensions/sheet size
    4. Extracts entities using optimal strategy per element
    5. Creates all entities in the active AutoCAD drawing

    Args:
        file_path: Path to the PDF file
        page: Page number to vectorize (1-indexed)
        dpi: Resolution for rendering (72-1200, default 300)
        model: Gemini model for analysis (gemini-2.0-flash recommended - has free tier)
        include_opencv: Use OpenCV for special regions (default True)
        create_layers: Create layers that don't exist (default True)

    Returns:
        Success result with complete pipeline results.

    Example:
        >>> result = await gemini_vectorize_pdf("mechanical_plan.pdf")
        >>> if result["success"]:
        ...     print(f"Drawing type: {result['data']['summary']['drawing_type']}")
        ...     print(f"Entities created: {result['data']['summary']['created']}")
        ...     print(f"Success rate: {result['data']['summary']['success_rate']}")
    """
    logger.info(
        "gemini_vectorize_pdf_called",
        file_path=file_path,
        page=page,
        dpi=dpi,
        model=model,
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

    try:
        # ═══════════════════════════════════════════════════════════════════
        # PHASE 1: PDF Intake (High-quality rendering, no bitonal)
        # ═══════════════════════════════════════════════════════════════════
        logger.info("gemini_vectorize_phase1_start", phase="PDF Intake")

        render_result = await render_pdf_high_quality_async(
            pdf_path=pdf_path,
            page_number=page - 1,
            dpi=dpi,
            convert_grayscale=True,
        )

        # ═══════════════════════════════════════════════════════════════════
        # PHASE 2: Gemini Understanding (Analyze original image)
        # ═══════════════════════════════════════════════════════════════════
        logger.info("gemini_vectorize_phase2_start", phase="Gemini Understanding")

        analyzer = DrawingAnalyzer(model=model)
        analysis = await analyzer.analyze(render_result.image_path)

        # ═══════════════════════════════════════════════════════════════════
        # PHASE 3: Coordinate Calibration (Map pixels to DWG units)
        # ═══════════════════════════════════════════════════════════════════
        logger.info("gemini_vectorize_phase3_start", phase="Coordinate Calibration")

        calibration = calibrate_from_analysis(
            analysis=analysis,
            image_width=render_result.width_px,
            image_height=render_result.height_px,
            image_dpi=dpi,
        )

        # ═══════════════════════════════════════════════════════════════════
        # PHASE 4: Adaptive Extraction (Direct / Guided / Selective)
        # ═══════════════════════════════════════════════════════════════════
        logger.info("gemini_vectorize_phase4_start", phase="Adaptive Extraction")

        if include_opencv:
            extraction = await extract_all(
                analysis, calibration, render_result.image_path
            )
        else:
            extraction = await extract_direct_only(analysis, calibration)

        # ═══════════════════════════════════════════════════════════════════
        # PHASE 5: AutoCAD Entity Creation (Draw in DWG)
        # ═══════════════════════════════════════════════════════════════════
        logger.info("gemini_vectorize_phase5_start", phase="AutoCAD Entity Creation")

        creation = await create_entities_in_autocad(
            extraction,
            create_layers=create_layers,
        )

        # Get drawing bounds
        bounds = estimate_drawing_bounds(analysis, calibration)

        # ═══════════════════════════════════════════════════════════════════
        # BUILD RESULT
        # ═══════════════════════════════════════════════════════════════════
        logger.info(
            "gemini_vectorize_pdf_complete",
            drawing_type=analysis.drawing_type,
            total_elements=analysis.total_elements,
            entities_extracted=extraction.total_entities,
            entities_created=creation.success_count,
            success_rate=f"{creation.success_rate:.1%}",
        )

        # Return compact summary to reduce token usage
        return success_result(
            data={
                "summary": {
                    "drawing_type": analysis.drawing_type,
                    "complexity": analysis.complexity,
                    "scale": analysis.scale,
                    "units": calibration.units,
                    "calibration_method": calibration.method,
                    "calibration_confidence": f"{calibration.confidence:.0%}",
                    "extracted": extraction.total_entities,
                    "created": creation.success_count,
                    "failed": creation.failure_count,
                    "success_rate": f"{creation.success_rate:.0%}",
                },
                "phases": {
                    "phase1_render": {"image_path": str(render_result.image_path)},
                    "phase2_analysis": _summarize_analysis(analysis),
                    "phase3_calibration": {
                        "scale_factor": calibration.scale_factor,
                        "units": calibration.units,
                        "method": calibration.method,
                    },
                    "phase4_extraction": _summarize_extraction(extraction),
                    "phase5_creation": _summarize_creation(creation),
                },
            },
            message=f"Vectorized {analysis.drawing_type} drawing: "
                    f"{creation.success_count}/{extraction.total_entities} entities created "
                    f"({creation.success_rate:.0%})",
        )

    except FileNotFoundError as e:
        return error_result(ErrorCode.ELEMENT_NOT_FOUND, str(e))
    except ValueError as e:
        return error_result(ErrorCode.INVALID_PARAMS, str(e))
    except Exception as e:
        logger.exception("gemini_vectorize_pdf_failed", error=str(e))
        return error_result(ErrorCode.INTERNAL_ERROR, f"Failed to vectorize PDF: {e}")


@mcp.tool()
async def gemini_create_from_extraction(
    extraction_json: str,
    create_layers: bool = True,
) -> dict[str, Any]:
    """
    Create AutoCAD entities from a previously extracted JSON.

    Use this to create entities from a saved ExtractionResult JSON,
    allowing you to re-run Phase 5 without re-analyzing the drawing.

    Args:
        extraction_json: JSON string of ExtractionResult.to_dict()
        create_layers: Whether to create layers that don't exist

    Returns:
        Success result with creation results.

    Example:
        >>> # First, extract entities
        >>> extract_result = await gemini_extract_pdf_entities("plan.pdf")
        >>> extraction_json = json.dumps(extract_result["data"]["extraction"])
        >>>
        >>> # Later, create entities from saved extraction
        >>> result = await gemini_create_from_extraction(extraction_json)
    """
    import json

    logger.info("gemini_create_from_extraction_called")

    try:
        # Parse the extraction JSON
        extraction_data = json.loads(extraction_json)

        # Reconstruct EntityToCreate objects
        entities = []
        for entity_dict in extraction_data.get("entities", []):
            entities.append(EntityToCreate.from_dict(entity_dict))

        # Create a minimal ExtractionResult
        extraction = ExtractionResult(
            entities=entities,
            primary_strategy=extraction_data.get("metadata", {}).get("primary_strategy", "direct"),
            drawing_type=extraction_data.get("metadata", {}).get("drawing_type", ""),
        )

        # Create entities in AutoCAD
        creation = await create_entities_in_autocad(
            extraction,
            create_layers=create_layers,
        )

        logger.info(
            "gemini_create_from_extraction_success",
            created=creation.success_count,
            failed=creation.failure_count,
        )

        # Return summarized result to reduce token usage
        return success_result(
            data=_summarize_creation(creation),
            message=f"Created {creation.success_count} of {len(entities)} entities "
                    f"({creation.success_rate:.0%} success rate)",
        )

    except json.JSONDecodeError as e:
        return error_result(
            ErrorCode.INVALID_PARAMS,
            f"Invalid JSON: {e}",
        )
    except Exception as e:
        logger.exception("gemini_create_from_extraction_failed", error=str(e))
        return error_result(ErrorCode.INTERNAL_ERROR, f"Failed to create entities: {e}")


# ==============================================================================
# PHASE 6: Validation & Self-Correction
# ==============================================================================


@mcp.tool()
async def gemini_validate_extraction(
    image_path: str,
    creation_handles: list[str],
    created_count: int,
    failed_count: int,
    max_iterations: int = 3,
    apply_corrections: bool = True,
    model: str = "gemini-2.0-flash",
) -> dict[str, Any]:
    """
    Validate created AutoCAD entities against the original drawing.

    Uses Gemini Vision to compare the original drawing image with the
    created entities and identify any issues or missing elements.

    Args:
        image_path: Path to the original drawing image (PNG from Phase 1)
        creation_handles: List of created entity handles from Phase 5
        created_count: Number of successfully created entities
        failed_count: Number of failed entity creations
        max_iterations: Maximum validation/correction iterations (default 3)
        apply_corrections: Whether to auto-apply corrections (default True)
        model: Gemini model for validation

    Returns:
        Success result with validation status and any issues found.

    Example:
        >>> result = await gemini_validate_extraction(
        ...     image_path="temp/plan_page0.png",
        ...     creation_handles=["1A", "1B", "1C"],
        ...     created_count=100,
        ...     failed_count=5,
        ... )
        >>> if result["success"]:
        ...     print(f"Status: {result['data']['status']}")
        ...     print(f"Accuracy: {result['data']['accuracy_estimate']}%")
    """
    logger.info(
        "gemini_validate_extraction_called",
        image_path=image_path,
        created_count=created_count,
        max_iterations=max_iterations,
    )

    # Validate image path
    img_path = Path(image_path)
    if not img_path.exists():
        return error_result(
            ErrorCode.ELEMENT_NOT_FOUND,
            f"Image file not found: {image_path}",
        )

    try:
        # Build a minimal AutoCADCreationResult for validation
        from .autocad_creation import CreationStatistics

        creation_result = AutoCADCreationResult(
            success=True,
            statistics=CreationStatistics(
                total_entities=created_count + failed_count,
                success_count=created_count,
                failure_count=failed_count,
            ),
            created_handles=creation_handles,
        )

        # Run validation
        validation = await validate_extraction(
            original_image_path=img_path,
            creation_result=creation_result,
            max_iterations=max_iterations,
            apply_corrections_enabled=apply_corrections,
            model=model,
        )

        logger.info(
            "gemini_validate_extraction_success",
            status=validation.status.value,
            accuracy=validation.accuracy_estimate,
            issues=len(validation.issues),
        )

        return success_result(
            data=validation.to_dict(),
            message=f"Validation {validation.status.value}: "
                    f"{validation.accuracy_estimate:.0f}% accurate, "
                    f"{len(validation.issues)} issues found",
        )

    except Exception as e:
        logger.exception("gemini_validate_extraction_failed", error=str(e))
        return error_result(ErrorCode.INTERNAL_ERROR, f"Validation failed: {e}")


@mcp.tool()
async def gemini_complete_pipeline(
    file_path: str,
    page: int = 1,
    dpi: int = 300,
    model: str = "gemini-2.0-flash",
    include_opencv: bool = True,
    create_layers: bool = True,
    validate: bool = True,
    max_validation_iterations: int = 3,
    apply_corrections: bool = True,
) -> dict[str, Any]:
    """
    Run the complete Gemini-First PDF-to-AutoCAD pipeline (Phases 1-6).

    This is the full pipeline that:
    1. Renders PDF at high quality (Phase 1)
    2. Analyzes with Gemini Vision (Phase 2)
    3. Calibrates coordinates (Phase 3)
    4. Extracts entities (Phase 4)
    5. Creates entities in AutoCAD (Phase 5)
    6. Validates and self-corrects (Phase 6)

    Args:
        file_path: Path to the PDF file
        page: Page number to process (1-indexed)
        dpi: Resolution for rendering (72-1200, default 300)
        model: Gemini model for analysis and validation
        include_opencv: Use OpenCV for special regions (default True)
        create_layers: Create layers that don't exist (default True)
        validate: Whether to run validation phase (default True)
        max_validation_iterations: Max validation iterations (default 3)
        apply_corrections: Whether to auto-apply corrections (default True)

    Returns:
        Success result with complete pipeline results.

    Example:
        >>> result = await gemini_complete_pipeline("floor_plan.pdf")
        >>> if result["success"]:
        ...     data = result["data"]
        ...     print(f"Drawing type: {data['summary']['drawing_type']}")
        ...     print(f"Entities created: {data['summary']['created']}")
        ...     print(f"Validation: {data['validation']['status']}")
    """
    logger.info(
        "gemini_complete_pipeline_called",
        file_path=file_path,
        page=page,
        dpi=dpi,
        validate=validate,
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

    try:
        # ═══════════════════════════════════════════════════════════════════
        # PHASE 1: PDF Intake (High-quality rendering)
        # ═══════════════════════════════════════════════════════════════════
        logger.info("gemini_complete_phase1_start", phase="PDF Intake")

        render_result = await render_pdf_high_quality_async(
            pdf_path=pdf_path,
            page_number=page - 1,
            dpi=dpi,
            convert_grayscale=True,
        )

        # ═══════════════════════════════════════════════════════════════════
        # PHASE 2: Gemini Understanding
        # ═══════════════════════════════════════════════════════════════════
        logger.info("gemini_complete_phase2_start", phase="Gemini Understanding")

        analyzer = DrawingAnalyzer(model=model)
        analysis = await analyzer.analyze(render_result.image_path)

        # ═══════════════════════════════════════════════════════════════════
        # PHASE 3: Coordinate Calibration
        # ═══════════════════════════════════════════════════════════════════
        logger.info("gemini_complete_phase3_start", phase="Coordinate Calibration")

        calibration = calibrate_from_analysis(
            analysis=analysis,
            image_width=render_result.width_px,
            image_height=render_result.height_px,
            image_dpi=dpi,
        )

        # ═══════════════════════════════════════════════════════════════════
        # PHASE 4: Adaptive Extraction
        # ═══════════════════════════════════════════════════════════════════
        logger.info("gemini_complete_phase4_start", phase="Adaptive Extraction")

        if include_opencv:
            extraction = await extract_all(
                analysis, calibration, render_result.image_path
            )
        else:
            extraction = await extract_direct_only(analysis, calibration)

        # ═══════════════════════════════════════════════════════════════════
        # PHASE 5: AutoCAD Entity Creation
        # ═══════════════════════════════════════════════════════════════════
        logger.info("gemini_complete_phase5_start", phase="AutoCAD Entity Creation")

        creation = await create_entities_in_autocad(
            extraction,
            create_layers=create_layers,
        )

        # Get drawing bounds
        bounds = estimate_drawing_bounds(analysis, calibration)

        # ═══════════════════════════════════════════════════════════════════
        # PHASE 6: Validation & Self-Correction (Optional)
        # ═══════════════════════════════════════════════════════════════════
        validation_data = None

        if validate:
            logger.info("gemini_complete_phase6_start", phase="Validation")

            validation = await validate_extraction(
                original_image_path=render_result.image_path,
                creation_result=creation,
                extraction_result=extraction,
                analysis=analysis,
                calibration=calibration,
                max_iterations=max_validation_iterations,
                apply_corrections_enabled=apply_corrections,
                model=model,
            )

            validation_data = validation.to_dict()

            logger.info(
                "gemini_complete_phase6_done",
                status=validation.status.value,
                accuracy=validation.accuracy_estimate,
            )

        # ═══════════════════════════════════════════════════════════════════
        # BUILD RESULT
        # ═══════════════════════════════════════════════════════════════════
        logger.info(
            "gemini_complete_pipeline_done",
            drawing_type=analysis.drawing_type,
            entities_created=creation.success_count,
            validation_status=validation.status.value if validate else "skipped",
        )

        # Build compact result to reduce token usage
        result_data = {
            "summary": {
                "drawing_type": analysis.drawing_type,
                "complexity": analysis.complexity,
                "scale": analysis.scale,
                "units": calibration.units,
                "calibration_method": calibration.method,
                "calibration_confidence": f"{calibration.confidence:.0%}",
                "extracted": extraction.total_entities,
                "created": creation.success_count,
                "failed": creation.failure_count,
                "success_rate": f"{creation.success_rate:.0%}",
            },
            "phases": {
                "phase1_render": {"image_path": str(render_result.image_path)},
                "phase2_analysis": _summarize_analysis(analysis),
                "phase3_calibration": {
                    "scale_factor": calibration.scale_factor,
                    "units": calibration.units,
                    "method": calibration.method,
                },
                "phase4_extraction": _summarize_extraction(extraction),
                "phase5_creation": _summarize_creation(creation),
            },
        }

        if validation_data:
            result_data["phases"]["phase6_validation"] = validation_data
            result_data["summary"]["validation_status"] = validation.status.value
            result_data["summary"]["validation_accuracy"] = f"{validation.accuracy_estimate:.0f}%"
            result_data["summary"]["validation_issues"] = len(validation.issues)

        # Build message
        if validate:
            message = (
                f"Complete pipeline finished: {creation.success_count} entities created, "
                f"validation {validation.status.value} ({validation.accuracy_estimate:.0f}% accurate)"
            )
        else:
            message = (
                f"Pipeline finished (no validation): {creation.success_count} entities created "
                f"({creation.success_rate:.0%} success rate)"
            )

        return success_result(data=result_data, message=message)

    except FileNotFoundError as e:
        return error_result(ErrorCode.ELEMENT_NOT_FOUND, str(e))
    except ValueError as e:
        return error_result(ErrorCode.INVALID_PARAMS, str(e))
    except Exception as e:
        logger.exception("gemini_complete_pipeline_failed", error=str(e))
        return error_result(ErrorCode.INTERNAL_ERROR, f"Pipeline failed: {e}")


# =============================================================================
# HYBRID EXTRACTION TOOL (Gemini + OpenCV + YOLO Fusion)
# =============================================================================

def _summarize_hybrid_extraction(result: HybridExtractionResult) -> dict:
    """Summarize hybrid extraction result for compact tool output."""
    return {
        "total_entities": result.total_entities,
        "by_source": {
            "gemini": result.gemini_entities,
            "opencv": result.opencv_entities,
            "yolo": result.yolo_entities,
        },
        "fusion": {
            "duplicates_merged": result.duplicates_merged,
            "conflicts_resolved": result.conflicts_resolved,
        },
        "by_type": {
            etype: len(elist)
            for etype, elist in get_entities_by_type(result).items()
        },
        "required_layers": get_required_layers(result)[:10],
        "required_blocks": get_required_blocks(result)[:10],
    }


@mcp.tool()
async def gemini_hybrid_extract(
    pdf_path: str,
    page: int = 1,
    dpi: int = 300,
    use_opencv_lines: bool = True,
    use_opencv_circles: bool = True,
    use_yolo_symbols: bool = True,
    opencv_line_min_length: int = 30,
    yolo_confidence: float = 0.5,
    enable_refinement: bool = True,
    refine_snap_to_grid: bool = True,
    refine_connect_endpoints: bool = True,
    refine_align_parallel: bool = True,
    use_ocr_text_anchoring: bool = True,
    ocr_min_confidence: float = 60.0,
    create_in_autocad: bool = True,
) -> dict:
    """
    HYBRID EXTRACTION: Gemini + OpenCV + YOLO + OCR fusion for optimal PDF to vector.

    This tool combines the strengths of multiple extraction methods:
    - **Gemini**: Semantic understanding (what & where), text/OCR, layer assignment
    - **OpenCV**: Pixel-perfect geometry (lines, circles, contours)
    - **YOLO**: Trained symbol detection (MEP devices, equipment)
    - **OCR (Tesseract)**: Pixel-accurate text positions anchoring
    - **Refinement**: Gemini reviews and adjusts coordinates for accuracy

    The hybrid approach produces superior results compared to any single method:
    - Gemini understands context but has coordinate drift (~10-100px)
    - OpenCV is geometrically precise but has no semantic understanding
    - YOLO detects symbols accurately but needs context for attributes
    - OCR anchors Gemini's text content to pixel-accurate positions
    - Refinement pass connects endpoints, aligns lines, snaps to grid

    Args:
        pdf_path: Path to the PDF file to process
        page: Page number to extract (1-indexed, default: 1)
        dpi: Resolution for rendering (default: 300)
        use_opencv_lines: Use OpenCV for line extraction (default: True)
        use_opencv_circles: Use OpenCV for circle extraction (default: True)
        use_yolo_symbols: Use YOLO for symbol detection (default: True)
        opencv_line_min_length: Minimum line length in pixels (default: 30)
        yolo_confidence: YOLO confidence threshold 0-1 (default: 0.5)
        enable_refinement: Enable Gemini refinement pass (default: True)
        refine_snap_to_grid: Snap coordinates to grid (default: True)
        refine_connect_endpoints: Connect nearby endpoints (default: True)
        refine_align_parallel: Align nearly-parallel lines (default: True)
        use_ocr_text_anchoring: Anchor text to OCR-detected positions (default: True)
        ocr_min_confidence: Minimum OCR confidence 0-100 (default: 60.0)
        create_in_autocad: If True, create entities in AutoCAD (default: True)

    Returns:
        Extraction result with entities from all sources, refined and deduplicated.
        Text positions are OCR-anchored for alignment with vectors.

    Example:
        >>> result = await gemini_hybrid_extract(
        ...     pdf_path="drawing.pdf",
        ...     enable_refinement=True,
        ...     use_ocr_text_anchoring=True,
        ... )
        >>> print(f"Refined: {result['summary']['refinement_adjustments']} adjustments")
        >>> print(f"OCR anchored: {result['summary']['ocr_text_anchored']} texts")
    """
    try:
        pdf_file = Path(pdf_path)
        if not pdf_file.exists():
            return error_result(ErrorCode.ELEMENT_NOT_FOUND, f"PDF not found: {pdf_path}")

        logger.info(
            "gemini_hybrid_extract_starting",
            pdf_path=pdf_path,
            page=page,
            use_opencv_lines=use_opencv_lines,
            use_opencv_circles=use_opencv_circles,
            use_yolo_symbols=use_yolo_symbols,
        )

        # Phase 1: Render PDF (convert 1-indexed to 0-indexed)
        page_0_indexed = page - 1
        if page_0_indexed < 0:
            return error_result(ErrorCode.INVALID_PARAMS, "Page number must be >= 1")

        render_result = await render_pdf_high_quality_async(
            pdf_path=pdf_file,
            page_number=page_0_indexed,
            dpi=dpi,
        )
        # render_pdf_high_quality_async raises exceptions on failure, no need to check success

        # Phase 2: Gemini Understanding
        analysis = await analyze_drawing(render_result.image_path)

        # Phase 3: Calibration
        calibration = calibrate_from_analysis(
            analysis=analysis,
            image_width=render_result.width_px,
            image_height=render_result.height_px,
            image_dpi=dpi,
        )

        # Phase 4: HYBRID Extraction with Refinement + OCR Text Anchoring
        config = HybridExtractionConfig(
            use_opencv_for_lines=use_opencv_lines,
            use_opencv_for_circles=use_opencv_circles,
            use_yolo_for_symbols=use_yolo_symbols,
            use_gemini_for_text=True,
            opencv_line_min_length=opencv_line_min_length,
            yolo_confidence_threshold=yolo_confidence,
            prefer_opencv_geometry=True,
            # OCR Text Anchoring settings
            use_ocr_for_text_positions=use_ocr_text_anchoring,
            ocr_min_confidence=ocr_min_confidence,
            # Refinement settings
            enable_refinement=enable_refinement,
            refine_snap_to_grid=refine_snap_to_grid,
            refine_connect_endpoints=refine_connect_endpoints,
            refine_align_parallel=refine_align_parallel,
        )

        extraction = await hybrid_extract_all(
            analysis=analysis,
            calibration=calibration,
            image_path=render_result.image_path,
            config=config,
        )

        # Store extraction to PostgreSQL for semantic search
        storage_result = await _store_extraction_to_database(
            extraction=extraction,
            pdf_path=pdf_file,
            calibration=calibration,
        )

        result_data = {
            "success": True,
            "summary": {
                "drawing_type": analysis.drawing_type,
                "total_entities": extraction.total_entities,
                "gemini_entities": extraction.gemini_entities,
                "opencv_entities": extraction.opencv_entities,
                "yolo_entities": extraction.yolo_entities,
                "duplicates_merged": extraction.duplicates_merged,
                "refinement_applied": extraction.refinement_applied,
                "refinement_adjustments": extraction.refinement_adjustments,
                "endpoints_connected": extraction.endpoints_connected,
                "lines_snapped": extraction.lines_snapped,
                "stored_to_db": storage_result.get("stored", 0),
            },
            "extraction": _summarize_hybrid_extraction(extraction),
            "calibration": {
                "method": calibration.method,
                "scale_factor": calibration.scale_factor,
                "units": calibration.units,
                "confidence": f"{calibration.confidence:.0%}",
            },
            "storage": storage_result,
            "image_path": str(render_result.image_path),
        }

        # Optional: Create in AutoCAD
        if create_in_autocad and extraction.entities:
            creation = await create_entities_in_autocad(extraction)
            result_data["creation"] = _summarize_creation(creation)
            result_data["summary"]["created"] = creation.success_count
            result_data["summary"]["failed"] = creation.failure_count

        # Build message
        refinement_msg = ""
        if extraction.refinement_applied:
            refinement_msg = f", refined {extraction.refinement_adjustments} coords"

        message = (
            f"Hybrid extraction complete: {extraction.total_entities} entities "
            f"(Gemini: {extraction.gemini_entities}, OpenCV: {extraction.opencv_entities}, "
            f"YOLO: {extraction.yolo_entities}{refinement_msg})"
        )

        return success_result(data=result_data, message=message)

    except FileNotFoundError as e:
        return error_result(ErrorCode.ELEMENT_NOT_FOUND, str(e))
    except ValueError as e:
        return error_result(ErrorCode.INVALID_PARAMS, str(e))
    except Exception as e:
        logger.exception("gemini_hybrid_extract_failed", error=str(e))
        return error_result(ErrorCode.INTERNAL_ERROR, f"Hybrid extraction failed: {e}")


# =============================================================================
# Phase C: Advanced Vectorization Tools
# =============================================================================


@mcp.tool()
async def phase_c_detect_junctions(
    image_path: str,
    confidence_threshold: float = 0.5,
    model_name: str = "hawpv3",
    snap_distance: float = 5.0,
) -> dict:
    """
    Detect T, L, X, Y junctions, corners, and endpoints in floor plan images.

    Uses HAWP (Holistically-Attracted Wireframe Parsing) neural network to
    detect structural junctions that can be used to improve line endpoint
    snapping and connection accuracy.

    Phase C.1 of Advanced Vectorization.

    Args:
        image_path: Path to floor plan or technical drawing image
        confidence_threshold: Minimum confidence for junction detection (0-1)
        model_name: Model to use (default: hawpv3)
        snap_distance: Distance threshold for endpoint snapping (pixels)

    Returns:
        dict with detected junctions, wireframe lines, and snapping suggestions
    """
    from .neural_junction_detection import (
        is_junction_detection_available,
        JunctionDetector,
        JunctionDetectionConfig,
    )

    logger.info(
        "phase_c_detect_junctions",
        image_path=image_path,
        confidence=confidence_threshold,
        model=model_name,
    )

    try:
        # Check availability
        if not is_junction_detection_available():
            return error_result(
                ErrorCode.MISSING_DEPENDENCY,
                "Junction detection requires PyTorch. Install with: pip install torch torchvision"
            )

        image_file = Path(image_path)
        if not image_file.exists():
            return error_result(
                ErrorCode.ELEMENT_NOT_FOUND,
                f"Image not found: {image_path}"
            )

        # Create config
        config = JunctionDetectionConfig(
            model_name=model_name,
            confidence_threshold=confidence_threshold,
        )

        # Run detection
        detector = JunctionDetector.get_instance(config)
        result = detector.detect(str(image_file), config)

        # Summarize junctions by type
        junction_counts = {}
        for junction in result.junctions:
            jtype = junction.junction_type.value
            junction_counts[jtype] = junction_counts.get(jtype, 0) + 1

        result_data = {
            "success": True,
            "total_junctions": len(result.junctions),
            "total_lines": len(result.lines),
            "junction_counts": junction_counts,
            "image_size": result.image_size,
            "processing_time_ms": result.processing_time_ms,
            "junctions": [
                {
                    "position": j.position,
                    "type": j.junction_type.value,
                    "confidence": f"{j.confidence:.2%}",
                    "connected_lines": j.connected_line_indices,
                }
                for j in result.junctions[:20]  # Limit output
            ],
            "lines": [
                {
                    "start": line.start,
                    "end": line.end,
                    "confidence": f"{line.confidence:.2%}",
                }
                for line in result.lines[:20]  # Limit output
            ],
        }

        message = (
            f"Detected {len(result.junctions)} junctions and {len(result.lines)} lines "
            f"in {result.processing_time_ms:.0f}ms"
        )
        return success_result(data=result_data, message=message)

    except Exception as e:
        logger.exception("phase_c_detect_junctions_failed", error=str(e))
        return error_result(ErrorCode.INTERNAL_ERROR, f"Junction detection failed: {e}")


@mcp.tool()
async def phase_c_bezier_splatting(
    image_path: str,
    num_curves: int = 64,
    iterations: int = 500,
    output_svg: str = "",
    curve_type: str = "cubic",
    learning_rate: float = 0.01,
) -> dict:
    """
    Vectorize an image using Bezier Splatting (150x faster curve fitting).

    Uses differentiable 2D Gaussian splatting to optimize Bezier curve control
    points, achieving fast convergence for stroke-based vectorization.

    Phase C.2 of Advanced Vectorization.

    Reference: arxiv 2503.16424 "Bezier Splatting"

    Args:
        image_path: Path to input image (grayscale line drawing works best)
        num_curves: Number of Bezier curves to fit (default: 64)
        iterations: Number of optimization iterations (default: 500)
        output_svg: Optional path to save SVG output (empty = don't save)
        curve_type: Type of curves: "linear", "quadratic", or "cubic"
        learning_rate: Optimizer learning rate (default: 0.01)

    Returns:
        dict with optimized curves, loss history, and optional SVG path
    """
    from .bezier_splatting import (
        is_bezier_splatting_available,
        bezier_splat,
        BezierSplattingConfig,
        CurveType,
    )

    logger.info(
        "phase_c_bezier_splatting",
        image_path=image_path,
        num_curves=num_curves,
        iterations=iterations,
    )

    try:
        # Check availability
        if not is_bezier_splatting_available():
            return error_result(
                ErrorCode.MISSING_DEPENDENCY,
                "Bezier Splatting requires PyTorch. Install with: pip install torch torchvision"
            )

        image_file = Path(image_path)
        if not image_file.exists():
            return error_result(
                ErrorCode.ELEMENT_NOT_FOUND,
                f"Image not found: {image_path}"
            )

        # Parse curve type
        try:
            curve_type_enum = CurveType(curve_type.lower())
        except ValueError:
            return error_result(
                ErrorCode.INVALID_PARAMS,
                f"Invalid curve_type: {curve_type}. Use 'linear', 'quadratic', or 'cubic'"
            )

        # Create config
        config = BezierSplattingConfig(
            num_curves=num_curves,
            curve_type=curve_type_enum,
            iterations=iterations,
            learning_rate=learning_rate,
        )

        # Run optimization
        output_path = Path(output_svg) if output_svg else None
        result = bezier_splat(image_file, config, output_path)

        result_data = {
            "success": True,
            "num_curves": len(result.curves),
            "final_loss": result.final_loss,
            "image_size": result.image_size,
            "processing_time_ms": result.processing_time_ms,
            "device": result.device,
            "curve_type": curve_type,
            "curves_preview": [
                {
                    "control_points": c.control_points,
                    "stroke_width": c.stroke_width,
                    "opacity": c.opacity,
                }
                for c in result.curves[:10]  # Limit output
            ],
        }

        if output_path:
            result_data["svg_path"] = str(output_path)

        message = (
            f"Bezier Splatting complete: {len(result.curves)} curves, "
            f"loss={result.final_loss:.6f}, time={result.processing_time_ms:.0f}ms"
        )
        return success_result(data=result_data, message=message)

    except Exception as e:
        logger.exception("phase_c_bezier_splatting_failed", error=str(e))
        return error_result(ErrorCode.INTERNAL_ERROR, f"Bezier Splatting failed: {e}")


@mcp.tool()
async def phase_c_live_vectorize(
    image_path: str,
    num_layers: int = 5,
    paths_per_layer: int = 1,
    output_svg: str = "",
    iterations_per_layer: int = 500,
) -> dict:
    """
    Vectorize an image using LIVE (Layer-wise Image Vectorization).

    Progressive coarse-to-fine vectorization that builds up the image
    layer by layer using closed Bezier paths. Each layer captures
    remaining detail from previous layers.

    Phase C.3 of Advanced Vectorization.

    Reference: "Towards Layer-wise Image Vectorization" (CVPR 2022)

    Args:
        image_path: Path to input image (color images work well)
        num_layers: Number of vector layers (default: 5)
        paths_per_layer: Closed paths per layer (default: 1)
        output_svg: Optional path to save SVG output (empty = don't save)
        iterations_per_layer: Optimization iterations per layer (default: 500)

    Returns:
        dict with vector layers, loss history, and optional SVG path
    """
    from .live_vectorization import (
        is_live_available,
        live_vectorize,
        LIVEConfig,
    )

    logger.info(
        "phase_c_live_vectorize",
        image_path=image_path,
        num_layers=num_layers,
        paths_per_layer=paths_per_layer,
    )

    try:
        # Check availability
        if not is_live_available():
            return error_result(
                ErrorCode.MISSING_DEPENDENCY,
                "LIVE vectorization requires PyTorch. Install with: pip install torch torchvision"
            )

        image_file = Path(image_path)
        if not image_file.exists():
            return error_result(
                ErrorCode.ELEMENT_NOT_FOUND,
                f"Image not found: {image_path}"
            )

        # Create config
        config = LIVEConfig(
            num_layers=num_layers,
            paths_per_layer=paths_per_layer,
            iterations_per_layer=iterations_per_layer,
        )

        # Run vectorization
        output_path = Path(output_svg) if output_svg else None
        result = live_vectorize(image_file, config, output_path)

        total_paths = sum(len(layer.paths) for layer in result.layers)

        result_data = {
            "success": True,
            "num_layers": len(result.layers),
            "total_paths": total_paths,
            "final_loss": result.final_loss,
            "image_size": result.image_size,
            "processing_time_ms": result.processing_time_ms,
            "device": result.device,
            "layers_preview": [
                {
                    "layer_index": layer.layer_index,
                    "num_paths": len(layer.paths),
                    "fill_color": layer.fill_color,
                }
                for layer in result.layers
            ],
        }

        if output_path:
            result_data["svg_path"] = str(output_path)

        message = (
            f"LIVE complete: {len(result.layers)} layers, {total_paths} paths, "
            f"loss={result.final_loss:.6f}, time={result.processing_time_ms:.0f}ms"
        )
        return success_result(data=result_data, message=message)

    except Exception as e:
        logger.exception("phase_c_live_vectorize_failed", error=str(e))
        return error_result(ErrorCode.INTERNAL_ERROR, f"LIVE vectorization failed: {e}")


@mcp.tool()
async def phase_c_visualize_pipeline(
    image_path: str,
    output_dir: str = "",
    run_junction_detection: bool = True,
    run_bezier_splatting: bool = True,
    run_live_vectorization: bool = True,
    bezier_num_curves: int = 64,
    bezier_iterations: int = 300,
    live_num_layers: int = 4,
    live_iterations: int = 300,
) -> dict:
    """
    Visualize the Phase C vectorization pipeline on an image.

    Creates a side-by-side comparison showing:
    1. Original input image
    2. Junction detection overlay (junctions + wireframe lines)
    3. Bezier Splatting result (SVG vectorization)
    4. LIVE layered vectorization result

    Use this tool to assess what each technique produces and identify gaps.

    Args:
        image_path: Path to input image (PNG, JPG, PDF page render)
        output_dir: Output directory (default: creates phase_c_output next to image)
        run_junction_detection: Enable C.1 Junction Detection
        run_bezier_splatting: Enable C.2 Bezier Splatting
        run_live_vectorization: Enable C.3 LIVE Vectorization
        bezier_num_curves: Number of Bezier curves to fit (more = finer detail)
        bezier_iterations: Optimization iterations for Bezier (more = better fit)
        live_num_layers: Number of LIVE layers (more = finer detail)
        live_iterations: Iterations per LIVE layer

    Returns:
        dict with paths to all output files and statistics
    """
    from .pipeline_visualizer import visualize_pipeline

    logger.info(
        "phase_c_visualize_pipeline",
        image_path=image_path,
        output_dir=output_dir,
    )

    try:
        image_file = Path(image_path)
        if not image_file.exists():
            return error_result(
                ErrorCode.ELEMENT_NOT_FOUND,
                f"Image not found: {image_path}"
            )

        out_dir = Path(output_dir) if output_dir else None

        result = await visualize_pipeline(
            image_path=image_file,
            output_dir=out_dir,
            run_junction_detection=run_junction_detection,
            run_bezier_splatting=run_bezier_splatting,
            run_live_vectorization=run_live_vectorization,
            bezier_num_curves=bezier_num_curves,
            bezier_iterations=bezier_iterations,
            live_num_layers=live_num_layers,
            live_iterations=live_iterations,
        )

        result_data = result.to_dict()
        result_data["success"] = True

        # Build summary message
        parts = []
        if result.junction_count > 0:
            parts.append(f"{result.junction_count} junctions")
        if result.bezier_curve_count > 0:
            parts.append(f"{result.bezier_curve_count} Bezier curves")
        if result.live_path_count > 0:
            parts.append(f"{result.live_layer_count} LIVE layers ({result.live_path_count} paths)")

        message = f"Pipeline visualization complete: {', '.join(parts) if parts else 'no results'}"

        if result.comparison_path:
            message += f". Comparison saved to: {result.comparison_path}"

        if result.errors:
            message += f". Warnings: {len(result.errors)} errors occurred."

        return success_result(data=result_data, message=message)

    except Exception as e:
        logger.exception("phase_c_visualize_pipeline_failed", error=str(e))
        return error_result(ErrorCode.INTERNAL_ERROR, f"Pipeline visualization failed: {e}")


# =============================================================================
# Phase D: RAG Symbol Recognition Tools
# =============================================================================

@mcp.tool()
async def symbol_recognize(
    image_path: str,
    bbox: Optional[str] = None,
    domain: Optional[str] = None,
    category: Optional[str] = None,
    top_k: int = 5,
    min_confidence: float = 0.5,
) -> dict[str, Any]:
    """
    Recognize a CAD symbol from an image using CLIP embeddings and RAG.

    This tool uses visual similarity search to identify symbols from image patches,
    returning matching block names and confidence scores.

    Args:
        image_path: Path to image containing the symbol
        bbox: Optional bounding box "x1,y1,x2,y2" (pixels). If not provided, uses full image.
        domain: Filter by domain: electrical, mechanical, plumbing, fire, architectural
        category: Filter by category: outlet, switch, diffuser, detector, etc.
        top_k: Number of top matches to return (default 5)
        min_confidence: Minimum confidence threshold 0-1 (default 0.5)

    Returns:
        Success result with list of matching symbols, or error result.

    Example:
        >>> result = await symbol_recognize("symbol.png", domain="electrical")
        >>> if result["success"]:
        ...     for match in result["data"]["matches"]:
        ...         print(f"{match['block_name']}: {match['confidence']:.1%}")
    """
    from .symbol_rag import (
        SymbolRAG,
        SymbolDomain,
        is_symbol_rag_available,
        extract_symbol_region,
    )
    from PIL import Image as PILImage
    from aec_agent.db.connection import get_database_pool

    logger.info(
        "symbol_recognize_called",
        image_path=image_path,
        bbox=bbox,
        domain=domain,
    )

    # Check dependencies
    if not is_symbol_rag_available():
        return error_result(
            ErrorCode.INTERNAL_ERROR,
            "Symbol RAG not available. Install: pip install torch transformers"
        )

    # Validate image path
    img_path = Path(image_path)
    if not img_path.exists():
        return error_result(
            ErrorCode.ELEMENT_NOT_FOUND,
            f"Image not found: {image_path}"
        )

    try:
        # Load image
        image = PILImage.open(img_path)

        # Extract region if bbox provided
        if bbox:
            try:
                coords = [int(x.strip()) for x in bbox.split(",")]
                if len(coords) != 4:
                    return error_result(
                        ErrorCode.INVALID_PARAMS,
                        "bbox must be 4 comma-separated integers: x1,y1,x2,y2"
                    )
                image = extract_symbol_region(image, tuple(coords))
            except ValueError:
                return error_result(
                    ErrorCode.INVALID_PARAMS,
                    "bbox coordinates must be integers"
                )

        # Validate domain
        domain_enum = None
        if domain:
            try:
                domain_enum = SymbolDomain(domain.lower())
            except ValueError:
                valid_domains = [d.value for d in SymbolDomain]
                return error_result(
                    ErrorCode.INVALID_PARAMS,
                    f"Invalid domain '{domain}'. Valid: {valid_domains}"
                )

        # Get database pool
        pool = await get_database_pool()
        if pool is None:
            return error_result(
                ErrorCode.INTERNAL_ERROR,
                "Database not configured. Set DATABASE_URL environment variable."
            )

        # Initialize RAG
        rag = SymbolRAG(pool)
        await rag.initialize()

        # Recognize symbol
        matches = await rag.recognize_symbol(
            image=image,
            domain=domain_enum,
            category=category,
            top_k=top_k,
            min_confidence=min_confidence,
        )

        # Format results
        match_data = []
        for m in matches:
            match_data.append({
                "symbol_id": str(m.symbol_id),
                "block_name": m.block_name,
                "display_name": m.display_name,
                "domain": m.domain.value,
                "category": m.category,
                "subcategory": m.subcategory,
                "layer": m.layer,
                "confidence": round(m.confidence, 4),
                "default_scale": m.default_scale,
                "default_rotation": m.default_rotation,
                "attributes": m.attributes,
            })

        if matches:
            best = matches[0]
            message = f"Found {len(matches)} matches. Best: {best.block_name} ({best.confidence:.1%})"
        else:
            message = "No matching symbols found above confidence threshold"

        return success_result(
            data={
                "matches": match_data,
                "match_count": len(matches),
                "domain_filter": domain,
                "category_filter": category,
            },
            message=message,
        )

    except Exception as e:
        logger.exception("symbol_recognize_failed", error=str(e))
        return error_result(ErrorCode.INTERNAL_ERROR, f"Symbol recognition failed: {e}")


@mcp.tool()
async def symbol_search(
    query: str,
    domain: Optional[str] = None,
    top_k: int = 5,
) -> dict[str, Any]:
    """
    Search for CAD symbols by text description using CLIP text embeddings.

    This tool searches the symbol library using natural language queries,
    returning matching symbols based on semantic similarity.

    Args:
        query: Text description (e.g., "smoke detector", "duplex outlet")
        domain: Filter by domain: electrical, mechanical, plumbing, fire, architectural
        top_k: Number of results to return (default 5)

    Returns:
        Success result with list of matching symbols, or error result.

    Example:
        >>> result = await symbol_search("fire alarm pull station")
        >>> if result["success"]:
        ...     for match in result["data"]["matches"]:
        ...         print(f"{match['block_name']}: {match['display_name']}")
    """
    from .symbol_rag import (
        SymbolRAG,
        SymbolDomain,
        is_symbol_rag_available,
    )
    from aec_agent.db.connection import get_database_pool

    logger.info("symbol_search_called", query=query, domain=domain)

    # Check dependencies
    if not is_symbol_rag_available():
        return error_result(
            ErrorCode.INTERNAL_ERROR,
            "Symbol RAG not available. Install: pip install torch transformers"
        )

    # Validate domain
    domain_enum = None
    if domain:
        try:
            domain_enum = SymbolDomain(domain.lower())
        except ValueError:
            valid_domains = [d.value for d in SymbolDomain]
            return error_result(
                ErrorCode.INVALID_PARAMS,
                f"Invalid domain '{domain}'. Valid: {valid_domains}"
            )

    try:
        # Get database pool
        pool = await get_database_pool()
        if pool is None:
            return error_result(
                ErrorCode.INTERNAL_ERROR,
                "Database not configured. Set DATABASE_URL environment variable."
            )

        # Initialize RAG
        rag = SymbolRAG(pool)
        await rag.initialize()

        # Search by text
        matches = await rag.search_by_description(
            description=query,
            domain=domain_enum,
            top_k=top_k,
        )

        # Format results
        match_data = []
        for m in matches:
            match_data.append({
                "symbol_id": str(m.symbol_id),
                "block_name": m.block_name,
                "display_name": m.display_name,
                "domain": m.domain.value,
                "category": m.category,
                "subcategory": m.subcategory,
                "layer": m.layer,
                "confidence": round(m.confidence, 4),
            })

        if matches:
            message = f"Found {len(matches)} symbols matching '{query}'"
        else:
            message = f"No symbols found matching '{query}'"

        return success_result(
            data={
                "query": query,
                "matches": match_data,
                "match_count": len(matches),
                "domain_filter": domain,
            },
            message=message,
        )

    except Exception as e:
        logger.exception("symbol_search_failed", error=str(e))
        return error_result(ErrorCode.INTERNAL_ERROR, f"Symbol search failed: {e}")


@mcp.tool()
async def symbol_library_stats() -> dict[str, Any]:
    """
    Get statistics about the symbol library.

    Returns counts of symbols by domain and embedding status.

    Returns:
        Success result with library statistics, or error result.

    Example:
        >>> result = await symbol_library_stats()
        >>> if result["success"]:
        ...     print(f"Total symbols: {result['data']['total']}")
        ...     for domain, data in result['data']['by_domain'].items():
        ...         print(f"  {domain}: {data['count']}")
    """
    from .symbol_library_seed import get_symbol_stats
    from aec_agent.db.connection import get_database_pool

    logger.info("symbol_library_stats_called")

    try:
        pool = await get_database_pool()
        if pool is None:
            return error_result(
                ErrorCode.INTERNAL_ERROR,
                "Database not configured. Set DATABASE_URL environment variable."
            )

        stats = await get_symbol_stats(pool)

        return success_result(
            data=stats,
            message=f"Symbol library: {stats['total']} symbols ({stats['with_embeddings']} with embeddings)",
        )

    except Exception as e:
        logger.exception("symbol_library_stats_failed", error=str(e))
        return error_result(ErrorCode.INTERNAL_ERROR, f"Failed to get stats: {e}")


@mcp.tool()
async def symbol_library_seed(
    generate_embeddings: bool = True,
) -> dict[str, Any]:
    """
    Seed the symbol library with standard CAD symbols.

    Populates the database with ~80 common electrical, mechanical, plumbing,
    fire alarm, and architectural symbols with NCS-compliant layer mapping.

    Args:
        generate_embeddings: Generate CLIP text embeddings for search (default True)

    Returns:
        Success result with seeding statistics, or error result.

    Example:
        >>> result = await symbol_library_seed()
        >>> if result["success"]:
        ...     print(f"Seeded {result['data']['symbols_added']} symbols")
    """
    from .symbol_library_seed import seed_symbol_library, get_symbol_stats
    from aec_agent.db.connection import get_database_pool

    logger.info("symbol_library_seed_called", generate_embeddings=generate_embeddings)

    try:
        pool = await get_database_pool()
        if pool is None:
            return error_result(
                ErrorCode.INTERNAL_ERROR,
                "Database not configured. Set DATABASE_URL environment variable."
            )

        # Seed symbols
        count = await seed_symbol_library(pool, generate_embeddings=generate_embeddings)

        # Get updated stats
        stats = await get_symbol_stats(pool)

        return success_result(
            data={
                "symbols_added": count,
                "total_in_library": stats["total"],
                "with_embeddings": stats["with_embeddings"],
                "by_domain": stats["by_domain"],
            },
            message=f"Seeded {count} symbols. Library now has {stats['total']} symbols.",
        )

    except Exception as e:
        logger.exception("symbol_library_seed_failed", error=str(e))
        return error_result(ErrorCode.INTERNAL_ERROR, f"Seeding failed: {e}")


# =============================================================================
# Best Practices Pipeline Tool
# =============================================================================


@mcp.tool()
async def run_best_practices_vectorization(
    pdf_path: str,
    page: int = 1,
    output_dir: str = "",
    dpi: int = 300,
    use_super_resolution: bool = True,
    use_symbol_rag: bool = True,
    straighten_tolerance_deg: float = 5.0,
    connect_endpoints: bool = True,
    gemini_visual_qa: bool = True,
) -> dict[str, Any]:
    """
    Run the Best Practices PDF to AutoCAD vectorization pipeline.

    This is the recommended 7-stage pipeline that combines the optimal algorithms
    for each step as documented in docs/BEST_ALGORITHMS_PIPELINE.md:

    1. PDF Rendering (PyMuPDF @ 300-600 DPI, Real-ESRGAN super-resolution)
    2. Preprocessing (NLM denoise + Hough deskew + 7-method ensemble binarization)
    3. Gemini Analysis (scale, type, MText, layers, element detection)
    4. Vector Extraction (LSD lines + Hough circles + HAWP junctions)
    5. Symbol Recognition (CLIP embeddings + pgvector RAG lookup)
    6. Validation (5° line straightening + endpoint connection + Gemini QA)
    7. Output (scaled entities ready for AutoCAD)

    Args:
        pdf_path: Path to the PDF file to vectorize
        page: Page number to process (1-indexed, default 1)
        output_dir: Output directory for intermediate files (empty string = pdf_dir/vectorized)
        dpi: Base DPI for rendering (default 300, auto-adjusts for small pages)
        use_super_resolution: Apply Real-ESRGAN 4x upscaling if DPI < 200 (default True)
        use_symbol_rag: Use CLIP + pgvector RAG for symbol recognition (default True)
        straighten_tolerance_deg: Snap lines to H/V/45° if within this tolerance (default 5.0)
        connect_endpoints: Connect nearby line endpoints (default True)
        gemini_visual_qa: Run Gemini visual QA validation pass (default True)

    Returns:
        Success result with pipeline results including:
        - entity_count: Total entities extracted
        - symbol_count: Symbols recognized via RAG
        - line_count: Lines extracted
        - text_count: Text elements extracted
        - stages: Results from each pipeline stage
        - entities: List of EntityToCreate objects (summarized)

    Example:
        >>> result = await run_best_practices_vectorization(
        ...     pdf_path="/path/to/drawing.pdf",
        ...     page=1,
        ...     use_symbol_rag=True,
        ... )
        >>> if result["success"]:
        ...     print(f"Extracted {result['data']['entity_count']} entities")
        ...     print(f"Recognized {result['data']['symbol_count']} symbols via RAG")
    """
    from .best_practices_pipeline import (
        BestPracticesPipeline,
        BestPracticesConfig,
        SymbolRecognitionMethod,
    )

    logger.info(
        "run_best_practices_vectorization_called",
        pdf_path=pdf_path,
        page=page,
        dpi=dpi,
        use_symbol_rag=use_symbol_rag,
    )

    try:
        # Validate PDF exists
        pdf_file = Path(pdf_path)
        if not pdf_file.exists():
            return error_result(
                ErrorCode.FILE_NOT_FOUND,
                f"PDF not found: {pdf_path}"
            )

        # Configure pipeline
        config = BestPracticesConfig(
            dpi=dpi,
            auto_dpi=True,
            super_resolution=use_super_resolution,
            symbol_method=(
                SymbolRecognitionMethod.RAG if use_symbol_rag
                else SymbolRecognitionMethod.HARDCODED
            ),
            straighten_lines=True,
            straighten_tolerance_deg=straighten_tolerance_deg,
            connect_endpoints=connect_endpoints,
            gemini_visual_qa=gemini_visual_qa,
        )

        # Run pipeline
        pipeline = BestPracticesPipeline(config)
        result = await pipeline.process_pdf(
            pdf_path=pdf_file,
            page=page,
            output_dir=Path(output_dir) if output_dir else None,
        )

        # Helper to get entity type as string
        def get_entity_type_str(entity):
            if hasattr(entity.entity_type, 'value'):
                return entity.entity_type.value
            return str(entity.entity_type)

        # Summarize entities for response
        entity_summary = []
        for entity in result.entities[:MAX_ELEMENTS_IN_RESULT]:
            entity_summary.append({
                "type": get_entity_type_str(entity),
                "layer": entity.layer,
                "properties": {
                    k: v for k, v in entity.properties.items()
                    if k in ("start", "end", "center", "radius", "content", "block_name")
                },
            })

        # Build stage summary
        stage_summary = []
        for stage in result.stages:
            stage_summary.append({
                "name": stage.stage,
                "success": stage.success,
                "duration_ms": round(stage.duration_ms, 2),
                "errors": stage.errors[:3] if stage.errors else [],
                "warnings": stage.warnings[:3] if stage.warnings else [],
            })

        response_data = {
            "success": result.success,
            "entity_count": result.entity_count,
            "symbol_count": result.symbol_count,
            "line_count": result.line_count,
            "text_count": result.text_count,
            "total_duration_ms": round(result.total_duration_ms, 2),
            "stages": stage_summary,
            "entities_sample": entity_summary,
            "entities_truncated": len(result.entities) > MAX_ELEMENTS_IN_RESULT,
            "image_path": str(result.image_path) if result.image_path else None,
            "symbols_recognized": result.symbols_recognized[:10],
        }

        if result.success:
            return success_result(
                data=response_data,
                message=(
                    f"Pipeline complete: {result.entity_count} entities, "
                    f"{result.symbol_count} symbols via RAG, "
                    f"{round(result.total_duration_ms / 1000, 1)}s"
                ),
            )
        else:
            # Collect errors from failed stages
            all_errors = []
            for stage in result.stages:
                if not stage.success and stage.errors:
                    all_errors.extend(stage.errors)

            return error_result(
                ErrorCode.INTERNAL_ERROR,
                f"Pipeline failed: {'; '.join(all_errors[:3])}"
            )

    except Exception as e:
        logger.exception("run_best_practices_vectorization_failed", error=str(e))
        return error_result(
            ErrorCode.INTERNAL_ERROR,
            f"Best practices pipeline failed: {e}"
        )
