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
            domain_str = m.domain.value if hasattr(m.domain, 'value') else str(m.domain)
            match_data.append({
                "symbol_id": str(m.symbol_id),
                "block_name": m.block_name,
                "display_name": m.display_name,
                "domain": domain_str,
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
            domain_str = m.domain.value if hasattr(m.domain, 'value') else str(m.domain)
            match_data.append({
                "symbol_id": str(m.symbol_id),
                "block_name": m.block_name,
                "display_name": m.display_name,
                "domain": domain_str,
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
