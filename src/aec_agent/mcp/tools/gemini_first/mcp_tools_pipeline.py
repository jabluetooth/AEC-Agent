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


from .mcp_tools_helpers import logger, MAX_ELEMENTS_IN_RESULT


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
    DEPRECATED: Use `vectorize_pdf` instead.

    MIGRATION: Replace with:
        >>> result = await vectorize_pdf(
        ...     pdf_path="drawing.pdf",
        ...     extraction_method="best",
        ...     preprocess=True,
        ...     use_symbol_rag=True,
        ...     refine_geometry=True,
        ...     validate=True,
        ... )

    Legacy 7-stage pipeline that combines the optimal algorithms
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


# =============================================================================
# Unified Pipeline Tool (Recommended)
# =============================================================================


@mcp.tool()
async def vectorize_pdf(
    pdf_path: str,
    page: int = 1,
    extraction_method: str = "hybrid",
    preprocess: bool = True,
    use_symbol_rag: bool = True,
    refine_geometry: bool = True,
    create_in_autocad: bool = True,
    validate: bool = False,
    dpi: int = 300,
    straighten_tolerance_deg: float = 5.0,
    connect_tolerance_px: float = 10.0,
    grid_size_px: float = 5.0,
) -> dict[str, Any]:
    """
    PRIMARY TOOL for PDF to AutoCAD vectorization. Use this tool whenever
    the user wants to vectorize a PDF, convert PDF to CAD, or create AutoCAD
    entities from a PDF drawing.

    This is the unified pipeline using Gemini Vision + OpenCV (no add-ons required).
    It consolidates all vectorization tools into a single, configurable entry point.

    Pipeline stages (configurable):
    1. PDF Rendering (always)
    2. Image Preprocessing (optional - denoise, deskew, binarize)
    3. Gemini Analysis (always - detects elements, scale, layers)
    4. Coordinate Calibration (always - maps pixels to DWG units)
    5. Entity Extraction (configurable method)
    6. Symbol Recognition (optional - RAG or hardcoded)
    7. Geometry Refinement (optional - straighten, connect, snap)
    8. AutoCAD Creation (optional)
    9. Validation (optional - Gemini visual QA)

    Args:
        pdf_path: Path to the PDF file to vectorize
        page: Page number to process (1-indexed, default 1)
        extraction_method: Extraction algorithm to use:
            - "direct": Gemini coordinates only (fast, less accurate)
            - "hybrid": Gemini + OpenCV (balanced, recommended)
            - "best": Optimal algorithm per entity type (highest quality)
            - "vtracer": Raster-to-vector (for scanned drawings)
        preprocess: Apply image preprocessing (denoise, deskew, binarize)
        use_symbol_rag: Use CLIP + pgvector RAG for symbol recognition
        refine_geometry: Apply geometry refinement (straighten, connect, snap)
        create_in_autocad: Create entities in AutoCAD after extraction
        validate: Run Gemini visual QA validation
        dpi: Rendering DPI (default 300)
        straighten_tolerance_deg: Snap lines to H/V/45° if within tolerance
        connect_tolerance_px: Connect endpoints within this distance
        grid_size_px: Snap to grid with this cell size

    Returns:
        Success result with:
        - total_entities: Total entities extracted
        - entities_by_type: Count by entity type
        - entities_created: Count created in AutoCAD (if enabled)
        - validation_passed: Whether validation passed (if enabled)
        - stages: Results from each pipeline stage
        - total_duration_ms: Total processing time

    Example:
        >>> # Simple extraction
        >>> result = await vectorize_pdf("/path/to/drawing.pdf")

        >>> # Full pipeline with AutoCAD creation
        >>> result = await vectorize_pdf(
        ...     pdf_path="/path/to/drawing.pdf",
        ...     extraction_method="best",
        ...     create_in_autocad=True,
        ...     validate=True,
        ... )

        >>> # Fast extraction without preprocessing
        >>> result = await vectorize_pdf(
        ...     pdf_path="/path/to/drawing.pdf",
        ...     extraction_method="direct",
        ...     preprocess=False,
        ...     refine_geometry=False,
        ... )
    """
    from .unified_pipeline import (
        UnifiedPipeline,
        PipelineConfig,
        ExtractionMethod,
        SymbolMethod,
    )

    logger.info(
        "vectorize_pdf_called",
        pdf_path=pdf_path,
        page=page,
        extraction_method=extraction_method,
        create_in_autocad=create_in_autocad,
    )

    try:
        # Validate PDF exists
        pdf_file = Path(pdf_path)
        if not pdf_file.exists():
            return error_result(
                ErrorCode.FILE_NOT_FOUND,
                f"PDF not found: {pdf_path}"
            )

        # Validate extraction method
        valid_methods = ["direct", "hybrid", "best", "vtracer"]
        if extraction_method not in valid_methods:
            return error_result(
                ErrorCode.INVALID_PARAMETER,
                f"Invalid extraction_method: {extraction_method}. Valid: {valid_methods}"
            )

        # Configure pipeline
        config = PipelineConfig(
            dpi=dpi,
            preprocess=preprocess,
            extraction_method=ExtractionMethod(extraction_method),
            symbol_method=SymbolMethod.RAG if use_symbol_rag else SymbolMethod.HARDCODED,
            refine_geometry=refine_geometry,
            straighten_tolerance_deg=straighten_tolerance_deg,
            connect_tolerance_px=connect_tolerance_px,
            grid_size_px=grid_size_px,
            create_in_autocad=create_in_autocad,
            validate=validate,
            gemini_visual_qa=validate,
        )

        # Validate page number (1-indexed from user, convert to 0-indexed)
        if page < 1:
            return error_result(
                ErrorCode.INVALID_PARAMETER,
                f"Page number must be >= 1, got {page}"
            )

        # Run unified pipeline (convert to 0-indexed)
        pipeline = UnifiedPipeline(config)
        result = await pipeline.process(
            pdf_path=str(pdf_file),
            page=page - 1,  # Convert 1-indexed to 0-indexed
        )

        # Build response
        response_data = result.to_dict()

        # Add entity sample if available
        if result.entities:
            entity_sample = []
            for entity in result.entities[:MAX_ELEMENTS_IN_RESULT]:
                entity_type = (
                    entity.entity_type.value
                    if hasattr(entity.entity_type, 'value')
                    else str(entity.entity_type)
                )
                entity_sample.append({
                    "type": entity_type,
                    "layer": entity.layer,
                    "properties": {
                        k: v for k, v in entity.properties.items()
                        if k in ("start", "end", "center", "radius", "content", "block_name")
                    },
                })
            response_data["entities_sample"] = entity_sample
            response_data["entities_truncated"] = len(result.entities) > MAX_ELEMENTS_IN_RESULT

        if result.success:
            return success_result(
                data=response_data,
                message=(
                    f"Vectorization complete: {result.total_entities} entities "
                    f"in {round(result.total_duration_ms / 1000, 1)}s"
                ),
            )
        else:
            return error_result(
                ErrorCode.INTERNAL_ERROR,
                f"Pipeline failed: {result.error}"
            )

    except Exception as e:
        logger.exception("vectorize_pdf_failed", error=str(e))
        return error_result(
            ErrorCode.INTERNAL_ERROR,
            f"Vectorization failed: {e}"
        )


# =============================================================================
# Scan2CAD Parity Tools - Batch Processing, DXF Export, Preview, Profiles
# =============================================================================


@mcp.tool()
async def batch_process_pdfs(
    input_dir: str,
    output_dir: str,
    file_pattern: str = "*.pdf",
    max_parallel: int = 4,
    extraction_method: str = "hybrid",
    export_dxf: bool = True,
    overwrite_existing: bool = False,
) -> dict[str, Any]:
    """
    Batch helper: Process multiple PDF files using vectorize_pdf internally.

    Use this when you need to vectorize many PDFs at once. For single files,
    use `vectorize_pdf` directly.

    Processes all matching PDFs in a directory with parallel execution,
    error isolation per file, and comprehensive reporting.

    Args:
        input_dir: Directory containing PDF files
        output_dir: Directory for output files (DXF, previews)
        file_pattern: Glob pattern for files (default "*.pdf")
        max_parallel: Maximum concurrent files (default 4)
        extraction_method: Extraction method (direct, hybrid, best, vtracer)
        export_dxf: Export results to DXF files (default True)
        overwrite_existing: Overwrite existing output files (default False)

    Returns:
        Batch result with status and per-file results
    """
    try:
        from pathlib import Path
        from .batch_processor import BatchConfig, BatchProcessor

        input_path = Path(input_dir)
        output_path = Path(output_dir)

        if not input_path.exists():
            return error_result(
                ErrorCode.FILE_NOT_FOUND,
                f"Input directory not found: {input_dir}"
            )

        # Create output directory if needed
        output_path.mkdir(parents=True, exist_ok=True)

        config = BatchConfig(
            input_dir=input_path,
            output_dir=output_path,
            file_pattern=file_pattern,
            max_parallel=max_parallel,
            extraction_method=extraction_method,
            export_dxf=export_dxf,
            overwrite_existing=overwrite_existing,
        )

        processor = BatchProcessor(config)
        result = await processor.process()

        return success_result(
            data=result.to_dict(),
            message=(
                f"Batch complete: {result.success_count}/{result.total_files} "
                f"succeeded ({result.success_rate:.0%})"
            ),
        )

    except Exception as e:
        logger.exception("batch_process_failed", error=str(e))
        return error_result(
            ErrorCode.INTERNAL_ERROR,
            f"Batch processing failed: {e}"
        )
