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


from .mcp_tools_helpers import logger, _summarize_analysis, _summarize_extraction, _summarize_creation

# ==============================================================================
# PHASE 5: AutoCAD Entity Creation (Draw in DWG)
# ==============================================================================


@mcp.tool()
async def gemini_create_entities(
    file_path: str,
    page: int = 1,
    dpi: int = 300,
    model: str = "gemini-1.5-flash",
    include_opencv: bool = True,
    create_layers: bool = True,
) -> dict[str, Any]:
    """
    DEPRECATED: Use vectorize_pdf with create_in_autocad=True instead.
    This tool only extracts text (MTEXT) reliably. For full geometry
    extraction (lines, circles, arcs), use vectorize_pdf with
    extraction_method="hybrid".

    Legacy tool that combines Phases 1-5 of the Gemini-First pipeline:
    1. Phase 1: Render PDF to high-quality image
    2. Phase 2: Analyze with Gemini Vision
    3. Phase 3: Calibrate coordinates
    4. Phase 4: Extract entities (Gemini-only, no OpenCV)
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
    model: str = "gemini-1.5-flash",
    include_opencv: bool = True,
    create_layers: bool = True,
) -> dict[str, Any]:
    """
    DEPRECATED: Use `vectorize_pdf` instead.

    Complete PDF-to-AutoCAD vectorization using Gemini-First pipeline.

    MIGRATION: Replace with:
        >>> result = await vectorize_pdf(
        ...     pdf_path="mechanical_plan.pdf",
        ...     extraction_method="hybrid",
        ...     create_in_autocad=True,
        ... )

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
        model: Gemini model for analysis
        include_opencv: Use OpenCV for special regions (default True)
        create_layers: Create layers that don't exist (default True)

    Returns:
        Success result with complete pipeline results.
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
    model: str = "gemini-1.5-flash",
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

        status_str = validation.status.value if hasattr(validation.status, 'value') else str(validation.status)
        logger.info(
            "gemini_validate_extraction_success",
            status=status_str,
            accuracy=validation.accuracy_estimate,
            issues=len(validation.issues),
        )

        return success_result(
            data=validation.to_dict(),
            message=f"Validation {status_str}: "
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
    model: str = "gemini-1.5-flash",
    include_opencv: bool = True,
    create_layers: bool = True,
    validate: bool = True,
    max_validation_iterations: int = 3,
    apply_corrections: bool = True,
) -> dict[str, Any]:
    """
    DEPRECATED: Use `vectorize_pdf` instead.

    Run the complete Gemini-First PDF-to-AutoCAD pipeline (Phases 1-6).

    MIGRATION: Replace with:
        >>> result = await vectorize_pdf(
        ...     pdf_path="floor_plan.pdf",
        ...     extraction_method="hybrid",
        ...     create_in_autocad=True,
        ...     validate=True,
        ... )

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
            validation_status_str = validation.status.value if hasattr(validation.status, 'value') else str(validation.status)

            logger.info(
                "gemini_complete_phase6_done",
                status=validation_status_str,
                accuracy=validation.accuracy_estimate,
            )

        # ═══════════════════════════════════════════════════════════════════
        # BUILD RESULT
        # ═══════════════════════════════════════════════════════════════════
        validation_status_for_log = (
            (validation.status.value if hasattr(validation.status, 'value') else str(validation.status))
            if validate else "skipped"
        )
        logger.info(
            "gemini_complete_pipeline_done",
            drawing_type=analysis.drawing_type,
            entities_created=creation.success_count,
            validation_status=validation_status_for_log,
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
            val_status = validation.status.value if hasattr(validation.status, 'value') else str(validation.status)
            result_data["phases"]["phase6_validation"] = validation_data
            result_data["summary"]["validation_status"] = val_status
            result_data["summary"]["validation_accuracy"] = f"{validation.accuracy_estimate:.0f}%"
            result_data["summary"]["validation_issues"] = len(validation.issues)

        # Build message
        if validate:
            val_status_msg = validation.status.value if hasattr(validation.status, 'value') else str(validation.status)
            message = (
                f"Complete pipeline finished: {creation.success_count} entities created, "
                f"validation {val_status_msg} ({validation.accuracy_estimate:.0f}% accurate)"
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
