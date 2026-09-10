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


from .mcp_tools_helpers import logger, _store_extraction_to_database, _summarize_hybrid_extraction, _summarize_creation


# =============================================================================
# HYBRID EXTRACTION TOOL (Gemini + OpenCV + YOLO Fusion)
# =============================================================================



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
    DEPRECATED: Use `vectorize_pdf` instead.

    MIGRATION: Replace with:
        >>> result = await vectorize_pdf(
        ...     pdf_path="drawing.pdf",
        ...     extraction_method="hybrid",
        ...     create_in_autocad=True,
        ... )

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
            jtype = junction.junction_type.value if hasattr(junction.junction_type, 'value') else str(junction.junction_type)
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
                    "type": j.junction_type.value if hasattr(j.junction_type, 'value') else str(j.junction_type),
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
