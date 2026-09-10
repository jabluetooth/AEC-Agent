"""
MCP tools for AutoCAD Raster Design integration — complete pipeline and
post-processing.

Split from the former monolithic raster_design.py. Contains:
- raster_pdf_to_vector_pipeline: full PDF/image-to-vector pipeline (DEPRECATED,
  see vectorize_pdf)
- raster_topology_cleanup: standalone NetworkX-based geometry post-processing
"""


import structlog

from aec_agent.mcp.concurrency import with_tool_lock
from aec_agent.mcp.server import get_lock, mcp
from aec_agent.mcp.sidecar_client import SidecarError, call_autocad_command

from .base import ErrorCode, error_result, success_result
from .pdf_converter import convert_pdf_to_bitonal_tiff

logger = structlog.get_logger(__name__)


# =============================================================================
# Complete PDF-to-Vector Pipeline
# =============================================================================

@mcp.tool()
@with_tool_lock(get_lock())
async def raster_pdf_to_vector_pipeline(
    file_path: str,
    page: int = 1,
    dpi: int = 300,
    scale: float = 1.0,
    mode: str = "auto",
    target_layer: str | None = None,
    text_layer: str | None = None,
    symbol_layer: str | None = None,
    fade_percent: int = 70,
    store_in_db: bool = True,
    min_line_length: int = 50,
    max_line_gap: int = 15,
    hough_threshold: int = 80,
    min_circle_radius: int = 20,
    max_circle_radius: int = 500,
    hough_circles_dp: float = 1.2,
    hough_circles_param1: float = 200.0,
    hough_circles_param2: float = 200.0,
    hough_circles_min_dist: int = 100,
    contour_epsilon_factor: float = 0.01,
    min_contour_points: int = 5,
    min_contour_area: float = 3000.0,
    ellipse_fit_threshold: float = 0.85,
    arc_coverage_min: float = 30.0,
    arc_coverage_max: float = 350.0,
    line_merge_angle_tol: float = 5.0,
    line_merge_dist_tol: float = 15.0,
    circle_merge_center_tol: float = 30.0,
    circle_merge_radius_tol: float = 20.0,
    # Signal Restoration
    signal_restore: bool = True,
    signal_close_kernel_length: int = 15,
    signal_close_angle_step: int = 15,
    # Iterative Masking
    mask_detected_circles: bool = True,
    mask_detected_lines: bool = True,
    mask_thickness: int = 5,
    # Skeletonization & Topology
    skeletonize: bool = True,
    topology_cleanup: bool = True,
    snap_tolerance: float = 5.0,
    # Phase 2.5: OCR Text Masking
    ocr_masking: bool = True,
    ocr_min_confidence: int = 60,
    ocr_lang: str = "eng",
    # Phase 2.5: Symbol Detection
    symbol_detection: bool = True,
    symbol_backend: str = "auto",  # "auto", "yolo", "template"
    symbol_threshold: float = 0.8,
    # Phase 2.5.1: YOLO Symbol Detection
    yolo_model_path: str | None = None,
    yolo_confidence: float = 0.5,
    yolo_iou_threshold: float = 0.45,
    # Phase 2.5: AEC Heuristics
    aec_heuristics: bool = True,
    orthogonal_snap: bool = True,
    orthogonal_angle_tolerance: float = 2.0,
    collinear_merge: bool = True,
) -> dict:
    """
    DEPRECATED: Use `vectorize_pdf` instead (Gemini + OpenCV, no add-ons required).

    MIGRATION: Replace with:
        >>> result = await vectorize_pdf(
        ...     pdf_path="drawing.pdf",
        ...     extraction_method="hybrid",
        ...     create_in_autocad=True,
        ... )

    REQUIRES AutoCAD Raster Design add-on. Only use this if you specifically
    need Raster Design features or the user explicitly requests traditional
    raster-to-vector conversion.

    This tool uses AutoCAD's Raster Design toolset to convert PDF or image
    files to AutoCAD vector entities.  Accepts PDF, TIFF, PNG, JPG, and BMP.
    Do NOT call raster_auto_vectorize, raster_convert_pdf,
    raster_attach_image, or raster_cleanup individually.

    Pipeline steps (all automatic, all inside this one call):
    1. Detects input type (PDF vs image file)
    2. For image files (TIFF/PNG/JPG/BMP): skips PDF steps, goes
       straight to attach → cleanup → vectorize
    3. For vector PDFs: imports directly via PDFIMPORT
    4. For scanned PDFs:
       a. Converts PDF to bitonal TIFF (Python-side, at specified DPI)
       b. Attaches bitonal TIFF to AutoCAD
       c. Despeckles (removes scan noise)
       d. Deskews (straightens rotation)
       e. (Phase 2.5) OCR text detection → MText entities
       f. (Phase 2.5) Symbol template matching → Block insertions
       g. Detects features via OpenCV (lines, circles, arcs, ellipses, polylines)
       h. (Phase 2.5) AEC heuristics: orthogonal snap, collinear merge
       i. Creates AutoCAD entities via draw commands
    5. Fades original raster image for background reference
    6. Extracts all entities and stores in PostgreSQL with geometry and embeddings

    Vectorization uses Python-side OpenCV (HoughLinesP, HoughCircles,
    findContours + fitEllipse) instead of Raster Design VTools, which are
    interactive and cannot be automated via SendStringToExecute.

    Phase 2.5 Features (optional, all default to False):
    - ocr_masking: Detect text via Tesseract, mask from image, create MText
    - symbol_detection: Match templates, mask from image, insert blocks
    - aec_heuristics: Snap near-orthogonal lines, merge collinear segments

    Args:
        file_path: Absolute path to the PDF file
        page: PDF page to import (default 1)
        dpi: Render resolution for PDF-to-TIFF conversion (default 300)
        scale: Import scale factor (default 1.0)
        mode: Detection mode — "auto", "vector", or "scanned" (default "auto")
        target_layer: Layer for vectorized geometry (optional)
        text_layer: Layer for MText entities from OCR (optional, uses target_layer if not set)
        symbol_layer: Layer for inserted blocks (optional, uses target_layer if not set)
        fade_percent: Raster fade percentage 0-100 (default 70)
        store_in_db: Store results in PostgreSQL (default True)
        min_line_length: Min line length in pixels for detection (default 50)
        max_line_gap: Max gap to merge line segments in pixels (default 15)
        hough_threshold: Line detection sensitivity — lower = more lines (default 80)
        min_circle_radius: Min circle radius in pixels (default 20)
        max_circle_radius: Max circle radius in pixels, 0=unlimited (default 500)
        hough_circles_dp: Accumulator resolution ratio — lower = finer (default 1.2)
        hough_circles_param1: Canny high threshold inside HoughCircles (default 200)
        hough_circles_param2: Circle center accumulator threshold — higher = fewer
                              but more confident circles (default 200)
        hough_circles_min_dist: Min distance between circle centers in pixels (default 100)
        contour_epsilon_factor: Polyline simplification factor (default 0.01)
        min_contour_points: Min points per polyline (default 5)
        min_contour_area: Min contour area in pixels to filter noise (default 3000)
        ellipse_fit_threshold: Goodness-of-fit for ellipse/arc detection 0-1 (default 0.85)
        arc_coverage_min: Min arc coverage in degrees to accept as arc (default 30)
        arc_coverage_max: Max arc coverage degrees before full ellipse (default 350)
        line_merge_angle_tol: Max angle diff in degrees to merge duplicate lines (default 5)
        line_merge_dist_tol: Max perpendicular distance in pixels to merge lines (default 15)
        circle_merge_center_tol: Max center distance in pixels to merge circles (default 30)
        circle_merge_radius_tol: Max radius diff in pixels to merge circles (default 20)
        signal_restore: Bridge gaps in dashed/broken lines via directional closing (default True)
        signal_close_kernel_length: Directional kernel length in pixels (default 15)
        signal_close_angle_step: Degrees between directional passes (default 15)
        mask_detected_circles: Erase detected circle pixels before contour pass (default True)
        mask_detected_lines: Erase detected line pixels before circle detection to
                             prevent line intersections being misidentified (default True)
        mask_thickness: Pixel thickness of the erasure mask (default 5)
        skeletonize: Reduce thick lines to 1px centerlines before detection (default True)
        topology_cleanup: Merge degree-2 breaks and snap dangling endpoints (default True)
        snap_tolerance: Max distance in drawing units to snap endpoints (default 5.0)
        ocr_masking: Enable OCR text detection and MText creation (default False)
        ocr_min_confidence: Minimum OCR confidence 0-100 (default 60)
        ocr_lang: Tesseract language code (default "eng")
        symbol_detection: Enable symbol detection and block insertion (default False)
        symbol_backend: Detection backend - "auto", "yolo", or "template" (default "auto")
        symbol_threshold: Template match confidence threshold 0-1 (default 0.8)
        yolo_model_path: Custom YOLO model path (.pt or .onnx), uses default if None
        yolo_confidence: YOLO confidence threshold 0-1 (default 0.5)
        yolo_iou_threshold: YOLO IoU threshold for NMS 0-1 (default 0.45)
        aec_heuristics: Enable AEC-specific geometric cleanup (default False)
        orthogonal_snap: Snap near-orthogonal lines to exact 0/90 degrees (default True)
        orthogonal_angle_tolerance: Degrees tolerance for orthogonal snapping (default 2.0)
        collinear_merge: Merge collinear line segments (default True)

    Returns:
        Pipeline results with step details, entity counts, and PostgreSQL project info

    Example:
        raster_pdf_to_vector_pipeline("C:/plans/floor1.pdf", mode="auto", store_in_db=True)
        raster_pdf_to_vector_pipeline("C:/scans/bracket.png", mode="scanned")
        # With Phase 2.5 features:
        raster_pdf_to_vector_pipeline("C:/plans/floor1.pdf", ocr_masking=True, aec_heuristics=True)
    """
    if not file_path or not file_path.strip():
        return error_result(ErrorCode.INVALID_PARAMS, "file_path is required")

    if mode not in ("auto", "vector", "scanned"):
        return error_result(ErrorCode.INVALID_PARAMS, "mode must be 'auto', 'vector', or 'scanned'")

    # Detect if input is an image file (not a PDF).
    # If so, skip PDF-specific steps and go straight to attach/vectorize.
    import os
    file_ext = os.path.splitext(file_path.strip())[1].lower()
    is_image_file = file_ext in (".tif", ".tiff", ".png", ".jpg", ".jpeg", ".bmp")

    steps_completed = []
    step_errors = []

    try:
        # Step 1: Record baseline entity count
        baseline = await call_autocad_command("raster_get_entity_count")
        baseline_count = 0
        if baseline.get("success") and baseline.get("data"):
            baseline_count = baseline["data"].get("total", 0)
        steps_completed.append({"step": "baseline_count", "count": baseline_count})

        # Step 2: Determine input type and processing mode
        detected_mode = mode

        if is_image_file:
            # ---- Image file: skip PDF steps, go straight to vectorize ----
            detected_mode = "scanned"
            tiff_path = file_path.strip()
            steps_completed.append({
                "step": "detect_input_type",
                "type": "image",
                "extension": file_ext,
                "skipped_pdf_steps": True,
            })
            logger.info(
                "Input is an image file, skipping PDF conversion",
                file_ext=file_ext,
            )
        else:
            # ---- PDF file: try vector import, fall back to scanned ----
            if mode == "auto":
                detected_mode = "vector"  # try vector first, fall back to scanned

            if detected_mode == "vector":
                # ---- Vector PDF: import via PDFIMPORT ----
                import_params = {
                    "file_path": file_path.strip(),
                    "page": page,
                    "insertion_point": [0.0, 0.0],
                    "scale": float(scale),
                    "rotation": 0.0,
                }
                if target_layer:
                    import_params["target_layer"] = target_layer.strip()

                import_result = await call_autocad_command("raster_import_pdf", import_params)
                steps_completed.append({
                    "step": "pdf_import_vector",
                    "success": import_result.get("success", False),
                })

                # Barrier: entity count blocks until PDFIMPORT finishes in OnIdle
                post_import = await call_autocad_command("raster_get_entity_count")
                post_import_count = 0
                if post_import.get("success") and post_import.get("data"):
                    post_import_count = post_import["data"].get("total", 0)

                new_entities = post_import_count - baseline_count
                steps_completed.append({
                    "step": "post_vector_import_count",
                    "count": post_import_count,
                    "new_entities": new_entities,
                })

                # Auto-detect fallback: if no entities added, switch to scanned
                if mode == "auto" and new_entities <= 0:
                    detected_mode = "scanned"
                    logger.info(
                        "Auto-detect: no vector entities from PDFIMPORT, switching to scanned pipeline"
                    )

        if detected_mode == "scanned" and not is_image_file:
            # ---- Scanned PDF: convert to TIFF first ----

            # Step A: Convert PDF to bitonal TIFF (Python-side)
            # AutoCAD Raster Design cannot attach PDF files directly.
            # We render the PDF page to a high-DPI bitonal TIFF first.
            try:
                tiff_path = convert_pdf_to_bitonal_tiff(
                    pdf_path=file_path.strip(),
                    page=page,
                    dpi=dpi,
                    threshold=128,
                )
                steps_completed.append({
                    "step": "convert_pdf_to_bitonal_tiff",
                    "success": True,
                    "tiff_path": tiff_path,
                })
            except Exception as e:
                logger.error("PDF to TIFF conversion failed", error=str(e), exc_info=True)
                steps_completed.append({
                    "step": "convert_pdf_to_bitonal_tiff",
                    "success": False,
                    "error": str(e),
                })
                return error_result(
                    ErrorCode.INTERNAL_ERROR,
                    f"PDF to bitonal TIFF conversion failed: {e}",
                    f"Steps completed: {[s['step'] for s in steps_completed]}",
                )

            # Step B: Attach the bitonal TIFF as a raster image in AutoCAD
            attach_params = {
                "file_path": tiff_path,
                "insertion_point": [0.0, 0.0],
                "scale": float(scale),
            }
            if target_layer:
                attach_params["target_layer"] = target_layer.strip()

            attach_result = await call_autocad_command("raster_attach_image", attach_params)
            steps_completed.append({
                "step": "attach_bitonal_tiff",
                "success": attach_result.get("success", False),
                "tiff_path": tiff_path,
            })

            # Despeckle: remove noise spots
            despeckle_result = await call_autocad_command("raster_cleanup", {
                "operation": "despeckle",
                "blob_size": 3,
            })
            steps_completed.append({
                "step": "despeckle",
                "success": despeckle_result.get("success", False),
            })

            # Barrier
            await call_autocad_command("raster_get_entity_count")

            # Deskew: straighten skewed scans
            deskew_result = await call_autocad_command("raster_cleanup", {
                "operation": "deskew",
            })
            steps_completed.append({
                "step": "deskew",
                "success": deskew_result.get("success", False),
            })

            # Barrier
            await call_autocad_command("raster_get_entity_count")

            # Vectorize: Python-side OpenCV detection → AutoCAD draw commands
            # Raster Design VTools (vline, vpline) are interactive and cannot
            # be automated. Instead, we detect features from the bitonal TIFF
            # using OpenCV and create AutoCAD entities via draw commands.
            #
            # IMPORTANT: pass the same ``scale`` used for image attachment so
            # that vectorized entity coordinates match the raster image.
            from .image_vectorizer import vectorize_bitonal_image

            try:
                detection = vectorize_bitonal_image(
                    image_path=tiff_path,
                    dpi=dpi,
                    scale=scale,
                    min_line_length=min_line_length,
                    max_line_gap=max_line_gap,
                    hough_threshold=hough_threshold,
                    min_circle_radius=min_circle_radius,
                    max_circle_radius=max_circle_radius,
                    hough_circles_dp=hough_circles_dp,
                    hough_circles_param1=hough_circles_param1,
                    hough_circles_param2=hough_circles_param2,
                    hough_circles_min_dist=hough_circles_min_dist,
                    contour_epsilon_factor=contour_epsilon_factor,
                    min_contour_points=min_contour_points,
                    min_contour_area=min_contour_area,
                    ellipse_fit_threshold=ellipse_fit_threshold,
                    arc_coverage_min=arc_coverage_min,
                    arc_coverage_max=arc_coverage_max,
                    line_merge_angle_tol=line_merge_angle_tol,
                    line_merge_dist_tol=line_merge_dist_tol,
                    circle_merge_center_tol=circle_merge_center_tol,
                    circle_merge_radius_tol=circle_merge_radius_tol,
                    signal_restore=signal_restore,
                    signal_close_kernel_length=signal_close_kernel_length,
                    signal_close_angle_step=signal_close_angle_step,
                    mask_detected_circles=mask_detected_circles,
                    mask_detected_lines=mask_detected_lines,
                    mask_thickness=mask_thickness,
                    skeletonize=skeletonize,
                    topology_cleanup=topology_cleanup,
                    snap_tolerance=snap_tolerance,
                    # Phase 2.5 parameters
                    ocr_masking=ocr_masking,
                    ocr_min_confidence=ocr_min_confidence,
                    ocr_lang=ocr_lang,
                    symbol_detection=symbol_detection,
                    symbol_backend=symbol_backend,
                    symbol_threshold=symbol_threshold,
                    # Phase 2.5.1: YOLO parameters
                    yolo_model_path=yolo_model_path,
                    yolo_confidence=yolo_confidence,
                    yolo_iou_threshold=yolo_iou_threshold,
                    aec_heuristics=aec_heuristics,
                    orthogonal_snap=orthogonal_snap,
                    orthogonal_angle_tolerance=orthogonal_angle_tolerance,
                    collinear_merge=collinear_merge,
                )
                steps_completed.append({
                    "step": "opencv_detect_features",
                    "success": True,
                    "lines": len(detection.lines),
                    "circles": len(detection.circles),
                    "arcs": len(detection.arcs),
                    "ellipses": len(detection.ellipses),
                    "polylines": len(detection.polylines),
                    "texts": len(detection.texts),
                    "blocks": len(detection.blocks),
                })
            except Exception as e:
                logger.error("OpenCV vectorization failed", error=str(e), exc_info=True)
                steps_completed.append({
                    "step": "opencv_detect_features",
                    "success": False,
                    "error": str(e),
                })
                detection = None

            # Create AutoCAD entities from detected features
            created_counts = {
                "lines": 0, "circles": 0, "arcs": 0,
                "ellipses": 0, "polylines": 0, "texts": 0,
                "blocks": 0, "errors": 0,
            }
            effective_text_layer = text_layer or target_layer
            effective_symbol_layer = symbol_layer or target_layer

            if detection:
                # Create lines
                for line in detection.lines:
                    try:
                        params = {
                            "start": [line.start[0], line.start[1], 0.0],
                            "end": [line.end[0], line.end[1], 0.0],
                        }
                        if target_layer:
                            params["layer"] = target_layer.strip()
                        await call_autocad_command("draw_line", params)
                        created_counts["lines"] += 1
                    except Exception as e:
                        logger.error("Failed to create line entity", error=str(e), exc_info=True)
                        created_counts["errors"] += 1

                # Create circles
                for circle in detection.circles:
                    try:
                        params = {
                            "center": [circle.center[0], circle.center[1], 0.0],
                            "radius": circle.radius,
                        }
                        if target_layer:
                            params["layer"] = target_layer.strip()
                        await call_autocad_command("draw_circle", params)
                        created_counts["circles"] += 1
                    except Exception as e:
                        logger.error("Failed to create circle entity", error=str(e), exc_info=True)
                        created_counts["errors"] += 1

                # Create arcs
                for arc in detection.arcs:
                    try:
                        params = {
                            "center": [arc.center[0], arc.center[1], 0.0],
                            "radius": arc.radius,
                            "start_angle": arc.start_angle,
                            "end_angle": arc.end_angle,
                        }
                        if target_layer:
                            params["layer"] = target_layer.strip()
                        await call_autocad_command("draw_arc", params)
                        created_counts["arcs"] += 1
                    except Exception as e:
                        logger.error("Failed to create arc entity", error=str(e), exc_info=True)
                        created_counts["errors"] += 1

                # Create ellipses
                for ellipse in detection.ellipses:
                    try:
                        params = {
                            "center": [ellipse.center[0], ellipse.center[1], 0.0],
                            "major_axis_endpoint": [
                                ellipse.major_axis_endpoint[0],
                                ellipse.major_axis_endpoint[1],
                                0.0,
                            ],
                            "axis_ratio": ellipse.axis_ratio,
                            "start_angle": ellipse.start_angle,
                            "end_angle": ellipse.end_angle,
                        }
                        if target_layer:
                            params["layer"] = target_layer.strip()
                        await call_autocad_command("draw_ellipse", params)
                        created_counts["ellipses"] += 1
                    except Exception as e:
                        logger.error("Failed to create ellipse entity", error=str(e), exc_info=True)
                        created_counts["errors"] += 1

                # Create polylines (with bulge for rounded corners)
                for pline in detection.polylines:
                    try:
                        pts = [[p[0], p[1], 0.0] for p in pline.points]
                        params = {
                            "points": pts,
                            "closed": pline.closed,
                        }
                        if pline.bulges and any(b != 0.0 for b in pline.bulges):
                            params["bulges"] = pline.bulges
                        if target_layer:
                            params["layer"] = target_layer.strip()
                        await call_autocad_command("draw_polyline", params)
                        created_counts["polylines"] += 1
                    except Exception as e:
                        logger.error("Failed to create polyline entity", error=str(e), exc_info=True)
                        created_counts["errors"] += 1

                # Create MText entities from OCR-detected text (Phase 2.5)
                for text in detection.texts:
                    try:
                        params = {
                            "text": text.text,
                            "position": [text.position[0], text.position[1], 0.0],
                            "height": text.height if text.height > 0 else 2.5,
                            "rotation": text.rotation,
                        }
                        if text.width and text.width > 0:
                            params["width"] = text.width
                        if effective_text_layer:
                            params["layer"] = effective_text_layer.strip()
                        await call_autocad_command("draw_mtext", params)
                        created_counts["texts"] += 1
                    except Exception as e:
                        logger.error("Failed to create mtext entity", error=str(e), exc_info=True)
                        created_counts["errors"] += 1

                # Insert block references from symbol detection (Phase 2.5)
                for block in detection.blocks:
                    try:
                        params = {
                            "block_name": block.block_name,
                            "position": [block.position[0], block.position[1], 0.0],
                            "scale": block.scale if block.scale > 0 else 1.0,
                            "rotation": block.rotation,
                        }
                        if effective_symbol_layer:
                            params["layer"] = effective_symbol_layer.strip()
                        await call_autocad_command("insert_block", params)
                        created_counts["blocks"] += 1
                    except Exception as e:
                        logger.error("Failed to insert block reference", error=str(e), exc_info=True)
                        created_counts["errors"] += 1

            total_created = (
                created_counts["lines"] + created_counts["circles"]
                + created_counts["arcs"] + created_counts["ellipses"]
                + created_counts["polylines"] + created_counts["texts"]
                + created_counts["blocks"]
            )
            steps_completed.append({
                "step": "create_autocad_entities",
                "success": total_created > 0,
                "created": created_counts,
                "total": total_created,
            })

            # Post-vectorize entity count
            post_vectorize = await call_autocad_command("raster_get_entity_count")
            post_vectorize_count = 0
            if post_vectorize.get("success") and post_vectorize.get("data"):
                post_vectorize_count = post_vectorize["data"].get("total", 0)
            steps_completed.append({
                "step": "post_vectorize_count",
                "count": post_vectorize_count,
                "new_entities": post_vectorize_count - baseline_count,
            })

            # Fade raster image for background reference
            fade_result = await call_autocad_command("raster_fade_image", {
                "fade_percent": fade_percent,
            })
            steps_completed.append({
                "step": "fade_image",
                "success": fade_result.get("success", False),
                "fade_percent": fade_percent,
            })

        # Step 3: Final entity count and breakdown
        final_count_result = await call_autocad_command("raster_get_entity_count")
        final_count = 0
        type_breakdown = {}
        layer_breakdown = {}
        if final_count_result.get("success") and final_count_result.get("data"):
            final_count = final_count_result["data"].get("total", 0)
            type_breakdown = final_count_result["data"].get("by_type", {})
            layer_breakdown = final_count_result["data"].get("by_layer", {})

        steps_completed.append({
            "step": "final_count",
            "count": final_count,
            "entities_added": final_count - baseline_count,
        })

        # Step 4: Store in PostgreSQL
        db_result = None
        if store_in_db:
            from aec_agent.mcp.server import get_database_pool, get_sync_manager

            pool = get_database_pool()
            sm = get_sync_manager()

            if pool and sm:
                try:
                    extraction = await sm.trigger_full_sync(
                        "autocad", file_path, force=True
                    )
                    db_result = {
                        "project_id": str(extraction.project_id),
                        "elements_extracted": extraction.elements_extracted,
                        "embeddings_generated": extraction.embeddings_generated,
                        "relationships_computed": extraction.relationships_computed,
                        "duration_ms": extraction.duration_ms,
                    }
                    if extraction.errors:
                        db_result["errors"] = extraction.errors

                    steps_completed.append({
                        "step": "postgresql_storage",
                        "success": extraction.success,
                        "elements": extraction.elements_extracted,
                    })
                except Exception as e:
                    logger.error("PostgreSQL storage failed", error=str(e), exc_info=True)
                    step_errors.append(f"PostgreSQL storage: {str(e)}")
                    steps_completed.append({
                        "step": "postgresql_storage",
                        "success": False,
                        "error": str(e),
                    })
            else:
                step_errors.append("PostgreSQL not configured — entities not stored")
                steps_completed.append({
                    "step": "postgresql_storage",
                    "success": False,
                    "error": "Database not configured",
                })

        return success_result(
            data={
                "pipeline_mode": detected_mode,
                "file_path": file_path,
                "entities_before": baseline_count,
                "entities_after": final_count,
                "entities_added": final_count - baseline_count,
                "by_entity_type": type_breakdown,
                "by_layer": layer_breakdown,
                "postgresql": db_result,
                "steps": steps_completed,
                "errors": step_errors if step_errors else None,
            },
            message=(
                f"Pipeline complete ({detected_mode}): "
                f"{final_count - baseline_count} entities added"
            ),
        )

    except SidecarError as e:
        return error_result(e.code, e.message, e.details)
    except Exception as e:
        logger.error("Pipeline failed", error=str(e), exc_info=True)
        return error_result(
            ErrorCode.INTERNAL_ERROR,
            f"Pipeline failed: {str(e)}",
            f"Steps completed: {[s['step'] for s in steps_completed]}"
        )


# =============================================================================
# Topology Cleanup (standalone post-processing)
# =============================================================================

@mcp.tool()
@with_tool_lock(get_lock())
async def raster_topology_cleanup(
    lines_json: list[dict],
    polylines_json: list[dict] | None = None,
    snap_tolerance: float = 5.0,
) -> dict:
    """
    Post-process vectorized geometry by merging degree-2 breaks and snapping
    dangling endpoints using a NetworkX graph.  Use this when previously
    vectorized output has fragmented lines that should be continuous.

    This tool operates on already-extracted geometry (JSON arrays of lines and
    polylines) and returns cleaned geometry.  It does NOT read images or call
    AutoCAD — it is a pure-Python graph operation.

    Args:
        lines_json: List of line dicts with "start" [x,y] and "end" [x,y].
        polylines_json: Optional list of polyline dicts with "points" [[x,y],...] and "closed" bool.
        snap_tolerance: Max distance in drawing units to snap dangling endpoints (default 5.0).

    Returns:
        Cleaned geometry with merged lines and snapped endpoints.

    Example:
        raster_topology_cleanup(
            lines_json=[{"start": [0,0], "end": [10,0]}, {"start": [10.1,0], "end": [20,0]}],
            snap_tolerance=1.0,
        )
    """
    try:
        from .image_vectorizer import (
            DetectedLine,
            DetectedPolyline,
            VectorizationResult,
        )
        from .topology import (
            build_segment_graph,
            graph_to_vectorization_result,
            merge_degree2_nodes,
            snap_dangling_endpoints,
        )
    except ImportError as e:
        return error_result(
            ErrorCode.INTERNAL_ERROR,
            f"Required dependency not installed: {e}. "
            "Install with: pip install networkx>=3.0",
        )

    if not lines_json:
        return error_result(ErrorCode.INVALID_PARAMS, "lines_json is required and must not be empty")

    # Build VectorizationResult from JSON input
    vr = VectorizationResult()
    for ld in lines_json:
        start = tuple(ld.get("start", [0, 0]))
        end = tuple(ld.get("end", [0, 0]))
        vr.lines.append(DetectedLine(start=start, end=end))

    for pd in (polylines_json or []):
        pts = [tuple(p) for p in pd.get("points", [])]
        closed = pd.get("closed", False)
        if len(pts) >= 2:
            vr.polylines.append(DetectedPolyline(points=pts, closed=closed))

    lines_before = len(vr.lines)
    polylines_before = len(vr.polylines)

    try:
        graph = build_segment_graph(vr, snap_tolerance)
        graph = merge_degree2_nodes(graph)
        graph = snap_dangling_endpoints(graph, snap_tolerance)
        cleaned = graph_to_vectorization_result(graph)

        # Serialize back to JSON
        cleaned_lines = [
            {"start": list(l.start), "end": list(l.end)}
            for l in cleaned.lines
        ]
        cleaned_polylines = [
            {"points": [list(p) for p in pl.points], "closed": pl.closed}
            for pl in cleaned.polylines
        ]

        return success_result(
            data={
                "lines_before": lines_before,
                "lines_after": len(cleaned.lines),
                "polylines_before": polylines_before,
                "polylines_after": len(cleaned.polylines),
                "lines": cleaned_lines,
                "polylines": cleaned_polylines,
            },
            message=(
                f"Topology cleanup: {lines_before} lines → {len(cleaned.lines)}, "
                f"{polylines_before} polylines → {len(cleaned.polylines)}"
            ),
        )
    except Exception as e:
        logger.error("Topology cleanup failed", error=str(e), exc_info=True)
        return error_result(ErrorCode.INTERNAL_ERROR, f"Topology cleanup failed: {e}")
