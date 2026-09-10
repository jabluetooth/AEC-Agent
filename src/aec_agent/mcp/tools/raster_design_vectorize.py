"""
MCP tools for AutoCAD Raster Design integration — vectorization.

Split from the former monolithic raster_design.py. Contains:
- raster_vectorize: Raster Design VTools vectorization (vline/vpline/varc/vcircle/vrect)
- raster_auto_vectorize: Python-side OpenCV detection → AutoCAD draw commands
"""


import structlog

from aec_agent.mcp.concurrency import with_tool_lock
from aec_agent.mcp.server import get_lock, mcp
from aec_agent.mcp.sidecar_client import SidecarError, call_autocad_command

from .base import ErrorCode, error_result, success_result
from .pdf_converter import convert_pdf_to_bitonal_tiff

logger = structlog.get_logger(__name__)


# =============================================================================
# VTools — Vectorize Raster Entities (vline, vpline, varc, vcircle, vrect)
# =============================================================================

@mcp.tool()
@with_tool_lock(get_lock())
async def raster_vectorize(
    tool: str = "vpline",
    method: str = "1p",
    points: list[list[float]] | None = None,
    target_layer: str | None = None,
) -> dict:
    """
    Vectorize raster entities to AutoCAD vector objects using Raster Design VTools.

    Uses the actual Raster Design vectorization commands (vline, vpline, varc,
    vcircle, vrect) to convert bitonal raster entities into native AutoCAD
    lines, polylines, arcs, circles, and rectangles.

    Requires AutoCAD Raster Design and a bitonal raster image to be attached.
    This operation is queued — use raster_get_entity_count to verify results.

    Args:
        tool: VTool to use. Options:
            - "vline": Convert raster line to vector line
            - "vpline": Convert raster line to vector polyline (default)
            - "varc": Convert raster arc to vector arc
            - "vcircle": Convert raster circle to vector circle
            - "vrect": Convert raster rectangle to vector rectangle
        method: Pick method. Options:
            - "1p": One-pick — single click on raster entity (default)
            - "2p": Multi-pick — click multiple points to define entity
        points: Click points as [[x, y], ...]. If omitted, AutoCAD
                prompts for interactive picking.
        target_layer: Layer to place vectorized entities on (optional)

    Returns:
        Queued operation status

    Example:
        raster_vectorize("vpline", method="1p", points=[[100, 200]])
    """
    valid_tools = {"vline", "vpline", "varc", "vcircle", "vrect"}
    if tool not in valid_tools:
        return error_result(
            ErrorCode.INVALID_PARAMS,
            f"Invalid tool: {tool}. Valid: {', '.join(sorted(valid_tools))}"
        )

    if method not in ("1p", "2p"):
        return error_result(ErrorCode.INVALID_PARAMS, "method must be '1p' or '2p'")

    params = {
        "tool": tool,
        "method": method,
    }
    if points:
        params["points"] = points
    if target_layer:
        params["target_layer"] = target_layer.strip()

    try:
        result = await call_autocad_command("raster_vectorize", params)
        return result
    except SidecarError as e:
        return error_result(e.code, e.message, e.details)
    except Exception as e:
        logger.error("Unexpected error in raster_vectorize", error=str(e), exc_info=True)
        return error_result(ErrorCode.INTERNAL_ERROR, f"Unexpected error: {str(e)}")


# =============================================================================
# Automated Vectorization (Python OpenCV → AutoCAD draw commands)
# =============================================================================

@mcp.tool()
@with_tool_lock(get_lock())
async def raster_auto_vectorize(
    image_path: str,
    dpi: int = 300,
    scale: float = 1.0,
    target_layer: str | None = None,
    text_layer: str | None = None,
    symbol_layer: str | None = None,
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
    LOW-LEVEL: OpenCV vectorization step only. DO NOT call this directly for
    PDF or image files — use ``raster_pdf_to_vector_pipeline`` instead, which
    handles the complete workflow (convert, attach, cleanup, vectorize, fade,
    store).

    This tool ONLY runs the OpenCV detection on an already-processed bitonal
    TIFF and creates AutoCAD entities. It does NOT convert PDFs, attach images,
    despeckle, deskew, fade, or store to PostgreSQL.

    Prerequisites before calling this tool:
    1. Image must already be a bitonal TIFF (use raster_convert_pdf for PDFs)
    2. Image must already be attached in AutoCAD (use raster_attach_image)
    3. Image should already be cleaned (despeckle/deskew via raster_cleanup)

    Phase 2.5 Features (optional):
    - OCR masking: Detect text via Tesseract, create MText entities
    - Symbol detection: Match templates, insert blocks
    - AEC heuristics: Snap orthogonal lines, merge collinear segments

    Args:
        image_path: Absolute path to an already-processed bitonal TIFF image
        dpi: Image resolution in DPI (default 300)
        scale: Coordinate scale factor — must match raster_attach_image scale (default 1.0)
        target_layer: Layer for created geometry entities (optional)
        text_layer: Layer for MText entities from OCR (optional, uses target_layer if not set)
        symbol_layer: Layer for inserted blocks (optional, uses target_layer if not set)
        min_line_length: Min line length in pixels (default 50)
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
        Vectorization summary with entity counts and creation results

    Example:
        # Prefer raster_pdf_to_vector_pipeline instead of calling this directly
        raster_auto_vectorize("C:/plans/floor1_page1_bitonal.tif", dpi=300, scale=1.0)
        # With Phase 2.5 features:
        raster_auto_vectorize("C:/plans/floor1.tif", ocr_masking=True, aec_heuristics=True)
    """
    from .image_vectorizer import vectorize_bitonal_image

    if not image_path or not image_path.strip():
        return error_result(ErrorCode.INVALID_PARAMS, "image_path is required")

    try:
        # Step 1: Detect features from the bitonal image using OpenCV
        detection = vectorize_bitonal_image(
            image_path=image_path.strip(),
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

        created = {
            "lines": 0, "circles": 0, "arcs": 0, "ellipses": 0,
            "polylines": 0, "texts": 0, "blocks": 0, "errors": 0,
        }

        # Step 2: Create AutoCAD line entities
        for line in detection.lines:
            try:
                params = {
                    "start": [line.start[0], line.start[1], 0.0],
                    "end": [line.end[0], line.end[1], 0.0],
                }
                if target_layer:
                    params["layer"] = target_layer.strip()
                await call_autocad_command("draw_line", params)
                created["lines"] += 1
            except Exception as e:
                logger.warning("Failed to create line", error=str(e))
                created["errors"] += 1

        # Step 3: Create AutoCAD circle entities
        for circle in detection.circles:
            try:
                params = {
                    "center": [circle.center[0], circle.center[1], 0.0],
                    "radius": circle.radius,
                }
                if target_layer:
                    params["layer"] = target_layer.strip()
                await call_autocad_command("draw_circle", params)
                created["circles"] += 1
            except Exception as e:
                logger.warning("Failed to create circle", error=str(e))
                created["errors"] += 1

        # Step 4: Create AutoCAD arc entities
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
                created["arcs"] += 1
            except Exception as e:
                logger.warning("Failed to create arc", error=str(e))
                created["errors"] += 1

        # Step 5: Create AutoCAD ellipse entities
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
                created["ellipses"] += 1
            except Exception as e:
                logger.warning("Failed to create ellipse", error=str(e))
                created["errors"] += 1

        # Step 6: Create AutoCAD polyline entities (with bulge for curved segments)
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
                created["polylines"] += 1
            except Exception as e:
                logger.warning("Failed to create polyline", error=str(e))
                created["errors"] += 1

        # Step 7: Create MText entities from OCR-detected text (Phase 2.5)
        effective_text_layer = text_layer or target_layer
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
                created["texts"] += 1
            except Exception as e:
                logger.warning("Failed to create MText", error=str(e), text=text.text[:30])
                created["errors"] += 1

        # Step 8: Insert block references from symbol detection (Phase 2.5)
        effective_symbol_layer = symbol_layer or target_layer
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
                created["blocks"] += 1
            except Exception as e:
                logger.warning(
                    "Failed to insert block",
                    error=str(e),
                    block_name=block.block_name,
                )
                created["errors"] += 1

        total_created = (
            created["lines"] + created["circles"] + created["arcs"]
            + created["ellipses"] + created["polylines"]
            + created["texts"] + created["blocks"]
        )

        return success_result(
            data={
                "image_path": image_path,
                "image_size_px": [detection.image_width_px, detection.image_height_px],
                "dpi": dpi,
                "scale": scale,
                "detected": {
                    "lines": len(detection.lines),
                    "circles": len(detection.circles),
                    "arcs": len(detection.arcs),
                    "ellipses": len(detection.ellipses),
                    "polylines": len(detection.polylines),
                    "texts": len(detection.texts),
                    "blocks": len(detection.blocks),
                },
                "created": created,
                "total_entities_created": total_created,
                "target_layer": target_layer,
                "text_layer": effective_text_layer,
                "symbol_layer": effective_symbol_layer,
                "phase25_features": {
                    "ocr_masking": ocr_masking,
                    "symbol_detection": symbol_detection,
                    "aec_heuristics": aec_heuristics,
                },
            },
            message=(
                f"Auto-vectorized: {total_created} entities created "
                f"({created['lines']} lines, {created['circles']} circles, "
                f"{created['arcs']} arcs, {created['ellipses']} ellipses, "
                f"{created['polylines']} polylines, {created['texts']} texts, "
                f"{created['blocks']} blocks)"
            ),
        )

    except FileNotFoundError as e:
        return error_result(ErrorCode.INVALID_PARAMS, str(e))
    except RuntimeError as e:
        return error_result(ErrorCode.INTERNAL_ERROR, str(e))
    except SidecarError as e:
        return error_result(e.code, e.message, e.details)
    except Exception as e:
        logger.error("Unexpected error in raster_auto_vectorize", error=str(e), exc_info=True)
        return error_result(ErrorCode.INTERNAL_ERROR, f"Unexpected error: {str(e)}")
