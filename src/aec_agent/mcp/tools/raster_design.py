"""
MCP tools for AutoCAD Raster Design integration.

Provides PDF-to-DWG conversion pipeline:
1. Import PDF into AutoCAD (vector PDFs)
2. Attach raster images (scanned PDFs / images)
3. Cleanup raster images (despeckle, deskew, threshold)
4. Vectorize raster to AutoCAD entities
5. OCR text extraction

Async commands (import_pdf, cleanup, vectorize, ocr) use SendStringToExecute
on the sidecar and return a "queued" status immediately. Use
raster_get_status / raster_get_entity_count to check results after execution.
"""

from typing import Optional, List

from aec_agent.mcp.server import mcp, get_lock
from aec_agent.mcp.concurrency import with_tool_lock
from aec_agent.mcp.sidecar_client import call_autocad_command, SidecarError
from .base import success_result, error_result, ErrorCode

import structlog

logger = structlog.get_logger(__name__)


# =============================================================================
# PDF Import
# =============================================================================

@mcp.tool()
@with_tool_lock(get_lock())
async def raster_import_pdf(
    file_path: str,
    page: int = 1,
    insertion_point_x: float = 0.0,
    insertion_point_y: float = 0.0,
    scale: float = 1.0,
    rotation: float = 0.0,
    target_layer: Optional[str] = None
) -> dict:
    """
    Import a PDF file into AutoCAD as vector entities.

    Uses AutoCAD's -PDFIMPORT command to convert PDF vector/text content
    into native AutoCAD lines, arcs, text, and hatches.

    For scanned (raster) PDFs, use raster_attach_image + raster_vectorize instead.

    This operation is queued — use raster_get_entity_count to verify results.

    Args:
        file_path: Absolute path to the PDF file
        page: PDF page number to import (default 1)
        insertion_point_x: X coordinate for insertion (default 0)
        insertion_point_y: Y coordinate for insertion (default 0)
        scale: Import scale factor (default 1.0)
        rotation: Rotation angle in degrees (default 0)
        target_layer: Layer to place imported geometry on (optional)

    Returns:
        Queued operation status with import parameters

    Example:
        raster_import_pdf("C:/plans/floor1.pdf", page=1, scale=1.0)
    """
    if not file_path or not file_path.strip():
        return error_result(ErrorCode.INVALID_PARAMS, "file_path is required")

    if page < 1:
        return error_result(ErrorCode.INVALID_PARAMS, "page must be >= 1")

    if scale <= 0:
        return error_result(ErrorCode.INVALID_PARAMS, "scale must be positive")

    params = {
        "file_path": file_path.strip(),
        "page": page,
        "insertion_point": [float(insertion_point_x), float(insertion_point_y)],
        "scale": float(scale),
        "rotation": float(rotation),
    }
    if target_layer:
        params["target_layer"] = target_layer.strip()

    try:
        result = await call_autocad_command("raster_import_pdf", params)
        return result
    except SidecarError as e:
        return error_result(e.code, e.message, e.details)
    except Exception as e:
        logger.error("Unexpected error in raster_import_pdf", error=str(e), exc_info=True)
        return error_result(ErrorCode.INTERNAL_ERROR, f"Unexpected error: {str(e)}")


# =============================================================================
# Raster Image Attach
# =============================================================================

@mcp.tool()
@with_tool_lock(get_lock())
async def raster_attach_image(
    file_path: str,
    image_name: Optional[str] = None,
    insertion_point_x: float = 0.0,
    insertion_point_y: float = 0.0,
    scale: float = 1.0,
    target_layer: Optional[str] = None
) -> dict:
    """
    Attach a raster image file to the current AutoCAD drawing.

    Creates an image reference that can then be processed with Raster Design
    tools (cleanup, vectorize, OCR). Use this for scanned PDFs or image files.

    Supported formats: TIFF, PNG, JPG, BMP, GIF, PCX, CAL, GP4.

    Args:
        file_path: Absolute path to the raster image file
        image_name: Name for the image in AutoCAD (default: filename)
        insertion_point_x: X coordinate for insertion (default 0)
        insertion_point_y: Y coordinate for insertion (default 0)
        scale: Image scale factor (default 1.0)
        target_layer: Layer to place the image on (optional)

    Returns:
        Image attachment details (handle, dimensions, position)

    Example:
        raster_attach_image("C:/scans/floor1.tiff", scale=1.0)
    """
    if not file_path or not file_path.strip():
        return error_result(ErrorCode.INVALID_PARAMS, "file_path is required")

    if scale <= 0:
        return error_result(ErrorCode.INVALID_PARAMS, "scale must be positive")

    params = {
        "file_path": file_path.strip(),
        "insertion_point": [float(insertion_point_x), float(insertion_point_y)],
        "scale": float(scale),
    }
    if image_name:
        params["image_name"] = image_name.strip()
    if target_layer:
        params["target_layer"] = target_layer.strip()

    try:
        result = await call_autocad_command("raster_attach_image", params)
        return result
    except SidecarError as e:
        return error_result(e.code, e.message, e.details)
    except Exception as e:
        logger.error("Unexpected error in raster_attach_image", error=str(e), exc_info=True)
        return error_result(ErrorCode.INTERNAL_ERROR, f"Unexpected error: {str(e)}")


# =============================================================================
# Raster Cleanup
# =============================================================================

@mcp.tool()
@with_tool_lock(get_lock())
async def raster_cleanup(
    operation: str = "despeckle",
    blob_size: int = 3,
    threshold_value: int = 128,
    brightness: int = 0,
    contrast: int = 0
) -> dict:
    """
    Clean up raster images using AutoCAD Raster Design tools.

    Requires AutoCAD Raster Design to be installed. Applies to all
    raster images in the current drawing.

    This operation is queued — results are applied asynchronously.

    Args:
        operation: Cleanup operation to perform. Options:
            - "despeckle": Remove small noise spots (default)
            - "deskew": Straighten skewed scanned images
            - "negate": Invert black/white colors
            - "mirror_x": Mirror image horizontally
            - "mirror_y": Mirror image vertically
            - "threshold": Adjust binary threshold (0-255)
            - "bias": Adjust brightness and contrast
            - "touchup": Enter interactive cleanup mode
        blob_size: Max speckle size in pixels for despeckle (default 3)
        threshold_value: Threshold value 0-255 for threshold operation (default 128)
        brightness: Brightness adjustment for bias operation (-100 to 100)
        contrast: Contrast adjustment for bias operation (-100 to 100)

    Returns:
        Queued operation status

    Example:
        raster_cleanup("despeckle", blob_size=5)
    """
    valid_ops = {"despeckle", "deskew", "negate", "mirror_x", "mirror_y", "threshold", "bias", "touchup"}
    if operation not in valid_ops:
        return error_result(
            ErrorCode.INVALID_PARAMS,
            f"Invalid operation: {operation}. Valid: {', '.join(sorted(valid_ops))}"
        )

    params = {
        "operation": operation,
        "blob_size": blob_size,
        "threshold_value": threshold_value,
        "brightness": brightness,
        "contrast": contrast,
    }

    try:
        result = await call_autocad_command("raster_cleanup", params)
        return result
    except SidecarError as e:
        return error_result(e.code, e.message, e.details)
    except Exception as e:
        logger.error("Unexpected error in raster_cleanup", error=str(e), exc_info=True)
        return error_result(ErrorCode.INTERNAL_ERROR, f"Unexpected error: {str(e)}")


# =============================================================================
# Vectorize
# =============================================================================

@mcp.tool()
@with_tool_lock(get_lock())
async def raster_vectorize(
    method: str = "auto",
    target_layer: Optional[str] = None,
    detect_polygons: bool = True,
    detect_arcs: bool = True,
    gap_tolerance: float = 0.5
) -> dict:
    """
    Vectorize raster images to AutoCAD vector entities using Raster Design.

    Converts raster lines, arcs, and shapes into native AutoCAD entities
    (lines, arcs, circles, polylines). Requires AutoCAD Raster Design.

    This operation is queued — use raster_get_entity_count to verify results.

    Args:
        method: Vectorization method. Options:
            - "auto": Let Raster Design choose the best method (default)
            - "outline": Trace outer edges of raster lines
            - "centerline": Find centerlines of raster lines (best for drawings)
            - "contour": Contour-based vectorization
        target_layer: Layer to place vectorized entities on (optional)
        detect_polygons: Detect and create closed polygons (default True)
        detect_arcs: Detect and create arcs/circles (default True)
        gap_tolerance: Gap tolerance for closing lines, in drawing units (default 0.5)

    Returns:
        Queued operation status with vectorization parameters

    Example:
        raster_vectorize("centerline", target_layer="Vectorized")
    """
    valid_methods = {"auto", "outline", "centerline", "contour"}
    if method not in valid_methods:
        return error_result(
            ErrorCode.INVALID_PARAMS,
            f"Invalid method: {method}. Valid: {', '.join(sorted(valid_methods))}"
        )

    params = {
        "method": method,
        "detect_polygons": detect_polygons,
        "detect_arcs": detect_arcs,
        "gap_tolerance": float(gap_tolerance),
    }
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
# OCR Text Extraction
# =============================================================================

@mcp.tool()
@with_tool_lock(get_lock())
async def raster_ocr_extract(
    target_layer: Optional[str] = None,
    text_height: float = 0.0,
    language: str = "eng"
) -> dict:
    """
    Extract text from raster images using Raster Design OCR.

    Recognizes text in raster images and converts to AutoCAD TEXT/MTEXT entities.
    Requires AutoCAD Raster Design to be installed.

    This operation is queued — use raster_get_entity_count to verify results.

    Args:
        target_layer: Layer to place extracted text entities on (optional)
        text_height: Text height in drawing units (0 = auto-detect)
        language: OCR language code (default "eng" for English)

    Returns:
        Queued operation status

    Example:
        raster_ocr_extract(target_layer="OCR_Text")
    """
    params = {
        "text_height": float(text_height),
        "language": language,
    }
    if target_layer:
        params["target_layer"] = target_layer.strip()

    try:
        result = await call_autocad_command("raster_ocr", params)
        return result
    except SidecarError as e:
        return error_result(e.code, e.message, e.details)
    except Exception as e:
        logger.error("Unexpected error in raster_ocr_extract", error=str(e), exc_info=True)
        return error_result(ErrorCode.INTERNAL_ERROR, f"Unexpected error: {str(e)}")


# =============================================================================
# Status & Query
# =============================================================================

@mcp.tool()
async def raster_get_status() -> dict:
    """
    Get information about all raster images in the current AutoCAD drawing.

    Returns image names, dimensions, file paths, resolution, and load status.
    Also checks if AutoCAD Raster Design is available.

    Returns:
        List of raster images with their properties, and Raster Design availability
    """
    try:
        result = await call_autocad_command("raster_get_status")
        return result
    except SidecarError as e:
        return error_result(e.code, e.message, e.details)
    except Exception as e:
        logger.error("Unexpected error in raster_get_status", error=str(e), exc_info=True)
        return error_result(ErrorCode.INTERNAL_ERROR, f"Unexpected error: {str(e)}")


@mcp.tool()
async def raster_get_entity_count(
    layer: Optional[str] = None
) -> dict:
    """
    Count entities in the drawing, optionally filtered by layer.

    Useful for checking results after PDF import or vectorization operations.
    Returns total count and breakdown by entity type and layer.

    Args:
        layer: Filter by layer name (optional, counts all if omitted)

    Returns:
        Entity counts (total, by type, by layer)

    Example:
        raster_get_entity_count("Vectorized")
    """
    params = {}
    if layer:
        params["layer"] = layer.strip()

    try:
        result = await call_autocad_command("raster_get_entity_count", params)
        return result
    except SidecarError as e:
        return error_result(e.code, e.message, e.details)
    except Exception as e:
        logger.error("Unexpected error in raster_get_entity_count", error=str(e), exc_info=True)
        return error_result(ErrorCode.INTERNAL_ERROR, f"Unexpected error: {str(e)}")


# =============================================================================
# Fade Raster Image
# =============================================================================

@mcp.tool()
@with_tool_lock(get_lock())
async def raster_fade_image(
    fade_percent: int = 70
) -> dict:
    """
    Fade raster images in the drawing to reduce opacity.

    After vectorization, use this to keep the original raster image as a
    low-opacity background reference while the vector entities are prominent.

    Args:
        fade_percent: Fade level 0-100 (0 = opaque, 100 = fully transparent, default 70)

    Returns:
        Fade operation result

    Example:
        raster_fade_image(70)
    """
    if fade_percent < 0 or fade_percent > 100:
        return error_result(ErrorCode.INVALID_PARAMS, "fade_percent must be 0-100")

    try:
        result = await call_autocad_command("raster_fade_image", {
            "fade_percent": fade_percent,
        })
        return result
    except SidecarError as e:
        return error_result(e.code, e.message, e.details)
    except Exception as e:
        logger.error("Unexpected error in raster_fade_image", error=str(e), exc_info=True)
        return error_result(ErrorCode.INTERNAL_ERROR, f"Unexpected error: {str(e)}")


# =============================================================================
# PostgreSQL Storage
# =============================================================================

@mcp.tool()
@with_tool_lock(get_lock())
async def raster_store_vectorized(
    file_path: Optional[str] = None,
    force: bool = True,
) -> dict:
    """
    Extract all entities from the current AutoCAD drawing and store in PostgreSQL.

    Performs full extraction including:
    - Entity metadata (type, layer, color, linetype)
    - Geometry (PostGIS spatial data)
    - Embeddings (pgvector for semantic search)
    - Spatial relationships between entities

    Use after vectorization to persist results for knowledge base referencing.

    Args:
        file_path: Path to DWG file (optional, uses active document if omitted)
        force: Force re-extraction even if unchanged (default True)

    Returns:
        Extraction summary with project_id, entity counts, and breakdown

    Example:
        raster_store_vectorized()
    """
    from aec_agent.mcp.server import get_database_pool, get_sync_manager

    pool = get_database_pool()
    sm = get_sync_manager()

    if not pool:
        return error_result(
            ErrorCode.MISSING_CONFIG,
            "PostgreSQL database not configured. Set DATABASE_URL."
        )

    if not sm:
        return error_result(
            ErrorCode.INTERNAL_ERROR,
            "Sync manager not available."
        )

    try:
        # Get file path from active drawing if not provided
        if not file_path:
            drawing_info = await call_autocad_command("get_drawing_info")
            if drawing_info.get("success") and drawing_info.get("data"):
                file_path = drawing_info["data"].get("file_path", "")
            if not file_path:
                return error_result(
                    ErrorCode.INVALID_PARAMS,
                    "No file_path provided and no active drawing found"
                )

        # Run full sync (extraction + embeddings + relationships)
        result = await sm.trigger_full_sync("autocad", file_path, force=force)

        if not result.success:
            return error_result(
                ErrorCode.INTERNAL_ERROR,
                "Extraction failed",
                "; ".join(result.errors)
            )

        # Get entity breakdown from the drawing
        count_result = await call_autocad_command("raster_get_entity_count")
        type_breakdown = {}
        layer_breakdown = {}
        if count_result.get("success") and count_result.get("data"):
            type_breakdown = count_result["data"].get("by_type", {})
            layer_breakdown = count_result["data"].get("by_layer", {})

        return success_result(
            data={
                "project_id": str(result.project_id),
                "elements_extracted": result.elements_extracted,
                "elements_updated": result.elements_updated,
                "embeddings_generated": result.embeddings_generated,
                "relationships_computed": result.relationships_computed,
                "duration_ms": result.duration_ms,
                "by_entity_type": type_breakdown,
                "by_layer": layer_breakdown,
            },
            message=f"Stored {result.elements_extracted} entities in PostgreSQL"
        )

    except SidecarError as e:
        return error_result(e.code, e.message, e.details)
    except Exception as e:
        logger.error("Failed to store vectorized entities", error=str(e), exc_info=True)
        return error_result(ErrorCode.INTERNAL_ERROR, f"Storage failed: {str(e)}")


# =============================================================================
# Complete PDF-to-Vector Pipeline
# =============================================================================

@mcp.tool()
@with_tool_lock(get_lock())
async def raster_pdf_to_vector_pipeline(
    file_path: str,
    page: int = 1,
    scale: float = 1.0,
    mode: str = "auto",
    vectorize_method: str = "centerline",
    target_layer: Optional[str] = None,
    skip_ocr: bool = False,
    fade_percent: int = 70,
    store_in_db: bool = True,
) -> dict:
    """
    Complete PDF-to-DWG pipeline with PostgreSQL storage.

    Orchestrates the full Raster Design workflow:
    1. Auto-detects PDF type (vector vs scanned)
    2. For vector PDFs: imports directly via PDFIMPORT
    3. For scanned PDFs: attaches image, converts to bitonal, despeckles,
       deskews, vectorizes centerlines, runs OCR text extraction
    4. Fades original raster image for background reference
    5. Extracts all entities and stores in PostgreSQL with geometry and embeddings

    Each async step blocks until the previous one completes via OnIdle barrier.

    Args:
        file_path: Absolute path to the PDF file
        page: PDF page to import (default 1)
        scale: Import scale factor (default 1.0)
        mode: Detection mode — "auto", "vector", or "scanned" (default "auto")
        vectorize_method: Vectorization method — "centerline", "outline", "auto" (default "centerline")
        target_layer: Layer for vectorized entities (optional)
        skip_ocr: Skip OCR text extraction (default False)
        fade_percent: Raster fade percentage 0-100 (default 70)
        store_in_db: Store results in PostgreSQL (default True)

    Returns:
        Pipeline results with step details, entity counts, and PostgreSQL project info

    Example:
        raster_pdf_to_vector_pipeline("C:/plans/floor1.pdf", mode="auto", store_in_db=True)
    """
    if not file_path or not file_path.strip():
        return error_result(ErrorCode.INVALID_PARAMS, "file_path is required")

    if mode not in ("auto", "vector", "scanned"):
        return error_result(ErrorCode.INVALID_PARAMS, "mode must be 'auto', 'vector', or 'scanned'")

    if vectorize_method not in ("auto", "outline", "centerline", "contour"):
        return error_result(ErrorCode.INVALID_PARAMS, "Invalid vectorize_method")

    steps_completed = []
    step_errors = []

    try:
        # Step 1: Record baseline entity count
        baseline = await call_autocad_command("raster_get_entity_count")
        baseline_count = 0
        if baseline.get("success") and baseline.get("data"):
            baseline_count = baseline["data"].get("total", 0)
        steps_completed.append({"step": "baseline_count", "count": baseline_count})

        # Step 2: Determine PDF type
        detected_mode = mode
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

        if detected_mode == "scanned":
            # ---- Scanned PDF: Raster Design pipeline ----

            # Attach PDF as raster image
            attach_params = {
                "file_path": file_path.strip(),
                "insertion_point": [0.0, 0.0],
                "scale": float(scale),
            }
            if target_layer:
                attach_params["target_layer"] = target_layer.strip()

            attach_result = await call_autocad_command("raster_attach_image", attach_params)
            steps_completed.append({
                "step": "attach_image",
                "success": attach_result.get("success", False),
            })

            # Threshold: convert to bitonal (black & white)
            threshold_result = await call_autocad_command("raster_cleanup", {
                "operation": "threshold",
                "threshold_value": 128,
            })
            steps_completed.append({
                "step": "threshold_bitonal",
                "success": threshold_result.get("success", False),
            })

            # Barrier: wait for threshold to complete
            await call_autocad_command("raster_get_entity_count")

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

            # Vectorize: trace lines from raster
            vectorize_params = {
                "method": vectorize_method,
                "detect_polygons": True,
                "detect_arcs": True,
                "gap_tolerance": 0.5,
            }
            if target_layer:
                vectorize_params["target_layer"] = target_layer.strip()

            vectorize_result = await call_autocad_command("raster_vectorize", vectorize_params)
            steps_completed.append({
                "step": "vectorize",
                "success": vectorize_result.get("success", False),
                "method": vectorize_method,
            })

            # Barrier: get post-vectorize count
            post_vectorize = await call_autocad_command("raster_get_entity_count")
            post_vectorize_count = 0
            if post_vectorize.get("success") and post_vectorize.get("data"):
                post_vectorize_count = post_vectorize["data"].get("total", 0)
            steps_completed.append({
                "step": "post_vectorize_count",
                "count": post_vectorize_count,
                "new_entities": post_vectorize_count - baseline_count,
            })

            # OCR: extract text from raster
            if not skip_ocr:
                ocr_params = {"language": "eng"}
                if target_layer:
                    ocr_params["target_layer"] = target_layer.strip()

                ocr_result = await call_autocad_command("raster_ocr", ocr_params)
                steps_completed.append({
                    "step": "ocr",
                    "success": ocr_result.get("success", False),
                })

                # Barrier
                await call_autocad_command("raster_get_entity_count")

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
