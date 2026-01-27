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
