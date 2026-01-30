"""
MCP tools for AutoCAD Raster Design integration.

Provides PDF-to-DWG conversion pipeline:
1. Convert PDF to bitonal TIFF (Python-side, via PyMuPDF + Pillow)
2. Import PDF into AutoCAD (vector PDFs via -PDFIMPORT)
3. Attach raster images (scanned PDFs converted to TIFF / raw images)
4. Cleanup raster images (despeckle, deskew, threshold)
5. Vectorize raster to AutoCAD entities (Python-side OpenCV detection)
6. Create entities in AutoCAD via draw commands

Vectorization uses Python-side OpenCV (HoughLinesP, HoughCircles,
findContours) rather than Raster Design VTools (vline, vpline, etc.),
because VTools are interactive and cannot be automated via
SendStringToExecute.

Async commands (import_pdf, cleanup) use SendStringToExecute on the sidecar
and return a "queued" status immediately. Use raster_get_status /
raster_get_entity_count to check results after execution.

IMPORTANT: AutoCAD Raster Design cannot attach PDF files directly. PDFs must
be converted to a supported raster format (bitonal TIFF) first using
raster_convert_pdf or the raster_pdf_to_vector_pipeline.
"""

from typing import Optional, List

from aec_agent.mcp.server import mcp, get_lock
from aec_agent.mcp.concurrency import with_tool_lock
from aec_agent.mcp.sidecar_client import call_autocad_command, SidecarError
from .base import success_result, error_result, ErrorCode
from .pdf_converter import convert_pdf_to_bitonal_tiff

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
# PDF to Bitonal TIFF Conversion (Python-side)
# =============================================================================

@mcp.tool()
async def raster_convert_pdf(
    file_path: str,
    page: int = 1,
    dpi: int = 300,
    threshold: int = 128,
    output_dir: Optional[str] = None,
) -> dict:
    """
    Convert a PDF file to a bitonal (1-bit black & white) TIFF image.

    AutoCAD Raster Design cannot attach PDF files directly. This tool converts
    a PDF page to a bitonal TIFF that can be attached via raster_attach_image
    and processed with Raster Design tools (cleanup, vectorize, OCR).

    The conversion runs on the Python server (no AutoCAD needed). Use the
    returned file path with raster_attach_image to load it into AutoCAD.

    Args:
        file_path: Absolute path to the PDF file
        page: PDF page number to convert (1-based, default 1)
        dpi: Render resolution (default 300 — good for construction drawings)
        threshold: Grayscale-to-bitonal threshold 0-255 (default 128).
                   Pixels darker than threshold become black.
        output_dir: Directory for output TIFF (default: same folder as PDF)

    Returns:
        Conversion result with output TIFF path, dimensions, and file size

    Example:
        result = raster_convert_pdf("C:/plans/floor1.pdf", page=1, dpi=300)
        tiff_path = result["data"]["tiff_path"]
        raster_attach_image(tiff_path, scale=1.0)
    """
    if not file_path or not file_path.strip():
        return error_result(ErrorCode.INVALID_PARAMS, "file_path is required")

    try:
        import os

        tiff_path = convert_pdf_to_bitonal_tiff(
            pdf_path=file_path.strip(),
            page=page,
            dpi=dpi,
            threshold=threshold,
            output_dir=output_dir,
        )

        file_size = os.path.getsize(tiff_path)

        return success_result(
            data={
                "tiff_path": tiff_path,
                "source_pdf": file_path.strip(),
                "page": page,
                "dpi": dpi,
                "threshold": threshold,
                "file_size_bytes": file_size,
            },
            message=f"Converted PDF page {page} to bitonal TIFF: {tiff_path}",
        )

    except FileNotFoundError as e:
        return error_result(ErrorCode.INVALID_PARAMS, str(e))
    except ValueError as e:
        return error_result(ErrorCode.INVALID_PARAMS, str(e))
    except RuntimeError as e:
        return error_result(ErrorCode.INTERNAL_ERROR, str(e))
    except Exception as e:
        logger.error("Unexpected error in raster_convert_pdf", error=str(e), exc_info=True)
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
# VTools — Vectorize Raster Entities (vline, vpline, varc, vcircle, vrect)
# =============================================================================

@mcp.tool()
@with_tool_lock(get_lock())
async def raster_vectorize(
    tool: str = "vpline",
    method: str = "1p",
    points: Optional[List[List[float]]] = None,
    target_layer: Optional[str] = None,
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
# Process Image — ibfilter (bitonal image filtering)
# =============================================================================

@mcp.tool()
@with_tool_lock(get_lock())
async def raster_process_image(
    filter_type: str = "skeletonize",
) -> dict:
    """
    Apply bitonal image filter using AutoCAD Raster Design ibfilter.

    Prepares bitonal raster images for vectorization by processing pixel data.
    "Skeletonize" reduces all lines to 1px width, which significantly improves
    VTool detection accuracy for vline/vpline/varc/vcircle.

    Requires AutoCAD Raster Design and a bitonal raster image.
    This operation is queued.

    Args:
        filter_type: Filter to apply. Options:
            - "smooth": Smooth jagged edges on lines
            - "thin": Reduce line width by one pixel per side
            - "thicken": Increase line width by one pixel per side
            - "separate": Separate touching lines at junctions
            - "skeletonize": Reduce all lines to 1px centerlines (default, best for vectorization)

    Returns:
        Queued operation status

    Example:
        raster_process_image("skeletonize")
    """
    valid_filters = {"smooth", "thin", "thicken", "separate", "skeletonize"}
    if filter_type not in valid_filters:
        return error_result(
            ErrorCode.INVALID_PARAMS,
            f"Invalid filter_type: {filter_type}. Valid: {', '.join(sorted(valid_filters))}"
        )

    try:
        result = await call_autocad_command("raster_process_image", {
            "filter_type": filter_type,
        })
        return result
    except SidecarError as e:
        return error_result(e.code, e.message, e.details)
    except Exception as e:
        logger.error("Unexpected error in raster_process_image", error=str(e), exc_info=True)
        return error_result(ErrorCode.INTERNAL_ERROR, f"Unexpected error: {str(e)}")


# =============================================================================
# REM Primitives — isline, isarc, iscircle, issmart
# =============================================================================

@mcp.tool()
@with_tool_lock(get_lock())
async def raster_create_primitive(
    primitive_type: str = "smart",
    point_x: Optional[float] = None,
    point_y: Optional[float] = None,
) -> dict:
    """
    Create a REM (Raster Entity Manipulation) primitive from raster data.

    Detects a raster entity (line, arc, or circle) and creates an overlay
    primitive that represents it as a vector object. Use "smart" to
    auto-detect the best primitive type.

    Provide coordinates to target a specific raster entity. If omitted,
    AutoCAD prompts for interactive picking.

    Args:
        primitive_type: Primitive type. Options:
            - "smart": Auto-detect best type (issmart, default)
            - "line": Line primitive (isline)
            - "arc": Arc primitive (isarc)
            - "circle": Circle primitive (iscircle)
        point_x: X coordinate of raster entity to convert (optional)
        point_y: Y coordinate of raster entity to convert (optional)

    Returns:
        Queued operation status

    Example:
        raster_create_primitive("smart", point_x=100.0, point_y=200.0)
    """
    valid_types = {"smart", "line", "arc", "circle"}
    if primitive_type not in valid_types:
        return error_result(
            ErrorCode.INVALID_PARAMS,
            f"Invalid primitive_type: {primitive_type}. Valid: {', '.join(sorted(valid_types))}"
        )

    params = {"primitive_type": primitive_type}
    if point_x is not None and point_y is not None:
        params["point"] = [float(point_x), float(point_y)]

    try:
        result = await call_autocad_command("raster_create_primitive", params)
        return result
    except SidecarError as e:
        return error_result(e.code, e.message, e.details)
    except Exception as e:
        logger.error("Unexpected error in raster_create_primitive", error=str(e), exc_info=True)
        return error_result(ErrorCode.INTERNAL_ERROR, f"Unexpected error: {str(e)}")


# =============================================================================
# Select Raster Entities — isebrcon, isebrsmart
# =============================================================================

@mcp.tool()
@with_tool_lock(get_lock())
async def raster_select_entities(
    method: str = "smart",
    corner1_x: Optional[float] = None,
    corner1_y: Optional[float] = None,
    corner2_x: Optional[float] = None,
    corner2_y: Optional[float] = None,
) -> dict:
    """
    Select raster entities within a rectangular region.

    Uses Raster Design's bitonal region selection to identify and select
    complete raster entities. Selected entities become REM objects that
    can be converted to primitives with raster_create_primitive.

    Args:
        method: Selection method. Options:
            - "smart": Smart entity detection (isebrsmart, default)
            - "crossing": Crossing rectangle selection (isebrcon)
        corner1_x: First corner X coordinate (optional)
        corner1_y: First corner Y coordinate (optional)
        corner2_x: Opposite corner X coordinate (optional)
        corner2_y: Opposite corner Y coordinate (optional)

    Returns:
        Queued operation status

    Example:
        raster_select_entities("smart", 0, 0, 1000, 1000)
    """
    valid_methods = {"smart", "crossing"}
    if method not in valid_methods:
        return error_result(
            ErrorCode.INVALID_PARAMS,
            f"Invalid method: {method}. Valid: {', '.join(sorted(valid_methods))}"
        )

    params = {"method": method}
    if (corner1_x is not None and corner1_y is not None and
            corner2_x is not None and corner2_y is not None):
        params["corner1"] = [float(corner1_x), float(corner1_y)]
        params["corner2"] = [float(corner2_x), float(corner2_y)]

    try:
        result = await call_autocad_command("raster_select_entities", params)
        return result
    except SidecarError as e:
        return error_result(e.code, e.message, e.details)
    except Exception as e:
        logger.error("Unexpected error in raster_select_entities", error=str(e), exc_info=True)
        return error_result(ErrorCode.INTERNAL_ERROR, f"Unexpected error: {str(e)}")


# =============================================================================
# Follower VTools — vfpline, vfcontour, vf3dpoly
# =============================================================================

@mcp.tool()
@with_tool_lock(get_lock())
async def raster_follower(
    follower_type: str = "polyline",
    start_point_x: Optional[float] = None,
    start_point_y: Optional[float] = None,
    target_layer: Optional[str] = None,
) -> dict:
    """
    Semi-automatic follower for tracing raster lines and contours.

    The follower traces along raster data from a starting point,
    automatically following the raster path and creating vector entities.
    Useful for complex geometry like contour lines and winding paths.

    Provide a starting point to begin tracing. If omitted, AutoCAD
    prompts for interactive picking.

    Args:
        follower_type: Follower type. Options:
            - "polyline": Follow raster polyline path (vfpline, default)
            - "contour": Follow raster contour line (vfcontour)
            - "3dpoly": Create 3D polyline from raster (vf3dpoly)
        start_point_x: X coordinate to start following from (optional)
        start_point_y: Y coordinate to start following from (optional)
        target_layer: Layer to place followed entities on (optional)

    Returns:
        Queued operation status

    Example:
        raster_follower("polyline", start_point_x=50.0, start_point_y=100.0)
    """
    valid_types = {"polyline", "contour", "3dpoly"}
    if follower_type not in valid_types:
        return error_result(
            ErrorCode.INVALID_PARAMS,
            f"Invalid follower_type: {follower_type}. Valid: {', '.join(sorted(valid_types))}"
        )

    params = {"follower_type": follower_type}
    if start_point_x is not None and start_point_y is not None:
        params["start_point"] = [float(start_point_x), float(start_point_y)]
    if target_layer:
        params["target_layer"] = target_layer.strip()

    try:
        result = await call_autocad_command("raster_follower", params)
        return result
    except SidecarError as e:
        return error_result(e.code, e.message, e.details)
    except Exception as e:
        logger.error("Unexpected error in raster_follower", error=str(e), exc_info=True)
        return error_result(ErrorCode.INTERNAL_ERROR, f"Unexpected error: {str(e)}")


# =============================================================================
# Text Recognition — irectext
# =============================================================================

@mcp.tool()
@with_tool_lock(get_lock())
async def raster_recognize_text(
    target_layer: Optional[str] = None,
) -> dict:
    """
    Recognize and convert raster text to AutoCAD TEXT entities.

    Uses Raster Design's irectext command to detect text in bitonal
    raster images and create native AutoCAD text entities.

    Configure recognition settings first with AutoCAD's irecsetup if needed.

    Args:
        target_layer: Layer to place recognized text entities on (optional)

    Returns:
        Queued operation status

    Example:
        raster_recognize_text(target_layer="OCR_Text")
    """
    params = {}
    if target_layer:
        params["target_layer"] = target_layer.strip()

    try:
        result = await call_autocad_command("raster_recognize_text", params)
        return result
    except SidecarError as e:
        return error_result(e.code, e.message, e.details)
    except Exception as e:
        logger.error("Unexpected error in raster_recognize_text", error=str(e), exc_info=True)
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
    target_layer: Optional[str] = None,
    min_line_length: int = 80,
    max_line_gap: int = 10,
    hough_threshold: int = 150,
    min_circle_radius: int = 20,
    max_circle_radius: int = 500,
    hough_circles_dp: float = 1.2,
    hough_circles_param1: float = 200.0,
    hough_circles_param2: float = 100.0,
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
    mask_detected_lines: bool = False,
    mask_thickness: int = 5,
    # Skeletonization & Topology
    skeletonize: bool = False,
    topology_cleanup: bool = False,
    snap_tolerance: float = 5.0,
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

    Args:
        image_path: Absolute path to an already-processed bitonal TIFF image
        dpi: Image resolution in DPI (default 300)
        scale: Coordinate scale factor — must match raster_attach_image scale (default 1.0)
        target_layer: Layer for created entities (optional)
        min_line_length: Min line length in pixels (default 80)
        max_line_gap: Max gap to merge line segments in pixels (default 10)
        hough_threshold: Line detection sensitivity — lower = more lines (default 150)
        min_circle_radius: Min circle radius in pixels (default 20)
        max_circle_radius: Max circle radius in pixels, 0=unlimited (default 500)
        hough_circles_dp: Accumulator resolution ratio — lower = finer (default 1.2)
        hough_circles_param1: Canny high threshold inside HoughCircles (default 200)
        hough_circles_param2: Circle center accumulator threshold — higher = fewer
                              but more confident circles (default 100)
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
        mask_detected_lines: Erase detected line pixels before contour pass (default False)
        mask_thickness: Pixel thickness of the erasure mask (default 5)
        skeletonize: Reduce thick lines to 1px centerlines before detection (default False)
        topology_cleanup: Merge degree-2 breaks and snap dangling endpoints (default False)
        snap_tolerance: Max distance in drawing units to snap endpoints (default 5.0)

    Returns:
        Vectorization summary with entity counts and creation results

    Example:
        # Prefer raster_pdf_to_vector_pipeline instead of calling this directly
        raster_auto_vectorize("C:/plans/floor1_page1_bitonal.tif", dpi=300, scale=1.0)
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
        )

        created = {"lines": 0, "circles": 0, "arcs": 0, "ellipses": 0, "polylines": 0, "errors": 0}

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

        total_created = (
            created["lines"] + created["circles"] + created["arcs"]
            + created["ellipses"] + created["polylines"]
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
                },
                "created": created,
                "total_entities_created": total_created,
                "target_layer": target_layer,
            },
            message=(
                f"Auto-vectorized: {total_created} entities created "
                f"({created['lines']} lines, {created['circles']} circles, "
                f"{created['arcs']} arcs, {created['ellipses']} ellipses, "
                f"{created['polylines']} polylines)"
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
    dpi: int = 300,
    scale: float = 1.0,
    mode: str = "auto",
    target_layer: Optional[str] = None,
    fade_percent: int = 70,
    store_in_db: bool = True,
    min_line_length: int = 80,
    max_line_gap: int = 10,
    hough_threshold: int = 150,
    min_circle_radius: int = 20,
    max_circle_radius: int = 500,
    hough_circles_dp: float = 1.2,
    hough_circles_param1: float = 200.0,
    hough_circles_param2: float = 100.0,
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
    mask_detected_lines: bool = False,
    mask_thickness: int = 5,
    # Skeletonization & Topology
    skeletonize: bool = False,
    topology_cleanup: bool = False,
    snap_tolerance: float = 5.0,
) -> dict:
    """
    PRIMARY TOOL for converting any PDF or image file to AutoCAD vector
    entities.  Accepts PDF, TIFF, PNG, JPG, and BMP files.  Use this tool
    whenever the user wants to vectorize, trace, or convert a file to CAD
    entities.  Do NOT call raster_auto_vectorize, raster_convert_pdf,
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
       e. Detects features via OpenCV (lines, circles, arcs, ellipses, polylines)
       f. Creates AutoCAD entities via draw commands
    4. Fades original raster image for background reference
    5. Extracts all entities and stores in PostgreSQL with geometry and embeddings

    Vectorization uses Python-side OpenCV (HoughLinesP, HoughCircles,
    findContours + fitEllipse) instead of Raster Design VTools, which are
    interactive and cannot be automated via SendStringToExecute.

    Args:
        file_path: Absolute path to the PDF file
        page: PDF page to import (default 1)
        dpi: Render resolution for PDF-to-TIFF conversion (default 300)
        scale: Import scale factor (default 1.0)
        mode: Detection mode — "auto", "vector", or "scanned" (default "auto")
        target_layer: Layer for vectorized entities (optional)
        fade_percent: Raster fade percentage 0-100 (default 70)
        store_in_db: Store results in PostgreSQL (default True)
        min_line_length: Min line length in pixels for detection (default 80)
        max_line_gap: Max gap to merge line segments in pixels (default 10)
        hough_threshold: Line detection sensitivity — lower = more lines (default 150)
        min_circle_radius: Min circle radius in pixels (default 20)
        max_circle_radius: Max circle radius in pixels, 0=unlimited (default 500)
        hough_circles_dp: Accumulator resolution ratio — lower = finer (default 1.2)
        hough_circles_param1: Canny high threshold inside HoughCircles (default 200)
        hough_circles_param2: Circle center accumulator threshold — higher = fewer
                              but more confident circles (default 100)
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
        mask_detected_lines: Erase detected line pixels before contour pass (default False)
        mask_thickness: Pixel thickness of the erasure mask (default 5)
        skeletonize: Reduce thick lines to 1px centerlines before detection (default False)
        topology_cleanup: Merge degree-2 breaks and snap dangling endpoints (default False)
        snap_tolerance: Max distance in drawing units to snap endpoints (default 5.0)

    Returns:
        Pipeline results with step details, entity counts, and PostgreSQL project info

    Example:
        raster_pdf_to_vector_pipeline("C:/plans/floor1.pdf", mode="auto", store_in_db=True)
        raster_pdf_to_vector_pipeline("C:/scans/bracket.png", mode="scanned")
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
                )
                steps_completed.append({
                    "step": "opencv_detect_features",
                    "success": True,
                    "lines": len(detection.lines),
                    "circles": len(detection.circles),
                    "arcs": len(detection.arcs),
                    "ellipses": len(detection.ellipses),
                    "polylines": len(detection.polylines),
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
                "ellipses": 0, "polylines": 0, "errors": 0,
            }

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
                    except Exception:
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
                    except Exception:
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
                    except Exception:
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
                    except Exception:
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
                    except Exception:
                        created_counts["errors"] += 1

            total_created = (
                created_counts["lines"] + created_counts["circles"]
                + created_counts["arcs"] + created_counts["ellipses"]
                + created_counts["polylines"]
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
    lines_json: List[dict],
    polylines_json: Optional[List[dict]] = None,
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
        from .topology import (
            build_segment_graph,
            merge_degree2_nodes,
            snap_dangling_endpoints,
            graph_to_vectorization_result,
        )
        from .image_vectorizer import (
            DetectedLine,
            DetectedPolyline,
            VectorizationResult,
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
