"""
MCP tools for AutoCAD Raster Design integration — PDF import & image attach.

Split from the former monolithic raster_design.py. Contains:
- raster_import_pdf: import a PDF into AutoCAD as vector entities (-PDFIMPORT)
- raster_convert_pdf: convert a PDF page to a bitonal TIFF (Python-side)
- raster_attach_image: attach a raster image file to the current drawing
"""


import structlog

from aec_agent.mcp.concurrency import with_tool_lock
from aec_agent.mcp.server import get_lock, mcp
from aec_agent.mcp.sidecar_client import SidecarError, call_autocad_command

from .base import ErrorCode, error_result, success_result
from .pdf_converter import convert_pdf_to_bitonal_tiff

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
    target_layer: str | None = None
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
    output_dir: str | None = None,
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
    image_name: str | None = None,
    insertion_point_x: float = 0.0,
    insertion_point_y: float = 0.0,
    scale: float = 1.0,
    target_layer: str | None = None
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
