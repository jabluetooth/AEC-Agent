"""
MCP tools for AutoCAD Raster Design integration — image cleanup & filtering.

Split from the former monolithic raster_design.py. Contains:
- raster_cleanup: despeckle/deskew/negate/mirror/threshold/bias/touchup
- raster_fade_image: fade raster images to reduce opacity
- raster_process_image: bitonal image filtering (ibfilter)
"""


import structlog

from aec_agent.mcp.concurrency import with_tool_lock
from aec_agent.mcp.server import get_lock, mcp
from aec_agent.mcp.sidecar_client import SidecarError, call_autocad_command

from .base import ErrorCode, error_result, success_result
from .pdf_converter import convert_pdf_to_bitonal_tiff

logger = structlog.get_logger(__name__)


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
