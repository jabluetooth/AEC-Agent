"""
MCP tools for AutoCAD Raster Design integration — OCR / text recognition.

Split from the former monolithic raster_design.py. Contains:
- raster_ocr_extract: extract text from raster images using Raster Design OCR
- raster_recognize_text: recognize raster text and convert to TEXT entities (irectext)
"""


import structlog

from aec_agent.mcp.concurrency import with_tool_lock
from aec_agent.mcp.server import get_lock, mcp
from aec_agent.mcp.sidecar_client import SidecarError, call_autocad_command

from .base import ErrorCode, error_result, success_result
from .pdf_converter import convert_pdf_to_bitonal_tiff

logger = structlog.get_logger(__name__)


# =============================================================================
# OCR Text Extraction
# =============================================================================

@mcp.tool()
@with_tool_lock(get_lock())
async def raster_ocr_extract(
    target_layer: str | None = None,
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
# Text Recognition — irectext
# =============================================================================

@mcp.tool()
@with_tool_lock(get_lock())
async def raster_recognize_text(
    target_layer: str | None = None,
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
