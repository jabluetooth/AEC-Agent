"""
Phase 1: PDF Intake & Rendering

This module renders PDFs to high-quality images WITHOUT destroying information.
Unlike the existing pdf_converter.py which converts to bitonal TIFF, this module
preserves the full quality of the original PDF for Gemini Vision analysis.

Key Principles:
- Keep RGB or high-quality grayscale (NO bitonal conversion)
- Use 300+ DPI for technical drawings
- Save as PNG (lossless compression)
- Preserve anti-aliasing and fine details
"""

from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Optional, Tuple

import numpy as np
import structlog

# Optional imports - gracefully handle if not installed
try:
    import fitz  # PyMuPDF
    HAS_PYMUPDF = True
except ImportError:
    HAS_PYMUPDF = False

try:
    from PIL import Image
    HAS_PIL = True
except ImportError:
    HAS_PIL = False

logger = structlog.get_logger(__name__)

# Constants
MIN_DPI = 72
MAX_DPI = 1200
DEFAULT_DPI = 300
GRAYSCALE_THRESHOLD = 0.01  # Max color variance to consider grayscale


@dataclass
class PDFRenderResult:
    """Result of rendering a PDF page to a high-quality image."""

    # Output image info
    image_path: Path
    width_px: int
    height_px: int
    dpi: int
    color_mode: Literal["RGB", "L"]  # RGB or Grayscale

    # Source PDF info
    original_pdf: Path
    page_number: int  # 0-indexed
    page_count: int

    # Page dimensions in points (1 point = 1/72 inch)
    page_width_pts: float
    page_height_pts: float

    # Computed properties
    @property
    def page_width_inches(self) -> float:
        """Page width in inches."""
        return self.page_width_pts / 72.0

    @property
    def page_height_inches(self) -> float:
        """Page height in inches."""
        return self.page_height_pts / 72.0

    @property
    def pixels_per_inch(self) -> float:
        """Actual pixels per inch (should match DPI)."""
        return self.width_px / self.page_width_inches

    @property
    def aspect_ratio(self) -> float:
        """Width / Height ratio."""
        return self.width_px / self.height_px if self.height_px > 0 else 1.0

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "image_path": str(self.image_path),
            "width_px": self.width_px,
            "height_px": self.height_px,
            "dpi": self.dpi,
            "color_mode": self.color_mode,
            "original_pdf": str(self.original_pdf),
            "page_number": self.page_number,
            "page_count": self.page_count,
            "page_width_pts": self.page_width_pts,
            "page_height_pts": self.page_height_pts,
            "page_width_inches": self.page_width_inches,
            "page_height_inches": self.page_height_inches,
            "pixels_per_inch": self.pixels_per_inch,
            "aspect_ratio": self.aspect_ratio,
        }


@dataclass
class PDFInfo:
    """Information about a PDF file."""

    path: Path
    page_count: int
    pages: list[dict] = field(default_factory=list)  # List of page info dicts

    # Metadata
    title: Optional[str] = None
    author: Optional[str] = None
    subject: Optional[str] = None
    creator: Optional[str] = None

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "path": str(self.path),
            "page_count": self.page_count,
            "pages": self.pages,
            "title": self.title,
            "author": self.author,
            "subject": self.subject,
            "creator": self.creator,
        }


def _check_dependencies() -> None:
    """Check that required dependencies are installed."""
    if not HAS_PYMUPDF:
        raise ImportError(
            "PyMuPDF (fitz) is required for PDF rendering. "
            "Install with: pip install pymupdf"
        )
    if not HAS_PIL:
        raise ImportError(
            "Pillow (PIL) is required for image processing. "
            "Install with: pip install Pillow"
        )


def is_effectively_grayscale(
    image: Image.Image,
    threshold: float = GRAYSCALE_THRESHOLD
) -> bool:
    """
    Check if an RGB image is effectively grayscale.

    Technical drawings are often black & white, so we can save space
    by converting to grayscale while preserving all visual information.

    Args:
        image: PIL Image in RGB mode
        threshold: Maximum average color variance (0-1) to consider grayscale

    Returns:
        True if image is effectively grayscale
    """
    if image.mode != "RGB":
        return image.mode == "L"

    # Convert to numpy for fast computation
    arr = np.array(image)

    # Calculate color variance: how much do R, G, B differ?
    r, g, b = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]

    # Mean absolute difference between channels
    diff_rg = np.abs(r.astype(np.float32) - g.astype(np.float32))
    diff_gb = np.abs(g.astype(np.float32) - b.astype(np.float32))

    avg_diff = (diff_rg.mean() + diff_gb.mean()) / 2.0 / 255.0

    # Return Python bool, not numpy bool
    return bool(avg_diff < threshold)


def get_pdf_info(pdf_path: str | Path) -> PDFInfo:
    """
    Get information about a PDF file without rendering.

    Args:
        pdf_path: Path to PDF file

    Returns:
        PDFInfo object with page count and metadata

    Raises:
        FileNotFoundError: If PDF file doesn't exist
        RuntimeError: If PDF cannot be opened
    """
    _check_dependencies()

    pdf_path = Path(pdf_path)

    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")

    try:
        doc = fitz.open(str(pdf_path))
    except Exception as e:
        raise RuntimeError(f"Failed to open PDF: {e}")

    try:
        # Get metadata
        metadata = doc.metadata or {}

        # Get page information
        pages = []
        for i, page in enumerate(doc):
            rect = page.rect
            pages.append({
                "page_number": i,
                "width_pts": rect.width,
                "height_pts": rect.height,
                "width_inches": rect.width / 72.0,
                "height_inches": rect.height / 72.0,
                "rotation": page.rotation,
            })

        return PDFInfo(
            path=pdf_path,
            page_count=len(doc),
            pages=pages,
            title=metadata.get("title"),
            author=metadata.get("author"),
            subject=metadata.get("subject"),
            creator=metadata.get("creator"),
        )
    finally:
        doc.close()


def render_pdf_page(
    pdf_path: str | Path,
    page_number: int = 0,
    dpi: int = DEFAULT_DPI,
) -> Tuple[Image.Image, dict]:
    """
    Render a single PDF page to a PIL Image.

    This is the low-level rendering function. For most use cases,
    use render_pdf_high_quality() which also saves to disk.

    Args:
        pdf_path: Path to PDF file
        page_number: Page to render (0-indexed)
        dpi: Resolution in dots per inch

    Returns:
        Tuple of (PIL Image, page_info dict)

    Raises:
        FileNotFoundError: If PDF file doesn't exist
        ValueError: If page_number is invalid or DPI out of range
        RuntimeError: If rendering fails
    """
    _check_dependencies()

    pdf_path = Path(pdf_path)

    # Validate inputs
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")

    if not MIN_DPI <= dpi <= MAX_DPI:
        raise ValueError(f"DPI must be between {MIN_DPI} and {MAX_DPI}, got {dpi}")

    try:
        doc = fitz.open(str(pdf_path))
    except Exception as e:
        raise RuntimeError(f"Failed to open PDF: {e}")

    try:
        # Validate page number
        if page_number < 0 or page_number >= len(doc):
            raise ValueError(
                f"Invalid page number {page_number}. "
                f"PDF has {len(doc)} pages (0-{len(doc)-1})"
            )

        page = doc[page_number]
        rect = page.rect

        # Calculate zoom factor for target DPI
        # PDF default is 72 DPI
        zoom = dpi / 72.0
        matrix = fitz.Matrix(zoom, zoom)

        # Render page to pixmap (RGB by default, no alpha)
        pixmap = page.get_pixmap(matrix=matrix, alpha=False)

        # Convert to PIL Image
        image = Image.frombytes(
            "RGB",
            [pixmap.width, pixmap.height],
            pixmap.samples
        )

        page_info = {
            "page_number": page_number,
            "page_count": len(doc),
            "width_pts": rect.width,
            "height_pts": rect.height,
            "width_px": pixmap.width,
            "height_px": pixmap.height,
            "dpi": dpi,
            "rotation": page.rotation,
        }

        return image, page_info

    finally:
        doc.close()


def render_pdf_high_quality(
    pdf_path: str | Path,
    page_number: int = 0,
    dpi: int = DEFAULT_DPI,
    output_dir: str | Path | None = None,
    output_filename: str | None = None,
    convert_grayscale: bool = True,
    grayscale_threshold: float = GRAYSCALE_THRESHOLD,
    enable_super_resolution: bool | None = None,  # Phase B: None = use settings
) -> PDFRenderResult:
    """
    Render a PDF page to a high-quality image file.

    This is the PRIMARY function for Phase 1 of the Gemini-First pipeline.
    It renders the PDF at high quality WITHOUT any lossy preprocessing.

    Key behaviors:
    - Renders at specified DPI (default 300)
    - Keeps RGB unless image is effectively grayscale
    - Saves as PNG (lossless compression)
    - NO bitonal conversion (preserves anti-aliasing, gradients)

    Args:
        pdf_path: Path to PDF file
        page_number: Page to render (0-indexed)
        dpi: Resolution in dots per inch (72-1200, default 300)
        output_dir: Directory for output image (default: temp directory)
        output_filename: Custom filename (default: {pdf_name}_page{N}.png)
        convert_grayscale: If True, convert to grayscale if image has no color
        grayscale_threshold: Color variance threshold for grayscale detection

    Returns:
        PDFRenderResult with image path and metadata

    Raises:
        FileNotFoundError: If PDF file doesn't exist
        ValueError: If parameters are invalid
        RuntimeError: If rendering fails

    Example:
        >>> result = render_pdf_high_quality("drawing.pdf", page_number=0, dpi=300)
        >>> print(f"Rendered to {result.image_path}")
        >>> print(f"Size: {result.width_px}x{result.height_px}")
        >>> print(f"Mode: {result.color_mode}")
    """
    _check_dependencies()

    pdf_path = Path(pdf_path)

    logger.info(
        "rendering_pdf_high_quality",
        pdf_path=str(pdf_path),
        page_number=page_number,
        dpi=dpi,
    )

    # Render the page
    image, page_info = render_pdf_page(pdf_path, page_number, dpi)

    # Optionally convert to grayscale if no color information
    color_mode: Literal["RGB", "L"] = "RGB"
    if convert_grayscale and is_effectively_grayscale(image, grayscale_threshold):
        image = image.convert("L")
        color_mode = "L"
        logger.debug("converted_to_grayscale", reason="no_color_variance")

    # Phase B: Apply super-resolution if enabled and DPI is below threshold
    effective_dpi = dpi
    super_resolution_applied = False

    from aec_agent.config.settings import get_settings
    settings = get_settings()

    # Use parameter if provided, otherwise use settings
    should_upscale = enable_super_resolution
    if should_upscale is None:
        should_upscale = settings.enable_super_resolution

    if should_upscale and dpi < settings.super_resolution_min_dpi_threshold:
        try:
            from .super_resolution import (
                RealESRGANUpscaler,
                is_super_resolution_available,
            )

            if is_super_resolution_available():
                import numpy as np

                upscaler = RealESRGANUpscaler.get_instance()
                if upscaler.is_available:
                    # Convert PIL Image to numpy array
                    image_array = np.array(image)

                    # Upscale
                    result = upscaler.upscale(image_array)

                    if result.scale_factor > 1:
                        # Convert back to PIL Image
                        from PIL import Image as PILImage
                        image = PILImage.fromarray(result.image)
                        effective_dpi = dpi * result.scale_factor
                        super_resolution_applied = True

                        logger.info(
                            "super_resolution_applied",
                            original_dpi=dpi,
                            effective_dpi=effective_dpi,
                            scale=result.scale_factor,
                            device=result.device_used,
                            time_ms=round(result.processing_time_ms, 2),
                        )
                else:
                    logger.debug("super_resolution_not_available", reason="model_not_loaded")
            else:
                logger.debug("super_resolution_not_available", reason="library_not_installed")
        except Exception as e:
            logger.warning("super_resolution_failed", error=str(e))

    # Determine output path
    if output_dir is None:
        output_dir = Path(tempfile.gettempdir()) / "aec_agent" / "gemini_first"
    else:
        output_dir = Path(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    if output_filename is None:
        output_filename = f"{pdf_path.stem}_page{page_number}.png"

    output_path = output_dir / output_filename

    # Save as PNG (lossless)
    image.save(output_path, "PNG", optimize=True)

    logger.info(
        "pdf_rendered_successfully",
        output_path=str(output_path),
        width_px=page_info["width_px"],
        height_px=page_info["height_px"],
        color_mode=color_mode,
        file_size_kb=output_path.stat().st_size / 1024,
    )

    # Get actual image dimensions after potential super-resolution
    actual_width, actual_height = image.size

    return PDFRenderResult(
        image_path=output_path,
        width_px=actual_width,
        height_px=actual_height,
        dpi=effective_dpi,  # Use effective DPI (may be scaled by super-resolution)
        color_mode=color_mode,
        original_pdf=pdf_path,
        page_number=page_number,
        page_count=page_info["page_count"],
        page_width_pts=page_info["width_pts"],
        page_height_pts=page_info["height_pts"],
    )


async def render_pdf_high_quality_async(
    pdf_path: str | Path,
    page_number: int = 0,
    dpi: int = DEFAULT_DPI,
    output_dir: str | Path | None = None,
    output_filename: str | None = None,
    convert_grayscale: bool = True,
    grayscale_threshold: float = GRAYSCALE_THRESHOLD,
) -> PDFRenderResult:
    """
    Async version of render_pdf_high_quality.

    Runs the CPU-bound rendering in a thread pool to avoid blocking.
    """
    import asyncio

    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(
        None,
        lambda: render_pdf_high_quality(
            pdf_path=pdf_path,
            page_number=page_number,
            dpi=dpi,
            output_dir=output_dir,
            output_filename=output_filename,
            convert_grayscale=convert_grayscale,
            grayscale_threshold=grayscale_threshold,
        )
    )


def render_all_pages(
    pdf_path: str | Path,
    dpi: int = DEFAULT_DPI,
    output_dir: str | Path | None = None,
    convert_grayscale: bool = True,
) -> list[PDFRenderResult]:
    """
    Render all pages of a PDF to high-quality images.

    Args:
        pdf_path: Path to PDF file
        dpi: Resolution in dots per inch
        output_dir: Directory for output images
        convert_grayscale: If True, convert to grayscale if no color

    Returns:
        List of PDFRenderResult, one per page
    """
    info = get_pdf_info(pdf_path)

    results = []
    for page_num in range(info.page_count):
        result = render_pdf_high_quality(
            pdf_path=pdf_path,
            page_number=page_num,
            dpi=dpi,
            output_dir=output_dir,
            convert_grayscale=convert_grayscale,
        )
        results.append(result)

    return results


# Comparison with existing bitonal converter
def compare_with_bitonal(
    pdf_path: str | Path,
    page_number: int = 0,
    dpi: int = DEFAULT_DPI,
) -> dict:
    """
    Compare high-quality rendering with bitonal conversion.

    This function demonstrates why Gemini-First (high-quality) is better
    than the existing bitonal approach.

    Returns dict with comparison metrics.
    """
    _check_dependencies()

    # Render high quality
    hq_result = render_pdf_high_quality(pdf_path, page_number, dpi)
    hq_image = Image.open(hq_result.image_path)

    # Render and convert to bitonal (what the old pipeline does)
    image, _ = render_pdf_page(pdf_path, page_number, dpi)
    gray = image.convert("L")
    bitonal = gray.point(lambda x: 255 if x > 128 else 0, mode="1")

    # Compare
    hq_array = np.array(hq_image)
    bitonal_array = np.array(bitonal.convert("L"))  # Convert back for comparison

    # Calculate information metrics
    hq_unique_values = len(np.unique(hq_array))
    bitonal_unique_values = len(np.unique(bitonal_array))  # Will be 2 (0 and 255)

    # Entropy (measure of information)
    def entropy(arr):
        hist, _ = np.histogram(arr.flatten(), bins=256, range=(0, 256))
        hist = hist[hist > 0]
        prob = hist / hist.sum()
        return -np.sum(prob * np.log2(prob))

    hq_entropy = entropy(hq_array)
    bitonal_entropy = entropy(bitonal_array)

    return {
        "high_quality": {
            "path": str(hq_result.image_path),
            "mode": hq_result.color_mode,
            "unique_values": hq_unique_values,
            "entropy": round(hq_entropy, 3),
            "file_size_kb": hq_result.image_path.stat().st_size / 1024,
        },
        "bitonal": {
            "mode": "1 (bitonal)",
            "unique_values": bitonal_unique_values,
            "entropy": round(bitonal_entropy, 3),
        },
        "information_preserved": f"{(hq_entropy / max(bitonal_entropy, 0.001)) * 100:.1f}%",
        "recommendation": "Use high_quality for Gemini-First pipeline",
    }
