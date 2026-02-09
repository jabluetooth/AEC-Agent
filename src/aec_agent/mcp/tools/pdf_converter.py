"""
PDF-to-bitonal-TIFF conversion utility for the Raster Design pipeline.

AutoCAD Raster Design cannot attach PDF files directly as raster images.
This module converts PDF pages to bitonal (1-bit black & white) TIFF images
that Raster Design can import, clean up, and vectorize.

Uses PyMuPDF (fitz) for PDF rendering and Pillow for bitonal conversion.
"""

import os
from pathlib import Path

import structlog

logger = structlog.get_logger(__name__)


def convert_pdf_to_bitonal_tiff(
    pdf_path: str,
    page: int = 1,
    dpi: int = 300,
    threshold: int = 128,
    output_dir: str | None = None,
) -> str:
    """
    Convert a PDF page to a bitonal (1-bit) TIFF image.

    Args:
        pdf_path: Absolute path to the PDF file.
        page: Page number (1-based) to convert.
        dpi: Resolution for rendering (default 300 — good for construction drawings).
        threshold: Grayscale threshold for bitonal conversion (0-255, default 128).
                   Pixels darker than threshold become black, lighter become white.
        output_dir: Directory to save the output TIFF. If None, saves alongside
                    the source PDF with a _pageN_bitonal.tif suffix.

    Returns:
        Absolute path to the generated bitonal TIFF file.

    Raises:
        FileNotFoundError: If the PDF file does not exist.
        ValueError: If the page number is out of range or parameters are invalid.
        RuntimeError: If conversion fails.
    """
    import fitz  # PyMuPDF
    from PIL import Image

    # Validate inputs
    pdf_path = os.path.abspath(pdf_path)
    if not os.path.isfile(pdf_path):
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")

    if page < 1:
        raise ValueError("page must be >= 1")

    if dpi < 72 or dpi > 1200:
        raise ValueError("dpi must be between 72 and 1200")

    if threshold < 0 or threshold > 255:
        raise ValueError("threshold must be between 0 and 255")

    # Determine output path
    pdf_stem = Path(pdf_path).stem
    tiff_name = f"{pdf_stem}_page{page}_bitonal.tif"

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, tiff_name)
    else:
        output_path = os.path.join(os.path.dirname(pdf_path), tiff_name)

    logger.info(
        "Converting PDF to bitonal TIFF",
        pdf_path=pdf_path,
        page=page,
        dpi=dpi,
        threshold=threshold,
        output_path=output_path,
    )

    try:
        # Open PDF and render the target page
        doc = fitz.open(pdf_path)

        if page > len(doc):
            doc.close()
            raise ValueError(
                f"Page {page} out of range. PDF has {len(doc)} page(s)."
            )

        # fitz uses 0-based page indexing
        pdf_page = doc[page - 1]

        # Render at target DPI (default PDF is 72 DPI, so scale = dpi/72)
        zoom = dpi / 72.0
        matrix = fitz.Matrix(zoom, zoom)
        pixmap = pdf_page.get_pixmap(matrix=matrix, alpha=False)

        # Convert PyMuPDF pixmap to Pillow Image
        img = Image.frombytes("RGB", (pixmap.width, pixmap.height), pixmap.samples)
        doc.close()

        # Convert to grayscale then to bitonal (1-bit)
        grayscale = img.convert("L")
        bitonal = grayscale.point(lambda px: 255 if px > threshold else 0, mode="1")

        # Save as TIFF with Group4 compression (optimal for bitonal)
        bitonal.save(
            output_path,
            format="TIFF",
            compression="group4",
            dpi=(dpi, dpi),
        )

        # Verify output
        file_size = os.path.getsize(output_path)
        logger.info(
            "Bitonal TIFF created",
            output_path=output_path,
            width=pixmap.width,
            height=pixmap.height,
            dpi=dpi,
            file_size_bytes=file_size,
        )

        return output_path

    except (ValueError, FileNotFoundError):
        raise
    except Exception as e:
        raise RuntimeError(f"PDF to bitonal TIFF conversion failed: {e}") from e
