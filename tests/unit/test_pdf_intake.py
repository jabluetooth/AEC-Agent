"""
Unit tests for Phase 1: PDF Intake & Rendering.

Tests the gemini_first.pdf_intake module which renders PDFs to
high-quality images WITHOUT destroying information.
"""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

# Import module under test
from aec_agent.mcp.tools.gemini_first.pdf_intake import (
    DEFAULT_DPI,
    GRAYSCALE_THRESHOLD,
    MAX_DPI,
    MIN_DPI,
    PDFInfo,
    PDFRenderResult,
    is_effectively_grayscale,
)


class TestIsEffectivelyGrayscale:
    """Tests for the is_effectively_grayscale function."""

    def test_pure_grayscale_image(self):
        """A pure grayscale image (R=G=B) should return True."""
        from PIL import Image

        # Create a grayscale gradient as RGB
        arr = np.zeros((100, 100, 3), dtype=np.uint8)
        for i in range(100):
            arr[i, :, :] = i * 2  # R=G=B for all pixels

        image = Image.fromarray(arr, mode="RGB")
        assert is_effectively_grayscale(image) is True

    def test_color_image(self):
        """An image with color variance should return False."""
        from PIL import Image

        # Create an image with color variance
        arr = np.zeros((100, 100, 3), dtype=np.uint8)
        arr[:50, :, 0] = 255  # Red top half
        arr[50:, :, 2] = 255  # Blue bottom half

        image = Image.fromarray(arr, mode="RGB")
        assert is_effectively_grayscale(image) is False

    def test_slight_color_variance(self):
        """Slight color variance below threshold should return True."""
        from PIL import Image

        # Create nearly grayscale with tiny variance
        arr = np.zeros((100, 100, 3), dtype=np.uint8)
        arr[:, :, 0] = 128  # R
        arr[:, :, 1] = 127  # G (slightly different)
        arr[:, :, 2] = 128  # B

        image = Image.fromarray(arr, mode="RGB")
        # Variance is about 0.4/255 ≈ 0.0016, well below default threshold
        assert is_effectively_grayscale(image, threshold=0.01) is True

    def test_grayscale_mode_image(self):
        """An image already in L mode should return True."""
        from PIL import Image

        arr = np.random.randint(0, 256, (100, 100), dtype=np.uint8)
        image = Image.fromarray(arr, mode="L")
        assert is_effectively_grayscale(image) is True

    def test_black_and_white_image(self):
        """A black and white image should return True."""
        from PIL import Image

        # Create black and white pattern
        arr = np.zeros((100, 100, 3), dtype=np.uint8)
        arr[::2, :, :] = 255  # White stripes

        image = Image.fromarray(arr, mode="RGB")
        assert is_effectively_grayscale(image) is True


class TestPDFRenderResult:
    """Tests for the PDFRenderResult dataclass."""

    def test_basic_creation(self, tmp_path):
        """Test creating a PDFRenderResult."""
        image_path = tmp_path / "test.png"
        image_path.touch()

        result = PDFRenderResult(
            image_path=image_path,
            width_px=2400,
            height_px=1800,
            dpi=300,
            color_mode="RGB",
            original_pdf=Path("test.pdf"),
            page_number=0,
            page_count=5,
            page_width_pts=612,  # 8.5 inches
            page_height_pts=792,  # 11 inches
        )

        assert result.width_px == 2400
        assert result.height_px == 1800
        assert result.dpi == 300
        assert result.color_mode == "RGB"
        assert result.page_number == 0
        assert result.page_count == 5

    def test_computed_properties(self, tmp_path):
        """Test computed properties like page_width_inches."""
        image_path = tmp_path / "test.png"
        image_path.touch()

        result = PDFRenderResult(
            image_path=image_path,
            width_px=2550,  # 8.5" at 300 DPI
            height_px=3300,  # 11" at 300 DPI
            dpi=300,
            color_mode="L",
            original_pdf=Path("test.pdf"),
            page_number=0,
            page_count=1,
            page_width_pts=612,  # 8.5"
            page_height_pts=792,  # 11"
        )

        assert abs(result.page_width_inches - 8.5) < 0.01
        assert abs(result.page_height_inches - 11.0) < 0.01
        assert abs(result.aspect_ratio - (2550 / 3300)) < 0.001

    def test_to_dict(self, tmp_path):
        """Test serialization to dictionary."""
        image_path = tmp_path / "test.png"
        image_path.touch()

        result = PDFRenderResult(
            image_path=image_path,
            width_px=2400,
            height_px=1800,
            dpi=300,
            color_mode="RGB",
            original_pdf=Path("test.pdf"),
            page_number=0,
            page_count=1,
            page_width_pts=612,
            page_height_pts=792,
        )

        d = result.to_dict()

        assert "image_path" in d
        assert "width_px" in d
        assert "dpi" in d
        assert "page_width_inches" in d  # Computed property included
        assert d["color_mode"] == "RGB"


class TestPDFInfo:
    """Tests for the PDFInfo dataclass."""

    def test_basic_creation(self):
        """Test creating a PDFInfo."""
        info = PDFInfo(
            path=Path("test.pdf"),
            page_count=10,
            pages=[
                {"page_number": i, "width_pts": 612, "height_pts": 792}
                for i in range(10)
            ],
            title="Test Drawing",
            author="Test Author",
        )

        assert info.page_count == 10
        assert len(info.pages) == 10
        assert info.title == "Test Drawing"

    def test_to_dict(self):
        """Test serialization to dictionary."""
        info = PDFInfo(
            path=Path("test.pdf"),
            page_count=1,
            pages=[{"page_number": 0}],
        )

        d = info.to_dict()

        assert "path" in d
        assert "page_count" in d
        assert d["page_count"] == 1


class TestConstants:
    """Test module constants."""

    def test_dpi_range(self):
        """Test DPI constants are reasonable."""
        assert MIN_DPI == 72
        assert MAX_DPI == 1200
        assert MIN_DPI < DEFAULT_DPI < MAX_DPI
        assert DEFAULT_DPI == 300

    def test_grayscale_threshold(self):
        """Test grayscale threshold is reasonable."""
        assert 0 < GRAYSCALE_THRESHOLD < 0.1


# Integration tests (require PyMuPDF and Pillow)
@pytest.mark.skipif(
    not all([
        pytest.importorskip("fitz", reason="PyMuPDF not installed"),
        pytest.importorskip("PIL", reason="Pillow not installed"),
    ]),
    reason="PDF rendering dependencies not available"
)
class TestPDFRendering:
    """Integration tests for actual PDF rendering."""

    @pytest.fixture
    def sample_pdf(self, tmp_path):
        """Create a simple PDF for testing."""
        try:
            import fitz

            # Create a simple PDF with text
            doc = fitz.open()
            page = doc.new_page(width=612, height=792)  # Letter size

            # Add some content
            page.insert_text(
                (72, 72),
                "Test Drawing",
                fontsize=24,
                color=(0, 0, 0)
            )
            page.draw_line((72, 100), (540, 100), color=(0, 0, 0), width=2)
            page.draw_rect((100, 200, 400, 500), color=(0, 0, 0), width=1)

            pdf_path = tmp_path / "test_drawing.pdf"
            doc.save(str(pdf_path))
            doc.close()

            return pdf_path
        except Exception:
            pytest.skip("Could not create test PDF")

    def test_render_pdf_high_quality(self, sample_pdf, tmp_path):
        """Test rendering a PDF to high-quality image."""
        from aec_agent.mcp.tools.gemini_first.pdf_intake import (
            render_pdf_high_quality,
        )

        result = render_pdf_high_quality(
            pdf_path=sample_pdf,
            page_number=0,
            dpi=300,
            output_dir=tmp_path,
        )

        assert result.image_path.exists()
        assert result.image_path.suffix == ".png"
        assert result.width_px > 0
        assert result.height_px > 0
        assert result.dpi == 300
        assert result.page_number == 0
        assert result.page_count == 1

    def test_render_preserves_quality(self, sample_pdf, tmp_path):
        """Test that rendering preserves quality (not bitonal)."""
        from PIL import Image

        from aec_agent.mcp.tools.gemini_first.pdf_intake import (
            render_pdf_high_quality,
        )

        result = render_pdf_high_quality(
            pdf_path=sample_pdf,
            dpi=300,
            output_dir=tmp_path,
            convert_grayscale=True,
        )

        # Load the rendered image
        image = Image.open(result.image_path)

        # Should be grayscale (L) or RGB, NOT bitonal (1)
        assert image.mode in ("RGB", "L")

        # Should have more than 2 unique values (unlike bitonal)
        arr = np.array(image)
        unique_values = len(np.unique(arr))
        assert unique_values > 2, "Image should not be bitonal"

    def test_get_pdf_info(self, sample_pdf):
        """Test getting PDF info without rendering."""
        from aec_agent.mcp.tools.gemini_first.pdf_intake import get_pdf_info

        info = get_pdf_info(sample_pdf)

        assert info.page_count == 1
        assert len(info.pages) == 1
        assert info.pages[0]["page_number"] == 0
        assert info.pages[0]["width_pts"] == 612
        assert info.pages[0]["height_pts"] == 792

    def test_render_invalid_page(self, sample_pdf):
        """Test rendering invalid page number raises error."""
        from aec_agent.mcp.tools.gemini_first.pdf_intake import (
            render_pdf_high_quality,
        )

        with pytest.raises(ValueError, match="Invalid page number"):
            render_pdf_high_quality(sample_pdf, page_number=99)

    def test_render_invalid_dpi(self, sample_pdf):
        """Test rendering with invalid DPI raises error."""
        from aec_agent.mcp.tools.gemini_first.pdf_intake import (
            render_pdf_high_quality,
        )

        with pytest.raises(ValueError, match="DPI must be between"):
            render_pdf_high_quality(sample_pdf, dpi=50)  # Below MIN_DPI

        with pytest.raises(ValueError, match="DPI must be between"):
            render_pdf_high_quality(sample_pdf, dpi=2000)  # Above MAX_DPI

    def test_file_not_found(self):
        """Test rendering nonexistent file raises error."""
        from aec_agent.mcp.tools.gemini_first.pdf_intake import (
            render_pdf_high_quality,
        )

        with pytest.raises(FileNotFoundError):
            render_pdf_high_quality("nonexistent.pdf")

    def test_compare_with_bitonal(self, sample_pdf):
        """Test comparison function shows quality difference."""
        from aec_agent.mcp.tools.gemini_first.pdf_intake import compare_with_bitonal

        comparison = compare_with_bitonal(sample_pdf)

        # High quality should have more unique values
        assert comparison["high_quality"]["unique_values"] > comparison["bitonal"]["unique_values"]

        # High quality should have higher entropy (more information)
        assert comparison["high_quality"]["entropy"] > comparison["bitonal"]["entropy"]

        # Bitonal should have exactly 2 unique values
        assert comparison["bitonal"]["unique_values"] == 2


class TestAsyncRendering:
    """Test async rendering function."""

    @pytest.mark.asyncio
    async def test_async_render(self, tmp_path):
        """Test async rendering wrapper."""
        # Skip if dependencies not available
        pytest.importorskip("fitz", reason="PyMuPDF not installed")
        pytest.importorskip("PIL", reason="Pillow not installed")

        import fitz

        from aec_agent.mcp.tools.gemini_first.pdf_intake import (
            render_pdf_high_quality_async,
        )

        # Create a simple PDF
        doc = fitz.open()
        page = doc.new_page()
        page.insert_text((72, 72), "Async Test")
        pdf_path = tmp_path / "async_test.pdf"
        doc.save(str(pdf_path))
        doc.close()

        # Test async rendering
        result = await render_pdf_high_quality_async(
            pdf_path=pdf_path,
            output_dir=tmp_path,
        )

        assert result.image_path.exists()
        assert result.width_px > 0
