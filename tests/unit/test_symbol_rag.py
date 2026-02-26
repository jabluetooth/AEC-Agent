"""
Unit tests for RAG Symbol Recognition.

Tests the Symbol RAG system including:
- CLIP encoder initialization and encoding
- Symbol library repository operations
- Symbol matching and search
- Integration with the pipeline
"""

from __future__ import annotations

import pytest
from dataclasses import dataclass
from typing import Optional
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import numpy as np


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def mock_pool():
    """Create a mock database pool."""
    pool = MagicMock()
    pool.fetch = AsyncMock(return_value=[])
    pool.fetchrow = AsyncMock(return_value=None)
    pool.fetchval = AsyncMock(return_value=None)
    pool.execute = AsyncMock()
    pool.transaction = MagicMock()
    return pool


@pytest.fixture
def sample_symbol_entry():
    """Create a sample symbol library entry."""
    from aec_agent.mcp.tools.gemini_first.symbol_rag import SymbolLibraryEntry

    return SymbolLibraryEntry(
        id=uuid4(),
        block_name="E-OUTL-DUP",
        display_name="Duplex Outlet",
        description="Standard duplex electrical outlet",
        domain="electrical",
        category="outlet",
        subcategory="duplex",
        layer="E-POWR-OUTL",
        embedding=[0.1] * 512,
        preview_image=None,
        default_scale=1.0,
        default_rotation=0.0,
        attributes={},
        standards=["NCS", "NEC"],
        jurisdiction="CA",
        code_references=["NEC 210.52"],
    )


@pytest.fixture
def sample_match_row():
    """Create a sample database row for symbol match."""
    return {
        "id": uuid4(),
        "block_name": "E-OUTL-DUP",
        "display_name": "Duplex Outlet",
        "domain": "electrical",
        "category": "outlet",
        "subcategory": "duplex",
        "layer": "E-POWR-OUTL",
        "attributes": {},
        "default_scale": 1.0,
        "default_rotation": 0.0,
        "distance": 0.15,  # 85% confidence
    }


# =============================================================================
# Test SymbolDomain Enum
# =============================================================================

class TestSymbolDomain:
    """Tests for SymbolDomain enum."""

    def test_domain_values(self):
        """Test domain enum values."""
        from aec_agent.mcp.tools.gemini_first.symbol_rag import SymbolDomain

        assert SymbolDomain.ELECTRICAL.value == "electrical"
        assert SymbolDomain.MECHANICAL.value == "mechanical"
        assert SymbolDomain.PLUMBING.value == "plumbing"
        assert SymbolDomain.FIRE.value == "fire"
        assert SymbolDomain.ARCHITECTURAL.value == "architectural"

    def test_domain_from_string(self):
        """Test creating domain from string."""
        from aec_agent.mcp.tools.gemini_first.symbol_rag import SymbolDomain

        assert SymbolDomain("electrical") == SymbolDomain.ELECTRICAL
        assert SymbolDomain("fire") == SymbolDomain.FIRE


# =============================================================================
# Test SymbolRecognitionConfig
# =============================================================================

class TestSymbolRecognitionConfig:
    """Tests for SymbolRecognitionConfig."""

    def test_default_config(self):
        """Test default configuration values."""
        from aec_agent.mcp.tools.gemini_first.symbol_rag import SymbolRecognitionConfig

        config = SymbolRecognitionConfig()
        assert config.embedding_dim == 512
        assert config.top_k == 5
        assert config.min_confidence == 0.5
        assert config.image_size == 224
        assert config.cache_embeddings is True

    def test_custom_config(self):
        """Test custom configuration."""
        from aec_agent.mcp.tools.gemini_first.symbol_rag import SymbolRecognitionConfig

        config = SymbolRecognitionConfig(
            top_k=10,
            min_confidence=0.7,
            device="cpu",
        )
        assert config.top_k == 10
        assert config.min_confidence == 0.7
        assert config.device == "cpu"


# =============================================================================
# Test SymbolLibraryEntry
# =============================================================================

class TestSymbolLibraryEntry:
    """Tests for SymbolLibraryEntry dataclass."""

    def test_entry_creation(self, sample_symbol_entry):
        """Test creating a symbol entry."""
        entry = sample_symbol_entry
        assert entry.block_name == "E-OUTL-DUP"
        assert entry.domain == "electrical"
        assert entry.category == "outlet"
        assert entry.layer == "E-POWR-OUTL"
        assert len(entry.embedding) == 512
        assert entry.is_active is True

    def test_entry_defaults(self):
        """Test entry default values."""
        from aec_agent.mcp.tools.gemini_first.symbol_rag import SymbolLibraryEntry

        entry = SymbolLibraryEntry(
            id=uuid4(),
            block_name="TEST",
            display_name="Test",
            description=None,
            domain="electrical",
            category="test",
            subcategory=None,
            layer="E-TEST",
            embedding=None,
            preview_image=None,
        )
        assert entry.default_scale == 1.0
        assert entry.default_rotation == 0.0
        assert entry.attributes == {}
        assert entry.standards == []


# =============================================================================
# Test SymbolMatch
# =============================================================================

class TestSymbolMatch:
    """Tests for SymbolMatch dataclass."""

    def test_match_creation(self):
        """Test creating a symbol match."""
        from aec_agent.mcp.tools.gemini_first.symbol_rag import SymbolMatch, SymbolDomain

        match = SymbolMatch(
            symbol_id=uuid4(),
            block_name="E-OUTL-DUP",
            display_name="Duplex Outlet",
            domain=SymbolDomain.ELECTRICAL,
            category="outlet",
            subcategory="duplex",
            layer="E-POWR-OUTL",
            confidence=0.85,
            distance=0.15,
        )
        assert match.block_name == "E-OUTL-DUP"
        assert match.confidence == 0.85
        assert match.domain == SymbolDomain.ELECTRICAL


# =============================================================================
# Test CLIPEncoder
# =============================================================================

class TestCLIPEncoder:
    """Tests for CLIPEncoder."""

    def test_singleton_pattern(self):
        """Test that CLIPEncoder uses singleton pattern."""
        from aec_agent.mcp.tools.gemini_first.symbol_rag import CLIPEncoder

        # Clear singleton
        CLIPEncoder._instance = None

        encoder1 = CLIPEncoder.get_instance()
        encoder2 = CLIPEncoder.get_instance()
        assert encoder1 is encoder2

        # Clean up
        CLIPEncoder._instance = None

    def test_device_detection_cpu_fallback(self):
        """Test CPU fallback when CUDA unavailable."""
        from aec_agent.mcp.tools.gemini_first.symbol_rag import CLIPEncoder

        # Clear singleton
        CLIPEncoder._instance = None

        encoder = CLIPEncoder()

        # Mock torch to simulate no CUDA
        with patch("aec_agent.mcp.tools.gemini_first.symbol_rag.TORCH_AVAILABLE", True):
            with patch("aec_agent.mcp.tools.gemini_first.symbol_rag.torch") as mock_torch:
                mock_torch.cuda.is_available.return_value = False
                mock_torch.backends.mps.is_available.return_value = False

                device = encoder._detect_device()
                assert device == "cpu"

        CLIPEncoder._instance = None

    @pytest.mark.skipif(
        True,  # Skip actual model loading in unit tests
        reason="CLIP model loading requires network and is slow"
    )
    async def test_initialize_loads_model(self):
        """Test that initialize loads the CLIP model."""
        from aec_agent.mcp.tools.gemini_first.symbol_rag import CLIPEncoder

        encoder = CLIPEncoder()
        await encoder.initialize()
        assert encoder._initialized is True

    def test_encode_without_init_raises(self):
        """Test that encoding without initialization raises error."""
        from aec_agent.mcp.tools.gemini_first.symbol_rag import CLIPEncoder

        CLIPEncoder._instance = None
        encoder = CLIPEncoder()

        with pytest.raises(RuntimeError, match="not initialized"):
            # Create a mock PIL image
            mock_image = MagicMock()
            mock_image.mode = "RGB"
            encoder.encode_image(mock_image)

        CLIPEncoder._instance = None


# =============================================================================
# Test SymbolRAGRepository
# =============================================================================

class TestSymbolRAGRepository:
    """Tests for SymbolRAGRepository."""

    @pytest.mark.asyncio
    async def test_add_symbol(self, mock_pool, sample_symbol_entry):
        """Test adding a symbol to the library."""
        from aec_agent.mcp.tools.gemini_first.symbol_rag import SymbolRAGRepository

        # Setup mock
        expected_id = uuid4()
        mock_pool.fetchval = AsyncMock(return_value=expected_id)

        repo = SymbolRAGRepository(mock_pool)
        result = await repo.add_symbol(sample_symbol_entry)

        assert result == expected_id
        mock_pool.fetchval.assert_called_once()

    @pytest.mark.asyncio
    async def test_search_similar_empty_results(self, mock_pool):
        """Test search with no results."""
        from aec_agent.mcp.tools.gemini_first.symbol_rag import SymbolRAGRepository

        mock_pool.fetch = AsyncMock(return_value=[])

        repo = SymbolRAGRepository(mock_pool)
        results = await repo.search_similar(
            embedding=[0.1] * 512,
            domain="electrical",
            limit=5,
        )

        assert results == []

    @pytest.mark.asyncio
    async def test_search_similar_with_results(self, mock_pool, sample_match_row):
        """Test search with results."""
        from aec_agent.mcp.tools.gemini_first.symbol_rag import SymbolRAGRepository

        mock_pool.fetch = AsyncMock(return_value=[sample_match_row])

        repo = SymbolRAGRepository(mock_pool)
        results = await repo.search_similar(
            embedding=[0.1] * 512,
            limit=5,
        )

        assert len(results) == 1
        assert results[0].block_name == "E-OUTL-DUP"
        assert results[0].confidence == pytest.approx(0.85, rel=0.01)

    @pytest.mark.asyncio
    async def test_search_with_domain_filter(self, mock_pool):
        """Test that domain filter is applied in query."""
        from aec_agent.mcp.tools.gemini_first.symbol_rag import SymbolRAGRepository

        mock_pool.fetch = AsyncMock(return_value=[])

        repo = SymbolRAGRepository(mock_pool)
        await repo.search_similar(
            embedding=[0.1] * 512,
            domain="fire",
            limit=5,
        )

        # Check that query was called with domain parameter
        call_args = mock_pool.fetch.call_args
        assert "fire" in call_args[0] or any("fire" in str(arg) for arg in call_args[0])

    @pytest.mark.asyncio
    async def test_get_symbol_count(self, mock_pool):
        """Test getting symbol count."""
        from aec_agent.mcp.tools.gemini_first.symbol_rag import SymbolRAGRepository

        mock_pool.fetchval = AsyncMock(return_value=42)

        repo = SymbolRAGRepository(mock_pool)
        count = await repo.get_symbol_count()

        assert count == 42

    @pytest.mark.asyncio
    async def test_get_symbol_count_by_domain(self, mock_pool):
        """Test getting symbol count filtered by domain."""
        from aec_agent.mcp.tools.gemini_first.symbol_rag import SymbolRAGRepository

        mock_pool.fetchval = AsyncMock(return_value=15)

        repo = SymbolRAGRepository(mock_pool)
        count = await repo.get_symbol_count(domain="electrical")

        assert count == 15

    @pytest.mark.asyncio
    async def test_record_usage(self, mock_pool):
        """Test recording symbol usage."""
        from aec_agent.mcp.tools.gemini_first.symbol_rag import SymbolRAGRepository

        repo = SymbolRAGRepository(mock_pool)
        symbol_id = uuid4()

        await repo.record_usage(
            symbol_id=symbol_id,
            project_id=None,
            confidence=0.85,
            image_hash="abc123",
        )

        mock_pool.execute.assert_called_once()


# =============================================================================
# Test SymbolRAG
# =============================================================================

class TestSymbolRAG:
    """Tests for SymbolRAG service."""

    @pytest.mark.asyncio
    async def test_initialize(self, mock_pool):
        """Test initializing the RAG service."""
        from aec_agent.mcp.tools.gemini_first.symbol_rag import SymbolRAG, CLIPEncoder

        # Mock encoder
        CLIPEncoder._instance = None
        mock_encoder = MagicMock()
        mock_encoder.initialize = AsyncMock()

        with patch.object(CLIPEncoder, "get_instance", return_value=mock_encoder):
            rag = SymbolRAG(mock_pool)
            await rag.initialize()

            assert rag._initialized is True
            mock_encoder.initialize.assert_called_once()

        CLIPEncoder._instance = None

    @pytest.mark.asyncio
    async def test_recognize_not_initialized(self, mock_pool):
        """Test that recognize auto-initializes."""
        from aec_agent.mcp.tools.gemini_first.symbol_rag import SymbolRAG, CLIPEncoder

        CLIPEncoder._instance = None

        # Mock encoder with all needed methods
        mock_encoder = MagicMock()
        mock_encoder.initialize = AsyncMock()
        mock_encoder.encode_image = MagicMock(return_value=np.zeros(512))

        # Mock repository
        mock_repo = MagicMock()
        mock_repo.search_similar = AsyncMock(return_value=[])
        mock_repo.record_usage = AsyncMock()

        # Mock PIL image
        mock_image = MagicMock()
        mock_image.mode = "RGB"

        with patch.object(CLIPEncoder, "get_instance", return_value=mock_encoder):
            rag = SymbolRAG(mock_pool)
            rag.repository = mock_repo

            # Should auto-initialize
            results = await rag.recognize_symbol(mock_image)

            assert results == []
            mock_encoder.initialize.assert_called_once()

        CLIPEncoder._instance = None

    def test_compute_image_hash(self, mock_pool):
        """Test image hash computation."""
        from aec_agent.mcp.tools.gemini_first.symbol_rag import SymbolRAG
        from PIL import Image
        import io

        rag = SymbolRAG(mock_pool)

        # Create a simple test image
        img = Image.new("RGB", (10, 10), color="red")

        hash1 = rag._compute_image_hash(img)
        hash2 = rag._compute_image_hash(img)

        assert hash1 == hash2
        assert len(hash1) == 32  # MD5 hex digest

    def test_create_preview(self, mock_pool):
        """Test preview thumbnail creation."""
        from aec_agent.mcp.tools.gemini_first.symbol_rag import SymbolRAG
        from PIL import Image

        rag = SymbolRAG(mock_pool)

        # Create a test image
        img = Image.new("RGB", (256, 256), color="blue")

        preview_bytes = rag._create_preview(img, size=64)

        assert isinstance(preview_bytes, bytes)
        assert len(preview_bytes) > 0

        # Verify it's a valid PNG
        preview_img = Image.open(io.BytesIO(preview_bytes))
        assert preview_img.size[0] <= 64
        assert preview_img.size[1] <= 64


# Import io for BytesIO
import io


# =============================================================================
# Test Utility Functions
# =============================================================================

class TestUtilityFunctions:
    """Tests for utility functions."""

    def test_is_symbol_rag_available(self):
        """Test availability check."""
        from aec_agent.mcp.tools.gemini_first.symbol_rag import is_symbol_rag_available

        # Should return True or False based on installed packages
        result = is_symbol_rag_available()
        assert isinstance(result, bool)

    def test_extract_symbol_region(self):
        """Test extracting a symbol region from an image."""
        from aec_agent.mcp.tools.gemini_first.symbol_rag import extract_symbol_region
        from PIL import Image

        # Create a test image
        img = Image.new("RGB", (200, 200), color="white")

        # Extract a region
        region = extract_symbol_region(img, (50, 50, 150, 150), padding=10)

        # Should be cropped to bbox + padding
        assert region.size[0] <= 120  # 100 + 2*10
        assert region.size[1] <= 120

    def test_extract_symbol_region_with_edge_padding(self):
        """Test region extraction with edge padding clamping."""
        from aec_agent.mcp.tools.gemini_first.symbol_rag import extract_symbol_region
        from PIL import Image

        img = Image.new("RGB", (100, 100), color="white")

        # Extract region at edge - padding should be clamped
        region = extract_symbol_region(img, (0, 0, 50, 50), padding=20)

        # Padding clamped to image bounds
        assert region.size[0] <= 70
        assert region.size[1] <= 70

    def test_image_to_base64(self):
        """Test image to base64 conversion."""
        from aec_agent.mcp.tools.gemini_first.symbol_rag import image_to_base64
        from PIL import Image

        img = Image.new("RGB", (10, 10), color="green")

        b64 = image_to_base64(img)

        assert isinstance(b64, str)
        assert len(b64) > 0

    def test_base64_to_image(self):
        """Test base64 to image conversion."""
        from aec_agent.mcp.tools.gemini_first.symbol_rag import (
            image_to_base64,
            base64_to_image,
        )
        from PIL import Image

        # Create original
        original = Image.new("RGB", (10, 10), color="purple")

        # Round trip
        b64 = image_to_base64(original)
        restored = base64_to_image(b64)

        assert restored.size == original.size
        assert restored.mode == original.mode


# =============================================================================
# Test Symbol Library Seed Data
# =============================================================================

class TestSymbolLibrarySeed:
    """Tests for symbol library seed data."""

    def test_all_symbols_list(self):
        """Test that ALL_SYMBOLS contains symbols."""
        from aec_agent.mcp.tools.gemini_first.symbol_library_seed import ALL_SYMBOLS

        assert len(ALL_SYMBOLS) > 50  # Should have ~80 symbols

    def test_electrical_symbols_exist(self):
        """Test electrical symbols are defined."""
        from aec_agent.mcp.tools.gemini_first.symbol_library_seed import ELECTRICAL_SYMBOLS

        assert len(ELECTRICAL_SYMBOLS) > 0

        # Check for common symbols
        block_names = [s.block_name for s in ELECTRICAL_SYMBOLS]
        assert "E-OUTL-DUP" in block_names
        assert "E-SWCH-1P" in block_names

    def test_fire_symbols_exist(self):
        """Test fire alarm symbols are defined."""
        from aec_agent.mcp.tools.gemini_first.symbol_library_seed import FIRE_SYMBOLS

        assert len(FIRE_SYMBOLS) > 0

        block_names = [s.block_name for s in FIRE_SYMBOLS]
        assert "F-DETC-SMOK" in block_names
        assert "F-PULL" in block_names

    def test_symbol_definition_fields(self):
        """Test that symbol definitions have required fields."""
        from aec_agent.mcp.tools.gemini_first.symbol_library_seed import ALL_SYMBOLS

        for sym in ALL_SYMBOLS:
            assert sym.block_name, f"Symbol missing block_name"
            assert sym.display_name, f"Symbol {sym.block_name} missing display_name"
            assert sym.description, f"Symbol {sym.block_name} missing description"
            assert sym.domain, f"Symbol {sym.block_name} missing domain"
            assert sym.category, f"Symbol {sym.block_name} missing category"
            assert sym.layer, f"Symbol {sym.block_name} missing layer"
            assert sym.standards, f"Symbol {sym.block_name} missing standards"

    def test_layer_naming_convention(self):
        """Test that layers follow NCS naming convention."""
        from aec_agent.mcp.tools.gemini_first.symbol_library_seed import ALL_SYMBOLS

        for sym in ALL_SYMBOLS:
            layer = sym.layer

            # NCS layers should have discipline prefix
            if sym.domain == "electrical":
                assert layer.startswith("E-"), f"Electrical symbol {sym.block_name} has non-E layer: {layer}"
            elif sym.domain == "mechanical":
                assert layer.startswith("M-"), f"Mechanical symbol {sym.block_name} has non-M layer: {layer}"
            elif sym.domain == "plumbing":
                assert layer.startswith("P-"), f"Plumbing symbol {sym.block_name} has non-P layer: {layer}"
            elif sym.domain == "fire":
                assert layer.startswith("F-"), f"Fire symbol {sym.block_name} has non-F layer: {layer}"
            elif sym.domain == "architectural":
                assert layer.startswith("A-"), f"Arch symbol {sym.block_name} has non-A layer: {layer}"


# =============================================================================
# Integration Tests (with mocked database)
# =============================================================================

class TestSymbolRAGIntegration:
    """Integration tests for Symbol RAG with mocked dependencies."""

    @pytest.mark.asyncio
    async def test_full_recognition_flow(self, mock_pool, sample_match_row):
        """Test full symbol recognition flow."""
        from aec_agent.mcp.tools.gemini_first.symbol_rag import (
            SymbolRAG,
            CLIPEncoder,
            SymbolDomain,
        )
        from PIL import Image

        CLIPEncoder._instance = None

        # Mock encoder
        mock_encoder = MagicMock()
        mock_encoder.initialize = AsyncMock()
        mock_encoder.encode_image = MagicMock(return_value=np.zeros(512))

        # Setup mock pool to return results
        mock_pool.fetch = AsyncMock(return_value=[sample_match_row])
        mock_pool.execute = AsyncMock()

        # Create test image
        test_image = Image.new("RGB", (64, 64), color="red")

        with patch.object(CLIPEncoder, "get_instance", return_value=mock_encoder):
            rag = SymbolRAG(mock_pool)
            await rag.initialize()

            results = await rag.recognize_symbol(
                image=test_image,
                domain=SymbolDomain.ELECTRICAL,
                top_k=3,
            )

            assert len(results) == 1
            assert results[0].block_name == "E-OUTL-DUP"
            assert results[0].confidence > 0.5

        CLIPEncoder._instance = None

    @pytest.mark.asyncio
    async def test_text_search_flow(self, mock_pool, sample_match_row):
        """Test text-based symbol search."""
        from aec_agent.mcp.tools.gemini_first.symbol_rag import (
            SymbolRAG,
            CLIPEncoder,
        )

        CLIPEncoder._instance = None

        # Mock encoder
        mock_encoder = MagicMock()
        mock_encoder.initialize = AsyncMock()
        mock_encoder.encode_text = MagicMock(return_value=np.zeros(512))

        mock_pool.fetch = AsyncMock(return_value=[sample_match_row])

        with patch.object(CLIPEncoder, "get_instance", return_value=mock_encoder):
            rag = SymbolRAG(mock_pool)
            await rag.initialize()

            results = await rag.search_by_description("duplex outlet")

            mock_encoder.encode_text.assert_called_once_with("duplex outlet")
            assert len(results) == 1

        CLIPEncoder._instance = None
