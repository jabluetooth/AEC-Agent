"""
Unit tests for the Best Practices PDF to AutoCAD Vectorization Pipeline.

Tests the 7-stage pipeline that combines optimal algorithms for each step:
1. PDF Rendering (PyMuPDF + Real-ESRGAN)
2. Preprocessing (NLM + Hough + ensemble binarization)
3. Gemini Analysis
4. Vector Extraction (LSD + HAWP + hybrid)
5. Symbol Recognition (CLIP + pgvector RAG)
6. Validation (line straightening + endpoint connection)
7. Output
"""

import pytest
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from pathlib import Path
import numpy as np
from PIL import Image

from aec_agent.mcp.tools.gemini_first.best_practices_pipeline import (
    BestPracticesPipeline,
    BestPracticesConfig,
    PipelineResult,
    StageResult,
    VectorizationMethod,
    SymbolRecognitionMethod,
    run_best_practices_pipeline,
    run_best_practices_pipeline_sync,
)


# =============================================================================
# Configuration Tests
# =============================================================================


class TestBestPracticesConfig:
    """Test configuration dataclass."""

    def test_default_values(self):
        """Test default configuration values."""
        config = BestPracticesConfig()

        # Stage 1: PDF Rendering
        assert config.dpi == 300
        assert config.auto_dpi is True
        assert config.convert_grayscale is True

        # Stage 2: Preprocessing
        assert config.denoise is True
        assert config.denoise_strength == 10
        assert config.deskew is True
        assert config.deskew_method == "hough"
        assert config.binarize is True
        assert config.binarize_method == "ensemble"
        assert config.super_resolution is True
        assert config.super_resolution_scale == 4

        # Stage 3: Gemini Analysis
        assert config.gemini_model == "gemini-2.0-flash"
        assert config.analyze_scale is True

        # Stage 4: Vector Extraction
        assert config.vectorization_method == VectorizationMethod.HYBRID
        assert config.extract_lines is True
        assert config.extract_circles is True
        assert config.use_hawp_junctions is True

        # Stage 5: Symbol Recognition
        assert config.symbol_method == SymbolRecognitionMethod.RAG
        assert config.symbol_min_confidence == 0.5
        assert config.symbol_top_k == 3

        # Stage 6: Validation
        assert config.straighten_lines is True
        assert config.straighten_tolerance_deg == 5.0
        assert config.connect_endpoints is True
        assert config.connect_tolerance_px == 10.0
        assert config.snap_to_grid is True
        assert config.gemini_visual_qa is True

        # Stage 7: Output
        assert config.use_ncs_layers is True
        assert config.output_format == "entities"

    def test_custom_config(self):
        """Test custom configuration values."""
        config = BestPracticesConfig(
            dpi=600,
            super_resolution=False,
            symbol_method=SymbolRecognitionMethod.HARDCODED,
            straighten_tolerance_deg=3.0,
            gemini_visual_qa=False,
        )

        assert config.dpi == 600
        assert config.super_resolution is False
        assert config.symbol_method == SymbolRecognitionMethod.HARDCODED
        assert config.straighten_tolerance_deg == 3.0
        assert config.gemini_visual_qa is False


class TestVectorizationMethod:
    """Test vectorization method enum."""

    def test_enum_values(self):
        """Test all vectorization methods exist."""
        assert VectorizationMethod.VTRACER.value == "vtracer"
        assert VectorizationMethod.BEZIER_SPLATTING.value == "bezier_splatting"
        assert VectorizationMethod.OPENCV_LSD.value == "opencv_lsd"
        assert VectorizationMethod.HYBRID.value == "hybrid"


class TestSymbolRecognitionMethod:
    """Test symbol recognition method enum."""

    def test_enum_values(self):
        """Test all symbol recognition methods exist."""
        assert SymbolRecognitionMethod.RAG.value == "rag"
        assert SymbolRecognitionMethod.HARDCODED.value == "hardcoded"
        assert SymbolRecognitionMethod.GEMINI_ONLY.value == "gemini_only"


# =============================================================================
# Result Dataclass Tests
# =============================================================================


class TestStageResult:
    """Test stage result dataclass."""

    def test_success_result(self):
        """Test successful stage result."""
        result = StageResult(
            stage="1_render_pdf",
            success=True,
            duration_ms=150.5,
            data={"image_path": "/tmp/image.png"},
        )

        assert result.stage == "1_render_pdf"
        assert result.success is True
        assert result.duration_ms == 150.5
        assert result.data["image_path"] == "/tmp/image.png"
        assert result.errors == []
        assert result.warnings == []

    def test_failure_result(self):
        """Test failed stage result."""
        result = StageResult(
            stage="3_gemini_analysis",
            success=False,
            duration_ms=50.0,
            errors=["API rate limit exceeded"],
        )

        assert result.success is False
        assert len(result.errors) == 1
        assert "rate limit" in result.errors[0]


class TestPipelineResult:
    """Test pipeline result dataclass."""

    def test_default_values(self):
        """Test default pipeline result values."""
        result = PipelineResult(
            success=False,
            pdf_path="/path/to/file.pdf",
            page=1,
        )

        assert result.success is False
        assert result.pdf_path == "/path/to/file.pdf"
        assert result.page == 1
        assert result.stages == []
        assert result.total_duration_ms == 0.0
        assert result.image_path is None
        assert result.entities == []
        assert result.symbols_recognized == []
        assert result.entity_count == 0
        assert result.symbol_count == 0

    def test_to_dict(self):
        """Test serialization to dictionary."""
        result = PipelineResult(
            success=True,
            pdf_path="/path/to/file.pdf",
            page=1,
            total_duration_ms=5000.0,
            entity_count=150,
            symbol_count=12,
            line_count=100,
            text_count=25,
        )
        result.stages.append(StageResult(
            stage="1_render_pdf",
            success=True,
            duration_ms=200.0,
        ))

        d = result.to_dict()

        assert d["success"] is True
        assert d["pdf_path"] == "/path/to/file.pdf"
        assert d["page"] == 1
        assert d["total_duration_ms"] == 5000.0
        assert len(d["stages"]) == 1
        assert d["stages"][0]["stage"] == "1_render_pdf"
        assert d["statistics"]["entity_count"] == 150
        assert d["statistics"]["symbol_count"] == 12


# =============================================================================
# Pipeline Class Tests
# =============================================================================


class TestBestPracticesPipeline:
    """Test the main pipeline class."""

    def test_init_default_config(self):
        """Test pipeline initialization with default config."""
        pipeline = BestPracticesPipeline()

        assert pipeline.config is not None
        assert pipeline.config.dpi == 300
        assert pipeline._initialized is False
        assert pipeline._symbol_rag is None

    def test_init_custom_config(self):
        """Test pipeline initialization with custom config."""
        config = BestPracticesConfig(
            dpi=600,
            symbol_method=SymbolRecognitionMethod.HARDCODED,
        )
        pipeline = BestPracticesPipeline(config=config)

        assert pipeline.config.dpi == 600
        assert pipeline.config.symbol_method == SymbolRecognitionMethod.HARDCODED

    @pytest.mark.asyncio
    async def test_initialize_with_hardcoded_symbols(self):
        """Test initialization when using hardcoded symbols (no RAG)."""
        config = BestPracticesConfig(
            symbol_method=SymbolRecognitionMethod.HARDCODED,
        )
        pipeline = BestPracticesPipeline(config=config)

        await pipeline.initialize()

        assert pipeline._initialized is True
        assert pipeline._symbol_rag is None  # No RAG initialized

    @pytest.mark.asyncio
    async def test_initialize_rag_fallback(self):
        """Test RAG initialization falls back to hardcoded on failure."""
        config = BestPracticesConfig(
            symbol_method=SymbolRecognitionMethod.RAG,
        )
        pipeline = BestPracticesPipeline(config=config)

        # Patch to simulate RAG not available
        with patch(
            "aec_agent.mcp.tools.gemini_first.symbol_rag.is_symbol_rag_available",
            return_value=False,
        ):
            await pipeline.initialize()

        assert pipeline._initialized is True
        # Should fall back to hardcoded
        assert pipeline.config.symbol_method == SymbolRecognitionMethod.HARDCODED


class TestSymbolTypeMapping:
    """Test symbol type to domain mapping."""

    def test_electrical_symbols(self):
        """Test electrical symbol type mapping."""
        config = BestPracticesConfig()
        pipeline = BestPracticesPipeline(config)

        from aec_agent.mcp.tools.gemini_first.symbol_rag import SymbolDomain

        assert pipeline._map_symbol_type_to_domain("outlet") == SymbolDomain.ELECTRICAL
        assert pipeline._map_symbol_type_to_domain("switch") == SymbolDomain.ELECTRICAL
        assert pipeline._map_symbol_type_to_domain("light") == SymbolDomain.ELECTRICAL
        assert pipeline._map_symbol_type_to_domain("panel") == SymbolDomain.ELECTRICAL

    def test_mechanical_symbols(self):
        """Test mechanical symbol type mapping."""
        config = BestPracticesConfig()
        pipeline = BestPracticesPipeline(config)

        from aec_agent.mcp.tools.gemini_first.symbol_rag import SymbolDomain

        assert pipeline._map_symbol_type_to_domain("diffuser") == SymbolDomain.MECHANICAL
        assert pipeline._map_symbol_type_to_domain("thermostat") == SymbolDomain.MECHANICAL
        assert pipeline._map_symbol_type_to_domain("vav") == SymbolDomain.MECHANICAL

    def test_plumbing_symbols(self):
        """Test plumbing symbol type mapping."""
        config = BestPracticesConfig()
        pipeline = BestPracticesPipeline(config)

        from aec_agent.mcp.tools.gemini_first.symbol_rag import SymbolDomain

        assert pipeline._map_symbol_type_to_domain("valve") == SymbolDomain.PLUMBING
        assert pipeline._map_symbol_type_to_domain("fixture") == SymbolDomain.PLUMBING
        assert pipeline._map_symbol_type_to_domain("sink") == SymbolDomain.PLUMBING

    def test_fire_symbols(self):
        """Test fire alarm symbol type mapping."""
        config = BestPracticesConfig()
        pipeline = BestPracticesPipeline(config)

        from aec_agent.mcp.tools.gemini_first.symbol_rag import SymbolDomain

        assert pipeline._map_symbol_type_to_domain("detector") == SymbolDomain.FIRE
        assert pipeline._map_symbol_type_to_domain("smoke") == SymbolDomain.FIRE
        assert pipeline._map_symbol_type_to_domain("pull_station") == SymbolDomain.FIRE

    def test_architectural_symbols(self):
        """Test architectural symbol type mapping."""
        config = BestPracticesConfig()
        pipeline = BestPracticesPipeline(config)

        from aec_agent.mcp.tools.gemini_first.symbol_rag import SymbolDomain

        assert pipeline._map_symbol_type_to_domain("door") == SymbolDomain.ARCHITECTURAL
        assert pipeline._map_symbol_type_to_domain("window") == SymbolDomain.ARCHITECTURAL
        assert pipeline._map_symbol_type_to_domain("stairs") == SymbolDomain.ARCHITECTURAL

    def test_unknown_symbol_returns_none(self):
        """Test unknown symbol type returns None."""
        config = BestPracticesConfig()
        pipeline = BestPracticesPipeline(config)

        assert pipeline._map_symbol_type_to_domain("unknown_type") is None
        assert pipeline._map_symbol_type_to_domain("xyz") is None


# =============================================================================
# Stage Tests (Mocked)
# =============================================================================


class TestStage1RenderPDF:
    """Test Stage 1: PDF Rendering."""

    @pytest.mark.asyncio
    async def test_render_pdf_handles_exception(self):
        """Test PDF rendering handles exceptions gracefully."""
        config = BestPracticesConfig(super_resolution=False)
        pipeline = BestPracticesPipeline(config)

        # Calling with non-existent file should return failure
        result = await pipeline._stage1_render_pdf(
            Path("/nonexistent/test.pdf"),
            page=1,
            output_dir=Path("/tmp/output"),
        )

        assert result.success is False
        assert result.stage == "1_render_pdf"
        assert len(result.errors) > 0


class TestStage2Preprocess:
    """Test Stage 2: Preprocessing."""

    @pytest.mark.asyncio
    async def test_preprocess_with_disabled_options(self):
        """Test preprocessing with all options disabled."""
        config = BestPracticesConfig(
            denoise=False,
            deskew=False,
            binarize=False,
            enhance_contrast=False,
        )
        pipeline = BestPracticesPipeline(config)

        # Create test image (grayscale)
        test_image = Image.new("L", (100, 100), color=128)

        result = await pipeline._stage2_preprocess(test_image, dpi=300)

        assert result.success is True
        assert result.stage == "2_preprocess"
        assert "processed_image" in result.data
        assert result.data["skew_angle"] == 0.0  # No deskew applied


class TestStage3Analyze:
    """Test Stage 3: Gemini Analysis."""

    @pytest.mark.asyncio
    async def test_analyze_handles_exception(self):
        """Test Gemini analysis handles exceptions gracefully."""
        config = BestPracticesConfig()
        pipeline = BestPracticesPipeline(config)

        test_image = Image.new("L", (100, 100), color=255)

        # Mock analyze_drawing to raise an exception
        with patch(
            "aec_agent.mcp.tools.gemini_first.gemini_understanding.analyze_drawing",
            new_callable=AsyncMock,
            side_effect=Exception("API error"),
        ):
            result = await pipeline._stage3_analyze(
                test_image,
                image_path=Path("/tmp/test.png"),
            )

        assert result.success is False
        assert result.stage == "3_gemini_analysis"
        assert len(result.errors) > 0
        assert "API error" in result.errors[0]


class TestStage5SymbolRAG:
    """Test Stage 5: Symbol Recognition (RAG)."""

    @pytest.mark.asyncio
    async def test_symbol_rag_no_symbols(self):
        """Test symbol recognition with no symbols detected."""
        config = BestPracticesConfig()
        pipeline = BestPracticesPipeline(config)

        # Mock analysis with no symbols
        mock_analysis = Mock()
        mock_analysis.elements = Mock()
        mock_analysis.elements.symbols = []

        test_image = Image.new("L", (100, 100), color=255)
        mock_calibration = Mock()

        result = await pipeline._stage5_recognize_symbols(
            test_image,
            mock_analysis,
            mock_calibration,
        )

        assert result.success is True
        assert result.stage == "5_symbol_rag"
        assert result.data["symbol_entities"] == []
        assert "No symbols" in result.warnings[0]

    @pytest.mark.asyncio
    async def test_symbol_rag_with_hardcoded_fallback(self):
        """Test symbol recognition falls back to hardcoded mapping."""
        config = BestPracticesConfig(
            symbol_method=SymbolRecognitionMethod.HARDCODED,
        )
        pipeline = BestPracticesPipeline(config)
        pipeline._symbol_rag = None  # No RAG

        # Mock analysis with symbols
        mock_symbol = Mock()
        mock_symbol.symbol_type = "outlet"
        mock_symbol.subtype = "duplex"
        mock_symbol.position = (100, 100)
        mock_symbol.rotation = 0.0

        mock_analysis = Mock()
        mock_analysis.elements = Mock()
        mock_analysis.elements.symbols = [mock_symbol]

        test_image = Image.new("L", (200, 200), color=255)
        mock_calibration = Mock()

        with patch(
            "aec_agent.mcp.tools.gemini_first.adaptive_extraction.extract_symbols_direct",
            return_value=[],  # Empty for simplicity
        ):
            result = await pipeline._stage5_recognize_symbols(
                test_image,
                mock_analysis,
                mock_calibration,
            )

        assert result.success is True
        assert result.data["method"] == "hardcoded"


class TestStage6Validation:
    """Test Stage 6: Validation & Correction."""

    @pytest.mark.asyncio
    async def test_validation_disabled_options(self):
        """Test validation with all options disabled."""
        config = BestPracticesConfig(
            straighten_lines=False,
            connect_endpoints=False,
            snap_to_grid=False,
            gemini_visual_qa=False,
        )
        pipeline = BestPracticesPipeline(config)

        # Create mock entities
        mock_entity = Mock()
        mock_entity.entity_type = Mock()
        mock_entity.entity_type.value = "LINE"
        mock_entity.layer = "A-WALL"
        mock_entity.properties = {"start": (0, 0), "end": (100, 2)}

        test_image = Image.new("L", (100, 100), color=255)
        mock_analysis = Mock()

        result = await pipeline._stage6_validate(
            [mock_entity],
            test_image,
            mock_analysis,
        )

        assert result.success is True
        assert result.stage == "6_validate"
        assert "entities" in result.data
        assert len(result.data["entities"]) == 1


# =============================================================================
# Convenience Function Tests
# =============================================================================


class TestConvenienceFunctions:
    """Test convenience functions."""

    @pytest.mark.asyncio
    async def test_run_best_practices_pipeline_file_not_found(self):
        """Test pipeline fails gracefully for missing file."""
        result = await run_best_practices_pipeline(
            pdf_path="/nonexistent/file.pdf",
        )

        assert result.success is False
        # Should fail at stage 1

    def test_run_best_practices_pipeline_sync_exists(self):
        """Test sync wrapper function exists and is callable."""
        assert callable(run_best_practices_pipeline_sync)


# =============================================================================
# Integration Tests (Mocked)
# =============================================================================


class TestPipelineIntegration:
    """Integration tests for the full pipeline (mocked)."""

    @pytest.mark.asyncio
    async def test_full_pipeline_mock(self):
        """Test full pipeline execution with mocked stages."""
        config = BestPracticesConfig(
            symbol_method=SymbolRecognitionMethod.HARDCODED,
            super_resolution=False,
            gemini_visual_qa=False,
        )
        pipeline = BestPracticesPipeline(config)

        # Create mock components
        mock_image = Image.new("L", (1000, 800), color=255)
        mock_render_result = Mock()
        mock_render_result.image_path = Path("/tmp/test.png")

        mock_analysis = Mock()
        mock_analysis.drawing_type = "floor_plan"
        mock_analysis.scale = "1/4\" = 1'-0\""
        mock_analysis.total_elements = 10
        mock_analysis.elements = Mock()
        mock_analysis.elements.symbols = []

        mock_calibration = Mock()
        mock_calibration.scale_factor = 48.0
        mock_calibration.units = "inch"
        mock_calibration.to_dwg = Mock(return_value=(0, 0))

        mock_entity = Mock()
        mock_entity.entity_type = Mock()
        mock_entity.entity_type.value = "LINE"
        mock_entity.layer = "A-WALL"
        mock_entity.properties = {"start": (0, 0), "end": (100, 0)}

        mock_extraction = Mock()
        mock_extraction.entities = [mock_entity]

        # Patch all external calls
        with patch.object(
            pipeline, "_stage1_render_pdf",
            new_callable=AsyncMock,
            return_value=StageResult(
                stage="1_render_pdf",
                success=True,
                duration_ms=100,
                data={"image_path": Path("/tmp/test.png"), "image": mock_image, "dpi": 300},
            ),
        ), patch.object(
            pipeline, "_stage2_preprocess",
            new_callable=AsyncMock,
            return_value=StageResult(
                stage="2_preprocess",
                success=True,
                duration_ms=50,
                data={"processed_image": mock_image},
            ),
        ), patch.object(
            pipeline, "_stage3_analyze",
            new_callable=AsyncMock,
            return_value=StageResult(
                stage="3_gemini_analysis",
                success=True,
                duration_ms=200,
                data={"analysis": mock_analysis, "calibration": mock_calibration},
            ),
        ), patch.object(
            pipeline, "_stage4_extract",
            new_callable=AsyncMock,
            return_value=StageResult(
                stage="4_extract",
                success=True,
                duration_ms=150,
                data={"entities": [mock_entity]},
            ),
        ), patch.object(
            pipeline, "_stage5_recognize_symbols",
            new_callable=AsyncMock,
            return_value=StageResult(
                stage="5_symbol_rag",
                success=True,
                duration_ms=100,
                data={"symbol_entities": [], "recognized_symbols": []},
            ),
        ), patch.object(
            pipeline, "_stage6_validate",
            new_callable=AsyncMock,
            return_value=StageResult(
                stage="6_validate",
                success=True,
                duration_ms=50,
                data={"entities": [mock_entity]},
            ),
        ), patch.object(
            pipeline, "_stage7_output",
            new_callable=AsyncMock,
            return_value=StageResult(
                stage="7_output",
                success=True,
                duration_ms=20,
                data={"entity_count": 1},
            ),
        ), patch(
            "pathlib.Path.exists",
            return_value=True,
        ), patch(
            "pathlib.Path.mkdir",
        ):
            result = await pipeline.process_pdf(
                pdf_path="/tmp/test.pdf",
                page=1,
            )

        assert result.success is True
        assert len(result.stages) == 7
        assert all(s.success for s in result.stages)
        assert result.entity_count == 1


# =============================================================================
# Edge Cases
# =============================================================================


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_empty_entities_list(self):
        """Test handling of empty entities list."""
        result = PipelineResult(
            success=True,
            pdf_path="/path/to/file.pdf",
            page=1,
            entities=[],
        )

        assert result.entity_count == 0
        assert result.line_count == 0

    @pytest.mark.asyncio
    async def test_stage_failure_stops_pipeline(self):
        """Test that stage failure stops pipeline execution."""
        config = BestPracticesConfig()
        pipeline = BestPracticesPipeline(config)

        # Make stage 1 fail
        with patch.object(
            pipeline, "_stage1_render_pdf",
            new_callable=AsyncMock,
            return_value=StageResult(
                stage="1_render_pdf",
                success=False,
                duration_ms=10,
                errors=["PDF file corrupted"],
            ),
        ), patch.object(
            pipeline, "initialize",
            new_callable=AsyncMock,
        ), patch(
            "pathlib.Path.exists",
            return_value=True,
        ), patch(
            "pathlib.Path.mkdir",
        ):
            result = await pipeline.process_pdf(
                pdf_path="/tmp/test.pdf",
                page=1,
            )

        assert result.success is False
        assert len(result.stages) == 1  # Only stage 1 ran
        assert "PDF file corrupted" in result.stages[0].errors

    def test_config_immutability(self):
        """Test that config can be customized independently."""
        config1 = BestPracticesConfig(dpi=300)
        config2 = BestPracticesConfig(dpi=600)

        assert config1.dpi != config2.dpi
