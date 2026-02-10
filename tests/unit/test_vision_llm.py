"""
Tests for Vision LLM integration.

Phase C: Symbol Intelligence - Tests for vision_llm.py module.
"""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest

from aec_agent.mcp.tools.vision_llm import (
    VisionProvider,
    VisionClassificationResult,
    VisionLLMClassifier,
    get_vision_classifier,
    is_vision_llm_available,
)


class TestVisionProvider:
    """Tests for VisionProvider enum."""

    def test_provider_values(self):
        """Should have all expected provider values."""
        assert VisionProvider.GEMINI.value == "gemini"
        assert VisionProvider.OPENAI.value == "openai"
        assert VisionProvider.ANTHROPIC.value == "anthropic"
        assert VisionProvider.AUTO.value == "auto"

    def test_provider_from_string(self):
        """Should create provider from string."""
        assert VisionProvider("gemini") == VisionProvider.GEMINI
        assert VisionProvider("openai") == VisionProvider.OPENAI


class TestVisionClassificationResult:
    """Tests for VisionClassificationResult dataclass."""

    def test_default_values(self):
        """Should have sensible defaults."""
        result = VisionClassificationResult(
            category="mechanical",
            type="valve",
            subtype="gate_valve",
        )
        assert result.category == "mechanical"
        assert result.type == "valve"
        assert result.subtype == "gate_valve"
        assert result.direction == "none"
        assert result.confidence == 0.0
        assert result.visual_features == []
        assert result.error is None

    def test_full_result(self):
        """Should store all classification data."""
        result = VisionClassificationResult(
            category="electrical",
            type="outlet",
            subtype="gfci_outlet",
            direction="down",
            size_hint="20A",
            system="power",
            confidence=0.95,
            visual_features=["wavy_line", "gfi_text"],
            reasoning="GFCI outlet identified by wavy line symbol",
            raw_response={"test": "data"},
            provider="gemini",
        )
        assert result.confidence == 0.95
        assert "wavy_line" in result.visual_features
        assert result.provider == "gemini"

    def test_error_result(self):
        """Should handle error results."""
        result = VisionClassificationResult(
            category="unknown",
            type="unknown",
            subtype="unknown",
            error="API rate limit exceeded",
        )
        assert result.error == "API rate limit exceeded"


class TestVisionLLMClassifier:
    """Tests for VisionLLMClassifier class."""

    def test_initialization_with_auto(self):
        """Should initialize with auto provider."""
        classifier = VisionLLMClassifier(provider=VisionProvider.AUTO)
        assert classifier.provider == VisionProvider.AUTO

    def test_initialization_with_string_provider(self):
        """Should accept string provider."""
        classifier = VisionLLMClassifier(provider="gemini")
        assert classifier.provider == VisionProvider.GEMINI

    def test_initialization_with_temperature(self):
        """Should accept temperature parameter."""
        classifier = VisionLLMClassifier(temperature=0.1)
        assert classifier.temperature == 0.1

    def test_parse_json_response_valid(self):
        """Should parse valid JSON response."""
        classifier = VisionLLMClassifier()
        json_text = '{"category": "mechanical", "subtype": "gate_valve"}'
        result = classifier._parse_json_response(json_text)
        assert result["category"] == "mechanical"
        assert result["subtype"] == "gate_valve"

    def test_parse_json_response_with_markdown(self):
        """Should handle JSON wrapped in markdown code blocks."""
        classifier = VisionLLMClassifier()

        # With ```json wrapper
        json_text = '```json\n{"category": "electrical"}\n```'
        result = classifier._parse_json_response(json_text)
        assert result["category"] == "electrical"

        # With ``` wrapper only
        json_text2 = '```\n{"type": "valve"}\n```'
        result2 = classifier._parse_json_response(json_text2)
        assert result2["type"] == "valve"

    def test_parse_json_response_invalid(self):
        """Should return empty dict for invalid JSON."""
        classifier = VisionLLMClassifier()
        result = classifier._parse_json_response("not valid json {")
        assert result == {}

    def test_image_to_base64(self):
        """Should convert numpy image to base64."""
        classifier = VisionLLMClassifier()

        # Create a simple test image
        image = np.zeros((100, 100), dtype=np.uint8)
        image[40:60, 40:60] = 255  # White square

        base64_str = classifier._image_to_base64(image)
        assert isinstance(base64_str, str)
        assert len(base64_str) > 0
        # PNG magic bytes in base64 start with "iVBOR"
        assert base64_str.startswith("iVBOR")


class TestVisionLLMClassifierAsync:
    """Async tests for VisionLLMClassifier."""

    @pytest.fixture
    def mock_image(self):
        """Create a mock test image."""
        return np.zeros((64, 64), dtype=np.uint8)

    @pytest.mark.asyncio
    async def test_classify_symbol_no_provider(self, mock_image):
        """Should return error when no provider available."""
        with patch.object(VisionLLMClassifier, '_get_available_provider', return_value=None):
            classifier = VisionLLMClassifier()
            result = await classifier.classify_symbol(
                image=mock_image,
                yolo_class="valve",
                yolo_category="mechanical",
                prompt="Test prompt",
            )
            assert result.error is not None
            assert "no" in result.error.lower() and "api" in result.error.lower()

    @pytest.mark.asyncio
    async def test_classify_symbol_with_gemini_mock(self, mock_image):
        """Should classify using Gemini (mocked)."""
        mock_response = MagicMock()
        mock_response.text = json.dumps({
            "category": "mechanical",
            "type": "valve",
            "subtype": "gate_valve",
            "direction": "up",
            "confidence": 0.92,
            "visual_features": ["bowtie_shape"],
            "reasoning": "Gate valve identified by bowtie shape",
        })

        mock_model = AsyncMock()
        mock_model.generate_content_async = AsyncMock(return_value=mock_response)

        classifier = VisionLLMClassifier(provider="gemini")
        classifier._gemini_model = mock_model

        with patch.object(classifier, '_get_available_provider', return_value=VisionProvider.GEMINI):
            result = await classifier.classify_symbol(
                image=mock_image,
                yolo_class="valve",
                yolo_category="mechanical",
                prompt="Test prompt",
                provider=VisionProvider.GEMINI,
            )

        assert result.subtype == "gate_valve"
        assert result.confidence == 0.92
        assert result.provider == "gemini"

    @pytest.mark.asyncio
    async def test_classify_symbol_with_openai_mock(self, mock_image):
        """Should classify using OpenAI (mocked)."""
        mock_response = MagicMock()
        mock_response.choices = [
            MagicMock(message=MagicMock(content=json.dumps({
                "category": "electrical",
                "type": "outlet",
                "subtype": "gfci_outlet",
                "confidence": 0.88,
            })))
        ]

        mock_client = AsyncMock()
        mock_client.chat.completions.create = AsyncMock(return_value=mock_response)

        classifier = VisionLLMClassifier(provider="openai")
        classifier._openai_client = mock_client

        with patch.object(classifier, '_get_available_provider', return_value=VisionProvider.OPENAI):
            result = await classifier.classify_symbol(
                image=mock_image,
                yolo_class="outlet",
                yolo_category="electrical",
                prompt="Test prompt",
                provider=VisionProvider.OPENAI,
            )

        assert result.subtype == "gfci_outlet"
        assert result.provider == "openai"

    @pytest.mark.asyncio
    async def test_classify_batch(self, mock_image):
        """Should classify multiple symbols."""
        classifier = VisionLLMClassifier()

        # Mock single classify to return different results
        call_count = 0
        async def mock_classify(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            return VisionClassificationResult(
                category="mechanical",
                type="valve",
                subtype=f"valve_{call_count}",
                confidence=0.9,
            )

        with patch.object(classifier, 'classify_symbol', side_effect=mock_classify):
            results = await classifier.classify_batch(
                images=[mock_image, mock_image, mock_image],
                yolo_classes=["valve", "valve", "valve"],
                yolo_categories=["mechanical", "mechanical", "mechanical"],
                prompts=["prompt1", "prompt2", "prompt3"],
            )

        assert len(results) == 3
        assert results[0].subtype == "valve_1"
        assert results[1].subtype == "valve_2"
        assert results[2].subtype == "valve_3"


class TestGetVisionClassifier:
    """Tests for get_vision_classifier helper."""

    def test_creates_classifier(self):
        """Should create classifier on first call."""
        # Clear the global instance
        import aec_agent.mcp.tools.vision_llm as vision_llm_module
        vision_llm_module._classifier = None

        classifier = get_vision_classifier(provider="auto")
        assert isinstance(classifier, VisionLLMClassifier)

    def test_returns_same_instance(self):
        """Should return same instance on subsequent calls."""
        classifier1 = get_vision_classifier()
        classifier2 = get_vision_classifier()
        assert classifier1 is classifier2


class TestIsVisionLLMAvailable:
    """Tests for is_vision_llm_available helper."""

    def test_available_with_gemini_key(self):
        """Should return True if Gemini API key is set."""
        mock_settings = MagicMock()
        mock_settings.gemini_api_key = "test-key"
        mock_settings.openai_api_key = None
        mock_settings.anthropic_api_key = None

        with patch('aec_agent.mcp.tools.vision_llm.get_settings', return_value=mock_settings):
            assert is_vision_llm_available() is True

    def test_available_with_openai_key(self):
        """Should return True if OpenAI API key is set."""
        mock_settings = MagicMock()
        mock_settings.gemini_api_key = None
        mock_settings.openai_api_key = "test-key"
        mock_settings.anthropic_api_key = None

        with patch('aec_agent.mcp.tools.vision_llm.get_settings', return_value=mock_settings):
            assert is_vision_llm_available() is True

    def test_available_with_anthropic_key(self):
        """Should return True if Anthropic API key is set."""
        mock_settings = MagicMock()
        mock_settings.gemini_api_key = None
        mock_settings.openai_api_key = None
        mock_settings.anthropic_api_key = "test-key"

        with patch('aec_agent.mcp.tools.vision_llm.get_settings', return_value=mock_settings):
            assert is_vision_llm_available() is True

    def test_not_available_without_keys(self):
        """Should return False if no API keys are set."""
        mock_settings = MagicMock()
        mock_settings.gemini_api_key = None
        mock_settings.openai_api_key = None
        mock_settings.anthropic_api_key = None

        with patch('aec_agent.mcp.tools.vision_llm.get_settings', return_value=mock_settings):
            assert is_vision_llm_available() is False


class TestVisionLLMClassifierProviderSelection:
    """Tests for provider auto-selection logic."""

    def test_auto_selects_gemini_first(self):
        """Should prefer Gemini when all providers available."""
        mock_settings = MagicMock()
        mock_settings.gemini_api_key = "gemini-key"
        mock_settings.openai_api_key = "openai-key"
        mock_settings.anthropic_api_key = "anthropic-key"

        with patch('aec_agent.mcp.tools.vision_llm.get_settings', return_value=mock_settings):
            classifier = VisionLLMClassifier(provider=VisionProvider.AUTO)
            provider = classifier._get_available_provider()
            assert provider == VisionProvider.GEMINI

    def test_auto_selects_openai_when_no_gemini(self):
        """Should use OpenAI when Gemini not available."""
        mock_settings = MagicMock()
        mock_settings.gemini_api_key = None
        mock_settings.openai_api_key = "openai-key"
        mock_settings.anthropic_api_key = None

        with patch('aec_agent.mcp.tools.vision_llm.get_settings', return_value=mock_settings):
            classifier = VisionLLMClassifier(provider=VisionProvider.AUTO)
            provider = classifier._get_available_provider()
            assert provider == VisionProvider.OPENAI

    def test_respects_explicit_provider(self):
        """Should use explicitly specified provider."""
        classifier = VisionLLMClassifier(provider=VisionProvider.ANTHROPIC)
        assert classifier.provider == VisionProvider.ANTHROPIC


class TestImageConversion:
    """Tests for image conversion utilities."""

    def test_grayscale_image(self):
        """Should handle grayscale images."""
        classifier = VisionLLMClassifier()
        gray = np.zeros((50, 50), dtype=np.uint8)
        base64_str = classifier._image_to_base64(gray)
        assert isinstance(base64_str, str)

    def test_rgb_image(self):
        """Should handle RGB images."""
        classifier = VisionLLMClassifier()
        rgb = np.zeros((50, 50, 3), dtype=np.uint8)
        base64_str = classifier._image_to_base64(rgb)
        assert isinstance(base64_str, str)

    def test_rgba_image(self):
        """Should handle RGBA images."""
        classifier = VisionLLMClassifier()
        rgba = np.zeros((50, 50, 4), dtype=np.uint8)
        base64_str = classifier._image_to_base64(rgba)
        assert isinstance(base64_str, str)

    def test_image_to_pil(self):
        """Should convert numpy to PIL Image."""
        classifier = VisionLLMClassifier()
        gray = np.zeros((50, 50), dtype=np.uint8)
        pil_img = classifier._image_to_pil(gray)

        from PIL import Image
        assert isinstance(pil_img, Image.Image)
        assert pil_img.size == (50, 50)
