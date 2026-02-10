"""
Tests for two-stage symbol classifier.

Phase C: Symbol Intelligence - Tests for symbol_classifier.py module.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest

from aec_agent.mcp.tools.document_classifier import DrawingType
from aec_agent.mcp.tools.semantic_ocr import AnnotationType, ParsedAnnotation
from aec_agent.mcp.tools.symbol_detection import DetectedBlock
from aec_agent.mcp.tools.symbol_classifier import (
    SmartSymbol,
    SymbolClassifier,
    YOLO_CONFIDENCE_THRESHOLD,
    GENERIC_CLASSES,
    detect_and_classify_symbols,
)


class TestSmartSymbol:
    """Tests for SmartSymbol dataclass."""

    def test_default_values(self):
        """Should have sensible defaults."""
        symbol = SmartSymbol(
            block_name="VALVE-GATE",
            position=(100.0, 200.0),
        )
        assert symbol.block_name == "VALVE-GATE"
        assert symbol.position == (100.0, 200.0)
        assert symbol.scale == 1.0
        assert symbol.rotation == 0.0
        assert symbol.confidence == 0.0
        assert symbol.classification_source == "yolo"
        assert symbol.associated_text == []
        assert symbol.specs == {}

    def test_full_symbol(self):
        """Should store all classification data."""
        symbol = SmartSymbol(
            block_name="P-VALV-GATE",
            position=(50.0, 75.0),
            scale=1.0,
            rotation=90.0,
            confidence=0.95,
            category="plumbing",
            type="valve",
            subtype="gate_valve",
            direction="up",
            size="3/4\"",
            system="domestic_cold_water",
            specs={"size_inches": 0.75},
            tag="V-101",
            classification_source="vision_llm",
            vision_features=["bowtie_shape"],
            reasoning="Gate valve identified by bowtie shape",
        )
        assert symbol.subtype == "gate_valve"
        assert symbol.system == "domestic_cold_water"
        assert symbol.classification_source == "vision_llm"

    def test_from_detected_block(self):
        """Should create SmartSymbol from DetectedBlock."""
        block = DetectedBlock(
            block_name="VALVE-GATE",
            position=(100.0, 200.0),
            scale=1.0,
            rotation=45.0,
            confidence=0.85,
            category="mechanical",
        )
        symbol = SmartSymbol.from_detected_block(block)

        assert symbol.block_name == "VALVE-GATE"
        assert symbol.position == (100.0, 200.0)
        assert symbol.rotation == 45.0
        assert symbol.confidence == 0.85
        assert symbol.category == "mechanical"
        assert symbol.type == "valve"  # Parsed from block_name
        assert symbol.classification_source == "yolo"
        assert symbol.original_detection is block


class TestGenericClasses:
    """Tests for GENERIC_CLASSES configuration."""

    def test_generic_classes_defined(self):
        """Should have common generic classes."""
        assert "valve" in GENERIC_CLASSES
        assert "outlet" in GENERIC_CLASSES
        assert "diffuser" in GENERIC_CLASSES
        assert "detector" in GENERIC_CLASSES

    def test_confidence_threshold(self):
        """Confidence threshold should be reasonable."""
        assert 0.7 <= YOLO_CONFIDENCE_THRESHOLD <= 0.95


class TestSymbolClassifier:
    """Tests for SymbolClassifier class."""

    def test_initialization_defaults(self):
        """Should initialize with sensible defaults."""
        classifier = SymbolClassifier()
        assert classifier.yolo_threshold == YOLO_CONFIDENCE_THRESHOLD
        assert classifier.text_distance == 100.0

    def test_initialization_custom_params(self):
        """Should accept custom parameters."""
        classifier = SymbolClassifier(
            yolo_confidence_threshold=0.7,
            vision_provider="gemini",
            text_association_distance=150.0,
        )
        assert classifier.yolo_threshold == 0.7
        assert classifier.vision_provider == "gemini"
        assert classifier.text_distance == 150.0

    def test_needs_vision_llm_low_confidence(self):
        """Should use Vision LLM for low confidence detections."""
        classifier = SymbolClassifier(
            yolo_confidence_threshold=0.85,
            enable_vision_llm=True,
        )
        # Mock the enable_vision_llm check
        classifier.enable_vision_llm = True

        low_conf_block = DetectedBlock(
            block_name="VALVE-GATE",
            position=(0, 0),
            confidence=0.7,
            category="mechanical",
        )
        assert classifier._needs_vision_llm(low_conf_block) is True

        high_conf_block = DetectedBlock(
            block_name="VALVE-GATE",
            position=(0, 0),
            confidence=0.95,
            category="mechanical",
        )
        # High confidence non-generic class doesn't need Vision LLM
        high_conf_block_specific = DetectedBlock(
            block_name="SPECIFIC-SYMBOL",
            position=(0, 0),
            confidence=0.95,
            category="mechanical",
        )
        assert classifier._needs_vision_llm(high_conf_block_specific) is False

    def test_needs_vision_llm_generic_class(self):
        """Should use Vision LLM for generic classes."""
        classifier = SymbolClassifier(enable_vision_llm=True)
        classifier.enable_vision_llm = True

        # Generic "valve" class needs subtyping
        valve_block = DetectedBlock(
            block_name="VALVE",
            position=(0, 0),
            confidence=0.95,  # High confidence
            category="mechanical",
        )
        assert classifier._needs_vision_llm(valve_block) is True

    def test_needs_vision_llm_disabled(self):
        """Should not use Vision LLM when disabled."""
        classifier = SymbolClassifier(enable_vision_llm=False)
        classifier.enable_vision_llm = False

        block = DetectedBlock(
            block_name="VALVE",
            position=(0, 0),
            confidence=0.5,
            category="mechanical",
        )
        assert classifier._needs_vision_llm(block) is False

    def test_find_nearby_text(self):
        """Should find text annotations near a position."""
        classifier = SymbolClassifier(text_association_distance=100.0)

        annotations = [
            ParsedAnnotation(
                raw_text="3/4\" GV",
                annotation_type=AnnotationType.SIZE,
                structured_data={"size": "3/4\""},
                confidence=0.9,
                position=(50.0, 50.0),  # Distance from (55, 52): sqrt(29) ≈ 5.39
            ),
            ParsedAnnotation(
                raw_text="DCW",
                annotation_type=AnnotationType.NOTE,
                structured_data={},
                confidence=0.8,
                position=(60.0, 55.0),  # Distance from (55, 52): sqrt(34) ≈ 5.83
            ),
            ParsedAnnotation(
                raw_text="FAR AWAY",
                annotation_type=AnnotationType.NOTE,
                structured_data={},
                confidence=0.8,
                position=(500.0, 500.0),  # Too far
            ),
        ]

        nearby = classifier._find_nearby_text((55.0, 52.0), annotations)

        assert len(nearby) == 2
        # Should be sorted by distance
        assert nearby[0].raw_text == "3/4\" GV"  # Closest (sqrt(29) ≈ 5.39)
        assert nearby[1].raw_text == "DCW"  # Second closest (sqrt(34) ≈ 5.83)

    def test_crop_symbol(self):
        """Should crop symbol region from image."""
        classifier = SymbolClassifier()

        # Create a test image
        image = np.zeros((300, 300), dtype=np.uint8)
        image[140:160, 140:160] = 255  # White square in center

        # Position in drawing units (with scale=1.0, center at 150, 150)
        crop = classifier._crop_symbol(
            image=image,
            position=(150.0, 150.0),  # In drawing units
            scale=1.0,
        )

        assert crop.shape[0] > 0
        assert crop.shape[1] > 0

    def test_update_block_name(self):
        """Should generate standard block name from classification."""
        classifier = SymbolClassifier()

        from aec_agent.mcp.tools.vision_llm import VisionClassificationResult

        result = VisionClassificationResult(
            category="mechanical",
            type="valve",
            subtype="gate_valve",
            confidence=0.9,
        )
        symbol = SmartSymbol(block_name="OLD", position=(0, 0))

        new_name = classifier._update_block_name(symbol, result)
        assert "VALV" in new_name
        assert "GATE" in new_name

    def test_get_specialized_prompt_valve(self):
        """Should return specialized prompt for valve."""
        classifier = SymbolClassifier()
        prompt = classifier._get_specialized_prompt("valve", ["3/4\" GV"])
        assert prompt is not None
        assert "valve" in prompt.lower() or "Gate" in prompt

    def test_get_specialized_prompt_diffuser(self):
        """Should return specialized prompt for diffuser."""
        classifier = SymbolClassifier()
        prompt = classifier._get_specialized_prompt("diffuser", ["24x24"])
        assert prompt is not None
        assert "diffuser" in prompt.lower() or "grille" in prompt.lower()

    def test_get_specialized_prompt_unknown(self):
        """Should return None for unknown types."""
        classifier = SymbolClassifier()
        prompt = classifier._get_specialized_prompt("unknown_type", [])
        assert prompt is None


class TestSymbolClassifierAsync:
    """Async tests for SymbolClassifier."""

    @pytest.fixture
    def mock_image(self):
        """Create a mock test image."""
        return np.zeros((300, 300), dtype=np.uint8)

    @pytest.mark.asyncio
    async def test_classify_symbols_basic(self, mock_image):
        """Should classify symbols without Vision LLM."""
        classifier = SymbolClassifier(enable_vision_llm=False)

        # Mock detect_symbols to return some blocks
        mock_blocks = [
            DetectedBlock(
                block_name="VALVE-GATE",
                position=(100.0, 100.0),
                confidence=0.9,
                category="mechanical",
            ),
            DetectedBlock(
                block_name="OUTLET-DUPLEX",
                position=(200.0, 200.0),
                confidence=0.85,
                category="electrical",
            ),
        ]

        with patch('aec_agent.mcp.tools.symbol_classifier.detect_symbols') as mock_detect:
            mock_detect.return_value = (mock_image, mock_blocks)

            masked, symbols = await classifier.classify_symbols(
                image=mock_image,
                drawing_type=DrawingType.MEP_PLAN,
                scale=1.0,
            )

        assert len(symbols) == 2
        assert all(isinstance(s, SmartSymbol) for s in symbols)
        assert symbols[0].block_name == "VALVE-GATE"
        assert symbols[1].block_name == "OUTLET-DUPLEX"
        assert all(s.classification_source == "yolo" for s in symbols)

    @pytest.mark.asyncio
    async def test_classify_symbols_with_text_association(self, mock_image):
        """Should associate nearby text with symbols."""
        classifier = SymbolClassifier(enable_vision_llm=False)

        mock_blocks = [
            DetectedBlock(
                block_name="VALVE",
                position=(100.0, 100.0),
                confidence=0.9,
                category="mechanical",
            ),
        ]

        annotations = [
            ParsedAnnotation(
                raw_text="3/4\" GV",
                annotation_type=AnnotationType.SIZE,
                structured_data={"size": "3/4\""},
                confidence=0.9,
                position=(105.0, 105.0),  # Very close
            ),
        ]

        with patch('aec_agent.mcp.tools.symbol_classifier.detect_symbols') as mock_detect:
            mock_detect.return_value = (mock_image, mock_blocks)

            masked, symbols = await classifier.classify_symbols(
                image=mock_image,
                drawing_type=DrawingType.PLUMBING_PLAN,
                parsed_annotations=annotations,
            )

        assert len(symbols) == 1
        assert len(symbols[0].associated_text) > 0
        assert symbols[0].specs.get("size") == "3/4\""

    @pytest.mark.asyncio
    async def test_classify_symbols_with_vision_llm(self, mock_image):
        """Should use Vision LLM for low confidence symbols."""
        # Create a larger image to avoid crop issues
        large_mock_image = np.zeros((300, 300), dtype=np.uint8)

        classifier = SymbolClassifier(
            enable_vision_llm=True,
            yolo_confidence_threshold=0.85,
        )
        classifier.enable_vision_llm = True

        mock_blocks = [
            DetectedBlock(
                block_name="VALVE",
                position=(150.0, 150.0),  # Center of image
                confidence=0.6,  # Low confidence - will trigger Vision LLM
                category="mechanical",
            ),
        ]

        # Mock Vision LLM result
        from aec_agent.mcp.tools.vision_llm import VisionClassificationResult
        mock_vision_result = VisionClassificationResult(
            category="mechanical",
            type="valve",
            subtype="gate_valve",
            direction="up",
            confidence=0.95,
            visual_features=["bowtie_shape"],
            reasoning="Gate valve identified",
            provider="gemini",
        )

        # Mock the vision classifier
        mock_vision_classifier = AsyncMock()
        mock_vision_classifier.classify_symbol = AsyncMock(return_value=mock_vision_result)

        with patch('aec_agent.mcp.tools.symbol_classifier.detect_symbols') as mock_detect:
            mock_detect.return_value = (large_mock_image, mock_blocks)

            with patch.object(classifier, '_get_vision_classifier', return_value=mock_vision_classifier):
                masked, symbols = await classifier.classify_symbols(
                    image=large_mock_image,
                    drawing_type=DrawingType.PLUMBING_PLAN,
                    scale=1.0,
                )

        assert len(symbols) == 1
        # Verify Vision LLM was used
        mock_vision_classifier.classify_symbol.assert_called_once()
        assert symbols[0].subtype == "gate_valve"
        assert symbols[0].classification_source == "vision_llm"
        assert symbols[0].confidence == 0.95

    @pytest.mark.asyncio
    async def test_classify_symbols_empty_results(self, mock_image):
        """Should handle empty detection results."""
        classifier = SymbolClassifier(enable_vision_llm=False)

        with patch('aec_agent.mcp.tools.symbol_classifier.detect_symbols') as mock_detect:
            mock_detect.return_value = (mock_image, [])

            masked, symbols = await classifier.classify_symbols(
                image=mock_image,
                drawing_type=DrawingType.HVAC_PLAN,
            )

        assert len(symbols) == 0


class TestDetectAndClassifySymbols:
    """Tests for detect_and_classify_symbols convenience function."""

    @pytest.fixture
    def mock_image(self):
        """Create a mock test image."""
        return np.zeros((200, 200), dtype=np.uint8)

    @pytest.mark.asyncio
    async def test_convenience_function(self, mock_image):
        """Should work as a convenience wrapper."""
        mock_blocks = [
            DetectedBlock(
                block_name="DIFFUSER-SUPPLY",
                position=(100.0, 100.0),
                confidence=0.9,
                category="mechanical",
            ),
        ]

        with patch('aec_agent.mcp.tools.symbol_classifier.detect_symbols') as mock_detect:
            with patch('aec_agent.mcp.tools.symbol_classifier.is_vision_llm_available', return_value=False):
                mock_detect.return_value = (mock_image, mock_blocks)

                masked, symbols = await detect_and_classify_symbols(
                    image=mock_image,
                    drawing_type=DrawingType.HVAC_PLAN,
                    enable_vision_llm=False,
                )

        assert len(symbols) == 1
        assert symbols[0].category == "mechanical"


class TestSmartSymbolTextAssociation:
    """Tests for text association with SmartSymbol."""

    def test_extract_tag_from_associated_text(self):
        """Should extract equipment tag from associated text."""
        annotations = [
            ParsedAnnotation(
                raw_text="VAV-101",
                annotation_type=AnnotationType.EQUIPMENT_TAG,
                structured_data={"tag": "VAV-101"},
                confidence=0.95,
                position=(0, 0),
            ),
        ]

        symbol = SmartSymbol(
            block_name="VAV-BOX",
            position=(5.0, 5.0),
            associated_text=annotations,
        )

        # Extract tag manually (simulating what classifier does)
        for ann in symbol.associated_text:
            if ann.annotation_type == AnnotationType.EQUIPMENT_TAG:
                symbol.tag = ann.raw_text
                break

        assert symbol.tag == "VAV-101"

    def test_extract_specs_from_associated_text(self):
        """Should extract specs from associated text."""
        annotations = [
            ParsedAnnotation(
                raw_text="24x24 SA 200 CFM",
                annotation_type=AnnotationType.EQUIPMENT_SPEC,
                structured_data={
                    "width": 24,
                    "height": 24,
                    "cfm": 200,
                    "air_type": "supply",
                },
                confidence=0.9,
                position=(0, 0),
            ),
        ]

        symbol = SmartSymbol(
            block_name="DIFFUSER-SUPPLY",
            position=(5.0, 5.0),
            associated_text=annotations,
        )

        # Update specs (simulating what classifier does)
        for ann in symbol.associated_text:
            if ann.structured_data:
                symbol.specs.update(ann.structured_data)

        assert symbol.specs["cfm"] == 200
        assert symbol.specs["width"] == 24


class TestSymbolClassifierEdgeCases:
    """Edge case tests for SymbolClassifier."""

    @pytest.fixture
    def mock_image(self):
        """Create a mock test image."""
        return np.zeros((100, 100), dtype=np.uint8)

    @pytest.mark.asyncio
    async def test_handles_vision_llm_error(self, mock_image):
        """Should handle Vision LLM errors gracefully."""
        classifier = SymbolClassifier(enable_vision_llm=True)
        classifier.enable_vision_llm = True

        mock_blocks = [
            DetectedBlock(
                block_name="VALVE",
                position=(50.0, 50.0),
                confidence=0.5,
                category="mechanical",
            ),
        ]

        with patch('aec_agent.mcp.tools.symbol_classifier.detect_symbols') as mock_detect:
            mock_detect.return_value = (mock_image, mock_blocks)

            # Mock vision classifier that raises an error
            mock_vision_classifier = AsyncMock()
            mock_vision_classifier.classify_symbol = AsyncMock(side_effect=Exception("API Error"))

            with patch.object(classifier, '_get_vision_classifier', return_value=mock_vision_classifier):
                masked, symbols = await classifier.classify_symbols(
                    image=mock_image,
                    drawing_type=DrawingType.PLUMBING_PLAN,
                )

        # Should still return the symbol, just with YOLO classification
        assert len(symbols) == 1
        assert symbols[0].classification_source == "yolo"

    @pytest.mark.asyncio
    async def test_handles_empty_crop(self, mock_image):
        """Should handle edge case where symbol crop is empty."""
        classifier = SymbolClassifier(enable_vision_llm=True)
        classifier.enable_vision_llm = True

        # Symbol at edge of image
        mock_blocks = [
            DetectedBlock(
                block_name="VALVE",
                position=(0.0, 0.0),  # At corner
                confidence=0.5,
                category="mechanical",
            ),
        ]

        with patch('aec_agent.mcp.tools.symbol_classifier.detect_symbols') as mock_detect:
            mock_detect.return_value = (mock_image, mock_blocks)

            masked, symbols = await classifier.classify_symbols(
                image=mock_image,
                drawing_type=DrawingType.MEP_PLAN,
            )

        # Should still work despite potential crop issues
        assert len(symbols) == 1
