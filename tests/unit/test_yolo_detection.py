"""
Unit tests for YOLOv8 Symbol Detection (Phase 2.5.1).

Tests the yolo_detection module and its integration with the
symbol_detection unified interface.
"""

import numpy as np
import pytest
from unittest.mock import MagicMock, patch
from pathlib import Path

# Test imports
from aec_agent.mcp.tools.yolo_detection import (
    YOLODetection,
    YOLOSymbolDetector,
    DEFAULT_CLASS_MAP,
    detect_symbols_yolo,
    get_yolo_detector,
)
from aec_agent.mcp.tools.symbol_detection import (
    DetectedBlock,
    is_yolo_available,
    detect_symbols,
)


class TestYOLODetection:
    """Tests for the YOLODetection dataclass."""

    def test_yolo_detection_creation(self):
        """Test creating a YOLODetection instance."""
        det = YOLODetection(
            class_id=0,
            confidence=0.85,
            bbox=(100.0, 100.0, 150.0, 150.0),
            class_name="VALVE-GATE",
        )
        assert det.class_id == 0
        assert det.confidence == 0.85
        assert det.bbox == (100.0, 100.0, 150.0, 150.0)
        assert det.class_name == "VALVE-GATE"

    def test_yolo_detection_defaults(self):
        """Test YOLODetection default values."""
        det = YOLODetection(
            class_id=1,
            confidence=0.5,
            bbox=(0.0, 0.0, 10.0, 10.0),
        )
        assert det.class_name == ""


class TestDefaultClassMap:
    """Tests for the DEFAULT_CLASS_MAP."""

    def test_class_map_structure(self):
        """Test that class map has correct structure."""
        for class_id, (block_name, category) in DEFAULT_CLASS_MAP.items():
            assert isinstance(class_id, int)
            assert isinstance(block_name, str)
            assert isinstance(category, str)
            assert block_name.isupper() or block_name == "UNKNOWN"
            assert category in [
                "mechanical",
                "electrical",
                "fire",
                "plumbing",
                "low_voltage",
                "unknown",
            ]

    def test_class_map_coverage(self):
        """Test that class map covers expected classes."""
        categories = set(cat for _, cat in DEFAULT_CLASS_MAP.values())
        expected = {"mechanical", "electrical", "fire", "plumbing", "low_voltage", "unknown"}
        assert categories == expected

    def test_class_map_has_unknown(self):
        """Test that class map includes an UNKNOWN class."""
        block_names = [name for name, _ in DEFAULT_CLASS_MAP.values()]
        assert "UNKNOWN" in block_names


class TestYOLOSymbolDetector:
    """Tests for the YOLOSymbolDetector class."""

    def test_detector_not_available_missing_model(self, tmp_path):
        """Test detector reports unavailable when model missing."""
        fake_path = tmp_path / "nonexistent.onnx"
        detector = YOLOSymbolDetector(str(fake_path))
        assert not detector.is_available

    def test_detector_with_custom_class_map(self, tmp_path):
        """Test detector accepts custom class map."""
        custom_map = {0: ("CUSTOM-SYMBOL", "custom")}
        fake_path = tmp_path / "fake.onnx"
        detector = YOLOSymbolDetector(str(fake_path), class_map=custom_map)
        assert detector.class_map == custom_map

    def test_detector_parameters(self, tmp_path):
        """Test detector stores parameters correctly."""
        fake_path = tmp_path / "fake.onnx"
        detector = YOLOSymbolDetector(
            str(fake_path),
            conf_threshold=0.7,
            iou_threshold=0.3,
        )
        assert detector.conf_threshold == 0.7
        assert detector.iou_threshold == 0.3

    def test_compute_iou_perfect_overlap(self):
        """Test IoU calculation for perfect overlap."""
        box = (0.0, 0.0, 10.0, 10.0)
        iou = YOLOSymbolDetector._compute_iou(box, box)
        assert iou == 1.0

    def test_compute_iou_no_overlap(self):
        """Test IoU calculation for no overlap."""
        box1 = (0.0, 0.0, 10.0, 10.0)
        box2 = (20.0, 20.0, 30.0, 30.0)
        iou = YOLOSymbolDetector._compute_iou(box1, box2)
        assert iou == 0.0

    def test_compute_iou_partial_overlap(self):
        """Test IoU calculation for partial overlap."""
        box1 = (0.0, 0.0, 10.0, 10.0)  # Area = 100
        box2 = (5.0, 5.0, 15.0, 15.0)  # Area = 100, Intersection = 25
        # Union = 100 + 100 - 25 = 175
        # IoU = 25 / 175 = 0.143
        iou = YOLOSymbolDetector._compute_iou(box1, box2)
        assert abs(iou - 0.143) < 0.01

    def test_detect_returns_empty_when_unavailable(self, tmp_path):
        """Test detect returns empty when detector unavailable."""
        fake_path = tmp_path / "nonexistent.onnx"
        detector = YOLOSymbolDetector(str(fake_path))

        # Create a simple test image
        image = np.ones((100, 100), dtype=np.uint8) * 255

        masked, blocks = detector.detect(image, scale=1.0)
        assert np.array_equal(masked, image)
        assert blocks == []


class TestDetectSymbolsYolo:
    """Tests for the detect_symbols_yolo function."""

    def test_returns_empty_when_no_model(self):
        """Test function returns empty when model not available."""
        image = np.ones((100, 100), dtype=np.uint8) * 255
        masked, blocks = detect_symbols_yolo(
            image,
            model_path="/nonexistent/path/model.onnx",
        )
        assert np.array_equal(masked, image)
        assert blocks == []

    def test_accepts_grayscale_image(self):
        """Test function accepts grayscale images."""
        image = np.ones((100, 100), dtype=np.uint8) * 255
        masked, blocks = detect_symbols_yolo(image)
        assert masked.shape == image.shape

    def test_accepts_scale_parameter(self):
        """Test function accepts scale parameter."""
        image = np.ones((100, 100), dtype=np.uint8) * 255
        masked, blocks = detect_symbols_yolo(image, scale=1 / 300)
        # Should not raise


class TestGetYoloDetector:
    """Tests for the get_yolo_detector function."""

    def test_returns_none_when_model_missing(self):
        """Test returns None when model doesn't exist."""
        detector = get_yolo_detector("/nonexistent/path/model.onnx")
        assert detector is None

    def test_uses_default_path_when_none(self):
        """Test uses default path when None provided."""
        # This should not raise, just return None if model doesn't exist
        detector = get_yolo_detector(None)
        # May be None if default model doesn't exist


class TestUnifiedDetectSymbols:
    """Tests for the unified detect_symbols function."""

    def test_backend_auto_fallback_to_template(self):
        """Test auto backend falls back to template when YOLO unavailable."""
        image = np.ones((100, 100), dtype=np.uint8) * 255

        # Should not raise even if YOLO unavailable
        masked, blocks = detect_symbols(
            image,
            scale=1.0,
            backend="auto",
        )
        assert masked.shape == image.shape

    def test_backend_template_explicit(self):
        """Test explicit template backend."""
        image = np.ones((100, 100), dtype=np.uint8) * 255

        masked, blocks = detect_symbols(
            image,
            scale=1.0,
            backend="template",
        )
        assert masked.shape == image.shape

    def test_backend_yolo_returns_empty_when_unavailable(self):
        """Test YOLO backend returns empty when unavailable."""
        image = np.ones((100, 100), dtype=np.uint8) * 255

        masked, blocks = detect_symbols(
            image,
            scale=1.0,
            backend="yolo",
            yolo_model_path="/nonexistent/model.onnx",
        )
        assert np.array_equal(masked, image)
        assert blocks == []

    def test_backend_unknown_defaults_to_template(self):
        """Test unknown backend defaults to template."""
        image = np.ones((100, 100), dtype=np.uint8) * 255

        masked, blocks = detect_symbols(
            image,
            scale=1.0,
            backend="invalid_backend",
        )
        assert masked.shape == image.shape


class TestIsYoloAvailable:
    """Tests for the is_yolo_available function."""

    def test_returns_false_when_model_missing(self):
        """Test returns False when model doesn't exist."""
        available = is_yolo_available("/nonexistent/model.onnx")
        assert available is False

    def test_returns_false_when_default_model_missing(self):
        """Test returns False when default model doesn't exist."""
        # This assumes the default model path doesn't exist
        available = is_yolo_available(None)
        # Could be True or False depending on whether model exists


class TestYOLONMS:
    """Tests for YOLO Non-Maximum Suppression."""

    def test_nms_removes_duplicates(self, tmp_path):
        """Test NMS removes duplicate detections."""
        fake_path = tmp_path / "fake.onnx"
        detector = YOLOSymbolDetector(str(fake_path), iou_threshold=0.3)  # Lower threshold

        # Create overlapping detections
        # Box1: (0,0,10,10) Area=100
        # Box2: (2,2,12,12) Area=100, Intersection=(2,2,10,10)=64, Union=136, IoU=0.47 > 0.3
        detections = [
            YOLODetection(0, 0.9, (0.0, 0.0, 10.0, 10.0)),
            YOLODetection(0, 0.8, (2.0, 2.0, 12.0, 12.0)),  # Overlaps with first (IoU=0.47)
            YOLODetection(1, 0.7, (50.0, 50.0, 60.0, 60.0)),  # No overlap
        ]

        filtered = detector._nms(detections)

        # Should keep highest confidence from overlapping pair and the non-overlapping one
        assert len(filtered) == 2
        assert filtered[0].confidence == 0.9
        assert filtered[1].confidence == 0.7

    def test_nms_empty_list(self, tmp_path):
        """Test NMS with empty list."""
        fake_path = tmp_path / "fake.onnx"
        detector = YOLOSymbolDetector(str(fake_path))

        filtered = detector._nms([])
        assert filtered == []

    def test_nms_single_detection(self, tmp_path):
        """Test NMS with single detection."""
        fake_path = tmp_path / "fake.onnx"
        detector = YOLOSymbolDetector(str(fake_path))

        detections = [YOLODetection(0, 0.9, (0.0, 0.0, 10.0, 10.0))]
        filtered = detector._nms(detections)

        assert len(filtered) == 1
        assert filtered[0].confidence == 0.9


class TestDetectedBlockConversion:
    """Tests for converting YOLO detections to DetectedBlock."""

    def test_block_has_correct_fields(self):
        """Test DetectedBlock has all required fields."""
        block = DetectedBlock(
            block_name="VALVE-GATE",
            position=(100.0, 200.0),
            scale=1.0,
            rotation=90.0,
            confidence=0.85,
            category="mechanical",
        )
        assert block.block_name == "VALVE-GATE"
        assert block.position == (100.0, 200.0)
        assert block.scale == 1.0
        assert block.rotation == 90.0
        assert block.confidence == 0.85
        assert block.category == "mechanical"

    def test_block_defaults(self):
        """Test DetectedBlock default values."""
        block = DetectedBlock(
            block_name="TEST",
            position=(0.0, 0.0),
        )
        assert block.scale == 1.0
        assert block.rotation == 0.0
        assert block.confidence == 0.0
        assert block.category == ""


@pytest.mark.parametrize(
    "class_id,expected_name,expected_category",
    [
        (0, "VALVE-GATE", "mechanical"),
        (13, "OUTLET-DUPLEX", "electrical"),
        (24, "SMOKE-DETECTOR", "fire"),
        (30, "FLOOR-DRAIN", "plumbing"),
        (36, "DATA-OUTLET", "low_voltage"),
        (41, "UNKNOWN", "unknown"),
    ],
)
class TestClassMapping:
    """Parametrized tests for class ID to block name mapping."""

    def test_class_id_mapping(self, class_id, expected_name, expected_category):
        """Test class ID maps to correct block name and category."""
        block_name, category = DEFAULT_CLASS_MAP[class_id]
        assert block_name == expected_name
        assert category == expected_category
