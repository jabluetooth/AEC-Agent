"""
YOLOv8 Symbol Detection for AEC Drawings.

This module provides neural network-based symbol detection using YOLOv8,
offering more robust detection compared to template matching. It can handle
variations in drawing styles, rotations, and scales.

Part of Phase 2.5.1: LLM-Enhanced Vectorization Pipeline.

Supports two inference backends:
1. Ultralytics (native) - requires ultralytics package
2. ONNX Runtime - for deployment without heavy dependencies

Usage:
    >>> detector = YOLOSymbolDetector("models/mep_symbols.onnx")
    >>> blocks = detector.detect(image, scale=1/300)
    >>> for block in blocks:
    ...     print(f"{block.block_name} at {block.position}")
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import structlog

# Re-export DetectedBlock from symbol_detection for consistency
from .symbol_detection import DetectedBlock

logger = structlog.get_logger(__name__)


# Default class mapping for MEP symbols
# Maps YOLO class indices to (block_name, category)
DEFAULT_CLASS_MAP: dict[int, tuple[str, str]] = {
    0: ("VALVE-GATE", "mechanical"),
    1: ("VALVE-BALL", "mechanical"),
    2: ("VALVE-BUTTERFLY", "mechanical"),
    3: ("VALVE-CHECK", "mechanical"),
    4: ("DIFFUSER-SUPPLY", "mechanical"),
    5: ("DIFFUSER-RETURN", "mechanical"),
    6: ("DIFFUSER-EXHAUST", "mechanical"),
    7: ("DUCT-ELBOW", "mechanical"),
    8: ("DUCT-TEE", "mechanical"),
    9: ("PUMP", "mechanical"),
    10: ("AHU", "mechanical"),
    11: ("FCU", "mechanical"),
    12: ("VAV-BOX", "mechanical"),
    13: ("OUTLET-DUPLEX", "electrical"),
    14: ("OUTLET-GFCI", "electrical"),
    15: ("OUTLET-QUAD", "electrical"),
    16: ("SWITCH-SINGLE", "electrical"),
    17: ("SWITCH-3WAY", "electrical"),
    18: ("SWITCH-DIMMER", "electrical"),
    19: ("PANEL-ELECTRICAL", "electrical"),
    20: ("LIGHT-FIXTURE", "electrical"),
    21: ("LIGHT-RECESSED", "electrical"),
    22: ("LIGHT-EXIT", "electrical"),
    23: ("JUNCTION-BOX", "electrical"),
    24: ("SMOKE-DETECTOR", "fire"),
    25: ("HEAT-DETECTOR", "fire"),
    26: ("PULL-STATION", "fire"),
    27: ("HORN-STROBE", "fire"),
    28: ("SPRINKLER-HEAD", "fire"),
    29: ("FIRE-EXTINGUISHER", "fire"),
    30: ("FLOOR-DRAIN", "plumbing"),
    31: ("CLEANOUT", "plumbing"),
    32: ("FIXTURE-SINK", "plumbing"),
    33: ("FIXTURE-TOILET", "plumbing"),
    34: ("FIXTURE-URINAL", "plumbing"),
    35: ("WATER-HEATER", "plumbing"),
    36: ("DATA-OUTLET", "low_voltage"),
    37: ("CAMERA-SECURITY", "low_voltage"),
    38: ("CARD-READER", "low_voltage"),
    39: ("SPEAKER", "low_voltage"),
    40: ("THERMOSTAT", "low_voltage"),
    41: ("UNKNOWN", "unknown"),  # Catch-all for unrecognized symbols
}


@dataclass
class YOLODetection:
    """Raw YOLO detection result before conversion to DetectedBlock."""

    class_id: int
    confidence: float
    bbox: tuple[float, float, float, float]  # x1, y1, x2, y2 in pixels
    class_name: str = ""


class YOLOSymbolDetector:
    """
    YOLOv8-based symbol detector for AEC drawings.

    Supports both native ultralytics models and ONNX exports.

    Args:
        model_path: Path to the model file (.pt for ultralytics, .onnx for ONNX Runtime)
        class_map: Optional mapping of class indices to (block_name, category).
                   If None, uses DEFAULT_CLASS_MAP.
        device: Device to run inference on ('cpu', 'cuda', 'mps'). Auto-detected if None.
        conf_threshold: Minimum confidence for detections (0-1).
        iou_threshold: IoU threshold for NMS (0-1).

    Example:
        >>> detector = YOLOSymbolDetector("models/mep_symbols.onnx")
        >>> blocks = detector.detect(binary_image, scale=1/300)
    """

    def __init__(
        self,
        model_path: str,
        class_map: dict[int, tuple[str, str]] | None = None,
        device: str | None = None,
        conf_threshold: float = 0.5,
        iou_threshold: float = 0.45,
    ):
        self.model_path = Path(model_path)
        self.class_map = class_map or DEFAULT_CLASS_MAP
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.device = device

        # Determine backend based on file extension
        self._backend: str | None = None
        self._model: Any = None
        self._onnx_session: Any = None
        self._input_size: tuple[int, int] = (640, 640)  # YOLO default

        if not self.model_path.exists():
            logger.warning(
                "YOLO model not found, detection will be skipped",
                path=str(self.model_path),
            )
            return

        suffix = self.model_path.suffix.lower()
        if suffix == ".onnx":
            self._init_onnx()
        elif suffix in (".pt", ".pth"):
            self._init_ultralytics()
        else:
            logger.error(
                "Unsupported model format",
                path=str(self.model_path),
                suffix=suffix,
            )

    def _init_ultralytics(self) -> None:
        """Initialize ultralytics YOLO model."""
        try:
            from ultralytics import YOLO

            self._model = YOLO(str(self.model_path))
            self._backend = "ultralytics"
            logger.info(
                "YOLO model loaded (ultralytics)",
                path=str(self.model_path),
                device=self.device or "auto",
            )
        except ImportError:
            logger.warning(
                "ultralytics not installed, YOLO detection unavailable. "
                "Install with: pip install ultralytics"
            )
        except Exception as e:
            logger.error(
                "Failed to load YOLO model",
                path=str(self.model_path),
                error=str(e),
            )

    def _init_onnx(self) -> None:
        """Initialize ONNX Runtime session."""
        try:
            import onnxruntime as ort

            # Configure providers (prefer CUDA if available)
            providers = ["CPUExecutionProvider"]
            if self.device == "cuda":
                providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
            elif self.device is None:
                # Auto-detect
                available = ort.get_available_providers()
                if "CUDAExecutionProvider" in available:
                    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]

            self._onnx_session = ort.InferenceSession(
                str(self.model_path),
                providers=providers,
            )

            # Get input shape
            input_info = self._onnx_session.get_inputs()[0]
            if len(input_info.shape) == 4:
                self._input_size = (input_info.shape[2], input_info.shape[3])

            self._backend = "onnx"
            logger.info(
                "YOLO model loaded (ONNX Runtime)",
                path=str(self.model_path),
                providers=providers,
                input_size=self._input_size,
            )
        except ImportError:
            logger.warning(
                "onnxruntime not installed, ONNX inference unavailable. "
                "Install with: pip install onnxruntime"
            )
        except Exception as e:
            logger.error(
                "Failed to load ONNX model",
                path=str(self.model_path),
                error=str(e),
            )

    @property
    def is_available(self) -> bool:
        """Check if the detector is ready for inference."""
        return self._backend is not None

    def detect(
        self,
        image: np.ndarray,
        scale: float = 1.0,
        mask_detections: bool = False,
    ) -> tuple[np.ndarray, list[DetectedBlock]]:
        """
        Detect symbols in the image using YOLO.

        Args:
            image: Input image (grayscale or BGR). White = background, black = ink.
            scale: Coordinate scale factor (pixel to drawing units). Typically 1/DPI.
            mask_detections: If True, mask detected regions in the returned image.

        Returns:
            Tuple of:
            - masked_image: Image with detected regions optionally masked
            - detected_blocks: List of DetectedBlock for AutoCAD insertion
        """
        if not self.is_available:
            logger.debug("YOLO detector not available, skipping")
            return image, []

        # Ensure image is in correct format
        if len(image.shape) == 2:
            # Grayscale to BGR for YOLO
            import cv2

            image_bgr = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        else:
            image_bgr = image

        height, width = image.shape[:2]

        # Run inference
        if self._backend == "ultralytics":
            raw_detections = self._detect_ultralytics(image_bgr)
        elif self._backend == "onnx":
            raw_detections = self._detect_onnx(image_bgr)
        else:
            return image, []

        # Convert to DetectedBlock
        blocks: list[DetectedBlock] = []
        masked_image = image.copy() if mask_detections else image

        for det in raw_detections:
            # Get block name and category from class map
            if det.class_id in self.class_map:
                block_name, category = self.class_map[det.class_id]
            else:
                block_name = f"SYMBOL-{det.class_id}"
                category = "unknown"

            # Calculate center position
            x1, y1, x2, y2 = det.bbox
            cx = (x1 + x2) / 2
            cy = (y1 + y2) / 2

            # Convert to drawing units (Y-flip for AutoCAD)
            pos_x = cx * scale
            pos_y = (height - cy) * scale

            # Estimate rotation from aspect ratio (simple heuristic)
            bbox_w = x2 - x1
            bbox_h = y2 - y1
            if bbox_h > bbox_w * 1.5:
                rotation = 90.0
            elif bbox_w > bbox_h * 1.5:
                rotation = 0.0
            else:
                rotation = 0.0  # Square-ish, assume upright

            blocks.append(
                DetectedBlock(
                    block_name=block_name,
                    position=(pos_x, pos_y),
                    scale=1.0,
                    rotation=rotation,
                    confidence=det.confidence,
                    category=category,
                )
            )

            # Mask the detection region if requested
            if mask_detections:
                import cv2

                # Determine background value
                bg_value = 255 if np.mean(image) > 128 else 0
                x1_int, y1_int = int(x1), int(y1)
                x2_int, y2_int = int(x2), int(y2)
                # Add padding
                pad = 2
                x1_int = max(0, x1_int - pad)
                y1_int = max(0, y1_int - pad)
                x2_int = min(width, x2_int + pad)
                y2_int = min(height, y2_int + pad)
                masked_image[y1_int:y2_int, x1_int:x2_int] = bg_value

        logger.info(
            "YOLO symbol detection complete",
            backend=self._backend,
            symbols_found=len(blocks),
        )

        return masked_image, blocks

    def _detect_ultralytics(self, image: np.ndarray) -> list[YOLODetection]:
        """Run inference using ultralytics YOLO."""
        results = self._model(
            image,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            device=self.device,
            verbose=False,
        )

        detections: list[YOLODetection] = []

        for result in results:
            if result.boxes is None:
                continue

            for box in result.boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                conf = float(box.conf[0].cpu().numpy())
                cls_id = int(box.cls[0].cpu().numpy())

                detections.append(
                    YOLODetection(
                        class_id=cls_id,
                        confidence=conf,
                        bbox=(float(x1), float(y1), float(x2), float(y2)),
                        class_name=result.names.get(cls_id, ""),
                    )
                )

        return detections

    def _detect_onnx(self, image: np.ndarray) -> list[YOLODetection]:
        """Run inference using ONNX Runtime."""
        import cv2

        # Preprocess image for YOLO
        orig_h, orig_w = image.shape[:2]
        target_h, target_w = self._input_size

        # Letterbox resize (maintain aspect ratio)
        scale_factor = min(target_w / orig_w, target_h / orig_h)
        new_w = int(orig_w * scale_factor)
        new_h = int(orig_h * scale_factor)

        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        # Pad to target size
        pad_w = (target_w - new_w) // 2
        pad_h = (target_h - new_h) // 2
        padded = np.full((target_h, target_w, 3), 114, dtype=np.uint8)
        padded[pad_h : pad_h + new_h, pad_w : pad_w + new_w] = resized

        # Normalize and transpose to NCHW
        blob = padded.astype(np.float32) / 255.0
        blob = np.transpose(blob, (2, 0, 1))
        blob = np.expand_dims(blob, 0)

        # Run inference
        input_name = self._onnx_session.get_inputs()[0].name
        outputs = self._onnx_session.run(None, {input_name: blob})

        # Parse YOLO output format
        # YOLOv8 output: (1, num_classes + 4, num_detections)
        # Transpose to (num_detections, num_classes + 4)
        output = outputs[0]
        if output.shape[1] < output.shape[2]:
            output = np.transpose(output[0], (1, 0))
        else:
            output = output[0]

        detections: list[YOLODetection] = []

        for row in output:
            # First 4 values are bbox (cx, cy, w, h)
            # Remaining values are class scores
            cx, cy, w, h = row[:4]
            class_scores = row[4:]

            # Get best class
            class_id = int(np.argmax(class_scores))
            confidence = float(class_scores[class_id])

            if confidence < self.conf_threshold:
                continue

            # Convert from letterbox coords to original image coords
            x1 = (cx - w / 2 - pad_w) / scale_factor
            y1 = (cy - h / 2 - pad_h) / scale_factor
            x2 = (cx + w / 2 - pad_w) / scale_factor
            y2 = (cy + h / 2 - pad_h) / scale_factor

            # Clip to image bounds
            x1 = max(0, min(orig_w, x1))
            y1 = max(0, min(orig_h, y1))
            x2 = max(0, min(orig_w, x2))
            y2 = max(0, min(orig_h, y2))

            if x2 > x1 and y2 > y1:
                detections.append(
                    YOLODetection(
                        class_id=class_id,
                        confidence=confidence,
                        bbox=(x1, y1, x2, y2),
                    )
                )

        # Apply NMS
        if detections:
            detections = self._nms(detections)

        return detections

    def _nms(self, detections: list[YOLODetection]) -> list[YOLODetection]:
        """Apply Non-Maximum Suppression to detections."""
        if not detections:
            return []

        # Sort by confidence descending
        detections = sorted(detections, key=lambda x: x.confidence, reverse=True)
        kept: list[YOLODetection] = []

        for det in detections:
            # Check IoU with kept detections
            should_keep = True
            for kept_det in kept:
                iou = self._compute_iou(det.bbox, kept_det.bbox)
                if iou > self.iou_threshold:
                    should_keep = False
                    break

            if should_keep:
                kept.append(det)

        return kept

    @staticmethod
    def _compute_iou(
        box1: tuple[float, float, float, float],
        box2: tuple[float, float, float, float],
    ) -> float:
        """Compute Intersection over Union between two boxes."""
        x1_1, y1_1, x2_1, y2_1 = box1
        x1_2, y1_2, x2_2, y2_2 = box2

        # Intersection
        xi1 = max(x1_1, x1_2)
        yi1 = max(y1_1, y1_2)
        xi2 = min(x2_1, x2_2)
        yi2 = min(y2_1, y2_2)

        inter_w = max(0, xi2 - xi1)
        inter_h = max(0, yi2 - yi1)
        inter_area = inter_w * inter_h

        # Union
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        union_area = area1 + area2 - inter_area

        if union_area == 0:
            return 0.0

        return inter_area / union_area


# Global detector instance (lazy initialization)
_detector: YOLOSymbolDetector | None = None


def get_yolo_detector(
    model_path: str | None = None,
    **kwargs: Any,
) -> YOLOSymbolDetector | None:
    """
    Get or create the global YOLO detector instance.

    Args:
        model_path: Path to model file. If None, uses default location.
        **kwargs: Additional arguments passed to YOLOSymbolDetector.

    Returns:
        YOLOSymbolDetector instance, or None if model not found.
    """
    global _detector

    if model_path is None:
        # Default model location
        this_file = Path(__file__)
        default_path = this_file.parent.parent.parent.parent.parent / "models" / "mep_symbols.onnx"
        model_path = str(default_path)

    # Create new detector if path changed or first call
    if _detector is None or str(_detector.model_path) != model_path:
        _detector = YOLOSymbolDetector(model_path, **kwargs)

    return _detector if _detector.is_available else None


def detect_symbols_yolo(
    image: np.ndarray,
    scale: float = 1.0,
    model_path: str | None = None,
    confidence: float = 0.5,
    iou_threshold: float = 0.45,
    mask_detections: bool = True,
    class_map: dict[int, tuple[str, str]] | None = None,
) -> tuple[np.ndarray, list[DetectedBlock]]:
    """
    Detect AEC symbols using YOLOv8.

    This is the main entry point for YOLO-based symbol detection, providing
    a similar interface to detect_and_mask_symbols() from the template
    matching module.

    Args:
        image: Binary image (white = background, black = ink).
        scale: Coordinate scale factor (pixel to drawing units). Typically 1/DPI.
        model_path: Path to YOLO model (.pt or .onnx). Uses default if None.
        confidence: Minimum confidence threshold (0-1).
        iou_threshold: IoU threshold for NMS (0-1).
        mask_detections: If True, mask detected symbols from the returned image.
        class_map: Optional custom class mapping. Uses DEFAULT_CLASS_MAP if None.

    Returns:
        Tuple of:
        - masked_image: Image with symbol regions optionally masked
        - detected_blocks: List of DetectedBlock for AutoCAD insertion

    Example:
        >>> masked, blocks = detect_symbols_yolo(binary_img, scale=1/300)
        >>> print(f"Found {len(blocks)} symbols")
        >>> for b in blocks:
        ...     print(f"  {b.block_name} ({b.confidence:.2f})")
    """
    detector = get_yolo_detector(
        model_path=model_path,
        conf_threshold=confidence,
        iou_threshold=iou_threshold,
        class_map=class_map,
    )

    if detector is None:
        logger.debug("YOLO detector not available, returning empty results")
        return image, []

    return detector.detect(image, scale=scale, mask_detections=mask_detections)
