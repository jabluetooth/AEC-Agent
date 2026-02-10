"""
Two-Stage Symbol Classification for AEC Drawings.

This module provides the orchestration layer for two-stage symbol classification:
1. Stage 1: YOLOv8 detection for fast symbol localization and initial classification
2. Stage 2: Vision LLM for detailed subtype classification (on ambiguous symbols)

Part of Phase C: Symbol Intelligence - enhances detection with semantic understanding.

The classifier:
- Uses YOLO for fast detection
- Only invokes Vision LLM when confidence is low or class is ambiguous
- Associates nearby text annotations with symbols
- Returns SmartSymbol objects with full semantic data

Usage:
    >>> classifier = SymbolClassifier()
    >>> smart_symbols = await classifier.classify_symbols(
    ...     image=binary_image,
    ...     drawing_type=DrawingType.HVAC_PLAN,
    ...     parsed_annotations=annotations,
    ... )
"""

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import structlog

from aec_agent.prompts.symbol_classification import (
    SYMBOL_SUBTYPES,
    get_symbol_classification_prompt,
    get_unknown_symbol_prompt,
    get_valve_classification_prompt,
    get_diffuser_classification_prompt,
    get_outlet_classification_prompt,
    get_detector_classification_prompt,
)
from .document_classifier import DrawingType
from .semantic_ocr import ParsedAnnotation
from .symbol_detection import DetectedBlock, detect_symbols
from .vision_llm import (
    VisionLLMClassifier,
    VisionClassificationResult,
    get_vision_classifier,
    is_vision_llm_available,
)

logger = structlog.get_logger(__name__)


# Thresholds for deciding when to use Vision LLM
YOLO_CONFIDENCE_THRESHOLD = 0.85  # Use Vision LLM if below this
GENERIC_CLASSES = {
    "valve", "outlet", "switch", "light", "diffuser",
    "damper", "detector", "fixture", "equipment", "device",
}  # Classes that benefit from Vision LLM subtyping


@dataclass
class SmartSymbol:
    """
    Semantically-enriched symbol with full classification data.

    Extends DetectedBlock with detailed subtype information,
    associated text, and CAD standards data.

    Attributes:
        # From DetectedBlock
        block_name: AutoCAD block name to insert
        position: (x, y) insertion point in drawing units
        scale: Block scale factor
        rotation: Rotation in degrees
        confidence: Detection confidence (0-1)
        category: MEP category (mechanical, electrical, etc.)

        # Enhanced classification
        type: General type (valve, outlet, diffuser, etc.)
        subtype: Specific subtype (gate_valve, duplex_outlet, etc.)
        direction: Symbol orientation (up, down, left, right, none)
        size: Size from nearby text or visual analysis
        system: MEP system (domestic_cold_water, supply_air, etc.)

        # Associated data
        specs: Specifications from nearby text
        tag: Equipment tag from nearby text
        associated_text: List of associated text annotations

        # Classification metadata
        classification_source: "yolo", "vision_llm", or "hybrid"
        vision_features: Visual features identified by Vision LLM
        reasoning: Classification reasoning
    """
    # Core identification
    block_name: str
    position: tuple[float, float]
    scale: float = 1.0
    rotation: float = 0.0
    confidence: float = 0.0
    category: str = ""

    # Enhanced classification
    type: str = ""
    subtype: str = ""
    direction: str = "none"
    size: str | None = None
    system: str | None = None

    # Associated data
    specs: dict[str, Any] = field(default_factory=dict)
    tag: str | None = None
    associated_text: list[ParsedAnnotation] = field(default_factory=list)

    # Classification metadata
    classification_source: str = "yolo"  # yolo, vision_llm, hybrid
    vision_features: list[str] = field(default_factory=list)
    reasoning: str = ""

    # Original detection (for reference)
    original_detection: DetectedBlock | None = None
    bbox: tuple[float, float, float, float] | None = None  # x1, y1, x2, y2

    @classmethod
    def from_detected_block(cls, block: DetectedBlock) -> "SmartSymbol":
        """Create SmartSymbol from a DetectedBlock."""
        # Parse type from block_name (e.g., "VALVE-GATE" -> "valve")
        block_type = block.block_name.split("-")[0].lower() if "-" in block.block_name else block.block_name.lower()

        return cls(
            block_name=block.block_name,
            position=block.position,
            scale=block.scale,
            rotation=block.rotation,
            confidence=block.confidence,
            category=block.category,
            type=block_type,
            subtype=block.block_name.lower().replace("-", "_"),
            classification_source="yolo",
            original_detection=block,
        )


class SymbolClassifier:
    """
    Two-stage symbol classifier with YOLO + Vision LLM.

    Stage 1: YOLO detection
        - Fast detection of symbol locations
        - Initial classification into generic classes
        - High-confidence detections used directly

    Stage 2: Vision LLM (selective)
        - Called for low-confidence detections
        - Called for generic classes needing subtyping
        - Returns detailed subtype and specifications

    Args:
        yolo_confidence_threshold: Use Vision LLM if YOLO confidence below this.
        enable_vision_llm: Whether to use Vision LLM at all.
        vision_provider: Vision LLM provider (gemini, openai, anthropic, auto).
        text_association_distance: Max distance (pixels) for text association.

    Example:
        >>> classifier = SymbolClassifier()
        >>> smart_symbols = await classifier.classify_symbols(
        ...     image=binary_image,
        ...     drawing_type=DrawingType.HVAC_PLAN,
        ... )
    """

    def __init__(
        self,
        yolo_confidence_threshold: float = YOLO_CONFIDENCE_THRESHOLD,
        enable_vision_llm: bool = True,
        vision_provider: str = "auto",
        text_association_distance: float = 100.0,
    ):
        self.yolo_threshold = yolo_confidence_threshold
        self.enable_vision_llm = enable_vision_llm and is_vision_llm_available()
        self.vision_provider = vision_provider
        self.text_distance = text_association_distance

        self._vision_classifier: VisionLLMClassifier | None = None

    def _get_vision_classifier(self) -> VisionLLMClassifier:
        """Get or create the Vision LLM classifier."""
        if self._vision_classifier is None:
            self._vision_classifier = get_vision_classifier(
                provider=self.vision_provider,
            )
        return self._vision_classifier

    def _needs_vision_llm(self, block: DetectedBlock) -> bool:
        """Determine if a detection needs Vision LLM classification."""
        if not self.enable_vision_llm:
            return False

        # Low confidence detection
        if block.confidence < self.yolo_threshold:
            return True

        # Generic class that benefits from subtyping
        block_type = block.block_name.split("-")[0].lower() if "-" in block.block_name else block.block_name.lower()
        if block_type in GENERIC_CLASSES:
            return True

        return False

    def _get_specialized_prompt(
        self,
        block_type: str,
        nearby_text: list[str],
    ) -> str | None:
        """Get a specialized prompt for common symbol types."""
        if block_type == "valve":
            return get_valve_classification_prompt(
                valve_image_description="",  # Will be filled by Vision LLM
                nearby_text=nearby_text,
            )
        elif block_type == "diffuser":
            return get_diffuser_classification_prompt(nearby_text=nearby_text)
        elif block_type == "outlet":
            return get_outlet_classification_prompt(nearby_text=nearby_text)
        elif block_type in ("detector", "smoke", "heat"):
            return get_detector_classification_prompt(nearby_text=nearby_text)

        return None

    def _find_nearby_text(
        self,
        position: tuple[float, float],
        annotations: list[ParsedAnnotation],
        max_distance: float | None = None,
    ) -> list[ParsedAnnotation]:
        """Find text annotations near a symbol position."""
        if max_distance is None:
            max_distance = self.text_distance

        nearby = []
        px, py = position

        for ann in annotations:
            ax, ay = ann.position
            distance = ((px - ax) ** 2 + (py - ay) ** 2) ** 0.5

            if distance <= max_distance:
                nearby.append((distance, ann))

        # Sort by distance and return annotations
        nearby.sort(key=lambda x: x[0])
        return [ann for _, ann in nearby]

    def _crop_symbol(
        self,
        image: np.ndarray,
        position: tuple[float, float],
        scale: float,
        padding: int = 10,
    ) -> np.ndarray:
        """Crop the symbol region from the image."""
        height, width = image.shape[:2]

        # Convert drawing units back to pixels
        px = int(position[0] / scale)
        py = int(height - position[1] / scale)  # Y-flip

        # Estimate symbol size (typical symbol is 20-50 pixels)
        half_size = 30

        # Crop with padding
        x1 = max(0, px - half_size - padding)
        y1 = max(0, py - half_size - padding)
        x2 = min(width, px + half_size + padding)
        y2 = min(height, py + half_size + padding)

        return image[y1:y2, x1:x2]

    def _update_block_name(
        self,
        symbol: SmartSymbol,
        vision_result: VisionClassificationResult,
    ) -> str:
        """Generate standard block name from classification."""
        category = vision_result.category.upper()
        type_abbrev = {
            "valve": "VALV",
            "outlet": "OUTL",
            "switch": "SWCH",
            "light": "LITE",
            "diffuser": "DIFF",
            "grille": "GRIL",
            "damper": "DAMP",
            "detector": "DTCT",
            "fixture": "FIXT",
            "pump": "PUMP",
            "fan": "FAN",
            "panel": "PANL",
        }.get(vision_result.type.lower(), vision_result.type[:4].upper())

        subtype = vision_result.subtype.upper().replace("_", "-")

        # Generate block name like "M-VALV-GATE" or "E-OUTL-GFCI"
        prefix = {
            "mechanical": "M",
            "electrical": "E",
            "plumbing": "P",
            "fire": "FA",
            "low_voltage": "LV",
        }.get(category.lower(), "X")

        return f"{prefix}-{type_abbrev}-{subtype}"

    async def classify_symbols(
        self,
        image: np.ndarray,
        drawing_type: DrawingType = DrawingType.MEP_PLAN,
        scale: float = 1 / 300,
        parsed_annotations: list[ParsedAnnotation] | None = None,
        yolo_backend: str = "auto",
        yolo_model_path: str | None = None,
        yolo_confidence: float = 0.5,
    ) -> tuple[np.ndarray, list[SmartSymbol]]:
        """
        Detect and classify symbols using the two-stage pipeline.

        Args:
            image: Binary image (white background, black ink).
            drawing_type: Type of drawing for context.
            scale: Pixel to drawing units scale factor.
            parsed_annotations: Pre-parsed text annotations.
            yolo_backend: YOLO backend ("auto", "yolo", "template").
            yolo_model_path: Custom YOLO model path.
            yolo_confidence: YOLO detection confidence threshold.

        Returns:
            Tuple of (masked_image, list of SmartSymbol).
        """
        # Stage 1: YOLO detection
        masked_image, detected_blocks = detect_symbols(
            image=image,
            scale=scale,
            backend=yolo_backend,
            yolo_model_path=yolo_model_path,
            yolo_confidence=yolo_confidence,
            mask_detections=True,
        )

        if not detected_blocks:
            logger.debug("No symbols detected by YOLO")
            return masked_image, []

        logger.info(
            "Stage 1 complete: YOLO detection",
            symbols_found=len(detected_blocks),
        )

        # Convert to SmartSymbols
        smart_symbols: list[SmartSymbol] = []
        vision_llm_candidates: list[tuple[int, DetectedBlock]] = []

        for i, block in enumerate(detected_blocks):
            symbol = SmartSymbol.from_detected_block(block)

            # Associate nearby text
            if parsed_annotations:
                symbol.associated_text = self._find_nearby_text(
                    block.position, parsed_annotations
                )

                # Extract specs from associated text
                for ann in symbol.associated_text[:3]:  # Top 3 closest
                    if ann.structured_data:
                        symbol.specs.update(ann.structured_data)
                    if ann.annotation_type.value == "equipment_tag":
                        symbol.tag = ann.raw_text

            # Check if Vision LLM is needed
            if self._needs_vision_llm(block):
                vision_llm_candidates.append((i, block))

            smart_symbols.append(symbol)

        # Stage 2: Vision LLM for candidates
        if vision_llm_candidates and self.enable_vision_llm:
            logger.info(
                "Stage 2: Vision LLM classification",
                candidates=len(vision_llm_candidates),
            )

            classifier = self._get_vision_classifier()

            for idx, block in vision_llm_candidates:
                symbol = smart_symbols[idx]

                # Crop symbol from image
                try:
                    symbol_crop = self._crop_symbol(image, block.position, scale)
                    if symbol_crop.size == 0:
                        continue
                except Exception as e:
                    logger.warning("Failed to crop symbol", error=str(e))
                    continue

                # Get nearby text for context
                nearby_text = [ann.raw_text for ann in symbol.associated_text[:5]]

                # Get appropriate prompt
                block_type = symbol.type
                prompt = self._get_specialized_prompt(block_type, nearby_text)
                if prompt is None:
                    prompt = get_symbol_classification_prompt(
                        yolo_class=block_type,
                        yolo_category=block.category,
                        drawing_type=drawing_type,
                        nearby_text=nearby_text,
                        yolo_confidence=block.confidence,
                    )

                # Classify with Vision LLM
                try:
                    result = await classifier.classify_symbol(
                        image=symbol_crop,
                        yolo_class=block_type,
                        yolo_category=block.category,
                        prompt=prompt,
                    )

                    if result.error:
                        logger.warning(
                            "Vision LLM classification failed",
                            error=result.error,
                            symbol=block_type,
                        )
                        continue

                    # Update symbol with Vision LLM results
                    symbol.subtype = result.subtype
                    symbol.direction = result.direction
                    symbol.system = result.system
                    symbol.vision_features = result.visual_features
                    symbol.reasoning = result.reasoning
                    symbol.classification_source = "vision_llm"

                    # Use higher confidence from Vision LLM if available
                    if result.confidence > symbol.confidence:
                        symbol.confidence = result.confidence

                    # Update block name based on classification
                    symbol.block_name = self._update_block_name(symbol, result)

                    # Extract size from Vision LLM if not from text
                    if result.size_hint and not symbol.size:
                        symbol.size = result.size_hint

                    logger.debug(
                        "Vision LLM classification complete",
                        symbol=block_type,
                        subtype=result.subtype,
                        confidence=result.confidence,
                    )

                except Exception as e:
                    logger.warning(
                        "Vision LLM classification error",
                        error=str(e),
                        symbol=block_type,
                    )

        logger.info(
            "Two-stage classification complete",
            total_symbols=len(smart_symbols),
            vision_llm_classified=len([s for s in smart_symbols if s.classification_source == "vision_llm"]),
        )

        return masked_image, smart_symbols

    async def classify_unknown_region(
        self,
        image: np.ndarray,
        region: np.ndarray,
        position: tuple[float, float],
        drawing_type: DrawingType,
        annotations: list[ParsedAnnotation] | None = None,
    ) -> SmartSymbol | None:
        """
        Classify an unknown symbol region using Vision LLM only.

        Used for symbols that YOLO doesn't recognize at all.

        Args:
            image: Full image for context.
            region: Cropped region containing the unknown symbol.
            position: Position in drawing units.
            drawing_type: Type of drawing.
            annotations: Nearby text annotations.

        Returns:
            SmartSymbol or None if classification fails.
        """
        if not self.enable_vision_llm:
            return None

        classifier = self._get_vision_classifier()
        nearby_text = [ann.raw_text for ann in (annotations or [])[:5]]

        prompt = get_unknown_symbol_prompt(
            drawing_type=drawing_type,
            nearby_text=nearby_text,
        )

        try:
            result = await classifier.identify_unknown_symbol(
                image=region,
                prompt=prompt,
            )

            if result.error:
                return None

            symbol = SmartSymbol(
                block_name=f"{result.category[:1].upper()}-{result.type.upper()}-{result.subtype.upper()}".replace("_", "-"),
                position=position,
                category=result.category,
                type=result.type,
                subtype=result.subtype,
                direction=result.direction,
                system=result.system,
                confidence=result.confidence,
                vision_features=result.visual_features,
                reasoning=result.reasoning,
                classification_source="vision_llm",
            )

            return symbol

        except Exception as e:
            logger.warning("Unknown symbol classification failed", error=str(e))
            return None


# Convenience function for the unified detection interface
async def detect_and_classify_symbols(
    image: np.ndarray,
    drawing_type: DrawingType = DrawingType.MEP_PLAN,
    scale: float = 1 / 300,
    parsed_annotations: list[ParsedAnnotation] | None = None,
    enable_vision_llm: bool = True,
    yolo_backend: str = "auto",
    yolo_confidence: float = 0.5,
    vision_provider: str = "auto",
) -> tuple[np.ndarray, list[SmartSymbol]]:
    """
    Detect and classify symbols with the two-stage pipeline.

    This is the main entry point for Phase C symbol intelligence.

    Args:
        image: Binary image (white background, black ink).
        drawing_type: Type of drawing for context.
        scale: Pixel to drawing units scale factor.
        parsed_annotations: Pre-parsed text annotations.
        enable_vision_llm: Whether to use Vision LLM for detailed classification.
        yolo_backend: YOLO backend ("auto", "yolo", "template").
        yolo_confidence: YOLO detection confidence threshold.
        vision_provider: Vision LLM provider (gemini, openai, anthropic, auto).

    Returns:
        Tuple of (masked_image, list of SmartSymbol).

    Example:
        >>> from aec_agent.mcp.tools.symbol_classifier import detect_and_classify_symbols
        >>> masked, symbols = await detect_and_classify_symbols(
        ...     image=binary_img,
        ...     drawing_type=DrawingType.HVAC_PLAN,
        ...     enable_vision_llm=True,
        ... )
        >>> for sym in symbols:
        ...     print(f"{sym.subtype} at {sym.position}")
    """
    classifier = SymbolClassifier(
        enable_vision_llm=enable_vision_llm,
        vision_provider=vision_provider,
    )

    return await classifier.classify_symbols(
        image=image,
        drawing_type=drawing_type,
        scale=scale,
        parsed_annotations=parsed_annotations,
        yolo_backend=yolo_backend,
        yolo_confidence=yolo_confidence,
    )
