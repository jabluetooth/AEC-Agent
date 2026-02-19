"""
OCR-based Text Position Anchoring.

This module uses Tesseract OCR to get pixel-accurate text positions,
then matches Gemini's semantically-corrected text content to those positions.

The problem: Gemini estimates text positions visually (can be off by 10-100+ pixels),
while OpenCV provides pixel-perfect geometry. This causes misalignment.

The solution: Use OCR to detect actual text bounding boxes, then anchor
Gemini's text content to those positions.

Usage:
    >>> anchored = await anchor_text_positions(analysis, image_path, calibration)
    >>> print(f"Anchored {len(anchored)} text elements")
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from pathlib import Path
from typing import List, Optional, Tuple, TYPE_CHECKING

import numpy as np
import structlog

if TYPE_CHECKING:
    from PIL import Image as PILImage
    from .gemini_understanding import DrawingAnalysis, DetectedText
    from .coordinate_calibration import ScaleCalibration
    from .adaptive_extraction import EntityToCreate

logger = structlog.get_logger(__name__)

# Check if pytesseract is available
TESSERACT_AVAILABLE = False
try:
    import pytesseract
    TESSERACT_AVAILABLE = True
except ImportError:
    pass


@dataclass
class OCRTextBox:
    """Text detected by OCR with bounding box."""
    text: str
    x: int  # Left position (pixels)
    y: int  # Top position (pixels)
    width: int  # Box width (pixels)
    height: int  # Box height (pixels)
    confidence: float  # OCR confidence (0-100)

    @property
    def center(self) -> Tuple[int, int]:
        """Get center point of the text box."""
        return (self.x + self.width // 2, self.y + self.height // 2)

    @property
    def bottom_left(self) -> Tuple[int, int]:
        """Get bottom-left point (typical text insertion point)."""
        return (self.x, self.y + self.height)


@dataclass
class TextAnchorResult:
    """Result of text position anchoring."""
    original_position: Tuple[int, int]  # Gemini's estimated position
    anchored_position: Tuple[int, int]  # OCR-derived accurate position
    ocr_text: str  # Text detected by OCR
    gemini_text: str  # Text from Gemini
    match_score: float  # Similarity score (0-1)
    height_px: int  # Detected text height


@dataclass
class TextAnchoringResult:
    """Complete result of text anchoring process."""
    anchored_texts: List[TextAnchorResult] = field(default_factory=list)
    unmatched_gemini: List[str] = field(default_factory=list)
    unmatched_ocr: List[str] = field(default_factory=list)
    total_texts: int = 0
    anchored_count: int = 0
    avg_offset: float = 0.0  # Average pixel offset between Gemini and OCR positions

    @property
    def anchor_rate(self) -> float:
        """Percentage of texts successfully anchored."""
        if self.total_texts == 0:
            return 0.0
        return self.anchored_count / self.total_texts * 100


def extract_text_with_ocr(
    image_path: Path,
    min_confidence: float = 60.0,
    lang: str = "eng",
) -> List[OCRTextBox]:
    """
    Extract text from image using Tesseract OCR.

    Args:
        image_path: Path to image file
        min_confidence: Minimum confidence threshold (0-100)
        lang: OCR language code

    Returns:
        List of OCRTextBox with detected text and positions
    """
    if not TESSERACT_AVAILABLE:
        logger.warning("tesseract_not_available")
        return []

    try:
        from PIL import Image
        import pytesseract

        # Load image
        image = Image.open(image_path)

        # Run OCR with bounding box data
        ocr_data = pytesseract.image_to_data(
            image,
            lang=lang,
            output_type=pytesseract.Output.DICT,
            config='--psm 6'  # Assume uniform block of text
        )

        results: List[OCRTextBox] = []

        # Process OCR results
        n_boxes = len(ocr_data['text'])
        for i in range(n_boxes):
            text = ocr_data['text'][i].strip()
            if not text:
                continue

            conf = float(ocr_data['conf'][i])
            if conf < min_confidence:
                continue

            results.append(OCRTextBox(
                text=text,
                x=int(ocr_data['left'][i]),
                y=int(ocr_data['top'][i]),
                width=int(ocr_data['width'][i]),
                height=int(ocr_data['height'][i]),
                confidence=conf,
            ))

        logger.info(
            "ocr_extraction_complete",
            total_boxes=n_boxes,
            filtered_results=len(results),
            min_confidence=min_confidence,
        )

        return results

    except Exception as e:
        logger.warning("ocr_extraction_failed", error=str(e))
        return []


def normalize_text(text: str) -> str:
    """Normalize text for comparison."""
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text.strip())
    # Lowercase
    text = text.lower()
    # Remove common OCR artifacts
    text = text.replace('|', 'l').replace('0', 'o')
    return text


def calculate_similarity(text1: str, text2: str) -> float:
    """
    Calculate similarity between two text strings.

    Uses SequenceMatcher for fuzzy matching, accounting for
    OCR errors and Gemini's corrections.
    """
    norm1 = normalize_text(text1)
    norm2 = normalize_text(text2)

    # Exact match
    if norm1 == norm2:
        return 1.0

    # Fuzzy match
    return SequenceMatcher(None, norm1, norm2).ratio()


def find_best_ocr_match(
    gemini_text: str,
    ocr_boxes: List[OCRTextBox],
    min_similarity: float = 0.6,
) -> Optional[Tuple[OCRTextBox, float]]:
    """
    Find the best OCR match for a Gemini text.

    Args:
        gemini_text: Text from Gemini (semantically corrected)
        ocr_boxes: Available OCR detections
        min_similarity: Minimum similarity score to accept

    Returns:
        Tuple of (best matching OCRTextBox, similarity score) or None
    """
    best_match: Optional[Tuple[OCRTextBox, float]] = None
    best_score = 0.0

    for box in ocr_boxes:
        score = calculate_similarity(gemini_text, box.text)

        if score > best_score and score >= min_similarity:
            best_score = score
            best_match = (box, score)

    return best_match


def anchor_text_to_ocr(
    gemini_texts: List["DetectedText"],
    ocr_boxes: List[OCRTextBox],
    min_similarity: float = 0.5,
    max_position_offset: int = 200,  # Max pixels to accept a match
) -> TextAnchoringResult:
    """
    Anchor Gemini text positions to OCR-detected positions.

    Uses a two-pass matching:
    1. First match by text similarity
    2. Then validate by position proximity

    Args:
        gemini_texts: Text elements from Gemini's analysis
        ocr_boxes: OCR-detected text boxes
        min_similarity: Minimum text similarity to accept
        max_position_offset: Maximum position offset to accept match

    Returns:
        TextAnchoringResult with anchored positions
    """
    result = TextAnchoringResult(total_texts=len(gemini_texts))

    # Track used OCR boxes
    used_boxes: set = set()
    offsets: List[float] = []

    for gemini_text in gemini_texts:
        # Find best text match
        best_match = None
        best_score = 0.0

        for i, box in enumerate(ocr_boxes):
            if i in used_boxes:
                continue

            score = calculate_similarity(gemini_text.content, box.text)

            if score > best_score and score >= min_similarity:
                # Check position proximity
                g_pos = gemini_text.position
                o_pos = box.bottom_left
                offset = np.sqrt((g_pos[0] - o_pos[0])**2 + (g_pos[1] - o_pos[1])**2)

                # If positions are reasonably close, it's likely the same text
                if offset <= max_position_offset:
                    best_score = score
                    best_match = (i, box, offset)

        if best_match:
            idx, box, offset = best_match
            used_boxes.add(idx)
            offsets.append(offset)

            result.anchored_texts.append(TextAnchorResult(
                original_position=gemini_text.position,
                anchored_position=box.bottom_left,
                ocr_text=box.text,
                gemini_text=gemini_text.content,
                match_score=best_score,
                height_px=box.height,
            ))
            result.anchored_count += 1
        else:
            result.unmatched_gemini.append(gemini_text.content)

    # Track unmatched OCR boxes
    for i, box in enumerate(ocr_boxes):
        if i not in used_boxes:
            result.unmatched_ocr.append(box.text)

    # Calculate average offset
    if offsets:
        result.avg_offset = float(np.mean(offsets))

    logger.info(
        "text_anchoring_complete",
        total_texts=result.total_texts,
        anchored=result.anchored_count,
        anchor_rate=f"{result.anchor_rate:.1f}%",
        avg_offset=f"{result.avg_offset:.1f}px",
        unmatched_gemini=len(result.unmatched_gemini),
        unmatched_ocr=len(result.unmatched_ocr),
    )

    return result


async def anchor_text_positions(
    analysis: "DrawingAnalysis",
    image_path: Path,
    calibration: "ScaleCalibration",
    min_similarity: float = 0.5,
    min_ocr_confidence: float = 60.0,
) -> List["EntityToCreate"]:
    """
    Anchor Gemini's text positions using OCR.

    This is the main entry point for text position anchoring.
    Returns MTEXT entities with OCR-anchored positions.

    Args:
        analysis: Gemini's drawing analysis
        image_path: Path to source image
        calibration: Coordinate calibration
        min_similarity: Minimum text similarity for matching
        min_ocr_confidence: Minimum OCR confidence threshold

    Returns:
        List of MTEXT EntityToCreate with anchored positions
    """
    from .adaptive_extraction import EntityToCreate, EntityType, ExtractionSource, get_layer_for_text

    entities: List[EntityToCreate] = []

    if not TESSERACT_AVAILABLE:
        logger.warning("tesseract_not_available", message="Falling back to Gemini positions")
        # Fall back to Gemini's approximate positions
        for text in analysis.elements.text:
            position_dwg = calibration.to_dwg(*text.position)
            height_dwg = calibration.scale_length(text.height_px)
            height_dwg = max(height_dwg, 0.1)
            layer = get_layer_for_text(text.text_type)

            entities.append(EntityToCreate(
                entity_type=EntityType.MTEXT,
                layer=layer,
                properties={
                    "content": text.content,
                    "position": position_dwg,
                    "height": height_dwg,
                    "text_type": text.text_type,
                },
                source=ExtractionSource.DIRECT,
                source_element=f"text_{text.text_type}",
            ))
        return entities

    # Extract text with OCR
    ocr_boxes = extract_text_with_ocr(
        image_path,
        min_confidence=min_ocr_confidence,
    )

    if not ocr_boxes:
        logger.warning("no_ocr_detections", message="Falling back to Gemini positions")
        # Fall back to Gemini positions
        for text in analysis.elements.text:
            position_dwg = calibration.to_dwg(*text.position)
            height_dwg = calibration.scale_length(text.height_px)
            height_dwg = max(height_dwg, 0.1)
            layer = get_layer_for_text(text.text_type)

            entities.append(EntityToCreate(
                entity_type=EntityType.MTEXT,
                layer=layer,
                properties={
                    "content": text.content,
                    "position": position_dwg,
                    "height": height_dwg,
                    "text_type": text.text_type,
                },
                source=ExtractionSource.DIRECT,
                source_element=f"text_{text.text_type}",
            ))
        return entities

    # Anchor text positions
    anchoring = anchor_text_to_ocr(
        analysis.elements.text,
        ocr_boxes,
        min_similarity=min_similarity,
    )

    # Create entities with anchored positions
    for anchor in anchoring.anchored_texts:
        # Use OCR position, but Gemini's content (semantically corrected)
        position_dwg = calibration.to_dwg(*anchor.anchored_position)
        height_dwg = calibration.scale_length(anchor.height_px)
        height_dwg = max(height_dwg, 0.1)

        # Find the original Gemini text for layer assignment
        text_type = "note"  # default
        for text in analysis.elements.text:
            if text.content == anchor.gemini_text:
                text_type = text.text_type
                break

        layer = get_layer_for_text(text_type)

        entities.append(EntityToCreate(
            entity_type=EntityType.MTEXT,
            layer=layer,
            properties={
                "content": anchor.gemini_text,  # Use Gemini's corrected text
                "position": position_dwg,  # Use OCR's accurate position
                "height": height_dwg,  # Use OCR's detected height
                "text_type": text_type,
                "anchor_score": anchor.match_score,
            },
            source=ExtractionSource.HYBRID,
            confidence=anchor.match_score,
            source_element=f"ocr_anchored_{text_type}",
        ))

    # For unmatched Gemini texts, use original positions
    for unmatched_content in anchoring.unmatched_gemini:
        # Find original text data
        for text in analysis.elements.text:
            if text.content == unmatched_content:
                position_dwg = calibration.to_dwg(*text.position)
                height_dwg = calibration.scale_length(text.height_px)
                height_dwg = max(height_dwg, 0.1)
                layer = get_layer_for_text(text.text_type)

                entities.append(EntityToCreate(
                    entity_type=EntityType.MTEXT,
                    layer=layer,
                    properties={
                        "content": text.content,
                        "position": position_dwg,
                        "height": height_dwg,
                        "text_type": text.text_type,
                    },
                    source=ExtractionSource.DIRECT,
                    confidence=0.5,  # Lower confidence for unanchored
                    source_element=f"text_unanchored_{text.text_type}",
                ))
                break

    logger.info(
        "text_position_anchoring_complete",
        total_entities=len(entities),
        ocr_anchored=anchoring.anchored_count,
        gemini_fallback=len(anchoring.unmatched_gemini),
        avg_offset=f"{anchoring.avg_offset:.1f}px",
    )

    return entities


# Export availability check
def is_ocr_available() -> bool:
    """Check if OCR (Tesseract) is available."""
    return TESSERACT_AVAILABLE
