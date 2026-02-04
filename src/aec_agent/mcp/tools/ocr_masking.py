"""
OCR Text Detection and Masking for AEC Vectorization.

This module detects text regions in bitonal images using Tesseract OCR,
masks them from the image to prevent geometric noise during line detection,
and returns text entities for AutoCAD MText creation.

Part of Phase 2.5: Semantic AEC Vectorization Pipeline.
"""

from dataclasses import dataclass
from typing import List, Tuple, Optional
import numpy as np
import structlog

logger = structlog.get_logger(__name__)


@dataclass
class DetectedText:
    """Text region detected by OCR."""

    text: str  # The recognized text string
    position: Tuple[float, float]  # Bottom-left corner in drawing units
    width: float  # Text box width in drawing units
    height: float  # Text height in drawing units (for AutoCAD)
    confidence: float  # OCR confidence 0-100
    rotation: float = 0.0  # Estimated rotation in degrees


def detect_and_mask_text(
    image: np.ndarray,
    scale: float = 1.0,
    min_confidence: int = 60,
    min_text_height_px: int = 8,
    max_text_height_px: int = 200,
    padding_px: int = 2,
    lang: str = "eng",
    invert_for_ocr: bool = True,
) -> Tuple[np.ndarray, List[DetectedText]]:
    """
    Detect text regions using Tesseract OCR and mask them from the image.

    This function runs OCR on the input image to find text bounding boxes,
    extracts the text content for later AutoCAD MText creation, and masks
    (erases) the text regions from the image so that subsequent line
    detection algorithms don't try to vectorize text as geometry.

    Args:
        image: Grayscale or binary image (numpy array). Expected to be
               "ink on background" format (black lines on white, or inverted).
        scale: Coordinate scale factor (pixel to drawing units).
               Typically 1/DPI for inch-based drawings.
        min_confidence: Minimum OCR confidence (0-100) to accept text.
               Lower values catch more text but may include false positives.
        min_text_height_px: Minimum text height in pixels to detect.
               Filters out noise that looks like tiny text.
        max_text_height_px: Maximum text height in pixels to detect.
               Filters out large regions that aren't actually text.
        padding_px: Pixels to expand bounding box for masking.
               Ensures complete text removal including anti-aliasing artifacts.
        lang: Tesseract language code (e.g., "eng", "eng+osd").
        invert_for_ocr: If True, inverts the image for OCR (Tesseract expects
               dark text on light background).

    Returns:
        Tuple of:
        - masked_image: Image with text regions filled with background color
        - detected_texts: List of DetectedText for AutoCAD MText creation

    Raises:
        RuntimeError: If Tesseract is not installed or fails to initialize.

    Example:
        >>> import cv2
        >>> img = cv2.imread("plan.tif", cv2.IMREAD_GRAYSCALE)
        >>> masked, texts = detect_and_mask_text(img, scale=1/300)
        >>> print(f"Found {len(texts)} text regions")
        >>> for t in texts:
        ...     print(f"  '{t.text}' at {t.position}")
    """
    try:
        import pytesseract
        from PIL import Image
    except ImportError as e:
        logger.warning(
            "pytesseract not installed, skipping OCR masking",
            error=str(e),
        )
        return image, []

    # Validate input
    if image is None or image.size == 0:
        logger.warning("Empty image provided to OCR masking")
        return image, []

    height_px, width_px = image.shape[:2]

    # Prepare image for Tesseract
    # Tesseract expects dark text on light background
    ocr_image = image.copy()
    if invert_for_ocr:
        # If image is inverted (white lines on black), flip it
        if np.mean(ocr_image) < 128:
            ocr_image = 255 - ocr_image

    # Convert numpy to PIL for pytesseract
    pil_image = Image.fromarray(ocr_image)

    try:
        # Get detailed OCR data including bounding boxes
        # Output includes: level, page_num, block_num, par_num, line_num, word_num,
        #                  left, top, width, height, conf, text
        ocr_data = pytesseract.image_to_data(
            pil_image,
            lang=lang,
            output_type=pytesseract.Output.DICT,
            config="--psm 11",  # Sparse text mode - good for engineering drawings
        )
    except pytesseract.TesseractNotFoundError:
        logger.error(
            "Tesseract OCR not installed. "
            "Install from https://github.com/UB-Mannheim/tesseract/wiki"
        )
        return image, []
    except Exception as e:
        logger.warning(
            "Tesseract OCR failed",
            error=str(e),
        )
        return image, []

    detected_texts: List[DetectedText] = []
    masked_image = image.copy()

    # Determine background value for masking
    # If image mean > 128, background is white (255); else black (0)
    background_value = 255 if np.mean(image) > 128 else 0

    n_boxes = len(ocr_data["text"])
    logger.debug(f"OCR found {n_boxes} potential text regions")

    for i in range(n_boxes):
        text = ocr_data["text"][i]
        if not isinstance(text, str):
            continue
        text = text.strip()

        # Skip empty text
        if not text:
            continue

        # Parse confidence (-1 means no confidence available)
        try:
            conf = int(ocr_data["conf"][i])
        except (ValueError, TypeError):
            conf = 0

        if conf < min_confidence:
            continue

        # Bounding box in pixels (top-left origin, as per image coordinates)
        x = int(ocr_data["left"][i])
        y = int(ocr_data["top"][i])
        w = int(ocr_data["width"][i])
        h = int(ocr_data["height"][i])

        # Filter by text height
        if h < min_text_height_px or h > max_text_height_px:
            continue

        # Filter out very wide boxes (likely not real text)
        if w > width_px * 0.8:
            continue

        # Mask the text region (fill with background color)
        x1 = max(0, x - padding_px)
        y1 = max(0, y - padding_px)
        x2 = min(width_px, x + w + padding_px)
        y2 = min(height_px, y + h + padding_px)
        masked_image[y1:y2, x1:x2] = background_value

        # Convert to drawing units (Y-flip for AutoCAD coordinate system)
        # AutoCAD origin is bottom-left; image origin is top-left
        pos_x = x * scale
        pos_y = (height_px - (y + h)) * scale  # Bottom-left of text box

        # Text height for AutoCAD is approximately 70-80% of bbox height
        # This accounts for descenders and ascenders in the bounding box
        text_height = h * scale * 0.75

        detected_texts.append(
            DetectedText(
                text=text,
                position=(pos_x, pos_y),
                width=w * scale,
                height=text_height,
                confidence=float(conf),
                rotation=0.0,  # TODO: detect rotation via baseline angle
            )
        )

    logger.info(
        "OCR masking complete",
        texts_found=len(detected_texts),
        min_confidence=min_confidence,
    )

    return masked_image, detected_texts


def merge_adjacent_texts(
    texts: List[DetectedText],
    horizontal_gap_tolerance: float = 5.0,
    vertical_tolerance: float = 2.0,
) -> List[DetectedText]:
    """
    Merge adjacent text regions that likely belong together.

    Tesseract often splits a single label into multiple words.
    This function merges horizontally adjacent text regions that
    are on the same baseline into single text entities.

    Args:
        texts: List of detected text regions.
        horizontal_gap_tolerance: Maximum horizontal gap to merge (drawing units).
        vertical_tolerance: Maximum vertical offset to consider same line.

    Returns:
        List of merged DetectedText regions.
    """
    if len(texts) <= 1:
        return texts

    # Sort by Y position (line), then X position (left to right)
    sorted_texts = sorted(texts, key=lambda t: (round(t.position[1], 1), t.position[0]))

    merged: List[DetectedText] = []
    current_group: List[DetectedText] = [sorted_texts[0]]

    for text in sorted_texts[1:]:
        last = current_group[-1]

        # Check if on same horizontal line
        y_diff = abs(text.position[1] - last.position[1])
        if y_diff > vertical_tolerance:
            # Different line - finalize current group and start new
            merged.append(_merge_text_group(current_group))
            current_group = [text]
            continue

        # Check horizontal gap
        last_right = last.position[0] + last.width
        gap = text.position[0] - last_right

        if gap <= horizontal_gap_tolerance:
            # Adjacent - add to group
            current_group.append(text)
        else:
            # Gap too large - finalize and start new
            merged.append(_merge_text_group(current_group))
            current_group = [text]

    # Don't forget the last group
    if current_group:
        merged.append(_merge_text_group(current_group))

    return merged


def _merge_text_group(group: List[DetectedText]) -> DetectedText:
    """Merge a group of adjacent texts into a single DetectedText."""
    if len(group) == 1:
        return group[0]

    # Combine text with spaces
    combined_text = " ".join(t.text for t in group)

    # Position is leftmost
    min_x = min(t.position[0] for t in group)
    avg_y = sum(t.position[1] for t in group) / len(group)

    # Width spans from leftmost to rightmost
    max_right = max(t.position[0] + t.width for t in group)
    total_width = max_right - min_x

    # Height is average
    avg_height = sum(t.height for t in group) / len(group)

    # Confidence is minimum (conservative)
    min_conf = min(t.confidence for t in group)

    return DetectedText(
        text=combined_text,
        position=(min_x, avg_y),
        width=total_width,
        height=avg_height,
        confidence=min_conf,
        rotation=group[0].rotation,
    )
