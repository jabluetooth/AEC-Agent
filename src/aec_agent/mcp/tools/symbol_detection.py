"""
Symbol Detection for AEC Drawings.

This module detects standard AEC symbols in bitonal images using either:
1. Template Matching (OpenCV) - default, no training required
2. YOLOv8 Neural Network - more robust, requires trained model

Part of Phase 2.5: Semantic AEC Vectorization Pipeline.

Usage:
    # Template matching (default)
    masked, blocks = detect_symbols(image, scale=1/300)

    # YOLO detection (if model available)
    masked, blocks = detect_symbols(image, scale=1/300, backend="yolo")

    # Auto-select best available backend
    masked, blocks = detect_symbols(image, scale=1/300, backend="auto")
"""

import math
from dataclasses import dataclass, field
from typing import List, Tuple, Optional
from pathlib import Path

import numpy as np
import structlog

logger = structlog.get_logger(__name__)


@dataclass
class DetectedBlock:
    """A symbol/block detected via template matching."""

    block_name: str  # AutoCAD block name to insert
    position: Tuple[float, float]  # Insertion point in drawing units
    scale: float = 1.0  # Block scale factor
    rotation: float = 0.0  # Rotation in degrees (0, 90, 180, 270)
    confidence: float = 0.0  # Match confidence 0-1
    category: str = ""  # "mechanical", "electrical", "fire", etc.


@dataclass
class SymbolTemplate:
    """A template image for matching."""

    name: str  # Template name (e.g., "valve_gate")
    block_name: str  # AutoCAD block name (e.g., "VALVE-GATE")
    category: str  # Category (e.g., "mechanical")
    image: "np.ndarray" = field(repr=False)  # Template image
    rotations: List[int] = field(default_factory=lambda: [0, 90, 180, 270])


def load_symbol_templates(
    template_dir: Optional[str] = None,
) -> List[SymbolTemplate]:
    """
    Load symbol template images from the assets directory.

    Templates should be:
    - PNG format, grayscale
    - White background (255), black symbol (0)
    - Approximately 32x32 to 64x64 pixels
    - Organized by category (mechanical/, electrical/, fire/, plumbing/)

    Args:
        template_dir: Path to template directory. If None, uses default
                      src/assets/templates/.

    Returns:
        List of SymbolTemplate objects ready for matching.

    Example:
        >>> templates = load_symbol_templates()
        >>> print(f"Loaded {len(templates)} templates")
    """
    try:
        import cv2
    except ImportError:
        logger.warning("OpenCV not available for template loading")
        return []

    if template_dir is None:
        # Default to package assets
        # Navigate from this file's location to assets/templates/
        this_file = Path(__file__)
        template_dir = this_file.parent.parent.parent.parent / "assets" / "templates"
    else:
        template_dir = Path(template_dir)

    if not template_dir.exists():
        logger.info(
            "Template directory not found, symbol detection will be skipped",
            path=str(template_dir),
        )
        return []

    templates: List[SymbolTemplate] = []
    categories = ["mechanical", "electrical", "fire", "plumbing", "low_voltage"]

    for category in categories:
        cat_dir = template_dir / category
        if not cat_dir.exists():
            continue

        for img_path in cat_dir.glob("*.png"):
            img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
            if img is None:
                logger.warning(f"Failed to load template: {img_path}")
                continue

            # Normalize: ensure white background (255), black symbol (0)
            # If image is inverted (black background), flip it
            if np.mean(img) < 128:
                img = cv2.bitwise_not(img)

            # Block name = filename without extension, uppercased, underscores to dashes
            block_name = img_path.stem.upper().replace("_", "-")

            templates.append(
                SymbolTemplate(
                    name=img_path.stem,
                    block_name=block_name,
                    category=category,
                    image=img,
                    rotations=[0, 90, 180, 270],  # Try all orthogonal rotations
                )
            )

    logger.info(
        "Symbol templates loaded",
        count=len(templates),
        directory=str(template_dir),
    )
    return templates


def detect_and_mask_symbols(
    image: np.ndarray,
    templates: List[SymbolTemplate],
    scale: float = 1.0,
    match_threshold: float = 0.8,
    nms_distance: float = 20.0,
    padding_px: int = 2,
) -> Tuple[np.ndarray, List[DetectedBlock]]:
    """
    Detect symbols using template matching and mask them from the image.

    This function runs OpenCV template matching against a library of standard
    AEC symbols. Detected symbols are masked (erased) from the image so that
    subsequent line detection algorithms don't try to vectorize them.

    Args:
        image: Binary image (white = background, black = ink).
        templates: List of SymbolTemplate objects to match against.
        scale: Coordinate scale factor (pixel to drawing units).
        match_threshold: Minimum correlation to accept a match (0-1).
                        Higher = fewer false positives, but may miss rotated symbols.
        nms_distance: Non-maximum suppression distance in pixels.
                      Prevents duplicate detections of the same symbol.
        padding_px: Pixels to expand mask around detected symbols.

    Returns:
        Tuple of:
        - masked_image: Image with symbol regions erased
        - detected_blocks: List of DetectedBlock for AutoCAD insertion

    Example:
        >>> templates = load_symbol_templates()
        >>> masked, blocks = detect_and_mask_symbols(binary_img, templates)
        >>> for b in blocks:
        ...     print(f"Found {b.block_name} at {b.position}")
    """
    try:
        import cv2
    except ImportError:
        logger.warning("OpenCV not available for symbol detection")
        return image, []

    if not templates:
        logger.debug("No templates provided, skipping symbol detection")
        return image, []

    height_px, width_px = image.shape[:2]
    masked_image = image.copy()

    # Collect all detections from all templates and rotations
    # Format: (cx, cy, w, h, block_name, category, confidence, rotation)
    all_detections: List[Tuple[float, float, int, int, str, str, float, int]] = []

    for template in templates:
        for rotation in template.rotations or [0]:
            # Rotate template if needed
            tmpl = template.image
            if rotation != 0:
                tmpl = _rotate_template(tmpl, rotation)

            if tmpl is None or tmpl.size == 0:
                continue

            # Skip if template is larger than image
            if tmpl.shape[0] > height_px or tmpl.shape[1] > width_px:
                continue

            # Template matching
            try:
                result = cv2.matchTemplate(image, tmpl, cv2.TM_CCOEFF_NORMED)
            except cv2.error as e:
                logger.warning(
                    "Template matching failed",
                    template=template.name,
                    error=str(e),
                )
                continue

            # Find peaks above threshold
            locations = np.where(result >= match_threshold)

            for pt in zip(*locations[::-1]):  # (x, y) pairs
                x, y = pt
                conf = float(result[y, x])

                # Center of the template
                cx = x + tmpl.shape[1] / 2
                cy = y + tmpl.shape[0] / 2

                all_detections.append(
                    (
                        cx,
                        cy,
                        tmpl.shape[1],
                        tmpl.shape[0],
                        template.block_name,
                        template.category,
                        conf,
                        rotation,
                    )
                )

    # Non-maximum suppression to remove duplicate detections
    detections = _nms_detections(all_detections, nms_distance)

    detected_blocks: List[DetectedBlock] = []

    # Determine background value for masking
    background_value = 255 if np.mean(image) > 128 else 0

    for cx, cy, w, h, block_name, category, conf, rotation in detections:
        # Mask the symbol region (fill with background)
        x1 = max(0, int(cx - w / 2 - padding_px))
        y1 = max(0, int(cy - h / 2 - padding_px))
        x2 = min(width_px, int(cx + w / 2 + padding_px))
        y2 = min(height_px, int(cy + h / 2 + padding_px))
        masked_image[y1:y2, x1:x2] = background_value

        # Convert to drawing units (Y-flip for AutoCAD)
        pos_x = cx * scale
        pos_y = (height_px - cy) * scale

        detected_blocks.append(
            DetectedBlock(
                block_name=block_name,
                position=(pos_x, pos_y),
                scale=1.0,
                rotation=float(rotation),
                confidence=conf,
                category=category,
            )
        )

    logger.info(
        "Symbol detection complete",
        templates_checked=len(templates),
        symbols_found=len(detected_blocks),
    )

    return masked_image, detected_blocks


def _rotate_template(image: np.ndarray, angle: int) -> np.ndarray:
    """
    Rotate template image by the given angle (must be 0, 90, 180, or 270).

    Uses cv2.rotate for exact 90-degree rotations (faster than warpAffine).
    """
    import cv2

    if angle == 0:
        return image
    elif angle == 90:
        return cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
    elif angle == 180:
        return cv2.rotate(image, cv2.ROTATE_180)
    elif angle == 270:
        return cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    else:
        # For arbitrary angles, use warpAffine (slower)
        center = (image.shape[1] // 2, image.shape[0] // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        return cv2.warpAffine(
            image,
            M,
            (image.shape[1], image.shape[0]),
            borderValue=255,  # White border
        )


def _nms_detections(
    detections: List[Tuple[float, float, int, int, str, str, float, int]],
    distance: float,
) -> List[Tuple[float, float, int, int, str, str, float, int]]:
    """
    Non-maximum suppression to remove duplicate detections.

    Keeps the detection with highest confidence when multiple detections
    are within the specified distance.

    Args:
        detections: List of (cx, cy, w, h, block_name, category, conf, rotation).
        distance: Maximum distance between centers to consider duplicates.

    Returns:
        Filtered list of detections.
    """
    if not detections:
        return []

    # Sort by confidence descending
    sorted_dets = sorted(detections, key=lambda x: x[6], reverse=True)
    kept: List[Tuple[float, float, int, int, str, str, float, int]] = []

    for det in sorted_dets:
        cx, cy = det[0], det[1]

        # Check if this detection is too close to a kept one
        is_duplicate = False
        for k in kept:
            kx, ky = k[0], k[1]
            dist = math.sqrt((cx - kx) ** 2 + (cy - ky) ** 2)
            if dist < distance:
                is_duplicate = True
                break

        if not is_duplicate:
            kept.append(det)

    return kept


# =============================================================================
# Unified Symbol Detection Interface
# =============================================================================


def is_yolo_available(model_path: Optional[str] = None) -> bool:
    """
    Check if YOLO detection is available.

    Args:
        model_path: Optional custom model path.

    Returns:
        True if YOLO model is available and loaded.
    """
    try:
        from .yolo_detection import get_yolo_detector

        detector = get_yolo_detector(model_path)
        return detector is not None and detector.is_available
    except ImportError:
        return False


def detect_symbols(
    image: np.ndarray,
    scale: float = 1.0,
    backend: str = "auto",
    # Template matching parameters
    match_threshold: float = 0.8,
    nms_distance: float = 20.0,
    template_dir: Optional[str] = None,
    # YOLO parameters
    yolo_model_path: Optional[str] = None,
    yolo_confidence: float = 0.5,
    yolo_iou_threshold: float = 0.45,
    # Common parameters
    mask_detections: bool = True,
) -> Tuple[np.ndarray, List[DetectedBlock]]:
    """
    Detect AEC symbols using the specified backend.

    This is the unified entry point for symbol detection, supporting both
    template matching and YOLOv8 neural network detection.

    Args:
        image: Binary image (white = background, black = ink).
        scale: Coordinate scale factor (pixel to drawing units). Typically 1/DPI.
        backend: Detection backend to use:
                 - "auto": Use YOLO if available, else template matching
                 - "yolo": Force YOLO (fails if not available)
                 - "template": Force template matching
        match_threshold: Template matching confidence threshold (0-1).
        nms_distance: Non-maximum suppression distance in pixels.
        template_dir: Custom template directory path.
        yolo_model_path: Custom YOLO model path (.pt or .onnx).
        yolo_confidence: YOLO confidence threshold (0-1).
        yolo_iou_threshold: YOLO IoU threshold for NMS (0-1).
        mask_detections: If True, mask detected symbols from returned image.

    Returns:
        Tuple of:
        - masked_image: Image with symbol regions optionally masked
        - detected_blocks: List of DetectedBlock for AutoCAD insertion

    Example:
        >>> # Auto-select best backend
        >>> masked, blocks = detect_symbols(binary_img, scale=1/300)

        >>> # Force YOLO
        >>> masked, blocks = detect_symbols(binary_img, backend="yolo")

        >>> # Force template matching with custom templates
        >>> masked, blocks = detect_symbols(
        ...     binary_img,
        ...     backend="template",
        ...     template_dir="custom/templates",
        ... )
    """
    # Determine which backend to use
    use_yolo = False

    if backend == "yolo":
        if is_yolo_available(yolo_model_path):
            use_yolo = True
        else:
            logger.error("YOLO backend requested but not available")
            return image, []

    elif backend == "auto":
        use_yolo = is_yolo_available(yolo_model_path)
        if use_yolo:
            logger.debug("Auto-selected YOLO backend")
        else:
            logger.debug("Auto-selected template matching backend")

    elif backend == "template":
        use_yolo = False

    else:
        logger.warning(f"Unknown backend '{backend}', using template matching")
        use_yolo = False

    # Run detection
    if use_yolo:
        from .yolo_detection import detect_symbols_yolo

        return detect_symbols_yolo(
            image,
            scale=scale,
            model_path=yolo_model_path,
            confidence=yolo_confidence,
            iou_threshold=yolo_iou_threshold,
            mask_detections=mask_detections,
        )
    else:
        templates = load_symbol_templates(template_dir)
        return detect_and_mask_symbols(
            image,
            templates,
            scale=scale,
            match_threshold=match_threshold,
            nms_distance=nms_distance,
            padding_px=2 if mask_detections else 0,
        )
