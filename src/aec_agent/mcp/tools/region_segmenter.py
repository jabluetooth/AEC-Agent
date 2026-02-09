"""
Region segmentation for AEC drawings.

Identifies and isolates different regions in engineering drawings:
- Title block (usually bottom-right or right edge)
- Legend (symbol definitions, usually right side)
- Drawing area (main content)
- Notes (text-heavy areas)
- Schedules (tabular data)
- Revision blocks

This is Layer 2 of the Semantic Intelligence Pipeline.
Each region type gets specialized processing.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np
import structlog

logger = structlog.get_logger(__name__)


class RegionType(str, Enum):
    """Type of drawing region."""
    TITLE_BLOCK = "title_block"     # Project info, sheet number
    LEGEND = "legend"               # Symbol definitions
    DRAWING_AREA = "drawing"        # Main drawing content
    NOTES = "notes"                 # General/construction notes
    SCHEDULE = "schedule"           # Tabular data
    REVISION_BLOCK = "revision"     # Revision history
    SCALE_BAR = "scale"             # Graphical scale
    BORDER = "border"               # Drawing border lines
    UNKNOWN = "unknown"


@dataclass
class DetectedRegion:
    """A detected region in the drawing."""
    region_type: RegionType
    bounds: tuple[int, int, int, int]  # x, y, width, height in pixels
    confidence: float  # 0.0 - 1.0
    text_content: str = ""  # Extracted text from region
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def x(self) -> int:
        return self.bounds[0]

    @property
    def y(self) -> int:
        return self.bounds[1]

    @property
    def width(self) -> int:
        return self.bounds[2]

    @property
    def height(self) -> int:
        return self.bounds[3]

    @property
    def area(self) -> int:
        return self.width * self.height

    @property
    def center(self) -> tuple[int, int]:
        return (self.x + self.width // 2, self.y + self.height // 2)

    def contains_point(self, x: int, y: int) -> bool:
        """Check if a point is inside this region."""
        return (self.x <= x < self.x + self.width and
                self.y <= y < self.y + self.height)

    def to_slice(self) -> tuple[slice, slice]:
        """Convert to numpy slice for array indexing."""
        return (slice(self.y, self.y + self.height),
                slice(self.x, self.x + self.width))


@dataclass
class SegmentationResult:
    """Result of region segmentation."""
    regions: list[DetectedRegion]
    image_width: int
    image_height: int
    title_block: DetectedRegion | None = None
    drawing_area: DetectedRegion | None = None
    legend: DetectedRegion | None = None

    def get_regions_by_type(self, region_type: RegionType) -> list[DetectedRegion]:
        """Get all regions of a specific type."""
        return [r for r in self.regions if r.region_type == region_type]

    def get_main_drawing_area(self) -> DetectedRegion | None:
        """Get the primary drawing area (largest DRAWING_AREA region)."""
        if self.drawing_area:
            return self.drawing_area
        drawing_regions = self.get_regions_by_type(RegionType.DRAWING_AREA)
        if drawing_regions:
            return max(drawing_regions, key=lambda r: r.area)
        return None


# Standard title block sizes (width x height) as fraction of sheet
# Based on common ANSI/ISO sheet sizes
TITLE_BLOCK_CONFIGS = {
    "ansi_d_bottom": {  # ANSI D (34x22), title block at bottom
        "position": "bottom",
        "width_fraction": (0.3, 0.5),   # 30-50% of sheet width
        "height_fraction": (0.08, 0.15),  # 8-15% of sheet height
    },
    "ansi_d_right": {  # Title block on right edge
        "position": "right",
        "width_fraction": (0.15, 0.25),
        "height_fraction": (0.3, 0.5),
    },
    "arch_d_bottom": {  # Architectural D (36x24)
        "position": "bottom-right",
        "width_fraction": (0.25, 0.4),
        "height_fraction": (0.1, 0.2),
    },
    "iso_a1": {  # ISO A1
        "position": "bottom-right",
        "width_fraction": (0.2, 0.35),
        "height_fraction": (0.1, 0.18),
    },
}


def segment_regions(
    image: np.ndarray,
    detect_title_block: bool = True,
    detect_legend: bool = True,
    detect_notes: bool = True,
    detect_schedules: bool = True,
    title_block_position: str = "auto",
    min_region_area_fraction: float = 0.01,
) -> SegmentationResult:
    """
    Segment a drawing image into functional regions.

    Uses a combination of:
    1. Template matching for standard title block sizes
    2. Line detection for region boundaries
    3. Text density analysis for notes/schedules

    Args:
        image: Grayscale or binary image (numpy array)
        detect_title_block: Whether to detect title block region
        detect_legend: Whether to detect legend region
        detect_notes: Whether to detect notes regions
        detect_schedules: Whether to detect schedule regions
        title_block_position: "auto", "bottom", "right", "bottom-right"
        min_region_area_fraction: Minimum region area as fraction of image

    Returns:
        SegmentationResult with detected regions
    """
    import cv2

    # Ensure grayscale
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    height, width = gray.shape[:2]
    regions: list[DetectedRegion] = []

    logger.info(
        "Starting region segmentation",
        image_size=f"{width}x{height}",
        detect_title_block=detect_title_block,
        detect_legend=detect_legend,
    )

    # Detect title block
    title_block = None
    if detect_title_block:
        title_block = _detect_title_block(gray, title_block_position)
        if title_block:
            regions.append(title_block)
            logger.info(
                "Title block detected",
                bounds=title_block.bounds,
                confidence=f"{title_block.confidence:.2f}",
            )

    # Detect legend (usually right side, above or beside title block)
    legend = None
    if detect_legend:
        legend = _detect_legend(gray, title_block)
        if legend:
            regions.append(legend)
            logger.info(
                "Legend detected",
                bounds=legend.bounds,
                confidence=f"{legend.confidence:.2f}",
            )

    # Detect notes regions (text-dense areas)
    if detect_notes:
        notes_regions = _detect_notes_regions(gray, regions, min_region_area_fraction)
        regions.extend(notes_regions)
        if notes_regions:
            logger.info(f"Detected {len(notes_regions)} notes regions")

    # Detect schedule regions (tabular data)
    if detect_schedules:
        schedule_regions = _detect_schedule_regions(gray, regions, min_region_area_fraction)
        regions.extend(schedule_regions)
        if schedule_regions:
            logger.info(f"Detected {len(schedule_regions)} schedule regions")

    # Determine main drawing area (everything not claimed by other regions)
    drawing_area = _compute_drawing_area(width, height, regions)
    if drawing_area:
        regions.insert(0, drawing_area)  # Add at beginning

    return SegmentationResult(
        regions=regions,
        image_width=width,
        image_height=height,
        title_block=title_block,
        drawing_area=drawing_area,
        legend=legend,
    )


def _detect_title_block(
    image: np.ndarray,
    position: str = "auto",
) -> DetectedRegion | None:
    """
    Detect the title block region.

    Title blocks are typically:
    - In the bottom-right corner
    - Have a distinctive rectangular border
    - Contain dense text (project name, sheet number, etc.)

    Args:
        image: Grayscale image
        position: "auto", "bottom", "right", "bottom-right"

    Returns:
        DetectedRegion for title block, or None if not found
    """
    import cv2

    height, width = image.shape[:2]

    # Define candidate regions based on position hint
    candidates = []

    if position in ("auto", "bottom-right"):
        # Check bottom-right quadrant
        candidates.append({
            "name": "bottom-right",
            "x": int(width * 0.6),
            "y": int(height * 0.75),
            "w": int(width * 0.4),
            "h": int(height * 0.25),
        })

    if position in ("auto", "bottom"):
        # Check bottom strip
        candidates.append({
            "name": "bottom",
            "x": int(width * 0.4),
            "y": int(height * 0.85),
            "w": int(width * 0.6),
            "h": int(height * 0.15),
        })

    if position in ("auto", "right"):
        # Check right strip
        candidates.append({
            "name": "right",
            "x": int(width * 0.8),
            "y": int(height * 0.5),
            "w": int(width * 0.2),
            "h": int(height * 0.5),
        })

    # Score each candidate by text density and border presence
    best_candidate = None
    best_score = 0.0

    for cand in candidates:
        x, y, w, h = cand["x"], cand["y"], cand["w"], cand["h"]

        # Ensure bounds are valid
        x = max(0, min(x, width - 1))
        y = max(0, min(y, height - 1))
        w = min(w, width - x)
        h = min(h, height - y)

        if w < 50 or h < 50:
            continue

        roi = image[y:y+h, x:x+w]

        # Score 1: Text density (title blocks have lots of text)
        # Binarize and count ink pixels
        _, binary = cv2.threshold(roi, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        ink_ratio = np.count_nonzero(binary) / (w * h)
        text_score = min(ink_ratio / 0.15, 1.0)  # Normalize: 15% ink = 1.0

        # Score 2: Border detection (title blocks have rectangular borders)
        edges = cv2.Canny(roi, 50, 150)
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, 50, minLineLength=min(w, h) * 0.3, maxLineGap=10)
        border_score = 0.0
        if lines is not None:
            # Count horizontal and vertical lines
            h_lines = sum(1 for l in lines if abs(l[0][1] - l[0][3]) < 10)
            v_lines = sum(1 for l in lines if abs(l[0][0] - l[0][2]) < 10)
            border_score = min((h_lines + v_lines) / 10, 1.0)

        # Score 3: Position (bottom-right is most common)
        position_score = 0.5
        if cand["name"] == "bottom-right":
            position_score = 1.0
        elif cand["name"] == "bottom":
            position_score = 0.7

        # Combined score
        score = (text_score * 0.4 + border_score * 0.4 + position_score * 0.2)

        if score > best_score:
            best_score = score
            best_candidate = {
                "bounds": (x, y, w, h),
                "confidence": score,
                "name": cand["name"],
            }

    # Require minimum confidence
    if best_candidate and best_candidate["confidence"] > 0.3:
        return DetectedRegion(
            region_type=RegionType.TITLE_BLOCK,
            bounds=best_candidate["bounds"],
            confidence=best_candidate["confidence"],
            metadata={"position": best_candidate["name"]},
        )

    return None


def _detect_legend(
    image: np.ndarray,
    title_block: DetectedRegion | None,
) -> DetectedRegion | None:
    """
    Detect the legend region.

    Legends are typically:
    - Above or beside the title block
    - Contain symbol graphics with text labels
    - Have consistent horizontal spacing (symbol columns)

    Args:
        image: Grayscale image
        title_block: Already detected title block (to exclude)

    Returns:
        DetectedRegion for legend, or None if not found
    """
    import cv2

    height, width = image.shape[:2]

    # Legends are typically on the right side
    # Look above the title block if present, otherwise check right edge

    if title_block:
        # Look above title block
        tb_x, tb_y, tb_w, tb_h = title_block.bounds

        # Legend candidate: same x range, above title block
        legend_height = int(height * 0.3)
        legend_y = max(0, tb_y - legend_height)
        actual_height = tb_y - legend_y

        if actual_height < 100:
            return None  # Not enough space for legend

        roi = image[legend_y:tb_y, tb_x:tb_x+tb_w]
    else:
        # Check right edge (rightmost 20% of width)
        roi_x = int(width * 0.8)
        roi = image[int(height*0.1):int(height*0.7), roi_x:]

        if roi.shape[0] < 100 or roi.shape[1] < 100:
            return None

        legend_y = int(height * 0.1)
        tb_x = roi_x
        tb_w = width - roi_x
        actual_height = int(height * 0.6)

    # Analyze ROI for legend characteristics
    # Legends have: horizontal rows of symbols + text

    # Binarize
    _, binary = cv2.threshold(roi, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Look for horizontal lines (row dividers)
    edges = cv2.Canny(roi, 50, 150)
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, 30, minLineLength=roi.shape[1] * 0.4, maxLineGap=20)

    h_line_count = 0
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            if abs(y1 - y2) < 5:  # Horizontal
                h_line_count += 1

    # Score: legends have multiple horizontal divider lines
    line_score = min(h_line_count / 5, 1.0)

    # Score: text density (legends have moderate text)
    ink_ratio = np.count_nonzero(binary) / binary.size
    text_score = 1.0 if 0.05 < ink_ratio < 0.25 else 0.5

    # Combined score
    confidence = (line_score * 0.6 + text_score * 0.4)

    if confidence > 0.3:
        bounds = (tb_x, legend_y, tb_w, actual_height)
        return DetectedRegion(
            region_type=RegionType.LEGEND,
            bounds=bounds,
            confidence=confidence,
        )

    return None


def _detect_notes_regions(
    image: np.ndarray,
    existing_regions: list[DetectedRegion],
    min_area_fraction: float = 0.01,
) -> list[DetectedRegion]:
    """
    Detect notes regions (text-heavy areas).

    Notes areas have:
    - High text density
    - Mostly horizontal text lines
    - Often in corners or margins

    Args:
        image: Grayscale image
        existing_regions: Already detected regions to exclude
        min_area_fraction: Minimum region area as fraction of image

    Returns:
        List of detected notes regions
    """
    import cv2

    height, width = image.shape[:2]
    min_area = int(width * height * min_area_fraction)

    # Binarize
    _, binary = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Create mask to exclude existing regions
    mask = np.ones((height, width), dtype=np.uint8) * 255
    for region in existing_regions:
        x, y, w, h = region.bounds
        mask[y:y+h, x:x+w] = 0

    # Apply mask
    masked = cv2.bitwise_and(binary, mask)

    # Look for text-dense areas using morphological operations
    # Dilate horizontally to connect text lines
    kernel_h = cv2.getStructuringElement(cv2.MORPH_RECT, (30, 3))
    dilated = cv2.dilate(masked, kernel_h, iterations=2)

    # Find contours of text blocks
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    notes_regions = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        area = w * h

        if area < min_area:
            continue

        # Check text density in original
        roi = binary[y:y+h, x:x+w]
        ink_ratio = np.count_nonzero(roi) / area

        # Notes have moderate-high text density
        if 0.08 < ink_ratio < 0.4:
            # Check if mostly horizontal lines (text)
            aspect_ratio = w / max(h, 1)
            if aspect_ratio > 1.5:  # Wider than tall = horizontal text block
                notes_regions.append(DetectedRegion(
                    region_type=RegionType.NOTES,
                    bounds=(x, y, w, h),
                    confidence=min(ink_ratio / 0.2, 1.0) * 0.8,
                ))

    return notes_regions


def _detect_schedule_regions(
    image: np.ndarray,
    existing_regions: list[DetectedRegion],
    min_area_fraction: float = 0.01,
) -> list[DetectedRegion]:
    """
    Detect schedule regions (tabular data).

    Schedules have:
    - Grid-like structure (horizontal + vertical lines)
    - Regular cell spacing
    - Header row with bold text

    Args:
        image: Grayscale image
        existing_regions: Already detected regions to exclude
        min_area_fraction: Minimum region area as fraction of image

    Returns:
        List of detected schedule regions
    """
    import cv2

    height, width = image.shape[:2]
    min_area = int(width * height * min_area_fraction)

    # Create mask to exclude existing regions
    mask = np.ones((height, width), dtype=np.uint8) * 255
    for region in existing_regions:
        x, y, w, h = region.bounds
        mask[y:y+h, x:x+w] = 0

    # Apply mask to image
    masked = cv2.bitwise_and(image, mask)

    # Detect lines
    edges = cv2.Canny(masked, 50, 150)
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, 50, minLineLength=50, maxLineGap=10)

    if lines is None or len(lines) < 10:
        return []

    # Separate horizontal and vertical lines
    h_lines = []
    v_lines = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        if abs(y1 - y2) < 5:  # Horizontal
            h_lines.append((x1, y1, x2, y2))
        elif abs(x1 - x2) < 5:  # Vertical
            v_lines.append((x1, y1, x2, y2))

    # Look for grid-like intersections
    # Group lines into potential table regions
    schedule_regions = []

    # Simple approach: find bounding box of line clusters
    if h_lines and v_lines:
        all_lines = h_lines + v_lines
        xs = [l[0] for l in all_lines] + [l[2] for l in all_lines]
        ys = [l[1] for l in all_lines] + [l[3] for l in all_lines]

        # Use line density to find table regions
        # (This is a simplified approach; more sophisticated would cluster lines)

        # For now, check if there's a high-density grid region
        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)

        w = max_x - min_x
        h = max_y - min_y
        area = w * h

        if area > min_area:
            # Count grid line density
            h_count = len(h_lines)
            v_count = len(v_lines)

            # Schedules have multiple horizontal AND vertical lines
            if h_count >= 3 and v_count >= 3:
                # Estimate confidence from grid regularity
                confidence = min((h_count + v_count) / 20, 0.9)

                schedule_regions.append(DetectedRegion(
                    region_type=RegionType.SCHEDULE,
                    bounds=(min_x, min_y, w, h),
                    confidence=confidence,
                    metadata={
                        "h_lines": h_count,
                        "v_lines": v_count,
                    },
                ))

    return schedule_regions


def _compute_drawing_area(
    width: int,
    height: int,
    regions: list[DetectedRegion],
) -> DetectedRegion | None:
    """
    Compute the main drawing area (everything not claimed by other regions).

    Args:
        width: Image width
        height: Image height
        regions: Already detected regions

    Returns:
        DetectedRegion for main drawing area
    """
    # Simple approach: largest unclaimed rectangular area
    # Start with full image, subtract other regions

    # Find the leftmost and topmost extent of non-drawing regions
    margin_right = width
    margin_bottom = height

    for region in regions:
        if region.region_type in (RegionType.TITLE_BLOCK, RegionType.LEGEND):
            # These are typically on the right/bottom
            if region.x < width * 0.5:
                continue  # Ignore if on left side
            margin_right = min(margin_right, region.x)

        if region.region_type == RegionType.TITLE_BLOCK:
            if region.y < height * 0.5:
                continue
            margin_bottom = min(margin_bottom, region.y)

    # Leave some margin from edges (typical drawing border)
    border_margin = int(min(width, height) * 0.02)

    drawing_x = border_margin
    drawing_y = border_margin
    drawing_w = margin_right - border_margin * 2
    drawing_h = margin_bottom - border_margin * 2

    if drawing_w < 100 or drawing_h < 100:
        # Fallback: use most of the image
        drawing_x = border_margin
        drawing_y = border_margin
        drawing_w = width - border_margin * 2
        drawing_h = height - border_margin * 2

    return DetectedRegion(
        region_type=RegionType.DRAWING_AREA,
        bounds=(drawing_x, drawing_y, drawing_w, drawing_h),
        confidence=0.9,
        metadata={"computed": True},
    )


def extract_title_block_text(
    image: np.ndarray,
    title_block: DetectedRegion,
) -> dict[str, str]:
    """
    Extract text from the title block region using OCR.

    Args:
        image: Full image
        title_block: Detected title block region

    Returns:
        Dictionary with extracted fields (sheet_number, sheet_name, etc.)
    """
    try:
        import pytesseract
    except ImportError:
        logger.warning("pytesseract not installed, cannot extract title block text")
        return {}

    x, y, w, h = title_block.bounds
    roi = image[y:y+h, x:x+w]

    # Run OCR
    try:
        text = pytesseract.image_to_string(roi, config="--psm 6")
    except Exception as e:
        logger.warning(f"OCR failed on title block: {e}")
        return {}

    # Parse common fields
    result = {"raw_text": text}

    # Look for sheet number patterns (E1.01, M-101, A100, etc.)
    import re

    sheet_pattern = r'\b([AEMPSFCL][-.]?\d{1,3}[.-]?\d{0,2})\b'
    sheet_matches = re.findall(sheet_pattern, text, re.IGNORECASE)
    if sheet_matches:
        result["sheet_number"] = sheet_matches[0].upper()

    # Look for date patterns
    date_pattern = r'\b(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})\b'
    date_matches = re.findall(date_pattern, text)
    if date_matches:
        result["date"] = date_matches[0]

    # Look for scale patterns
    scale_pattern = r'\b(1[:/=]\s*\d+|SCALE[:\s]*\d+)\b'
    scale_matches = re.findall(scale_pattern, text, re.IGNORECASE)
    if scale_matches:
        result["scale"] = scale_matches[0]

    return result


def create_region_mask(
    image_shape: tuple[int, int],
    regions: list[DetectedRegion],
    include_types: list[RegionType] | None = None,
    exclude_types: list[RegionType] | None = None,
) -> np.ndarray:
    """
    Create a binary mask for specified region types.

    Args:
        image_shape: (height, width) of the image
        regions: List of detected regions
        include_types: Region types to include (white in mask)
        exclude_types: Region types to exclude (black in mask)

    Returns:
        Binary mask (255 = included, 0 = excluded)
    """
    height, width = image_shape
    mask = np.zeros((height, width), dtype=np.uint8)

    if include_types:
        # Start with black, add white for included regions
        for region in regions:
            if region.region_type in include_types:
                x, y, w, h = region.bounds
                mask[y:y+h, x:x+w] = 255
    elif exclude_types:
        # Start with white, subtract excluded regions
        mask[:] = 255
        for region in regions:
            if region.region_type in exclude_types:
                x, y, w, h = region.bounds
                mask[y:y+h, x:x+w] = 0
    else:
        # Default: include everything
        mask[:] = 255

    return mask
