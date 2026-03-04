"""
Linetype Detection Module.

Detects line types (CONTINUOUS, DASHED, HIDDEN, CENTER, PHANTOM, DOTTED) by
analyzing pixel patterns along line segments using FFT and pattern matching.

This module enables proper linetype preservation when vectorizing raster
drawings, matching Scan2CAD's capability to detect dashed/dotted lines.

Algorithm:
1. Sample pixels along the line path
2. Apply FFT to detect periodic patterns (gaps/dashes)
3. Classify based on gap/dash ratios and pattern frequency
4. Map to AutoCAD linetype names

References:
- AutoCAD Linetype Definition: https://help.autodesk.com/view/ACD/2024/ENU/
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Tuple
import math

import numpy as np
from scipy import signal
from scipy.fft import fft, fftfreq
import structlog

logger = structlog.get_logger(__name__)


class LinetypeName(str, Enum):
    """AutoCAD standard linetype names."""
    CONTINUOUS = "Continuous"
    DASHED = "DASHED"
    HIDDEN = "HIDDEN"
    CENTER = "CENTER"
    PHANTOM = "PHANTOM"
    DOTTED = "DOT"
    DASHDOT = "DASHDOT"
    DIVIDE = "DIVIDE"
    BORDER = "BORDER"


# AutoCAD linetype patterns (in drawing units, normalized)
# Pattern: [dash, gap, dash, gap, ...] where positive = dash, negative = gap
LINETYPE_PATTERNS = {
    LinetypeName.CONTINUOUS: [],  # No pattern
    LinetypeName.DASHED: [0.5, -0.25],  # ___  ___  ___
    LinetypeName.HIDDEN: [0.25, -0.125],  # __  __  __
    LinetypeName.CENTER: [1.25, -0.25, 0.25, -0.25],  # ____  _  ____
    LinetypeName.PHANTOM: [1.25, -0.25, 0.25, -0.25, 0.25, -0.25],  # ____  _  _  ____
    LinetypeName.DOTTED: [0.0, -0.25],  # .  .  .
    LinetypeName.DASHDOT: [0.5, -0.25, 0.0, -0.25],  # ___  .  ___
    LinetypeName.DIVIDE: [0.5, -0.25, 0.0, -0.25, 0.0, -0.25],  # ___  .  .  ___
    LinetypeName.BORDER: [0.5, -0.25, 0.5, -0.25, 0.0, -0.25],  # ___  ___  .  ___
}


@dataclass
class LinetypeResult:
    """Result of linetype detection for a single line."""
    linetype: LinetypeName
    confidence: float  # 0.0 to 1.0
    pattern: List[float] = field(default_factory=list)  # Detected pattern in pixels
    dash_ratio: float = 0.0  # Ratio of dash to total length
    gap_count: int = 0  # Number of gaps detected
    dominant_frequency: float = 0.0  # FFT dominant frequency (gaps per pixel)

    def to_dict(self) -> dict:
        return {
            "linetype": self.linetype.value,
            "confidence": self.confidence,
            "pattern": self.pattern,
            "dash_ratio": self.dash_ratio,
            "gap_count": self.gap_count,
            "dominant_frequency": self.dominant_frequency,
        }


@dataclass
class LinetypeConfig:
    """Configuration for linetype detection."""
    sample_width: int = 3  # Width of sampling perpendicular to line
    min_line_length: int = 30  # Minimum line length for reliable detection
    min_gaps_for_pattern: int = 2  # Minimum gaps to consider non-continuous
    gap_threshold: float = 0.3  # Intensity threshold for gap detection (0-1)
    fft_peak_threshold: float = 0.1  # Minimum FFT peak to consider periodic
    smoothing_kernel: int = 3  # Kernel size for intensity smoothing

    # Pattern classification thresholds
    dash_ratio_continuous: float = 0.95  # Above this = CONTINUOUS
    dash_ratio_dashed: Tuple[float, float] = (0.5, 0.75)  # DASHED range
    dash_ratio_hidden: Tuple[float, float] = (0.4, 0.6)  # HIDDEN range
    dash_ratio_dotted: float = 0.2  # Below this = DOTTED


def sample_line_intensity(
    binary_image: np.ndarray,
    start: Tuple[int, int],
    end: Tuple[int, int],
    sample_width: int = 3,
) -> np.ndarray:
    """
    Sample pixel intensities along a line path.

    Args:
        binary_image: Binary image (0 = background, 255 = foreground)
        start: (x, y) start point
        end: (x, y) end point
        sample_width: Width of sampling perpendicular to line

    Returns:
        1D array of intensity values along the line (0.0 to 1.0)
    """
    x1, y1 = start
    x2, y2 = end

    # Calculate line length and direction
    dx = x2 - x1
    dy = y2 - y1
    length = int(math.sqrt(dx * dx + dy * dy))

    if length < 2:
        return np.array([1.0])

    # Normalize direction
    dx_norm = dx / length
    dy_norm = dy / length

    # Perpendicular direction for width sampling
    perp_x = -dy_norm
    perp_y = dx_norm

    # Sample points along the line
    intensities = []
    h, w = binary_image.shape[:2]

    for i in range(length):
        # Center point on line
        cx = x1 + dx_norm * i
        cy = y1 + dy_norm * i

        # Sample across width and take max (to handle thin lines)
        max_intensity = 0.0
        for offset in range(-sample_width // 2, sample_width // 2 + 1):
            px = int(cx + perp_x * offset)
            py = int(cy + perp_y * offset)

            if 0 <= px < w and 0 <= py < h:
                # Normalize to 0-1 range
                intensity = binary_image[py, px] / 255.0
                max_intensity = max(max_intensity, intensity)

        intensities.append(max_intensity)

    return np.array(intensities)


def detect_gaps(
    intensities: np.ndarray,
    gap_threshold: float = 0.3,
    min_gap_length: int = 2,
) -> List[Tuple[int, int]]:
    """
    Detect gaps (regions of low intensity) in the intensity profile.

    Args:
        intensities: 1D array of intensity values (0.0 to 1.0)
        gap_threshold: Intensity below this is considered a gap
        min_gap_length: Minimum length of gap to be counted

    Returns:
        List of (start_idx, end_idx) tuples for each gap
    """
    gaps = []
    in_gap = False
    gap_start = 0

    for i, intensity in enumerate(intensities):
        if intensity < gap_threshold:
            if not in_gap:
                in_gap = True
                gap_start = i
        else:
            if in_gap:
                in_gap = False
                if i - gap_start >= min_gap_length:
                    gaps.append((gap_start, i))

    # Handle gap at end of line
    if in_gap and len(intensities) - gap_start >= min_gap_length:
        gaps.append((gap_start, len(intensities)))

    return gaps


def analyze_pattern_fft(
    intensities: np.ndarray,
    smoothing_kernel: int = 3,
) -> Tuple[float, List[float]]:
    """
    Analyze intensity pattern using FFT to detect periodicity.

    Args:
        intensities: 1D array of intensity values
        smoothing_kernel: Kernel size for smoothing

    Returns:
        Tuple of (dominant_frequency, list of significant frequencies)
    """
    if len(intensities) < 8:
        return 0.0, []

    # Smooth the signal
    if smoothing_kernel > 1:
        kernel = np.ones(smoothing_kernel) / smoothing_kernel
        smoothed = np.convolve(intensities, kernel, mode='same')
    else:
        smoothed = intensities

    # Apply FFT
    n = len(smoothed)
    yf = fft(smoothed - np.mean(smoothed))  # Remove DC component
    xf = fftfreq(n)[:n // 2]

    # Get magnitude spectrum (only positive frequencies)
    magnitude = 2.0 / n * np.abs(yf[:n // 2])

    # Find peaks in magnitude spectrum
    if len(magnitude) < 3:
        return 0.0, []

    # Find dominant frequency (excluding DC)
    peak_idx = np.argmax(magnitude[1:]) + 1 if len(magnitude) > 1 else 0
    dominant_freq = xf[peak_idx] if peak_idx < len(xf) else 0.0

    # Find all significant frequencies
    threshold = np.max(magnitude) * 0.3
    significant_freqs = [xf[i] for i in range(1, len(magnitude))
                        if magnitude[i] > threshold and i < len(xf)]

    return abs(dominant_freq), significant_freqs


def extract_pattern(
    intensities: np.ndarray,
    gaps: List[Tuple[int, int]],
) -> List[float]:
    """
    Extract the dash-gap pattern from detected gaps.

    Args:
        intensities: 1D intensity array
        gaps: List of (start, end) gap positions

    Returns:
        Pattern as [dash_length, gap_length, ...] in pixels
    """
    if not gaps:
        return []

    pattern = []
    prev_end = 0

    for gap_start, gap_end in gaps:
        # Dash length (from previous gap end to this gap start)
        dash_len = gap_start - prev_end
        if dash_len > 0:
            pattern.append(float(dash_len))

        # Gap length
        gap_len = gap_end - gap_start
        pattern.append(float(-gap_len))  # Negative for gaps (AutoCAD convention)

        prev_end = gap_end

    # Final dash (from last gap to end)
    final_dash = len(intensities) - prev_end
    if final_dash > 0:
        pattern.append(float(final_dash))

    return pattern


def classify_linetype(
    dash_ratio: float,
    gap_count: int,
    dominant_freq: float,
    pattern: List[float],
    config: LinetypeConfig,
) -> Tuple[LinetypeName, float]:
    """
    Classify the linetype based on analysis results.

    Args:
        dash_ratio: Ratio of dash pixels to total pixels
        gap_count: Number of gaps detected
        dominant_freq: Dominant FFT frequency
        pattern: Extracted pattern
        config: Detection configuration

    Returns:
        Tuple of (LinetypeName, confidence)
    """
    # Continuous: no gaps or very high dash ratio
    if gap_count < config.min_gaps_for_pattern or dash_ratio >= config.dash_ratio_continuous:
        confidence = min(1.0, dash_ratio + 0.1)
        return LinetypeName.CONTINUOUS, confidence

    # Check for complex patterns (CENTER, PHANTOM, etc.)
    if len(pattern) >= 6:
        # Count dash/gap alternation patterns
        dash_lengths = [p for p in pattern if p > 0]
        gap_lengths = [-p for p in pattern if p < 0]

        if len(dash_lengths) >= 2:
            # Check for short-long-short pattern (CENTER/PHANTOM)
            dash_variance = np.var(dash_lengths) if len(dash_lengths) > 1 else 0

            if dash_variance > 0.1 * np.mean(dash_lengths) ** 2:
                # High variance suggests CENTER or PHANTOM pattern
                if len(pattern) >= 8:
                    return LinetypeName.PHANTOM, 0.7
                return LinetypeName.CENTER, 0.75

    # Dotted: very low dash ratio (mostly gaps with tiny dots)
    if dash_ratio < config.dash_ratio_dotted:
        return LinetypeName.DOTTED, 0.8

    # Hidden vs Dashed based on dash ratio
    if config.dash_ratio_hidden[0] <= dash_ratio <= config.dash_ratio_hidden[1]:
        return LinetypeName.HIDDEN, 0.8

    if config.dash_ratio_dashed[0] <= dash_ratio <= config.dash_ratio_dashed[1]:
        return LinetypeName.DASHED, 0.85

    # Default to DASHED for ambiguous cases with gaps
    if gap_count >= config.min_gaps_for_pattern:
        return LinetypeName.DASHED, 0.6

    return LinetypeName.CONTINUOUS, 0.5


def detect_linetype(
    binary_image: np.ndarray,
    line_start: Tuple[int, int],
    line_end: Tuple[int, int],
    config: Optional[LinetypeConfig] = None,
) -> LinetypeResult:
    """
    Detect the linetype of a line segment in a binary image.

    This is the main entry point for linetype detection. It analyzes the
    pixel pattern along the line to determine if it's continuous, dashed,
    dotted, or another pattern.

    Args:
        binary_image: Binary image (grayscale, 0=background, 255=foreground)
        line_start: (x, y) start point of the line
        line_end: (x, y) end point of the line
        config: Optional configuration parameters

    Returns:
        LinetypeResult with detected linetype and confidence

    Example:
        >>> result = detect_linetype(image, (10, 10), (200, 10))
        >>> print(f"Linetype: {result.linetype.value}, confidence: {result.confidence:.2f}")
    """
    if config is None:
        config = LinetypeConfig()

    # Calculate line length
    dx = line_end[0] - line_start[0]
    dy = line_end[1] - line_start[1]
    length = int(math.sqrt(dx * dx + dy * dy))

    # Short lines default to continuous
    if length < config.min_line_length:
        return LinetypeResult(
            linetype=LinetypeName.CONTINUOUS,
            confidence=0.5,
            pattern=[],
            dash_ratio=1.0,
            gap_count=0,
        )

    try:
        # Sample intensity along the line
        intensities = sample_line_intensity(
            binary_image,
            line_start,
            line_end,
            sample_width=config.sample_width,
        )

        # Smooth intensities
        if config.smoothing_kernel > 1:
            kernel = np.ones(config.smoothing_kernel) / config.smoothing_kernel
            intensities = np.convolve(intensities, kernel, mode='same')

        # Detect gaps
        gaps = detect_gaps(intensities, gap_threshold=config.gap_threshold)

        # Calculate dash ratio
        total_gap_length = sum(end - start for start, end in gaps)
        dash_ratio = 1.0 - (total_gap_length / len(intensities)) if len(intensities) > 0 else 1.0

        # Analyze with FFT for periodicity
        dominant_freq, _ = analyze_pattern_fft(intensities, config.smoothing_kernel)

        # Extract pattern
        pattern = extract_pattern(intensities, gaps)

        # Classify linetype
        linetype, confidence = classify_linetype(
            dash_ratio=dash_ratio,
            gap_count=len(gaps),
            dominant_freq=dominant_freq,
            pattern=pattern,
            config=config,
        )

        return LinetypeResult(
            linetype=linetype,
            confidence=confidence,
            pattern=pattern,
            dash_ratio=dash_ratio,
            gap_count=len(gaps),
            dominant_frequency=dominant_freq,
        )

    except Exception as e:
        logger.warning("linetype_detection_failed", error=str(e))
        return LinetypeResult(
            linetype=LinetypeName.CONTINUOUS,
            confidence=0.3,
        )


def detect_linetypes_batch(
    binary_image: np.ndarray,
    lines: List[Tuple[Tuple[int, int], Tuple[int, int]]],
    config: Optional[LinetypeConfig] = None,
) -> List[LinetypeResult]:
    """
    Detect linetypes for multiple lines efficiently.

    Args:
        binary_image: Binary image
        lines: List of ((x1, y1), (x2, y2)) line segments
        config: Optional configuration

    Returns:
        List of LinetypeResult for each line
    """
    if config is None:
        config = LinetypeConfig()

    results = []
    for start, end in lines:
        result = detect_linetype(binary_image, start, end, config)
        results.append(result)

    return results


def map_to_autocad_linetype(linetype: LinetypeName) -> str:
    """
    Map LinetypeName to AutoCAD linetype string for entity creation.

    Args:
        linetype: Detected linetype

    Returns:
        AutoCAD linetype name (e.g., "DASHED", "HIDDEN")
    """
    return linetype.value


# Mapping for common pattern names
PATTERN_NAME_MAP = {
    "continuous": LinetypeName.CONTINUOUS,
    "solid": LinetypeName.CONTINUOUS,
    "dashed": LinetypeName.DASHED,
    "dash": LinetypeName.DASHED,
    "hidden": LinetypeName.HIDDEN,
    "invisible": LinetypeName.HIDDEN,
    "center": LinetypeName.CENTER,
    "centerline": LinetypeName.CENTER,
    "phantom": LinetypeName.PHANTOM,
    "dotted": LinetypeName.DOTTED,
    "dot": LinetypeName.DOTTED,
    "dashdot": LinetypeName.DASHDOT,
    "divide": LinetypeName.DIVIDE,
    "border": LinetypeName.BORDER,
}


def parse_linetype_name(name: str) -> LinetypeName:
    """
    Parse a linetype name string to LinetypeName enum.

    Args:
        name: Linetype name (case-insensitive)

    Returns:
        LinetypeName enum value
    """
    normalized = name.lower().strip()
    return PATTERN_NAME_MAP.get(normalized, LinetypeName.CONTINUOUS)
