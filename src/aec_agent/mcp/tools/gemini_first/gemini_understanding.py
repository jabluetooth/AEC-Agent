"""
Phase 2: Gemini Understanding - Drawing Analysis with Vision AI.

This module uses Gemini Vision to fully analyze a drawing BEFORE any extraction.
It identifies drawing type, elements, text, symbols, and recommends extraction strategies.

This is the "brain" of the Gemini-First pipeline - understanding first, then extracting.

Key Outputs:
- Drawing type classification (floor_plan, electrical, mechanical, etc.)
- Element inventory with pixel locations
- Text content with semantic corrections
- Symbol identification with subtypes
- Calibration hints for coordinate mapping
- Extraction strategy recommendations

Usage:
    >>> analyzer = DrawingAnalyzer()
    >>> analysis = await analyzer.analyze(image_path)
    >>> print(f"Type: {analysis.drawing_type}, Elements: {analysis.total_elements}")
"""

from __future__ import annotations

import base64
import json
import re
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Literal, Optional

import structlog

from aec_agent.config.settings import get_settings

logger = structlog.get_logger(__name__)


class DrawingType(str, Enum):
    """Types of technical drawings."""
    FLOOR_PLAN = "floor_plan"
    ELECTRICAL = "electrical"
    MECHANICAL = "mechanical"
    PLUMBING = "plumbing"
    FIRE_ALARM = "fire_alarm"
    REFLECTED_CEILING = "reflected_ceiling"
    SITE_PLAN = "site_plan"
    DETAIL = "detail"
    SECTION = "section"
    ELEVATION = "elevation"
    SCHEDULE = "schedule"
    DIAGRAM = "diagram"
    OTHER = "other"


class ExtractionStrategy(str, Enum):
    """Extraction strategies for different element types."""
    DIRECT = "direct"  # Use Gemini coordinates directly
    GUIDED_RASTERIZATION = "guided_rasterization"  # Use Raster Design VTools
    SELECTIVE_OPENCV = "selective_opencv"  # Use OpenCV for specific regions
    HYBRID = "hybrid"  # Combination of strategies
    SKIP = "skip"  # Don't extract


@dataclass
class RegionBounds:
    """Bounding box for a region in pixel coordinates."""
    x1: int
    y1: int
    x2: int
    y2: int
    content: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "bounds": [self.x1, self.y1, self.x2, self.y2],
            "content": self.content,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "RegionBounds":
        bounds = data.get("bounds", [0, 0, 0, 0])
        return cls(
            x1=bounds[0] if len(bounds) > 0 else 0,
            y1=bounds[1] if len(bounds) > 1 else 0,
            x2=bounds[2] if len(bounds) > 2 else 0,
            y2=bounds[3] if len(bounds) > 3 else 0,
            content=data.get("content"),
        )


@dataclass
class DetectedLine:
    """Line detected in the drawing."""
    start: tuple[int, int]
    end: tuple[int, int]
    line_type: str = "other"  # wall, duct, pipe, wire, dimension, leader, other
    linetype: str = "continuous"  # continuous, dashed, dotted, hidden, center
    layer_suggestion: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "start": list(self.start),
            "end": list(self.end),
            "type": self.line_type,
            "linetype": self.linetype,
            "layer_suggestion": self.layer_suggestion,
        }


@dataclass
class DetectedArc:
    """Arc detected in the drawing."""
    center: tuple[int, int]
    radius: int
    start_angle: float
    end_angle: float
    arc_type: str = "other"  # door_swing, curved_wall, other

    def to_dict(self) -> dict:
        return {
            "center": list(self.center),
            "radius": self.radius,
            "start_angle": self.start_angle,
            "end_angle": self.end_angle,
            "type": self.arc_type,
        }


@dataclass
class DetectedCircle:
    """Circle detected in the drawing."""
    center: tuple[int, int]
    radius: int
    circle_type: str = "other"  # column, equipment, symbol, other

    def to_dict(self) -> dict:
        return {
            "center": list(self.center),
            "radius": self.radius,
            "type": self.circle_type,
        }


@dataclass
class DetectedText:
    """Text detected in the drawing."""
    content: str
    position: tuple[int, int]
    height_px: int = 12
    text_type: str = "note"  # room_name, dimension, equipment_tag, note, title, label
    associated_with: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "content": self.content,
            "position": list(self.position),
            "height_px": self.height_px,
            "type": self.text_type,
            "associated_with": self.associated_with,
        }


@dataclass
class DetectedSymbol:
    """Symbol detected in the drawing."""
    symbol_type: str  # diffuser, outlet, switch, valve, fixture, detector, device, equipment
    subtype: Optional[str] = None  # supply_square, duplex, gate_valve, etc.
    position: tuple[int, int] = (0, 0)
    rotation: float = 0
    size: Optional[str] = None
    tag: Optional[str] = None
    associated_text: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "type": self.symbol_type,
            "subtype": self.subtype,
            "position": list(self.position),
            "rotation": self.rotation,
            "size": self.size,
            "tag": self.tag,
            "associated_text": self.associated_text,
        }


@dataclass
class DetectedDimension:
    """Dimension detected in the drawing."""
    value: str  # As shown on drawing, e.g., "20'-0\""
    numeric_value: Optional[float] = None  # Parsed number
    unit: str = "inches"
    start: tuple[int, int] = (0, 0)
    end: tuple[int, int] = (0, 0)
    text_position: tuple[int, int] = (0, 0)

    def to_dict(self) -> dict:
        return {
            "value": self.value,
            "numeric_value": self.numeric_value,
            "unit": self.unit,
            "start": list(self.start),
            "end": list(self.end),
            "text_position": list(self.text_position),
        }


@dataclass
class CalibrationHint:
    """Hint for scale calibration."""
    hint_type: str  # dimension, known_object, grid_spacing, sheet_border
    description: str
    pixel_measurement: float
    real_measurement: str
    confidence: float = 0.8

    def to_dict(self) -> dict:
        return {
            "type": self.hint_type,
            "description": self.description,
            "pixel_measurement": self.pixel_measurement,
            "real_measurement": self.real_measurement,
            "confidence": self.confidence,
        }


@dataclass
class SpecialRegion:
    """Region requiring special extraction handling."""
    bounds: tuple[int, int, int, int]
    strategy: str  # guided_rasterization, selective_opencv
    reason: str
    expected_pattern: Optional[str] = None  # parallel_lines, hatching, grid

    def to_dict(self) -> dict:
        return {
            "bounds": list(self.bounds),
            "strategy": self.strategy,
            "reason": self.reason,
            "expected_pattern": self.expected_pattern,
        }


@dataclass
class DrawingElements:
    """All elements detected in the drawing."""
    lines: list[DetectedLine] = field(default_factory=list)
    arcs: list[DetectedArc] = field(default_factory=list)
    circles: list[DetectedCircle] = field(default_factory=list)
    text: list[DetectedText] = field(default_factory=list)
    symbols: list[DetectedSymbol] = field(default_factory=list)
    dimensions: list[DetectedDimension] = field(default_factory=list)

    @property
    def total_count(self) -> int:
        return (
            len(self.lines) + len(self.arcs) + len(self.circles) +
            len(self.text) + len(self.symbols) + len(self.dimensions)
        )

    def to_dict(self) -> dict:
        return {
            "lines": {
                "count": len(self.lines),
                "items": [line.to_dict() for line in self.lines],
            },
            "arcs": {
                "count": len(self.arcs),
                "items": [arc.to_dict() for arc in self.arcs],
            },
            "circles": {
                "count": len(self.circles),
                "items": [circle.to_dict() for circle in self.circles],
            },
            "text": {
                "count": len(self.text),
                "items": [text.to_dict() for text in self.text],
            },
            "symbols": {
                "count": len(self.symbols),
                "items": [symbol.to_dict() for symbol in self.symbols],
            },
            "dimensions": {
                "count": len(self.dimensions),
                "items": [dim.to_dict() for dim in self.dimensions],
            },
        }


@dataclass
class ExtractionStrategyConfig:
    """Recommended extraction strategy for the drawing."""
    primary_strategy: str = "direct"  # direct, guided_rasterization, hybrid
    rationale: str = ""
    per_element_strategy: dict[str, str] = field(default_factory=dict)
    special_regions: list[SpecialRegion] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "primary_strategy": self.primary_strategy,
            "rationale": self.rationale,
            "per_element_strategy": self.per_element_strategy,
            "special_regions": [r.to_dict() for r in self.special_regions],
        }


@dataclass
class DrawingAnalysis:
    """Complete analysis of a technical drawing."""
    # Drawing metadata
    drawing_type: str
    scale: Optional[str] = None
    sheet_size: Optional[str] = None
    units: str = "imperial"
    complexity: str = "medium"
    description: str = ""

    # Detected regions
    title_block: Optional[RegionBounds] = None
    drawing_area: Optional[RegionBounds] = None
    legend: Optional[RegionBounds] = None
    notes_area: Optional[RegionBounds] = None

    # Detected elements
    elements: DrawingElements = field(default_factory=DrawingElements)

    # Calibration hints
    calibration_hints: list[CalibrationHint] = field(default_factory=list)

    # Extraction strategy
    extraction_strategy: ExtractionStrategyConfig = field(default_factory=ExtractionStrategyConfig)

    # Raw response for debugging
    raw_response: Optional[dict] = None

    # Provider info
    provider: str = ""
    model: str = ""

    @property
    def total_elements(self) -> int:
        return self.elements.total_count

    def to_dict(self) -> dict:
        return {
            "drawing_analysis": {
                "type": self.drawing_type,
                "scale": self.scale,
                "sheet_size": self.sheet_size,
                "units": self.units,
                "complexity": self.complexity,
                "description": self.description,
            },
            "regions": {
                "title_block": self.title_block.to_dict() if self.title_block else None,
                "drawing_area": self.drawing_area.to_dict() if self.drawing_area else None,
                "legend": self.legend.to_dict() if self.legend else None,
                "notes": self.notes_area.to_dict() if self.notes_area else None,
            },
            "elements": self.elements.to_dict(),
            "calibration_hints": [h.to_dict() for h in self.calibration_hints],
            "extraction_strategy": self.extraction_strategy.to_dict(),
            "metadata": {
                "provider": self.provider,
                "model": self.model,
                "total_elements": self.total_elements,
            },
        }


# Gemini prompt for drawing analysis
UNDERSTANDING_PROMPT = """Analyze this technical drawing. Return ONLY JSON (no markdown):

{
  "drawing_analysis": {"type": "floor_plan|electrical|mechanical|plumbing|fire_alarm|reflected_ceiling|site_plan|detail|section|elevation|schedule|diagram|other", "scale": "1/4\"=1'-0\" or null", "sheet_size": "ARCH D or null", "units": "imperial|metric", "complexity": "simple|medium|complex", "description": "brief"},
  "regions": {"title_block": {"bounds": [x1,y1,x2,y2], "content": "..."} or null, "drawing_area": {"bounds": [x1,y1,x2,y2]}, "legend": null, "notes": null},
  "elements": {
    "lines": [{"start": [x,y], "end": [x,y], "type": "wall|duct|pipe|wire|dimension|leader|other", "linetype": "continuous|dashed|dotted|hidden|center", "layer_suggestion": "NCS name"}],
    "arcs": [{"center": [x,y], "radius": px, "start_angle": deg, "end_angle": deg, "type": "door_swing|curved_wall|other"}],
    "circles": [{"center": [x,y], "radius": px, "type": "column|equipment|symbol|other"}],
    "text": [{"content": "text (correct errors)", "position": [x,y], "height_px": px, "type": "room_name|dimension|equipment_tag|note|title|label|other", "associated_with": "labeled element"}],
    "symbols": [{"type": "diffuser|outlet|switch|valve|fixture|detector|device|equipment|other", "subtype": "specific", "position": [x,y], "rotation": deg, "size": "24x24", "tag": "tag", "associated_text": ["labels"]}],
    "dimensions": [{"value": "shown", "numeric_value": num, "unit": "inches|feet|mm|m", "start": [x,y], "end": [x,y], "text_position": [x,y]}]
  },
  "calibration_hints": [{"type": "dimension|known_object|grid_spacing|sheet_border", "description": "what", "pixel_measurement": px, "real_measurement": "value+units", "confidence": 0-1}],
  "extraction_strategy": {"primary_strategy": "direct|guided_rasterization|hybrid", "rationale": "why", "per_element_strategy": {"walls": "direct|guided|skip", "ductwork": "...", "piping": "...", "electrical": "...", "text": "direct", "symbols": "direct", "dimensions": "direct"}, "special_regions": [{"bounds": [x1,y1,x2,y2], "strategy": "guided_rasterization|selective_opencv", "reason": "why", "expected_pattern": "parallel_lines|hatching|grid|curves"}]}
}

Rules: Pixel coords from top-left (0,0). Correct OCR errors semantically. Identify symbol subtypes. Max 50 items per category.
"""


class DrawingAnalyzer:
    """
    Analyzes drawings using Gemini Vision AI.

    This is the core of Phase 2 - understanding the drawing before extraction.
    Supports Gemini Pro Vision with fallback to other providers.

    Args:
        model: Gemini model to use (gemini-2.0-flash recommended - has free tier).
        temperature: LLM temperature (lower = more deterministic).
        max_output_tokens: Maximum tokens in response.

    Example:
        >>> analyzer = DrawingAnalyzer()
        >>> analysis = await analyzer.analyze(Path("drawing.png"))
        >>> print(f"Type: {analysis.drawing_type}")
        >>> print(f"Elements: {analysis.total_elements}")
    """

    def __init__(
        self,
        model: str = "gemini-2.0-flash",
        temperature: float = 0.1,
        max_output_tokens: int = 8192,
    ):
        self.settings = get_settings()
        self.model_name = model
        self.temperature = temperature
        self.max_output_tokens = max_output_tokens
        self._gemini_model = None

    def _get_model_name(self) -> str:
        """Get the mapped model name for the new SDK."""
        # Map legacy model names to Gemini 2.0 Flash (has free tier)
        # Pro models don't have free tier - always use Flash for cost savings
        model_mapping = {
            "gemini-1.5-pro": "gemini-2.0-flash",
            "gemini-1.5-flash": "gemini-2.0-flash",
            "gemini-1.5-pro-latest": "gemini-2.0-flash",
            "gemini-1.5-flash-latest": "gemini-2.0-flash",
            "gemini-pro-latest": "gemini-2.0-flash",  # Pro has no free tier
            "gemini-flash-latest": "gemini-2.0-flash",
        }

        model_name = self.model_name
        if model_name in model_mapping:
            original_model = model_name
            model_name = model_mapping[model_name]
            logger.info(
                "gemini_model_mapped",
                original_model=original_model,
                mapped_model=model_name,
                reason="Using Gemini 2.x equivalent"
            )

        return model_name

    def _get_gemini_client(self):
        """Get the Gemini client (new SDK)."""
        if self._gemini_model is None:
            try:
                from . import get_gemini_client
                self._gemini_model = get_gemini_client(self.settings.gemini_api_key)
                logger.debug(
                    "gemini_client_initialized",
                    model=self._get_model_name(),
                )
            except ImportError:
                logger.error(
                    "google-genai not installed. "
                    "Install with: pip install google-genai"
                )
                raise

        return self._gemini_model

    def _load_image(self, image_path: Path) -> "PIL.Image.Image":
        """Load image from path."""
        try:
            from PIL import Image
            return Image.open(image_path)
        except ImportError:
            raise ImportError("Pillow not installed. Install with: pip install Pillow")

    def _parse_response(self, response_text: str) -> dict:
        """Parse JSON response from Gemini, handling potential formatting issues."""
        # Try direct JSON parse first
        try:
            return json.loads(response_text)
        except json.JSONDecodeError:
            pass

        # Try to extract JSON from markdown code blocks
        json_match = re.search(r'```(?:json)?\s*([\s\S]*?)\s*```', response_text)
        if json_match:
            json_str = json_match.group(1)
            try:
                return json.loads(json_str)
            except json.JSONDecodeError:
                # Try to repair truncated JSON by closing brackets
                repaired = self._repair_truncated_json(json_str)
                if repaired:
                    return repaired

        # Try to find raw JSON object
        json_match = re.search(r'\{[\s\S]*\}', response_text)
        if json_match:
            try:
                return json.loads(json_match.group())
            except json.JSONDecodeError:
                repaired = self._repair_truncated_json(json_match.group())
                if repaired:
                    return repaired

        raise ValueError(f"Failed to parse response as JSON: {response_text[:500]}...")

    def _repair_truncated_json(self, json_str: str) -> Optional[dict]:
        """Attempt to repair truncated JSON by closing open brackets."""
        # Count open brackets
        open_braces = json_str.count('{') - json_str.count('}')
        open_brackets = json_str.count('[') - json_str.count(']')

        if open_braces <= 0 and open_brackets <= 0:
            return None

        # Try to close the JSON properly
        repaired = json_str.rstrip()
        # Remove trailing comma if present
        if repaired.endswith(','):
            repaired = repaired[:-1]
        # Close arrays then objects
        repaired += ']' * open_brackets + '}' * open_braces

        try:
            return json.loads(repaired)
        except json.JSONDecodeError:
            return None

    def _parse_analysis(self, data: dict) -> DrawingAnalysis:
        """Convert raw JSON to DrawingAnalysis dataclass."""
        # Parse drawing analysis
        da = data.get("drawing_analysis", {})

        # Parse regions
        regions = data.get("regions", {})
        title_block = None
        drawing_area = None
        legend = None
        notes_area = None

        if regions.get("title_block"):
            title_block = RegionBounds.from_dict(regions["title_block"])
        if regions.get("drawing_area"):
            drawing_area = RegionBounds.from_dict(regions["drawing_area"])
        if regions.get("legend"):
            legend = RegionBounds.from_dict(regions["legend"])
        if regions.get("notes"):
            notes_area = RegionBounds.from_dict(regions["notes"])

        # Parse elements
        elements_data = data.get("elements", {})
        elements = DrawingElements()

        # Parse lines
        for line_data in elements_data.get("lines", []):
            if isinstance(line_data, dict) and "start" in line_data and "end" in line_data:
                # Ensure exactly 2 coordinates (Gemini sometimes returns more)
                start = line_data["start"][:2] if len(line_data["start"]) >= 2 else [0, 0]
                end = line_data["end"][:2] if len(line_data["end"]) >= 2 else [0, 0]
                elements.lines.append(DetectedLine(
                    start=tuple(start),
                    end=tuple(end),
                    line_type=line_data.get("type", "other"),
                    linetype=line_data.get("linetype", "continuous"),
                    layer_suggestion=line_data.get("layer_suggestion"),
                ))

        # Helper to safely extract 2D coordinates
        def _safe_coord(data, key, default=(0, 0)):
            val = data.get(key, default)
            if isinstance(val, (list, tuple)) and len(val) >= 2:
                return tuple(val[:2])
            return default

        # Parse arcs
        for arc_data in elements_data.get("arcs", []):
            if isinstance(arc_data, dict) and "center" in arc_data:
                elements.arcs.append(DetectedArc(
                    center=_safe_coord(arc_data, "center"),
                    radius=arc_data.get("radius", 0),
                    start_angle=arc_data.get("start_angle", 0),
                    end_angle=arc_data.get("end_angle", 360),
                    arc_type=arc_data.get("type", "other"),
                ))

        # Parse circles
        for circle_data in elements_data.get("circles", []):
            if isinstance(circle_data, dict) and "center" in circle_data:
                elements.circles.append(DetectedCircle(
                    center=_safe_coord(circle_data, "center"),
                    radius=circle_data.get("radius", 0),
                    circle_type=circle_data.get("type", "other"),
                ))

        # Parse text
        for text_data in elements_data.get("text", []):
            if isinstance(text_data, dict) and "content" in text_data:
                elements.text.append(DetectedText(
                    content=text_data["content"],
                    position=_safe_coord(text_data, "position"),
                    height_px=text_data.get("height_px", 12),
                    text_type=text_data.get("type", "note"),
                    associated_with=text_data.get("associated_with"),
                ))

        # Parse symbols
        for symbol_data in elements_data.get("symbols", []):
            if isinstance(symbol_data, dict) and "type" in symbol_data:
                elements.symbols.append(DetectedSymbol(
                    symbol_type=symbol_data["type"],
                    subtype=symbol_data.get("subtype"),
                    position=_safe_coord(symbol_data, "position"),
                    rotation=symbol_data.get("rotation", 0),
                    size=symbol_data.get("size"),
                    tag=symbol_data.get("tag"),
                    associated_text=symbol_data.get("associated_text", []),
                ))

        # Parse dimensions
        for dim_data in elements_data.get("dimensions", []):
            if isinstance(dim_data, dict) and "value" in dim_data:
                elements.dimensions.append(DetectedDimension(
                    value=dim_data["value"],
                    numeric_value=dim_data.get("numeric_value"),
                    unit=dim_data.get("unit", "inches"),
                    start=_safe_coord(dim_data, "start"),
                    end=_safe_coord(dim_data, "end"),
                    text_position=_safe_coord(dim_data, "text_position"),
                ))

        # Parse calibration hints
        calibration_hints = []
        for hint_data in data.get("calibration_hints", []):
            if isinstance(hint_data, dict) and "type" in hint_data:
                calibration_hints.append(CalibrationHint(
                    hint_type=hint_data["type"],
                    description=hint_data.get("description", ""),
                    pixel_measurement=hint_data.get("pixel_measurement", 0),
                    real_measurement=hint_data.get("real_measurement", ""),
                    confidence=hint_data.get("confidence", 0.8),
                ))

        # Parse extraction strategy
        strategy_data = data.get("extraction_strategy", {})
        special_regions = []
        for region_data in strategy_data.get("special_regions", []):
            if isinstance(region_data, dict) and "bounds" in region_data:
                special_regions.append(SpecialRegion(
                    bounds=tuple(region_data["bounds"]),
                    strategy=region_data.get("strategy", "guided_rasterization"),
                    reason=region_data.get("reason", ""),
                    expected_pattern=region_data.get("expected_pattern"),
                ))

        extraction_strategy = ExtractionStrategyConfig(
            primary_strategy=strategy_data.get("primary_strategy", "direct"),
            rationale=strategy_data.get("rationale", ""),
            per_element_strategy=strategy_data.get("per_element_strategy", {}),
            special_regions=special_regions,
        )

        return DrawingAnalysis(
            drawing_type=da.get("type", "other"),
            scale=da.get("scale"),
            sheet_size=da.get("sheet_size"),
            units=da.get("units", "imperial"),
            complexity=da.get("complexity", "medium"),
            description=da.get("description", ""),
            title_block=title_block,
            drawing_area=drawing_area,
            legend=legend,
            notes_area=notes_area,
            elements=elements,
            calibration_hints=calibration_hints,
            extraction_strategy=extraction_strategy,
            raw_response=data,
            provider="gemini",
            model=self.model_name,
        )

    async def analyze(
        self,
        image_path: Path,
        context: Optional[str] = None,
    ) -> DrawingAnalysis:
        """
        Analyze a drawing image using Gemini Vision.

        Args:
            image_path: Path to the drawing image (PNG, JPG, etc.)
            context: Optional context/hints about the drawing

        Returns:
            DrawingAnalysis with complete understanding of the drawing

        Raises:
            FileNotFoundError: If image file doesn't exist
            ValueError: If Gemini response cannot be parsed
            RuntimeError: If Gemini API call fails
        """
        image_path = Path(image_path)
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")

        logger.info(
            "analyzing_drawing",
            image_path=str(image_path),
            model=self.model_name,
        )

        # Load image
        image = self._load_image(image_path)

        # Build prompt
        prompt = UNDERSTANDING_PROMPT
        if context:
            prompt = f"## CONTEXT\n{context}\n\n{prompt}"

        try:
            # Get Gemini client (new SDK)
            client = self._get_gemini_client()

            # Call Gemini Vision with retry for rate limits
            from . import gemini_call_with_retry

            response_text = await gemini_call_with_retry(
                client,
                [prompt, image],
                generation_config={
                    "temperature": self.temperature,
                    "max_output_tokens": self.max_output_tokens,
                },
                model_name=self._get_model_name(),
            )
            logger.debug(
                "gemini_response_received",
                response_length=len(response_text),
            )

            parsed = self._parse_response(response_text)
            analysis = self._parse_analysis(parsed)

            logger.info(
                "drawing_analysis_complete",
                drawing_type=analysis.drawing_type,
                complexity=analysis.complexity,
                total_elements=analysis.total_elements,
                lines=len(analysis.elements.lines),
                text=len(analysis.elements.text),
                symbols=len(analysis.elements.symbols),
            )

            return analysis

        except Exception as e:
            logger.exception("gemini_analysis_failed", error=str(e))
            raise RuntimeError(f"Failed to analyze drawing: {e}")


async def analyze_drawing(
    image_path: Path | str,
    model: str = "gemini-2.0-flash",
    context: Optional[str] = None,
) -> DrawingAnalysis:
    """
    Convenience function to analyze a drawing image.

    Args:
        image_path: Path to the drawing image
        model: Gemini model to use
        context: Optional context about the drawing

    Returns:
        DrawingAnalysis with complete understanding

    Example:
        >>> analysis = await analyze_drawing("floor_plan.png")
        >>> print(f"Type: {analysis.drawing_type}")
        >>> print(f"Found {len(analysis.elements.symbols)} symbols")
    """
    analyzer = DrawingAnalyzer(model=model)
    return await analyzer.analyze(Path(image_path), context=context)
