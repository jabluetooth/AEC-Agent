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
UNDERSTANDING_PROMPT = """
Analyze this technical/engineering drawing image and provide a complete understanding.

## INSTRUCTIONS

You are looking at an ORIGINAL, high-quality rendering of a technical drawing (architectural, electrical, mechanical, plumbing, or similar).

Provide your analysis as JSON with the following structure:

```json
{
  "drawing_analysis": {
    "type": "<floor_plan|electrical|mechanical|plumbing|fire_alarm|reflected_ceiling|site_plan|detail|section|elevation|schedule|diagram|other>",
    "scale": "<scale notation if visible, e.g., '1/4\" = 1'-0\"' or null>",
    "sheet_size": "<detected sheet size, e.g., 'ARCH D (24x36)' or null>",
    "units": "<imperial|metric>",
    "complexity": "<simple|medium|complex>",
    "description": "<brief description of what this drawing shows>"
  },

  "regions": {
    "title_block": {"bounds": [x1, y1, x2, y2], "content": "<extracted title block info>"} or null,
    "drawing_area": {"bounds": [x1, y1, x2, y2]},
    "legend": {"bounds": [x1, y1, x2, y2]} or null,
    "notes": {"bounds": [x1, y1, x2, y2]} or null
  },

  "elements": {
    "lines": [
      {
        "start": [x, y],
        "end": [x, y],
        "type": "<wall|duct|pipe|wire|dimension|leader|other>",
        "linetype": "<continuous|dashed|dotted|hidden|center>",
        "layer_suggestion": "<NCS layer name>"
      }
    ],
    "arcs": [
      {"center": [x, y], "radius": <pixels>, "start_angle": <deg>, "end_angle": <deg>, "type": "<door_swing|curved_wall|other>"}
    ],
    "circles": [
      {"center": [x, y], "radius": <pixels>, "type": "<column|equipment|symbol|other>"}
    ],
    "text": [
      {
        "content": "<text content - correct any obvious errors>",
        "position": [x, y],
        "height_px": <approximate height in pixels>,
        "type": "<room_name|dimension|equipment_tag|note|title|label|other>",
        "associated_with": "<what this text labels, if applicable>"
      }
    ],
    "symbols": [
      {
        "type": "<diffuser|outlet|switch|valve|fixture|detector|device|equipment|other>",
        "subtype": "<specific subtype, e.g., 'supply_square', 'duplex', 'gate_valve'>",
        "position": [x, y],
        "rotation": <degrees, 0 if upright>,
        "size": "<size if visible, e.g., '24x24'>",
        "tag": "<equipment tag if visible>",
        "associated_text": ["<nearby text labels>"]
      }
    ],
    "dimensions": [
      {
        "value": "<dimension value as shown>",
        "numeric_value": <parsed number>,
        "unit": "<inches|feet|mm|m>",
        "start": [x, y],
        "end": [x, y],
        "text_position": [x, y]
      }
    ]
  },

  "calibration_hints": [
    {
      "type": "<dimension|known_object|grid_spacing|sheet_border>",
      "description": "<what this is>",
      "pixel_measurement": <pixels>,
      "real_measurement": "<value with units>",
      "confidence": <0-1>
    }
  ],

  "extraction_strategy": {
    "primary_strategy": "<direct|guided_rasterization|hybrid>",
    "rationale": "<why this strategy>",
    "per_element_strategy": {
      "walls": "<direct|guided|skip>",
      "ductwork": "<direct|guided|skip>",
      "piping": "<direct|guided|skip>",
      "electrical": "<direct|guided|skip>",
      "text": "direct",
      "symbols": "direct",
      "dimensions": "direct"
    },
    "special_regions": [
      {
        "bounds": [x1, y1, x2, y2],
        "strategy": "<guided_rasterization|selective_opencv>",
        "reason": "<why special handling>",
        "expected_pattern": "<parallel_lines|hatching|grid|curves>"
      }
    ]
  }
}
```

## IMPORTANT NOTES

1. **Coordinates**: Use pixel coordinates from top-left origin (0,0).
2. **Text content**: Read text semantically - correct obvious OCR-like errors based on context.
3. **Symbols**: Identify symbol TYPE and SUBTYPE (e.g., diffuser → supply_square_diffuser).
4. **Line types**: Distinguish solid, dashed, dotted, hidden, center lines.
5. **Associations**: Note which text labels which elements.
6. **Calibration**: Identify ANY measurable references for scale calibration.
7. **Limit results**: For large drawings, include the most important/representative items (up to 50 per category).

Return ONLY the JSON, no markdown formatting or explanation.
"""


class DrawingAnalyzer:
    """
    Analyzes drawings using Gemini Vision AI.

    This is the core of Phase 2 - understanding the drawing before extraction.
    Supports Gemini Pro Vision with fallback to other providers.

    Args:
        model: Gemini model to use (gemini-pro-latest, gemini-flash-latest, or legacy gemini-1.5-pro/flash which auto-map to 2.x).
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
        model: str = "gemini-pro-latest",
        temperature: float = 0.1,
        max_output_tokens: int = 8192,
    ):
        self.settings = get_settings()
        self.model_name = model
        self.temperature = temperature
        self.max_output_tokens = max_output_tokens
        self._gemini_model = None

    async def _get_gemini_model(self):
        """Lazy-load Gemini model."""
        if self._gemini_model is None:
            try:
                import google.generativeai as genai

                if not self.settings.gemini_api_key:
                    raise ValueError(
                        "GEMINI_API_KEY not set. Set it in environment or .env file."
                    )

                genai.configure(api_key=self.settings.gemini_api_key)
                
                # Map legacy model names to current Gemini 2.x models
                # Gemini 1.5 models have been deprecated and replaced with Gemini 2.x
                model_name = self.model_name
                model_mapping = {
                    "gemini-1.5-pro": "gemini-pro-latest",  # Maps to gemini-2.5-pro
                    "gemini-1.5-flash": "gemini-flash-latest",  # Maps to gemini-2.5-flash
                    "gemini-1.5-pro-latest": "gemini-pro-latest",
                    "gemini-1.5-flash-latest": "gemini-flash-latest",
                }
                
                # Apply mapping if needed
                if model_name in model_mapping:
                    original_model = model_name
                    model_name = model_mapping[model_name]
                    logger.info(
                        "gemini_model_mapped",
                        original_model=original_model,
                        mapped_model=model_name,
                        reason="Gemini 1.5 models deprecated, using Gemini 2.x equivalent"
                    )
                
                self._gemini_model = genai.GenerativeModel(model_name)
                logger.debug(
                    "gemini_model_initialized",
                    model=model_name,
                    original_request=self.model_name,
                )
            except ImportError:
                logger.error(
                    "google-generativeai not installed. "
                    "Install with: pip install google-generativeai"
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
            try:
                return json.loads(json_match.group(1))
            except json.JSONDecodeError:
                pass

        # Try to find raw JSON object
        json_match = re.search(r'\{[\s\S]*\}', response_text)
        if json_match:
            try:
                return json.loads(json_match.group())
            except json.JSONDecodeError:
                pass

        raise ValueError(f"Failed to parse response as JSON: {response_text[:500]}...")

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
                elements.lines.append(DetectedLine(
                    start=tuple(line_data["start"]),
                    end=tuple(line_data["end"]),
                    line_type=line_data.get("type", "other"),
                    linetype=line_data.get("linetype", "continuous"),
                    layer_suggestion=line_data.get("layer_suggestion"),
                ))

        # Parse arcs
        for arc_data in elements_data.get("arcs", []):
            if isinstance(arc_data, dict) and "center" in arc_data:
                elements.arcs.append(DetectedArc(
                    center=tuple(arc_data["center"]),
                    radius=arc_data.get("radius", 0),
                    start_angle=arc_data.get("start_angle", 0),
                    end_angle=arc_data.get("end_angle", 360),
                    arc_type=arc_data.get("type", "other"),
                ))

        # Parse circles
        for circle_data in elements_data.get("circles", []):
            if isinstance(circle_data, dict) and "center" in circle_data:
                elements.circles.append(DetectedCircle(
                    center=tuple(circle_data["center"]),
                    radius=circle_data.get("radius", 0),
                    circle_type=circle_data.get("type", "other"),
                ))

        # Parse text
        for text_data in elements_data.get("text", []):
            if isinstance(text_data, dict) and "content" in text_data:
                elements.text.append(DetectedText(
                    content=text_data["content"],
                    position=tuple(text_data.get("position", [0, 0])),
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
                    position=tuple(symbol_data.get("position", [0, 0])),
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
                    start=tuple(dim_data.get("start", [0, 0])),
                    end=tuple(dim_data.get("end", [0, 0])),
                    text_position=tuple(dim_data.get("text_position", [0, 0])),
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
            # Get Gemini model
            model = await self._get_gemini_model()

            # Call Gemini Vision
            response = await model.generate_content_async(
                [prompt, image],
                generation_config={
                    "temperature": self.temperature,
                    "max_output_tokens": self.max_output_tokens,
                }
            )

            # Parse response
            response_text = response.text
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
    model: str = "gemini-pro-latest",
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
