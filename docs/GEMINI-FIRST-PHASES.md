# Gemini-First Architecture: Phase-by-Phase Implementation Guide

## Overview

This document breaks down the Gemini-First PDF-to-AutoCAD pipeline into **6 implementation phases**, each with clear deliverables, dependencies, and success criteria.

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                         IMPLEMENTATION ROADMAP                                  │
└─────────────────────────────────────────────────────────────────────────────────┘

Phase 1: PDF Intake & Rendering (No preprocessing)
    │
    ▼
Phase 2: Gemini Understanding (Analyze original image)
    │
    ▼
Phase 3: Coordinate Calibration (Map pixels to DWG units)
    │
    ▼
Phase 4: Adaptive Extraction (Direct / Guided / Selective)
    │
    ▼
Phase 5: AutoCAD Entity Creation (Draw in DWG)
    │
    ▼
Phase 6: Validation & Self-Correction (Gemini verifies)
```

---

# Phase 1: PDF Intake & Rendering

## Purpose
Render PDF to high-quality image WITHOUT destroying information. No bitonal conversion.

## Input
- PDF file path
- Page number (for multi-page PDFs)
- Target DPI (default: 300)

## Output
- High-quality RGB or grayscale image (PNG/TIFF)
- Image metadata (dimensions, DPI, color mode)

## Implementation

```python
# phase1_pdf_intake.py

import fitz  # PyMuPDF
from PIL import Image
from pathlib import Path
from dataclasses import dataclass

@dataclass
class PDFRenderResult:
    image_path: Path
    width_px: int
    height_px: int
    dpi: int
    color_mode: str  # "RGB" or "L" (grayscale)
    page_number: int
    original_pdf: Path

async def render_pdf_high_quality(
    pdf_path: Path,
    page_number: int = 0,
    dpi: int = 300,
    output_dir: Path = None
) -> PDFRenderResult:
    """
    Render PDF page to high-quality image.

    CRITICAL: Do NOT convert to bitonal. Keep full color/grayscale.
    """
    doc = fitz.open(pdf_path)
    page = doc[page_number]

    # Calculate zoom for target DPI (PDF default is 72 DPI)
    zoom = dpi / 72.0
    matrix = fitz.Matrix(zoom, zoom)

    # Render page to pixmap (RGB by default)
    pixmap = page.get_pixmap(matrix=matrix, alpha=False)

    # Convert to PIL Image
    image = Image.frombytes("RGB", [pixmap.width, pixmap.height], pixmap.samples)

    # Determine if grayscale would preserve all info
    # (technical drawings are often black & white)
    if is_effectively_grayscale(image):
        image = image.convert("L")
        color_mode = "L"
    else:
        color_mode = "RGB"

    # Save as PNG (lossless)
    output_dir = output_dir or Path("./temp")
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / f"{pdf_path.stem}_page{page_number}.png"
    image.save(output_path, "PNG")

    doc.close()

    return PDFRenderResult(
        image_path=output_path,
        width_px=pixmap.width,
        height_px=pixmap.height,
        dpi=dpi,
        color_mode=color_mode,
        page_number=page_number,
        original_pdf=pdf_path
    )

def is_effectively_grayscale(image: Image.Image, threshold: float = 0.01) -> bool:
    """Check if RGB image is effectively grayscale."""
    import numpy as np
    arr = np.array(image)
    r, g, b = arr[:,:,0], arr[:,:,1], arr[:,:,2]
    diff = np.abs(r.astype(float) - g) + np.abs(g.astype(float) - b)
    return (diff.mean() / 255.0) < threshold
```

## Key Principles

| DO | DON'T |
|----|-------|
| Keep RGB or high-quality grayscale | Convert to bitonal (1-bit) |
| Use 300+ DPI | Use less than 200 DPI |
| Save as PNG (lossless) | Save as JPEG (lossy) |
| Preserve anti-aliasing | Apply threshold |

## Success Criteria

- [ ] PDF renders at specified DPI
- [ ] No information loss (color/grayscale preserved)
- [ ] Output image is lossless format
- [ ] Metadata captured correctly

## Dependencies
- PyMuPDF (`fitz`)
- Pillow (`PIL`)

---

# Phase 2: Gemini Understanding

## Purpose
Use Gemini Vision to fully understand the drawing BEFORE any extraction. This is the "brain" phase.

## Input
- High-quality image from Phase 1
- Drawing context (optional hints from user)

## Output
- Complete drawing analysis (JSON)
- Element inventory with locations
- Recommended extraction strategy per element

## Implementation

```python
# phase2_gemini_understanding.py

import google.generativeai as genai
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional
import json

@dataclass
class DrawingAnalysis:
    drawing_type: str  # "floor_plan", "electrical", "mechanical", etc.
    scale: Optional[str]  # "1/4\" = 1'-0\"" or None
    sheet_size: Optional[str]  # "ARCH D", "24x36", etc.
    units: str  # "imperial" or "metric"
    complexity: str  # "simple", "medium", "complex"

    regions: dict  # title_block, drawing_area, legend bounds
    elements: dict  # categorized elements with locations

    extraction_strategy: dict  # recommended approach per element type
    calibration_hints: list  # known dimensions for scale calibration

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
    "title_block": {"bounds": [x1, y1, x2, y2], "content": "<extracted title block info>"},
    "drawing_area": {"bounds": [x1, y1, x2, y2]},
    "legend": {"bounds": [x1, y1, x2, y2]} or null,
    "notes": {"bounds": [x1, y1, x2, y2]} or null,
    "schedules": [{"bounds": [x1, y1, x2, y2], "type": "<schedule type>"}]
  },

  "elements": {
    "lines": {
      "count": <approximate count>,
      "categories": [
        {
          "type": "<wall|duct|pipe|wire|dimension|leader|other>",
          "linetype": "<continuous|dashed|dotted|hidden|center>",
          "layer_suggestion": "<NCS layer name>",
          "items": [
            {"start": [x, y], "end": [x, y]}
          ]
        }
      ]
    },

    "arcs": {
      "count": <approximate count>,
      "items": [
        {"center": [x, y], "radius": <pixels>, "start_angle": <deg>, "end_angle": <deg>, "type": "<door_swing|curved_wall|other>"}
      ]
    },

    "circles": {
      "count": <approximate count>,
      "items": [
        {"center": [x, y], "radius": <pixels>, "type": "<column|equipment|symbol|other>"}
      ]
    },

    "text": {
      "count": <approximate count>,
      "items": [
        {
          "content": "<text content - correct any obvious errors>",
          "position": [x, y],
          "height_px": <approximate height in pixels>,
          "type": "<room_name|dimension|equipment_tag|note|title|label|other>",
          "associated_with": "<what this text labels, if applicable>"
        }
      ]
    },

    "symbols": {
      "count": <approximate count>,
      "items": [
        {
          "type": "<diffuser|outlet|switch|valve|fixture|detector|device|equipment|other>",
          "subtype": "<specific subtype, e.g., 'supply_square', 'duplex', 'gate_valve'>",
          "position": [x, y],
          "rotation": <degrees, 0 if upright>,
          "size": "<size if visible, e.g., '24x24'>",
          "tag": "<equipment tag if visible>",
          "associated_text": ["<nearby text labels>"]
        }
      ]
    },

    "dimensions": {
      "count": <approximate count>,
      "items": [
        {
          "value": "<dimension value as shown>",
          "numeric_value": <parsed number>,
          "unit": "<inches|feet|mm|m>",
          "start": [x, y],
          "end": [x, y],
          "text_position": [x, y]
        }
      ]
    }
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
    "primary_strategy": "<direct_extraction|guided_rasterization|hybrid>",
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
        "reason": "<why special handling>"
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

Return ONLY the JSON, no markdown formatting.
"""

async def analyze_drawing_with_gemini(
    image_path: Path,
    model: str = "gemini-2.0-flash"  # Flash has free tier, Pro does not
) -> DrawingAnalysis:
    """
    Use Gemini Vision to fully understand the drawing.
    """
    # Configure Gemini
    genai.configure(api_key=os.environ["GEMINI_API_KEY"])
    model = genai.GenerativeModel(model)

    # Load image
    image = Image.open(image_path)

    # Send to Gemini
    response = await model.generate_content_async(
        [UNDERSTANDING_PROMPT, image],
        generation_config={
            "temperature": 0.1,  # Low temperature for consistent output
            "max_output_tokens": 8192
        }
    )

    # Parse JSON response
    try:
        analysis_json = json.loads(response.text)
    except json.JSONDecodeError:
        # Try to extract JSON from response
        import re
        json_match = re.search(r'\{[\s\S]*\}', response.text)
        if json_match:
            analysis_json = json.loads(json_match.group())
        else:
            raise ValueError(f"Failed to parse Gemini response as JSON: {response.text[:500]}")

    # Convert to dataclass
    return DrawingAnalysis(
        drawing_type=analysis_json["drawing_analysis"]["type"],
        scale=analysis_json["drawing_analysis"].get("scale"),
        sheet_size=analysis_json["drawing_analysis"].get("sheet_size"),
        units=analysis_json["drawing_analysis"].get("units", "imperial"),
        complexity=analysis_json["drawing_analysis"].get("complexity", "medium"),
        regions=analysis_json.get("regions", {}),
        elements=analysis_json.get("elements", {}),
        extraction_strategy=analysis_json.get("extraction_strategy", {}),
        calibration_hints=analysis_json.get("calibration_hints", [])
    )
```

## What Gemini Identifies

| Category | Details Extracted |
|----------|------------------|
| **Drawing Type** | Floor plan, electrical, mechanical, plumbing, etc. |
| **Scale** | Scale notation if visible |
| **Regions** | Title block, drawing area, legend, notes bounds |
| **Lines** | Start/end points, type (wall/duct/pipe), linetype |
| **Arcs** | Center, radius, angles, type (door swing, curve) |
| **Circles** | Center, radius, type (column, equipment) |
| **Text** | Content (corrected), position, type, associations |
| **Symbols** | Type, subtype, position, rotation, associated text |
| **Dimensions** | Value, start/end points, numeric parsing |
| **Calibration** | Known measurements for scale calculation |

## Success Criteria

- [ ] Gemini returns valid JSON
- [ ] Drawing type correctly identified
- [ ] All major elements inventoried
- [ ] Text content semantically corrected
- [ ] Symbols typed and subtyped
- [ ] Calibration hints provided

## Dependencies
- `google-generativeai`
- Gemini API key

---

# Phase 3: Coordinate Calibration

## Purpose
Convert Gemini's pixel coordinates to AutoCAD DWG units using calibration hints.

## Input
- Drawing analysis from Phase 2
- Image dimensions from Phase 1

## Output
- Scale factor (units per pixel)
- Coordinate transform function

## Implementation

```python
# phase3_coordinate_calibration.py

from dataclasses import dataclass
from typing import Callable, Tuple
import re

@dataclass
class ScaleCalibration:
    scale_factor: float  # DWG units per pixel
    units: str  # "inches", "feet", "mm", "m"
    confidence: float  # 0-1
    method: str  # "dimension", "scale_notation", "sheet_size", "manual"

    # Coordinate transform
    image_height: int  # For Y-axis flip

    def to_dwg(self, pixel_x: float, pixel_y: float) -> Tuple[float, float]:
        """Convert pixel coordinates to DWG coordinates."""
        dwg_x = pixel_x * self.scale_factor
        dwg_y = (self.image_height - pixel_y) * self.scale_factor  # Y-axis flip
        return (dwg_x, dwg_y)

def calibrate_from_analysis(
    analysis: DrawingAnalysis,
    image_width: int,
    image_height: int,
    image_dpi: int = 300
) -> ScaleCalibration:
    """
    Determine scale factor using multiple methods, ranked by confidence.
    """
    calibrations = []

    # Method 1: Known dimension (highest confidence)
    for hint in analysis.calibration_hints:
        if hint["type"] == "dimension" and hint.get("pixel_measurement") and hint.get("real_measurement"):
            real_value, real_unit = parse_measurement(hint["real_measurement"])
            if real_value and hint["pixel_measurement"] > 0:
                factor = real_value / hint["pixel_measurement"]
                calibrations.append(ScaleCalibration(
                    scale_factor=factor,
                    units=real_unit,
                    confidence=hint.get("confidence", 0.9),
                    method="dimension",
                    image_height=image_height
                ))

    # Method 2: Scale notation (high confidence)
    if analysis.scale:
        factor, units = parse_scale_notation(analysis.scale, image_dpi)
        if factor:
            calibrations.append(ScaleCalibration(
                scale_factor=factor,
                units=units,
                confidence=0.85,
                method="scale_notation",
                image_height=image_height
            ))

    # Method 3: Sheet size (medium confidence)
    if analysis.sheet_size:
        sheet_width, sheet_height = parse_sheet_size(analysis.sheet_size)
        if sheet_width:
            factor_x = sheet_width / image_width
            factor_y = sheet_height / image_height
            factor = (factor_x + factor_y) / 2
            calibrations.append(ScaleCalibration(
                scale_factor=factor,
                units="inches",
                confidence=0.7,
                method="sheet_size",
                image_height=image_height
            ))

    # Method 4: Default based on DPI (low confidence)
    # Assume 1 pixel = 1/DPI inches on paper
    default_factor = 1.0 / image_dpi
    calibrations.append(ScaleCalibration(
        scale_factor=default_factor,
        units="inches",
        confidence=0.3,
        method="dpi_default",
        image_height=image_height
    ))

    # Return highest confidence calibration
    calibrations.sort(key=lambda c: c.confidence, reverse=True)
    return calibrations[0]

def parse_measurement(measurement_str: str) -> Tuple[float, str]:
    """Parse measurement string like '20'-0\"' or '100mm' into (value, unit)."""
    # Feet and inches: 20'-0", 20' 0", 20'0"
    feet_inches = re.match(r"(\d+)['\s-]+(\d+)?[\""]?", measurement_str)
    if feet_inches:
        feet = float(feet_inches.group(1))
        inches = float(feet_inches.group(2) or 0)
        return (feet * 12 + inches, "inches")

    # Plain inches: 24", 24 in
    inches_match = re.match(r"(\d+\.?\d*)\s*[\""]|(\d+\.?\d*)\s*in", measurement_str)
    if inches_match:
        return (float(inches_match.group(1) or inches_match.group(2)), "inches")

    # Millimeters: 100mm, 100 mm
    mm_match = re.match(r"(\d+\.?\d*)\s*mm", measurement_str, re.IGNORECASE)
    if mm_match:
        return (float(mm_match.group(1)), "mm")

    # Meters: 1.5m, 1.5 m
    m_match = re.match(r"(\d+\.?\d*)\s*m(?!m)", measurement_str, re.IGNORECASE)
    if m_match:
        return (float(m_match.group(1)) * 1000, "mm")  # Convert to mm

    # Plain number (assume inches)
    num_match = re.match(r"(\d+\.?\d*)", measurement_str)
    if num_match:
        return (float(num_match.group(1)), "inches")

    return (None, None)

def parse_scale_notation(scale_str: str, dpi: int) -> Tuple[float, str]:
    """
    Parse scale notation like '1/4" = 1'-0"' into scale factor.

    Returns: (units_per_pixel, unit_name)
    """
    # Format: 1/4" = 1'-0" means 0.25" on paper = 12" in real life
    # So 1 paper inch = 48 real inches

    # Match: fraction" = feet'-inches"
    match = re.match(
        r"(\d+)/(\d+)[\"\"]\s*=\s*(\d+)['\s-]+(\d+)?[\""]?",
        scale_str
    )
    if match:
        paper_inches = float(match.group(1)) / float(match.group(2))
        real_feet = float(match.group(3))
        real_inches = float(match.group(4) or 0)
        real_total_inches = real_feet * 12 + real_inches

        # paper_inches on paper = real_total_inches in reality
        # 1 paper inch = real_total_inches / paper_inches real inches
        inches_per_paper_inch = real_total_inches / paper_inches

        # At given DPI, 1 pixel = 1/dpi paper inches
        # So 1 pixel = (1/dpi) * inches_per_paper_inch real inches
        inches_per_pixel = (1.0 / dpi) * inches_per_paper_inch

        return (inches_per_pixel, "inches")

    # Match: 1:100 (metric)
    match = re.match(r"1\s*:\s*(\d+)", scale_str)
    if match:
        scale_ratio = float(match.group(1))
        # 1 paper unit = scale_ratio real units
        # At given DPI, 1 pixel = 1/dpi paper inches = 25.4/dpi paper mm
        mm_per_pixel = (25.4 / dpi) * scale_ratio
        return (mm_per_pixel, "mm")

    return (None, None)

def parse_sheet_size(sheet_str: str) -> Tuple[float, float]:
    """Parse sheet size to (width_inches, height_inches)."""
    SHEET_SIZES = {
        "ARCH A": (9, 12),
        "ARCH B": (12, 18),
        "ARCH C": (18, 24),
        "ARCH D": (24, 36),
        "ARCH E": (36, 48),
        "ANSI A": (8.5, 11),
        "ANSI B": (11, 17),
        "ANSI C": (17, 22),
        "ANSI D": (22, 34),
        "ANSI E": (34, 44),
        "A0": (33.1, 46.8),
        "A1": (23.4, 33.1),
        "A2": (16.5, 23.4),
        "A3": (11.7, 16.5),
        "A4": (8.3, 11.7),
    }

    # Try to match known sheet sizes
    for name, (w, h) in SHEET_SIZES.items():
        if name.lower() in sheet_str.lower():
            return (w, h)

    # Try to parse dimensions: 24x36, 24" x 36"
    match = re.search(r"(\d+)\s*[x×]\s*(\d+)", sheet_str)
    if match:
        return (float(match.group(1)), float(match.group(2)))

    return (None, None)
```

## Calibration Methods (Priority Order)

| Priority | Method | Confidence | Source |
|----------|--------|------------|--------|
| 1 | Known Dimension | 90% | Dimension line with pixel measurement |
| 2 | Scale Notation | 85% | "1/4" = 1'-0"" in title block |
| 3 | Sheet Size | 70% | "ARCH D (24x36)" |
| 4 | DPI Default | 30% | Fallback: 1 pixel = 1/DPI inches |

## Success Criteria

- [ ] Scale factor calculated
- [ ] Y-axis flip handled (PDF top-down → AutoCAD bottom-up)
- [ ] Unit system determined (imperial/metric)
- [ ] Transform function tested

---

# Phase 4: Adaptive Extraction

## Purpose
Choose and execute the optimal extraction strategy for each element based on Gemini's analysis.

## Input
- Drawing analysis from Phase 2
- Scale calibration from Phase 3

## Output
- List of entities to create (with DWG coordinates)

## Three Extraction Strategies

### Strategy A: Direct Extraction

**When**: Clean drawings, simple geometry, text, symbols

```python
# phase4a_direct_extraction.py

from dataclasses import dataclass
from typing import List

@dataclass
class EntityToCreate:
    entity_type: str  # "line", "arc", "circle", "mtext", "block"
    layer: str
    properties: dict  # type-specific properties
    source: str = "direct"  # extraction method used

async def direct_extraction(
    analysis: DrawingAnalysis,
    calibration: ScaleCalibration
) -> List[EntityToCreate]:
    """
    Create entities directly from Gemini's coordinate output.
    No image processing. Pure semantic-to-entity mapping.
    """
    entities = []

    # Extract lines
    for category in analysis.elements.get("lines", {}).get("categories", []):
        layer = category.get("layer_suggestion", "0")
        linetype = category.get("linetype", "CONTINUOUS")

        for item in category.get("items", []):
            start_dwg = calibration.to_dwg(*item["start"])
            end_dwg = calibration.to_dwg(*item["end"])

            entities.append(EntityToCreate(
                entity_type="line",
                layer=layer,
                properties={
                    "start": start_dwg,
                    "end": end_dwg,
                    "linetype": linetype.upper()
                }
            ))

    # Extract arcs
    for arc in analysis.elements.get("arcs", {}).get("items", []):
        center_dwg = calibration.to_dwg(*arc["center"])
        radius_dwg = arc["radius"] * calibration.scale_factor

        entities.append(EntityToCreate(
            entity_type="arc",
            layer=get_layer_for_type(arc.get("type", "other")),
            properties={
                "center": center_dwg,
                "radius": radius_dwg,
                "start_angle": arc["start_angle"],
                "end_angle": arc["end_angle"]
            }
        ))

    # Extract circles
    for circle in analysis.elements.get("circles", {}).get("items", []):
        center_dwg = calibration.to_dwg(*circle["center"])
        radius_dwg = circle["radius"] * calibration.scale_factor

        entities.append(EntityToCreate(
            entity_type="circle",
            layer=get_layer_for_type(circle.get("type", "other")),
            properties={
                "center": center_dwg,
                "radius": radius_dwg
            }
        ))

    # Extract text
    for text in analysis.elements.get("text", {}).get("items", []):
        position_dwg = calibration.to_dwg(*text["position"])
        height_dwg = text.get("height_px", 12) * calibration.scale_factor

        entities.append(EntityToCreate(
            entity_type="mtext",
            layer=get_text_layer(text.get("type", "note")),
            properties={
                "content": text["content"],
                "position": position_dwg,
                "height": max(height_dwg, 0.1)  # Minimum height
            }
        ))

    # Extract symbols as blocks
    for symbol in analysis.elements.get("symbols", {}).get("items", []):
        position_dwg = calibration.to_dwg(*symbol["position"])
        block_name = map_symbol_to_block(symbol["type"], symbol.get("subtype"))

        entities.append(EntityToCreate(
            entity_type="block",
            layer=get_symbol_layer(symbol["type"]),
            properties={
                "block_name": block_name,
                "position": position_dwg,
                "rotation": symbol.get("rotation", 0),
                "scale": 1.0,
                "attributes": {
                    "TAG": symbol.get("tag", ""),
                    "SIZE": symbol.get("size", "")
                }
            }
        ))

    return entities
```

### Strategy B: Guided Rasterization

**When**: Complex curves, connected paths, precise tracing needed

```python
# phase4b_guided_rasterization.py

@dataclass
class RasterCommand:
    tool: str  # "VFPLINE", "VFCONTOUR", "VARC", "VCIRCLE", "VLINE"
    start_point: Tuple[float, float]  # DWG coordinates
    layer: str
    options: dict

async def guided_rasterization(
    analysis: DrawingAnalysis,
    calibration: ScaleCalibration,
    image_path: Path
) -> List[RasterCommand]:
    """
    Generate Raster Design commands for complex regions.
    Gemini provides guidance; VTools do precise tracing.
    """
    commands = []

    # Process special regions that need guided rasterization
    for region in analysis.extraction_strategy.get("special_regions", []):
        if region["strategy"] != "guided_rasterization":
            continue

        # Get Gemini's guidance for this region
        region_guidance = await get_region_guidance(analysis, region, calibration)

        for path in region_guidance.get("paths", []):
            start_dwg = calibration.to_dwg(*path["start_pixel"])

            tool = map_path_type_to_vtool(path["type"])

            commands.append(RasterCommand(
                tool=tool,
                start_point=start_dwg,
                layer=path.get("layer", "0"),
                options={
                    "expected_end": calibration.to_dwg(*path["end_pixel"]) if "end_pixel" in path else None,
                    "gap_jump": path.get("gap_jump", 3),
                    "corner_threshold": path.get("corner_threshold", 45)
                }
            ))

    return commands

def map_path_type_to_vtool(path_type: str) -> str:
    """Map path type to Raster Design VTool command."""
    mapping = {
        "polyline": "VFPLINE",
        "contour": "VFCONTOUR",
        "arc": "VARC",
        "circle": "VCIRCLE",
        "line": "VLINE",
        "rectangle": "VRECT"
    }
    return mapping.get(path_type, "VFPLINE")
```

### Strategy C: Selective OpenCV

**When**: Repetitive patterns, hatching, dense details

```python
# phase4c_selective_opencv.py

import cv2
import numpy as np

async def selective_opencv(
    analysis: DrawingAnalysis,
    calibration: ScaleCalibration,
    original_image: np.ndarray
) -> List[EntityToCreate]:
    """
    Use OpenCV on specific regions where it excels.
    Gemini pre-filters; OpenCV batch-detects; Gemini validates.
    """
    entities = []

    for region in analysis.extraction_strategy.get("special_regions", []):
        if region["strategy"] != "selective_opencv":
            continue

        # 1. Crop region from ORIGINAL image (not bitonal!)
        bounds = region["bounds"]
        cropped = original_image[bounds[1]:bounds[3], bounds[0]:bounds[2]]

        # 2. Minimal preprocessing - adaptive threshold, NOT bitonal
        gray = cv2.cvtColor(cropped, cv2.COLOR_RGB2GRAY)
        processed = cv2.adaptiveThreshold(
            gray, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            blockSize=11, C=2
        )

        # 3. Detect based on expected pattern
        pattern = region.get("expected_pattern", "lines")

        if pattern == "parallel_lines":
            detected = detect_parallel_lines_opencv(
                processed,
                expected_angle=region.get("angle", 0),
                expected_spacing=region.get("spacing")
            )
        elif pattern == "hatching":
            detected = detect_hatching_opencv(processed)
        elif pattern == "grid":
            detected = detect_grid_opencv(processed)
        else:
            continue

        # 4. Convert to entities with offset for region
        offset_x, offset_y = bounds[0], bounds[1]
        for item in detected:
            if item["type"] == "line":
                start = (item["start"][0] + offset_x, item["start"][1] + offset_y)
                end = (item["end"][0] + offset_x, item["end"][1] + offset_y)

                entities.append(EntityToCreate(
                    entity_type="line",
                    layer=region.get("layer", "0"),
                    properties={
                        "start": calibration.to_dwg(*start),
                        "end": calibration.to_dwg(*end),
                        "linetype": "CONTINUOUS"
                    },
                    source="selective_opencv"
                ))

    return entities
```

## Extraction Strategy Selection

```python
# phase4_coordinator.py

async def extract_all(
    analysis: DrawingAnalysis,
    calibration: ScaleCalibration,
    image_path: Path
) -> Tuple[List[EntityToCreate], List[RasterCommand]]:
    """
    Coordinate all extraction strategies based on Gemini's recommendations.
    """
    entities = []
    raster_commands = []

    # Load original image for selective OpenCV
    original_image = cv2.imread(str(image_path))

    # 1. Direct extraction for most elements
    primary = analysis.extraction_strategy.get("primary_strategy", "direct_extraction")

    if primary in ["direct_extraction", "hybrid"]:
        direct_entities = await direct_extraction(analysis, calibration)
        entities.extend(direct_entities)

    # 2. Guided rasterization for complex regions
    if primary in ["guided_rasterization", "hybrid"]:
        raster_cmds = await guided_rasterization(analysis, calibration, image_path)
        raster_commands.extend(raster_cmds)

    # 3. Selective OpenCV for special regions
    for region in analysis.extraction_strategy.get("special_regions", []):
        if region["strategy"] == "selective_opencv":
            opencv_entities = await selective_opencv(analysis, calibration, original_image)
            entities.extend(opencv_entities)

    return entities, raster_commands
```

## Success Criteria

- [ ] All elements extracted according to strategy
- [ ] Coordinates correctly transformed
- [ ] Block names mapped correctly
- [ ] Layers assigned according to NCS

---

# Phase 5: AutoCAD Entity Creation

## Purpose
Create entities in AutoCAD using the existing MCP tools.

## Input
- List of entities to create from Phase 4
- Optional: Raster commands for guided rasterization

## Output
- Entities created in AutoCAD
- Entity handles for validation

## Implementation

```python
# phase5_autocad_creation.py

from typing import List, Dict

async def create_entities_in_autocad(
    entities: List[EntityToCreate],
    raster_commands: List[RasterCommand] = None
) -> Dict[str, List[str]]:
    """
    Create all extracted entities in AutoCAD.
    Returns: {"success": [handles], "failed": [descriptions]}
    """
    results = {"success": [], "failed": []}

    # Ensure required layers exist
    layers_needed = set(e.layer for e in entities)
    await ensure_layers_exist(layers_needed)

    # Create each entity
    for entity in entities:
        try:
            if entity.entity_type == "line":
                result = await draw_line(
                    start=entity.properties["start"],
                    end=entity.properties["end"],
                    layer=entity.layer,
                    linetype=entity.properties.get("linetype", "CONTINUOUS")
                )

            elif entity.entity_type == "arc":
                result = await draw_arc(
                    center=entity.properties["center"],
                    radius=entity.properties["radius"],
                    start_angle=entity.properties["start_angle"],
                    end_angle=entity.properties["end_angle"],
                    layer=entity.layer
                )

            elif entity.entity_type == "circle":
                result = await draw_circle(
                    center=entity.properties["center"],
                    radius=entity.properties["radius"],
                    layer=entity.layer
                )

            elif entity.entity_type == "mtext":
                result = await draw_mtext(
                    content=entity.properties["content"],
                    position=entity.properties["position"],
                    height=entity.properties["height"],
                    layer=entity.layer
                )

            elif entity.entity_type == "block":
                result = await insert_block(
                    block_name=entity.properties["block_name"],
                    position=entity.properties["position"],
                    rotation=entity.properties.get("rotation", 0),
                    scale=entity.properties.get("scale", 1.0),
                    layer=entity.layer,
                    attributes=entity.properties.get("attributes", {})
                )

            if result.get("success"):
                results["success"].append(result.get("handle", "unknown"))
            else:
                results["failed"].append(f"{entity.entity_type} at {entity.properties}")

        except Exception as e:
            results["failed"].append(f"{entity.entity_type}: {str(e)}")

    # Execute raster commands if any
    if raster_commands:
        for cmd in raster_commands:
            try:
                result = await execute_raster_command(cmd)
                if result.get("success"):
                    results["success"].append(result.get("handle", "raster"))
                else:
                    results["failed"].append(f"Raster {cmd.tool}: {result.get('error')}")
            except Exception as e:
                results["failed"].append(f"Raster {cmd.tool}: {str(e)}")

    return results

async def ensure_layers_exist(layers: set):
    """Create any layers that don't exist."""
    for layer in layers:
        if layer == "0":
            continue
        # Use layer creation MCP tool
        await create_layer_if_not_exists(layer)

async def execute_raster_command(cmd: RasterCommand) -> dict:
    """Execute a Raster Design VTool command."""
    # First ensure raster image is attached
    # Then execute the VTool command via SendStringToExecute
    command_str = build_raster_command_string(cmd)
    return await send_command_to_autocad(command_str)
```

## Success Criteria

- [ ] All direct extraction entities created
- [ ] Raster commands executed (if applicable)
- [ ] Failed entities logged for review
- [ ] Entity handles captured for validation

---

# Phase 6: Validation & Self-Correction

## Purpose
Use Gemini to verify extracted entities against the original PDF and correct any issues.

## Input
- Original image from Phase 1
- List of created entities from Phase 5
- Drawing analysis from Phase 2

## Output
- Validation result (approved/issues_found)
- Corrections applied (if needed)

## Implementation

```python
# phase6_validation.py

from dataclasses import dataclass
from typing import List, Optional

@dataclass
class ValidationResult:
    status: str  # "approved" | "issues_found"
    accuracy_estimate: float  # 0-100
    issues: List[dict]
    corrections: List[dict]
    iteration: int

VALIDATION_PROMPT = """
Compare the extracted entities against the original drawing.

## ORIGINAL DRAWING
[Image attached]

## EXTRACTED ENTITIES
```json
{entities_json}
```

## INSTRUCTIONS

Check for these issues:

1. **MISSING ELEMENTS**: Elements in original not in extracted list
2. **EXTRA ELEMENTS**: False positives that shouldn't exist
3. **POSITION ERRORS**: Elements in wrong location (>5% deviation)
4. **TEXT ERRORS**: Wrong content, missing characters, OCR errors
5. **SYMBOL ERRORS**: Wrong type, wrong attributes, wrong orientation
6. **CONNECTIVITY ISSUES**: Gaps or overlaps that shouldn't exist

For each issue, provide a correction action:
- ADD: Add missing element
- REMOVE: Remove false positive
- MODIFY: Change element properties
- REPLACE: Replace with correct element

Return JSON:
```json
{
  "validation_status": "approved" | "issues_found",
  "accuracy_estimate": <0-100>,
  "issues": [
    {
      "type": "missing_element|extra_element|position_error|text_error|symbol_error|connectivity",
      "description": "<what's wrong>",
      "location": [x, y],
      "severity": "critical|major|minor"
    }
  ],
  "corrections": [
    {
      "action": "ADD|REMOVE|MODIFY|REPLACE",
      "entity_type": "<line|arc|circle|mtext|block>",
      "properties": { ... }
    }
  ]
}
```
"""

async def validate_extraction(
    original_image_path: Path,
    created_entities: List[dict],
    analysis: DrawingAnalysis,
    calibration: ScaleCalibration,
    max_iterations: int = 3
) -> ValidationResult:
    """
    Validate extracted entities against original and self-correct.
    """
    iteration = 0
    current_entities = created_entities.copy()

    while iteration < max_iterations:
        iteration += 1

        # Format entities for prompt
        entities_json = json.dumps(current_entities, indent=2)

        # Send to Gemini for validation
        image = Image.open(original_image_path)

        response = await gemini_model.generate_content_async(
            [VALIDATION_PROMPT.replace("{entities_json}", entities_json), image],
            generation_config={"temperature": 0.1}
        )

        validation = json.loads(response.text)

        if validation["validation_status"] == "approved":
            return ValidationResult(
                status="approved",
                accuracy_estimate=validation["accuracy_estimate"],
                issues=[],
                corrections=[],
                iteration=iteration
            )

        # Apply corrections
        if validation["corrections"]:
            applied = await apply_corrections(
                validation["corrections"],
                calibration
            )

            # Update entity list
            current_entities = await get_current_entities()
        else:
            # No corrections but not approved - manual review needed
            return ValidationResult(
                status="manual_review_needed",
                accuracy_estimate=validation["accuracy_estimate"],
                issues=validation["issues"],
                corrections=[],
                iteration=iteration
            )

    # Max iterations reached
    return ValidationResult(
        status="max_iterations",
        accuracy_estimate=validation.get("accuracy_estimate", 0),
        issues=validation.get("issues", []),
        corrections=[],
        iteration=iteration
    )

async def apply_corrections(
    corrections: List[dict],
    calibration: ScaleCalibration
) -> List[dict]:
    """Apply Gemini's corrections to AutoCAD."""
    applied = []

    for correction in corrections:
        action = correction["action"]

        if action == "ADD":
            # Add missing element
            entity = EntityToCreate(
                entity_type=correction["entity_type"],
                layer=correction.get("layer", "0"),
                properties=correction["properties"]
            )
            result = await create_entity(entity)
            applied.append({"action": "ADD", "result": result})

        elif action == "REMOVE":
            # Remove false positive
            result = await erase_entity(correction["entity_handle"])
            applied.append({"action": "REMOVE", "result": result})

        elif action == "MODIFY":
            # Modify existing entity
            result = await modify_entity(
                correction["entity_handle"],
                correction["new_properties"]
            )
            applied.append({"action": "MODIFY", "result": result})

        elif action == "REPLACE":
            # Replace entity
            await erase_entity(correction["entity_handle"])
            entity = EntityToCreate(
                entity_type=correction["entity_type"],
                layer=correction.get("layer", "0"),
                properties=correction["properties"]
            )
            result = await create_entity(entity)
            applied.append({"action": "REPLACE", "result": result})

    return applied
```

## Validation Loop Flow

```
┌──────────────────────────────────────────────────────────────┐
│                    VALIDATION LOOP                           │
└──────────────────────────────────────────────────────────────┘

        ┌─────────────────────────────┐
        │  Created Entities (Phase 5) │
        └──────────────┬──────────────┘
                       │
            ┌──────────▼──────────┐
        ┌───│  Gemini Validation  │◀────────────────┐
        │   └──────────┬──────────┘                 │
        │              │                            │
        │   ┌──────────▼──────────┐                 │
        │   │  Status Check       │                 │
        │   └──────────┬──────────┘                 │
        │              │                            │
        │   ┌──────────┴──────────┐                 │
        │   │                     │                 │
        │   ▼                     ▼                 │
        │ APPROVED            ISSUES_FOUND          │
        │   │                     │                 │
        │   ▼                     ▼                 │
        │ ┌─────────┐    ┌───────────────────┐      │
        │ │  Done!  │    │ Apply Corrections │──────┘
        │ │  Exit   │    │ (iteration < max) │
        │ └─────────┘    └───────────────────┘
        │
        │ (iteration >= max)
        │   │
        │   ▼
        │ ┌──────────────────────┐
        └─│  Manual Review       │
          │  (Return issues)     │
          └──────────────────────┘
```

## Success Criteria

- [ ] Validation prompt returns valid JSON
- [ ] Corrections applied successfully
- [ ] Approved within 3 iterations (typical case)
- [ ] Issues logged for manual review if needed

---

# Complete Pipeline Integration

## Main Entry Point

```python
# gemini_first_pipeline.py

from pathlib import Path
from dataclasses import dataclass

@dataclass
class PipelineResult:
    success: bool
    entities_created: int
    accuracy: float
    validation_iterations: int
    errors: List[str]
    warnings: List[str]

async def gemini_first_vectorize(
    pdf_path: Path,
    page_number: int = 0,
    dpi: int = 300,
    max_validation_iterations: int = 3
) -> PipelineResult:
    """
    Complete Gemini-First PDF to AutoCAD vectorization pipeline.
    """
    errors = []
    warnings = []

    try:
        # ══════════════════════════════════════════════════════════════════
        # PHASE 1: PDF Intake (No preprocessing)
        # ══════════════════════════════════════════════════════════════════
        print("Phase 1: Rendering PDF...")
        render_result = await render_pdf_high_quality(
            pdf_path=pdf_path,
            page_number=page_number,
            dpi=dpi
        )

        # ══════════════════════════════════════════════════════════════════
        # PHASE 2: Gemini Understanding
        # ══════════════════════════════════════════════════════════════════
        print("Phase 2: Analyzing with Gemini...")
        analysis = await analyze_drawing_with_gemini(
            image_path=render_result.image_path
        )
        print(f"  - Drawing type: {analysis.drawing_type}")
        print(f"  - Complexity: {analysis.complexity}")
        print(f"  - Strategy: {analysis.extraction_strategy.get('primary_strategy')}")

        # ══════════════════════════════════════════════════════════════════
        # PHASE 3: Coordinate Calibration
        # ══════════════════════════════════════════════════════════════════
        print("Phase 3: Calibrating coordinates...")
        calibration = calibrate_from_analysis(
            analysis=analysis,
            image_width=render_result.width_px,
            image_height=render_result.height_px,
            image_dpi=dpi
        )
        print(f"  - Scale factor: {calibration.scale_factor:.6f} {calibration.units}/px")
        print(f"  - Method: {calibration.method} (confidence: {calibration.confidence:.0%})")

        # ══════════════════════════════════════════════════════════════════
        # PHASE 4: Adaptive Extraction
        # ══════════════════════════════════════════════════════════════════
        print("Phase 4: Extracting entities...")
        entities, raster_commands = await extract_all(
            analysis=analysis,
            calibration=calibration,
            image_path=render_result.image_path
        )
        print(f"  - Direct entities: {len(entities)}")
        print(f"  - Raster commands: {len(raster_commands)}")

        # ══════════════════════════════════════════════════════════════════
        # PHASE 5: AutoCAD Entity Creation
        # ══════════════════════════════════════════════════════════════════
        print("Phase 5: Creating entities in AutoCAD...")
        creation_result = await create_entities_in_autocad(
            entities=entities,
            raster_commands=raster_commands
        )
        print(f"  - Succeeded: {len(creation_result['success'])}")
        print(f"  - Failed: {len(creation_result['failed'])}")

        if creation_result["failed"]:
            warnings.extend(creation_result["failed"])

        # ══════════════════════════════════════════════════════════════════
        # PHASE 6: Validation & Self-Correction
        # ══════════════════════════════════════════════════════════════════
        print("Phase 6: Validating extraction...")
        validation = await validate_extraction(
            original_image_path=render_result.image_path,
            created_entities=entities,
            analysis=analysis,
            calibration=calibration,
            max_iterations=max_validation_iterations
        )
        print(f"  - Status: {validation.status}")
        print(f"  - Accuracy: {validation.accuracy_estimate:.1f}%")
        print(f"  - Iterations: {validation.iteration}")

        if validation.issues:
            for issue in validation.issues:
                if issue.get("severity") == "critical":
                    errors.append(issue["description"])
                else:
                    warnings.append(issue["description"])

        return PipelineResult(
            success=validation.status == "approved",
            entities_created=len(creation_result["success"]),
            accuracy=validation.accuracy_estimate,
            validation_iterations=validation.iteration,
            errors=errors,
            warnings=warnings
        )

    except Exception as e:
        errors.append(str(e))
        return PipelineResult(
            success=False,
            entities_created=0,
            accuracy=0,
            validation_iterations=0,
            errors=errors,
            warnings=warnings
        )
```

## Usage Example

```python
# Example usage
import asyncio

async def main():
    result = await gemini_first_vectorize(
        pdf_path=Path("./drawings/floor_plan.pdf"),
        page_number=0,
        dpi=300
    )

    print("\n" + "="*60)
    print("PIPELINE COMPLETE")
    print("="*60)
    print(f"Success: {result.success}")
    print(f"Entities created: {result.entities_created}")
    print(f"Accuracy: {result.accuracy:.1f}%")
    print(f"Validation iterations: {result.validation_iterations}")

    if result.errors:
        print(f"\nErrors ({len(result.errors)}):")
        for e in result.errors:
            print(f"  - {e}")

    if result.warnings:
        print(f"\nWarnings ({len(result.warnings)}):")
        for w in result.warnings:
            print(f"  - {w}")

asyncio.run(main())
```

---

# Summary: Phase-by-Phase

| Phase | Purpose | Input | Output | Key Technology |
|-------|---------|-------|--------|----------------|
| **1** | PDF Intake | PDF file | High-quality PNG | PyMuPDF |
| **2** | Understanding | PNG image | Drawing analysis JSON | Gemini Vision |
| **3** | Calibration | Analysis + Image dims | Scale factor + Transform | Math |
| **4** | Extraction | Analysis + Calibration | Entity list + Raster commands | Strategy selection |
| **5** | Creation | Entity list | AutoCAD entities | MCP tools |
| **6** | Validation | Original + Entities | Corrections applied | Gemini Vision |

## Data Flow Diagram

```
┌─────────┐     ┌─────────┐     ┌─────────┐     ┌─────────┐     ┌─────────┐     ┌─────────┐
│ Phase 1 │────▶│ Phase 2 │────▶│ Phase 3 │────▶│ Phase 4 │────▶│ Phase 5 │────▶│ Phase 6 │
│         │     │         │     │         │     │         │     │         │     │         │
│  PDF    │     │ Gemini  │     │ Scale   │     │ Extract │     │ Create  │     │ Validate│
│  Render │     │ Analyze │     │ Calibr. │     │ Entities│     │ in CAD  │     │ & Fix   │
└─────────┘     └─────────┘     └─────────┘     └─────────┘     └─────────┘     └─────────┘
     │               │               │               │               │               │
     ▼               ▼               ▼               ▼               ▼               ▼
  ┌─────┐       ┌─────────┐     ┌─────────┐     ┌─────────┐     ┌─────────┐     ┌─────────┐
  │ PNG │       │ JSON    │     │ Scale   │     │ Entity  │     │ DWG     │     │ Final   │
  │Image│       │ Analysis│     │ Factor  │     │ List    │     │ Entities│     │ DWG     │
  └─────┘       └─────────┘     └─────────┘     └─────────┘     └─────────┘     └─────────┘
```

---

*Document Version: 1.1*
*Last Updated: 2026-02-16*
*Note: Uses gemini-2.0-flash by default (has free tier). Pro models do not have free tier.*
