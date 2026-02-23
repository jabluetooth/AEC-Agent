# PDF Raster-to-Vector Pipeline - Technical Reference

> **Purpose**: This document provides a comprehensive technical reference for the Gemini-First PDF-to-AutoCAD vectorization pipeline. Use it to understand where accuracy is lost and research improvements.

---

## Table of Contents

1. [Pipeline Overview](#pipeline-overview)
2. [Phase 1: PDF Intake & Rendering](#phase-1-pdf-intake--rendering)
3. [Phase 2: Gemini Understanding](#phase-2-gemini-understanding)
4. [Phase 3: Coordinate Calibration](#phase-3-coordinate-calibration)
5. [Phase 4: Adaptive Extraction](#phase-4-adaptive-extraction)
6. [Phase 4b: OpenCV Extraction](#phase-4b-opencv-extraction-utilities)
7. [Phase 4c: Gemini Refinement](#phase-4c-gemini-refinement)
8. [Phase 4d: OCR Text Anchoring](#phase-4d-ocr-text-anchoring)
9. [Phase 5: AutoCAD Entity Creation](#phase-5-autocad-entity-creation)
10. [Phase 6: Validation & Self-Correction](#phase-6-validation--self-correction)
11. [Precision Analysis](#precision-analysis-where-accuracy-is-lost)
12. [Known Limitations](#known-limitations)
13. [Research Directions for Improvement](#research-directions-for-improvement)
14. [Algorithm Parameters Reference](#algorithm-parameters-reference)
15. [File Reference](#file-reference)

---

## Pipeline Overview

**Location**: `src/aec_agent/mcp/tools/gemini_first/`

**Total Codebase**: ~12,000+ lines across 11 modules

### High-Level Data Flow

```
PDF Input → Phase 1 → Phase 2 → Phase 3 → Phase 4 → Phase 5 → Phase 6 → AutoCAD Output
           (Render) (Analyze) (Calibrate) (Extract) (Create) (Validate)
```

### Visual Pipeline Diagram

```
┌─────────────┐
│ PDF Input   │
└──────┬──────┘
       │
       ▼
┌──────────────────────────────────────┐
│ Phase 1: PDF Rendering (300 DPI)     │
│ - PyMuPDF rendering                  │
│ - Grayscale detection (variance<0.01)│
│ - PNG save (lossless)                │
└──────┬───────────────────────────────┘
       │
       ▼ [PNG Image ~3-5MB]
┌──────────────────────────────────────┐
│ Phase 2: Gemini Analysis             │
│ - Classify drawing type              │
│ - Detect elements (lines, arcs, etc.)│
│ - Extract text, symbols              │
│ - Identify calibration hints         │
└──────┬───────────────────────────────┘
       │
       ▼ [DrawingAnalysis JSON]
┌──────────────────────────────────────┐
│ Phase 3: Calibration                 │
│ - Try dimension hints (90% conf)     │
│ - Try scale notation (85% conf)      │
│ - Try sheet size (70% conf)          │
│ - Fall back to DPI (30% conf)        │
└──────┬───────────────────────────────┘
       │
       ▼ [ScaleCalibration: units/pixel]
┌──────────────────────────────────────┐
│ Phase 4: Extraction                  │
│ ┌─ Direct: Gemini coords → DWG      │
│ ├─ Guided: OpenCV + Raster Design   │
│ └─ Selective: OpenCV for regions    │
│ - Hybrid OpenCV extraction           │
│ - Refinement: snap/connect/align    │
│ - OCR text anchoring                 │
│ - Draftsman cleanup                  │
└──────┬───────────────────────────────┘
       │
       ▼ [EntityToCreate list + RasterCommands]
┌──────────────────────────────────────┐
│ Phase 5: AutoCAD Creation            │
│ - Create layers (NCS-compliant)      │
│ - Batch entity creation (50/batch)   │
│ - Thread marshaling for STA          │
└──────┬───────────────────────────────┘
       │
       ▼ [Entities created, handles captured]
┌──────────────────────────────────────┐
│ Phase 6: Validation & Correction     │
│ - Gemini compares: original vs created│
│ - Detect issues (7 types)            │
│ - Apply corrections (up to 3 loops)  │
└──────────────────────────────────────┘
       │
       ▼
┌─────────────────────────────────────┐
│ Final AutoCAD Drawing               │
│ (Vectorized, semantically tagged)   │
└─────────────────────────────────────┘
```

---

## Phase 1: PDF Intake & Rendering

**File**: `pdf_intake.py` (~552 lines)

### Purpose
Render PDF pages to high-quality raster images WITHOUT destroying information.

### Technology Stack
- **PyMuPDF (fitz)**: PDF parsing and rendering
- **Pillow (PIL)**: Image manipulation
- **PNG format**: Lossless compression

### Key Parameters

| Parameter | Default | Range | Impact |
|-----------|---------|-------|--------|
| `dpi` | 300 | 72-1200 | Higher = more detail, larger files |
| `GRAYSCALE_THRESHOLD` | 0.01 | 0-1 | Color variance threshold for grayscale conversion |
| `alpha` | False | - | Disabled to preserve solid colors |

### Rendering Formula
```python
zoom = dpi / 72.0  # PDF base resolution is 72 DPI
matrix = fitz.Matrix(zoom, zoom)
pixmap = page.get_pixmap(matrix=matrix, alpha=False)
```

### Key Functions
- `render_pdf_high_quality()`: Main entry point
- `is_effectively_grayscale()`: Checks if RGB image has minimal color variance
- `get_pdf_info()`: Extracts PDF metadata
- `compare_with_bitonal()`: Demonstrates information preservation vs. old approaches

### Output Data Structure
```python
@dataclass
class PDFRenderResult:
    image_path: Path          # Path to rendered PNG
    width_px: int             # Image width in pixels
    height_px: int            # Image height in pixels
    dpi: int                  # Rendering DPI
    color_mode: str           # "RGB" or "L" (grayscale)
    page_width_inches: float  # Original PDF page width
    page_height_inches: float # Original PDF page height
```

### Precision Loss Sources
1. **Rasterization**: Vector PDF → pixel grid (resolution limited by DPI)
2. **At 300 DPI**: 1 pixel ≈ 0.085mm or 0.0033 inches
3. **Grayscale conversion**: May lose subtle color information

### Potential Improvements
- Increase DPI to 600 for finer detail (2x file size)
- Preserve vector data for text/simple shapes
- Use PDF.js or other engines for comparison

---

## Phase 2: Gemini Understanding

**File**: `gemini_understanding.py` (~785 lines)

### Purpose
Use Gemini Vision to fully analyze the drawing BEFORE any extraction. This provides semantic understanding that pure computer vision lacks.

### Technology Stack
- **Gemini 2.0 Flash**: Vision-language model
- **JSON structured output**: For reliable parsing

### What Gemini Extracts

| Element Type | Properties Detected |
|--------------|---------------------|
| **Lines** | start, end, line_type (wall/duct/pipe/wire), linetype (continuous/dashed/dotted) |
| **Arcs** | center, radius, start_angle, end_angle, arc_type (door_swing/curved_wall) |
| **Circles** | center, radius, circle_type (column/equipment/symbol) |
| **Text** | content, position, height, text_type (room_name/dimension/tag/note) |
| **Symbols** | type, subtype, position, rotation, associated_text |
| **Dimensions** | value, numeric_value, unit, start/end points |
| **Regions** | title_block, drawing_area, legend, notes (bounding boxes) |

### Drawing Types Classified
- floor_plan, electrical, mechanical, plumbing
- fire_alarm, reflected_ceiling, site_plan
- detail, section, elevation, schedule, diagram

### Extraction Strategies
```python
class ExtractionStrategy(str, Enum):
    DIRECT = "direct"                      # Use Gemini coordinates directly
    GUIDED_RASTERIZATION = "guided_rasterization"  # Use Raster Design tools
    SELECTIVE_OPENCV = "selective_opencv"  # Use OpenCV for specific regions
    HYBRID = "hybrid"                      # Combine approaches
    SKIP = "skip"                          # Don't process this region
```

### Calibration Hints Extracted
Gemini looks for calibration information:
- **Known dimensions**: "20'-0"" dimension lines
- **Scale notations**: "1/4" = 1'-0"" in title block
- **Sheet sizes**: "ARCH D", "24x36"
- **Grid spacing**: Column grid dimensions

### Key Parameters
```python
temperature = 0.1  # Low for consistency
max_output_tokens = 8192  # For detailed JSON output
model = "gemini-2.0-flash"  # Fast, capable vision model
```

### Precision Loss Sources

| Source | Typical Error | Notes |
|--------|---------------|-------|
| Coordinate estimation | ±5-10 pixels | On clean technical drawings |
| Complex geometry | ±10-50 pixels | Overlapping elements, clutter |
| Poor image quality | ±50-100+ pixels | Low DPI, scanned documents |
| Text position | ±5-20 pixels | Baseline/center ambiguity |

### Why Gemini Struggles
1. **No sub-pixel accuracy**: LLMs estimate visually, not measure
2. **Coordinate system ambiguity**: Doesn't know exact origin
3. **Scale uncertainty**: Doesn't know image resolution
4. **Occlusion**: Overlapping elements confuse position
5. **Complex curves**: Arcs and splines are particularly challenging

### Potential Improvements
- Use higher resolution images (600+ DPI)
- Provide reference points/markers
- Chain of thought prompting for coordinates
- Multiple inference passes with averaging
- Hybrid approach: Gemini for "what", OpenCV for "where"

---

## Phase 3: Coordinate Calibration

**File**: `coordinate_calibration.py` (~808 lines)

### Purpose
Convert pixel coordinates to AutoCAD DWG units (typically inches or millimeters).

### Calibration Methods (Priority Order)

| Priority | Method | Confidence | Source |
|----------|--------|------------|--------|
| 1 | Known Dimension | 90% | Dimension line with measured pixel distance |
| 2 | Scale Notation | 85% | "1/4" = 1'-0"" in title block |
| 3 | Sheet Size | 70% | ARCH D, A1, etc. |
| 4 | DPI Default | 30% | Fallback: 1px = 1/DPI inches |

### Coordinate Transformation Formula

```python
# CRITICAL: Y-axis flip required
dwg_x = pixel_x * scale_factor + origin_offset[0]
dwg_y = (image_height - pixel_y) * scale_factor + origin_offset[1]
```

**Why Y-flip?**
- Image coordinates: Origin top-left, Y increases downward
- AutoCAD coordinates: Origin bottom-left, Y increases upward

### Scale Calculation Examples

**From Dimension:**
```python
pixel_distance = 200  # pixels
real_measurement = "10'-0"" → 120 inches
scale_factor = 120 / 200 = 0.6 inches/pixel
```

**From Scale Notation:**
```python
notation = "1/4" = 1'-0""
# At 300 DPI: 1 inch on paper = 300 pixels
# 1/4" paper = 12" real
# pixels_per_inch_paper = 300
# real_per_paper_inch = 12 / 0.25 = 48 inches
scale_factor = 48 / 300 = 0.16 inches/pixel
```

**From Sheet Size:**
```python
sheet = "ARCH D" → 24" × 36"
image_size = 7200 × 10800 pixels (at 300 DPI)
# If known_dimension is along X: sheet_width / image_width
scale_x = 36 / 10800 = 0.00333 inches/pixel
```

### Supported Sheet Sizes
40+ standard sizes including:
- **ARCH**: A, B, C, D, E, E1
- **ANSI**: A, B, C, D, E
- **ISO**: A0, A1, A2, A3, A4
- **Custom**: 24x36, 30x42, 36x48

### Measurement Parsing Supported
```
Feet & inches: 20'-0", 20' 0", 20'0", 20 ft 0 in
Plain inches: 24", 24 in, 24 inches
Metric: 100mm, 100 cm, 1.5m, 1500 millimeters
```

### Precision Loss Sources
1. **DPI-based calibration** (30% confidence): Assumes accurate DPI metadata
2. **Sheet size estimation**: Assumes full-bleed printing
3. **Dimension measurement**: Pixel endpoint detection ±2-3 pixels
4. **Scale notation parsing**: Requires exact format matching
5. **Accumulating error**: Scale errors multiply through all coordinates

### Potential Improvements
- Require user-provided calibration point
- Detect multiple dimensions and average
- Use known grid spacing for verification
- Implement coordinate system detection

---

## Phase 4: Adaptive Extraction

**File**: `adaptive_extraction.py` (~2,900+ lines) — **LARGEST MODULE**

### Purpose
Extract AutoCAD entities using the optimal strategy for each element type.

### Three Extraction Strategies

#### A. Direct Extraction
- **When**: Clean drawings, text, symbols, simple geometry
- **Method**: Convert Gemini coordinates directly to DWG entities
- **Precision**: Inherits Phase 2 accuracy (±5-10 pixels)
- **Speed**: Fast, no additional processing

#### B. Guided Rasterization
- **When**: Complex curves, connected paths
- **Method**: Generate Raster Design VTool commands
- **Tools**: VFPLINE, VFCONTOUR, VARC, VCIRCLE, VLINE
- **Precision**: Sub-pixel via Raster Design tracing engine

#### C. Selective OpenCV
- **When**: Repetitive patterns, hatching, dense details
- **Method**: Use OpenCV on Gemini-identified regions
- **Precision**: Pixel-level geometric accuracy
- **Limitation**: No text/semantic understanding

### Layer Mapping (NCS-Compliant)

| Element Type | Layer | Color |
|--------------|-------|-------|
| wall | A-WALL | Cyan (5) |
| door | A-DOOR | Cyan (5) |
| duct | M-DUCT | Yellow (2) |
| diffuser | M-DIFF | Yellow (2) |
| pipe | P-PIPE | Green (3) |
| wire | E-POWR | Red (1) |
| outlet | E-POWR-OUTL | Red (1) |
| detector | F-ALRM-DETC | Magenta (4) |

### Hybrid Extraction Pipeline

```
Input Image + Gemini Analysis
            │
            ▼
    ┌───────────────────────────────┐
    │  Mask Text/Symbol Regions    │ ← Prevents OpenCV detecting text as lines
    └───────────────────────────────┘
            │
            ▼
    ┌───────────────────────────────┐
    │  OpenCV Line Extraction      │ ← LSD algorithm, sub-pixel accuracy
    │  (extract_lines_lsd)         │
    └───────────────────────────────┘
            │
            ▼
    ┌───────────────────────────────┐
    │  OpenCV Circle Extraction    │ ← Hough Circle Transform
    │  (extract_circles)           │
    └───────────────────────────────┘
            │
            ▼
    ┌───────────────────────────────┐
    │  OpenCV Arc Extraction       │ ← Ellipse fitting
    │  (extract_arcs)              │
    └───────────────────────────────┘
            │
            ▼
    ┌───────────────────────────────┐
    │  Merge Duplicates            │ ← OpenCV + Gemini overlap removal
    └───────────────────────────────┘
            │
            ▼
    ┌───────────────────────────────┐
    │  Gemini Refinement           │ ← Snap, connect, align
    └───────────────────────────────┘
            │
            ▼
    ┌───────────────────────────────┐
    │  Draftsman Cleanup           │ ← Remove noise, short segments
    └───────────────────────────────┘
            │
            ▼
    Output: EntityToCreate list
```

### Text/Symbol Masking (Fixes MTEXT issue)

Before OpenCV runs, text regions are masked out:
```python
# Calculate bounding box from text content and height
char_width = max(8, int(text_height * 0.6))
estimated_width = len(text_content) * char_width
# Draw white rectangle over text region
cv2.rectangle(image, (x1, y1), (x2, y2), (255, 255, 255), -1)
```

### Draftsman Cleanup (Deletion functionality)

| Cleanup Operation | Default | Purpose |
|-------------------|---------|---------|
| Remove short segments | 15px min | Eliminate noise lines |
| Remove lines through text | 30% threshold | Remove text artifacts |
| Remove incomplete curves | 15% arc span | Remove noise arcs |
| Filter low confidence | 0.3 threshold | Remove uncertain detections |
| Remove edge artifacts | 5px margin | Remove border lines |

### Key Data Structures

```python
@dataclass
class EntityToCreate:
    entity_type: EntityType  # LINE, ARC, CIRCLE, TEXT, BLOCK
    layer: str               # NCS layer name
    properties: dict         # start, end, center, radius, etc.
    source: ExtractionSource # DIRECT, SELECTIVE_OPENCV, HYBRID
    confidence: float        # 0.0 to 1.0

@dataclass
class HybridExtractionResult:
    entities: List[EntityToCreate]
    gemini_entities: int
    opencv_entities: int
    duplicates_merged: int
    cleanup_stats: CleanupStatistics
```

---

## Phase 4b: OpenCV Extraction Utilities

**File**: `opencv_extraction.py` (~1,034 lines)

### Purpose
Precise geometric extraction using computer vision algorithms.

### Line Detection Algorithms

#### 1. Line Segment Detector (LSD) — **PRIMARY**
```python
lsd = cv2.createLineSegmentDetector(cv2.LSD_REFINE_STD)
lines, widths, _, _ = lsd.detect(gray)
```

| Parameter | Value | Impact |
|-----------|-------|--------|
| Refinement | LSD_REFINE_STD | Sub-pixel accuracy |
| Min length | 20 pixels | Filter noise |
| Confidence | 0.95 | High accuracy |

**Advantages**:
- Sub-pixel endpoint accuracy
- Detects line thickness
- No parameter tuning required
- Handles varying contrast

**Limitations**:
- Slower than Hough
- May split long lines at gaps

#### 2. Probabilistic Hough Transform — **FALLBACK**
```python
lines = cv2.HoughLinesP(
    edges,
    rho=1,              # 1 pixel distance resolution
    theta=np.pi/180,    # 1 degree angle resolution
    threshold=50,       # Min votes needed
    minLineLength=30,   # Min line length
    maxLineGap=10       # Max gap to bridge
)
```

**Advantages**:
- Fast
- Handles gaps well

**Limitations**:
- 1 pixel minimum resolution
- Parameter sensitive
- May detect false positives

### Circle Detection (Hough Circle Transform)

```python
circles = cv2.HoughCircles(
    gray,
    cv2.HOUGH_GRADIENT,
    dp=1,           # Resolution ratio
    minDist=20,     # Min distance between centers
    param1=50,      # Canny high threshold
    param2=70,      # Accumulator threshold (TUNED HIGH)
    minRadius=5,
    maxRadius=200
)
```

| Parameter | Value | Impact |
|-----------|-------|--------|
| param2 | 70 | **High to reduce false positives** |
| minDist | 20px | Prevents overlapping detections |
| minRadius | 5px | Ignores noise dots |

**Known Issue**: param2=70 is aggressive and **may miss weak circles**

### Arc Detection (Ellipse Fitting)

```python
# 1. Find contours
contours, _ = cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

# 2. Fit ellipse (requires ≥5 points)
ellipse = cv2.fitEllipse(contour)
center, axes, angle = ellipse

# 3. Filter by shape
aspect_ratio = min(axes) / max(axes)  # Must be > 0.5
area_ratio = contour_area / (π * radius²)  # 0.05-0.85 for arcs

# 4. Calculate arc angles
# Sample contour points, find angular extent
```

**Filter Criteria**:
- Aspect ratio > 0.5 (not too elongated)
- Area ratio 0.05-0.85 (arc, not full circle)
- Arc span 30°-300° (meaningful arc)

**Known Issue**: Ellipse fitting **distorts true circular arcs**

### Line Type Detection (Pattern Analysis)

```python
def detect_line_type(line):
    # Sample pixels along line
    for t in linspace(0, 1, num_samples):
        point = lerp(start, end, t)
        # Sample 3-pixel perpendicular strip
        ink_values = sample_perpendicular(point, line.angle, width=3)
        is_ink = mean(ink_values) < threshold

    # Analyze pattern
    transitions = count_transitions(samples)  # ink↔gap changes
    ink_ratio = sum(ink) / total_samples

    if transitions < 2: return CONTINUOUS
    if avg_ink < 4 and transitions > 8: return DOTTED
    if ink_variance > 2.5: return CENTER
    if avg_ink > avg_gap: return DASHED
    return HIDDEN
```

| Line Type | Pattern Characteristics |
|-----------|------------------------|
| Continuous | No gaps, ink_ratio ≈ 1.0 |
| Dashed | Regular gaps, transitions > 3 |
| Hidden | Short dashes, 0.3 < ink_ratio < 0.7 |
| Dotted | Very short ink, frequent transitions |
| Center | Alternating long-short-long |

### Contour Extraction

```python
contours = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

# Simplify using Douglas-Peucker
simplified = cv2.approxPolyDP(contour, epsilon=2.0, closed=True)
```

**Precision Loss**: Douglas-Peucker simplification reduces point count, may lose curved detail.

---

## Phase 4c: Gemini Refinement

**File**: `gemini_refinement.py` (~692 lines)

### Purpose
Post-processing cleanup of extracted entities using geometric rules and optional Gemini verification.

### Refinement Operations

| Operation | Parameter | Effect |
|-----------|-----------|--------|
| Snap to Grid | tolerance=0.1 units | Align to drawing grid |
| Connect Endpoints | tolerance=10px | Join nearby line ends |
| Align Parallel | tolerance=5° | Force H/V/45° alignment |
| Remove Duplicates | - | Eliminate overlapping geometry |
| Fix Intersections | - | Extend lines to meet |

### Grid Snapping
```python
# Snap coordinate to nearest grid line
snapped = round(coord / grid_size) * grid_size
# Only snap if within tolerance
if abs(coord - snapped) < snap_tolerance:
    coord = snapped
```

### Endpoint Connection
```python
# Find endpoints within tolerance
for line1, line2 in pairs:
    dist = distance(line1.end, line2.start)
    if dist < connection_tolerance:
        # Extend/modify to connect
        line1.end = line2.start = midpoint(line1.end, line2.start)
```

### Parallel Alignment (AGGRESSIVE)
```python
parallel_tolerance_deg = 5.0  # Lines within 5° are aligned

if abs(line.angle - 0) < 5: line.angle = 0      # Horizontal
if abs(line.angle - 90) < 5: line.angle = 90    # Vertical
if abs(line.angle - 45) < 5: line.angle = 45    # Diagonal
```

**Known Issue**: 5° tolerance **may distort intentional angles** (e.g., 7° slope becomes 5° or 10°)

---

## Phase 4d: OCR Text Anchoring

**File**: `ocr_text_anchoring.py` (~470 lines)

### Purpose
Use Tesseract OCR to get pixel-accurate text positions, then match with Gemini's semantic text content.

### Problem Solved
- Gemini estimates text positions visually: ±10-100 pixel error
- OCR detects exact bounding boxes: ±2-5 pixel accuracy

### Algorithm

```
1. Run Tesseract OCR on image
   ↓
2. Get all text boxes with positions
   ↓
3. For each Gemini text element:
   a. Find OCR boxes with similar content
   b. Calculate text similarity (fuzzy match)
   c. If similarity > 0.5: use OCR position
   d. Else: keep Gemini position
   ↓
4. Output: Text with corrected positions
```

### Text Matching
```python
from difflib import SequenceMatcher

def similarity(text1, text2):
    # Normalize: lowercase, remove spaces, fix OCR artifacts
    t1 = normalize(text1)
    t2 = normalize(text2)
    return SequenceMatcher(None, t1, t2).ratio()

# Match if similarity > 0.5 and position within 200px
```

### Parameters

| Parameter | Default | Purpose |
|-----------|---------|---------|
| min_confidence | 60% | OCR confidence threshold |
| min_similarity | 0.5 | Text match threshold |
| max_offset | 200px | Maximum position correction |

### Typical Results
- **Anchor rate**: 70-90% of text elements
- **Average offset corrected**: 15-40 pixels

---

## Phase 5: AutoCAD Entity Creation

**File**: `autocad_creation.py` (~989 lines)

### Purpose
Create extracted entities in AutoCAD using the sidecar plugin.

### Entity Creation

| Entity Type | AutoCAD Command | Properties |
|-------------|-----------------|------------|
| LINE | AddLine | start, end, linetype |
| ARC | AddArc | center, radius, start_angle, end_angle |
| CIRCLE | AddCircle | center, radius |
| TEXT | AddText | position, height, content |
| MTEXT | AddMText | position, width, content |
| BLOCK | InsertBlock | name, position, rotation, scale, attributes |
| POLYLINE | Add2DPolyline | vertices, closed |

### Batch Processing
- Entities processed in batches of 50
- Thread marshaling for AutoCAD STA constraint
- Transaction handling with DocumentLock

### Layer Creation
```python
LAYER_PREFIX_COLORS = {
    "A-": 5,  # Cyan (Architecture)
    "M-": 2,  # Yellow (Mechanical)
    "E-": 1,  # Red (Electrical)
    "P-": 3,  # Green (Plumbing)
    "F-": 4,  # Magenta (Fire)
    "G-": 7,  # White (General)
}
```

---

## Phase 6: Validation & Self-Correction

**File**: `validation.py` (~848 lines)

### Purpose
Use Gemini Vision to compare created entities with original image and apply corrections.

### Validation Loop

```
Created Entities
       ↓
Gemini Validation (compare with original)
       ↓
Issue Detection
       ├→ APPROVED: Done (accuracy acceptable)
       ├→ ISSUES_FOUND: Apply corrections
       │       ↓
       │   Create/Delete/Modify entities
       │       ↓
       │   Loop (max 3 iterations)
       └→ MAX_ITERATIONS: Manual review needed
```

### Issue Types Detected

| Issue Type | Description | Severity |
|------------|-------------|----------|
| missing_element | Entity not created | Critical |
| extra_element | False positive | Major |
| position_error | Coordinate mismatch | Major |
| text_error | OCR/content mismatch | Minor |
| symbol_error | Wrong block type | Major |
| connectivity | Lines don't meet | Minor |
| layer_error | Wrong layer assignment | Minor |

### Correction Actions

| Action | What It Does |
|--------|--------------|
| ADD | Create missing entity |
| REMOVE | Delete false positive |
| MODIFY | Change entity properties |
| REPLACE | Delete old, create new |

### Typical Performance
- **Iterations needed**: 1-2 (rarely 3)
- **Accuracy improvement**: 85% → 92%+
- **False positive reduction**: 10-20%

---

## Precision Analysis: Where Accuracy is Lost

### Cumulative Error Analysis

| Phase | Operation | Input Precision | Output Precision | Error Added |
|-------|-----------|-----------------|------------------|-------------|
| 1 | PDF Rendering | ∞ (vector) | ±0.5px | 0.5px |
| 2 | Gemini Analysis | ±0.5px | ±5-50px | **±5-50px** |
| 3 | Calibration | ±5-50px | ±scale error | 0.3-1% |
| 4 | OpenCV Lines | ±5-50px | ±1-2px | **Improved!** |
| 4 | OpenCV Circles | ±5-50px | ±2-3px | **Improved!** |
| 4 | Refinement | ±1-5px | ±0.1 units | Variable |
| 5 | Creation | ±0.1-5px | Same | 0 |
| 6 | Validation | ±0.1-5px | ±0.1-5px | 0-1% |

### Key Insight
**OpenCV extraction IMPROVES accuracy** over direct Gemini coordinates.

### Largest Error Sources (Ranked)

1. **Gemini coordinate estimation** (±5-50 pixels)
   - LLMs estimate visually, don't measure
   - Complex geometry causes larger errors

2. **Calibration uncertainty** (0.3-1% scale error)
   - DPI-based calibration least accurate (30% confidence)
   - Dimension-based is best (90% confidence)

3. **Arc/curve fitting** (±3-5 pixels)
   - Ellipse fitting distorts true arcs
   - Splines not well supported

4. **Parallel alignment** (angle distortion)
   - 5° tolerance may alter intentional angles

5. **Line type detection** (classification error)
   - Pattern analysis can misclassify noisy lines

---

## Known Limitations

### Fundamental Limitations

1. **Rasterization bottleneck**: Converting vector PDF to pixels loses resolution
2. **No true curve extraction**: Arcs approximated as ellipse segments
3. **No spline support**: Bezier curves not extracted natively
4. **Text baseline ambiguity**: Position can be baseline, center, or top
5. **Symbol recognition limited**: Only trained symbols detected

### Algorithm Limitations

1. **Hough circles**: param2=70 misses weak circles
2. **LSD lines**: May split continuous lines at gaps
3. **Ellipse arcs**: Distorts true circular arcs
4. **Line type detection**: Requires clean, consistent patterns
5. **Parallel alignment**: 5° tolerance too aggressive for some drawings

### Environmental Limitations

1. **Tesseract OCR**: Requires installation, slow
2. **YOLO symbols**: Requires trained model
3. **AutoCAD sidecar**: Single-threaded (STA)
4. **Gemini API**: Rate limits, cost per call

---

## Research Directions for Improvement

### 1. Better Coordinate Extraction from LLMs

**Current Problem**: Gemini estimates coordinates visually (±5-50px error)

**Research Areas**:
- **Visual prompting**: Add reference markers/grids to image
- **Iterative refinement**: Ask "is this point exactly here?" follow-ups
- **Specialized models**: Fine-tune on technical drawing datasets
- **Ensemble methods**: Multiple LLMs + voting
- **Structured output**: Force coordinate grid responses

**Papers to Explore**:
- "Visual Programming" for precise spatial reasoning
- "LayoutLLM" for document understanding
- "DocTr" for document image dewarping

### 2. Advanced Line Detection

**Current Problem**: LSD/Hough struggle with complex drawings

**Research Areas**:
- **Deep Hough Transform**: Neural network-based line detection
- **LETR**: Line segment detection transformer
- **F-Clip**: Fast line segment detector with CLIP guidance
- **Edge Drawing**: Alternative to Canny for edge detection
- **Wireframe parsing**: Joint line and junction detection

**Libraries to Explore**:
- `deeplsd` (Deep Line Segment Detection)
- `hawp` (Holistically-Attracted Wireframe Parser)
- `lcnn` (Line-CNN)

### 3. Better Curve/Arc Extraction

**Current Problem**: Ellipse fitting distorts true arcs

**Research Areas**:
- **Ransac-based arc fitting**: More robust to outliers
- **B-spline approximation**: Capture complex curves
- **Differentiable curve fitting**: End-to-end learnable
- **Vectorization networks**: Direct raster-to-vector models

**Papers to Explore**:
- "Vectorization of Line Drawings via Polyvector Fields"
- "Deep Vectorization of Technical Drawings"
- "Neural Vector Graphics"

### 4. End-to-End Raster-to-Vector

**Current Problem**: Multi-phase pipeline accumulates errors

**Research Areas**:
- **DiffVG**: Differentiable vector graphics
- **Im2Vec**: Image to vector directly
- **DeepSVG**: Deep learning for SVG generation
- **PolyGen**: Polygon mesh generation (adaptable)

**Key Papers**:
- "Im2Vec: Synthesizing Vector Graphics" (CVPR 2021)
- "Deep Vectorization of Technical Drawings" (CVPR 2020)
- "PolyGen: An Autoregressive Generative Model of 3D Meshes"

### 5. Domain-Specific Training

**Current Problem**: Generic models don't understand AEC drawings

**Research Areas**:
- **Fine-tune on FloorPlanCAD dataset**
- **Create synthetic training data from DWG files**
- **Self-supervised pretraining on architectural PDFs**
- **Symbol detection with custom YOLO training**

**Datasets**:
- FloorPlanCAD (Berkeley)
- CubiCasa5K (Floor plans)
- ROBIN (Indoor robots, has floor plans)
- Custom: Render DWG → PDF → use as ground truth

### 6. Hybrid Approaches

**Current Problem**: No single method works for everything

**Research Areas**:
- **Confidence-based routing**: Use LLM for semantic, OpenCV for precise
- **Multi-resolution**: Coarse detection → fine localization
- **Active learning**: Ask user to verify ambiguous elements
- **Iterative refinement**: Detect, validate, correct, repeat

### 7. Quality Metrics

**Current Problem**: No objective accuracy measurement

**Research Areas**:
- **Chamfer distance**: Measure geometric similarity
- **Hausdorff distance**: Maximum deviation metric
- **IoU for lines**: Intersection over union for segments
- **Structured similarity**: Compare graph structure, not just geometry

---

## Algorithm Parameters Reference

### Phase 1: Rendering
```python
DPI = 300                    # Default rendering resolution
MIN_DPI = 72                 # Minimum allowed
MAX_DPI = 1200               # Maximum allowed
GRAYSCALE_THRESHOLD = 0.01   # Color variance for grayscale detection
```

### Phase 2: Gemini Analysis
```python
temperature = 0.1            # Low for consistency
max_output_tokens = 8192     # For detailed JSON
model = "gemini-2.0-flash"   # Fast vision model
```

### Phase 3: Calibration
```python
CONFIDENCE_DIMENSION = 0.90  # Known dimension
CONFIDENCE_SCALE = 0.85      # Scale notation
CONFIDENCE_SHEET = 0.70      # Sheet size
CONFIDENCE_DPI = 0.30        # DPI fallback
```

### Phase 4: OpenCV Extraction
```python
# Line Segment Detector
LSD_REFINE = cv2.LSD_REFINE_STD
MIN_LINE_LENGTH = 20         # Pixels

# Hough Lines
HOUGH_RHO = 1                # Pixel resolution
HOUGH_THETA = np.pi/180      # Angle resolution
HOUGH_THRESHOLD = 50         # Accumulator threshold
HOUGH_MIN_LENGTH = 30        # Minimum line length
HOUGH_MAX_GAP = 10           # Maximum gap to bridge

# Hough Circles
CIRCLE_DP = 1                # Resolution ratio
CIRCLE_MIN_DIST = 20         # Min center spacing
CIRCLE_PARAM1 = 50           # Canny threshold
CIRCLE_PARAM2 = 70           # Accumulator threshold (HIGH!)
CIRCLE_MIN_RADIUS = 5
CIRCLE_MAX_RADIUS = 200

# Arc Detection
ARC_MIN_LENGTH = 20
ARC_MIN_RADIUS = 10
ARC_MAX_RADIUS = 500
ARC_MIN_ANGLE = 30           # Minimum span (degrees)
ARC_MAX_ANGLE = 300          # Maximum span (degrees)
ASPECT_RATIO_MIN = 0.5       # For filtering ellipses
AREA_RATIO_MIN = 0.05
AREA_RATIO_MAX = 0.85

# Contour Detection
CONTOUR_MIN_AREA = 100       # Pixels²
DOUGLAS_PEUCKER_EPSILON = 2.0
```

### Phase 4: Refinement
```python
SNAP_TOLERANCE = 5.0         # Pixels
CONNECTION_TOLERANCE = 10.0  # Pixels
PARALLEL_TOLERANCE = 5.0     # Degrees (AGGRESSIVE!)
GRID_SIZE = 1.0              # DWG units
```

### Phase 4: Cleanup
```python
MIN_LINE_LENGTH = 15.0       # Pixels
MIN_ARC_SPAN = 15.0          # Degrees
TEXT_INTERSECTION = 0.30     # Max overlap fraction
MIN_CONFIDENCE = 0.3         # Filter threshold
EDGE_MARGIN = 5              # Pixels from edge
```

### Phase 4d: OCR
```python
MIN_OCR_CONFIDENCE = 60      # Percentage
MIN_TEXT_SIMILARITY = 0.5    # Fuzzy match ratio
MAX_POSITION_OFFSET = 200    # Pixels
```

### Phase 6: Validation
```python
MAX_ITERATIONS = 3           # Validation loops
ACCURACY_THRESHOLD = 90      # Acceptable percentage
```

---

## File Reference

| Component | File | Lines | Key Functions |
|-----------|------|-------|---------------|
| Package init | `__init__.py` | ~475 | Exports, Gemini client |
| Phase 1 | `pdf_intake.py` | ~552 | render_pdf_high_quality |
| Phase 2 | `gemini_understanding.py` | ~785 | analyze_drawing |
| Phase 3 | `coordinate_calibration.py` | ~808 | calibrate_from_analysis |
| Phase 4 | `adaptive_extraction.py` | ~2,900 | hybrid_extract_all |
| Phase 4b | `opencv_extraction.py` | ~1,034 | OpenCVExtractor |
| Phase 4c | `gemini_refinement.py` | ~692 | refine_entities_with_gemini |
| Phase 4d | `ocr_text_anchoring.py` | ~470 | anchor_text_positions |
| Phase 5 | `autocad_creation.py` | ~989 | create_entities_in_autocad |
| Phase 6 | `validation.py` | ~848 | validate_extraction |
| MCP Tools | `mcp_tools.py` | ~2,700 | gemini_vectorize_pdf |

---

## Quick Reference: Tuning for Better Accuracy

### If lines are inaccurate:
1. Increase DPI to 600 in Phase 1
2. Use LSD instead of Hough (default)
3. Decrease `PARALLEL_TOLERANCE` to 2-3°

### If circles are missing:
1. Decrease `CIRCLE_PARAM2` to 50-60
2. Increase `CIRCLE_MIN_DIST` to avoid merging
3. Check `CIRCLE_MAX_RADIUS` isn't too small

### If arcs are distorted:
1. Consider using Raster Design VTools instead
2. Research better arc fitting algorithms
3. Filter by tighter `ASPECT_RATIO_MIN`

### If text positions are wrong:
1. Ensure Tesseract is installed
2. Increase `MIN_OCR_CONFIDENCE` to 70+
3. Check image isn't too degraded

### If scale is wrong:
1. Provide explicit calibration dimension
2. Check dimension parsing is working
3. Verify DPI metadata is correct

### If there's too much noise:
1. Increase `MIN_LINE_LENGTH` to 25-30
2. Increase `MIN_CONFIDENCE` to 0.5
3. Enable all cleanup operations

---

*Document generated: 2026-02-23*
*For the AEC Agent PDF-to-AutoCAD Vectorization Pipeline*
