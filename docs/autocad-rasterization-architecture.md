# AutoCAD PDF Rasterization Pipeline Architecture

## Gemini-First Approach: Understanding Before Processing

---

# 1. Executive Summary

## The Core Problem with Current Implementation

The **existing pipeline** suffers from a fundamental architectural flaw: **preprocessing destroys information before the system understands what it's looking at.**

```
CURRENT PROBLEM: Information Loss Before Understanding
─────────────────────────────────────────────────────────────────────────

PDF (Rich Information)          Bitonal TIFF (Degraded)
┌─────────────────────────┐     ┌─────────────────────────┐
│ • Grayscale shading     │     │ • Binary only (1-bit)   │
│ • Line thickness detail │     │ • Thick lines = blobs   │
│ • Subtle text rendering │ ──▶ │ • Text = noise pixels   │
│ • Symbol fine details   │     │ • Details destroyed     │
│ • Color information     │     │ • Context lost          │
└─────────────────────────┘     └─────────────────────────┘
                                          │
                                          ▼
                                 OpenCV Sees Garbage
                                 ─────────────────────
                                 • False circles at intersections
                                 • Broken lines from thresholding
                                 • Text fragments detected as geometry
                                 • Symbols unrecognizable
                                 • Excessive noise throughout
```

**Result**: No amount of parameter tuning can recover information that was destroyed during preprocessing.

## The New Proposed Approach: Gemini-First

The **proposed pipeline** inverts the architecture: **understand the drawing FIRST using Gemini Vision on the original PDF, THEN decide how to extract.**

```
PROPOSED SOLUTION: Understanding Before Processing
─────────────────────────────────────────────────────────────────────────

PDF (Original Quality)
         │
         ▼
┌─────────────────────────────────────────────────────────────────────┐
│                      GEMINI VISION (FIRST)                          │
│                                                                     │
│  Analyzes ORIGINAL image with full quality:                         │
│  • "This is an MEP floor plan at 1/4" = 1'-0" scale"               │
│  • "Walls are parallel lines, 6" apart, layer A-WALL"              │
│  • "I see 12 diffuser symbols (24x24 supply air)"                  │
│  • "Text labels: Room 101, Room 102, Conference Room"              │
│  • "Ductwork is dashed lines, pipes are single solid lines"        │
│                                                                     │
│  NO PREPROCESSING. NO INFORMATION LOSS. FULL CONTEXT.              │
└─────────────────────────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    ADAPTIVE EXTRACTION                              │
│                                                                     │
│  Based on understanding, Gemini chooses optimal strategy:           │
│  • Direct extraction (Gemini outputs coordinates)                   │
│  • Guided rasterization (Raster Design with precise instructions)  │
│  • Selective OpenCV (only for specific regions that need it)       │
└─────────────────────────────────────────────────────────────────────┘
```

## Verdict: Why Gemini-First is Superior

| Aspect | Current (Preprocess First) | Proposed (Gemini First) |
|--------|---------------------------|-------------------------|
| **Information available** | Degraded bitonal | Original full quality |
| **Context awareness** | None (pixels only) | Full semantic understanding |
| **Noise** | High (preprocessing artifacts) | Zero (no preprocessing) |
| **Text handling** | OCR on degraded image | Direct reading of original |
| **Symbol recognition** | Template match on noise | Vision model on clean image |
| **Adaptability** | Fixed pipeline | Dynamic strategy per drawing |
| **Accuracy potential** | Limited by preprocessing | Limited only by Gemini capability |

## Recommendation

**Replace the current preprocessing-first pipeline with a Gemini-First architecture.**

The preprocessing step is the source of the noise problem. By putting Gemini Vision first, we:
1. Eliminate information loss entirely
2. Gain semantic understanding before any processing
3. Enable adaptive extraction strategies
4. Achieve higher accuracy with less complexity

---

# 2. Current System Analysis

## How the Existing Pipeline Works (And Why It Fails)

```
                         CURRENT PIPELINE (PROBLEMATIC)
┌─────────────────────────────────────────────────────────────────────────────────┐
│                                                                                 │
│  ┌──────────┐    ┌─────────────┐    ┌──────────────┐    ┌────────────────────┐ │
│  │ PDF File │───▶│ PyMuPDF     │───▶│ OpenCV       │───▶│ Semantic Pipeline  │ │
│  │          │    │ BITONAL     │    │ (Sees noise) │    │ (Too late to fix)  │ │
│  │          │    │ CONVERSION  │    │              │    │                    │ │
│  └──────────┘    └─────────────┘    └──────────────┘    └────────────────────┘ │
│                         │                   │                                   │
│                         │                   │                                   │
│               INFORMATION LOSS      NOISE AMPLIFICATION                         │
│               ─────────────────     ────────────────────                        │
│               • Color → gone        • False edges                               │
│               • Grayscale → gone    • Broken lines                              │
│               • Fine detail → gone  • Spurious circles                          │
│               • Text clarity → gone • Text as geometry                          │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘
```

### The Preprocessing Problem in Detail

**Step 1: PDF to Bitonal TIFF** (`pdf_converter.py`)
```python
# This is where information dies
fitz.open(pdf_path)
→ render at 300 DPI
→ convert RGB → Grayscale → Bitonal (threshold 128)  # <-- PROBLEM
→ save as TIFF

# What's lost:
# - Grayscale shading (dimension shadows, fills)
# - Color coding (red = fire, blue = plumbing)
# - Anti-aliased edges (become jagged)
# - Fine text (becomes unreadable)
# - Symbol details (become blobs)
```

**Step 2: OpenCV Detection** (`image_vectorizer.py`)
```python
# OpenCV operates on degraded image
# It can only work with what it's given

# Problems cascade:
# 1. Thick lines from anti-aliasing artifacts
# 2. Line breaks from threshold cutting through
# 3. False circles at every intersection
# 4. Text characters detected as geometry
# 5. Symbols unrecognizable as pixel noise
```

**Step 3: Semantic Enhancement** (Phases A-F)
```python
# By this point, it's too late
# The 6-phase pipeline tries to make sense of garbage
# OCR fails on degraded text
# Symbol matching fails on degraded symbols
# Classification guesses based on noise
```

### Why Parameter Tuning Can't Fix This

| Parameter | What it does | Why it can't help |
|-----------|--------------|-------------------|
| `threshold` | Binary cutoff (0-255) | Any value loses information |
| `dpi` | Resolution | Higher DPI = more noise detail |
| `min_line_length` | Filter short lines | Real lines also get filtered |
| `hough_threshold` | Detection sensitivity | Low = more noise, High = miss lines |
| `despeckle` | Remove small blobs | Also removes small text/details |

**The fundamental issue**: You cannot recover information that was destroyed. No algorithm can reconstruct what the preprocessing removed.

## Current System Strengths (Worth Keeping)

Despite the preprocessing problem, some components are valuable:

| Component | Value | Keep? |
|-----------|-------|-------|
| AutoCAD sidecar integration | Solid, well-tested | Yes |
| MCP tool framework | Clean architecture | Yes |
| PostgreSQL storage | Spatial + embeddings work well | Yes |
| Draw commands | `draw_line`, `draw_circle`, etc. work | Yes |
| 6-phase semantic pipeline concepts | Good ideas, wrong input | Rearchitect |

## Current System Limitations (Must Fix)

| Limitation | Impact | Root Cause |
|------------|--------|------------|
| **Excessive noise** | False positives everywhere | Bitonal preprocessing |
| **Broken lines** | Gaps in continuous runs | Threshold artifacts |
| **Text detection failures** | OCR on degraded image | Information loss |
| **Symbol recognition failures** | Matching noise to templates | Information loss |
| **Manual cleanup required** | 20%+ entities need fixing | Accumulated errors |

---

# 3. New Proposed Architecture: Gemini-First

## Core Philosophy

**"Understand first, then extract."**

Instead of:
```
PDF → Destroy information → Try to detect → Try to understand
```

We do:
```
PDF → Understand fully → Extract intelligently → Verify
```

## System Architecture Diagram

```
                           GEMINI-FIRST ARCHITECTURE
┌──────────────────────────────────────────────────────────────────────────────────┐
│                                                                                  │
│  ┌────────────────────────────────────────────────────────────────────────────┐ │
│  │                    PHASE 1: UNDERSTANDING                                  │ │
│  │                    (Gemini Vision on ORIGINAL PDF)                         │ │
│  │  ┌──────────────────────────────────────────────────────────────────────┐ │ │
│  │  │                                                                      │ │ │
│  │  │  INPUT: Original PDF (no preprocessing, full quality)               │ │ │
│  │  │                                                                      │ │ │
│  │  │  GEMINI ANALYZES:                                                    │ │ │
│  │  │  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐ ┌─────────────┐   │ │ │
│  │  │  │ Drawing     │ │ Element     │ │ Text        │ │ Symbols     │   │ │ │
│  │  │  │ Type &      │ │ Inventory   │ │ Content     │ │ Identified  │   │ │ │
│  │  │  │ Scale       │ │ & Locations │ │ & Meaning   │ │ & Typed     │   │ │ │
│  │  │  └─────────────┘ └─────────────┘ └─────────────┘ └─────────────┘   │ │ │
│  │  │                                                                      │ │ │
│  │  │  OUTPUT: Complete semantic understanding + extraction plan           │ │ │
│  │  │                                                                      │ │ │
│  │  └──────────────────────────────────────────────────────────────────────┘ │ │
│  └────────────────────────────────────────────────────────────────────────────┘ │
│                                        │                                         │
│                                        ▼                                         │
│  ┌────────────────────────────────────────────────────────────────────────────┐ │
│  │                    PHASE 2: STRATEGY SELECTION                             │ │
│  │                    (Gemini decides HOW to extract)                         │ │
│  │                                                                            │ │
│  │  Based on drawing complexity and element types:                            │ │
│  │                                                                            │ │
│  │  ┌──────────────────┐  ┌──────────────────┐  ┌──────────────────┐         │ │
│  │  │ DIRECT           │  │ GUIDED           │  │ SELECTIVE        │         │ │
│  │  │ EXTRACTION       │  │ RASTERIZATION    │  │ OPENCV           │         │ │
│  │  │                  │  │                  │  │                  │         │ │
│  │  │ Gemini outputs   │  │ Gemini directs   │  │ OpenCV on        │         │ │
│  │  │ coordinates      │  │ Raster Design    │  │ specific regions │         │ │
│  │  │ directly         │  │ tools            │  │ only             │         │ │
│  │  │                  │  │                  │  │                  │         │ │
│  │  │ Best for:        │  │ Best for:        │  │ Best for:        │         │ │
│  │  │ • Simple drawings│  │ • Complex curves │  │ • Dense details  │         │ │
│  │  │ • Clear elements │  │ • Precise tracing│  │ • Hatching       │         │ │
│  │  │ • Text & symbols │  │ • Connected paths│  │ • Repetitive     │         │ │
│  │  └──────────────────┘  └──────────────────┘  └──────────────────┘         │ │
│  │                                                                            │ │
│  └────────────────────────────────────────────────────────────────────────────┘ │
│                                        │                                         │
│                                        ▼                                         │
│  ┌────────────────────────────────────────────────────────────────────────────┐ │
│  │                    PHASE 3: EXTRACTION                                     │ │
│  │                    (Execute chosen strategy)                               │ │
│  │                                                                            │ │
│  │  ┌──────────────────────────────────────────────────────────────────────┐ │ │
│  │  │                     AutoCAD Integration                              │ │ │
│  │  │                                                                      │ │ │
│  │  │  draw_line, draw_circle, draw_arc, draw_polyline                    │ │ │
│  │  │  draw_mtext, insert_block                                           │ │ │
│  │  │  Raster Design VTools (when needed)                                 │ │ │
│  │  │                                                                      │ │ │
│  │  └──────────────────────────────────────────────────────────────────────┘ │ │
│  └────────────────────────────────────────────────────────────────────────────┘ │
│                                        │                                         │
│                                        ▼                                         │
│  ┌────────────────────────────────────────────────────────────────────────────┐ │
│  │                    PHASE 4: VALIDATION                                     │ │
│  │                    (Gemini verifies against original)                      │ │
│  │                                                                            │ │
│  │  Gemini compares extracted entities to original PDF:                       │ │
│  │  • Missing elements? → Extract them                                        │ │
│  │  • Wrong positions? → Correct them                                         │ │
│  │  • Text errors? → Fix them                                                 │ │
│  │  • Symbol mismatches? → Replace them                                       │ │
│  │                                                                            │ │
│  │  Loop until Gemini approves or max iterations reached                      │ │
│  └────────────────────────────────────────────────────────────────────────────┘ │
│                                        │                                         │
│                                        ▼                                         │
│  ┌────────────────────────────────────────────────────────────────────────────┐ │
│  │                         OUTPUT: Clean DWG                                  │ │
│  │                         Accurate, validated, minimal manual cleanup        │ │
│  └────────────────────────────────────────────────────────────────────────────┘ │
│                                                                                  │
└──────────────────────────────────────────────────────────────────────────────────┘
```

## Component Breakdown

### Gemini Vision (The Brain)

**Role**: First point of contact with the PDF. Understands everything before any processing.

**Input**: Original PDF rendered at high quality (no bitonal conversion)

**Capabilities**:
| Capability | How Gemini Does It |
|------------|-------------------|
| **Drawing type detection** | Recognizes floor plans, electrical, mechanical, etc. |
| **Scale inference** | Reads scale notation, measures known elements |
| **Element identification** | Sees lines, arcs, circles, text, symbols directly |
| **Text reading** | Reads text as text, not OCR on pixels |
| **Symbol recognition** | Identifies symbol types from visual appearance |
| **Line type detection** | Sees solid vs. dashed vs. dotted clearly |
| **Layer inference** | Understands which elements belong together |
| **Spatial relationships** | Knows what connects to what |

**Output**: Structured understanding document (JSON)

```json
{
  "drawing_analysis": {
    "type": "mechanical_floor_plan",
    "scale": "1/4\" = 1'-0\"",
    "units": "imperial",
    "sheet_size": "24x36",
    "complexity": "medium",
    "estimated_entities": 450
  },
  "regions": [
    {"type": "title_block", "bounds": [0, 0, 800, 150]},
    {"type": "drawing_area", "bounds": [0, 150, 2400, 1800]},
    {"type": "legend", "bounds": [2000, 150, 2400, 600]}
  ],
  "elements": {
    "lines": [
      {"start": [100, 200], "end": [500, 200], "type": "wall", "linetype": "continuous"},
      {"start": [100, 200], "end": [100, 600], "type": "wall", "linetype": "continuous"}
    ],
    "arcs": [
      {"center": [500, 200], "radius": 30, "start_angle": 0, "end_angle": 90, "type": "door_swing"}
    ],
    "text": [
      {"content": "ROOM 101", "position": [300, 400], "height": 12, "layer": "A-ANNO"},
      {"content": "24x24 SA", "position": [350, 350], "meaning": "24x24 inch supply air diffuser"}
    ],
    "symbols": [
      {"type": "diffuser", "subtype": "supply_square", "position": [350, 350], "size": "24x24"},
      {"type": "thermostat", "position": [480, 400]}
    ]
  },
  "extraction_strategy": {
    "recommended": "direct_extraction",
    "rationale": "Drawing is clean, elements are clearly defined, direct coordinate extraction will work well"
  }
}
```

### Extraction Strategies

#### Strategy A: Direct Extraction

**When to use**: Simple to medium complexity drawings with clear, well-defined elements.

**How it works**:
1. Gemini identifies each element with coordinates
2. System directly creates AutoCAD entities from Gemini's output
3. No image processing at all

```
Gemini says:              System executes:
─────────────────────     ─────────────────────────────────────
"Line from (100,200)      draw_line(
 to (500,200),             start=(100,200),
 layer A-WALL,             end=(500,200),
 continuous"               layer="A-WALL",
                           linetype="CONTINUOUS"
                          )
```

**Advantages**:
- Zero preprocessing noise
- Direct semantic-to-entity mapping
- Fastest execution path

**Challenges**:
- Coordinate precision depends on Gemini's visual accuracy
- Need calibration points for scale

#### Strategy B: Guided Rasterization

**When to use**: Complex curves, connected paths, when precise pixel tracing is needed.

**How it works**:
1. Gemini identifies regions and element types
2. Gemini provides specific instructions for Raster Design tools
3. Raster Design traces the actual pixels with intelligent guidance

```
Gemini says:              System executes:
─────────────────────     ─────────────────────────────────────
"Complex curve at         1. Attach raster image
 region (100,100)-        2. Set layer M-DUCT
 (300,300), use           3. Execute VFPLINE starting at (150,150)
 VFPline to trace         4. Follow until endpoint
 the ductwork,
 layer M-DUCT"
```

**Advantages**:
- Pixel-accurate tracing
- Native connectivity
- Handles complex geometry well

**When to prefer over direct extraction**:
- Irregular curves that are hard to describe mathematically
- Very dense areas where precise tracing matters
- When original line quality is high

#### Strategy C: Selective OpenCV

**When to use**: Specific regions with repetitive patterns, hatching, or dense details.

**How it works**:
1. Gemini identifies regions that would benefit from automated detection
2. Only those regions are preprocessed and processed with OpenCV
3. Results are integrated with directly extracted elements

```
Gemini says:              System executes:
─────────────────────     ─────────────────────────────────────
"Region (500,500)-        1. Crop only that region
 (700,700) contains       2. Preprocess minimally (not bitonal!)
 dense hatching,          3. Run line detection
 use OpenCV for           4. Filter by Gemini's expected pattern
 parallel line            5. Create entities
 detection"
```

**Advantages**:
- Efficient for repetitive patterns
- OpenCV is fast for batch detection
- Gemini pre-filters to reduce noise

**Key difference from current approach**:
- Only used for specific regions, not entire drawing
- Gemini validates results against expectations
- Preprocessing is minimal and targeted

### Validation Loop

**Purpose**: Ensure extraction accuracy before completing.

**Process**:
```
┌─────────────────────────────────────────────────────────────────┐
│                     VALIDATION LOOP                             │
└─────────────────────────────────────────────────────────────────┘

        ┌─────────────────────────────┐
        │  Extracted entities (DWG)   │
        └──────────────┬──────────────┘
                       │
                       ▼
        ┌─────────────────────────────┐
        │  Gemini compares to         │
        │  original PDF               │◀──────────────────────┐
        └──────────────┬──────────────┘                       │
                       │                                      │
                       ▼                                      │
        ┌─────────────────────────────┐                       │
        │  Discrepancies found?       │                       │
        └──────────────┬──────────────┘                       │
                       │                                      │
            ┌──────────┴──────────┐                           │
            │                     │                           │
            ▼                     ▼                           │
    ┌───────────────┐     ┌───────────────┐                   │
    │     YES       │     │      NO       │                   │
    │               │     │               │                   │
    │ Generate      │     │ Validation    │                   │
    │ corrections   │     │ complete      │                   │
    └───────┬───────┘     └───────────────┘                   │
            │                                                 │
            ▼                                                 │
    ┌───────────────┐                                         │
    │ Apply fixes   │─────────────────────────────────────────┘
    │ to DWG        │
    └───────────────┘
```

**What Gemini checks**:
- Missing elements (something in PDF not in DWG)
- Extra elements (false positives)
- Position errors (element in wrong place)
- Text errors (wrong content or placement)
- Symbol mismatches (wrong type or attributes)
- Connectivity issues (gaps or overlaps)

---

# 4. Detailed Comparison: Current vs. Gemini-First

## Side-by-Side Architecture Comparison

```
CURRENT ARCHITECTURE                    GEMINI-FIRST ARCHITECTURE
────────────────────────────────────    ────────────────────────────────────

PDF                                     PDF
 │                                       │
 ▼                                       ▼
┌─────────────────────┐                 ┌─────────────────────┐
│ Bitonal Conversion  │                 │ Gemini Vision       │
│ (DESTROYS INFO)     │                 │ (UNDERSTANDS ALL)   │
└─────────┬───────────┘                 └─────────┬───────────┘
          │                                       │
          ▼                                       ▼
┌─────────────────────┐                 ┌─────────────────────┐
│ OpenCV Detection    │                 │ Strategy Selection  │
│ (SEES NOISE)        │                 │ (INTELLIGENT)       │
└─────────┬───────────┘                 └─────────┬───────────┘
          │                                       │
          ▼                                       ▼
┌─────────────────────┐                 ┌─────────────────────┐
│ Semantic Pipeline   │                 │ Targeted Extraction │
│ (TOO LATE)          │                 │ (PRECISE)           │
└─────────┬───────────┘                 └─────────┬───────────┘
          │                                       │
          ▼                                       ▼
┌─────────────────────┐                 ┌─────────────────────┐
│ AutoCAD Drawing     │                 │ Validation Loop     │
│ (NOISE PROPAGATES)  │                 │ (SELF-CORRECTING)   │
└─────────┬───────────┘                 └─────────┬───────────┘
          │                                       │
          ▼                                       ▼
┌─────────────────────┐                 ┌─────────────────────┐
│ Manual Cleanup      │                 │ Clean Output        │
│ (20%+ REQUIRED)     │                 │ (<5% CLEANUP)       │
└─────────────────────┘                 └─────────────────────┘
```

## Feature Comparison Table

| Feature | Current Pipeline | Gemini-First Pipeline |
|---------|-----------------|----------------------|
| **First step** | Destroy information (bitonal) | Understand information (Gemini) |
| **Information preserved** | ~30% (binary only) | 100% (original PDF) |
| **Context awareness** | None until Phase D | Immediate, full context |
| **Text handling** | Tesseract OCR on degraded | Gemini reads original directly |
| **Symbol recognition** | Template match on noise | Vision recognition on clean |
| **Line detection** | Edge detection on artifacts | Semantic identification |
| **Decision making** | Fixed pipeline, same for all | Adaptive per drawing |
| **Noise level** | High (preprocessing artifacts) | Zero (no preprocessing) |
| **Validation** | None (hope for the best) | Gemini verifies against original |
| **Manual cleanup** | 20%+ entities | <5% entities |
| **Processing path** | One size fits all | Optimized per element type |

## Information Flow Comparison

### Current: Information Destruction Pipeline

```
Stage           Information Level
──────────────────────────────────────────────────────────
PDF Input       ████████████████████████████████ 100%
                        │
Bitonal Convert         ▼
                ██████████░░░░░░░░░░░░░░░░░░░░░░  30%
                        │
OpenCV Detect           ▼
                ████████░░░░░░░░░░░░░░░░░░░░░░░░  25%
                        │
Add Semantic            ▼
                ██████████░░░░░░░░░░░░░░░░░░░░░░  30%
                        │
Final Output            ▼
                ██████████████░░░░░░░░░░░░░░░░░░  40%

Result: Lost 60% of information, high noise
```

### Gemini-First: Information Preservation Pipeline

```
Stage           Information Level
──────────────────────────────────────────────────────────
PDF Input       ████████████████████████████████ 100%
                        │
Gemini Analyze          ▼
                ██████████████████████████████░░  95%
                        │
Extract                 ▼
                ████████████████████████████░░░░  90%
                        │
Validate+Fix            ▼
                ██████████████████████████████░░  95%

Result: Preserved 95% of information, minimal noise
```

## Noise Comparison

### Current Pipeline Noise Sources

| Source | Type | Impact |
|--------|------|--------|
| Threshold conversion | Systematic | Breaks lines, destroys gradients |
| Anti-aliasing loss | Systematic | Creates jagged edges |
| Small detail loss | Systematic | Text and symbols become blobs |
| Edge detection errors | Random | False positives at intersections |
| Hough voting noise | Random | Ghost circles, phantom lines |
| Template matching failures | Random | Wrong symbol matches |

### Gemini-First Noise Sources

| Source | Type | Impact | Mitigation |
|--------|------|--------|------------|
| Gemini coordinate imprecision | Systematic | ±1-2 pixels | Calibration points |
| Vision model errors | Random | Rare misidentification | Validation loop |
| Complex region extraction | Random | Occasional gaps | Re-extraction |

**Net noise reduction: ~90%**

---

# 5. Detailed Workflow

## Step-by-Step Processing Pipeline

### Step 1: PDF Intake

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           STEP 1: PDF INTAKE                                │
└─────────────────────────────────────────────────────────────────────────────┘

User provides: technical_drawing.pdf

System actions:
1. Render PDF at high quality (300+ DPI)
2. DO NOT convert to bitonal
3. Keep as RGB or high-quality grayscale
4. Identify page dimensions and any embedded metadata
5. Prepare for Gemini Vision analysis

Output: High-quality raster image ready for Gemini
```

### Step 2: Gemini Understanding Phase

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    STEP 2: GEMINI UNDERSTANDING                             │
└─────────────────────────────────────────────────────────────────────────────┘

Input: Original quality PDF image

Gemini Prompt:
┌─────────────────────────────────────────────────────────────────────────────┐
│ Analyze this technical drawing and provide:                                 │
│                                                                             │
│ 1. DRAWING IDENTIFICATION                                                   │
│    - Type (floor plan, electrical, mechanical, plumbing, etc.)              │
│    - Scale (if shown or inferable)                                          │
│    - Sheet size and orientation                                             │
│                                                                             │
│ 2. REGION MAPPING                                                           │
│    - Title block location and content                                       │
│    - Main drawing area bounds                                               │
│    - Legend location if present                                             │
│    - Notes/schedule locations                                               │
│                                                                             │
│ 3. ELEMENT INVENTORY                                                        │
│    For each element type, provide:                                          │
│    - Count (approximate)                                                    │
│    - Locations (coordinates or regions)                                     │
│    - Specific attributes (linetype, layer suggestion)                       │
│                                                                             │
│    Element types to identify:                                               │
│    - Lines (walls, ductwork, piping, wiring)                                │
│    - Arcs (door swings, curved walls)                                       │
│    - Circles (columns, equipment)                                           │
│    - Text (room names, dimensions, notes)                                   │
│    - Symbols (diffusers, outlets, valves, fixtures)                         │
│    - Dimensions (with values)                                               │
│                                                                             │
│ 4. EXTRACTION STRATEGY RECOMMENDATION                                       │
│    - Which strategy for which elements                                      │
│    - Any special handling needed                                            │
│    - Complexity assessment                                                  │
│                                                                             │
│ Return as structured JSON.                                                  │
└─────────────────────────────────────────────────────────────────────────────┘

Gemini Response (example):
{
  "drawing_type": "mechanical_floor_plan",
  "scale": "1/8\" = 1'-0\"",
  "sheet_size": "ARCH D (24x36)",
  "complexity": "medium",

  "regions": {
    "title_block": {"bounds": [2100, 0, 2400, 150], "content": "Project: Office Building, Sheet: M-101"},
    "drawing_area": {"bounds": [0, 150, 2100, 1650]},
    "legend": {"bounds": [2100, 150, 2400, 600]}
  },

  "elements": {
    "walls": {
      "count": 45,
      "strategy": "direct_extraction",
      "items": [
        {"start": [100, 200], "end": [800, 200], "thickness": 6, "layer": "A-WALL"},
        {"start": [100, 200], "end": [100, 900], "thickness": 6, "layer": "A-WALL"}
      ]
    },
    "ductwork": {
      "count": 28,
      "strategy": "direct_extraction",
      "items": [
        {"start": [150, 300], "end": [600, 300], "width": 24, "linetype": "dashed", "layer": "M-DUCT"}
      ]
    },
    "diffusers": {
      "count": 12,
      "strategy": "direct_extraction",
      "items": [
        {"type": "supply_square", "position": [300, 400], "size": "24x24", "tag": "SD-1"},
        {"type": "return_square", "position": [500, 400], "size": "24x24", "tag": "RD-1"}
      ]
    },
    "text_labels": {
      "count": 35,
      "items": [
        {"content": "CONFERENCE ROOM", "position": [400, 500], "height": 12},
        {"content": "200 CFM", "position": [305, 390], "associated_with": "diffuser at (300,400)"}
      ]
    }
  },

  "extraction_strategy": {
    "primary": "direct_extraction",
    "rationale": "Clean drawing, well-defined elements, text is legible",
    "special_regions": [
      {"region": [1800, 800, 2000, 1000], "strategy": "guided_rasterization", "reason": "Complex piping detail"}
    ]
  }
}
```

### Step 3: Strategy Execution

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                     STEP 3: STRATEGY EXECUTION                              │
└─────────────────────────────────────────────────────────────────────────────┘

Based on Gemini's analysis, execute appropriate strategy for each element:

FOR DIRECT EXTRACTION ELEMENTS:
───────────────────────────────────────────────────────────────────────────────
Gemini provided:
  {"start": [100, 200], "end": [800, 200], "layer": "A-WALL"}

System executes:
  draw_line(
    start=(100 * scale_factor, 200 * scale_factor),
    end=(800 * scale_factor, 200 * scale_factor),
    layer="A-WALL",
    linetype="CONTINUOUS"
  )

FOR SYMBOLS:
───────────────────────────────────────────────────────────────────────────────
Gemini provided:
  {"type": "supply_square", "position": [300, 400], "size": "24x24"}

System executes:
  insert_block(
    block_name="M-DIFF-SQ-SUPPLY",
    position=(300 * scale_factor, 400 * scale_factor),
    scale=1.0,
    layer="M-DIFF"
  )

FOR TEXT:
───────────────────────────────────────────────────────────────────────────────
Gemini provided:
  {"content": "CONFERENCE ROOM", "position": [400, 500], "height": 12}

System executes:
  draw_mtext(
    content="CONFERENCE ROOM",
    position=(400 * scale_factor, 500 * scale_factor),
    height=12,
    layer="A-ANNO-TEXT"
  )

FOR GUIDED RASTERIZATION REGIONS:
───────────────────────────────────────────────────────────────────────────────
Gemini provided:
  {"region": [1800, 800, 2000, 1000], "strategy": "guided_rasterization"}

System executes:
  1. Ensure raster image is attached
  2. Zoom to region
  3. Execute Raster Design VFPline at Gemini-specified start points
  4. Follow until endpoints
```

### Step 4: Coordinate Calibration

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                   STEP 4: COORDINATE CALIBRATION                            │
└─────────────────────────────────────────────────────────────────────────────┘

Challenge: Gemini provides pixel coordinates, AutoCAD needs DWG coordinates.

Solution: Use calibration points.

Method 1: Known Dimension Calibration
───────────────────────────────────────────────────────────────────────────────
Gemini identifies:
  - A dimension line showing "20'-0""
  - Pixel distance: 320 pixels

Calculation:
  scale_factor = 240 inches / 320 pixels = 0.75 inches/pixel

Method 2: Sheet Size Calibration
───────────────────────────────────────────────────────────────────────────────
Gemini identifies:
  - Sheet size: ARCH D (24" x 36")
  - Image size: 2400 x 3600 pixels

Calculation:
  scale_factor_x = 36" / 3600 pixels = 0.01 inches/pixel
  scale_factor_y = 24" / 2400 pixels = 0.01 inches/pixel

Method 3: Reference Point Calibration
───────────────────────────────────────────────────────────────────────────────
Gemini identifies two known points:
  - Grid intersection A1 at pixel (100, 100), should be (0, 0) in DWG
  - Grid intersection B1 at pixel (420, 100), should be (32', 0) in DWG

Calculation:
  scale_factor = 32' / 320 pixels = 1.2 inches/pixel

Application:
  dwg_x = pixel_x * scale_factor
  dwg_y = (image_height - pixel_y) * scale_factor  # Y-axis flip
```

### Step 5: Validation Loop

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                      STEP 5: VALIDATION LOOP                                │
└─────────────────────────────────────────────────────────────────────────────┘

After extraction, Gemini validates by comparing output to original.

Validation Prompt:
┌─────────────────────────────────────────────────────────────────────────────┐
│ Compare the extracted entities against the original drawing.               │
│                                                                             │
│ Original: [PDF image]                                                       │
│ Extracted: [List of entities created]                                       │
│                                                                             │
│ Check for:                                                                  │
│ 1. MISSING ELEMENTS - Anything in original not in extracted                │
│ 2. EXTRA ELEMENTS - False positives that shouldn't exist                   │
│ 3. POSITION ERRORS - Elements in wrong location                            │
│ 4. TEXT ERRORS - Wrong text content                                        │
│ 5. SYMBOL ERRORS - Wrong symbol type or attributes                         │
│ 6. CONNECTIVITY - Gaps or overlaps that shouldn't exist                    │
│                                                                             │
│ For each issue found, provide correction instructions.                     │
└─────────────────────────────────────────────────────────────────────────────┘

Gemini Response (example):
{
  "validation_result": "issues_found",
  "issues": [
    {
      "type": "missing_element",
      "description": "Diffuser at approximately (700, 450) not extracted",
      "correction": {
        "action": "insert_block",
        "block_name": "M-DIFF-SQ-SUPPLY",
        "position": [700, 450],
        "size": "24x24"
      }
    },
    {
      "type": "text_error",
      "description": "Text at (400, 500) shows 'CONFRENCE' should be 'CONFERENCE'",
      "correction": {
        "action": "edit_mtext",
        "position": [400, 500],
        "new_content": "CONFERENCE ROOM"
      }
    },
    {
      "type": "position_error",
      "description": "Wall line endpoint at (800, 200) should connect to (800, 205)",
      "correction": {
        "action": "modify_line",
        "line_id": "wall_3",
        "new_end": [800, 205]
      }
    }
  ],
  "overall_accuracy": "94%",
  "recommendation": "Apply corrections and re-validate"
}

System applies corrections and loops until:
- validation_result == "approved" OR
- iteration_count >= max_iterations (e.g., 3)
```

---

# 6. Extraction Strategies Deep Dive

## Strategy A: Direct Extraction

### Overview

Gemini analyzes the image and outputs exact coordinates for each element. The system creates AutoCAD entities directly from these coordinates without any image processing.

### When to Use

| Drawing Characteristic | Use Direct Extraction? |
|-----------------------|------------------------|
| Clean, high-quality scan | Yes |
| Simple geometry (mostly lines) | Yes |
| Text is clearly legible | Yes |
| Symbols are standard and recognizable | Yes |
| Low element density | Yes |
| Complex curves | No (use guided) |
| Dense hatching | No (use selective OpenCV) |
| Poor quality / noisy original | Maybe (with more validation) |

### Implementation

```python
async def direct_extraction(gemini_analysis: dict) -> list[Entity]:
    """
    Create AutoCAD entities directly from Gemini's analysis.
    No image processing. Pure semantic-to-entity mapping.
    """
    entities = []

    # Get scale factor from calibration
    scale = calculate_scale_factor(gemini_analysis)

    # Process lines
    for line in gemini_analysis["elements"].get("lines", []):
        entity = await draw_line(
            start=scale_point(line["start"], scale),
            end=scale_point(line["end"], scale),
            layer=line.get("layer", "0"),
            linetype=line.get("linetype", "CONTINUOUS")
        )
        entities.append(entity)

    # Process arcs
    for arc in gemini_analysis["elements"].get("arcs", []):
        entity = await draw_arc(
            center=scale_point(arc["center"], scale),
            radius=arc["radius"] * scale.factor,
            start_angle=arc["start_angle"],
            end_angle=arc["end_angle"],
            layer=arc.get("layer", "0")
        )
        entities.append(entity)

    # Process circles
    for circle in gemini_analysis["elements"].get("circles", []):
        entity = await draw_circle(
            center=scale_point(circle["center"], scale),
            radius=circle["radius"] * scale.factor,
            layer=circle.get("layer", "0")
        )
        entities.append(entity)

    # Process symbols (block insertions)
    for symbol in gemini_analysis["elements"].get("symbols", []):
        block_name = map_symbol_to_block(symbol["type"], symbol.get("subtype"))
        entity = await insert_block(
            block_name=block_name,
            position=scale_point(symbol["position"], scale),
            scale=symbol.get("scale", 1.0),
            rotation=symbol.get("rotation", 0),
            layer=get_layer_for_symbol(symbol["type"])
        )
        entities.append(entity)

    # Process text
    for text in gemini_analysis["elements"].get("text", []):
        entity = await draw_mtext(
            content=text["content"],
            position=scale_point(text["position"], scale),
            height=text.get("height", 12) * scale.factor,
            layer=text.get("layer", "0")
        )
        entities.append(entity)

    return entities
```

### Accuracy Considerations

| Factor | Impact | Mitigation |
|--------|--------|------------|
| Gemini coordinate precision | ±1-3 pixels | Use calibration points, validation loop |
| Scale inference accuracy | ±5% | Use known dimensions for calibration |
| Y-axis flip | Systematic | Always apply: `y_dwg = height - y_pixel` |
| Symbol rotation detection | ±15° | Gemini can infer from context |

## Strategy B: Guided Rasterization

### Overview

Gemini provides precise instructions for AutoCAD Raster Design tools. The tools trace the actual raster pixels, but Gemini tells them where to start, what layer to use, and when to stop.

### When to Use

| Drawing Characteristic | Use Guided Rasterization? |
|-----------------------|--------------------------|
| Complex, irregular curves | Yes |
| Splines and free-form shapes | Yes |
| Connected pathways (piping, ductwork) | Yes |
| High-precision requirements | Yes |
| Simple straight lines | No (direct is faster) |
| Standard symbols | No (direct is faster) |
| Text | No (direct is better) |

### Implementation

```python
async def guided_rasterization(gemini_analysis: dict, region: dict) -> list[Entity]:
    """
    Use Raster Design tools with Gemini's guidance.
    Gemini tells us where and how; VTools do the precise tracing.
    """
    entities = []

    # Ensure raster image is attached
    await ensure_raster_attached(gemini_analysis["source_image"])

    # Get Gemini's guidance for this region
    guidance = region.get("guidance", {})

    # For each path Gemini identified
    for path in guidance.get("paths", []):
        # Set the target layer
        await set_current_layer(path["layer"])

        # Determine which VTool to use
        vtool = select_vtool(path["type"])

        if vtool == "VFPLINE":
            # Follower polyline - traces connected pixels
            entity = await execute_vfpline(
                start_point=path["start"],
                expected_end=path.get("end"),  # Optional endpoint guidance
                options={
                    "gap_jump": path.get("gap_jump", 3),
                    "corner_threshold": path.get("corner_threshold", 45)
                }
            )
        elif vtool == "VFCONTOUR":
            # Contour follower - traces closed shapes
            entity = await execute_vfcontour(
                start_point=path["start"],
                direction=path.get("direction", "clockwise")
            )
        elif vtool == "VARC":
            # Arc primitive
            entity = await execute_varc(
                region=path["region"],
                expected_center=path.get("center")
            )

        entities.append(entity)

    return entities

def select_vtool(path_type: str) -> str:
    """Select the appropriate Raster Design tool based on path type."""
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

### Hybrid Approach: Gemini + VTools

```
Gemini's Role:                    VTool's Role:
─────────────────────────────     ─────────────────────────────
"Start VFPline at (100,200)"      Traces exact pixels from there
"This is ductwork, layer M-DUCT"  Uses specified layer
"Follow until you reach (500,200)" Knows when to stop
"It should be one continuous path" Maintains connectivity
"Expect corners at ~45 degrees"    Handles corners appropriately
```

## Strategy C: Selective OpenCV

### Overview

For specific regions with patterns better suited to automated detection (dense hatching, repetitive elements), use OpenCV but only on those targeted regions.

### Key Difference from Current Pipeline

| Current Pipeline | Selective OpenCV |
|-----------------|------------------|
| Process entire image | Process specific regions only |
| Bitonal conversion of everything | Minimal preprocessing of region |
| No context/expectations | Gemini provides expected pattern |
| Accept all detections | Filter by Gemini's expectations |
| No validation | Gemini validates results |

### When to Use

| Pattern Type | Use Selective OpenCV? |
|-------------|----------------------|
| Parallel line hatching | Yes |
| Grid patterns | Yes |
| Repetitive symbols | Yes |
| Dense mechanical details | Yes |
| Regular text blocks | Maybe |
| Irregular curves | No |
| Unique symbols | No |

### Implementation

```python
async def selective_opencv(
    gemini_analysis: dict,
    region: dict,
    original_image: np.ndarray
) -> list[Entity]:
    """
    Use OpenCV on specific regions where it excels.
    Gemini pre-filters and post-validates.
    """
    entities = []

    # 1. Crop just this region from the ORIGINAL (not bitonal) image
    bounds = region["bounds"]
    cropped = original_image[bounds[1]:bounds[3], bounds[0]:bounds[2]]

    # 2. Minimal preprocessing - NOT bitonal conversion
    # Use adaptive threshold to preserve detail
    gray = cv2.cvtColor(cropped, cv2.COLOR_RGB2GRAY)
    processed = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        blockSize=11, C=2
    )

    # 3. Detect based on expected pattern
    expected_pattern = region.get("expected_pattern")

    if expected_pattern == "parallel_lines":
        # Detect parallel lines using Hough with tight angle constraints
        lines = detect_parallel_lines(
            processed,
            expected_angle=region.get("angle", 0),
            expected_spacing=region.get("spacing"),
            tolerance=5  # degrees
        )

        # Filter by Gemini's expectations
        for line in lines:
            if matches_expectation(line, region):
                entities.append(create_line_entity(line, bounds, region["layer"]))

    elif expected_pattern == "grid":
        # Detect grid pattern
        h_lines, v_lines = detect_grid(
            processed,
            expected_h_spacing=region.get("h_spacing"),
            expected_v_spacing=region.get("v_spacing")
        )
        for line in h_lines + v_lines:
            entities.append(create_line_entity(line, bounds, region["layer"]))

    # 4. Gemini validates the results
    validation = await gemini_validate_region(
        original_region=cropped,
        detected_entities=entities,
        expectations=region
    )

    if validation["issues"]:
        # Apply corrections
        entities = apply_corrections(entities, validation["corrections"])

    return entities
```

---

# 7. Text Handling: Semantic Understanding

## The Core Difference

### Current: OCR on Degraded Image

```
PDF → Bitonal → Tesseract OCR → Pattern Matching → Maybe Correct

Problems:
• Image is degraded before OCR sees it
• OCR makes character-level errors
• Pattern matching tries to fix but often fails
• Context is lost
```

### Gemini-First: Direct Semantic Reading

```
PDF → Gemini Vision → Understands Meaning Directly

Advantages:
• Gemini sees original quality
• Gemini reads meaning, not characters
• Context informs understanding
• Self-correcting via validation
```

## Examples of Semantic Text Understanding

### Example 1: Ambiguous Characters

```
┌─────────────────────────────────────────────────────────────────────────────┐
│ Visual in PDF:    "1OO"  (zero vs letter O is unclear)                     │
│                                                                             │
│ Current OCR:      "1OO" or "100" or "1O0" (random guess)                   │
│                                                                             │
│ Gemini sees:      - This is a dimension text                               │
│                   - It's associated with a dimension line                   │
│                   - The line spans approximately 100 units at this scale   │
│                   - Therefore: "100" (one hundred)                         │
│                                                                             │
│ Gemini outputs:   {"content": "100", "type": "dimension", "value": 100}    │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Example 2: Partial Occlusion

```
┌─────────────────────────────────────────────────────────────────────────────┐
│ Visual in PDF:    "CON___CE ROOM"  (middle is obscured/damaged)            │
│                                                                             │
│ Current OCR:      "CON CE ROOM" or "CONCE ROOM" (incomplete)               │
│                                                                             │
│ Gemini sees:      - This is a room label                                   │
│                   - Room contains conference table with chairs              │
│                   - Common room type in office buildings                    │
│                   - Therefore: "CONFERENCE ROOM"                           │
│                                                                             │
│ Gemini outputs:   {"content": "CONFERENCE ROOM", "type": "room_name"}      │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Example 3: Technical Abbreviations

```
┌─────────────────────────────────────────────────────────────────────────────┐
│ Visual in PDF:    "24x24 SA 200CFM"                                        │
│                                                                             │
│ Current OCR:      "24x24 SA 200CFM" (correct but no understanding)         │
│                                                                             │
│ Gemini sees:      - This is associated with a diffuser symbol               │
│                   - "24x24" means 24 inches by 24 inches                   │
│                   - "SA" means Supply Air                                   │
│                   - "200CFM" means 200 cubic feet per minute               │
│                   - This is a supply air diffuser specification            │
│                                                                             │
│ Gemini outputs:   {                                                        │
│                     "content": "24x24 SA 200CFM",                          │
│                     "type": "equipment_tag",                               │
│                     "parsed": {                                            │
│                       "size_width": 24,                                    │
│                       "size_height": 24,                                   │
│                       "size_unit": "inches",                               │
│                       "air_type": "supply",                                │
│                       "cfm": 200                                           │
│                     },                                                     │
│                     "associated_symbol": "diffuser_at_350_400"             │
│                   }                                                        │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Example 4: Handwritten Annotations

```
┌─────────────────────────────────────────────────────────────────────────────┐
│ Visual in PDF:    [messy handwritten "verify in field"]                    │
│                                                                             │
│ Current OCR:      "venly n fled" or similar garbage                        │
│                                                                             │
│ Gemini sees:      - This is a handwritten annotation                       │
│                   - Common construction markup phrase                       │
│                   - Context suggests field verification needed             │
│                   - Therefore: "VERIFY IN FIELD"                           │
│                                                                             │
│ Gemini outputs:   {                                                        │
│                     "content": "VERIFY IN FIELD",                          │
│                     "type": "annotation",                                  │
│                     "style": "handwritten",                                │
│                     "layer": "A-ANNO-NOTE"                                 │
│                   }                                                        │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Text Extraction Implementation

```python
async def extract_text_with_gemini(image: bytes) -> list[TextEntity]:
    """
    Extract all text using Gemini's semantic understanding.
    No OCR preprocessing. Direct visual reading.
    """

    prompt = """
    Read all text visible in this technical drawing.

    For each text element, provide:
    1. content: The exact text content (correct any obvious errors based on context)
    2. position: Approximate [x, y] pixel coordinates
    3. height: Approximate text height in pixels
    4. type: What kind of text is this?
       - dimension: Measurement values
       - room_name: Room labels
       - equipment_tag: Equipment identifiers
       - note: General notes
       - title: Drawing titles
       - schedule: Tabular data
    5. parsed: For technical text, parse into structured data
    6. associated_with: If this text labels something, what does it label?

    Use your understanding of technical drawings to:
    - Correct OCR-like errors (0 vs O, 1 vs l, etc.)
    - Complete partially visible text
    - Understand abbreviations
    - Infer meaning from context

    Return as JSON array.
    """

    response = await gemini.generate(
        prompt=prompt,
        image=image
    )

    text_entities = []
    for item in response["text_items"]:
        entity = TextEntity(
            content=item["content"],
            position=item["position"],
            height=item["height"],
            text_type=item["type"],
            parsed_data=item.get("parsed"),
            associated_element=item.get("associated_with"),
            layer=get_text_layer(item["type"])
        )
        text_entities.append(entity)

    return text_entities
```

---

# 8. Symbol Recognition: Vision-Based Matching

## Current vs. Gemini-First Approach

### Current: Template Matching on Degraded Image

```
Problem Flow:
PDF → Bitonal (degraded) → Template Match → Frequent Failures

Issues:
• Template matching requires near-exact pixel match
• Degraded image doesn't match clean templates
• Rotation handling is brittle
• Scale variations cause failures
• Novel symbols completely fail
```

### Gemini-First: Visual Understanding

```
Improved Flow:
PDF → Gemini Vision → Symbol Understanding → Block Insertion

Advantages:
• Gemini trained on millions of images
• Understands visual similarity, not pixel identity
• Handles rotation, scale, style variations
• Can identify novel symbols by description
• Provides semantic attributes
```

## Symbol Recognition Process

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    GEMINI SYMBOL RECOGNITION                                │
└─────────────────────────────────────────────────────────────────────────────┘

Step 1: Gemini identifies symbol regions
───────────────────────────────────────────────────────────────────────────────
Gemini scans the drawing and identifies:
• "Symbol at (300, 400): appears to be a square supply air diffuser"
• "Symbol at (500, 600): appears to be a duplex electrical outlet"
• "Symbol at (150, 350): appears to be a gate valve"

Step 2: Gemini provides detailed classification
───────────────────────────────────────────────────────────────────────────────
For each symbol:
{
  "position": [300, 400],
  "type": "diffuser",
  "subtype": "supply_square",
  "attributes": {
    "size": "24x24",
    "cfm": 200,
    "connection": "top"
  },
  "rotation": 0,
  "associated_text": ["24x24 SA", "200 CFM", "SD-1"],
  "confidence": 0.95
}

Step 3: Map to AutoCAD blocks
───────────────────────────────────────────────────────────────────────────────
symbol_to_block_mapping = {
    ("diffuser", "supply_square"): "M-DIFF-SQ-SUPPLY",
    ("diffuser", "return_square"): "M-DIFF-SQ-RETURN",
    ("outlet", "duplex"): "E-OUTL-DUPLEX",
    ("valve", "gate"): "P-VALV-GATE",
    ...
}

block_name = symbol_to_block_mapping[(symbol.type, symbol.subtype)]

Step 4: Insert block with attributes
───────────────────────────────────────────────────────────────────────────────
insert_block(
    name="M-DIFF-SQ-SUPPLY",
    position=(300, 400),
    rotation=0,
    attributes={
        "SIZE": "24x24",
        "CFM": "200",
        "TAG": "SD-1"
    }
)
```

## Handling Unknown Symbols

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                  UNKNOWN SYMBOL HANDLING                                    │
└─────────────────────────────────────────────────────────────────────────────┘

When Gemini encounters an unfamiliar symbol:

1. Describe it visually:
   "Square shape with diagonal cross, approximately 0.5" x 0.5""

2. Infer from context:
   "Located on mechanical drawing, near ductwork, likely HVAC-related"

3. Check database by description:
   query_symbols(shape="square", has_diagonal_cross=True, discipline="mechanical")

4. If found → use that block

5. If not found → options:
   a. Create placeholder block with description
   b. Flag for manual review
   c. Search manufacturer catalogs (web search)
   d. Ask user for clarification

6. Learn for future:
   Add to symbol database for next time
```

## Symbol Database Integration

```python
async def recognize_and_insert_symbols(gemini_analysis: dict) -> list[Entity]:
    """
    Recognize symbols using Gemini and insert as blocks.
    """
    entities = []
    unknown_symbols = []

    for symbol in gemini_analysis["elements"].get("symbols", []):
        # Try to find matching block
        block_name = find_block_for_symbol(
            symbol_type=symbol["type"],
            symbol_subtype=symbol.get("subtype"),
            attributes=symbol.get("attributes", {})
        )

        if block_name:
            # Insert known block
            entity = await insert_block(
                block_name=block_name,
                position=scale_point(symbol["position"]),
                rotation=symbol.get("rotation", 0),
                scale=symbol.get("scale", 1.0),
                attributes=symbol.get("attributes", {})
            )
            entities.append(entity)
        else:
            # Handle unknown symbol
            result = await handle_unknown_symbol(symbol)
            if result.success:
                entities.append(result.entity)
            else:
                unknown_symbols.append(symbol)

    # Report unknown symbols for manual review
    if unknown_symbols:
        await flag_for_review(unknown_symbols)

    return entities

def find_block_for_symbol(symbol_type: str, symbol_subtype: str, attributes: dict) -> str:
    """
    Look up the appropriate AutoCAD block name for a symbol.
    """
    # Primary lookup: exact type + subtype match
    key = (symbol_type, symbol_subtype)
    if key in SYMBOL_BLOCK_MAP:
        return SYMBOL_BLOCK_MAP[key]

    # Secondary lookup: type only
    if symbol_type in SYMBOL_TYPE_MAP:
        return SYMBOL_TYPE_MAP[symbol_type]

    # Database query by attributes
    result = query_symbol_database(
        type=symbol_type,
        subtype=symbol_subtype,
        **attributes
    )
    if result:
        return result.block_name

    return None  # Unknown symbol
```

---

# 9. Validation and Self-Correction

## Validation Loop Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                      VALIDATION LOOP                                        │
└─────────────────────────────────────────────────────────────────────────────┘

                    ┌─────────────────────────────┐
                    │  Extraction Complete        │
                    │  (Entities in AutoCAD)      │
                    └──────────────┬──────────────┘
                                   │
                                   ▼
                    ┌─────────────────────────────┐
        ┌──────────│  Gemini Validation           │
        │          │                              │
        │          │  Compare extracted entities  │
        │          │  against original PDF        │
        │          └──────────────┬───────────────┘
        │                         │
        │                         ▼
        │          ┌─────────────────────────────┐
        │          │  Issues Found?              │
        │          └──────────────┬──────────────┘
        │                         │
        │              ┌──────────┴──────────┐
        │              │                     │
        │              ▼                     ▼
        │       ┌───────────┐         ┌───────────┐
        │       │    YES    │         │    NO     │
        │       └─────┬─────┘         └─────┬─────┘
        │             │                     │
        │             ▼                     ▼
        │  ┌──────────────────────┐  ┌──────────────────────┐
        │  │ Generate Corrections │  │ Validation Complete  │
        │  │                      │  │ Output: Clean DWG    │
        │  │ - Add missing        │  └──────────────────────┘
        │  │ - Remove extras      │
        │  │ - Fix positions      │
        │  │ - Correct text       │
        │  │ - Swap symbols       │
        │  └──────────┬───────────┘
        │             │
        │             ▼
        │  ┌──────────────────────┐
        │  │ Apply Corrections    │
        └──│ to AutoCAD           │
           └──────────────────────┘

           (Loop continues until no issues or max iterations)
```

## Validation Prompt

```python
VALIDATION_PROMPT = """
Compare the extracted entities against the original drawing.

ORIGINAL DRAWING:
[PDF image attached]

EXTRACTED ENTITIES:
{entities_json}

CHECK FOR:

1. MISSING ELEMENTS
   - Are there any elements in the original that are not in the extracted list?
   - Look for: lines, arcs, circles, text, symbols, dimensions

2. EXTRA ELEMENTS (False Positives)
   - Are there any extracted elements that don't exist in the original?
   - These should be removed.

3. POSITION ERRORS
   - Are any elements in the wrong location?
   - Check both absolute position and relative alignment.

4. TEXT ERRORS
   - Is any text content wrong?
   - Are there OCR-style errors?
   - Is text in wrong position or wrong size?

5. SYMBOL ERRORS
   - Are any symbols the wrong type?
   - Are symbol attributes incorrect?
   - Are symbols in wrong orientation?

6. CONNECTIVITY ISSUES
   - Are there gaps between elements that should connect?
   - Are there overlaps that shouldn't exist?

For each issue found, provide a correction action:
- ADD: {entity details}
- REMOVE: {entity ID}
- MODIFY: {entity ID, new properties}
- REPLACE: {entity ID, new entity}

Return JSON with:
{
  "validation_status": "approved" | "issues_found",
  "accuracy_estimate": 0-100,
  "issues": [...],
  "corrections": [...]
}
"""
```

## Correction Application

```python
async def apply_corrections(corrections: list[dict]) -> None:
    """
    Apply Gemini's corrections to the AutoCAD drawing.
    """
    for correction in corrections:
        action = correction["action"]

        if action == "ADD":
            # Add missing element
            entity_type = correction["entity_type"]
            if entity_type == "line":
                await draw_line(**correction["properties"])
            elif entity_type == "arc":
                await draw_arc(**correction["properties"])
            elif entity_type == "circle":
                await draw_circle(**correction["properties"])
            elif entity_type == "text":
                await draw_mtext(**correction["properties"])
            elif entity_type == "block":
                await insert_block(**correction["properties"])

        elif action == "REMOVE":
            # Remove false positive
            await erase_entity(correction["entity_id"])

        elif action == "MODIFY":
            # Modify existing entity
            await modify_entity(
                entity_id=correction["entity_id"],
                new_properties=correction["new_properties"]
            )

        elif action == "REPLACE":
            # Replace with different entity
            await erase_entity(correction["entity_id"])
            await create_entity(correction["replacement"])
```

## Validation Metrics

| Metric | Target | Measurement |
|--------|--------|-------------|
| **Entity count accuracy** | ±2% | Extracted / Original |
| **Position accuracy** | ±0.5% of dimension | RMS position error |
| **Text accuracy** | >98% | Correct / Total text |
| **Symbol accuracy** | >95% | Correct / Total symbols |
| **Connectivity** | >99% | Connected / Should connect |
| **Iterations needed** | <3 | Average validation loops |

---

# 10. Implementation Considerations

## System Requirements

### Software Requirements

| Component | Requirement | Purpose |
|-----------|-------------|---------|
| **Python** | 3.9-3.12 | Core runtime |
| **AutoCAD** | 2021-2025 | CAD platform |
| **Raster Design** | 2021-2025 (optional) | VTools for guided rasterization |
| **Gemini API** | `gemini-1.5-pro-vision` | Understanding + orchestration |
| **PostgreSQL** | 15+ with PostGIS, pgvector | Storage |

### Hardware Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| **CPU** | 4 cores | 8+ cores |
| **RAM** | 16 GB | 32 GB |
| **GPU** | - | NVIDIA (for local inference backup) |
| **Network** | Stable internet | Low latency (<100ms) |

### API Costs

| Operation | Approximate Cost | Notes |
|-----------|-----------------|-------|
| Gemini understanding (per page) | $0.02-0.05 | Depends on image size |
| Gemini validation (per iteration) | $0.01-0.02 | Smaller payload |
| Total per page (typical) | $0.05-0.15 | Including validation loops |

## Coordinate System Handling

### PDF to AutoCAD Coordinate Mapping

```
PDF/Image Coordinates:          AutoCAD Coordinates:
─────────────────────          ─────────────────────
Origin: Top-Left (0,0)         Origin: Bottom-Left (0,0)
Y increases downward           Y increases upward

Conversion:
  dwg_x = pdf_x * scale_factor
  dwg_y = (image_height - pdf_y) * scale_factor

Example:
  Image size: 2400 x 1800 pixels
  Scale: 1/4" = 1'-0" (0.25" per foot)
  Pixel density: 300 DPI

  scale_factor = 12" / (0.25" * 300 pixels) = 0.16 inches/pixel

  PDF point (500, 300):
    dwg_x = 500 * 0.16 = 80"
    dwg_y = (1800 - 300) * 0.16 = 240"
    AutoCAD point: (80", 240") = (6'-8", 20'-0")
```

### Calibration Strategy

```python
async def calibrate_coordinates(gemini_analysis: dict, image_size: tuple) -> ScaleFactor:
    """
    Determine coordinate scale factor using multiple methods.
    """
    calibrations = []

    # Method 1: Known dimension
    if "dimensions" in gemini_analysis:
        for dim in gemini_analysis["dimensions"]:
            if dim.get("value") and dim.get("pixel_length"):
                factor = dim["value"] / dim["pixel_length"]
                calibrations.append(("dimension", factor, dim.get("confidence", 0.8)))

    # Method 2: Scale notation
    if "scale" in gemini_analysis:
        scale_text = gemini_analysis["scale"]  # e.g., "1/4\" = 1'-0\""
        factor = parse_scale_notation(scale_text, image_dpi=300)
        calibrations.append(("notation", factor, 0.9))

    # Method 3: Sheet size
    if "sheet_size" in gemini_analysis:
        sheet = gemini_analysis["sheet_size"]  # e.g., "ARCH D"
        sheet_dims = SHEET_SIZES[sheet]  # (36", 24")
        factor_x = sheet_dims[0] / image_size[0]
        factor_y = sheet_dims[1] / image_size[1]
        calibrations.append(("sheet", (factor_x + factor_y) / 2, 0.7))

    # Use highest confidence calibration
    calibrations.sort(key=lambda x: x[2], reverse=True)
    return ScaleFactor(calibrations[0][1])
```

## Error Handling

### Gemini API Errors

```python
async def gemini_with_retry(prompt: str, image: bytes, max_retries: int = 3) -> dict:
    """
    Call Gemini API with retry logic for common errors.
    """
    for attempt in range(max_retries):
        try:
            response = await gemini.generate(prompt=prompt, image=image)
            return response

        except RateLimitError:
            # Wait and retry
            wait_time = 2 ** attempt  # Exponential backoff
            await asyncio.sleep(wait_time)

        except ContentFilterError:
            # Image flagged - try with different prompt
            prompt = sanitize_prompt(prompt)

        except InvalidResponseError:
            # Response wasn't valid JSON - try again with stricter prompt
            prompt = add_json_instructions(prompt)

        except TimeoutError:
            # Request timed out - retry
            continue

    raise GeminiExtractionError("Max retries exceeded")
```

### Extraction Errors

```python
async def safe_extraction(gemini_analysis: dict) -> ExtractionResult:
    """
    Perform extraction with comprehensive error handling.
    """
    result = ExtractionResult()

    for element in gemini_analysis.get("elements", {}).get("all", []):
        try:
            entity = await extract_element(element)
            result.add_success(entity)

        except CoordinateError as e:
            # Coordinate out of bounds or invalid
            result.add_error(element, "coordinate_error", str(e))

        except BlockNotFoundError as e:
            # Symbol block doesn't exist
            result.add_warning(element, "block_not_found", str(e))
            # Insert placeholder instead
            entity = await insert_placeholder(element)
            result.add_success(entity)

        except LayerError as e:
            # Layer doesn't exist - create it
            await create_layer(element.get("layer"))
            entity = await extract_element(element)
            result.add_success(entity)

    return result
```

## Migration from Current System

### Phase 1: Parallel Operation (Weeks 1-2)

```
Both systems run:
┌─────────────────────────────────────────────────────────────────────────────┐
│                                                                             │
│   PDF ─┬──▶ [Current Pipeline] ──▶ DWG_A                                   │
│        │                                                                    │
│        └──▶ [Gemini-First]     ──▶ DWG_B                                   │
│                                                                             │
│   Compare results, measure accuracy, identify gaps                         │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Phase 2: Gradual Shift (Weeks 3-4)

```
Route by drawing type:
┌─────────────────────────────────────────────────────────────────────────────┐
│                                                                             │
│   PDF ──▶ [Classifier]                                                      │
│                │                                                            │
│                ├──▶ Simple/Clean ──▶ [Gemini-First]                        │
│                │                                                            │
│                └──▶ Complex/Noisy ──▶ [Current Pipeline]                   │
│                                                                             │
│   Gradually shift more types to Gemini-First as confidence grows           │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Phase 3: Gemini-First Default (Weeks 5+)

```
Gemini-First is primary:
┌─────────────────────────────────────────────────────────────────────────────┐
│                                                                             │
│   PDF ──▶ [Gemini-First] ──▶ DWG                                           │
│                │                                                            │
│                └──▶ (fallback on failure) ──▶ [Current Pipeline]           │
│                                                                             │
│   Current pipeline becomes fallback only                                   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

# 11. Success Metrics

## Accuracy KPIs

| Metric | Current System | Target (Gemini-First) | Improvement |
|--------|---------------|----------------------|-------------|
| **Line accuracy** | 85% | 97% | +12% |
| **Circle accuracy** | 80% | 98% | +18% |
| **Text accuracy** | 75% | 95% | +20% |
| **Symbol accuracy** | 70% | 92% | +22% |
| **Overall accuracy** | 78% | 95% | +17% |

## Noise Reduction

| Metric | Current System | Target (Gemini-First) | Reduction |
|--------|---------------|----------------------|-----------|
| **False positive lines** | 15% | 2% | -87% |
| **False positive circles** | 12% | 1% | -92% |
| **Broken connections** | 18% | 3% | -83% |
| **Text errors** | 25% | 5% | -80% |

## Efficiency Metrics

| Metric | Current System | Target (Gemini-First) | Change |
|--------|---------------|----------------------|--------|
| **Manual cleanup time** | 20 min/page | 3 min/page | -85% |
| **Processing time** | 10 sec/page | 45 sec/page | +350% (trade-off) |
| **Total time (process + cleanup)** | 30 min/page | 4 min/page | -87% |
| **API cost** | $0.00/page | $0.10/page | +$0.10 |

## Quality Benchmarks

| Benchmark | Pass Criteria | Measurement |
|-----------|--------------|-------------|
| **Entity completeness** | >98% of original entities extracted | Extracted / Original |
| **Position accuracy** | <0.5% deviation | RMS position error |
| **Text completeness** | >95% of text correct | Correct text / Total |
| **Symbol completeness** | >90% correct identification | Correct / Total symbols |
| **Connectivity** | >99% proper connections | Connected / Should connect |
| **Validation iterations** | <3 average | Loops until approved |

---

# 12. Future Enhancements

## Short-Term (1-2 months)

1. **Multi-page handling**: Process entire drawing sets with sheet cross-referencing
2. **Block library expansion**: Add 500+ standard AEC blocks
3. **Caching layer**: Cache Gemini responses for similar regions
4. **Batch processing API**: REST API for automated processing

## Medium-Term (3-6 months)

1. **Learning from corrections**: Train local model on Gemini corrections
2. **Custom fine-tuning**: Fine-tune Gemini on AEC-specific drawings
3. **Real-time processing**: Stream results as extraction progresses
4. **Quality prediction**: Predict quality before processing completes

## Long-Term (6-12 months)

1. **Local inference option**: Run smaller vision model locally for cost reduction
2. **3D inference**: Infer 3D information from 2D drawings
3. **Cross-drawing intelligence**: Understand relationships across sheet sets
4. **Automated design validation**: Check extracted geometry against codes

## Scalability Considerations

### Horizontal Scaling

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        SCALED ARCHITECTURE                                  │
└─────────────────────────────────────────────────────────────────────────────┘

                         ┌───────────────┐
                         │  Load Balancer │
                         └───────┬───────┘
                                 │
          ┌──────────────────────┼──────────────────────┐
          │                      │                      │
          ▼                      ▼                      ▼
    ┌───────────┐          ┌───────────┐          ┌───────────┐
    │ Worker 1  │          │ Worker 2  │          │ Worker 3  │
    │           │          │           │          │           │
    │ Gemini    │          │ Gemini    │          │ Gemini    │
    │ API Key A │          │ API Key B │          │ API Key C │
    └─────┬─────┘          └─────┬─────┘          └─────┬─────┘
          │                      │                      │
          └──────────────────────┼──────────────────────┘
                                 │
                                 ▼
                    ┌────────────────────────┐
                    │  AutoCAD Server Pool   │
                    │  (Multiple instances)  │
                    └────────────────────────┘
```

### Cost Optimization

| Strategy | Impact | Implementation |
|----------|--------|----------------|
| Response caching | -30% API calls | Cache similar region analyses |
| Batch requests | -20% costs | Group multiple regions per call |
| Tiered processing | -40% costs | Use cheaper models for simple elements |
| Local preprocessing | -25% costs | Filter obvious elements locally first |

---

# 13. Summary

## The Core Insight

**The current pipeline's noise problem is caused by destroying information before understanding it.**

Bitonal conversion throws away grayscale, color, and fine detail. OpenCV then operates on this degraded image, producing noisy results. No amount of parameter tuning can recover lost information.

## The Solution

**Put Gemini FIRST. Understand before processing.**

By analyzing the original PDF with Gemini Vision before any preprocessing:
- We preserve 100% of the information
- We gain semantic understanding (what things ARE, not just what pixels exist)
- We can choose optimal extraction strategies per element
- We can validate results against the original
- We eliminate preprocessing noise entirely

## Expected Results

| Metric | Improvement |
|--------|-------------|
| **Accuracy** | +17% overall |
| **Noise reduction** | -87% false positives |
| **Manual cleanup** | -85% time |
| **Text recognition** | +20% accuracy |
| **Symbol recognition** | +22% accuracy |

## Trade-offs

| Cost | Benefit |
|------|---------|
| +$0.10/page API cost | -85% manual cleanup time |
| +35 sec processing time | +17% accuracy |
| API dependency | Adaptive, intelligent processing |
| New implementation effort | Fundamentally better architecture |

## Conclusion

The Gemini-First architecture is not just an improvement—it's a paradigm shift from "process then understand" to "understand then process." This eliminates the root cause of the noise problem rather than trying to mitigate its symptoms.

---

*Document Version: 2.0 - Gemini-First Architecture*
*Date: 2025*
*Status: Proposed for Implementation*
