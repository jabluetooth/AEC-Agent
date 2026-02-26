# Best Algorithms Pipeline: PDF to AutoCAD Vectorization

> **Comprehensive guide for engineering drawing vectorization**
> Compiled from academic research, industry tools, and existing codebase analysis

---

## Pipeline Overview

```
PDF → Image → Preprocess → Gemini Analysis → Vector Extraction → Validation → AutoCAD Output
 │      │         │              │                  │               │            │
 ▼      ▼         ▼              ▼                  ▼               ▼            ▼
High   300+    Denoise,      Scale, Type,      Lines, Arcs,    Straighten,   Scaled
DPI    DPI     Binarize,     MText, Layers,   Circles, Text,   Connect,     DWG
       PNG     Deskew        Symbols (RAG)    Symbols          Validate     Output
```

---

## Stage 1: PDF Upload & Rendering

### Goal
Convert PDF pages to high-quality raster images without information loss.

### Best Algorithms

| Algorithm | Library | Performance | Recommendation |
|-----------|---------|-------------|----------------|
| **PyMuPDF (fitz)** | Python | Fastest, 300+ DPI | **PRIMARY** |
| pdf2image (Poppler) | Python | Good quality | Fallback |
| ImageMagick | CLI | Universal | Legacy support |

### Implementation

```python
# Current: pdf_intake.py - render_pdf_high_quality()
DPI = 300  # Minimum for engineering drawings
# 600 DPI for detailed drawings with small text
# 1200 DPI for archival/maximum precision

Output: PNG (lossless) - NEVER JPEG for technical drawings
Color Mode: Auto-detect (grayscale if variance < 0.01)
```

### Enhancements to Consider

1. **Adaptive DPI Selection**
   - Analyze PDF vector content complexity
   - Auto-select 300/600/1200 based on smallest text height
   - Rule: Text must be ≥ 20 pixels high for reliable OCR

2. **Multi-Page Handling**
   - Parallel page rendering
   - Page-type classification (detail vs full sheet)

---

## Stage 2: Image Preprocessing & Enhancement

### Goal
Prepare image for optimal vectorization: clean noise, correct skew, enhance contrast.

### Best Algorithms by Task

#### 2.1 Denoising

| Algorithm | Best For | Library | Speed |
|-----------|----------|---------|-------|
| **Non-Local Means (NLM)** | Preserving edges + details | OpenCV | Medium |
| **DnCNN** | Deep learning (trained) | PyTorch | GPU-fast |
| Bilateral Filter | Edge-preserving smoothing | OpenCV | Fast |
| Median Filter | Salt-and-pepper noise | OpenCV | Fastest |
| Wavelet Denoising | Fine detail preservation | PyWavelets | Medium |

**Recommendation:** Use **NLM** (fastNlMeansDenoising) for engineering drawings - preserves fine lines.

```python
# Current: preprocessing.py - enhance_contrast()
cv2.fastNlMeansDenoising(img, h=10, templateWindowSize=7, searchWindowSize=21)
```

#### 2.2 Deskewing

| Algorithm | Accuracy | Speed | Best For |
|-----------|----------|-------|----------|
| **Hough Transform** | High | Medium | Line-heavy drawings |
| Projection Profile | High | Fast | Text-heavy documents |
| Image Moments | Medium | Fastest | Quick estimation |
| FFT-based | High | Slow | Complex rotations |

**Recommendation:** Use **Hough Transform** as primary, fall back to projection profile.

```python
# Current: preprocessing.py - detect_skew_hough()
# Detect dominant line angles, compute median, rotate image
```

#### 2.3 Binarization

| Algorithm | Strengths | Weaknesses |
|-----------|-----------|------------|
| **Otsu** | Global optimal threshold | Fails on uneven lighting |
| Adaptive Gaussian | Local adaptation | Can create noise |
| **Sauvola** | Document-optimized | Slower |
| Niblack | Strong contrast | Over-sensitive |
| Bradley | O(1) per pixel | Requires tuning |

**Recommendation:** Use **7-method ensemble voting** (current implementation is excellent).

```python
# Current: binarization.py - ensemble_binarize()
# Pixel is foreground if votes > (7 methods × 0.5)
# Methods: Otsu, Adaptive Gaussian, Adaptive Mean, Niblack, Sauvola, Wolf-Jolion, Bradley
```

#### 2.4 Super-Resolution (Low DPI Enhancement)

| Algorithm | Scale | Speed | Quality |
|-----------|-------|-------|---------|
| **Real-ESRGAN** | 4x | GPU: Fast, CPU: Slow | Excellent |
| ESPCN | 2-4x | Fast | Good |
| Bicubic | Any | Instant | Poor |

**Recommendation:** Use **Real-ESRGAN** for PDFs < 200 DPI.

```python
# Current: super_resolution.py - RealESRGANUpscaler
# 4x upscaling, tile-based processing, GPU acceleration
```

### Preprocessing Pipeline Order

```
1. Super-Resolution (if DPI < 200)
2. Grayscale conversion
3. Denoising (NLM)
4. Deskewing (Hough)
5. Contrast enhancement (CLAHE)
6. Binarization (Ensemble 7-method)
7. Border removal
```

---

## Stage 3: Gemini Analysis (Semantic Understanding)

### Goal
Extract semantic information: scale, drawing type, layers, elements, positions.

### 3.1 Scale Detection

| Method | Reliability | Implementation |
|--------|-------------|----------------|
| **Dimension text parsing** | High | Regex + OCR |
| **Scale notation** | High | "1/4" = 1'-0"" parsing |
| Sheet size inference | Medium | Match to ARCH/ANSI/ISO |
| Reference object | Low | Known-size elements |

**Recommendation:** Hierarchical approach - try dimension first, then scale notation, then sheet size.

```python
# Current: coordinate_calibration.py
# Calibration hierarchy:
# 1. calibrate_from_dimension() - Parse "10'-6"" text
# 2. calibrate_from_scale_notation() - Parse "1/8" = 1'-0""
# 3. calibrate_from_sheet_size() - Match ARCH D = 24"×36"
```

### 3.2 Drawing Type Classification

| Type | Indicators | Layer Prefix |
|------|------------|--------------|
| Floor Plan | Walls, doors, rooms | A- |
| Electrical | Outlets, panels, circuits | E- |
| Mechanical | Ducts, equipment, diffusers | M- |
| Plumbing | Pipes, fixtures, drains | P- |
| Fire Alarm | Devices, NAC, SLC | F- |
| Reflected Ceiling | Grid, lights, diffusers | A-CLNG |

```python
# Current: gemini_understanding.py - DrawingType enum
# Gemini classifies from image content
```

### 3.3 Element Extraction Targets

| Element | Detection Method | Output |
|---------|-----------------|--------|
| **MText/Dimensions** | OCR + Gemini | Content, position, height |
| **Lines** | Type classification | Wall, duct, pipe, wire, dimension |
| **Arcs** | Context analysis | Door swing, curved wall |
| **Circles** | Size + context | Column, equipment, symbol |
| **Symbols** | RAG lookup | Block name, attributes |
| **Layers** | Semantic inference | A-WALL, M-HVAC-DUCT |

### 3.4 Symbol Recognition with RAG

**Current Gap:** Symbol detection uses basic classification.

**Recommended Enhancement:**

```
┌─────────────────────────────────────────────────────────────┐
│                    RAG Symbol Pipeline                       │
├─────────────────────────────────────────────────────────────┤
│  1. Detect symbol regions (Gemini/YOLO)                     │
│  2. Extract symbol image patches                            │
│  3. Generate embeddings (CLIP/DINOv2)                       │
│  4. Query vector database (pgvector)                        │
│  5. Retrieve matching CAD blocks                            │
│  6. Return: block_name, insertion_point, rotation, scale    │
└─────────────────────────────────────────────────────────────┘
```

**Vector Database Schema:**
```sql
CREATE TABLE symbol_embeddings (
    id SERIAL PRIMARY KEY,
    block_name TEXT,
    category TEXT,  -- electrical, mechanical, plumbing, fire
    embedding vector(512),  -- CLIP embedding
    preview_image BYTEA,
    metadata JSONB
);
```

**Models for Symbol Embedding:**
- **CLIP** (OpenAI) - Best for general symbols
- **DINOv2** (Meta) - Better for technical/CAD symbols
- **Custom fine-tuned** - Best for domain-specific symbols

---

## Stage 4: Vector Extraction

### Goal
Convert pixels to precise CAD geometry (lines, arcs, circles, polylines).

### 4.1 Line Detection

| Algorithm | Accuracy | Speed | Best For |
|-----------|----------|-------|----------|
| **LSD (Line Segment Detector)** | Highest | Fast | Engineering drawings |
| Hough Transform (HoughLinesP) | Good | Medium | Simple drawings |
| EDLines | High | Fast | Real-time |
| HAWP (Neural) | Highest | Slow | Complex junctions |

**Recommendation:** Use **LSD** as primary, **HAWP** for complex floor plans.

```python
# Current: opencv_extraction.py - OpenCVExtractor.extract_lines()
lsd = cv2.createLineSegmentDetector()
lines = lsd.detect(gray_image)
```

**HAWP for Junction Detection:**
```python
# Current: neural_junction_detection.py
# Detects T, L, X, Y junctions with high accuracy
# HAWPv3 for self-supervised, works on out-of-distribution images
```

### 4.2 Circle Detection

| Algorithm | False Positives | Speed | Notes |
|-----------|-----------------|-------|-------|
| **Hough Circles** | Medium (tuned) | Fast | param2=70 reduces FP |
| Contour fitting | Low | Medium | More reliable |
| RANSAC circle fit | Low | Slow | Best accuracy |

**Recommendation:** Hough with high param2, validate with contour.

```python
# Current: opencv_extraction.py
cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, dp=1, minDist=20,
                 param1=50, param2=70, minRadius=5, maxRadius=500)
```

### 4.3 Arc & Fillet Detection

| Method | Capability | Implementation |
|--------|------------|----------------|
| **Contour-based** | Partial circles, fillets | OpenCV findContours |
| Ellipse fitting | Full ellipse arcs | cv2.fitEllipse |
| Bezier approximation | Smooth curves | VTracer |

```python
# Current: opencv_extraction.py - extract_arcs()
# Detect partial circles from contours
```

### 4.4 Vectorization Algorithms

| Algorithm | Speed | Quality | Token Count |
|-----------|-------|---------|-------------|
| **VTracer** | O(n) - FAST | High | 4.5k-20k |
| Potrace | O(n²) - Slow | High | Medium |
| **Bezier Splatting** | 150x faster than DiffVG | Highest | Compact |
| LIVE | 5h for 2K image | Semantic layers | ~18k |

**Recommendation:** Use **VTracer** for speed, **Bezier Splatting** for quality.

```python
# Current: vtracer_extraction.py - VTracerExtractor
# O(n) vectorization, originally designed for historic blueprint scans
# Output: SVG paths with Bezier curves
```

**Bezier Splatting (NeurIPS 2025):**
```python
# Current: bezier_splatting.py
# 30x faster forward, 150x faster backward than DiffVG
# Produces clean, editable SVG output
```

### 4.5 Text Extraction & Positioning

| Tool | Accuracy | Speed | Integration |
|------|----------|-------|-------------|
| **Tesseract 5** | Good (with training) | Medium | pytesseract |
| **eDOCr2** | 93.75% recall | Medium | Specialized |
| EasyOCR | Good | Fast | GPU-accelerated |
| Google Vision | Excellent | API call | Cloud-based |
| Gemini | Context-aware | API call | Semantic |

**Recommendation:** Tesseract for positions + Gemini for semantic correction.

```python
# Current: ocr_text_anchoring.py
# 1. Tesseract extracts bounding boxes (pixel-perfect)
# 2. Gemini provides semantic text (corrected content)
# 3. Match Gemini text to Tesseract boxes
# 4. Use Tesseract position + Gemini content
```

### 4.6 Hybrid Extraction Strategy

```
┌──────────────────────────────────────────────────────────────┐
│                   HYBRID EXTRACTION                           │
├──────────────────────────────────────────────────────────────┤
│                                                               │
│   Gemini (Semantic)  ──┐                                     │
│   Weight: 0.4          │                                     │
│                        ├──► Weighted Merge ──► Final Entities │
│   OpenCV (Geometric) ──┤                                     │
│   Weight: 0.3          │                                     │
│                        │                                     │
│   YOLO (Optional)  ────┘                                     │
│   Weight: 0.3                                                │
│                                                               │
└──────────────────────────────────────────────────────────────┘
```

```python
# Current: adaptive_extraction.py - hybrid_extract_all()
HybridExtractionConfig:
  gemini_weight: 0.4
  opencv_weight: 0.3
  yolo_weight: 0.3
  min_confidence: 0.5
  merge_strategy: "weighted_average"
```

---

## Stage 5: Validation & Correction

### Goal
Ensure lines are straight, endpoints connected, geometry valid.

### 5.1 Line Straightening

| Condition | Action | Tolerance |
|-----------|--------|-----------|
| Near horizontal (±5°) | Snap to 0° | 5° |
| Near vertical (±5°) | Snap to 90° | 5° |
| Near 45° (±5°) | Snap to 45° | 5° |
| Other angles | Keep original | - |

```python
# Current: gemini_refinement.py
# Aggressive line straightening with 5° tolerance
def straighten_line(angle):
    if abs(angle % 90) < 5:
        return round(angle / 90) * 90
    if abs(angle % 45) < 5:
        return round(angle / 45) * 45
    return angle
```

### 5.2 Endpoint Connection

| Scenario | Action | Tolerance |
|----------|--------|-----------|
| Endpoints within 10px | Connect to midpoint | 10px |
| T-junction | Extend to intersection | 15px |
| Near grid point | Snap to grid | 5px |

```python
# Current: gemini_refinement.py - connect_nearby_endpoints()
RefinementConfig:
  snap_to_grid: True (tolerance 5.0 px)
  connect_endpoints: True (tolerance 10.0 px)
  align_parallel_lines: True (tolerance 5.0°)
```

### 5.3 Relationship Validation (MText ↔ Lines)

**New Recommendation:**

```
┌─────────────────────────────────────────────────────────────┐
│              MTEXT-LINE RELATIONSHIP VALIDATION              │
├─────────────────────────────────────────────────────────────┤
│  1. Find dimension text (e.g., "10'-6"")                    │
│  2. Locate nearby extension lines                           │
│  3. Verify line endpoints align with dimension              │
│  4. If mismatch: adjust line OR flag for review             │
│  5. For leaders: trace from text to geometry                │
└─────────────────────────────────────────────────────────────┘
```

### 5.4 Visual Validation with Gemini

```python
# Current: validation.py
# 1. Render original PDF image
# 2. Render created DWG screenshot
# 3. Send both to Gemini for comparison
# 4. Gemini identifies: missing, extra, misplaced elements
# 5. Auto-correct or flag for manual review
```

---

## Stage 6: AutoCAD Output

### Goal
Create properly scaled DWG with correct layers, linetypes, and text.

### 6.1 Coordinate Conversion

```python
# Pixel to DWG units:
scale_factor = drawing_scale / dpi  # e.g., (1/96) / 300

DWG_x = pixel_x * scale_factor + origin_x
DWG_y = (image_height - pixel_y) * scale_factor + origin_y  # Y-flip
```

### 6.2 Layer Mapping

| Element Type | Layer Name | Color |
|--------------|------------|-------|
| Walls | A-WALL | 6 (Magenta) |
| Doors | A-DOOR | 6 (Magenta) |
| HVAC Ducts | M-HVAC-DUCT | 3 (Green) |
| Plumbing | P-PLUMB-PIPE | 5 (Cyan) |
| Electrical | E-POWER | 1 (Red) |
| Fire Alarm | F-FIRE-ALARM | 4 (Yellow) |
| Dimensions | A-DIMS | 7 (White) |

### 6.3 Linetype Mapping

| Detected Type | AutoCAD Linetype |
|---------------|------------------|
| Continuous | CONTINUOUS |
| Dashed | DASHED |
| Dotted | DOT |
| Center | CENTER |
| Hidden | HIDDEN |

### 6.4 Text Height Standards

```python
MIN_TEXT_HEIGHT = 0.09375  # 3/32" (industry standard minimum)
# Scale appropriately based on drawing scale
```

---

## Recommended Enhancements (Not Yet Implemented)

### Priority 1: RAG Symbol Recognition

```python
# New module: symbol_rag.py
class SymbolRAG:
    def __init__(self, vector_db: AsyncSession):
        self.encoder = CLIPEncoder()  # or DINOv2

    async def identify_symbol(self, image_patch: np.ndarray) -> SymbolMatch:
        embedding = self.encoder.encode(image_patch)
        matches = await self.vector_db.query(
            "SELECT block_name, similarity FROM symbol_embeddings "
            "ORDER BY embedding <-> $1 LIMIT 5",
            embedding
        )
        return SymbolMatch(
            block_name=matches[0].block_name,
            confidence=matches[0].similarity,
            alternatives=[m.block_name for m in matches[1:]]
        )
```

### Priority 2: Deep Learning Text Detection

Replace/augment Tesseract with transformer-based OCR:

```python
# eDOCr2 integration for 93.75% text recall
# or fine-tuned TrOCR for engineering documents
```

### Priority 3: Drawing2CAD Integration

Sequence-to-sequence learning for direct CAD command generation:

```
Image → Encoder → Latent Vector → Dual Decoder → CAD Commands
```

### Priority 4: Multi-Scale Processing

```python
# Process at multiple scales and merge results
scales = [0.5, 1.0, 2.0]
all_entities = []
for scale in scales:
    scaled_img = cv2.resize(img, scale=scale)
    entities = extract_entities(scaled_img)
    all_entities.extend(rescale_entities(entities, 1/scale))
merged = deduplicate_and_merge(all_entities)
```

---

## Algorithm Comparison Summary

### Preprocessing

| Task | Best Algorithm | Fallback |
|------|---------------|----------|
| PDF Render | PyMuPDF 300 DPI | pdf2image |
| Denoise | NLM | Bilateral |
| Deskew | Hough Transform | Projection Profile |
| Binarize | 7-Method Ensemble | Sauvola |
| Upscale | Real-ESRGAN 4x | Bicubic |

### Extraction

| Task | Best Algorithm | Fallback |
|------|---------------|----------|
| Lines | LSD | HoughLinesP |
| Circles | Hough (param2=70) | Contour fit |
| Arcs | Contour-based | Ellipse fit |
| Junctions | HAWP v3 | Manual rules |
| Vectorize | VTracer O(n) | Bezier Splatting |
| OCR | Tesseract + Gemini | EasyOCR |
| Symbols | RAG (CLIP/pgvector) | Gemini classify |

### Validation

| Task | Best Algorithm | Tolerance |
|------|---------------|-----------|
| Straighten | Angle snapping | 5° |
| Connect | Endpoint merge | 10px |
| Grid snap | Round to grid | 5px |
| Visual QA | Gemini comparison | - |

---

## References

### Academic Papers
- [Deep Vectorization of Technical Drawings](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123580579.pdf) (ECCV 2020)
- [HAWP: Holistically-Attracted Wireframe Parsing](https://github.com/cherubicXN/hawp) (CVPR 2020, TPAMI 2023)
- [Bezier Splatting](https://arxiv.org/abs/2503.16424) (NeurIPS 2025)
- [LIVE: Layer-wise Image Vectorization](https://ma-xu.github.io/LIVE/) (CVPR 2022 Oral)
- [Drawing2CAD](https://arxiv.org/pdf/2508.18733) (Sequence-to-sequence)
- [eDOCr2: OCR for Mechanical Drawings](https://www.mdpi.com/2075-1702/13/3/254)

### Tools & Libraries
- [VTracer](https://github.com/visioncortex/vtracer) - O(n) vectorization
- [Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN) - Super-resolution
- [Tesseract 5](https://github.com/tesseract-ocr/tesseract) - OCR
- [PyMuPDF](https://pymupdf.readthedocs.io/) - PDF rendering

### Commercial Solutions
- [Print2CAD 2024 AI](https://rvtplugins.com/posts/backtocad-technologies-llc/print2cad-2024-ai---pdf-to-cad-batch-converter/7735133729366296627)
- [CADirect 2026 AI](https://solutions.backtocad.com/features/caddirect)

---

## Current Implementation Status

| Stage | Module | Status |
|-------|--------|--------|
| 1. PDF Render | `pdf_intake.py` | Complete |
| 2. Preprocess | `preprocessing.py`, `binarization.py` | Complete |
| 2b. Super-Res | `super_resolution.py` | Complete |
| 3. Gemini Analysis | `gemini_understanding.py` | Complete |
| 3b. Calibration | `coordinate_calibration.py` | Complete |
| 4. Extraction | `adaptive_extraction.py`, `opencv_extraction.py` | Complete |
| 4b. VTracer | `vtracer_extraction.py` | Complete |
| 4c. Neural | `neural_junction_detection.py`, `bezier_splatting.py` | Complete |
| 5. Validation | `validation.py`, `gemini_refinement.py` | Complete |
| 6. AutoCAD | `autocad_creation.py` | Complete |
| **RAG Symbols** | - | **Not Implemented** |
| **Drawing2CAD** | - | **Not Implemented** |

---

*Last Updated: 2026-02-26*
