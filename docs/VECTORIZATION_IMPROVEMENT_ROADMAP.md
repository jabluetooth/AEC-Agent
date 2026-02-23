# Vectorization Improvement Roadmap

> **Purpose**: Implementation roadmap based on audit of `PDF_TO_VECTOR_NEW.md` against current AEC Agent pipeline.
>
> **Goal**: Improve PDF raster-to-vector accuracy from ~85% to 95%+

---

## Executive Summary

The research document `PDF_TO_VECTOR_NEW.md` outlines 7 phases of advanced raster-to-vector techniques. After auditing against our current implementation, this roadmap identifies **12 high-impact improvements** organized into **4 implementation phases**.

### Current vs. Target State

| Capability | Current State | Target State | Gap Level |
|------------|---------------|--------------|-----------|
| Image upscaling | None (raw PDF render) | Real-ESRGAN neural super-resolution | **Critical** |
| Binarization | Single adaptive threshold | Ensemble pixel voting (7 algorithms) | **High** |
| Text separation | Gemini visual estimation | Fletcher-Kasturi + skeleton segmentation | **High** |
| Line tracing | OpenCV LSD/Hough | VTracer + neural junctions | **Medium** |
| Curve fitting | Ellipse fitting (distorts arcs) | Bézier Splatting / DiffVG | **Critical** |
| Post-processing | Basic cleanup | SVGO + Visvalingam-Whyatt | **Medium** |
| OCR | Tesseract (implemented) | ✅ Done | Low |
| Vector storage | PostgreSQL pgvector | ✅ Done | Low |

### Estimated Impact

| Improvement | Accuracy Boost | Implementation Effort |
|-------------|----------------|----------------------|
| Real-ESRGAN upscaling | +5-8% | Medium |
| Ensemble binarization | +3-5% | Low |
| VTracer integration | +5-10% | Medium |
| Bézier Splatting curves | +8-12% | High |
| Text-graphics separation | +3-5% | Medium |
| Post-processing optimization | +2-3% | Low |

**Combined potential**: 85% → 95%+ accuracy

---

## Gap Analysis: Current Implementation vs. Research

### Phase I: Image Preprocessing (Research Doc)

| Technique | Research Recommendation | Current Implementation | Gap |
|-----------|------------------------|----------------------|-----|
| **Super-Resolution** | Real-ESRGAN (L1 + GAN loss) | ❌ None - raw PyMuPDF render | **CRITICAL** |
| **Binarization** | Ensemble pixel voting (7 algos) | ⚠️ Single adaptive threshold | HIGH |
| **Deskewing** | Automated horizontal alignment | ⚠️ Not implemented | MEDIUM |

**Current code** (`pdf_intake.py`):
```python
# Just renders at fixed DPI - no enhancement
pixmap = page.get_pixmap(matrix=matrix, alpha=False)
```

**Impact**: Low-resolution PDFs (72-150 DPI) produce unusable output.

---

### Phase II: Text-Graphics Separation (Research Doc)

| Technique | Research Recommendation | Current Implementation | Gap |
|-----------|------------------------|----------------------|-----|
| **Connected Component Analysis** | Fletcher-Kasturi algorithm | ❌ Gemini visual only | HIGH |
| **Size/Elongation Thresholding** | Histogram-based classification | ❌ None | HIGH |
| **Skeleton Segmentation** | 3-4 distance transform | ❌ None | MEDIUM |
| **Intersection Handling** | Junction detection + severing | ❌ None | MEDIUM |

**Current code** (`gemini_understanding.py`):
```python
# Gemini estimates text positions visually - no algorithmic separation
# Results in ±10-50 pixel error
```

**Impact**: Text characters detected as lines; lines through text cause artifacts.

---

### Phase III: OCR Integration (Research Doc)

| Technique | Research Recommendation | Current Implementation | Gap |
|-----------|------------------------|----------------------|-----|
| **Tesseract LSTM** | v4.0/5.0 with line-level processing | ✅ Implemented | DONE |
| **Coordinate Mapping** | Bounding box extraction | ✅ Implemented | DONE |
| **Invisible Text Layer** | Zero-opacity overlay | ⚠️ Partial (not injected into output) | LOW |
| **Dual-Layer Reconstruction** | Visual + semantic layers | ⚠️ Partial | LOW |

**Current code** (`ocr_text_anchoring.py`):
```python
# Good implementation - just needs invisible layer injection
```

**Impact**: OCR works but doesn't produce searchable PDFs.

---

### Phase IV: Vectorization Algorithms (Research Doc)

| Technique | Research Recommendation | Current Implementation | Gap |
|-----------|------------------------|----------------------|-----|
| **Potrace** | Global Bézier optimization | ❌ Not used | LOW (inferior) |
| **VTracer** | O(n) hierarchical clustering | ❌ Not used | **CRITICAL** |
| **Neural Junctions** | For architectural plans | ❌ Not used | HIGH |
| **Integer Programming** | Enforce 90° constraints | ❌ Not used | MEDIUM |

**Current code** (`opencv_extraction.py`):
```python
# Uses LSD + Hough - good but limited
lsd = cv2.createLineSegmentDetector(cv2.LSD_REFINE_STD)
circles = cv2.HoughCircles(...)  # param2=70 too aggressive
```

**Impact**: Missing complex curves; no color handling; O(n²) on large images.

---

### Phase V: Differentiable Vector Graphics (Research Doc)

| Technique | Research Recommendation | Current Implementation | Gap |
|-----------|------------------------|----------------------|-----|
| **DiffVG** | Gradient-based curve fitting | ❌ Not implemented | HIGH |
| **LIVE** | Layered semantic vectorization | ❌ Not implemented | HIGH |
| **Bézier Splatting** | 150x faster than DiffVG | ❌ Not implemented | **CRITICAL** |

**Current code**: Ellipse fitting only (distorts true arcs)
```python
# opencv_extraction.py - ellipse fitting is inaccurate
ellipse = cv2.fitEllipse(contour)  # Forces circular arcs into ellipses
```

**Impact**: Arcs, curves, and complex shapes are poorly represented.

---

### Phase VI: Post-Processing (Research Doc)

| Technique | Research Recommendation | Current Implementation | Gap |
|-----------|------------------------|----------------------|-----|
| **SVGO** | XML/DOM optimization | ❌ Not used (we output DWG) | N/A |
| **Coordinate Truncation** | 2-3 decimal precision | ✅ Implicit in DWG | DONE |
| **Ramer-Douglas-Peucker** | Line simplification | ⚠️ Not implemented | MEDIUM |
| **Visvalingam-Whyatt** | Area-based decimation | ⚠️ Not implemented | MEDIUM |

**Current code** (`adaptive_extraction.py`):
```python
# Basic cleanup only - no advanced simplification
_remove_short_segments()  # Just removes short lines
```

**Impact**: Output has excessive vertices; could be 50%+ smaller.

---

### Phase VII: Vector Databases (Research Doc)

| Technique | Research Recommendation | Current Implementation | Gap |
|-----------|------------------------|----------------------|-----|
| **ChromaDB** | Local RAG | ❌ Not used | LOW |
| **Pinecone** | Cloud vector search | ❌ Not used | LOW |
| **PostgreSQL pgvector** | Embedded vectors | ✅ Implemented | DONE |
| **Semantic Indexing** | RAG integration | ⚠️ Partial | LOW |

**Current implementation**: PostgreSQL with pgvector extension is implemented and working.

**Impact**: Good foundation; could add ChromaDB for local LLM integration.

---

## Implementation Roadmap

### Phase A: Quick Wins (1-2 weeks)
**Low effort, immediate impact**

#### A.1: Ensemble Binarization
**Effort**: Low | **Impact**: +3-5% accuracy

Replace single adaptive threshold with pixel voting:

```
Implementation:
1. Add binarization.py module
2. Implement 7 algorithms:
   - Otsu (global)
   - Adaptive Gaussian
   - Adaptive Mean
   - Niblack
   - Sauvola
   - Wolf-Jolion
   - Bradley
3. Majority vote per pixel
4. Integrate into pdf_intake.py

Files to modify:
- src/aec_agent/mcp/tools/gemini_first/pdf_intake.py
- NEW: src/aec_agent/mcp/tools/gemini_first/binarization.py

Dependencies: None (OpenCV already available)
```

#### A.2: Deskewing
**Effort**: Low | **Impact**: +1-2% accuracy

```
Implementation:
1. Detect skew angle using Hough lines
2. Rotate image to horizontal
3. Apply before binarization

Code pattern:
def deskew(image):
    edges = cv2.Canny(image, 50, 150)
    lines = cv2.HoughLines(edges, 1, np.pi/180, 200)
    angles = [line[0][1] for line in lines]
    median_angle = np.median(angles)
    rotated = ndimage.rotate(image, median_angle * 180/np.pi)
    return rotated
```

#### A.3: Line Simplification (RDP)
**Effort**: Low | **Impact**: +2% file size reduction

```
Implementation:
1. Add simplification.py module
2. Implement Ramer-Douglas-Peucker
3. Apply to polylines before AutoCAD creation

Dependencies: shapely or custom implementation
```

---

### Phase B: Core Improvements (2-4 weeks)
**Medium effort, significant impact**

#### B.1: VTracer Integration
**Effort**: Medium | **Impact**: +5-10% accuracy

VTracer is the most impactful single improvement:

```
Implementation:
1. pip install vtracer (Rust library with Python bindings)
2. Create vtracer_extraction.py module
3. Replace/augment OpenCV line detection with VTracer
4. Handle color clustering for multi-color drawings

Key advantages:
- O(n) complexity (vs O(n²) for Potrace)
- Native color support
- Better handling of corners and sharp edges
- Hierarchical clustering for complex drawings

Integration points:
- hybrid_extract_all() in adaptive_extraction.py
- New strategy: VTRACER alongside OPENCV

API usage:
import vtracer
svg_str = vtracer.convert_raw_image_to_svg(
    img_bytes,
    colormode='color',  # 'color' or 'binary'
    hierarchical='stacked',  # 'stacked' or 'cutout'
    mode='spline',  # 'spline', 'polygon', or 'none'
    filter_speckle=4,
    color_precision=6,
    layer_difference=16,
    corner_threshold=60,
    length_threshold=4.0,
    max_iterations=10,
    splice_threshold=45,
    path_precision=3
)
```

#### B.2: Real-ESRGAN Super-Resolution
**Effort**: Medium | **Impact**: +5-8% accuracy

Critical for low-resolution PDFs:

```
Implementation:
1. pip install realesrgan
2. Create super_resolution.py module
3. Apply before OpenCV extraction
4. Cache upscaled images to avoid reprocessing

Usage:
from realesrgan import RealESRGAN
model = RealESRGAN('cuda', scale=4)  # 4x upscaling
upscaled = model.predict(image)

Integration:
- Add to pdf_intake.py as optional step
- Enable for images < 200 DPI
- Configurable via HybridExtractionConfig

Considerations:
- GPU required for speed (CPU is 10-50x slower)
- Model weights ~60MB
- Processing time: ~2-5 seconds per image on GPU
```

#### B.3: Fletcher-Kasturi Text Separation
**Effort**: Medium | **Impact**: +3-5% accuracy

Algorithmic text/graphics separation:

```
Implementation:
1. Create text_graphics_separation.py
2. Connected component analysis
3. Size/elongation thresholding
4. Hough transform for text line detection
5. Route text to OCR, graphics to vectorizer

Algorithm:
1. Find connected components
2. Calculate bounding box histogram
3. Threshold: T = E + 3σ (E=avg size, σ=std dev)
4. Elongation filter: width/height ratio
5. Group into text lines using Hough

Benefits:
- Removes text characters from line detection
- Cleaner vectorization output
- Enables skeleton segmentation for intersections
```

---

### Phase C: Advanced Techniques (4-8 weeks)
**High effort, transformational impact**

#### C.1: Neural Junction Detection
**Effort**: High | **Impact**: +5-8% for architectural plans

Specialized for floor plans and schematics:

```
Implementation:
1. Train or use pretrained junction detection model
2. Integrate with floor plan parsing
3. Integer programming for constraint enforcement

Research to implement:
- "Raster-to-Vector: Revisiting Floorplan Transformation" (Liu et al.)
- https://github.com/art-programmer/FloorplanTransformation

Key features:
- Detects wall corners, door endpoints, window frames
- Enforces 90° constraints via integer programming
- ~90% precision/recall on residential floorplans

Dependencies:
- PyTorch
- CVXPY or Gurobi for integer programming
- Pre-trained model (~500MB)
```

#### C.2: Bézier Splatting for Curves
**Effort**: High | **Impact**: +8-12% for curved drawings

The most accurate curve fitting available:

```
Implementation:
1. Port Bézier Splatting from https://github.com/xiliu8006/Bezier_splatting
2. Create bezier_splatting.py module
3. Use for arc/curve extraction

Key advantages:
- 150x faster than DiffVG
- Sub-pixel accuracy
- Escapes local minima via adaptive pruning
- Native Bézier output (perfect for AutoCAD)

Algorithm:
1. Initialize random Bézier curves
2. Render to raster via 2D Gaussian splatting
3. Compute loss vs target image
4. Backpropagate to curve control points
5. Prune low-contribution curves
6. Densify high-variance regions
7. Iterate until convergence

Integration:
- Replace ellipse fitting in opencv_extraction.py
- Use for arc, circle, and complex curve detection
- Output native Bézier splines to AutoCAD
```

#### C.3: LIVE (Layered Image Vectorization)
**Effort**: High | **Impact**: +5-8% for complex drawings

Semantic layer-wise vectorization:

```
Implementation:
1. Port from https://github.com/ma-xu/LIVE
2. Integrate Score Distillation Sampling
3. Coarse-to-fine layer generation

When to use:
- Complex multi-layer drawings
- Colored illustrations
- When VTracer output is too noisy

Note: LIVE is slower than Bézier Splatting but produces
better-organized, layer-separated output.
```

---

### Phase D: Polish & Integration (2-4 weeks)
**Final optimization and integration**

#### D.1: Hybrid Strategy Router
**Effort**: Medium | **Impact**: Overall pipeline optimization

Intelligent routing based on image analysis:

```
Implementation:
1. Analyze input image characteristics
2. Route to optimal extraction strategy

Routing logic:
IF image_complexity < 0.3 AND colors == 1:
    use VTracer (binary mode)
ELIF drawing_type == "floorplan":
    use Neural Junctions + VTracer
ELIF has_complex_curves:
    use Bézier Splatting for curves + VTracer for lines
ELSE:
    use Hybrid OpenCV (current)

Metrics to analyze:
- Edge density (Canny edge count / image area)
- Color count (k-means clustering)
- Structural regularity (Hough line strength)
- Text density (connected component ratio)
```

#### D.2: Visvalingam-Whyatt Decimation
**Effort**: Low | **Impact**: 50%+ vertex reduction

For organic shapes and complex curves:

```
Implementation:
1. Calculate effective area for each vertex
2. Build min-heap priority queue
3. Iteratively remove lowest-area vertices
4. Recalculate adjacent areas
5. Stop at target vertex count or area threshold

Code pattern:
import heapq

def visvalingam_whyatt(points, target_count):
    areas = calculate_triangle_areas(points)
    heap = [(area, i) for i, area in enumerate(areas)]
    heapq.heapify(heap)

    while len(points) > target_count:
        _, idx = heapq.heappop(heap)
        points.pop(idx)
        recalculate_adjacent(points, idx, heap)

    return points
```

#### D.3: Searchable PDF Output (Optional)
**Effort**: Low | **Impact**: Document searchability

Inject invisible text layer:

```
Implementation:
1. Use PyMuPDF to create new PDF
2. Add vector graphics layer
3. Overlay invisible text from OCR
4. Set text opacity to 0

Benefits:
- Searchable output PDFs
- Copy/paste text support
- Accessibility compliance
```

---

## Implementation Priority Matrix

| Improvement | Impact | Effort | Priority | Phase |
|-------------|--------|--------|----------|-------|
| VTracer Integration | High | Medium | **P0** | B.1 |
| Real-ESRGAN Upscaling | High | Medium | **P0** | B.2 |
| Ensemble Binarization | Medium | Low | **P1** | A.1 |
| Bézier Splatting | Very High | High | **P1** | C.2 |
| Fletcher-Kasturi | Medium | Medium | **P2** | B.3 |
| Neural Junctions | High | High | **P2** | C.1 |
| Deskewing | Low | Low | **P3** | A.2 |
| RDP Simplification | Low | Low | **P3** | A.3 |
| Visvalingam-Whyatt | Low | Low | **P3** | D.2 |
| LIVE | Medium | High | **P4** | C.3 |
| Searchable PDF | Low | Low | **P4** | D.3 |

---

## Dependencies & Requirements

### Python Packages to Add

```txt
# Phase A
# (No new dependencies - uses existing OpenCV)

# Phase B
vtracer>=0.6.0          # Rust-based vectorizer
realesrgan>=0.3.0       # Super-resolution
basicsr>=1.4.2          # Real-ESRGAN dependency
gfpgan>=1.3.8           # Face enhancement (optional)

# Phase C
torch>=2.0.0            # For neural models
torchvision>=0.15.0     # Vision utilities
cvxpy>=1.4.0            # Integer programming (junctions)
# OR
gurobipy>=10.0.0        # Commercial solver (faster)

# Phase D
shapely>=2.0.0          # Geometry simplification
```

### Hardware Requirements

| Phase | CPU | GPU | RAM | Disk |
|-------|-----|-----|-----|------|
| A | Any | - | 4GB | - |
| B | 4+ cores | Recommended | 8GB | 1GB (models) |
| C | 8+ cores | Required | 16GB | 2GB (models) |
| D | Any | - | 4GB | - |

### Model Weights to Download

```
# Real-ESRGAN (~60MB)
https://github.com/xinntao/Real-ESRGAN/releases

# Bézier Splatting (if using pretrained)
https://github.com/xiliu8006/Bezier_splatting/releases

# Floor Plan Junction Detection (~500MB)
https://github.com/art-programmer/FloorplanTransformation/releases
```

---

## Success Metrics

### Accuracy Targets

| Metric | Current | Phase A | Phase B | Phase C | Phase D |
|--------|---------|---------|---------|---------|---------|
| Line accuracy | 80% | 83% | 90% | 93% | 95% |
| Circle accuracy | 75% | 78% | 85% | 90% | 92% |
| Arc accuracy | 60% | 63% | 75% | 88% | 90% |
| Text accuracy | 85% | 88% | 92% | 94% | 95% |
| Overall | 75% | 80% | 86% | 91% | 93% |

### Performance Targets

| Metric | Current | Target |
|--------|---------|--------|
| Processing time (A3 @ 300DPI) | 15-30s | <20s |
| Memory usage (peak) | 2GB | <4GB |
| GPU utilization | 0% | 50-80% |
| Output file size | Baseline | -30% |

### Quality Targets

| Metric | Current | Target |
|--------|---------|--------|
| Vertex count per line | High | -50% |
| False positive rate | 15% | <5% |
| Text-as-geometry rate | 10% | <2% |
| Curve distortion (arcs) | 20% | <5% |

---

## Risk Assessment

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| GPU not available | Medium | High | CPU fallback (slower) |
| Real-ESRGAN artifacts | Low | Medium | Quality thresholds |
| VTracer SVG parsing | Low | Low | Custom parser |
| Bézier Splatting complexity | Medium | High | Incremental adoption |
| Neural junction training data | High | Medium | Use pretrained model |
| Performance regression | Medium | Medium | Benchmark testing |

---

## Recommended Implementation Order

```
Week 1-2:   Phase A (Quick Wins)
            ├── A.1 Ensemble Binarization
            ├── A.2 Deskewing
            └── A.3 RDP Simplification

Week 3-4:   Phase B.1 (VTracer)
            └── Integrate VTracer, benchmark against OpenCV

Week 5-6:   Phase B.2 (Real-ESRGAN)
            └── Add super-resolution for low-DPI inputs

Week 7-8:   Phase B.3 (Text Separation)
            └── Fletcher-Kasturi implementation

Week 9-12:  Phase C.2 (Bézier Splatting)
            └── Curve fitting upgrade (highest impact)

Week 13-14: Phase C.1 (Neural Junctions) - Optional
            └── Only if architectural plans are priority

Week 15-16: Phase D (Polish)
            ├── D.1 Hybrid Router
            └── D.2 Visvalingam-Whyatt
```

---

## References

### Research Papers
1. Real-ESRGAN: Training Real-World Blind Super-Resolution (Wang et al., 2021)
2. Raster-to-Vector: Revisiting Floorplan Transformation (Liu et al., 2017)
3. Bézier Splatting for Fast Differentiable Vector Graphics (2025)
4. LIVE: Layered Image Vectorization (Ma et al., 2022)
5. Text/Graphics Separation Revisited (Tombre et al., 2002)

### Code Repositories
- VTracer: https://github.com/visioncortex/vtracer
- Real-ESRGAN: https://github.com/xinntao/Real-ESRGAN
- Bézier Splatting: https://github.com/xiliu8006/Bezier_splatting
- FloorplanTransformation: https://github.com/art-programmer/FloorplanTransformation
- LIVE: https://github.com/ma-xu/LIVE
- DiffVG: https://github.com/BachiLi/diffvg

---

*Document created: 2026-02-23*
*Based on audit of PDF_TO_VECTOR_NEW.md*
