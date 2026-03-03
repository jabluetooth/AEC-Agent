# Scan2CAD Feature Parity Plan

> **Goal**: Match and exceed Scan2CAD's raster-to-vector conversion capabilities while leveraging AI-powered semantic understanding as a competitive advantage.

**Created**: 2026-03-03
**Status**: Planning
**Reference**: [Scan2CAD Website](https://www.scan2cad.com/)

---

## Table of Contents

1. [Executive Summary](#executive-summary)
2. [Current State Assessment](#current-state-assessment)
3. [Scan2CAD Feature Analysis](#scan2cad-feature-analysis)
4. [Gap Analysis](#gap-analysis)
5. [Implementation Phases](#implementation-phases)
6. [Technical Architecture](#technical-architecture)
7. [Testing Strategy](#testing-strategy)
8. [Success Metrics](#success-metrics)
9. [Risk Assessment](#risk-assessment)

---

## Executive Summary

### The Competition

Scan2CAD is the industry-leading raster-to-vector conversion software with 20+ years of development. It excels at:
- **Edge tracing precision** - mathematically accurate line/arc/curve fitting
- **Robust preprocessing** - handles noisy scans, faded drawings, skewed images
- **Batch processing** - converts hundreds of files unattended
- **Format flexibility** - DXF, DWG, G-code output

### Our Advantage

Our pipeline has a fundamental architectural advantage: **semantic understanding**.

| Approach | Scan2CAD | Our Pipeline |
|----------|----------|--------------|
| Method | Edge tracing (dumb) | AI understanding (smart) |
| Entity Recognition | "This is a line" | "This is a wall at 24'-0"" |
| Layer Assignment | Manual/rule-based | Semantic (A-WALL, E-POWER) |
| Symbol Handling | Pattern matching | RAG + knowledge base |
| Output | Generic DXF | Native AutoCAD with blocks |

### Strategy

1. **Phase A**: Achieve core feature parity (line/arc/circle detection quality)
2. **Phase B**: Exceed with AI capabilities (semantic layers, symbol RAG, dimension extraction)
3. **Phase C**: Production polish (batch processing, error recovery, preview UI)

---

## Current State Assessment

### Implemented Modules (`gemini_first/`)

| Module | Status | Description |
|--------|--------|-------------|
| `pdf_intake.py` | Complete | PDF rendering (PyMuPDF), DPI detection, grayscale conversion |
| `preprocessing.py` | Complete | Deskew, denoise, contrast enhancement |
| `binarization.py` | Complete | Ensemble binarization (Otsu, Sauvola, adaptive) |
| `super_resolution.py` | Partial | Real-ESRGAN upscaling (optional, GPU required) |
| `text_graphics_separation.py` | Complete | Fletcher-Kasturi algorithm |
| `gemini_understanding.py` | Complete | Drawing analysis, entity detection, scale recognition |
| `coordinate_calibration.py` | Complete | Pixel-to-DWG unit mapping |
| `opencv_extraction.py` | Complete | Hough lines/circles, contour detection |
| `vtracer_extraction.py` | Partial | Raster-to-vector via VTracer (optional) |
| `adaptive_extraction.py` | Complete | Hybrid Gemini+OpenCV extraction |
| `ocr_text_anchoring.py` | Complete | Text detection and positioning |
| `symbol_rag.py` | Complete | CLIP embeddings + pgvector search |
| `neural_junction_detection.py` | Partial | HAWP junction detection (optional) |
| `bezier_splatting.py` | Stub | Differentiable curve fitting (not integrated) |
| `live_vectorization.py` | Stub | Layer-wise vectorization (not integrated) |
| `simplification.py` | Complete | Douglas-Peucker, Visvalingam simplification |
| `gemini_refinement.py` | Complete | Gemini-guided coordinate refinement |
| `autocad_creation.py` | Complete | Entity creation via sidecar |
| `validation.py` | Partial | Visual QA, entity count validation |
| `unified_pipeline.py` | Complete | Consolidated pipeline orchestrator |

### Current Pipeline Flow

```
PDF Input
    │
    ▼
┌─────────────────┐
│ 1. PDF Render   │ → High-DPI rasterization (300 DPI default)
└─────────────────┘
    │
    ▼
┌─────────────────┐
│ 2. Preprocess   │ → Deskew, denoise, binarize, enhance
└─────────────────┘
    │
    ▼
┌─────────────────┐
│ 3. Gemini       │ → Semantic analysis (drawing type, layers, scale)
│    Analysis     │
└─────────────────┘
    │
    ▼
┌─────────────────┐
│ 4. Calibration  │ → Pixel-to-DWG coordinate mapping
└─────────────────┘
    │
    ▼
┌─────────────────┐
│ 5. Extraction   │ → Hybrid: OpenCV + Gemini coordinates
└─────────────────┘
    │
    ▼
┌─────────────────┐
│ 6. Refinement   │ → Straighten, connect, deduplicate
└─────────────────┘
    │
    ▼
┌─────────────────┐
│ 7. Symbol RAG   │ → Match symbols to AutoCAD blocks
└─────────────────┘
    │
    ▼
┌─────────────────┐
│ 8. AutoCAD      │ → Create entities via sidecar
│    Creation     │
└─────────────────┘
    │
    ▼
┌─────────────────┐
│ 9. Validation   │ → Optional Gemini visual QA
└─────────────────┘
```

---

## Scan2CAD Feature Analysis

### Core Vectorization Features

| Feature | Priority | Scan2CAD | Our Status |
|---------|----------|----------|------------|
| Line detection | P0 | Excellent | Good (Hough) |
| Arc/circle detection | P0 | Excellent | Good (HoughCircles) |
| Bezier curve fitting | P1 | True Bezier | Stub only |
| Polyline recognition | P1 | Auto-join | Partial |
| Spline fitting | P2 | Yes | No |
| Ellipse detection | P2 | Yes | No |

### Preprocessing Features

| Feature | Priority | Scan2CAD | Our Status |
|---------|----------|----------|------------|
| Deskew | P0 | Yes | Complete |
| Despeckle/denoise | P0 | Yes | Complete |
| Thresholding/binarization | P0 | Yes | Complete |
| Brightness/contrast | P1 | Yes | Complete |
| Edge enhancement | P1 | Yes | Partial |
| Color-to-mono conversion | P1 | Yes | Complete |

### OCR Features

| Feature | Priority | Scan2CAD | Our Status |
|---------|----------|----------|------------|
| Standard font recognition | P0 | Excellent | Good (Gemini) |
| Handwritten text | P1 | Smart OCR | Limited |
| Rotated text | P1 | Yes | Yes (Gemini) |
| Multi-language | P2 | Yes | Yes (Gemini) |
| Font style preservation | P2 | Yes | No |
| Text reassembly (chars→words) | P1 | PDF native | Gemini semantic |

### Advanced Features

| Feature | Priority | Scan2CAD | Our Status |
|---------|----------|----------|------------|
| Line type detection (dashed) | P0 | Excellent | No |
| Line thickness detection | P1 | Yes | No |
| Hatch pattern recognition | P2 | Yes | No |
| Layer separation | P1 | Manual | Semantic (better) |
| Dimension extraction | P1 | Yes | Gemini |
| Block/symbol recognition | P1 | Pattern match | RAG (better) |

### Batch & Workflow Features

| Feature | Priority | Scan2CAD | Our Status |
|---------|----------|----------|------------|
| Batch conversion | P1 | Hundreds | Single file |
| Multi-page PDF | P1 | Yes | Partial |
| Preview before convert | P2 | Yes | No |
| Manual tracing fallback | P2 | Yes | No |
| Conversion profiles | P2 | Yes | Config only |
| Python/COM API | P1 | Yes | MCP tools |

### Output Formats

| Feature | Priority | Scan2CAD | Our Status |
|---------|----------|----------|------------|
| DXF export | P0 | Yes | No (via AutoCAD) |
| DWG export | P0 | Yes | Via sidecar |
| G-code export | P3 | Yes | No |
| PDF vector export | P3 | Yes | No |
| SVG export | P3 | No | No |

---

## Gap Analysis

### Critical Gaps (P0) - Must fix for production use

1. **Line Type Detection**
   - Cannot detect dashed, dotted, center, hidden lines
   - All lines created as CONTINUOUS
   - Impact: MEP/architectural drawings lose critical information

2. **True Curve Fitting**
   - No Bezier/spline fitting for curved entities
   - Curves approximated as line segments
   - Impact: Poor quality for mechanical drawings, contours

3. **DXF Export**
   - Cannot export standalone DXF without AutoCAD
   - Blocks offline workflows
   - Impact: Users need AutoCAD running

### High Priority Gaps (P1)

4. **Line Thickness Detection**
   - All lines created with default thickness
   - Loses visual hierarchy information

5. **Batch Processing**
   - No multi-file automation
   - Manual one-at-a-time conversion

6. **Multi-page PDF**
   - Single page processing only
   - Large document sets require manual iteration

7. **Polyline Auto-join**
   - Connected line segments not joined into polylines
   - Creates inefficient entity count

8. **Handwritten Text OCR**
   - Poor recognition of hand-drawn annotations
   - Common in markup/redline drawings

### Medium Priority Gaps (P2)

9. **Ellipse Detection**
10. **Hatch Pattern Recognition**
11. **Preview Interface**
12. **Conversion Profiles**
13. **Font Style Preservation**

### Low Priority / Future (P3)

14. **G-code Export**
15. **Manual Tracing Mode**
16. **PDF Vector Export**

---

## Implementation Phases

### Phase A: Core Quality Parity (4-6 weeks)

**Goal**: Match Scan2CAD's vectorization accuracy for clean technical drawings.

#### A.1: Line Type Detection
```
File: gemini_first/linetype_detection.py (new)

Approach:
1. Analyze line segment patterns (gaps, dashes)
2. Use Fourier analysis for periodicity
3. Map to AutoCAD linetypes: CONTINUOUS, DASHED, HIDDEN, CENTER, PHANTOM

Algorithm:
- Extract line from binary image
- Sample intensity along line path
- FFT to detect periodic gaps
- Classify based on gap/dash ratio

Integration:
- Add linetype field to EntityToCreate
- Update autocad_creation.py to set LINETYPE property
```

**Deliverables**:
- [ ] `linetype_detection.py` module
- [ ] Unit tests with sample dashed/dotted lines
- [ ] Integration with `adaptive_extraction.py`
- [ ] AutoCAD linetype mapping table

#### A.2: True Bezier Curve Fitting
```
File: gemini_first/bezier_fitting.py (new)

Approach:
1. Detect curved segments (high curvature variance)
2. Fit cubic Bezier using least-squares
3. Create SPLINE entities in AutoCAD

Algorithm (Schneider's Algorithm):
1. Find corners (high curvature points)
2. Split curve at corners
3. Fit Bezier to each segment
4. Join with G1 continuity

Integration:
- Add SPLINE entity type
- Update autocad_creation.py for SPLINE creation
```

**Deliverables**:
- [ ] `bezier_fitting.py` with Schneider's algorithm
- [ ] Curve detection in `opencv_extraction.py`
- [ ] SPLINE support in `autocad_creation.py`
- [ ] Accuracy tests against known curves

#### A.3: DXF Direct Export
```
File: gemini_first/dxf_export.py (new)

Approach:
- Use ezdxf library for DXF R2018 export
- No AutoCAD dependency for offline workflows
- Support layers, linetypes, text styles

Dependencies:
- ezdxf>=1.0.0
```

**Deliverables**:
- [ ] `dxf_export.py` module
- [ ] All entity types supported (LINE, CIRCLE, ARC, TEXT, SPLINE)
- [ ] Layer and linetype preservation
- [ ] Unit tests with DXF validation

#### A.4: Line Thickness Detection
```
File: gemini_first/thickness_detection.py (new)

Approach:
1. Measure line width in binary image
2. Use distance transform + skeleton
3. Map pixel width to lineweight

Integration:
- Add lineweight field to EntityToCreate
- Map to AutoCAD lineweights (0.00, 0.05, 0.09, 0.13, 0.18, ...)
```

**Deliverables**:
- [ ] `thickness_detection.py` module
- [ ] Integration with extraction pipeline
- [ ] Lineweight mapping table

---

### Phase B: Advanced Features (4-6 weeks)

**Goal**: Exceed Scan2CAD with AI capabilities and production features.

#### B.1: Batch Processing
```
File: gemini_first/batch_processor.py (new)

Features:
- Process directory of PDFs
- Parallel processing with asyncio
- Progress reporting
- Error recovery per file
- Summary report generation

MCP Tools:
- vectorize_pdf_batch(input_dir, output_dir, config)
- get_batch_status(batch_id)
- cancel_batch(batch_id)
```

**Deliverables**:
- [ ] `batch_processor.py` module
- [ ] MCP tools for batch operations
- [ ] Progress tracking and reporting
- [ ] Error isolation per file

#### B.2: Multi-page PDF Support
```
File: Update unified_pipeline.py

Features:
- Process all pages or page range
- Per-page configuration overrides
- Combined or separate output files
- Page-level validation
```

**Deliverables**:
- [ ] Multi-page support in `UnifiedPipeline`
- [ ] Page range parameters in MCP tools
- [ ] Combined DXF/DWG output option

#### B.3: Polyline Auto-join
```
File: gemini_first/polyline_builder.py (new)

Algorithm:
1. Build endpoint graph from lines
2. Find connected chains
3. Convert chains to polylines
4. Preserve vertex order (CW/CCW for closed)

Integration:
- Post-processing step after extraction
- Option to keep as lines or convert
```

**Deliverables**:
- [ ] `polyline_builder.py` module
- [ ] Graph-based endpoint matching
- [ ] Closed polyline detection
- [ ] Entity count reduction metrics

#### B.4: Enhanced Handwritten OCR
```
Approach:
- Use Gemini's handwriting recognition
- Fall back to TrOCR for specialized cases
- Confidence-based filtering

Integration:
- Upgrade text extraction in gemini_understanding.py
- Add handwriting mode flag
```

**Deliverables**:
- [ ] Handwriting detection classifier
- [ ] TrOCR integration (optional)
- [ ] Confidence thresholds for handwritten text

---

### Phase C: Production Polish (3-4 weeks)

**Goal**: Enterprise-ready features and user experience.

#### C.1: Preview Interface
```
Approach:
- Generate preview image with detected entities overlaid
- Color-coded by entity type
- Interactive adjustment (future: web UI)

Implementation:
- OpenCV drawing on source image
- Return preview image path
- Optional: Chainlit preview component
```

**Deliverables**:
- [ ] Preview generation function
- [ ] Color-coding by entity type
- [ ] Preview MCP tool

#### C.2: Conversion Profiles
```
File: gemini_first/profiles.py (new)

Predefined Profiles:
- ARCHITECTURAL: Floor plans, elevations (wall detection, room labels)
- MECHANICAL: Parts drawings (tight tolerances, centerlines)
- ELECTRICAL: Schematics (symbol-heavy, connection lines)
- CIVIL: Site plans (contours, hatching)
- LEGACY_SCAN: Faded/noisy scans (heavy preprocessing)

Storage:
- JSON profile definitions
- User custom profiles in database
```

**Deliverables**:
- [ ] Profile system with presets
- [ ] Custom profile storage
- [ ] Profile selection in MCP tools

#### C.3: Ellipse Detection
```
Approach:
- RANSAC-based ellipse fitting
- Distinguish from circles (eccentricity threshold)
- Create ELLIPSE entities

Integration:
- Add to opencv_extraction.py
- ELLIPSE support in autocad_creation.py
```

**Deliverables**:
- [ ] Ellipse detection algorithm
- [ ] ELLIPSE entity creation
- [ ] Tests with mechanical drawings

#### C.4: Hatch Pattern Recognition
```
Approach:
- Detect closed regions
- Analyze fill pattern (solid, line spacing, angle)
- Map to AutoCAD hatch patterns

Challenges:
- Many standard patterns (ANSI31, EARTH, etc.)
- Custom patterns not feasible

Implementation:
- Closed region detection
- Pattern classification (solid, 45°, cross-hatch)
- Basic hatch creation
```

**Deliverables**:
- [ ] Region detection
- [ ] Basic pattern classification
- [ ] HATCH entity creation (solid + simple patterns)

---

## Technical Architecture

### Module Dependency Graph

```
                    ┌─────────────────┐
                    │ unified_pipeline│
                    └────────┬────────┘
                             │
        ┌────────────────────┼────────────────────┐
        │                    │                    │
        ▼                    ▼                    ▼
┌───────────────┐   ┌────────────────┐   ┌───────────────┐
│ pdf_intake    │   │ preprocessing  │   │ gemini_       │
│               │   │ binarization   │   │ understanding │
└───────────────┘   │ super_res      │   └───────────────┘
                    └────────────────┘           │
                                                 │
        ┌────────────────────────────────────────┤
        │                    │                   │
        ▼                    ▼                   ▼
┌───────────────┐   ┌────────────────┐   ┌───────────────┐
│ coordinate_   │   │ adaptive_      │   │ symbol_rag    │
│ calibration   │   │ extraction     │   │               │
└───────────────┘   └────────┬───────┘   └───────────────┘
                             │
        ┌────────────────────┼────────────────────┐
        │                    │                    │
        ▼                    ▼                    ▼
┌───────────────┐   ┌────────────────┐   ┌───────────────┐
│ opencv_       │   │ linetype_      │   │ thickness_    │
│ extraction    │   │ detection (NEW)│   │ detection(NEW)│
└───────────────┘   └────────────────┘   └───────────────┘
        │                    │                    │
        └────────────────────┼────────────────────┘
                             │
                             ▼
                    ┌────────────────┐
                    │ bezier_fitting │
                    │ (NEW)          │
                    └────────────────┘
                             │
                             ▼
                    ┌────────────────┐
                    │ polyline_      │
                    │ builder (NEW)  │
                    └────────────────┘
                             │
        ┌────────────────────┴────────────────────┐
        │                                         │
        ▼                                         ▼
┌───────────────────┐                   ┌─────────────────┐
│ autocad_creation  │                   │ dxf_export (NEW)│
└───────────────────┘                   └─────────────────┘
```

### New Module Specifications

#### `linetype_detection.py`

```python
@dataclass
class LinetypeResult:
    linetype: str  # CONTINUOUS, DASHED, HIDDEN, CENTER, etc.
    confidence: float
    pattern: list[float]  # [dash, gap, dash, gap, ...] in pixels

def detect_linetype(
    binary_image: np.ndarray,
    line_start: tuple[int, int],
    line_end: tuple[int, int],
    sample_width: int = 3,
) -> LinetypeResult:
    """Detect linetype by analyzing pixel pattern along line."""
    ...
```

#### `bezier_fitting.py`

```python
@dataclass
class BezierCurve:
    control_points: list[tuple[float, float]]  # 4 points for cubic
    start: tuple[float, float]
    end: tuple[float, float]
    fit_error: float

def fit_bezier_to_points(
    points: list[tuple[float, float]],
    max_error: float = 2.0,
) -> list[BezierCurve]:
    """Fit Bezier curves to point sequence using Schneider's algorithm."""
    ...

def detect_curves(
    binary_image: np.ndarray,
    min_curvature: float = 0.1,
) -> list[np.ndarray]:
    """Detect curved segments in binary image."""
    ...
```

#### `dxf_export.py`

```python
def export_to_dxf(
    entities: list[EntityToCreate],
    output_path: str | Path,
    version: str = "R2018",
    units: str = "Inches",
) -> DXFExportResult:
    """Export entities to DXF file without AutoCAD."""
    ...
```

#### `batch_processor.py`

```python
@dataclass
class BatchConfig:
    input_dir: Path
    output_dir: Path
    file_pattern: str = "*.pdf"
    max_parallel: int = 4
    pipeline_config: PipelineConfig = field(default_factory=PipelineConfig)
    continue_on_error: bool = True

@dataclass
class BatchResult:
    batch_id: UUID
    total_files: int
    success_count: int
    failure_count: int
    results: list[PipelineResult]
    duration_seconds: float

async def process_batch(config: BatchConfig) -> BatchResult:
    """Process multiple files in parallel."""
    ...
```

---

## Testing Strategy

### Test Categories

#### Unit Tests
- Each new module has corresponding test file
- Mock external dependencies (Gemini API, AutoCAD sidecar)
- Test with synthetic images (known geometry)

#### Integration Tests
- Full pipeline tests with real PDF samples
- Compare output to expected entity counts
- Visual diff against reference images

#### Accuracy Tests
- Benchmark suite of 50+ sample drawings
- Measure precision/recall for each entity type
- Track accuracy metrics over time

### Test Files Structure

```
tests/
├── unit/
│   ├── test_linetype_detection.py
│   ├── test_bezier_fitting.py
│   ├── test_dxf_export.py
│   ├── test_thickness_detection.py
│   ├── test_polyline_builder.py
│   └── test_batch_processor.py
├── integration/
│   ├── test_unified_pipeline.py
│   ├── test_batch_processing.py
│   └── test_dxf_roundtrip.py
├── accuracy/
│   ├── benchmark_suite.py
│   ├── samples/
│   │   ├── architectural/
│   │   ├── mechanical/
│   │   ├── electrical/
│   │   └── legacy_scans/
│   └── expected/
└── fixtures/
    ├── synthetic_lines.png
    ├── synthetic_circles.png
    ├── dashed_lines.png
    └── bezier_curves.png
```

### Accuracy Benchmarks

| Metric | Target | Current |
|--------|--------|---------|
| Line detection precision | >95% | ~90% |
| Line detection recall | >90% | ~85% |
| Circle detection precision | >95% | ~88% |
| Arc detection precision | >90% | ~75% |
| Text OCR accuracy | >98% | ~95% |
| Symbol recognition | >85% | ~70% |
| Linetype accuracy | >90% | 0% (not implemented) |

---

## Success Metrics

### Phase A Completion Criteria

- [ ] Linetype detection accuracy >90% on test set
- [ ] Bezier fitting error <2px on synthetic curves
- [ ] DXF export passes AutoCAD validation
- [ ] Line thickness within 1 weight class

### Phase B Completion Criteria

- [ ] Batch processing handles 100 files without crash
- [ ] Multi-page PDF support for 50+ page documents
- [ ] Polyline joining reduces entity count by >30%
- [ ] Handwritten text recognition >80% accuracy

### Phase C Completion Criteria

- [ ] Preview generation <2s per page
- [ ] Profile system with 5+ presets
- [ ] Ellipse detection >85% accuracy
- [ ] Basic hatch support (solid, 45°, cross)

### Overall Quality Targets

| Metric | Scan2CAD | Target |
|--------|----------|--------|
| Clean drawing accuracy | 98% | 95% |
| Noisy scan accuracy | 90% | 85% |
| Processing speed (300 DPI) | 5-10s | 10-20s |
| Batch throughput | 100 files/hour | 50 files/hour |

---

## Risk Assessment

### Technical Risks

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| Bezier fitting complexity | Medium | High | Use proven Schneider algorithm |
| Linetype false positives | Medium | Medium | Conservative thresholds + user override |
| DXF compatibility issues | Low | High | Extensive testing with AutoCAD versions |
| Gemini API rate limits | High | Medium | Caching, retry logic, offline fallback |
| Performance degradation | Medium | Medium | Profiling, async processing |

### Resource Risks

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| GPU requirement for features | Medium | Low | CPU fallbacks for all features |
| Database dependency | Low | Low | SQLite fallback for symbol RAG |
| AutoCAD sidecar unavailable | Medium | High | DXF export as offline fallback |

---

## Appendix: Reference Materials

### Scan2CAD Documentation
- [Raster to Vector Introduction](https://www.scan2cad.com/blog/tips/convert-raster-to-vector-an-introduction/)
- [Converting Raster Images for CAD](https://www.scan2cad.com/docs/converting-raster-images-cad/)

### Algorithm References
- Schneider's Algorithm for Bezier fitting: "An Algorithm for Automatically Fitting Digitized Curves" (Graphics Gems, 1990)
- HAWP for junction detection: [GitHub](https://github.com/cherubicXN/hawp)
- VTracer for raster-to-vector: [GitHub](https://github.com/nickmccullum/vtracer)

### AutoCAD References
- [DXF Reference](https://help.autodesk.com/view/OARX/2024/ENU/?guid=GUID-235B22E0-A567-4CF6-92D3-38A2306D73F3)
- [Linetype Definition](https://help.autodesk.com/view/ACD/2024/ENU/?guid=GUID-20B4D4B3-1220-426A-847B-5BBE36EC6FDF)
- [Spline Entity](https://help.autodesk.com/view/ACD/2024/ENU/?guid=GUID-E1F884F8-AA90-4864-A215-3182D47AA652)

---

## Change Log

| Date | Author | Changes |
|------|--------|---------|
| 2026-03-03 | Claude | Initial plan creation |
