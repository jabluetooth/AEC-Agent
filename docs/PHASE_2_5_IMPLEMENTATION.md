# Phase 2.5: Semantic AEC Vectorization Pipeline - Implementation Reference

**Completed:** February 2026
**Status:** Complete

---

## Overview

Phase 2.5 transforms the existing OpenCV-based vectorization pipeline into a 5-stage semantic pipeline with:
- OCR text masking (Tesseract)
- Symbol template matching
- AEC geometric heuristics (orthogonal snapping, collinear merging)
- New sidecar commands (`draw_mtext`, `insert_block`)

### Success Metrics (Targets)

| Metric | Target |
|--------|--------|
| Entity Reduction | >=60% fewer lines |
| Text Accuracy | >=85% as MText |
| Geometric Precision | 100% orthogonal (88-92 deg -> exact 0/90) |
| Processing Time | <15 sec/page |

---

## Architecture

### 5-Stage Pipeline

```
+-------------------------------------------------------------------------+
|                        VECTORIZATION PIPELINE                           |
+-------------------------------------------------------------------------+
|                                                                         |
|  +------------+    +------------+    +------------+                     |
|  |  Stage 1   |    |  Stage 2   |    |  Stage 3   |                     |
|  | OCR Masking|--->|   Symbol   |--->|   Signal   |                     |
|  | (Tesseract)|    | Detection  |    |   Restore  |                     |
|  +------------+    +------------+    +------------+                     |
|        |                 |                 |                            |
|        v                 v                 v                            |
|  DetectedText[]   DetectedBlock[]    Binary Image                       |
|                                                                         |
|  +------------+    +------------+                                       |
|  |  Stage 4   |    |  Stage 5   |                                       |
|  |  Geometry  |--->|    AEC     |---> VectorizationResult               |
|  | Detection  |    | Heuristics |                                       |
|  +------------+    +------------+                                       |
|        |                 |                                              |
|        v                 v                                              |
|  Lines, Circles,   Snapped Lines,                                       |
|  Arcs, Polylines   Merged Segments                                      |
|                                                                         |
+-------------------------------------------------------------------------+
```

---

## Files Created

### 1. src/aec_agent/mcp/tools/ocr_masking.py

**Purpose:** Tesseract OCR integration for detecting and masking text regions.

**Key Components:**

```python
@dataclass
class DetectedText:
    text: str                           # Recognized text content
    position: Tuple[float, float]       # Insertion point (drawing units)
    width: float                        # Bounding box width
    height: float                       # Text height for MText
    confidence: float                   # OCR confidence 0-100
    rotation: float = 0.0               # Rotation in degrees

def detect_and_mask_text(
    image: np.ndarray,
    scale: float = 1.0,
    min_confidence: int = 60,
    lang: str = "eng",
    merge_distance: float = 10.0,
) -> Tuple[np.ndarray, List[DetectedText]]:
    """
    Detect text regions using Tesseract OCR and mask them from the image.

    Returns:
        - masked_image: Image with text regions erased (white)
        - detected_texts: List of DetectedText for MText creation
    """
```

**Dependencies:**
- `pytesseract>=0.3.10`
- Tesseract-OCR system installation

---

### 2. src/aec_agent/utils/geometry_cleanup.py

**Purpose:** AEC-specific geometric cleanup utilities for orthogonal snapping and collinear line merging.

**Key Components:**

```python
@dataclass
class MergedLine:
    start: Tuple[float, float]
    end: Tuple[float, float]
    linetype: str = "CONTINUOUS"  # "CONTINUOUS", "DASHED", "HIDDEN"
    segment_count: int = 1

def snap_to_orthogonal(
    lines: List[Tuple[float, float, float, float]],
    angle_tolerance: float = 2.0,
) -> List[Tuple[float, float, float, float]]:
    """
    Snap near-orthogonal lines to exact 0/90/180/270 degrees.

    Engineering drawings have lines INTENDED to be H/V but scanning
    introduces small angular errors. Lines within +/-tolerance are snapped.
    """

def merge_collinear_lines(
    lines: List[Tuple[float, float, float, float]],
    angle_tolerance: float = 2.0,
    distance_tolerance: float = 5.0,
    gap_tolerance: float = 20.0,
    min_gap_for_dashed: float = 2.0,
) -> Tuple[List[MergedLine], List[MergedLine]]:
    """
    Merge collinear line segments into single lines.

    Returns:
        - continuous_lines: Merged lines with linetype="CONTINUOUS"
        - dashed_lines: Merged lines with linetype="DASHED"
    """
```

**Algorithm Details:**
- Orthogonal snapping preserves line length while adjusting endpoints
- Collinear merging uses 1D projection to detect gaps/overlaps
- Dashed pattern detection via regular gap analysis

---

### 3. src/aec_agent/mcp/tools/symbol_detection.py

**Purpose:** Template matching for detecting standard AEC symbols.

**Key Components:**

```python
@dataclass
class DetectedBlock:
    block_name: str                     # AutoCAD block name to insert
    position: Tuple[float, float]       # Insertion point (drawing units)
    scale: float = 1.0                  # Block scale factor
    rotation: float = 0.0               # Rotation (0, 90, 180, 270)
    confidence: float = 0.0             # Match confidence 0-1
    category: str = ""                  # "mechanical", "electrical", etc.

@dataclass
class SymbolTemplate:
    name: str                           # Template name
    block_name: str                     # AutoCAD block name
    category: str                       # Category folder
    image: np.ndarray                   # Template image
    rotations: List[int]                # Angles to try [0, 90, 180, 270]

def load_symbol_templates(
    template_dir: Optional[str] = None,
) -> List[SymbolTemplate]:
    """Load PNG templates from src/assets/templates/"""

def detect_and_mask_symbols(
    image: np.ndarray,
    templates: List[SymbolTemplate],
    scale: float = 1.0,
    match_threshold: float = 0.8,
    nms_distance: float = 20.0,
    padding_px: int = 2,
) -> Tuple[np.ndarray, List[DetectedBlock]]:
    """
    Detect symbols using cv2.matchTemplate and mask them.

    Returns:
        - masked_image: Image with symbol regions erased
        - detected_blocks: List of DetectedBlock for insertion
    """
```

**Algorithm Details:**
- Uses `cv2.TM_CCOEFF_NORMED` for template matching
- Tests all 4 orthogonal rotations (0, 90, 180, 270 degrees)
- Non-maximum suppression prevents duplicate detections

---

### 4. src/assets/templates/ Directory

**Structure:**
```
templates/
    README.md              # Template creation guidelines
    mechanical/            # HVAC, piping symbols
    electrical/            # Outlets, switches, panels
    fire/                  # Smoke detectors, pull stations
    plumbing/              # Floor drains, cleanouts
    low_voltage/           # Data ports, cameras, card readers
```

**Template Requirements:**
- **Format:** PNG (grayscale)
- **Size:** 32x32 to 64x64 pixels recommended
- **Colors:** White background (255), Black symbol (0)
- **Naming:** `lowercase_with_underscores.png` -> Block name: `UPPERCASE-WITH-DASHES`

---

### 5. tests/unit/test_geometry_cleanup.py

**Test Coverage:** 29 tests

| Test Class | Tests | Description |
|------------|-------|-------------|
| TestSnapToOrthogonal | 11 | Horizontal/vertical snapping, tolerance, edge cases |
| TestMergeCollinearLines | 7 | Segment merging, gap handling, perpendicular rejection |
| TestDashedLineDetection | 5 | Regular pattern detection, noise filtering |
| TestDetectGapsInCluster | 2 | Gap calculation between segments |
| TestHelperFunctions | 2 | Conversion utilities |
| TestMergedLineDataclass | 2 | Dataclass defaults and custom values |

---

## Files Modified

### 1. src/aec_agent/mcp/tools/image_vectorizer.py

**Changes:**

#### Added to DetectedLine dataclass:
```python
@dataclass
class DetectedLine:
    start: Tuple[float, float]
    end: Tuple[float, float]
    linetype: str = "CONTINUOUS"  # NEW: "CONTINUOUS", "DASHED", "HIDDEN"
```

#### Added to VectorizationResult dataclass:
```python
@dataclass
class VectorizationResult:
    # ... existing fields ...
    texts: List["DetectedText"] = field(default_factory=list)    # NEW
    blocks: List["DetectedBlock"] = field(default_factory=list)  # NEW
```

#### New parameters for vectorize_bitonal_image():
```python
def vectorize_bitonal_image(
    # ... existing params ...
    # Phase 2.5: OCR Text Masking
    ocr_masking: bool = False,
    ocr_min_confidence: int = 60,
    ocr_lang: str = "eng",
    # Phase 2.5: Symbol Detection
    symbol_detection: bool = False,
    symbol_threshold: float = 0.8,
    # Phase 2.5: AEC Heuristics
    aec_heuristics: bool = False,
    orthogonal_snap: bool = True,
    orthogonal_angle_tolerance: float = 2.0,
    collinear_merge: bool = True,
) -> VectorizationResult:
```

#### Stage Integration Points:

**Stage 1 - OCR Masking** (after binary_reference, before skeletonization):
```python
if ocr_masking:
    from .ocr_masking import detect_and_mask_text
    binary, detected_texts = detect_and_mask_text(
        binary, scale, ocr_min_confidence, ocr_lang
    )
    result.texts = detected_texts
```

**Stage 2 - Symbol Detection** (after OCR masking):
```python
if symbol_detection:
    from .symbol_detection import load_symbol_templates, detect_and_mask_symbols
    templates = load_symbol_templates()
    binary, detected_blocks = detect_and_mask_symbols(
        binary, templates, scale, symbol_threshold
    )
    result.blocks = detected_blocks
```

**Stage 5 - AEC Heuristics** (after geometry detection, before return):
```python
if aec_heuristics and result.lines:
    from aec_agent.utils.geometry_cleanup import (
        snap_to_orthogonal, merge_collinear_lines
    )
    line_tuples = [(l.start[0], l.start[1], l.end[0], l.end[1]) for l in result.lines]

    if orthogonal_snap:
        line_tuples = snap_to_orthogonal(line_tuples, orthogonal_angle_tolerance)

    if collinear_merge:
        continuous, dashed = merge_collinear_lines(line_tuples)
        # Convert back to DetectedLine with linetype
```

---

### 2. src/aec_agent/mcp/tools/raster_design.py

**Changes:**

#### New parameters for raster_auto_vectorize():
```python
async def raster_auto_vectorize(
    # ... existing params ...
    text_layer: Optional[str] = None,      # NEW
    symbol_layer: Optional[str] = None,    # NEW
    # Phase 2.5: OCR Text Masking
    ocr_masking: bool = False,
    ocr_min_confidence: int = 60,
    ocr_lang: str = "eng",
    # Phase 2.5: Symbol Detection
    symbol_detection: bool = False,
    symbol_threshold: float = 0.8,
    # Phase 2.5: AEC Heuristics
    aec_heuristics: bool = False,
    orthogonal_snap: bool = True,
    orthogonal_angle_tolerance: float = 2.0,
    collinear_merge: bool = True,
) -> dict:
```

#### MText Creation (Step 7):
```python
for text in detection.texts:
    params = {
        "text": text.text,
        "position": [text.position[0], text.position[1], 0.0],
        "height": text.height if text.height > 0 else 2.5,
        "rotation": text.rotation,
    }
    if text.width and text.width > 0:
        params["width"] = text.width
    if effective_text_layer:
        params["layer"] = effective_text_layer.strip()
    await call_autocad_command("draw_mtext", params)
```

#### Block Insertion (Step 8):
```python
for block in detection.blocks:
    params = {
        "block_name": block.block_name,
        "position": [block.position[0], block.position[1], 0.0],
        "scale": block.scale if block.scale > 0 else 1.0,
        "rotation": block.rotation,
    }
    if effective_symbol_layer:
        params["layer"] = effective_symbol_layer.strip()
    await call_autocad_command("insert_block", params)
```

#### Same changes applied to raster_pdf_to_vector_pipeline().

---

### 3. src/sidecars/autocad/Models/CommandParams.cs

**Added Classes:**

```csharp
public class DrawMTextParams
{
    public string Text { get; set; }
    public double[] Position { get; set; }
    public double Height { get; set; } = 2.5;
    public double Width { get; set; }
    public double Rotation { get; set; }
    public string Layer { get; set; }
    public string Style { get; set; }
}

public class InsertBlockParams
{
    public string BlockName { get; set; }
    public double[] Position { get; set; }
    public double Scale { get; set; } = 1.0;
    public double Rotation { get; set; }
    public string Layer { get; set; }
    public Dictionary<string, string> Attributes { get; set; }
}
```

**Modified DrawLineParams:**
```csharp
public class DrawLineParams
{
    public double[] Start { get; set; }
    public double[] End { get; set; }
    public string Layer { get; set; }
    public int? Color { get; set; }
    public string Linetype { get; set; }  // NEW
}
```

---

### 4. src/sidecars/autocad/Commands/DrawingCommands.cs

**Added Methods:**

```csharp
public object DrawMText(object parameters, Document doc, Transaction tr)
{
    var param = Deserialize<DrawMTextParams>(parameters);
    var db = doc.Database;
    var bt = (BlockTable)tr.GetObject(db.BlockTableId, OpenMode.ForRead);
    var btr = (BlockTableRecord)tr.GetObject(bt[BlockTableRecord.ModelSpace], OpenMode.ForWrite);

    var mtext = new MText();
    mtext.SetDatabaseDefaults();
    mtext.Location = new Point3d(param.Position[0], param.Position[1],
                                  param.Position.Length > 2 ? param.Position[2] : 0);
    mtext.Contents = param.Text;
    mtext.TextHeight = param.Height > 0 ? param.Height : 2.5;

    if (param.Width > 0)
        mtext.Width = param.Width;
    if (param.Rotation != 0)
        mtext.Rotation = param.Rotation * Math.PI / 180.0;
    // ... layer and style handling ...

    btr.AppendEntity(mtext);
    tr.AddNewlyCreatedDBObject(mtext, true);
    return new { handle = mtext.Handle.ToString(), /* ... */ };
}

public object InsertBlock(object parameters, Document doc, Transaction tr)
{
    var param = Deserialize<InsertBlockParams>(parameters);
    var db = doc.Database;
    var bt = (BlockTable)tr.GetObject(db.BlockTableId, OpenMode.ForRead);

    if (!bt.Has(param.BlockName))
        throw new ArgumentException($"Block '{param.BlockName}' not found");

    var blockId = bt[param.BlockName];
    var insertPt = new Point3d(param.Position[0], param.Position[1],
                               param.Position.Length > 2 ? param.Position[2] : 0);

    var blockRef = new BlockReference(insertPt, blockId);
    blockRef.ScaleFactors = new Scale3d(param.Scale);
    blockRef.Rotation = param.Rotation * Math.PI / 180.0;
    // ... layer and attribute handling ...

    var btr = (BlockTableRecord)tr.GetObject(bt[BlockTableRecord.ModelSpace], OpenMode.ForWrite);
    btr.AppendEntity(blockRef);
    tr.AddNewlyCreatedDBObject(blockRef, true);
    return new { handle = blockRef.Handle.ToString(), /* ... */ };
}
```

**Modified DrawLine() for linetype support:**
```csharp
if (!string.IsNullOrEmpty(param.Linetype))
{
    LinetypeTable ltTable = (LinetypeTable)tr.GetObject(db.LinetypeTableId, OpenMode.ForRead);
    if (ltTable.Has(param.Linetype))
        line.LinetypeId = ltTable[param.Linetype];
}
```

---

### 5. src/sidecars/autocad/Commands/CommandRouter.cs

**Added to _handlers dictionary:**
```csharp
{ "draw_mtext", _drawingCommands.DrawMText },
{ "insert_block", _drawingCommands.InsertBlock },
```

---

## Usage Examples

### Basic Usage (Backward Compatible)
```python
# Existing behavior unchanged - all Phase 2.5 features default to False
result = await raster_pdf_to_vector_pipeline("C:/plans/floor1.pdf")
```

### With OCR Text Detection
```python
result = await raster_pdf_to_vector_pipeline(
    "C:/plans/floor1.pdf",
    ocr_masking=True,
    ocr_min_confidence=70,
    text_layer="OCR-TEXT",
)
# Creates MText entities for detected text
```

### With Symbol Detection
```python
result = await raster_pdf_to_vector_pipeline(
    "C:/plans/mep-plan.pdf",
    symbol_detection=True,
    symbol_threshold=0.85,
    symbol_layer="SYMBOLS",
)
# Inserts block references for detected symbols
```

### With AEC Heuristics
```python
result = await raster_pdf_to_vector_pipeline(
    "C:/plans/floor1.pdf",
    aec_heuristics=True,
    orthogonal_angle_tolerance=2.0,  # Snap lines within +/-2deg of H/V
)
# Produces cleaner orthogonal geometry
```

### Full Pipeline
```python
result = await raster_pdf_to_vector_pipeline(
    "C:/plans/floor1.pdf",
    # Layers
    target_layer="GEOMETRY",
    text_layer="TEXT",
    symbol_layer="SYMBOLS",
    # Phase 2.5 features
    ocr_masking=True,
    ocr_min_confidence=60,
    symbol_detection=True,
    symbol_threshold=0.8,
    aec_heuristics=True,
    orthogonal_snap=True,
    orthogonal_angle_tolerance=2.0,
    collinear_merge=True,
)
```

---

## Dependencies

### Python (add to pyproject.toml)
```toml
pytesseract = ">=0.3.10"
```

### System Requirements
- **Tesseract-OCR** installed
  - Windows: https://github.com/UB-Mannheim/tesseract/wiki
  - Set `TESSERACT_CMD` environment variable if not in PATH

### Existing (already in project)
- opencv-contrib-python
- numpy
- scikit-image
- networkx

---

## Error Handling

All Phase 2.5 features have graceful degradation:

| Condition | Behavior |
|-----------|----------|
| Tesseract not installed | OCR skipped, warning logged, continues |
| No templates exist | Symbol detection skipped, info logged |
| Block not in drawing | Error returned with message |
| Linetype not found | Falls back to CONTINUOUS |

---

## Testing

### Run Unit Tests
```bash
pytest tests/unit/test_geometry_cleanup.py -v
```

### Test Coverage
```
src/aec_agent/utils/geometry_cleanup.py    96% coverage
```

### Integration Testing
Test with real PDFs using:
```python
# Enable all features
result = await raster_pdf_to_vector_pipeline(
    "test.pdf",
    ocr_masking=True,
    symbol_detection=True,
    aec_heuristics=True,
)
print(f"Created: {result['data']['created']}")
```

---

## Future Enhancements

### Phase 2.5.1 (Planned)
- [ ] YOLOv8 symbol detection for varied drawing styles
- [ ] Vision LLM for semantic understanding
- [ ] Multi-language OCR support

### Template Library Expansion
- [ ] Add more standard AEC symbol templates
- [ ] Create template extraction tool from existing DWG blocks
- [ ] Support for scaled template matching

---

## File Reference Summary

| File | Type | Description |
|------|------|-------------|
| src/aec_agent/mcp/tools/ocr_masking.py | New | Tesseract OCR integration |
| src/aec_agent/utils/geometry_cleanup.py | New | AEC geometric heuristics |
| src/aec_agent/mcp/tools/symbol_detection.py | New | Template matching |
| src/assets/templates/ | New | Symbol template directory |
| tests/unit/test_geometry_cleanup.py | New | Unit tests (29 tests) |
| src/aec_agent/mcp/tools/image_vectorizer.py | Modified | Pipeline integration |
| src/aec_agent/mcp/tools/raster_design.py | Modified | MText/block creation |
| src/sidecars/autocad/Models/CommandParams.cs | Modified | New param classes |
| src/sidecars/autocad/Commands/DrawingCommands.cs | Modified | New commands |
| src/sidecars/autocad/Commands/CommandRouter.cs | Modified | Command registration |
