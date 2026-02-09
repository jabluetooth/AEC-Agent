# Semantic Intelligence Plan: From Dumb Geometry to Smart AEC Objects

**Created:** 2026-02-09
**Status:** Planning
**Goal:** Transform the vectorization pipeline from outputting raw geometry (lines, circles, text) into producing semantically-rich AEC objects that understand what they represent.

---

## The Problem: Dumb Geometry

### Current State

```
PDF Input                    Current Output
┌─────────────────┐         ┌─────────────────────────────────────┐
│ ┌───┐           │         │ Line: (0,0) → (100,0)               │
│ │   │  ROOM 101 │   →     │ Line: (100,0) → (100,80)            │
│ │   │  245 SF   │         │ Line: (100,80) → (0,80)             │
│ └───┘           │         │ Line: (0,80) → (0,0)                │
│   ◇─────────────│         │ Text: "ROOM 101" @ (20, 50)         │
│  VALVE          │         │ Text: "245 SF" @ (20, 40)           │
└─────────────────┘         │ Text: "VALVE" @ (10, 10)            │
                            │ Circle: center=(5,15), r=3          │
                            │ Line: (8,15) → (50,15)              │
                            └─────────────────────────────────────┘
```

**What we have:**
- 4 lines forming a rectangle (but we don't know it's a room boundary)
- Text strings (but we don't know "ROOM 101" is a room name vs. a label)
- A circle + lines (but we don't know it's a gate valve symbol)
- No relationships between elements
- No layer intelligence
- No MEP system classification

**What we need:**
```
Intelligent Output
┌─────────────────────────────────────────────────────────────────┐
│ Room:                                                           │
│   name: "ROOM 101"                                              │
│   area_sf: 245                                                  │
│   boundary: Polygon[(0,0), (100,0), (100,80), (0,80)]          │
│   layer: "A-ROOM"                                               │
│                                                                 │
│ Valve:                                                          │
│   type: "gate_valve"                                            │
│   block_name: "VALVE-GATE"                                      │
│   position: (5, 15)                                             │
│   connected_to: ["PIPE-001"]                                    │
│   layer: "P-VALV"                                               │
│   system: "plumbing"                                            │
│                                                                 │
│ Pipe:                                                           │
│   from: (8, 15)                                                 │
│   to: (50, 15)                                                  │
│   connected_elements: ["VALVE-001"]                             │
│   layer: "P-PIPE"                                               │
│   system: "plumbing"                                            │
└─────────────────────────────────────────────────────────────────┘
```

---

## The Solution: 5-Layer Semantic Pipeline

### Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        SEMANTIC VECTORIZATION PIPELINE                       │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────────┐                                                            │
│  │   PDF/TIF   │                                                            │
│  └──────┬──────┘                                                            │
│         │                                                                   │
│         ▼                                                                   │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ LAYER 1: Document Classification (LLM)                              │   │
│  │ "What type of drawing is this?"                                     │   │
│  │ → floor_plan | mep_plan | single_line | detail | schedule          │   │
│  └──────┬──────────────────────────────────────────────────────────────┘   │
│         │                                                                   │
│         ▼                                                                   │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ LAYER 2: Region Segmentation (CV + LLM)                             │   │
│  │ "Where are the different zones?"                                    │   │
│  │ → title_block | legend | drawing_area | notes | schedules          │   │
│  └──────┬──────────────────────────────────────────────────────────────┘   │
│         │                                                                   │
│         ▼                                                                   │
│  ┌──────┴──────┬──────────────┬──────────────┐                             │
│  │             │              │              │                             │
│  ▼             ▼              ▼              ▼                             │
│ ┌────────┐  ┌────────┐  ┌──────────┐  ┌───────────┐                        │
│ │ TEXT   │  │ SYMBOL │  │ GEOMETRY │  │ SCHEDULE  │                        │
│ │ Stream │  │ Stream │  │ Stream   │  │ Stream    │                        │
│ └───┬────┘  └───┬────┘  └────┬─────┘  └─────┬─────┘                        │
│     │           │            │              │                              │
│     ▼           ▼            ▼              ▼                              │
│ ┌────────┐  ┌────────┐  ┌──────────┐  ┌───────────┐                        │
│ │Semantic│  │ YOLOv8 │  │ OpenCV   │  │ Table     │                        │
│ │  OCR   │  │ + VLM  │  │ + Rules  │  │ Parser    │                        │
│ └───┬────┘  └───┬────┘  └────┬─────┘  └─────┬─────┘                        │
│     │           │            │              │                              │
│     └───────────┴─────┬──────┴──────────────┘                              │
│                       │                                                    │
│                       ▼                                                    │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ LAYER 3: Element Classification (LLM + Rules)                       │   │
│  │ "What is each detected element?"                                    │   │
│  │ → wall | door | window | duct | pipe | valve | outlet | fixture    │   │
│  └──────┬──────────────────────────────────────────────────────────────┘   │
│         │                                                                   │
│         ▼                                                                   │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ LAYER 4: Relationship Inference (Graph + LLM)                       │   │
│  │ "How are elements connected?"                                       │   │
│  │ → room_contains | pipe_connects | duct_branches | valve_on_pipe    │   │
│  └──────┬──────────────────────────────────────────────────────────────┘   │
│         │                                                                   │
│         ▼                                                                   │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ LAYER 5: Knowledge Grounding (pgvector + Rules)                     │   │
│  │ "What CAD standards apply?"                                         │   │
│  │ → layer_name | block_name | color | linetype | attributes          │   │
│  └──────┬──────────────────────────────────────────────────────────────┘   │
│         │                                                                   │
│         ▼                                                                   │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ OUTPUT: Semantic AEC Entities                                       │   │
│  │ • Rooms with names, areas, boundaries                               │   │
│  │ • Equipment with types, specs, connections                          │   │
│  │ • Systems with topology and flow direction                          │   │
│  │ • Proper layers, blocks, attributes                                 │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Implementation Layers

### Layer 1: Document Classification

**Current State:** None
**Goal:** Automatically determine drawing type to configure pipeline

**Implementation:**

```python
# New file: src/aec_agent/mcp/tools/document_classifier.py

class DrawingType(Enum):
    FLOOR_PLAN = "floor_plan"           # Architectural floor plan
    MEP_PLAN = "mep_plan"               # Mechanical/Electrical/Plumbing
    ELECTRICAL_PLAN = "electrical"       # Power, lighting
    HVAC_PLAN = "hvac"                  # Ductwork, equipment
    PLUMBING_PLAN = "plumbing"          # Piping, fixtures
    FIRE_ALARM = "fire_alarm"           # FA devices, wiring
    SINGLE_LINE = "single_line"         # Electrical single-line diagram
    RISER_DIAGRAM = "riser"             # Vertical system diagram
    DETAIL = "detail"                   # Construction detail
    SCHEDULE = "schedule"               # Equipment/room schedule
    LEGEND = "legend"                   # Symbol legend
    UNKNOWN = "unknown"

async def classify_document(
    image: np.ndarray,
    title_block_text: Optional[str] = None,
) -> DocumentClassification:
    """
    Use LLM to classify drawing type.

    Strategy:
    1. Extract title block region (bottom-right corner typically)
    2. OCR the title block for sheet name/number clues
    3. If ambiguous, send low-res thumbnail to Vision LLM
    4. Return classification with confidence
    """
```

**Pipeline Configuration by Drawing Type:**

| Drawing Type | Symbol Detection | OCR Focus | Geometry Rules | Layer Mapping |
|--------------|------------------|-----------|----------------|---------------|
| Floor Plan | Doors, windows | Room names, areas | Wall centerlines | A-* layers |
| MEP Plan | All MEP symbols | Equipment tags | Pipe/duct runs | M-*, E-*, P-* |
| Electrical | Outlets, panels | Circuit IDs | Conduit paths | E-* layers |
| Single-Line | Electrical symbols | Equipment labels | Connection lines | E-DIAG-* |
| Detail | Minimal | Dimensions, notes | All geometry | As-is |

---

### Layer 2: Region Segmentation

**Current State:** None (processes entire image uniformly)
**Goal:** Identify and isolate different regions for specialized processing

**Implementation:**

```python
# New file: src/aec_agent/mcp/tools/region_segmenter.py

class RegionType(Enum):
    TITLE_BLOCK = "title_block"     # Usually bottom-right
    LEGEND = "legend"               # Symbol definitions
    DRAWING_AREA = "drawing"        # Main content
    NOTES = "notes"                 # Text-heavy areas
    SCHEDULE = "schedule"           # Tabular data
    REVISION_BLOCK = "revision"     # Revision history
    SCALE_BAR = "scale"             # Graphical scale

@dataclass
class DetectedRegion:
    type: RegionType
    bounds: Tuple[int, int, int, int]  # x, y, width, height
    confidence: float

async def segment_regions(
    image: np.ndarray,
    drawing_type: DrawingType,
) -> List[DetectedRegion]:
    """
    Segment drawing into functional regions.

    Strategy:
    1. Use template matching for standard title block sizes
    2. Detect horizontal/vertical dividing lines
    3. Identify text-dense regions (likely notes/schedules)
    4. Main drawing area = largest remaining region
    """
```

**Why This Matters:**

| Region | Processing Strategy |
|--------|---------------------|
| Title Block | OCR only → extract project info, sheet number |
| Legend | OCR + symbol matching → build local symbol dictionary |
| Drawing Area | Full pipeline → geometry + symbols + semantic OCR |
| Notes | Text-only OCR → general notes, no geometry |
| Schedule | Table extraction → structured data (rooms, equipment) |

---

### Layer 3: Semantic Text Understanding (Phase 2.5.3)

**Current State:** Raw Tesseract OCR → unstructured strings
**Goal:** Parse text into structured, meaningful data

**The Problem:**

```
Raw OCR Output:
"ROOM 101"
"245 SF"
"24x24 SA"
"200 CFM"
"3/4\" GV"
"TYP. (3)"
```

These are just strings. The system doesn't know:
- "ROOM 101" is a room identifier
- "245 SF" is an area measurement
- "24x24 SA 200 CFM" describes a supply air diffuser
- "3/4\" GV" means a 3/4-inch gate valve
- "TYP. (3)" means "typical, 3 locations"

**Implementation:**

```python
# New file: src/aec_agent/mcp/tools/semantic_ocr.py

class AnnotationType(Enum):
    ROOM_NAME = "room_name"
    ROOM_NUMBER = "room_number"
    AREA = "area"
    EQUIPMENT_TAG = "equipment_tag"
    EQUIPMENT_SPEC = "equipment_spec"
    DIMENSION = "dimension"
    ELEVATION = "elevation"
    NOTE = "note"
    QUANTITY = "quantity"
    SIZE = "size"
    FLOW_RATE = "flow_rate"
    REFERENCE = "reference"

@dataclass
class ParsedAnnotation:
    raw_text: str
    annotation_type: AnnotationType
    structured_data: dict
    confidence: float
    position: Tuple[float, float]

async def parse_annotation(
    text: str,
    context: DrawingType,
    nearby_symbols: List[DetectedBlock] = None,
) -> ParsedAnnotation:
    """
    Use LLM to parse raw OCR text into structured data.

    Example prompts by context:

    MEP Plan:
    "Parse this MEP annotation: '24x24 SA 200 CFM'
     Return JSON: {type, width, height, airflow_cfm, direction}"

    Result: {
        "type": "supply_air_diffuser",
        "width_inches": 24,
        "height_inches": 24,
        "airflow_cfm": 200,
        "direction": "supply"
    }
    """
```

**Parsing Rules by Annotation Type:**

| Pattern | Type | Structured Output |
|---------|------|-------------------|
| `ROOM \d+` | room_number | `{"room": "101"}` |
| `\d+ SF` | area | `{"area_sf": 245}` |
| `\d+x\d+ SA` | equipment_spec | `{"width": 24, "height": 24, "type": "supply_air"}` |
| `\d+ CFM` | flow_rate | `{"cfm": 200}` |
| `\d+[/-]\d+"? [A-Z]+` | size + type | `{"size_inches": 0.75, "type": "gate_valve"}` |
| `TYP\.? \(\d+\)` | quantity | `{"quantity": 3, "typical": true}` |
| `EL\.? [\d.]+` | elevation | `{"elevation_ft": 10.5}` |

**Text-to-Element Association:**

```python
async def associate_text_to_elements(
    texts: List[ParsedAnnotation],
    symbols: List[DetectedBlock],
    geometry: List[DetectedLine],
) -> List[ElementWithAnnotations]:
    """
    Associate parsed text with nearby symbols/geometry.

    Rules:
    1. Equipment specs → nearest equipment symbol (within 2x symbol size)
    2. Room names → enclosing room boundary
    3. Dimensions → associated line segment
    4. Notes → general drawing notes (no association)
    """
```

---

### Layer 4: Symbol Intelligence (Phase 2.5.1 + 2.5.2)

**Current State:** YOLOv8 detection (Phase 2.5.1 complete), but no semantic understanding
**Goal:** Not just detect symbols, but understand what they are and how they connect

**The Gap:**

```
Current YOLOv8 Output:
┌─────────────────────────────────────┐
│ DetectedBlock:                      │
│   class_id: 12                      │
│   class_name: "valve"               │  ← Generic "valve"
│   position: (45.2, 78.3)            │
│   confidence: 0.89                  │
│   block_name: "VALVE"               │  ← Generic block
└─────────────────────────────────────┘

What We Need:
┌─────────────────────────────────────┐
│ SmartSymbol:                        │
│   category: "plumbing"              │
│   type: "valve"                     │
│   subtype: "gate_valve"             │  ← Specific type
│   size: "3/4 inch"                  │  ← From nearby text
│   block_name: "P-VALV-GATE-075"     │  ← Standard block
│   layer: "P-VALV"                   │  ← Correct layer
│   connected_to: ["PIPE-001"]        │  ← Connectivity
│   system: "domestic_cold_water"     │  ← System assignment
└─────────────────────────────────────┘
```

**Implementation:**

```python
# Enhanced: src/aec_agent/mcp/tools/symbol_classifier.py

class SymbolClassifier:
    """
    Two-stage symbol classification:
    1. YOLOv8: Fast detection of symbol locations
    2. Vision LLM: Detailed classification of ambiguous symbols
    """

    async def classify_symbol(
        self,
        symbol_image: np.ndarray,
        yolo_prediction: DetectedBlock,
        nearby_text: List[ParsedAnnotation],
        drawing_context: DrawingType,
    ) -> SmartSymbol:
        """
        Enhance YOLOv8 detection with semantic understanding.

        Steps:
        1. If YOLO confidence > 0.9 and class is specific → use directly
        2. If YOLO class is generic (e.g., "valve") → query Vision LLM
        3. Associate nearby text annotations (size, type, tag)
        4. Query Knowledge Base for block name and layer
        """

        # Step 1: Check if Vision LLM needed
        if self._needs_vision_llm(yolo_prediction):
            detailed_class = await self._query_vision_llm(symbol_image)
        else:
            detailed_class = yolo_prediction.class_name

        # Step 2: Find associated text
        specs = self._find_nearby_specs(yolo_prediction.position, nearby_text)

        # Step 3: Query Knowledge Base for CAD standards
        cad_info = await self._query_knowledge_base(
            category=detailed_class.category,
            type=detailed_class.type,
            size=specs.get("size"),
        )

        return SmartSymbol(
            category=detailed_class.category,
            type=detailed_class.type,
            subtype=detailed_class.subtype,
            size=specs.get("size"),
            block_name=cad_info.block_name,
            layer=cad_info.layer,
            color=cad_info.color,
            position=yolo_prediction.position,
        )
```

**Vision LLM Classification (Phase 2.5.2):**

```python
VISION_LLM_PROMPT = """
Analyze this MEP symbol from an engineering drawing.

Context: This is a {drawing_type} drawing.
Nearby text: {nearby_text}

Identify the symbol and return JSON:
{
    "category": "mechanical|electrical|plumbing|fire|low_voltage",
    "type": "valve|outlet|diffuser|detector|fixture|...",
    "subtype": "gate|ball|butterfly|duplex|quad|...",
    "direction": "up|down|left|right|none",
    "size_hint": "string if visible, null otherwise",
    "confidence": 0.0-1.0
}

If unsure, set confidence < 0.7 and provide best guess.
"""
```

---

### Layer 5: Geometry Intelligence

**Current State:** Raw lines, circles, polylines with no meaning
**Goal:** Understand what geometry represents (walls, ducts, pipes, etc.)

**Implementation:**

```python
# New file: src/aec_agent/mcp/tools/geometry_classifier.py

class GeometryType(Enum):
    WALL = "wall"
    DOOR_SWING = "door_swing"
    WINDOW = "window"
    DUCT = "duct"
    PIPE = "pipe"
    CONDUIT = "conduit"
    CENTERLINE = "centerline"
    DIMENSION_LINE = "dimension"
    LEADER = "leader"
    HIDDEN_LINE = "hidden"
    PROPERTY_LINE = "property"
    UNKNOWN = "unknown"

async def classify_geometry(
    lines: List[DetectedLine],
    circles: List[DetectedCircle],
    polylines: List[DetectedPolyline],
    drawing_type: DrawingType,
    nearby_symbols: List[SmartSymbol],
    nearby_text: List[ParsedAnnotation],
) -> List[ClassifiedGeometry]:
    """
    Classify raw geometry into meaningful AEC elements.

    Classification Rules:

    1. WALLS (Floor Plan):
       - Parallel line pairs with consistent spacing (4", 6", 8")
       - Or thick lines that were skeletonized
       - Orthogonal snapping applied

    2. DUCTS (MEP Plan):
       - Parallel line pairs with spacing matching duct sizes
       - Connected to diffuser/grille symbols
       - Rectangular pattern

    3. PIPES (Plumbing):
       - Single lines connecting valve/fitting symbols
       - May have size annotation nearby

    4. DIMENSION LINES:
       - Lines with arrow/tick endpoints
       - Text annotation at midpoint
       - Parallel to measured element

    5. LEADERS:
       - Line ending at symbol/text
       - Other end has arrow or dot
    """
```

**Geometry Pattern Recognition:**

| Pattern | Indicators | Classification |
|---------|------------|----------------|
| Parallel lines, 4-8" apart, orthogonal | Wall thickness | `WALL` |
| Rectangle connecting to diffuser | Duct size annotation | `DUCT` |
| Single line between valves | Pipe size text | `PIPE` |
| Arc with radius = door width | Door symbol nearby | `DOOR_SWING` |
| Line with arrows, text midpoint | Numeric text | `DIMENSION_LINE` |
| Dashed line | Linetype pattern | `HIDDEN_LINE` or `CENTERLINE` |

---

### Layer 6: Relationship Inference

**Current State:** No relationships between elements
**Goal:** Build a graph of how elements connect and contain each other

**Implementation:**

```python
# New file: src/aec_agent/mcp/tools/relationship_builder.py

class RelationType(Enum):
    CONTAINS = "contains"           # Room contains equipment
    CONNECTED_TO = "connected_to"   # Pipe connects to valve
    BRANCHES_FROM = "branches_from" # Duct branches from main
    FEEDS = "feeds"                 # Panel feeds circuit
    SERVES = "serves"               # Diffuser serves room
    ADJACENT_TO = "adjacent_to"     # Room adjacent to room
    ON_LEVEL = "on_level"           # Element on floor level

@dataclass
class ElementRelationship:
    source_id: str
    target_id: str
    relationship: RelationType
    metadata: dict  # e.g., {"flow_direction": "supply"}

async def build_relationships(
    rooms: List[Room],
    equipment: List[SmartSymbol],
    systems: List[ClassifiedGeometry],
) -> List[ElementRelationship]:
    """
    Infer relationships between detected elements.

    Strategies:

    1. CONTAINMENT:
       - Point-in-polygon test: equipment inside room boundary
       - Result: room CONTAINS equipment

    2. CONNECTIVITY:
       - Endpoint proximity: pipe ends at valve position
       - Result: pipe CONNECTED_TO valve

    3. TOPOLOGY:
       - Build graph from connected geometry
       - Trace flow paths through valves
       - Identify branches and mains

    4. SPATIAL:
       - Rooms sharing wall segments are ADJACENT_TO
       - Equipment near room center SERVES room
    """
```

**Relationship Examples:**

```
Input Elements:
- Room "ROOM 101" with boundary polygon
- Diffuser "DIFF-001" at position inside room
- Duct segment connecting to diffuser
- VAV box symbol upstream

Inferred Relationships:
┌────────────────────────────────────────────────────┐
│ ROOM-101 ←─CONTAINS─ DIFF-001                      │
│ DIFF-001 ←─CONNECTED_TO─ DUCT-001                  │
│ DUCT-001 ←─BRANCHES_FROM─ DUCT-MAIN-001            │
│ VAV-001 ←─FEEDS─ DUCT-001                          │
│ DIFF-001 ←─SERVES─ ROOM-101                        │
└────────────────────────────────────────────────────┘
```

---

### Layer 7: Knowledge Base Grounding (Phase 3)

**Current State:** No CAD standards knowledge
**Goal:** Apply correct layers, blocks, colors, and attributes based on standards

**Implementation:**

```python
# New file: src/aec_agent/mcp/tools/knowledge_query.py

@dataclass
class CADStandards:
    layer: str
    color: int
    linetype: str
    block_name: str
    attributes: dict

async def query_cad_standards(
    element_type: str,
    system: str,
    size: Optional[str] = None,
    project_standards: Optional[str] = None,
) -> CADStandards:
    """
    Query Knowledge Base for CAD insertion rules.

    Priority:
    1. Project-specific standards (if defined)
    2. Company standards
    3. Default AEC standards (NCS-based)

    Example Query:
    "What are the CAD standards for a 3/4 inch gate valve on
     a domestic cold water system?"

    Response:
    {
        "layer": "P-DOMW-VALV",
        "color": 5,  # Blue
        "linetype": "CONTINUOUS",
        "block_name": "P-VALV-GATE",
        "attributes": {
            "SIZE": "3/4\"",
            "TYPE": "GATE",
            "TAG": ""  # To be filled
        }
    }
    """
```

**Knowledge Base Structure:**

```yaml
# knowledge_base/cad_standards/plumbing.yaml

plumbing:
  systems:
    domestic_cold_water:
      abbreviation: "DCW"
      layer_prefix: "P-DOMW"
      color: 5  # Blue

    domestic_hot_water:
      abbreviation: "DHW"
      layer_prefix: "P-DOMH"
      color: 1  # Red

  elements:
    valve:
      gate:
        block_name: "P-VALV-GATE"
        layer_suffix: "-VALV"
        sizes: ["1/2\"", "3/4\"", "1\"", "1-1/4\"", "1-1/2\"", "2\""]

      ball:
        block_name: "P-VALV-BALL"
        layer_suffix: "-VALV"

    pipe:
      layer_suffix: "-PIPE"
      linetype: "CONTINUOUS"
      lineweight_by_size:
        "1/2\"": 0.25
        "3/4\"": 0.35
        "1\"": 0.50
```

---

## Complete Pipeline Flow

### Before (Current)

```
PDF → Tesseract → "ROOM 101" (raw string)
    → OpenCV → Line(0,0 → 100,0) (raw geometry)
    → OpenCV → Circle(5,15,3) (raw circle)
    → AutoCAD: draw_line, draw_circle, draw_mtext

Result: Disconnected primitives on Layer 0
```

### After (Semantic Pipeline)

```
PDF
  → Layer 1: Document Classification
    → "This is an MEP Plan (Plumbing)"

  → Layer 2: Region Segmentation
    → Drawing area, title block, legend identified

  → Layer 3: Parallel Streams
    │
    ├─ Text Stream:
    │   → Tesseract: "ROOM 101", "245 SF", "3/4\" GV"
    │   → Semantic OCR: room_number=101, area=245, valve_size=0.75"
    │   → Text-to-Element: "3/4\" GV" → nearest valve symbol
    │
    ├─ Symbol Stream:
    │   → YOLOv8: valve detected at (5,15), conf=0.87
    │   → Vision LLM: gate_valve (not ball, not butterfly)
    │   → Knowledge Base: block=P-VALV-GATE, layer=P-DOMW-VALV
    │
    └─ Geometry Stream:
        → OpenCV: lines, circles detected
        → Geometry Classifier: this line connects valve to fixture
        → Classification: PIPE, domestic_cold_water

  → Layer 4: Relationship Builder
    → Valve CONNECTED_TO Pipe
    → Pipe CONNECTED_TO Fixture
    → Fixture IN Room "ROOM 101"

  → Layer 5: Knowledge Grounding
    → Apply layer P-DOMW-PIPE to pipe
    → Apply block P-VALV-GATE to valve
    → Apply layer A-ROOM to room boundary

  → AutoCAD Output:
    → INSERT P-VALV-GATE on layer P-DOMW-VALV with SIZE="3/4\""
    → PLINE on layer P-DOMW-PIPE with correct lineweight
    → Room boundary on layer A-ROOM with ROOM attribute

Result: Intelligent AEC objects with correct layers, blocks,
        attributes, and relationships stored in PostgreSQL
```

---

## Implementation Phases

### Phase A: Foundation (2 weeks)

| Task | File | Description |
|------|------|-------------|
| A1 | `document_classifier.py` | LLM-based drawing type classification |
| A2 | `region_segmenter.py` | Title block, legend, drawing area detection |
| A3 | Update `image_vectorizer.py` | Integrate classification into pipeline |

**Deliverables:**
- Pipeline knows what type of drawing it's processing
- Regions processed with appropriate strategies
- Title block text extracted as project metadata

### Phase B: Semantic Text (2 weeks)

| Task | File | Description |
|------|------|-------------|
| B1 | `semantic_ocr.py` | LLM parsing of annotations |
| B2 | `text_associator.py` | Link text to nearby elements |
| B3 | `prompts/annotation_parsing.py` | Domain-specific prompts |

**Deliverables:**
- "24x24 SA 200 CFM" → structured diffuser spec
- Text associated with correct symbols
- Room names linked to room boundaries

### Phase C: Symbol Intelligence (2 weeks)

| Task | File | Description |
|------|------|-------------|
| C1 | `vision_llm.py` | Gemini/GPT-4o integration for symbol ID |
| C2 | Update `symbol_detection.py` | Two-stage detection pipeline |
| C3 | `prompts/symbol_classification.py` | Vision LLM prompts |

**Deliverables:**
- Generic "valve" → specific "gate_valve" or "ball_valve"
- Unknown symbols identified via Vision LLM
- Symbol specs extracted from nearby text

### Phase D: Geometry Intelligence (2 weeks)

| Task | File | Description |
|------|------|-------------|
| D1 | `geometry_classifier.py` | Pattern-based geometry classification |
| D2 | Update `geometry_cleanup.py` | System-aware processing |
| D3 | `topology_analyzer.py` | Graph-based connectivity |

**Deliverables:**
- Lines classified as walls, ducts, pipes, etc.
- Parallel lines merged into single elements with width
- Connected geometry forms system graphs

### Phase E: Relationships (1 week)

| Task | File | Description |
|------|------|-------------|
| E1 | `relationship_builder.py` | Infer element relationships |
| E2 | Update `repository.py` | Store relationships in PostgreSQL |

**Deliverables:**
- Room-contains-equipment relationships
- Pipe-connects-to-valve relationships
- System topology stored in database

### Phase F: Knowledge Grounding (2 weeks)

| Task | File | Description |
|------|------|-------------|
| F1 | `knowledge_query.py` | CAD standards lookup |
| F2 | Create `knowledge_base/` | Standards YAML files |
| F3 | Update sidecar commands | Apply layers/blocks/attributes |

**Deliverables:**
- Elements created with correct layers
- Standard blocks inserted (not raw geometry)
- Attributes populated from parsed text

---

## Success Metrics

| Metric | Current | Target | Measurement |
|--------|---------|--------|-------------|
| Symbol classification accuracy | 0% | 85% | Correct block name assigned |
| Text parsing accuracy | 0% | 90% | Structured data matches manual review |
| Layer assignment accuracy | 0% | 95% | Correct layer per element type |
| Relationship detection | 0% | 80% | Correct containment/connectivity |
| Entity reduction | 0% | 60% | Raw primitives → AEC objects |
| Processing time | ~10s | <30s | Per page with full pipeline |

---

## Cost Analysis

| Component | Provider | Cost per Page | Notes |
|-----------|----------|---------------|-------|
| Document Classification | Groq (Llama 3.3) | $0.00 | Free tier, 1 call/page |
| Semantic OCR | Groq (Llama 3.3) | $0.00 | ~10-50 calls/page |
| Vision LLM (symbols) | Gemini Pro Vision | ~$0.02 | ~5-20 symbols/page |
| Vision LLM (fallback) | GPT-4o | ~$0.05 | Only for ambiguous |
| YOLOv8 | Local | $0.00 | ONNX inference |
| Knowledge Base | PostgreSQL | $0.00 | Local pgvector |
| **Total per page** | | **~$0.02-0.10** | Depends on complexity |

---

## File Summary

### New Files to Create

| File | Purpose | Phase |
|------|---------|-------|
| `src/aec_agent/mcp/tools/document_classifier.py` | Drawing type classification | A |
| `src/aec_agent/mcp/tools/region_segmenter.py` | Region detection | A |
| `src/aec_agent/mcp/tools/semantic_ocr.py` | Text parsing | B |
| `src/aec_agent/mcp/tools/text_associator.py` | Text-element linking | B |
| `src/aec_agent/mcp/tools/vision_llm.py` | Vision LLM integration | C |
| `src/aec_agent/mcp/tools/geometry_classifier.py` | Geometry understanding | D |
| `src/aec_agent/mcp/tools/topology_analyzer.py` | System graphs | D |
| `src/aec_agent/mcp/tools/relationship_builder.py` | Element relationships | E |
| `src/aec_agent/mcp/tools/knowledge_query.py` | CAD standards lookup | F |
| `src/aec_agent/prompts/annotation_parsing.py` | OCR prompts | B |
| `src/aec_agent/prompts/symbol_classification.py` | Vision prompts | C |
| `knowledge_base/cad_standards/*.yaml` | Layer/block rules | F |

### Files to Modify

| File | Changes | Phase |
|------|---------|-------|
| `image_vectorizer.py` | Integrate all layers | All |
| `raster_design.py` | Pipeline orchestration | All |
| `symbol_detection.py` | Two-stage detection | C |
| `geometry_cleanup.py` | System-aware processing | D |
| `repository.py` | Store relationships | E |

---

## Conclusion

The current pipeline produces **dumb geometry** — lines, circles, and text that have no understanding of what they represent. This plan transforms it into a **semantic intelligence pipeline** that:

1. **Understands the drawing** — Knows it's a plumbing plan vs. an electrical plan
2. **Parses text meaningfully** — "3/4\" GV" becomes a structured gate valve specification
3. **Identifies symbols specifically** — Not just "valve" but "gate_valve on DCW system"
4. **Classifies geometry** — This line is a pipe, that pair of lines is a wall
5. **Infers relationships** — The valve is on the pipe that serves Room 101
6. **Applies standards** — Correct layers, blocks, and attributes per company/project standards

The result: **CAD files that are as intelligent as hand-drawn ones**, with proper layering, standard blocks, attributes, and database relationships for downstream querying and analysis.

---

*Document created: 2026-02-09*
*Related: FUTURE_ROADMAP.md, PHASE_2_5_IMPLEMENTATION.md, SESSION_CONTEXT.md*
