# AI Session Context
> **DO NOT DELETE**. This file maintains the continuity of work between AI coding sessions.

## 🟢 Current Focus
**Objective:** Gemini-First PDF to AutoCAD Pipeline — Phases 1-5 COMPLETE
**Last Action:** Gemini-First Phase 5 Implementation (2026-02-16):
- Created `autocad_creation.py` — AutoCAD entity creation module (~900 lines)
- Creates entities in AutoCAD from Phase 4 ExtractionResult
- Dataclasses: `EntityCreationResult`, `LayerCreationResult`, `CreationStatistics`, `AutoCADCreationResult`
- Core functions: `create_entities_in_autocad()`, `create_entities_batch()`, `create_single_entity()`, `create_layer_if_needed()`
- Entity creators: `create_line_entity()`, `create_arc_entity()`, `create_circle_entity()`, `create_text_entity()`, `create_block_entity()`, `create_polyline_entity()`
- MCP Tools: `gemini_create_entities` (extract + create), `gemini_vectorize_pdf` (complete PDF-to-AutoCAD pipeline), `gemini_create_from_extraction` (create from saved JSON)
- NCS-compliant layer colors: `LAYER_PREFIX_COLORS` (A-=white, M-=cyan, E-=red, P-=blue, F-=red, T-=green)
- Automatic layer creation for required layers
- Progress tracking with success/failure statistics
- 55 unit tests passing for Phase 5 components

**Previous Action:** Gemini-First Phase 4 Implementation (2026-02-14):
- Created `adaptive_extraction.py` — Adaptive entity extraction module (~750 lines)
- Three extraction strategies: Direct, Guided Rasterization, Selective OpenCV
- Dataclasses: `EntityToCreate`, `RasterCommand`, `ExtractionResult`
- 58 unit tests passing for Phase 4 components

**Previous Action:** Gemini-First Phase 3 Implementation (2026-02-14):
- Created `coordinate_calibration.py` — Pixel-to-DWG coordinate conversion module (~600 lines)
- Converts Gemini's pixel coordinates to AutoCAD drawing units
- Multiple calibration methods ranked by confidence: dimension (90%), scale notation (85%), sheet size (70%), DPI default (30%)
- Dataclasses: `ScaleCalibration` (with to_dwg(), to_pixels(), scale_length(), convert_units_to() methods)
- MCP Tools: `gemini_calibrate_coordinates`, `gemini_calibrate_manual`, `gemini_convert_coordinates`, `gemini_parse_scale`, `gemini_parse_measurement`, `gemini_analyze_pdf_calibrated`
- 71 unit tests passing for Phase 3 components

**Previous Action:** Gemini-First Phase 2 Implementation (2026-02-13):
- Created `gemini_understanding.py` — Gemini Vision drawing analysis module (~550 lines)
- Analyzes drawings BEFORE extraction to understand content
- Output: DrawingAnalysis with elements, regions, calibration hints, extraction strategy
- Dataclasses: DrawingAnalysis, DrawingElements, DetectedLine, DetectedArc, DetectedCircle, DetectedText, DetectedSymbol, DetectedDimension, CalibrationHint, SpecialRegion
- MCP Tools: `gemini_analyze_drawing`, `gemini_analyze_pdf`, `gemini_get_extraction_strategy`
- 38 unit tests passing for Phase 2 components

**Previous Action:** Gemini-First Phase 1 Implementation (2026-02-13):
- Created `gemini_first/` package for quality-preserving PDF vectorization
- Created `pdf_intake.py` — high-quality PDF rendering (NO bitonal conversion)
- Created `mcp_tools.py` — MCP tool wrappers (7 tools registered total)
- Key difference: Preserves grayscale/color information for Gemini Vision analysis
- Tools: `gemini_render_pdf`, `gemini_get_pdf_info`, `gemini_render_all_pages`, `gemini_compare_rendering_quality`
- Dataclasses: `PDFRenderResult`, `PDFInfo`
- Functions: `render_pdf_high_quality()`, `render_pdf_high_quality_async()`, `is_effectively_grayscale()`
- Integrated with existing MCP server (direct `@mcp.tool()` decorators)
- 20 unit tests passing for Phase 1 components

**Previous Action:** Gemini LLM Integration (2026-02-12):
- Added `KnowledgeLLM` class for LLM-powered CAD standards queries
- Added `LLMQueryResult` dataclass for LLM query responses
- Added `query_with_llm()` convenience function
- Supports Gemini (primary), OpenAI, and Anthropic providers with auto-fallback
- Methods: `query()`, `classify_element()`, `get_code_reference()`, `suggest_attributes()`
- System prompt with NCS layer naming, MEP standards, and building code knowledge
- JSON response parsing with fallback for plain text
- Added 20 new unit tests for KnowledgeLLM class
- All 132 knowledge_query tests passing

**Previous Action:** Phase F Implementation (2026-02-12):
- Created `knowledge_query.py` — CAD standards lookup module with 1500+ lines
- Added `CADStandards` dataclass (layer, color, linetype, lineweight, block_name, attributes)
- Added `Discipline` enum (ARCHITECTURAL, MECHANICAL, ELECTRICAL, PLUMBING, FIRE, LOW_VOLTAGE)
- Added `SystemType` enum with 25+ MEP system types (supply_air, domestic_cold_water, fire_alarm, etc.)
- Added `GroundedElement` dataclass for elements with standards applied
- Created `knowledge_base/cad_standards/` directory with 6 YAML files (plumbing, mechanical, electrical, fire, low_voltage, architectural)
- Added `query_cad_standards()` async function — Priority lookup: project → company → YAML → defaults
- Added `ground_element()` and `ground_elements()` for batch standards application
- Added convenience functions: `get_layer_for_element()`, `get_block_name()`, `get_color_for_system()`
- 50+ block name patterns across all MEP categories (valves, diffusers, outlets, detectors, etc.)
- NCS-based layer naming (P-DOMW-VALV, M-HVAC-DIFF, E-POWR-OUTL, F-ALRM-DETC, T-DATA-OUTL)
- Updated `image_vectorizer.py` with Phase F integration and new parameters
- Added 132 unit tests for Phase F components (including Gemini LLM tests)
- All tests passing (700+ total)

**Previous Action:** Phase E Implementation (2026-02-11):
- Created `relationship_builder.py` — Relationship inference engine with 20+ relationship types
- Added 39 unit tests for Phase E components

**Previous Action:** Phase D Implementation (2026-02-11):
- Created `geometry_classifier.py` — Pattern-based geometry classification (walls, ducts, pipes)
- Created `topology_analyzer.py` — Graph-based system connectivity analysis
- Added 88 new unit tests for Phase D components

**Previous Action:** Phase C Implementation (2026-02-10):
- Created `vision_llm.py` — Vision LLM integration (Gemini/GPT-4o/Claude) for symbol classification
- Created `symbol_classifier.py` — Two-stage detection pipeline (YOLO → Vision LLM)
- Added 89 new unit tests for Phase C components

**Phase A (COMPLETE):**
- `document_classifier.py` — Drawing type classification (14 types)
- `region_segmenter.py` — Title block, legend, drawing area detection
- `SEMANTIC_INTELLIGENCE_PLAN.md` — 5-layer semantic pipeline roadmap

**Phase B (COMPLETE):**
- `AnnotationType` enum with 20 annotation types (room_name, equipment_tag, flow_rate, size, etc.)
- `parse_annotation_by_patterns()` — Fast regex-based parsing (no LLM cost)
- `associate_text_to_elements()` — Links text to nearest symbols using distance + type rules
- `enrich_elements_with_text()` — Creates enriched elements with tags, specs, annotations
- Pipeline parameters: `semantic_ocr`, `text_association`, `semantic_llm_fallback`

**Phase C (COMPLETE):**
- `VisionLLMClassifier` class supporting Gemini, OpenAI, and Anthropic vision models
- `SmartSymbol` dataclass with full semantic data (subtype, direction, system, specs, associated text)
- `SymbolClassifier` orchestrates two-stage pipeline (YOLO confidence threshold → Vision LLM)
- `SYMBOL_SUBTYPES` dictionary with 50+ subtypes across 5 MEP categories
- Specialized prompts for valves, diffusers, outlets, and detectors
- Pipeline parameters: `vision_llm_classification`, `vision_llm_provider`, `vision_llm_confidence_threshold`

**Phase D (COMPLETE):**
- `GeometryClassifier` class for pattern-based geometry classification (parallel lines → walls/ducts)
- `TopologyAnalyzer` class for graph-based MEP system connectivity analysis
- `GeometryType` enum with 30+ types (wall, duct, pipe, conduit, dimension_line, leader, etc.)
- `GeometrySystem` enum with 15 MEP systems (supply_air, return_air, domestic_cold, power, etc.)
- `TopologyGraph` with BFS/DFS pathfinding, connected components, branch identification
- Classification based on: parallel line spacing, linetype, drawing context, nearby symbols/text
- Pipeline parameters: `geometry_classification` (bool, default True)

**Phase E (COMPLETE):**
- `RelationshipBuilder` class for multi-strategy relationship inference
- `RelationType` enum with 20+ types (contains, connected_to, branches_from, feeds, serves, adjacent_to, near, etc.)
- `InferredRelationship` dataclass with confidence, distance, system, bidirectional, reasoning
- Point-in-polygon containment detection (room CONTAINS equipment)
- Endpoint proximity connectivity detection (pipe CONNECTED_TO valve)
- Spatial analysis (room ADJACENT_TO room, symbol NEAR symbol)
- Flow path detection from TopologyGraph (source FEEDS terminal, branch BRANCHES_FROM main)
- Text labeling detection (annotation LABELS symbol)
- Repository methods: `create_relationships_batch()`, `get_connectivity_graph()`, `find_path_between_elements()`
- Pipeline parameters: `relationship_inference` (bool, default True)

**Phase F (COMPLETE):**
- `knowledge_query.py` — CAD standards lookup module with hierarchical lookup (project → company → YAML → defaults)
- `CADStandards` dataclass with layer, color, linetype, lineweight, block_name, attributes
- `Discipline` enum (8 disciplines: architectural, mechanical, electrical, plumbing, fire, low_voltage, civil, structural)
- `SystemType` enum with 25+ MEP system types (domestic_cold_water, supply_air, fire_alarm, data, security, etc.)
- `GroundedElement` dataclass for elements with standards applied
- `ground_elements()` async function for batch standards application with statistics
- Created `knowledge_base/cad_standards/` directory with 6 YAML files:
  - `plumbing.yaml` — Valves, pipes, fittings, fixtures, equipment (CPC standards)
  - `mechanical.yaml` — Ducts, diffusers, grilles, dampers, VAV, AHU (CMC standards)
  - `electrical.yaml` — Outlets, switches, lights, panels, conduit (CEC/Title 24)
  - `fire.yaml` — Smoke/heat detectors, horn/strobes, sprinklers, FACP (NFPA 72)
  - `low_voltage.yaml` — Data, voice, security cameras, access control (TIA/EIA)
  - `architectural.yaml` — Walls, doors, windows, rooms (NCS)
- 50+ block name patterns: P-VALV-GATE, M-DIFF-SQ, E-OUTL-DUP, F-DETC-SMOK, T-DATA-RJ45
- NCS-based layer naming: discipline-major-minor (P-DOMW-VALV, M-HVAC-DIFF, E-POWR-OUTL)
- Color coding by system: Blue=cold water/supply air, Red=hot water/power/fire, Cyan=return air
- Convenience functions: `get_layer_for_element()`, `get_block_name()`, `get_color_for_system()`
- Pipeline parameters: `knowledge_grounding` (bool), `project_standards`, `company_standards`
- 112 unit tests covering all components

**Semantic Intelligence Pipeline: COMPLETE (Phases A-F)**
- All 6 phases of the Semantic Intelligence Pipeline are now implemented
- Full pipeline: Document Classification → Region Segmentation → Semantic OCR → Symbol Intelligence → Geometry Intelligence → Relationship Inference → Knowledge Grounding

**Next Step:** Gemini-First Phase 6 (Validation & Self-Correction) — Gemini verifies output and corrects errors.

**Gemini-First Architecture:** This is a new approach to PDF vectorization that addresses noise issues in the current bitonal pipeline. Instead of preprocessing first (which destroys information), Gemini-First:
1. **Phase 1 (COMPLETE):** Render PDF to high-quality PNG (preserves grayscale/color, anti-aliasing)
2. **Phase 2 (COMPLETE):** Gemini Vision analyzes the original image to understand drawing content
3. **Phase 3 (COMPLETE):** Coordinate calibration (pixels → DWG units)
4. **Phase 4 (COMPLETE):** Adaptive extraction (direct/guided/selective strategies)
5. **Phase 5 (COMPLETE):** AutoCAD entity creation (draw lines, arcs, text, blocks)
6. **Phase 6 (NEXT):** Validation & self-correction (Gemini verifies output)

## 📊 Repository Status (as of 2026-02-02)

### What's Built & Working
| Component | Files | Status |
|-----------|-------|--------|
| **4-layer architecture** (UI → MCP → Sidecars → CAD) | Full stack | Implemented |
| **MCP Server** (FastMCP, 50 tools, SSE, tool lock) | `src/aec_agent/mcp/` | Implemented |
| **AutoCAD Sidecar** (.NET 4.8, thread queue, HTTP) | `src/sidecars/autocad/` (14 C# files) | Compiled |
| **Revit Sidecar** (pyRevit, Routes API, events) | `src/sidecars/revit/` (14 Python files) | Implemented |
| **Chainlit Frontend** (chat UI, SSE, app context) | `src/aec_agent/frontend/` | Implemented |
| **Multi-LLM Support** (OpenAI, Anthropic, Groq, Gemini, Azure, HF) | `src/aec_agent/frontend/agent.py` | Implemented |
| **Configuration** (42 env vars, Pydantic Settings) | `src/aec_agent/config/settings.py` | Implemented |
| **Intent Classification** (keyword + optional embeddings) | `src/aec_agent/intent/` | Implemented |
| **Conversation Memory** (summarization, preferences) | `src/aec_agent/memory/` | Implemented |
| **Token Optimization** (compression, filtering, tiers) | `src/aec_agent/frontend/tool_optimization.py` | Implemented |
| **DB Schema** (PostGIS 3.6.1 + pgvector 0.8.1, 15 tables) | `src/aec_agent/db/` | Running |
| **Extraction Pipeline** (AutoCAD + Revit extractors, sync) | `src/aec_agent/extraction/` | Code ready |
| **Semantic Search** (all-MiniLM-L6-v2, vector similarity) | `src/aec_agent/semantic/` | Code ready |
| **MEP Domain Logic** (rules engine, seed data, validators) | `src/aec_agent/domain/` | Code ready |
| **LLM Provider Fallback** (Groq -> Gemini -> OpenAI chain) | `src/aec_agent/frontend/agent.py` | Implemented |
| **REST Sidecar API** (/tools/notify_file_opened, /health) | `src/aec_agent/server.py` | Implemented |
| **Cached Context Tool** (get_file_context from PostgreSQL) | `src/aec_agent/mcp/tools/metadata.py` | Implemented |
| **Test Suite** (242 tests, pytest markers) | `tests/` | Passing |

### What's NOT Yet Done
| Component | Blocker | Roadmap Phase |
|-----------|---------|---------------|
| **End-to-end extraction test** (real CAD file → DB) | Needs running sidecar | Phase 1 (integration) |
| **Knowledge base files** (codes, standards, formulas) | Content creation | Phase 3 |
| **Raster Design integration** (PDF → DWG) | Full pipeline built w/ bitonal conversion, needs e2e test with real PDF | Phase 2 (complete, needs testing) |
| **Gemini-First Pipeline** (quality-preserving PDF → DWG) | Phases 1-5 complete, Phase 6 pending | Phase 2.6 (in progress) |
| **Element placement tools** (`place_revit_family`, `place_autocad_block`) | Tool development | Phase 4 |
| **Engineering calculations** (`calculate_ventilation`, `calculate_duct_size`, etc.) | Tool development | Phase 4 |
| **Routing/pathfinding** (`find_route`, `create_duct_run`) | Algorithm dev | Phase 4 |
| **Code validation** (CMC, CEC, CPC, CFC, LAMC) | Knowledge base | Phase 4+ |

## 📅 Roadmap Status
- [x] **Core Infrastructure** — 4-layer stack, MCP server, sidecars, frontend, config
- [x] **MCP Tool Framework** — 33 tools across 6 categories, tool lock, optimization
- [x] **AutoCAD Sidecar** — .NET plugin with STA thread marshaling, HTTP server
- [x] **Revit Sidecar** — pyRevit extension with Routes API, event hooks
- [x] **Database Schema** — Alembic migrations, PostGIS + pgvector, async pooling
- [x] **Extraction Pipeline** — AutoCAD + Revit extractors, sync manager, file hash detection
- [x] **Semantic Search** — Embeddings service, similarity search, description generator
- [x] **Intent & Memory** — Classifier, conversation summarizer, user preferences
- [x] **Token Optimization** — Dynamic compression, smart routing, tier-based loading
- [x] **CLAUDE.md cleanup** — Trimmed to 61 lines, references FUTURE_ROADMAP.md
- [x] **Phase 1: DB Foundation** (COMPLETE)
    - [x] PostgreSQL 18.1 running locally with PostGIS 3.6.1 + pgvector 0.8.1
    - [x] Alembic migrations executed — 15 tables created
    - [x] DATABASE_URL format fixed for asyncpg compatibility
    - [x] LLM provider fallback chain (Groq → Gemini → OpenAI → Anthropic)
    - [x] REST endpoint for sidecar file-open notifications
    - [x] `get_file_context` tool for cached context (90% token reduction)
    - [x] All `metadata` + `mep_tools` registered in MCP server
    - [x] 179 tests passing
    - [ ] End-to-end integration test with real CAD file (needs sidecar running)
- [x] **Phase 2: Raster Design** (COMPLETE)
    - [x] Raster Design sidecar commands (import_pdf, attach_image, cleanup, vectorize, ocr, get_status, get_entity_count)
    - [x] Async command support (SendStringToExecute path in OnIdle, IsAsyncCommand router)
    - [x] `extract_all_entities` C# command — full geometry extraction with pagination
    - [x] `raster_fade_image` C# command — fade raster images for background reference
    - [x] `raster_store_vectorized` MCP tool — extract + store in PostgreSQL with embeddings/relationships
    - [x] `raster_pdf_to_vector_pipeline` MCP tool — auto-detect PDF type, bitonal, despeckle, deskew, vectorize, OCR, fade, store
    - [x] Fixed `stream_entities()` to use `extract_all_entities` (was calling nonexistent `extract_batch`)
    - [x] `raster_convert_pdf` MCP tool + `pdf_converter.py` — Python-side PDF-to-bitonal-TIFF (PyMuPDF + Pillow, 300 DPI, Group4 compression)
    - [x] Fixed pipeline: scanned branch now converts PDF→TIFF before attaching (Raster Design cannot attach PDFs directly)
    - [x] Replaced non-existent `IVECTORIZE` with actual VTools (`vline`, `vpline`, `varc`, `vcircle`, `vrect`)
    - [x] Replaced non-existent `IOCR` with `irectext` (Raster Design text recognition)
    - [x] Added `raster_process_image` (ibfilter: smooth/thin/thicken/separate/skeletonize)
    - [x] Added `raster_create_primitive` (REM: issmart/isline/isarc/iscircle)
    - [x] Added `raster_select_entities` (isebrsmart/isebrcon — region selection)
    - [x] Added `raster_follower` (vfpline/vfcontour/vf3dpoly — semi-auto tracing)
    - [x] Added `raster_recognize_text` (irectext — raster text to AutoCAD text)
    - [x] **OpenCV auto-vectorization** — Replaced interactive VTools with Python-side OpenCV (HoughLinesP, HoughCircles, findContours)
    - [x] `raster_auto_vectorize` MCP tool — standalone OpenCV detect → AutoCAD draw entities
    - [x] `image_vectorizer.py` — OpenCV feature detection module (DetectedLine, DetectedCircle, DetectedPolyline)
    - [x] Pipeline now: bitonal TIFF → attach → despeckle → deskew → **OpenCV detect** → **draw_line/draw_polyline/draw_circle** → fade → store
    - [ ] End-to-end test with real PDF + running sidecar
- [x] **Phase 2.5.1: YOLOv8 Symbol Detection** (COMPLETE)
    - [x] `yolo_detection.py` — YOLOv8 detector with ultralytics + ONNX Runtime backends
    - [x] `train_yolo_symbols.py` — training script with synthetic dataset generation (42 classes)
    - [x] Unified `detect_symbols()` interface with `backend="auto"|"yolo"|"template"`
    - [x] Pipeline parameters: `symbol_backend`, `yolo_model_path`, `yolo_confidence`, `yolo_iou_threshold`
    - [x] `models/` directory with README.md
    - [x] Optional dependencies: `ultralytics>=8.0.0`, `onnxruntime>=1.15.0` via `pip install aec-agent[yolo]`
    - [x] 34 unit tests for YOLO detection
- [x] **Phase 2.5.2: Vision LLM Symbol Classification** (COMPLETE)
    - [x] `vision_llm.py` — VisionLLMClassifier with Gemini/GPT-4o/Claude support
    - [x] `symbol_classifier.py` — Two-stage pipeline (YOLO → Vision LLM for ambiguous symbols)
    - [x] `prompts/symbol_classification.py` — Domain-specific Vision LLM prompts for symbol ID
    - [x] `SmartSymbol` dataclass with subtype, direction, system, specs, associated text
    - [x] SYMBOL_SUBTYPES with 50+ subtypes across 5 MEP categories
    - [x] Updated `symbol_detection.py` with `detect_symbols_smart()` async function
    - [x] Updated `image_vectorizer.py` with Vision LLM integration
    - [x] 89 unit tests for Phase C components
    - [x] All 456 tests passing
- [x] **Phase 2.5.3: Semantic OCR Parsing** (COMPLETE)
    - [x] `semantic_ocr.py` — Pattern-based annotation parsing (20+ types)
    - [x] `text_associator.py` — Text-to-element association
    - [x] `prompts/annotation_parsing.py` — Domain-specific LLM prompts
    - [x] Pipeline integration with `semantic_ocr` and `text_association` parameters
    - [x] 69 unit tests for Phase B components
    - [x] All 367 tests passing
- [x] **Phase D: Geometry Intelligence** (COMPLETE)
    - [x] `geometry_classifier.py` — Pattern-based geometry classification (walls, ducts, pipes)
    - [x] `topology_analyzer.py` — Graph-based system connectivity analysis
    - [x] `GeometryType` enum with 30+ AEC element types
    - [x] `GeometrySystem` enum with 15 MEP systems
    - [x] `TopologyGraph` with BFS/DFS pathfinding and connected components
    - [x] `ClassifiedLine`, `ParallelLinePair`, `TopologyNode`, `TopologyEdge` dataclasses
    - [x] Classification rules: wall thickness detection, duct sizing, pipe run identification
    - [x] Pipeline integration with `geometry_classification` parameter
    - [x] 88 unit tests for Phase D components
    - [x] All tests passing
- [x] **Phase E: Relationship Inference** (COMPLETE)
    - [x] `relationship_builder.py` — Multi-strategy relationship inference engine
    - [x] `RelationType` enum with 20+ relationship types
    - [x] `InferredRelationship` dataclass with confidence scoring
    - [x] Point-in-polygon containment detection (rooms contain equipment)
    - [x] Endpoint proximity connectivity detection (pipes connect to valves)
    - [x] Spatial analysis (room adjacency, symbol proximity)
    - [x] Flow relationship detection from topology (source feeds terminal)
    - [x] Updated `repository.py` with batch relationship creation
    - [x] Pipeline integration with `relationship_inference` parameter
    - [x] 39 unit tests for Phase E components
    - [x] All tests passing
- [x] **Phase F: Knowledge Grounding** (COMPLETE with Gemini LLM)
    - [x] `knowledge_query.py` — CAD standards lookup module (1500+ lines)
    - [x] `CADStandards` dataclass with layer, color, linetype, lineweight, block_name, attributes
    - [x] `Discipline` enum (8 disciplines) and `SystemType` enum (25+ systems)
    - [x] `GroundedElement` dataclass for elements with standards applied
    - [x] Hierarchical lookup: project → company → YAML → built-in defaults
    - [x] Created `knowledge_base/cad_standards/` directory with 6 YAML files
    - [x] 50+ block name patterns (P-VALV-GATE, M-DIFF-SQ, E-OUTL-DUP, etc.)
    - [x] NCS-based layer naming (P-DOMW-VALV, M-HVAC-DIFF, E-POWR-OUTL)
    - [x] Color coding by system (Blue=cold/supply, Red=hot/power/fire)
    - [x] Convenience functions: `get_layer_for_element()`, `get_block_name()`, `get_color_for_system()`
    - [x] Pipeline parameters: `knowledge_grounding`, `project_standards`, `company_standards`
    - [x] **Gemini LLM Integration:**
        - [x] `KnowledgeLLM` class for LLM-powered CAD standards queries
        - [x] `LLMQueryResult` dataclass for structured LLM responses
        - [x] Methods: `query()`, `classify_element()`, `get_code_reference()`, `suggest_attributes()`
        - [x] Auto-fallback provider chain: Gemini → OpenAI → Anthropic
        - [x] `query_with_llm()` convenience function for quick queries
    - [x] 132 unit tests for Phase F components (including 20 LLM tests)
    - [x] All tests passing
- [ ] **Phase 3: Knowledge Base Query Tools** (Pending — MCP tool for codes/standards/formulas)
- [ ] **Phase 4: HVAC Autonomous Design** (Pending)
- [ ] **Phase 5-8: Fire/LV/Electrical/Plumbing** (Pending)
- [ ] **Phase 9: Multi-System Coordination** (Pending)
- [ ] **Phase 10: Intelligence & Learning** (Ongoing)

## 📈 Metrics
| Metric | Value |
|--------|-------|
| Python LOC | ~24,000 |
| MCP Tools | 50 (7 categories) |
| Config Parameters | 48 env vars |
| Test Files | 22 (750+ tests) |
| DB Tables | 15 (projects, elements, relationships, + 12 MEP/domain tables) |
| Sidecar Files | 29 total (15 C#, 14 Python) |
| Knowledge Base YAML Files | 6 (plumbing, mechanical, electrical, fire, low_voltage, architectural) |

## 🧠 Brain Dump (Context for Next Session)
- **Phase 2.5.1 YOLOv8 Symbol Detection is COMPLETE.** Neural network-based symbol detection added as alternative to template matching. Files: `yolo_detection.py`, `train_yolo_symbols.py`, `models/README.md`. Use `symbol_backend="yolo"` or `"auto"` to enable. Supports 42 MEP symbol classes across mechanical, electrical, fire, plumbing, and low_voltage categories. Install with `pip install aec-agent[yolo]`.
- **YOLOv8 Detection Architecture:** `YOLOSymbolDetector` class supports both ultralytics (native .pt) and ONNX Runtime (.onnx) backends. Auto-detects based on file extension. ONNX recommended for deployment (no heavy ultralytics dependency). Includes letterbox preprocessing, NMS, and coordinate conversion to drawing units.
- **Training Script:** `scripts/train_yolo_symbols.py` has 4 commands: `prepare` (synthetic dataset), `train` (YOLOv8), `export` (ONNX), `validate` (test). Synthetic data uses simple geometric shapes (valves=bowtie, outlets=circle+lines, etc.) with augmentation (rotation, scale, noise, blur).
- **Unified Symbol Detection:** `detect_symbols()` in `symbol_detection.py` is the unified entry point. `backend="auto"` (default) uses YOLO if model exists, else template matching. `backend="yolo"` forces YOLO (returns empty if unavailable). `backend="template"` forces template matching.
- **New Parameters in Pipeline:** `symbol_backend`, `yolo_model_path`, `yolo_confidence`, `yolo_iou_threshold` added to `image_vectorizer.py`, `raster_design.py::raster_auto_vectorize()`, and `raster_design.py::raster_pdf_to_vector_pipeline()`.
- **Phase 2 Raster Design is COMPLETE.** 17 MCP tools + 14 sidecar commands for full PDF-to-DWG-to-PostgreSQL pipeline. Vectorizer refactored with lines-first detection, FLD, circle validation, topology cleanup.
- **Raster Design MCP tools (17):** `raster_convert_pdf`, `raster_import_pdf`, `raster_attach_image`, `raster_cleanup`, `raster_vectorize` (VTools: vline/vpline/varc/vcircle/vrect — interactive, for manual use), `raster_auto_vectorize` (**Python OpenCV — automated**), `raster_process_image` (ibfilter), `raster_create_primitive` (issmart/isline/isarc/iscircle), `raster_select_entities` (isebrsmart/isebrcon), `raster_follower` (vfpline/vfcontour/vf3dpoly), `raster_recognize_text` (irectext), `raster_ocr_extract`, `raster_get_status`, `raster_get_entity_count`, `raster_fade_image`, `raster_store_vectorized`, `raster_pdf_to_vector_pipeline`.
- **OpenCV auto-vectorization (CRITICAL):** Raster Design VTools (`vline`, `vpline`, `varc`, `vcircle`, `vrect`) are **interactive** — they require mouse clicks and CANNOT be automated via `SendStringToExecute`. The pipeline uses Python-side OpenCV instead: `image_vectorizer.py` detects lines (FastLineDetector primary + HoughLinesP fallback), circles (HoughCircles with pixel validation), and polylines (findContours + approxPolyDP) from the bitonal TIFF, then creates AutoCAD entities via `draw_line`/`draw_polyline`/`draw_circle` sidecar commands.
- **Vectorizer refactoring (2026-02-02):** Reversed detection order (lines first, mask, then circles). Added FastLineDetector (opencv-contrib) as primary with HoughLinesP fallback. Added `_validate_circles_by_ink()` — samples 36 points around circumference, rejects circles with <35% ink. Tuned: param2 100→200, min_line_length 80→50, max_line_gap 10→15, hough_threshold 150→80. Enabled mask_detected_lines + topology_cleanup by default. Test: Lines 85→1,973 (+2,221%), Circles 205→6 (-97%).
- **Groq tool prefix fix (2026-02-02):** `AppContext.get_tool_prefix()` now returns `"autocad_,raster_"` for AUTOCAD context (was `"autocad_"` only). Without this, `raster_*` tools were filtered out when intent detected AutoCAD, causing Groq 400 error.
- **Coordinate conversion:** pixel → drawing units: `x_dwg = px_x / dpi`, `y_dwg = (height - px_y) / dpi` (Y-axis flip from image top-left to AutoCAD bottom-left origin).
- **PDF-to-bitonal conversion:** `raster_convert_pdf` and `pdf_converter.py` use PyMuPDF + Pillow to render PDF pages at 300 DPI and convert to 1-bit TIFF (Group4 compression). REQUIRED because AutoCAD Raster Design cannot attach PDF files directly.
- **Actual Raster Design commands (CRITICAL):** `IVECTORIZE` and `IOCR` do NOT exist. The actual commands are: VTools (`vline`, `vpline`, `varc`, `vcircle`, `vrect`), Followers (`vfpline`, `vfcontour`), REM Primitives (`isline`, `isarc`, `iscircle`, `issmart`), Image Processing (`ibfilter`), Text Recognition (`irectext`), Entity Selection (`isebrcon`, `isebrsmart`). **ALL are interactive** — the pipeline uses OpenCV instead.
- **Key pipeline tool:** `raster_pdf_to_vector_pipeline` orchestrates: auto-detect PDF type → (scanned: convert to bitonal TIFF → attach TIFF → despeckle → deskew → **OpenCV detect features** → **draw_line/draw_polyline/draw_circle**) / (vector: PDFIMPORT) → fade raster → extract to PostgreSQL with embeddings and relationships.
- **Natural blocking via OnIdle:** Async sidecar commands (SendStringToExecute) process sequentially in OnIdle. The next HTTP request after an async command acts as a natural barrier — no polling needed. Entity count calls serve as barriers in the pipeline.
- **Async command pattern:** `IsAsyncCommand()` on CommandRouter. OnIdle handler uses DocumentLock-only path (no Transaction) for async commands. Async: import_pdf, cleanup, vectorize, ocr. Sync: attach_image, get_status, get_entity_count, extract_all_entities, fade_image.
- **`extract_all_entities`** C# command: full geometry extraction with type-specific handling (Line, Circle, Arc, Polyline, Ellipse, Spline, Text, MText, BlockReference, Hatch). Supports pagination (offset + limit) and layer filter. Returns handle, type, layer, color, linetype, geometry, bounds.
- **Bug fixes in this session:** (1) Fixed scanned PDF branch passing raw PDF to `raster_attach_image` → now converts to bitonal TIFF first. (2) Replaced `IVECTORIZE` (doesn't exist) with actual VTools. (3) Replaced `IOCR` (doesn't exist) with `irectext`. (4) Added 5 new sidecar commands + 5 new MCP tools. (5) **Replaced interactive VTools with OpenCV** — VTools require mouse clicks and can't be automated. Created `image_vectorizer.py` + `raster_auto_vectorize` tool. Pipeline now uses OpenCV detection → AutoCAD draw commands. (6) Fixed Groq 400 error — `AppContext.get_tool_prefix()` missing `raster_` prefix for AUTOCAD context. (7) Vectorizer refactoring — lines-first detection, FLD, circle validation, topology cleanup, tuned thresholds. (8) Fixed Unicode `→` in log messages (cp1252 encoding). (9) Added `autocad_delete_entity` to MINIMAL_DESCRIPTIONS. Previous fixes: `DatabasePool._verify_extensions()` init order, missing `raster_design` import, `sync_cache` 401 error, `stream_entities()` nonexistent `extract_batch`.
- **Phase 1 is COMPLETE.** PostgreSQL 18.1 running with PostGIS + pgvector, 15 tables created, all tests passing.
- **Provider fallback chain** works: Groq (primary, fastest) → Gemini → OpenAI → Anthropic. Configure via `FALLBACK_PROVIDERS` env var.
- **REST endpoints** added alongside MCP SSE: `POST /tools/notify_file_opened` (sidecar hooks), `GET /health`.
- **`get_file_context`** MCP tool returns compact summary from PostgreSQL cache (element counts by type/layer/category). This is the key to 90% token reduction.
- **DATABASE_URL** uses `postgresql://` format (plain asyncpg). Alembic env.py auto-converts to `+asyncpg` for SQLAlchemy.
- `connection.py` strips `+asyncpg` prefix automatically if passed.
- The `PLAN.md` file has the detailed schema and SQL. **Read it for technical details.**
- `FUTURE_ROADMAP.md` has the 10-phase vision, tool gap analysis, and knowledge base architecture.
- Extraction pipeline has debounced sync (2s) and file hash detection to skip unchanged files.
- Tool lock enforces max 1 concurrent operation (STA constraint).
- The `knowledge_base/cad_standards/` directory now contains 6 YAML files for CAD standards (plumbing, mechanical, electrical, fire, low_voltage, architectural).
- MEP domain seed data exists inline in `src/aec_agent/domain/seed_data.py` (HVAC clearance rules).
- **New files (Phase B):** `src/aec_agent/mcp/tools/semantic_ocr.py` (pattern-based annotation parsing), `src/aec_agent/mcp/tools/text_associator.py` (text-element association), `src/aec_agent/prompts/annotation_parsing.py` (LLM prompts), `tests/unit/test_semantic_ocr.py` (44 tests), `tests/unit/test_text_associator.py` (25 tests).
- **Phase B key functions:** `parse_annotation_by_patterns()` — fast regex parsing for CFM, equipment tags, room numbers, sizes, etc. `associate_text_to_elements()` — distance-based association with type-specific rules. `enrich_elements_with_text()` — combine symbols with their associated text data.
- **New VectorizationResult fields (Phase B):** `parsed_annotations` (List[ParsedAnnotation]), `text_associations` (List[TextAssociation]), `enriched_elements` (elements with associated text).
- **New files (Phase C / Phase 2.5.2):** `src/aec_agent/mcp/tools/vision_llm.py` (Vision LLM integration), `src/aec_agent/mcp/tools/symbol_classifier.py` (two-stage classifier), `src/aec_agent/prompts/symbol_classification.py` (Vision prompts), `tests/unit/test_vision_llm.py` (25 tests), `tests/unit/test_symbol_classifier.py` (27 tests), `tests/unit/test_symbol_classification_prompts.py` (37 tests).
- **Phase C key components:** `VisionLLMClassifier` supports Gemini, OpenAI GPT-4o, and Anthropic Claude for symbol image classification. `SmartSymbol` dataclass extends DetectedBlock with subtype, direction, system, specs, and associated text. `SymbolClassifier` orchestrates two-stage pipeline: YOLO for fast detection, Vision LLM for ambiguous/low-confidence symbols.
- **Phase C key functions:** `detect_symbols_smart()` — async wrapper for two-stage detection. `classify_symbols()` — main classifier method. `detect_and_classify_symbols()` — convenience function. Specialized prompts: `get_valve_classification_prompt()`, `get_diffuser_classification_prompt()`, `get_outlet_classification_prompt()`, `get_detector_classification_prompt()`.
- **New VectorizationResult fields (Phase C):** `smart_symbols` (List[SmartSymbol]), `vision_llm_used` (bool).
- **New Pipeline Parameters (Phase C):** `vision_llm_classification` (bool), `vision_llm_provider` ("auto"|"gemini"|"openai"|"anthropic"), `vision_llm_confidence_threshold` (float, default 0.85).
- **SYMBOL_SUBTYPES:** Comprehensive dictionary with 50+ subtypes across 5 MEP categories: mechanical (valves, diffusers, dampers, equipment, fittings), electrical (outlets, switches, lights, panels, devices), fire (detection, notification, suppression, control), plumbing (valves, fixtures, equipment, fittings), low_voltage (data, security, audio_visual, control).
- **New files (Phase 2.5.1):** `src/aec_agent/mcp/tools/yolo_detection.py` (YOLOv8 detector), `scripts/train_yolo_symbols.py` (training script), `models/README.md` (model docs), `tests/unit/test_yolo_detection.py` (34 tests).
- **New files (Phase 2):** `src/aec_agent/mcp/tools/image_vectorizer.py` (OpenCV detection), `src/aec_agent/mcp/tools/pdf_converter.py` (PDF→bitonal TIFF).
- **New dependencies (optional Vision LLM):** `google-generativeai>=0.3.0` (Gemini), `openai>=1.0.0`, `anthropic>=0.8.0` — any one provider sufficient.
- **New dependencies (optional YOLO):** `ultralytics>=8.0.0`, `onnxruntime>=1.15.0` — install via `pip install aec-agent[yolo]`.
- **New files (Phase D):** `src/aec_agent/mcp/tools/geometry_classifier.py` (pattern-based geometry classification), `src/aec_agent/mcp/tools/topology_analyzer.py` (graph-based connectivity), `tests/unit/test_geometry_classifier.py` (48 tests), `tests/unit/test_topology_analyzer.py` (40 tests).
- **Phase D key classes:** `GeometryClassifier` — classifies lines/polylines/circles into AEC elements based on parallel spacing, linetype, and context. `TopologyAnalyzer` — builds graph from classified geometry and symbols for connectivity analysis.
- **Phase D key functions:** `classify_geometry()` — main entry point for geometry classification. `build_system_graph()` — creates TopologyGraph from lines and symbols. `find_connected_equipment()` — traces connectivity between equipment.
- **Phase D classification rules:** Walls detected by parallel lines 3-12" apart. Ducts detected by parallel lines 4-48" apart near diffuser symbols. Pipes detected by single lines connecting valve/fitting symbols. Dimension lines detected by lines with numeric text at midpoint.
- **Phase D topology features:** `TopologyGraph.find_path()` — BFS pathfinding between nodes. `TopologyGraph.find_connected_components()` — identifies isolated system segments. `TopologyAnalyzer.find_system_paths()` — traces from sources to terminals. `TopologyAnalyzer.identify_branches()` — identifies main vs branch runs.
- **New VectorizationResult fields (Phase D):** `classified_geometry` (List[ClassifiedLine]), `geometry_classification_stats` (dict), `parallel_pairs` (List[ParallelLinePair]), `wall_centerlines`, `duct_boundaries`, `pipe_runs`, `system_topology` (TopologyGraph).
- **New Pipeline Parameter (Phase D):** `geometry_classification` (bool, default True) — enables pattern-based geometry classification.
- **New files (Phase E):** `src/aec_agent/mcp/tools/relationship_builder.py` (relationship inference), `tests/unit/test_relationship_builder.py` (39 tests).
- **Phase E key classes:** `RelationshipBuilder` — builds relationships using containment, connectivity, spatial, and flow strategies. `InferredRelationship` — captures source, target, type, confidence, distance, system, and metadata.
- **Phase E relationship types (20+):** CONTAINS, CONTAINED_BY, CONNECTED_TO, BRANCHES_FROM, MERGES_INTO, FEEDS, FED_BY, SUPPLIES, RETURNS_TO, SERVES, SERVED_BY, ADJACENT_TO, NEAR, ABOVE, BELOW, ON_LEVEL, HOSTS, HOSTED_BY, LABELS, LABELED_BY, REFERENCES, REFERENCED_BY.
- **Phase E key functions:** `build_relationships()` — main entry point. `find_elements_in_room()` — point-in-polygon test. `find_connected_chain()` — traces connected symbols through geometry.
- **Phase E algorithms:** Point-in-polygon (ray casting) for containment. Endpoint proximity for connectivity. Edge overlap detection for room adjacency. BFS pathfinding for flow relationships.
- **Repository updates (Phase E):** `create_relationships_batch()` — batch upsert. `get_relationships_by_type()` — filter by type. `get_containment_relationships()` — room-contains queries. `get_connectivity_graph()` — adjacency list. `find_path_between_elements()` — BFS pathfinding.
- **New VectorizationResult fields (Phase E):** `inferred_relationships` (List[InferredRelationship]), `containment_relationships`, `connectivity_relationships`, `relationship_statistics`.
- **New Pipeline Parameter (Phase E):** `relationship_inference` (bool, default True) — enables relationship inference.
- **New dependencies (Phase 2):** `opencv-python-headless>=4.8.0` (or `opencv-contrib-python` for FastLineDetector), `numpy>=1.24.0`, `PyMuPDF>=1.24.0`, `Pillow>=10.0.0`, `networkx>=3.0` (topology cleanup).
- **New files (Phase F):** `src/aec_agent/mcp/tools/knowledge_query.py` (CAD standards lookup + Gemini LLM integration, 1500+ lines), `knowledge_base/cad_standards/plumbing.yaml`, `knowledge_base/cad_standards/mechanical.yaml`, `knowledge_base/cad_standards/electrical.yaml`, `knowledge_base/cad_standards/fire.yaml`, `knowledge_base/cad_standards/low_voltage.yaml`, `knowledge_base/cad_standards/architectural.yaml`, `tests/unit/test_knowledge_query.py` (132 tests).
- **Phase F key classes:** `CADStandards` — layer, color, linetype, lineweight, block_name, attributes. `Discipline` — 8 AEC disciplines (architectural, mechanical, electrical, plumbing, fire, low_voltage, civil, structural). `SystemType` — 25+ MEP system types (domestic_cold_water, supply_air, fire_alarm, data, security, etc.). `GroundedElement` — element with standards applied.
- **Phase F key functions:** `query_cad_standards()` — async hierarchical lookup (project → company → YAML → defaults). `ground_element()` — apply standards to single element. `ground_elements()` — batch grounding with statistics. `get_layer_for_element()`, `get_block_name()`, `get_color_for_system()` — convenience functions. `query_with_llm()` — LLM-powered natural language query for CAD standards.
- **Phase F Gemini LLM integration:** `KnowledgeLLM` class supports Gemini (primary), OpenAI, and Anthropic providers with auto-fallback. Methods include `query()` (natural language CAD standards questions), `classify_element()` (element classification from text description), `get_code_reference()` (building code references for elements), and `suggest_attributes()` (attribute suggestions for blocks). System prompt includes NCS layer naming conventions, MEP standards, building codes (CMC, CEC, CPC, CFC, NFPA 72), and color coding rules. Uses `gemini-1.5-flash` model for cost-effective queries.
- **Phase F block name patterns (50+):** Valves (P-VALV-GATE, P-VALV-BALL, P-VALV-BTRF), Diffusers (M-DIFF-SQ, M-DIFF-RD, M-DIFF-LN), Outlets (E-OUTL-DUP, E-OUTL-GFCI), Fire (F-DETC-SMOK, F-ANUN-HS, F-SPKL-PND), Low Voltage (T-DATA-RJ45, T-SECU-CAM-D).
- **Phase F layer naming (NCS-based):** Format is `{Discipline}-{Major}-{Minor}-{Suffix}`. Examples: P-DOMW-VALV (plumbing domestic water valve), M-HVAC-DIFF (mechanical HVAC diffuser), E-POWR-OUTL (electrical power outlet), F-ALRM-DETC (fire alarm detector).
- **Phase F color coding:** Blue(5)=cold water/supply air/data, Red(1)=hot water/power/fire, Cyan(4)=return air/storm, Green(3)=sanitary/voice, Magenta(6)=vent/exhaust/AV, Yellow(2)=gas/lighting.
- **New VectorizationResult fields (Phase F):** `grounded_elements` (List[GroundedElement]), `knowledge_grounding_stats` (dict with applied/missing/coverage).
- **New Pipeline Parameters (Phase F):** `knowledge_grounding` (bool, default True), `project_standards` (dict), `company_standards` (dict).
- **New dependency (optional YAML):** `PyYAML>=6.0` — for loading custom standards from YAML files.
- **Semantic Intelligence Pipeline COMPLETE.** All 6 phases (A-F) now implemented. Full pipeline transforms raw geometry into semantically-rich AEC objects with proper layers, blocks, attributes, and relationships.
- **CURRENT PRIORITY: Gemini-First Pipeline.** This is a new approach to PDF vectorization that puts Gemini Vision FIRST (before preprocessing) to understand drawing content before extraction. Addresses noise issues in the current bitonal pipeline. Phases 1-5 are COMPLETE (PDF Intake → Gemini Understanding → Coordinate Calibration → Adaptive Extraction → AutoCAD Entity Creation). Phase 6 (Validation & Self-Correction) is NEXT.
- **Gemini-First Phase 1 (COMPLETE):** High-quality PDF rendering without bitonal conversion. Files: `src/aec_agent/mcp/tools/gemini_first/pdf_intake.py`. MCP tools: `gemini_render_pdf`, `gemini_get_pdf_info`, `gemini_render_all_pages`, `gemini_compare_rendering_quality`. Key difference: preserves grayscale/color for AI analysis.
- **Gemini-First Phase 2 (COMPLETE):** Gemini Vision drawing analysis. Files: `src/aec_agent/mcp/tools/gemini_first/gemini_understanding.py`. MCP tools: `gemini_analyze_drawing`, `gemini_analyze_pdf`, `gemini_get_extraction_strategy`. Key outputs: `DrawingAnalysis` dataclass with elements, regions, calibration hints, and recommended extraction strategy. 38 unit tests.
- **Gemini-First Phase 3 (COMPLETE):** Coordinate calibration. Files: `src/aec_agent/mcp/tools/gemini_first/coordinate_calibration.py`. MCP tools: `gemini_calibrate_coordinates`, `gemini_calibrate_manual`, `gemini_convert_coordinates`, `gemini_parse_scale`, `gemini_parse_measurement`, `gemini_analyze_pdf_calibrated`. Key outputs: `ScaleCalibration` dataclass with `to_dwg()`, `to_pixels()`, `scale_length()`, `convert_units_to()` methods. Calibration method priority: dimension (90%) > scale notation (85%) > sheet size (70%) > DPI default (30%). Supports imperial (20'-6", 24") and metric (100mm, 1.5m) measurements. Standard sheet sizes: ARCH A-E, ANSI A-E, ISO A0-A4. 71 unit tests.
- **Gemini-First Phase 4 (COMPLETE):** Adaptive extraction. Files: `src/aec_agent/mcp/tools/gemini_first/adaptive_extraction.py`. MCP tools: `gemini_extract_entities`, `gemini_extract_pdf_entities`, `gemini_get_layer_mapping`, `gemini_get_block_mapping`, `gemini_get_required_layers`, `gemini_get_required_blocks`. Key outputs: `EntityToCreate`, `RasterCommand`, `ExtractionResult` dataclasses. Three extraction strategies: Direct (Gemini coordinates → entities), Guided rasterization (VTool commands for complex regions), Selective OpenCV (HoughLinesP/HoughCircles for patterns). NCS-compliant layer mapping: `ELEMENT_TYPE_TO_LAYER` (walls, ducts, outlets, etc.), `SYMBOL_TO_BLOCK` (50+ MEP symbols). Helper functions: `get_layer_for_element_type()`, `get_block_name()`, `get_entities_by_type()`, `get_required_layers()`. 58 unit tests.
- **Gemini-First Phase 5 (COMPLETE):** AutoCAD entity creation. Files: `src/aec_agent/mcp/tools/gemini_first/autocad_creation.py`. MCP tools: `gemini_create_entities` (extract + create), `gemini_vectorize_pdf` (complete PDF-to-AutoCAD pipeline), `gemini_create_from_extraction` (create from saved JSON). Key outputs: `AutoCADCreationResult`, `EntityCreationResult`, `LayerCreationResult`, `CreationStatistics` dataclasses. Core function: `create_entities_in_autocad()` takes ExtractionResult from Phase 4 and creates entities via sidecar commands (draw_line, draw_arc, draw_circle, draw_mtext, insert_block). Entity creators: `create_line_entity()`, `create_arc_entity()`, `create_circle_entity()`, `create_text_entity()`, `create_block_entity()`, `create_polyline_entity()`. Automatic layer creation with NCS-compliant colors: `LAYER_PREFIX_COLORS` (A-=7/white, M-=4/cyan, E-=1/red, P-=5/blue, F-=1/red, T-=3/green). Helper functions: `get_color_for_layer()`, `get_entity_type_stats()`, `get_failed_by_type()`. 55 unit tests.
- **Gemini-First architecture docs:** `docs/autocad-rasterization-architecture.md` (full architecture), `docs/GEMINI-FIRST-PHASES.md` (phase breakdown with code snippets).
- **Next priority (Phase 6):** Implement validation & self-correction — Gemini verifies the created entities against the original PDF and identifies/corrects errors.
- **Future priority:** Phase 3 (Knowledge Base Query Tools) from FUTURE_ROADMAP.md — building `query_knowledge_base` MCP tool for codes, standards, and engineering formulas.

## 📂 Key Files to Read First
1. `docs/SESSION_CONTEXT.md` (This file)
2. `CLAUDE.md` (Project rules — compact)
3. `docs/PLAN.md` (Technical spec for DB migration)
4. `docs/FUTURE_ROADMAP.md` (Long-term 10-phase vision)
