# Completed Phases Archive
> Historical implementation details. Reference only — not read every session.

## Gemini-First Pipeline (Phases 1-6) — Completed 2026-02-16

### Phase 6: Validation & Self-Correction
- File: `src/aec_agent/mcp/tools/gemini_first/validation.py` (~850 lines)
- MCP tools: `gemini_validate_extraction`, `gemini_complete_pipeline`
- Dataclasses: `ValidationResult`, `ValidationIssue`, `Correction`, `CorrectionResult`
- Enums: `ValidationStatus`, `IssueType` (8 types), `IssueSeverity`, `CorrectionAction`
- 35 unit tests

### Phase 5: AutoCAD Entity Creation
- File: `src/aec_agent/mcp/tools/gemini_first/autocad_creation.py` (~900 lines)
- MCP tools: `gemini_create_entities`, `gemini_vectorize_pdf`, `gemini_create_from_extraction`
- Entity creators: line, arc, circle, text, block, polyline
- NCS-compliant layer colors: `LAYER_PREFIX_COLORS`
- 55 unit tests

### Phase 4: Adaptive Extraction
- File: `src/aec_agent/mcp/tools/gemini_first/adaptive_extraction.py` (~750 lines)
- Three strategies: Direct, Guided Rasterization, Selective OpenCV
- Dataclasses: `EntityToCreate`, `RasterCommand`, `ExtractionResult`
- 58 unit tests

### Phase 3: Coordinate Calibration
- File: `src/aec_agent/mcp/tools/gemini_first/coordinate_calibration.py` (~600 lines)
- Calibration methods: dimension (90%) > scale notation (85%) > sheet size (70%) > DPI default (30%)
- Dataclass: `ScaleCalibration` with `to_dwg()`, `to_pixels()`, `scale_length()`, `convert_units_to()`
- 71 unit tests

### Phase 2: Gemini Understanding
- File: `src/aec_agent/mcp/tools/gemini_first/gemini_understanding.py` (~550 lines)
- MCP tools: `gemini_analyze_drawing`, `gemini_analyze_pdf`, `gemini_get_extraction_strategy`
- Output: `DrawingAnalysis` with elements, regions, calibration hints
- 38 unit tests

### Phase 1: PDF Intake
- File: `src/aec_agent/mcp/tools/gemini_first/pdf_intake.py`
- MCP tools: `gemini_render_pdf`, `gemini_get_pdf_info`, `gemini_render_all_pages`, `gemini_compare_rendering_quality`
- Key: Preserves grayscale/color (no bitonal conversion)
- 20 unit tests

---

## Semantic Intelligence Pipeline (Phases A-F) — Completed 2026-02-12

### Phase F: Knowledge Grounding
- File: `src/aec_agent/mcp/tools/knowledge_query.py` (1500+ lines)
- `CADStandards`, `Discipline`, `SystemType`, `GroundedElement` dataclasses
- `KnowledgeLLM` class with Gemini/OpenAI/Anthropic fallback
- 6 YAML files in `knowledge_base/cad_standards/`
- 132 unit tests

### Phase E: Relationship Inference
- File: `src/aec_agent/mcp/tools/relationship_builder.py`
- `RelationType` enum (20+ types), `InferredRelationship` dataclass
- Strategies: containment, connectivity, spatial, flow
- 39 unit tests

### Phase D: Geometry Intelligence
- Files: `geometry_classifier.py`, `topology_analyzer.py`
- `GeometryType` (30+ types), `GeometrySystem` (15 systems)
- `TopologyGraph` with BFS/DFS pathfinding
- 88 unit tests

### Phase C: Vision LLM Symbol Classification
- Files: `vision_llm.py`, `symbol_classifier.py`
- Two-stage: YOLO detection → Vision LLM classification
- `SmartSymbol` dataclass, 50+ SYMBOL_SUBTYPES
- 89 unit tests

### Phase B: Semantic OCR
- Files: `semantic_ocr.py`, `text_associator.py`
- `AnnotationType` enum (20 types), pattern-based parsing
- 69 unit tests

### Phase A: Document Classification
- Files: `document_classifier.py`, `region_segmenter.py`
- 14 drawing types, title block/legend/drawing area detection

---

## Phase 2: Raster Design — Completed 2026-02-02

### MCP Tools (17)
`raster_convert_pdf`, `raster_import_pdf`, `raster_attach_image`, `raster_cleanup`, `raster_vectorize`, `raster_auto_vectorize`, `raster_process_image`, `raster_create_primitive`, `raster_select_entities`, `raster_follower`, `raster_recognize_text`, `raster_ocr_extract`, `raster_get_status`, `raster_get_entity_count`, `raster_fade_image`, `raster_store_vectorized`, `raster_pdf_to_vector_pipeline`

### Key Implementation Notes
- **OpenCV auto-vectorization:** VTools are interactive (require mouse clicks). Pipeline uses Python-side OpenCV: `image_vectorizer.py` with HoughLinesP, HoughCircles, findContours
- **PDF-to-bitonal:** `pdf_converter.py` uses PyMuPDF + Pillow (300 DPI, Group4 TIFF)
- **Vectorizer tuning:** Lines-first detection, FastLineDetector, circle validation (36-point circumference sampling, >35% ink required)

---

## Phase 2.5.1: YOLOv8 Symbol Detection — Completed 2026-02-10

- Files: `yolo_detection.py`, `scripts/train_yolo_symbols.py`
- 42 MEP symbol classes
- Backends: ultralytics (.pt) or ONNX Runtime (.onnx)
- 34 unit tests

---

## Phase 1: DB Foundation — Completed 2026-02-01

- PostgreSQL 18.1 with PostGIS 3.6.1 + pgvector 0.8.1
- 15 tables via Alembic migrations
- LLM fallback chain: Groq → Gemini → OpenAI → Anthropic
- `get_file_context` tool for 90% token reduction
