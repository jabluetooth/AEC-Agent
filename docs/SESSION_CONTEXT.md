# Session Context
> **DO NOT DELETE**. Maintains continuity between AI coding sessions.

## Current Focus
**Objective:** Phase 3 — Knowledge Base Query Tools
**Status:** Best Practices Pipeline COMPLETE ✅

## Next Steps
1. Run database migration: `alembic upgrade head` (adds symbol_library table)
2. Seed symbol library: `python -m aec_agent.mcp.tools.gemini_first.symbol_library_seed`
3. Build `query_knowledge_base` MCP tool for codes/standards/formulas
4. End-to-end test with real PDF + running sidecar
5. Phase 4: Element placement tools (`place_revit_family`, `place_autocad_block`)

## Quick Reference
| Area | Key File/Directory |
|------|-------------------|
| Gemini-First pipeline | `src/aec_agent/mcp/tools/gemini_first/` |
| **Hybrid extraction** | `adaptive_extraction.py` → `hybrid_extract_all()` |
| **OpenCV utilities** | `opencv_extraction.py` → `OpenCVExtractor` |
| **Phase B: VTracer** | `vtracer_extraction.py` → O(n) vectorization |
| **Phase B: Super-Resolution** | `super_resolution.py` → Real-ESRGAN upscaling |
| **Phase B: Text/Graphics** | `text_graphics_separation.py` → Fletcher-Kasturi |
| **Phase D: Symbol RAG** | `symbol_rag.py` → CLIP embeddings + pgvector |
| **Symbol Library Seed** | `symbol_library_seed.py` → 80+ CAD symbols |
| **Best Practices Pipeline** | `best_practices_pipeline.py` → 7-stage orchestrator |
| YOLO detection | `src/aec_agent/mcp/tools/yolo_detection.py` |
| CAD standards YAML | `knowledge_base/cad_standards/*.yaml` |
| Main MCP tools | `run_best_practices_vectorization`, `gemini_complete_pipeline` |
| Architecture docs | `docs/autocad-rasterization-architecture.md` |
| Phase breakdown | `docs/GEMINI-FIRST-PHASES.md` |
| Long-term roadmap | `docs/FUTURE_ROADMAP.md` |

## Repository Metrics
| Metric | Value |
|--------|-------|
| Python LOC | ~25,000 |
| MCP Tools | 51 (7 categories) |
| Tests | 780+ passing |
| DB Tables | 15 |

## Architecture Summary
**4-layer stack:** Chainlit UI (8000) → FastMCP Server (54321) → Sidecars (AutoCAD .NET / Revit pyRevit) → PostgreSQL (PostGIS + pgvector)

**Critical:** CAD apps are STA — all API calls marshal to main thread via queues.

## Completed Work (Details in `docs/COMPLETED_PHASES.md`)
- [x] Core Infrastructure (MCP server, sidecars, frontend, config)
- [x] Phase 1: DB Foundation (PostgreSQL 18.1, PostGIS, pgvector, 15 tables)
- [x] Phase 2: Raster Design (17 MCP tools, OpenCV vectorization)
- [x] Phase 2.5: YOLOv8 + Vision LLM symbol detection
- [x] Semantic Intelligence Pipeline (Phases A-F complete)
- [x] Gemini-First Pipeline (Phases 1-6 complete, 277 unit tests)
- [x] **Hybrid Extraction Pipeline** (Gemini + OpenCV + YOLO fusion, 28 tests)
- [x] **Gemini Refinement Pass** (snap to grid, connect endpoints, align parallel)
- [x] **Line Type Detection** (continuous, dashed, dotted, center line detection)
- [x] **OCR Text Anchoring** (Tesseract OCR for pixel-accurate text positions)
- [x] **Arc/Fillet Detection** (OpenCV extract_arcs for partial circles, fillets)
- [x] **PostgreSQL Storage** (Hybrid extraction stores semantic data for queries)
- [x] **Line Straightening** (5° tolerance for aggressive alignment to H/V/45°)
- [x] **Line Weight Support** (thickness from OpenCV passed through to AutoCAD)
- [x] **Circle Detection Tuning** (param2=70 to reduce false positives)
- [x] **Phase B: VTracer Integration** (O(n) vectorizer, optional alternative to OpenCV)
- [x] **Phase B: Real-ESRGAN Super-Resolution** (4x upscaling for low-DPI PDFs, GPU+CPU)
- [x] **Phase B: Fletcher-Kasturi Separation** (text/graphics separation via connected components)
- [x] **Repository Cleanup** (removed dead code, outdated docs, consolidated files)
- [x] **Phase D: RAG Symbol Recognition** (CLIP embeddings + pgvector for symbol lookup)
  - `symbol_library` table with 512-dim CLIP embeddings
  - 80+ standard CAD symbols (electrical, mechanical, plumbing, fire, architectural)
  - MCP tools: `symbol_recognize`, `symbol_search`, `symbol_library_stats`, `symbol_library_seed`
  - 34 unit tests passing
- [x] **Best Practices Pipeline** (7-stage PDF to AutoCAD orchestrator with optimal algorithms)
  - Stage 1: PDF Rendering (PyMuPDF @ 300-600 DPI + Real-ESRGAN super-resolution)
  - Stage 2: Preprocessing (NLM denoise + Hough deskew + 7-method ensemble binarization)
  - Stage 3: Gemini Analysis (scale, type, MText, layers, element detection)
  - Stage 4: Vector Extraction (LSD lines + Hough circles + HAWP junctions + VTracer)
  - Stage 5: Symbol Recognition (CLIP embeddings + pgvector RAG lookup)
  - Stage 6: Validation (5° line straightening + endpoint connection + Gemini QA)
  - Stage 7: Output (scaled entities ready for AutoCAD)
  - MCP tool: `run_best_practices_vectorization`
  - 30 unit tests passing

## Key Patterns
- **Tool results:** Return `{"success": False, "error": {...}}` — never raise exceptions
- **Sidecar calls:** `httpx.Timeout(120.0, connect=5.0)` + retry decorator
- **Coordinate conversion:** `x_dwg = px_x / dpi`, `y_dwg = (height - px_y) / dpi`
