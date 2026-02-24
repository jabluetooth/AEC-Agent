# Session Context
> **DO NOT DELETE**. Maintains continuity between AI coding sessions.

## Current Focus
**Objective:** Phase 3 — Knowledge Base Query Tools
**Status:** Gemini-First Pipeline COMPLETE (6 phases) ✅

## Next Steps
1. Build `query_knowledge_base` MCP tool for codes/standards/formulas
2. End-to-end test with real PDF + running sidecar
3. Phase 4: Element placement tools (`place_revit_family`, `place_autocad_block`)

## Quick Reference
| Area | Key File/Directory |
|------|-------------------|
| Gemini-First pipeline | `src/aec_agent/mcp/tools/gemini_first/` |
| **Hybrid extraction** | `adaptive_extraction.py` → `hybrid_extract_all()` |
| **OpenCV utilities** | `opencv_extraction.py` → `OpenCVExtractor` |
| **Phase B: VTracer** | `vtracer_extraction.py` → O(n) vectorization |
| **Phase B: Super-Resolution** | `super_resolution.py` → Real-ESRGAN upscaling |
| **Phase B: Text/Graphics** | `text_graphics_separation.py` → Fletcher-Kasturi |
| YOLO detection | `src/aec_agent/mcp/tools/yolo_detection.py` |
| CAD standards YAML | `knowledge_base/cad_standards/*.yaml` |
| Main MCP tools | `gemini_complete_pipeline`, `gemini_vectorize_pdf` |
| Architecture docs | `docs/autocad-rasterization-architecture.md` |
| Phase breakdown | `docs/GEMINI-FIRST-PHASES.md` |
| Long-term roadmap | `docs/FUTURE_ROADMAP.md` |

## Repository Metrics
| Metric | Value |
|--------|-------|
| Python LOC | ~24,000 |
| MCP Tools | 50 (7 categories) |
| Tests | 750+ passing |
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

## Key Patterns
- **Tool results:** Return `{"success": False, "error": {...}}` — never raise exceptions
- **Sidecar calls:** `httpx.Timeout(120.0, connect=5.0)` + retry decorator
- **Coordinate conversion:** `x_dwg = px_x / dpi`, `y_dwg = (height - px_y) / dpi`
