# AI Session Context
> **DO NOT DELETE**. This file maintains the continuity of work between AI coding sessions.

## 🟢 Current Focus
**Objective:** Phase 2 Raster Design — COMPLETE (full Raster Design toolset + OpenCV auto-vectorization).
**Last Action:** Replaced interactive Raster Design VTools with Python-side OpenCV vectorization. VTools (`vline`, `vpline`, etc.) require mouse clicks and cannot be automated via `SendStringToExecute`. Created `image_vectorizer.py` (OpenCV: HoughLinesP, HoughCircles, findContours) and `raster_auto_vectorize` MCP tool. Updated pipeline: PDF→bitonal TIFF→attach→despeckle→deskew→**OpenCV detect features**→**draw_line/draw_polyline/draw_circle**→fade→store. All Python compiles clean.
**Next Step:** End-to-end test with real PDF + running sidecar, then Phase 3 (Knowledge Base).

## 📊 Repository Status (as of 2026-01-28)

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
| **Test Suite** (179 tests, pytest markers) | `tests/` | Passing |

### What's NOT Yet Done
| Component | Blocker | Roadmap Phase |
|-----------|---------|---------------|
| **End-to-end extraction test** (real CAD file → DB) | Needs running sidecar | Phase 1 (integration) |
| **Knowledge base files** (codes, standards, formulas) | Content creation | Phase 3 |
| **Raster Design integration** (PDF → DWG) | Full pipeline built w/ bitonal conversion, needs e2e test with real PDF | Phase 2 (complete, needs testing) |
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
- [ ] **Phase 3: Knowledge Base** (Pending — `knowledge_base/` dir not yet created)
- [ ] **Phase 4: HVAC Autonomous Design** (Pending)
- [ ] **Phase 5-8: Fire/LV/Electrical/Plumbing** (Pending)
- [ ] **Phase 9: Multi-System Coordination** (Pending)
- [ ] **Phase 10: Intelligence & Learning** (Ongoing)

## 📈 Metrics
| Metric | Value |
|--------|-------|
| Python LOC | ~16,000 |
| MCP Tools | 50 (7 categories) |
| Config Parameters | 42 env vars |
| Test Files | 11 (179 tests) |
| DB Tables | 15 (projects, elements, relationships, + 12 MEP/domain tables) |
| Sidecar Files | 29 total (15 C#, 14 Python) |

## 🧠 Brain Dump (Context for Next Session)
- **Phase 2 Raster Design is COMPLETE.** 17 MCP tools + 14 sidecar commands for full PDF-to-DWG-to-PostgreSQL pipeline.
- **Raster Design MCP tools (17):** `raster_convert_pdf`, `raster_import_pdf`, `raster_attach_image`, `raster_cleanup`, `raster_vectorize` (VTools: vline/vpline/varc/vcircle/vrect — interactive, for manual use), `raster_auto_vectorize` (**Python OpenCV — automated**), `raster_process_image` (ibfilter), `raster_create_primitive` (issmart/isline/isarc/iscircle), `raster_select_entities` (isebrsmart/isebrcon), `raster_follower` (vfpline/vfcontour/vf3dpoly), `raster_recognize_text` (irectext), `raster_ocr_extract`, `raster_get_status`, `raster_get_entity_count`, `raster_fade_image`, `raster_store_vectorized`, `raster_pdf_to_vector_pipeline`.
- **OpenCV auto-vectorization (CRITICAL):** Raster Design VTools (`vline`, `vpline`, `varc`, `vcircle`, `vrect`) are **interactive** — they require mouse clicks and CANNOT be automated via `SendStringToExecute`. The pipeline uses Python-side OpenCV instead: `image_vectorizer.py` detects lines (HoughLinesP), circles (HoughCircles), and polylines (findContours + approxPolyDP) from the bitonal TIFF, then creates AutoCAD entities via `draw_line`/`draw_polyline`/`draw_circle` sidecar commands.
- **Coordinate conversion:** pixel → drawing units: `x_dwg = px_x / dpi`, `y_dwg = (height - px_y) / dpi` (Y-axis flip from image top-left to AutoCAD bottom-left origin).
- **PDF-to-bitonal conversion:** `raster_convert_pdf` and `pdf_converter.py` use PyMuPDF + Pillow to render PDF pages at 300 DPI and convert to 1-bit TIFF (Group4 compression). REQUIRED because AutoCAD Raster Design cannot attach PDF files directly.
- **Actual Raster Design commands (CRITICAL):** `IVECTORIZE` and `IOCR` do NOT exist. The actual commands are: VTools (`vline`, `vpline`, `varc`, `vcircle`, `vrect`), Followers (`vfpline`, `vfcontour`), REM Primitives (`isline`, `isarc`, `iscircle`, `issmart`), Image Processing (`ibfilter`), Text Recognition (`irectext`), Entity Selection (`isebrcon`, `isebrsmart`). **ALL are interactive** — the pipeline uses OpenCV instead.
- **Key pipeline tool:** `raster_pdf_to_vector_pipeline` orchestrates: auto-detect PDF type → (scanned: convert to bitonal TIFF → attach TIFF → despeckle → deskew → **OpenCV detect features** → **draw_line/draw_polyline/draw_circle**) / (vector: PDFIMPORT) → fade raster → extract to PostgreSQL with embeddings and relationships.
- **Natural blocking via OnIdle:** Async sidecar commands (SendStringToExecute) process sequentially in OnIdle. The next HTTP request after an async command acts as a natural barrier — no polling needed. Entity count calls serve as barriers in the pipeline.
- **Async command pattern:** `IsAsyncCommand()` on CommandRouter. OnIdle handler uses DocumentLock-only path (no Transaction) for async commands. Async: import_pdf, cleanup, vectorize, ocr. Sync: attach_image, get_status, get_entity_count, extract_all_entities, fade_image.
- **`extract_all_entities`** C# command: full geometry extraction with type-specific handling (Line, Circle, Arc, Polyline, Ellipse, Spline, Text, MText, BlockReference, Hatch). Supports pagination (offset + limit) and layer filter. Returns handle, type, layer, color, linetype, geometry, bounds.
- **Bug fixes in this session:** (1) Fixed scanned PDF branch passing raw PDF to `raster_attach_image` → now converts to bitonal TIFF first. (2) Replaced `IVECTORIZE` (doesn't exist) with actual VTools. (3) Replaced `IOCR` (doesn't exist) with `irectext`. (4) Added 5 new sidecar commands + 5 new MCP tools. (5) **Replaced interactive VTools with OpenCV** — VTools require mouse clicks and can't be automated. Created `image_vectorizer.py` + `raster_auto_vectorize` tool. Pipeline now uses OpenCV detection → AutoCAD draw commands. Previous fixes: `DatabasePool._verify_extensions()` init order, missing `raster_design` import, `sync_cache` 401 error, `stream_entities()` nonexistent `extract_batch`.
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
- The `knowledge_base/` directory structure is planned in FUTURE_ROADMAP.md but not yet created.
- MEP domain seed data exists inline in `src/aec_agent/domain/seed_data.py` (HVAC clearance rules).
- **New files:** `src/aec_agent/mcp/tools/image_vectorizer.py` (OpenCV detection), `src/aec_agent/mcp/tools/pdf_converter.py` (PDF→bitonal TIFF).
- **New dependencies:** `opencv-python-headless>=4.8.0`, `numpy>=1.24.0`, `PyMuPDF>=1.24.0`, `Pillow>=10.0.0`.
- **Next priority:** End-to-end integration test with real PDF + running sidecar, then Phase 3 (Knowledge Base).

## 📂 Key Files to Read First
1. `docs/SESSION_CONTEXT.md` (This file)
2. `CLAUDE.md` (Project rules — compact)
3. `docs/PLAN.md` (Technical spec for DB migration)
4. `docs/FUTURE_ROADMAP.md` (Long-term 10-phase vision)
