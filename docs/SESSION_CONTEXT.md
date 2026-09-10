# Session Context
> **DO NOT DELETE**. Maintains continuity between AI coding sessions.

## Current Focus
**Objective:** Phase 3 — Knowledge Base Query Tools
**Status:** Knowledge Base MCP Tools COMPLETE ✅

## Next Steps
1. End-to-end test with real PDF + running sidecar
2. Phase 4: Element placement tools (`place_revit_family`, `place_autocad_block`)
3. Phase 4: Engineering calculations (`calculate_hvac_load`, `calculate_circuit`)

## Recently Completed (2026-09-10, round 3)
- ✅ **Implemented the safe server-side half of `docs/MEP_INTEGRATION_TODO.md`** (Tasks 4, 5, 12, 10):
  - `src/aec_agent/mcp/tools/workflow_tools.py` — `list_workflows`, `start_workflow` (wraps `WorkflowExecutor`)
  - `src/aec_agent/mcp/tools/validation_tools.py` — `validate_elements`, `get_suggestions` (wraps `RulesEngine`)
  - `src/aec_agent/mcp/tools/memory_tools.py` — `store_fact`, `recall_facts` (wraps `ProjectMemory`)
  - `aec-agent seed-rules` CLI command (`cli.py`) — loads default HVAC rules into `domain_rules` via `MEPKnowledgeBase.add_rule()` (idempotent, `ON CONFLICT DO UPDATE`)
  - All 6 new tools verified against actual method signatures in `rules_engine.py`/`executor.py`/`project_memory.py` (not just the TODO doc's snippets, which had at least one wrong function name: `get_hvac_seed_rules` → actually `get_default_hvac_rules`)
  - Registered in both `mcp/tools/__init__.py` and `src/aec_agent/server.py`'s explicit import list
  - `docs/MEP_INTEGRATION_TODO.md` annotated with the architecture correction and done/not-done status
- ⏸️ **Not done** (needs frontend redesign, see MEP_INTEGRATION_TODO.md's 2026-09-10 correction note): Tasks 1, 2, 3, 6, 7, 8, 9, 11 — these require a new imperative MCP tool call from `agent.py`/`app.py` rather than direct DB object instantiation in the frontend process.
- No Python interpreter was available to import-test these new files — recommend `python -c "import aec_agent.server"` on a machine with the environment before relying on them.

## Recently Completed (2026-09-10, round 2)
- ✅ **Fixed `document_classifier.py`** (`_call_groq`/`_call_gemini`/`_call_openai`): now read API keys/model names from `settings.py` (`groq_api_key`, `groq_model`, `gemini_api_key`, `gemini_model`, `openai_api_key`) instead of raw `os.environ.get(...)` + hardcoded model strings, per CLAUDE.md's "all config via Pydantic Settings" rule. Verified `settings.py` has no `env_prefix`/`case_sensitive=True` that would change env var mapping — zero behavior change. Preserved the original `GOOGLE_API_KEY` fallback for Gemini exactly.
- ✅ **Added error logging to 7 silent `except Exception:` blocks** in `raster_design.py` (now `raster_design_pipeline.py`'s AutoCAD batch entity-creation loop) — purely additive, no control-flow change.
- ✅ **Split `raster_design.py`** (was 2,042 lines, 18 tools) into 7 files: `raster_design_import.py`, `raster_design_cleanup.py`, `raster_design_vectorize.py`, `raster_design_pipeline.py`, `raster_design_ocr.py`, `raster_design_entities.py`, `raster_design_status.py`. Fixed a real breakage found mid-split: `src/aec_agent/server.py` had its own direct import list (`from aec_agent.mcp.tools import (..., raster_design, ...)`) separate from `mcp/tools/__init__.py` — updated to import the 7 new modules instead.
- ✅ **Extracted 3 classes from `unified_pipeline.py`** (1,910 → 1,382 lines): `GeometryRefinementPipeline` → `unified_pipeline_refinement.py`, `ExtractionFactory` → `unified_pipeline_factory.py`, `DebugVisualizer` → `unified_pipeline_visualizer.py`. `UnifiedPipeline` class itself (~1,050 lines, deep internal coupling) deliberately left untouched. `gemini_first/__init__.py` updated.
- ⚠️ **Correction to a subagent audit claim**: the 4 orphaned domain/memory modules were mis-cited as documented in `docs/AEC_AGENT_OVERVIEW.md` — the actual integration plan is `docs/MEP_INTEGRATION_TODO.md` (14 tasks). Verified the doc's factory-function signatures (`get_project_memory`, `get_user_preferences`, `get_rules_engine`, `get_knowledge_base`, `get_workflow_executor`) all match current code exactly.
- 🔴 **Found a real architecture gap in `MEP_INTEGRATION_TODO.md`**: it assumes `frontend/agent.py` (Chainlit UI process) can directly hold a `db_pool` and instantiate `ProjectMemory`/`RulesEngine`/etc. in-process. In reality, **only the FastMCP server process (`mcp/server.py`) ever calls `initialize_database_pool()`** — the frontend has zero direct DB access today and only reaches the system via `self.mcp_client.call_tool(...)` over SSE. Tasks 4/5/12/10 (new `workflow_tools.py`/`validation_tools.py`/`memory_tools.py` MCP tool files + HVAC seed data) are architecturally correct as documented and are the next planned step. Tasks 1/2/3/6/7/8/9/11 (frontend-side wiring) need redesigning around a new imperative MCP tool call (e.g. `get_session_context`) rather than direct object instantiation — not yet implemented, by user's choice to do the safe half first.
- User decision recorded: wire the 4 orphaned domain/memory modules in (not delete, not leave parked) — see architecture note above for the corrected approach.

## Recently Completed (2026-09-10)
- ✅ **Repo audit** of `gemini_first/` pipeline (32 files, ~32,900 lines): most modules are alive via lazy imports inside `@mcp.tool()` functions (not dead pipeline debris, contrary to first appearance from commit history); `polyline_builder.py` and `ellipse_detection.py` confirmed unwired but NOT dead — they're unchecked planned deliverables in `docs/SCAN2CAD_PARITY_PLAN.md` (sections B.3/C.3), fully built + unit-tested, awaiting integration decision.
- ✅ **Split `mcp_tools.py`** (was 4,660 lines, 47 `@mcp.tool()` registrations) into 11 focused sibling files: `mcp_tools_helpers.py`, `mcp_tools_rendering.py`, `mcp_tools_analysis.py`, `mcp_tools_calibration.py`, `mcp_tools_extraction.py`, `mcp_tools_creation.py`, `mcp_tools_hybrid.py`, `mcp_tools_symbol.py`, `mcp_tools_pipeline.py`, `mcp_tools_export.py`, `mcp_tools_knowledge.py`. Pure structural move, zero logic changes, verified via grep/diff (no test runner available — see note below). `gemini_first/__init__.py` updated accordingly.
- ✅ **Added 4 project-scoped Claude Code skills** under `.claude/skills/` (auto-load in future sessions on this repo): `aec-new-mcp-tool` (scaffolding new tools), `aec-pattern-audit` (checking CLAUDE.md pattern compliance + known false-positive traps), `aec-sidecar-debug` (STA/timeout/circuit-breaker debugging playbook), `aec-standards-lookup` (using/extending the Phase 3 CAD standards + code-reference knowledge base). Built after evaluating and rejecting 3 external marketplace skills (generic PDF-OCR skill, unrelated FreeCAD-mechanical-parts skill repo) as poor fits for this project's AutoCAD/Revit + LA-jurisdiction MEP domain.
- ⚠️ **No Python interpreter on the primary dev machine** (confirmed: no python.exe, venv, or conda found anywhere on the Windows box these sessions run on). `pytest`/`ruff`/`mypy` cannot be run there — any refactor done in that environment is verified via grep/manual diff only and needs a real import-smoke-test + test run on a machine with the environment installed before being trusted. Flag this explicitly rather than assuming CI-equivalent verification happened.
- Found but NOT fixed (flagged for decision, avoid silently changing behavior): `enable_bezier_splatting` and `enable_live_vectorization` settings in `config/settings.py` are defined but never read anywhere — `phase_c_bezier_splatting`/`phase_c_live_vectorize` run unconditionally regardless of the flag.

## Recently Completed (2026-03-06)
- ✅ Database migration: `alembic upgrade head` (symbol_library table verified)
- ✅ Symbol library seeded: 77 CAD symbols with CLIP embeddings
  - 24 electrical, 14 mechanical, 14 plumbing, 12 fire, 13 architectural
- ✅ Knowledge Base MCP Tools added (6 new tools):
  - `query_cad_standards` - Query NCS layer/block/color for element types
  - `query_knowledge_base` - LLM-powered natural language query
  - `get_code_reference` - Building code references (CFC, NFPA, CEC, etc.)
  - `classify_element` - Classify MEP element from text description
  - `get_mep_rules` - Get MEP design rules (clearance, sizing, routing)
  - `get_system_priorities` - MEP coordination priorities for clash detection

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
| MCP Tools | 57 (8 categories) |
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
- [x] **Knowledge Base MCP Tools** (Phase 3 - Query codes/standards/formulas)
  - `query_cad_standards`: Query NCS layer/block/color for element types
  - `query_knowledge_base`: LLM-powered natural language query
  - `get_code_reference`: Building code references (CA: CFC/CEC/CPC/CMC, NFPA 72)
  - `classify_element`: Classify MEP element from text description
  - `get_mep_rules`: MEP design rules (clearance, sizing, routing)
  - `get_system_priorities`: MEP coordination priorities for clash detection
  - 132 unit tests (127 passing)

## Key Patterns
- **Tool results:** Return `{"success": False, "error": {...}}` — never raise exceptions
- **Sidecar calls:** `httpx.Timeout(120.0, connect=5.0)` + retry decorator
- **Coordinate conversion:** `x_dwg = px_x / dpi`, `y_dwg = (height - px_y) / dpi`
