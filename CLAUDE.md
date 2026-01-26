# CLAUDE.md

## Start Here
**ALWAYS read `docs/SESSION_CONTEXT.md` first.** It has the active plan, completed tasks, and next steps. Do not re-scan the repo unless that file is missing.

## Project Overview
AEC Agent bridges LLMs with AutoCAD/Revit via MCP. Four layers: **Chainlit UI** (8000) → **FastMCP Server** (54321, concurrency=1) → **Sidecars** (AutoCAD .NET / Revit pyRevit, ports 20000-30000) → **PostgreSQL** (PostGIS + pgvector).

**Critical:** Both CAD apps are STA — all API calls must marshal to main thread via queues.

## Key Files

| Path | Purpose |
|------|---------|
| `src/aec_agent/config/settings.py` | All config via Pydantic Settings (env vars) |
| `src/aec_agent/config/versions.py` | Version compatibility matrix |
| `src/sidecars/autocad/` | AutoCAD .NET sidecar plugin |
| `docs/FUTURE_ROADMAP.md` | Long-term vision, tool gaps, phased dev plan |
| `docs/SESSION_CONTEXT.md` | Active plan, completed tasks, next steps |
| `docs/PLAN.md` | Technical spec for current phase |

## Commands
```bash
pip install -e ".[dev]"        # Install
pytest                         # All tests
pytest -m unit                 # Unit only
pytest -m integration          # Integration only
pytest --cov=src/aec_agent     # Coverage
mypy src/                      # Type check
ruff check src/                # Lint
black src/ tests/              # Format
```

## Patterns
- **Tool results:** Return `{"success": False, "error": {"code": N, "message": "..."}}` — never raise exceptions.
- **Sidecar calls:** Always use `httpx.Timeout(120.0, connect=5.0)` + `@retry(stop=stop_after_attempt(3), wait=wait_exponential(...))`.
- **Thread marshaling (AutoCAD):** Enqueue to `_jobQueue` → process in `OnIdle` handler with `DocumentLock` + `Transaction`.

## Version Support
- **AutoCAD:** 2021-2025 (.NET Framework 4.8) | **Revit:** 2021-2025 (pyRevit 4.8.x/5.0+) | **Python:** 3.9-3.12

## Future Roadmap
**Full details: `docs/FUTURE_ROADMAP.md`**

Vision: **AI that autonomously designs MEP, LV, and Fire Alarm systems** in AutoCAD/Revit. Jurisdiction: **Los Angeles, CA**.

| Phase | Focus | Status |
|-------|-------|--------|
| 1 | Foundation (cache, Groq/Gemini) | Pending |
| 2 | Raster Design (PDF to DWG) | Pending |
| 3 | Knowledge Base (LA codes, catalogs, formulas) | Pending |
| 4 | HVAC Design (CMC + ASHRAE) | Pending |
| 5 | Fire Alarm Design (CFC + NFPA 72) | Pending |
| 6 | Low Voltage Design (TIA/EIA + BICSI) | Pending |
| 7 | Electrical Design (CEC + Title 24) | Pending |
| 8 | Plumbing Design (CPC) | Pending |
| 9 | Multi-System Coordination | Pending |
| 10 | Intelligence & Learning | Ongoing |

**Currently 34% ready.** Missing: element placement, engineering calcs, routing, code validation, knowledge base query. Build order: `query_knowledge_base` → `place_revit_family` → `calculate_*` → `find_route`.
