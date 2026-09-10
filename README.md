# BDG-MCP (AEC Agent)

AI-powered automation that bridges Large Language Models with AutoCAD and Revit via the [Model Context Protocol](https://modelcontextprotocol.io) (MCP). It acts as an intelligent sidecar, letting an LLM drive native geometric operations in AutoCAD and Revit for MEP (Mechanical, Electrical, Plumbing), HVAC, Fire Alarm, and Low Voltage design workflows — with semantic search and spatial awareness backed by PostgreSQL.

**Jurisdiction focus:** Los Angeles, CA building codes (CFC, NFPA 72, CEC, CPC, CMC, Title 24).

## Architecture

```
Chainlit UI (8000)  →  FastMCP Server (54321, concurrency=1)  →  Sidecars  →  PostgreSQL
   chat interface         tool registration & intent            AutoCAD .NET      PostGIS
   SSE streaming           classification, retry/circuit          Revit pyRevit    pgvector
                           breaker, token optimization           (ports 20000-30000)
```

- **Chainlit UI** — chat interface for natural-language CAD commands.
- **FastMCP Server** — registers 60+ MCP tools, classifies MEP intent, enforces `concurrency=1` for CAD safety, retries/circuit-breaks sidecar calls.
- **Sidecars** — AutoCAD (.NET Framework 4.8) and Revit (pyRevit) processes. Both CAD apps are **STA** (single-threaded apartment): all API calls marshal to the main thread via a job queue processed on `OnIdle`, inside a `DocumentLock` + `Transaction`.
- **PostgreSQL** — PostGIS for spatial queries, pgvector for semantic/symbol search, plus project memory, domain rules, and workflow state.

## Key Capabilities

| Area | What it does |
|------|--------------|
| **PDF → AutoCAD vectorization** | Gemini-First pipeline: PDF rendering, drawing analysis, coordinate calibration, adaptive/hybrid extraction (Gemini + OpenCV + YOLO fusion), symbol recognition (CLIP + pgvector RAG), AutoCAD entity creation |
| **Raster Design pipeline** | PDF-to-DWG via AutoCAD Raster Design: import, cleanup, OpenCV-based vectorization, OCR, and topology cleanup |
| **Knowledge base** | NCS CAD layer/block/color standards and building-code references (CFC, NFPA 72, CEC, CPC, CMC), queryable by MCP tools |
| **MEP workflows** | Pre-defined multi-step workflow templates that execute without a per-step LLM round trip |
| **Domain validation** | Clearance, sizing, routing, and access rule checks against MEP domain rules, with design suggestions |
| **Project memory** | Persistent project facts/decisions and conversation summaries, recallable across a session |
| **Symbol recognition** | CLIP embeddings + pgvector RAG lookup against a seeded CAD symbol library |

## Project Layout

```
src/aec_agent/
├── config/          Pydantic Settings (env vars) + AutoCAD/Revit/pyRevit version matrix
├── frontend/        Chainlit UI, multi-LLM backend (OpenAI/Anthropic/Azure/HF/Groq/Gemini)
├── mcp/             FastMCP server, sidecar HTTP client (retry + circuit breaker), tools/
├── intent/          MEP-aware intent classification (hybrid keyword + embedding)
├── db/              Async PostgreSQL pool, SQLAlchemy models, Alembic migrations
├── semantic/        sentence-transformers embeddings, pgvector search
├── domain/          MEP knowledge base, rules engine, validators
├── workflows/       Workflow templates and executor
├── memory/          Project memory (facts, summaries) and user preferences
└── utils/           Logging, version checking

src/sidecars/
├── autocad/         AutoCAD .NET sidecar plugin (STA thread queue + HTTP listener)
└── revit/           Revit pyRevit extension

knowledge_base/cad_standards/   NCS layer/block/color YAML, per discipline
docs/                           Architecture notes, phase plans, session continuity
```

## Getting Started

**Requirements:** Python 3.9–3.12, PostgreSQL 15+ with PostGIS and pgvector, AutoCAD 2021–2025 and/or Revit 2021–2025 for live CAD control.

```bash
# Install (editable, with dev tools)
pip install -e ".[dev]"

# Configure environment
cp .env.example .env   # add API keys, DATABASE_URL, etc.

# Apply database migrations
alembic upgrade head

# Seed default HVAC domain rules
aec-agent seed-rules

# Check system compatibility (AutoCAD/Revit/Python versions)
aec-agent check

# Run everything (MCP server + Chainlit UI)
aec-agent launch

# ...or run components independently
aec-agent server   # FastMCP server only
aec-agent ui       # Chainlit UI only (requires server running)
```

## Development

```bash
pytest                         # All tests
pytest -m unit                 # Unit tests only
pytest -m integration          # Integration tests only
pytest --cov=src/aec_agent     # With coverage
mypy src/                      # Type check
ruff check src/                # Lint
black src/ tests/              # Format
```

## Version Support

| Component | Supported versions |
|-----------|---------------------|
| AutoCAD | 2021–2025 (.NET Framework 4.8) |
| Revit | 2021–2025 (pyRevit 4.8.x / 5.0+) |
| Python | 3.9–3.12 |

## Roadmap

| Phase | Focus | Status |
|-------|-------|--------|
| 1 | Foundation (cache, Groq/Gemini) | Complete |
| 2 | Raster Design (PDF to DWG) | Complete |
| 3 | Knowledge Base (LA codes, catalogs, formulas) | Complete |
| 4 | HVAC Design (CMC + ASHRAE) | Pending |
| 5 | Fire Alarm Design (CFC + NFPA 72) | Pending |
| 6 | Low Voltage Design (TIA/EIA + BICSI) | Pending |
| 7 | Electrical Design (CEC + Title 24) | Pending |
| 8 | Plumbing Design (CPC) | Pending |
| 9 | Multi-System Coordination | Pending |
| 10 | Intelligence & Learning | Ongoing |

See `docs/FUTURE_ROADMAP.md` for the full plan and `docs/SESSION_CONTEXT.md` for current progress.

## Documentation

- [`CLAUDE.md`](CLAUDE.md) — project conventions and patterns for AI-assisted development
- [`docs/SESSION_CONTEXT.md`](docs/SESSION_CONTEXT.md) — active plan, completed work, next steps
- [`docs/FUTURE_ROADMAP.md`](docs/FUTURE_ROADMAP.md) — long-term vision and phased build plan
- [`docs/MEP_INTEGRATION_TODO.md`](docs/MEP_INTEGRATION_TODO.md) — remaining MEP feature integration work

## License

MIT
