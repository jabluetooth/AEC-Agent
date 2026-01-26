# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 🚀 workflow: Start Here
**ALWAYS read `docs/SESSION_CONTEXT.md` first.**
This file contains the current active plan, completed tasks, and immediate next steps. Do not re-scan the entire repository or ask for a plan unless `SESSION_CONTEXT.md` is missing or empty.

## Project Overview

AEC Agent is an AI-powered automation system bridging LLMs with AutoCAD and Revit. It uses Model Context Protocol (MCP) for natural language CAD control, with semantic search and spatial awareness via PostgreSQL.

## Architecture

Four-layer stack with SSE/HTTP transport:

```
┌─────────────────┐
│ Chainlit UI     │ Port 8000
│ (Frontend)      │
└────────┬────────┘
         │ SSE
┌────────▼────────┐
│ FastMCP Server  │ Port 54321
│ (Middleware)    │ asyncio.Lock for concurrency=1
└────────┬────────┘
         │ HTTP (retry + circuit breaker)
┌────────▼────────┐
│ Sidecars        │ Ports 20000-30000
│ AutoCAD: .NET   │ Thread queue → Application.Idle + Extraction
│ Revit: pyRevit  │ Routes API → ExternalEvent + Extraction
└────────┬────────┘
         │
┌────────▼────────┐
│ CAD Kernels     │ Single-Threaded Apartment (STA)
│ (AutoCAD/Revit) │
└────────┬────────┘
         │
┌────────▼────────┐
│ PostgreSQL      │ PostGIS + pgvector
│ (Data Layer)    │ Spatial queries, semantic search, relationships
└─────────────────┘
```

**Critical constraints:**
- Both CAD apps are STA. All API calls must be marshaled to main thread via queues.
- PostgreSQL required for spatial/semantic features. SQLite remains for session cache only.

## Commands

```bash
# Install (editable mode with dev dependencies)
pip install -e ".[dev]"

# Run all tests
pytest

# Run single test file
pytest tests/unit/test_settings.py -v

# Run with coverage
pytest --cov=src/aec_agent --cov-report=html

# Type checking
mypy src/

# Linting
ruff check src/

# Format code
black src/ tests/

# Check system compatibility
aec-agent check

# Show version matrix
aec-agent version --matrix
```

## Key Files

| Path | Purpose |
|------|---------|
| `src/aec_agent/config/settings.py` | All configuration via Pydantic Settings (env vars) |
| `src/aec_agent/config/versions.py` | Version compatibility matrix (AutoCAD/Revit/pyRevit) |
| `src/aec_agent/utils/version_checker.py` | Environment validation utility |
| `src/sidecars/autocad/` | AutoCAD .NET sidecar plugin |

## Configuration

All settings via environment variables or `.env` file. Key settings:

**LLM:**
- `LLM_PROVIDER`: openai, anthropic, azure_openai, huggingface, groq
- `GROQ_API_KEY`: Groq API key (if using groq provider)
- `GROQ_MODEL`: Model name (default: moonshotai/kimi-k2-instruct)

**Sidecar:**
- `MCP_LISTENER_PORT`: Sidecar port (default: 20000)
- `SESSION_TOKEN`: UUID for sidecar authentication
- `SIDECAR_READ_TIMEOUT`: 120s (CAD operations can be slow)
- `MAX_CONCURRENT_TOOLS`: 1 (CAD safety—one operation at a time)

**Token Optimization:**
- `COMPRESS_TOOL_SCHEMAS`: true (reduce tool description tokens)
- `SMART_TOOL_ROUTING`: true (filter tools by AutoCAD/Revit context)
- `MAX_HISTORY_MESSAGES`: 20 (conversation history limit)

**Database (future):**
- `DATABASE_URL`: PostgreSQL connection string
- `EMBEDDING_MODEL`: Sentence transformer model (default: all-MiniLM-L6-v2)

## Patterns

**Tool results:** Return structured objects, not exceptions
```python
return {"success": False, "error": {"code": 4002, "message": "Sidecar timeout"}}
```

**Sidecar calls:** Always use retry + timeout
```python
SIDECAR_TIMEOUT = httpx.Timeout(120.0, connect=5.0)
@retry(stop=stop_after_attempt(3), wait=wait_exponential(...))
async def call_sidecar(url, payload, token): ...
```

**Thread marshaling (AutoCAD):** Queue + Idle event
```csharp
_jobQueue.Enqueue(action);
// Later, in OnIdle handler:
using (DocumentLock lock = doc.LockDocument()) {
    using (Transaction tr = ...) { action(); tr.Commit(); }
}
```

## Version Support

- **AutoCAD:** 2021-2025 (2024 recommended), .NET Framework 4.8
- **Revit:** 2021-2025 (2024 recommended), pyRevit 4.8.x (5.0+ for Revit 2025)
- **Python:** 3.9-3.12 (3.11 recommended)

## Test Markers

```bash
pytest -m unit          # Unit tests only
pytest -m integration   # Integration tests
pytest -m security      # Security tests
pytest -m "not slow"    # Skip slow tests
```
