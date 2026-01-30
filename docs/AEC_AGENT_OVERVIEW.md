# AEC Agent - Complete Repository Overview & Audit

## Executive Summary

AEC Agent is an AI-powered automation system that bridges Large Language Models (LLMs) with AutoCAD and Revit for MEP (Mechanical, Electrical, Plumbing), HVAC, and Low Voltage system design workflows. It uses Model Context Protocol (MCP) for natural language CAD control with semantic search and spatial awareness via PostgreSQL.

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                        USER INTERFACE                            │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │              Chainlit UI (Port 8000)                     │    │
│  │  - Chat interface for natural language commands          │    │
│  │  - Real-time streaming responses                         │    │
│  │  - 21 language translations                              │    │
│  └─────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────┘
                              │ SSE
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                     MCP MIDDLEWARE LAYER                         │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │           FastMCP Server (Port 54321)                    │    │
│  │  - Tool registration & discovery                         │    │
│  │  - asyncio.Lock for concurrency=1 (CAD safety)          │    │
│  │  - Intent classification (MEP-aware)                     │    │
│  │  - Token optimization (40-60% reduction)                 │    │
│  └─────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────┘
                              │ HTTP (retry + circuit breaker)
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                      CAD SIDECAR LAYER                           │
│  ┌────────────────────┐    ┌────────────────────────────────┐   │
│  │  AutoCAD Sidecar   │    │      Revit Sidecar             │   │
│  │  (.NET Framework)  │    │      (pyRevit)                 │   │
│  │  Port 20000-30000  │    │      Port 20000-30000          │   │
│  │                    │    │                                │   │
│  │  - Thread queue    │    │  - Routes API                  │   │
│  │  - Application.Idle│    │  - ExternalEvent               │   │
│  │  - Entity extract  │    │  - Element extraction          │   │
│  └────────────────────┘    └────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                       DATA LAYER                                 │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │              PostgreSQL Database                         │    │
│  │  - PostGIS (spatial queries)                            │    │
│  │  - pgvector (semantic search)                           │    │
│  │  - Element relationships                                │    │
│  │  - Project metadata                                     │    │
│  └─────────────────────────────────────────────────────────┘    │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │              SQLite Cache                                │    │
│  │  - Session state                                        │    │
│  │  - Active project tracking                              │    │
│  └─────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────┘
```

---

## Directory Structure

```
aec/
├── src/
│   └── aec_agent/
│       ├── config/              # Settings & version matrix
│       │   ├── settings.py      # Pydantic settings (env vars)
│       │   └── versions.py      # AutoCAD/Revit/pyRevit versions
│       │
│       ├── frontend/            # Chainlit UI & LLM agent
│       │   ├── app.py           # Chainlit handlers
│       │   ├── agent.py         # Multi-LLM backend support
│       │   ├── mcp_client.py    # SSE client for MCP server
│       │   └── tool_optimization.py  # Token reduction strategies
│       │
│       ├── mcp/                 # MCP server & tools
│       │   ├── server.py        # FastMCP server setup
│       │   ├── concurrency.py   # asyncio.Lock utilities
│       │   ├── sidecar_client.py # HTTP client with retry
│       │   └── tools/
│       │       ├── autocad.py   # AutoCAD tools (layers, drawing)
│       │       ├── revit.py     # Revit tools (levels, walls)
│       │       ├── metadata.py  # Semantic search tools
│       │       └── smart_tools.py # Context-aware filtering
│       │
│       ├── intent/              # MEP intent classification
│       │   ├── classifier.py    # Hybrid keyword+embedding classifier
│       │   ├── models.py        # Intent data models
│       │   └── patterns.py      # MEP domain patterns
│       │
│       ├── db/                  # PostgreSQL data layer
│       │   ├── connection.py    # Async pool management
│       │   ├── models.py        # SQLAlchemy ORM
│       │   ├── repository.py    # Data access patterns
│       │   ├── queries/
│       │   │   ├── spatial.py   # PostGIS queries
│       │   │   └── relationships.py
│       │   └── migrations/      # Alembic migrations
│       │
│       ├── semantic/            # Semantic search
│       │   ├── embeddings.py    # sentence-transformers
│       │   ├── search.py        # pgvector queries
│       │   └── description_generator.py
│       │
│       ├── extraction/          # CAD data extraction
│       │   ├── autocad_extractor.py
│       │   ├── revit_extractor.py
│       │   └── sync_manager.py
│       │
│       ├── domain/              # AEC domain logic
│       │   ├── models.py        # Domain entities
│       │   ├── knowledge.py     # AEC domain knowledge
│       │   └── rules_engine.py  # Business rules
│       │
│       ├── workflows/           # MEP workflows
│       │   ├── executor.py      # Workflow execution
│       │   ├── models.py        # State machines
│       │   └── templates.py     # Workflow templates
│       │
│       ├── memory/              # Context management
│       │   ├── project_memory.py
│       │   └── user_preferences.py
│       │
│       ├── cache/               # SQLite session cache
│       │   └── sqlite_cache.py
│       │
│       └── utils/               # Utilities
│           ├── logging.py       # Structlog config
│           └── version_checker.py
│
├── src/sidecars/
│   ├── autocad/                 # AutoCAD .NET sidecar
│   └── revit/                   # Revit pyRevit sidecar
│
├── tests/
│   ├── unit/                    # Unit tests
│   └── test_mcp_server.py       # Integration tests
│
├── scripts/                     # Launch scripts
├── docs/                        # Additional documentation
└── implementation_phase/        # Design documents
```

---

## MEP Domain Support Assessment

### Current MEP Capabilities

| Domain | Status | Tools | Notes |
|--------|--------|-------|-------|
| **HVAC** | Partial | `find_elements`, `get_nearby_elements` | Category filters defined |
| **Electrical** | Partial | `find_elements`, `get_nearby_elements` | Category filters defined |
| **Plumbing** | Partial | `find_elements`, `get_nearby_elements` | Category filters defined |
| **Fire Protection** | Partial | `find_elements` | Basic support |
| **Low Voltage** | Limited | Via electrical category | Needs dedicated patterns |

### MEP Category Filters (Implemented)

```python
# HVAC
revit_categories: ["Mechanical Equipment", "Ducts", "Duct Fittings",
                   "Duct Accessories", "Air Terminals", "Flex Ducts"]
autocad_layers: ["M-HVAC", "M-DUCT", "M-EQUIP", "MECH", "HVAC"]

# Electrical
revit_categories: ["Electrical Equipment", "Electrical Fixtures",
                   "Conduits", "Cable Trays", "Lighting Fixtures"]
autocad_layers: ["E-POWER", "E-LITE", "E-EQUIP", "ELEC"]

# Plumbing
revit_categories: ["Plumbing Fixtures", "Pipes", "Pipe Fittings",
                   "Pipe Accessories"]
autocad_layers: ["P-SANR", "P-DOME", "P-FIXT", "PLUMB"]

# Fire Protection
revit_categories: ["Sprinklers", "Fire Alarm Devices"]
autocad_layers: ["F-SPKL", "F-ALRM", "FIRE", "SPRINKLER"]
```

### Intent Classification (MEP-Aware)

The system classifies user intents into MEP domains:

```python
# Supported domains
MEPDomain.HVAC          # Heating, ventilation, air conditioning
MEPDomain.ELECTRICAL    # Power, lighting, low voltage
MEPDomain.PLUMBING      # Water, drainage, gas
MEPDomain.FIRE_PROTECTION # Sprinklers, alarms
MEPDomain.GENERAL       # Non-MEP queries

# Supported actions
MEPAction.QUERY         # "Find all ducts"
MEPAction.CREATE        # "Draw a duct"
MEPAction.MODIFY        # "Change duct size"
MEPAction.DELETE        # "Remove element"
MEPAction.ANALYZE       # "Check clearances"
MEPAction.ROUTE         # "Route from A to B"
MEPAction.SCHEDULE      # "List all VAV boxes"
MEPAction.COORDINATE    # "Check for clashes"
```

---

## Token Optimization Analysis

### Current Token Optimization Features

| Feature | Setting | Default | Token Savings |
|---------|---------|---------|---------------|
| Tool Schema Compression | `compress_tool_schemas` | `True` | 30-50% |
| Smart Tool Routing | `smart_tool_routing` | `True` | 40-60% |
| History Trimming | `max_history_messages` | 20 | Variable |
| Result Truncation | `max_tool_result_chars` | 2000 | Variable |
| Compression Mode | `tool_compression_mode` | `standard` | Configurable |

### Compression Modes

| Mode | Description | Use Case |
|------|-------------|----------|
| `full` | No compression | Development/debugging |
| `standard` | Remove examples, compress whitespace | Default usage |
| `minimal` | Types only, no descriptions | Budget LLMs (Groq, HF) |
| `ultra` | Required params only | Extreme token limits |

### Token Estimation

```python
# Current implementation
def estimate_token_count(text: str) -> int:
    """Rough token estimate (~4 chars per token)."""
    return len(text) // 4
```

### Tool Tier System

```python
TOOL_TIERS = {
    "essential": [  # Always loaded
        "ping", "autocad_draw_line", "autocad_draw_circle",
        "autocad_draw_rectangle", "autocad_list_layers",
        "revit_create_wall", "revit_list_levels", "revit_get_document_status"
    ],
    "standard": [   # Loaded by default
        "autocad_create_layer", "autocad_set_layer_state",
        "autocad_get_drawing_info", "autocad_get_entities",
        "revit_create_level", "revit_list_walls", "revit_list_rooms"
    ],
    "advanced": [   # Requires database
        "find_elements", "get_nearby_elements", "get_related_elements",
        "resolve_coordinates", "sync_metadata"
    ]
}
```

---

## LLM Provider Support

| Provider | Status | Token Optimization |
|----------|--------|-------------------|
| OpenAI | Full | Standard compression |
| Anthropic | Full | Standard compression |
| Azure OpenAI | Full | Standard compression |
| Hugging Face | Full | Minimal compression (auto) |
| Groq | Full | Minimal compression (auto) |

---

## Audit Findings & Recommendations

### Strengths

1. **Well-structured token optimization** - Multi-level compression system
2. **MEP domain awareness** - Intent classification for relevant tool loading
3. **Flexible LLM support** - 5 providers with auto-optimization
4. **Proper CAD thread safety** - asyncio.Lock for STA marshaling
5. **Semantic search ready** - pgvector + sentence-transformers

### Areas for Improvement

#### 1. Token Optimization (High Priority)

**Issue**: Tool descriptions still verbose in some cases
**Recommendation**:
- Add token budget tracking per conversation
- Implement dynamic compression based on context length
- Add prompt caching for system prompts

#### 2. MEP-Specific Tools (Medium Priority)

**Issue**: Generic tools, no MEP-specific operations
**Recommendation**: Add tools like:
- `route_duct` - Automatic duct routing
- `check_clearances` - MEP clearance validation
- `size_pipe` - Pipe sizing calculations
- `balance_system` - HVAC balancing
- `circuit_analysis` - Electrical load analysis

#### 3. Low Voltage Support (Medium Priority)

**Issue**: Low voltage lumped with electrical
**Recommendation**:
- Add dedicated `LOW_VOLTAGE` MEP domain
- Add patterns for: security, fire alarm, BMS, AV, data/telecom
- Add AutoCAD layers: `LV-DATA`, `LV-SEC`, `LV-FA`, `LV-AV`

#### 4. Output Token Optimization (Medium Priority)

**Issue**: Tool results can be verbose
**Recommendation**:
- Enable `enable_result_summarization` by default
- Add streaming result compression
- Implement selective field filtering

#### 5. Context Optimization (Low Priority)

**Issue**: Full history sent each time
**Recommendation**:
- Implement conversation summarization
- Add semantic memory for relevant context retrieval

---

## Configuration Reference

### Key Environment Variables

```bash
# LLM Configuration
LLM_PROVIDER=openai              # openai, anthropic, azure_openai, huggingface, groq
OPENAI_API_KEY=sk-...
GROQ_API_KEY=gsk_...
GROQ_MODEL=moonshotai/kimi-k2-instruct

# Token Optimization
COMPRESS_TOOL_SCHEMAS=true
SMART_TOOL_ROUTING=true
MAX_HISTORY_MESSAGES=20
TOOL_COMPRESSION_MODE=standard   # full, standard, minimal, ultra
TOOL_TIER=standard               # essential, standard, advanced
MAX_TOOL_RESULT_CHARS=2000

# Intent Classification
ENABLE_INTENT_CLASSIFICATION=true
USE_INTENT_EMBEDDINGS=false      # Set true for embedding similarity
INTENT_MIN_CONFIDENCE=0.3
MEP_DOMAIN_PRIORITY=hvac

# Database (for semantic search)
DATABASE_URL=postgresql://user:pass@host:5432/db
EMBEDDING_MODEL=all-MiniLM-L6-v2

# Sidecar
MCP_LISTENER_PORT=20000
SESSION_TOKEN=<uuid>
SIDECAR_READ_TIMEOUT=120
MAX_CONCURRENT_TOOLS=1
```

---

## Quick Start Commands

```bash
# Install
pip install -e ".[dev]"

# Run tests
pytest

# Start full system
aec-agent launch

# Start MCP server only
aec-agent server

# Start UI only
aec-agent ui

# Check compatibility
aec-agent check
```

---

## Version Support

| Software | Supported Versions | Recommended |
|----------|-------------------|-------------|
| AutoCAD | 2021-2025 | 2024 |
| Revit | 2021-2025 | 2024 |
| pyRevit | 4.8.x, 5.0+ | 4.8.x (5.0+ for Revit 2025) |
| Python | 3.9-3.12 | 3.11 |
| .NET Framework | 4.8 | 4.8 |

---

## File Statistics

| Metric | Count |
|--------|-------|
| Total Python Files | 78 |
| Source Files | 67 |
| Test Files | 11 |
| MCP Tools | 20+ |
| Supported Languages (UI) | 21 |
| LLM Providers | 5 |
| Database Migrations | 2 |

---

*Document generated: 2026-01-23*
*Repository: AEC Agent v0.1.0*
