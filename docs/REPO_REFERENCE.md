# AEC Agent Repository Reference

> **Quick Reference for Editing Without AI Scanning**  
> This file documents the entire repository structure, key files, and their contents.

---

## Project Overview

**AEC Agent** - AI-powered automation assistant for AutoCAD and Revit. Uses MCP (Model Context Protocol) to communicate with sidecar plugins in CAD applications.

### Architecture
```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│   Chainlit UI   │────▶│   MCP Server    │────▶│  CAD Sidecars   │
│  (Frontend)     │     │  (Python)       │     │  (C# Plugins)   │
└─────────────────┘     └─────────────────┘     └─────────────────┘
```

---

## Directory Structure

```
aec/
├── .env                          # Environment configuration
├── pyproject.toml                # Python project config & dependencies
├── requirements.txt              # Production dependencies
├── requirements-dev.txt          # Dev dependencies (pytest, ruff, etc.)
├── CLAUDE.md                     # Claude AI assistant instructions
├── README.md                     # Project readme
├── chainlit.md                   # Chainlit welcome markdown
│
├── scripts/                      # Startup/utility scripts
│   ├── Start-AECAgent-Dev.ps1    # PowerShell dev launcher
│   ├── AECAgent-Logon.ps1        # Logon script
│   ├── AECAgent-Logon.cmd        # Batch logon script
│   └── README.md                 # Scripts documentation
│
├── src/
│   ├── aec_agent/                # Main Python package
│   │   ├── __init__.py           # Package init, version info
│   │   ├── cli.py                # CLI entry point
│   │   ├── launcher.py           # Process orchestration
│   │   ├── server.py             # MCP server entry
│   │   │
│   │   ├── config/               # Configuration
│   │   │   ├── __init__.py
│   │   │   ├── settings.py       # Pydantic settings (env vars)
│   │   │   └── versions.py       # Version compatibility matrix
│   │   │
│   │   ├── frontend/             # Chainlit UI
│   │   │   ├── __init__.py
│   │   │   ├── app.py            # Chainlit app handlers
│   │   │   ├── agent.py          # LLM agent (OpenAI, Anthropic, Azure, HF)
│   │   │   └── mcp_client.py     # HTTP client for MCP server
│   │   │
│   │   ├── mcp/                  # MCP Server & Tools
│   │   │   ├── __init__.py
│   │   │   ├── server.py         # FastMCP server setup
│   │   │   ├── concurrency.py    # Thread locking utilities
│   │   │   ├── sidecar_client.py # HTTP client for CAD sidecars
│   │   │   └── tools/            # MCP tool definitions
│   │   │       ├── __init__.py
│   │   │       ├── base.py       # Base tool utilities
│   │   │       ├── common.py     # Shared tools (check_sidecar)
│   │   │       ├── autocad.py    # AutoCAD tools
│   │   │       └── revit.py      # Revit tools
│   │   │
│   │   ├── cache/                # Caching layer
│   │   │   └── sqlite_cache.py   # SQLite-based cache
│   │   │
│   │   └── utils/                # Utilities
│   │       ├── __init__.py
│   │       ├── logging.py        # Structlog setup
│   │       └── version_checker.py # Version compatibility checks
│   │
│   └── sidecars/                 # C# Sidecar Plugins
│       └── autocad/              # AutoCAD Sidecar
│           ├── AECAgentSidecar.csproj
│           ├── SidecarPlugin.cs      # Plugin entry point
│           ├── Commands/
│           │   ├── CommandRouter.cs  # Routes commands
│           │   ├── DrawingCommands.cs # Line, circle, rectangle
│           │   ├── LayerCommands.cs  # Layer operations
│           │   └── QueryCommands.cs  # Get entities, info
│           ├── HttpServer/
│           │   ├── SidecarListener.cs # HTTP listener
│           │   └── RequestHandler.cs  # Request processing
│           ├── Models/
│           │   ├── CommandParams.cs  # Parameter DTOs
│           │   ├── CommandResponse.cs # Response DTOs
│           │   └── JobRequest.cs     # Job request model
│           ├── Utils/
│           │   ├── Logger.cs         # Logging utility
│           │   ├── MetricsCollector.cs
│           │   └── SecurityValidator.cs
│           └── Properties/
│               └── AssemblyInfo.cs
│
├── tests/                        # Test suite
│   ├── __init__.py
│   ├── test_mcp_server.py        # MCP server tests
│   └── unit/                     # Unit tests
│
├── implementation_phase/         # Design documents
│   ├── 00_audit_and_recommendations.md
│   ├── 00_version_matrix.md
│   ├── 02_autocad_sidecar.md
│   ├── 03_revit_sidecar.md
│   ├── 04_mcp_middleware.md
│   ├── 05_frontend.md
│   ├── 06_governance.md
│   ├── 07_error_handling.md
│   └── 08_testing_and_validation.md
│
└── htmlcov/                      # Coverage reports (generated)
```

---

## Key Python Files

### `src/aec_agent/cli.py`
**CLI Entry Point** - Commands: `launch`, `server`, `ui`, `check`, `config`, `version`, `init`

Functions:
- `cmd_launch(args)` - Launch full system (MCP + Chainlit)
- `cmd_server(args)` - Run MCP server only
- `cmd_ui(args)` - Run Chainlit UI only
- `cmd_check(args)` - Run compatibility checks
- `cmd_config(args)` - Display configuration
- `cmd_version(args)` - Display version info
- `cmd_init(args)` - Initialize environment
- `main(argv)` - Main entry point

---

### `src/aec_agent/launcher.py`
**Process Orchestration** - Manages MCP server and Chainlit processes

Classes/Functions:
- `find_free_port(start_port, max_attempts)` - Find available port
- `wait_for_port(port, timeout, interval)` - Wait for server to start
- `ProcessManager` - Class managing subprocess lifecycle
  - `__init__()` - Initialize process list
  - `start_process(args, env, name)` - Start subprocess
  - `shutdown()` - Gracefully shutdown all processes
- `launch()` - Main launcher function

---

### `src/aec_agent/config/settings.py`
**Configuration** - Pydantic settings from environment

Enums:
- `Environment` - DEVELOPMENT, STAGING, PRODUCTION
- `LogFormat` - JSON, TEXT
- `LLMProvider` - OPENAI, ANTHROPIC, AZURE_OPENAI, HUGGINGFACE

Class `Settings`:
- `env` - Environment mode
- `debug` - Debug flag
- `log_level` - DEBUG, INFO, WARNING, ERROR
- `log_format` - JSON or TEXT
- `mcp_server_host` - MCP server host
- `mcp_server_port` - MCP server port
- `chainlit_host` - Chainlit host
- `chainlit_port` - Chainlit port
- `sidecar_autocad_port` - AutoCAD sidecar port
- `sidecar_revit_port` - Revit sidecar port
- `sidecar_timeout` - Request timeout
- `sidecar_connect_timeout` - Connection timeout
- `session_token` - Authentication token
- `llm_provider` - LLM provider
- `openai_api_key` - OpenAI key
- `openai_model` - OpenAI model name
- `anthropic_api_key` - Anthropic key
- `anthropic_model` - Anthropic model name
- `azure_openai_*` - Azure OpenAI settings
- `huggingface_*` - HuggingFace settings
- `cache_enabled` - Enable caching
- `cache_dir` - Cache directory
- `cache_ttl_seconds` - Cache TTL

---

### `src/aec_agent/frontend/agent.py`
**LLM Agent** - Handles LLM conversations with tool calling

Dataclasses:
- `Message` - Conversation message (role, content, tool_calls)
- `ToolCall` - Tool call from LLM (id, name, arguments)
- `AgentResponse` - Response (content, tool_calls, finished)

Classes:
- `LLMBackend` (ABC) - Abstract base for LLM backends
  - `generate(messages, tools)` - Generate response
  - `stream(messages, tools)` - Stream response
- `OpenAIBackend(LLMBackend)` - OpenAI GPT implementation
- `AnthropicBackend(LLMBackend)` - Anthropic Claude implementation
- `AzureOpenAIBackend(LLMBackend)` - Azure OpenAI implementation
- `HuggingFaceBackend(LLMBackend)` - HuggingFace implementation
- `AECAgent` - Main agent class
  - `__init__(mcp_client)` - Initialize with MCP client
  - `process_message(content)` - Async generator for responses
  - `clear_history()` - Clear conversation

---

### `src/aec_agent/frontend/app.py`
**Chainlit App** - User-facing chat interface

Decorated Functions:
- `@on_chat_start` - Initialize chat session
- `@on_message` - Handle user messages
- `@on_chat_end` - Cleanup session
- `@on_stop` - Handle stop generation
- `@action_callback("clear_history")` - Clear history action
- `@action_callback("check_sidecar")` - Check sidecar status
- `@set_chat_profiles` - Define chat profiles
- `@on_settings_update` - Handle settings updates

---

### `src/aec_agent/mcp/sidecar_client.py`
**Sidecar HTTP Client** - Communication with CAD sidecars

Functions:
- `get_client()` - Get/create HTTP client
- `close_client()` - Close HTTP client
- `get_sidecar_url(endpoint)` - Build sidecar URL
- `get_auth_headers(sidecar_type)` - Get auth headers
- `call_sidecar(endpoint, method, payload, sidecar_type)` - Call sidecar
- `call_autocad_command(command, params)` - Call AutoCAD command
- `check_sidecar_health(sidecar_type)` - Health check

Exceptions:
- `SidecarError` - Base exception
- `SidecarTimeoutError` - Timeout error
- `SidecarConnectionError` - Connection error
- `SidecarCircuitOpenError` - Circuit breaker open

---

### `src/aec_agent/mcp/tools/autocad.py`
**AutoCAD Tools** - MCP tools for AutoCAD automation

Functions (decorated with `@mcp.tool()`):
- `autocad_list_layers()` - List all layers
- `autocad_create_layer(name, color)` - Create layer
- `autocad_set_layer_state(name, is_on, is_frozen)` - Set layer state
- `autocad_draw_line(start_x, start_y, end_x, end_y, layer)` - Draw line
- `autocad_draw_circle(center_x, center_y, radius, layer)` - Draw circle
- `autocad_draw_rectangle(corner1_x, corner1_y, corner2_x, corner2_y, layer)` - Draw rectangle
- `autocad_get_drawing_info()` - Get drawing info
- `autocad_get_entities(layer, entity_type, limit)` - Get entities

---

### `src/aec_agent/mcp/tools/revit.py`
**Revit Tools** - MCP tools for Revit automation

Functions (decorated with `@mcp.tool()`):
- `revit_list_levels()` - List all levels
- `revit_create_level(name, elevation)` - Create level
- `revit_list_walls()` - List all walls
- `revit_create_wall(start_x, start_y, end_x, end_y, level_name, height)` - Create wall
- `revit_list_rooms(use_cache)` - List all rooms
- `revit_get_document_status()` - Get document info
- `revit_delete_element(element_id)` - Delete element

---

## Key C# Files (AutoCAD Sidecar)

### `src/sidecars/autocad/SidecarPlugin.cs`
**Plugin Entry Point** - AutoCAD plugin initialization

### `src/sidecars/autocad/Commands/CommandRouter.cs`
**Command Router** - Routes incoming commands to handlers

### `src/sidecars/autocad/Commands/DrawingCommands.cs`
**Drawing Commands** - Line, circle, rectangle creation

### `src/sidecars/autocad/Commands/LayerCommands.cs`
**Layer Commands** - Layer CRUD operations

### `src/sidecars/autocad/Commands/QueryCommands.cs`
**Query Commands** - Entity queries, drawing info

### `src/sidecars/autocad/HttpServer/SidecarListener.cs`
**HTTP Listener** - HTTP server for receiving commands

### `src/sidecars/autocad/HttpServer/RequestHandler.cs`
**Request Handler** - Process incoming HTTP requests

### `src/sidecars/autocad/Models/CommandParams.cs`
**Parameter DTOs** - Data transfer objects for parameters

### `src/sidecars/autocad/Models/CommandResponse.cs`
**Response DTOs** - Standard response format

---

## Environment Variables (.env)

```env
# Environment
AEC_ENV=development
AEC_DEBUG=true
AEC_LOG_LEVEL=DEBUG
AEC_LOG_FORMAT=text

# MCP Server
AEC_MCP_SERVER_HOST=127.0.0.1
AEC_MCP_SERVER_PORT=20987

# Chainlit
AEC_CHAINLIT_HOST=0.0.0.0
AEC_CHAINLIT_PORT=8000

# Sidecars
AEC_SIDECAR_AUTOCAD_PORT=25603
AEC_SIDECAR_REVIT_PORT=20001
AEC_SIDECAR_TIMEOUT=30.0
AEC_SIDECAR_CONNECT_TIMEOUT=5.0

# Authentication
SESSION_TOKEN=your-session-token-here
MCP_LISTENER_PORT=25603

# LLM Provider (openai, anthropic, azure_openai, huggingface)
AEC_LLM_PROVIDER=openai
OPENAI_API_KEY=sk-...
AEC_OPENAI_MODEL=gpt-4o

# Or for Anthropic
# AEC_LLM_PROVIDER=anthropic
# ANTHROPIC_API_KEY=sk-ant-...
# AEC_ANTHROPIC_MODEL=claude-3-5-sonnet-20241022

# Cache
AEC_CACHE_ENABLED=true
AEC_CACHE_DIR=${LOCALAPPDATA}/AECAgent/cache
AEC_CACHE_TTL_SECONDS=300
```

---

## Running the Application

```powershell
# Activate virtual environment
.\venv\Scripts\Activate.ps1

# Launch full system (MCP + UI)
python -m aec_agent.cli launch

# Run MCP server only
python -m aec_agent.cli server

# Run Chainlit UI only (assumes MCP running)
python -m aec_agent.cli ui

# Check compatibility
python -m aec_agent.cli check

# Show config
python -m aec_agent.cli config

# Show version
python -m aec_agent.cli version
```

---

## Testing

```powershell
# Run all tests
pytest

# Run with coverage
pytest --cov=src/aec_agent --cov-report=html

# Run specific test file
pytest tests/test_mcp_server.py

# Run linter
ruff check src/
```

---

## API Endpoints

### MCP Server (default: http://127.0.0.1:20987)
- `GET /health` - Health check
- `POST /tools/list` - List available tools
- `POST /tools/call` - Execute a tool

### AutoCAD Sidecar (default: http://127.0.0.1:25603)
- `POST /` - Execute command (Bearer token auth)
  ```json
  {"command": "draw_line", "params": {"start": {"x": 0, "y": 0}, "end": {"x": 100, "y": 100}}}
  ```

### Revit Sidecar (default: http://127.0.0.1:20001)
- `GET /mcp/levels` - List levels
- `POST /mcp/levels/create` - Create level
- `GET /mcp/walls` - List walls
- `POST /mcp/walls/create` - Create wall
- `GET /mcp/rooms` - List rooms
- `GET /mcp/status` - Document status

---

*Generated: 2026-01-19*
