# Version Compatibility Matrix

**Objective:** Define supported versions for all software components to ensure compatibility and guide deployment decisions.

## 1. Core Application Versions

### AutoCAD Compatibility

| AutoCAD Version | .NET Framework | ObjectARX SDK | Status | Notes |
|-----------------|----------------|---------------|--------|-------|
| AutoCAD 2021 | 4.8 | 2021 | Supported | LTS version |
| AutoCAD 2022 | 4.8 | 2022 | Supported | |
| AutoCAD 2023 | 4.8 | 2023 | Supported | |
| AutoCAD 2024 | 4.8 | 2024 | **Recommended** | Latest stable |
| AutoCAD 2025 | 4.8 | 2025 | Testing | Verify API changes |

**Key API Requirements:**
- `Application.DocumentManager.ExecuteInCommandContextAsync` - Available 2015+
- `Application.Idle` event - All versions
- `System.Net.HttpListener` - .NET 4.8 native

### Revit Compatibility

| Revit Version | .NET Framework | pyRevit Version | Status | Notes |
|---------------|----------------|-----------------|--------|-------|
| Revit 2021 | 4.8 | 4.8.x | Supported | End of support 2024 |
| Revit 2022 | 4.8 | 4.8.x | Supported | |
| Revit 2023 | 4.8 | 4.8.x | Supported | |
| Revit 2024 | 4.8 | 4.8.x | **Recommended** | |
| Revit 2025 | 8.0 | 5.0+ | Testing | **.NET 8 migration** |

**Critical Note:** Revit 2025 moves to .NET 8. pyRevit 5.0+ required.

### pyRevit Versions

| pyRevit Version | Revit Support | Routes API | Status |
|-----------------|---------------|------------|--------|
| 4.8.12 | 2019-2024 | Yes | **Recommended** |
| 4.8.13+ | 2019-2024 | Yes | Supported |
| 5.0.x | 2025+ | Yes | For Revit 2025 only |

**Routes API Port Configuration:**
```python
# In pyrevit.config or extension startup
from pyrevit import routes
routes.set_port(int(os.environ.get("MCP_LISTENER_PORT", 48884)))
```

## 2. Python Environment

### MCP Middleware Requirements

| Component | Minimum | Recommended | Maximum |
|-----------|---------|-------------|---------|
| Python | 3.9 | 3.11 | 3.12 |
| FastMCP | 0.1.0 | Latest | - |
| httpx | 0.24.0 | 0.27.x | - |
| asyncio | stdlib | stdlib | - |
| SQLite | 3.35 | 3.45 | - |

### Python Dependencies (requirements.txt)
```
fastmcp>=0.1.0
httpx>=0.27.0
chainlit>=1.0.0
tenacity>=8.2.0
python-dotenv>=1.0.0
aiosqlite>=0.19.0
pydantic>=2.0.0
```

### Chainlit Frontend

| Chainlit Version | Python | Features | Status |
|------------------|--------|----------|--------|
| 0.7.x | 3.8+ | Basic Steps UI | Legacy |
| 1.0.x | 3.9+ | Full Steps, Auth | **Recommended** |
| 1.1.x | 3.10+ | Enhanced UI | Supported |

## 3. Infrastructure Requirements

### AWS Instance Specifications

| Component | Minimum | Recommended | Notes |
|-----------|---------|-------------|-------|
| Instance Type | g4dn.2xlarge | **g4dn.4xlarge** | 4 users/instance |
| vCPU | 8 | 16 | CAD is single-threaded but needs headroom |
| RAM | 32 GB | 64 GB | Revit models can be large |
| GPU | NVIDIA T4 | NVIDIA T4 | vGPU time-slicing |
| Storage (OS) | 100 GB SSD | 200 GB SSD | Windows + Apps |
| Storage (Temp) | NVMe | NVMe Instance Store | Ephemeral, fast I/O |

### Windows Server

| OS Version | Status | Notes |
|------------|--------|-------|
| Windows Server 2019 | Supported | Ensure latest patches |
| Windows Server 2022 | **Recommended** | Better RDP performance |

### NVIDIA Drivers

| Driver Type | Version | Status |
|-------------|---------|--------|
| NVIDIA GRID | 16.x | **Recommended** |
| NVIDIA GRID | 15.x | Supported |

## 4. LLM Provider Compatibility

| Provider | Model | Context Window | Tool Calling | Status |
|----------|-------|----------------|--------------|--------|
| OpenAI | GPT-4o | 128K | Yes | **Recommended** |
| OpenAI | GPT-4-turbo | 128K | Yes | Supported |
| Anthropic | Claude 3.5 Sonnet | 200K | Yes | **Recommended** |
| Anthropic | Claude 3 Opus | 200K | Yes | Supported |
| Azure OpenAI | GPT-4o | 128K | Yes | Enterprise option |

**MCP Tool Calling Requirements:**
- Function/tool calling support required
- JSON mode recommended
- Streaming support for real-time feedback

## 5. Network & Security Requirements

### Port Ranges

| Component | Default Port | Dynamic Range | Protocol |
|-----------|--------------|---------------|----------|
| Chainlit UI | 8000 | 8000-8100 | HTTP |
| MCP Server (SSE) | 54321 | 54000-55000 | HTTP/SSE |
| AutoCAD Sidecar | N/A | 20000-30000 | HTTP |
| Revit Sidecar | 48884 | 20000-30000 | HTTP |

### Firewall Rules

| Rule | Source | Destination | Port | Action |
|------|--------|-------------|------|--------|
| User to Chainlit | User IP | Server | 8000 | Allow |
| Localhost IPC | 127.0.0.1 | 127.0.0.1 | 20000-55000 | Allow |
| Block External Sidecar | 0.0.0.0/0 | Server | 20000-30000 | **Deny** |

## 6. Development Environment

### Recommended IDE Setup

| Tool | Purpose | Version |
|------|---------|---------|
| Visual Studio 2022 | .NET Sidecar development | 17.8+ |
| VS Code | Python development | Latest |
| Python Extension | Linting, debugging | Latest |
| Pylance | Type checking | Latest |

### Build Tools

| Tool | Version | Purpose |
|------|---------|---------|
| MSBuild | 17.x | .NET compilation |
| ILRepack | 2.0.x | Assembly merging (replaces ILMerge) |
| pip | 23.x+ | Python packages |
| pytest | 8.x | Testing |

## 7. Upgrade Path

### From Revit 2024 to 2025

1. **pyRevit Migration:**
   - Install pyRevit 5.0+
   - Update extension to support .NET 8 APIs
   - Test all routes for compatibility

2. **Breaking Changes:**
   - Some Revit API methods deprecated
   - Verify all FilteredElementCollector usage
   - Test ExternalEvent marshaling

### From AutoCAD 2024 to 2025

1. **ObjectARX Update:**
   - Recompile against new SDK
   - Test all API calls

2. **Expected Changes:**
   - Minimal breaking changes
   - Verify transaction handling

## 8. Known Issues & Workarounds

| Issue | Affected Versions | Workaround |
|-------|-------------------|------------|
| pyRevit Routes slow startup | 4.8.x | Lazy-load routes module |
| HttpListener URL reservation | All Windows | Run `netsh http add urlacl` as admin |
| AutoCAD COM timeout | 2021-2023 | Increase `Application.SetSystemVariable("CMDECHO", 0)` |
| Revit Journal file locks | All | Redirect journals to NVMe |

## 9. Deprecation Schedule

| Component | Deprecation Date | End of Support | Action Required |
|-----------|------------------|----------------|-----------------|
| Revit 2021 | 2024-01-01 | 2024-06-30 | Upgrade to 2024 |
| AutoCAD 2021 | 2024-06-01 | 2025-01-01 | Upgrade to 2024 |
| Python 3.9 | 2025-01-01 | 2025-10-01 | Migrate to 3.11 |
| ILMerge | Immediate | N/A | Use ILRepack |
