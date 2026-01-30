# AEC Agent Scripts

Scripts for deploying and running AEC Agent in enterprise and development environments.

## Quick Start (Development)

For local development without GPO:

```powershell
# Option 1: Set up environment only
.\Start-AECAgent-Dev.ps1

# Option 2: Set up environment + launch AutoCAD
.\Start-AECAgent-Dev.ps1 -LaunchAutoCAD

# Option 3: Set up environment + launch AutoCAD + launch Agent
.\Start-AECAgent-Dev.ps1 -LaunchAutoCAD -LaunchAgent
```

After running, check that AutoCAD shows:
```
[AEC Agent] Sidecar started on port 20XXX
```

## Scripts Overview

| Script | Purpose |
|--------|---------|
| `AECAgent-Logon.ps1` | GPO logon script - sets SESSION_TOKEN and MCP_LISTENER_PORT |
| `AECAgent-Logon.cmd` | Batch wrapper for GPO deployment |
| `Start-AECAgent-Dev.ps1` | Development helper - sets up env and optionally launches apps |

## GPO Deployment (Enterprise)

### Prerequisites

1. Active Directory domain
2. AutoCAD 2021-2025 installed on workstations
3. AEC Agent sidecar plugin installed in AutoCAD

### Deployment Steps

#### 1. Copy Scripts to Network Share

```
\\domain.local\NETLOGON\AECAgent\
├── AECAgent-Logon.ps1
└── AECAgent-Logon.cmd
```

#### 2. Create Group Policy Object

1. Open **Group Policy Management Console** (gpmc.msc)
2. Right-click your OU → **Create a GPO**
3. Name it: `AEC Agent Configuration`

#### 3. Configure Logon Script

**Option A: User Configuration (Recommended)**
```
User Configuration
└── Policies
    └── Windows Settings
        └── Scripts (Logon/Logoff)
            └── Logon
                └── Add: \\domain.local\NETLOGON\AECAgent\AECAgent-Logon.cmd
```

**Option B: Computer Configuration**
```
Computer Configuration
└── Policies
    └── Windows Settings
        └── Scripts (Startup/Shutdown)
            └── Startup
                └── Add: \\domain.local\NETLOGON\AECAgent\AECAgent-Logon.cmd
```

#### 4. Configure PowerShell Execution Policy (if needed)

If PowerShell scripts are blocked:
```
Computer Configuration
└── Policies
    └── Administrative Templates
        └── Windows Components
            └── Windows PowerShell
                └── Turn on Script Execution: Enabled
                    └── Execution Policy: Allow local scripts and remote signed scripts
```

### Verification

After user logs in:

1. **Check environment variables:**
   ```powershell
   echo $env:SESSION_TOKEN
   echo $env:MCP_LISTENER_PORT
   ```

2. **Check log file:**
   ```
   %LOCALAPPDATA%\AECAgent\logs\logon.log
   ```

3. **Launch AutoCAD and check command line:**
   ```
   [AEC Agent] Sidecar started on port 20XXX
   ```

## How It Works

### Session Flow

```
┌─────────────────────────────────────────────────────────────┐
│  1. User Logs In                                            │
│     └── GPO runs AECAgent-Logon.ps1                         │
│         ├── Generates SESSION_TOKEN (UUID)                  │
│         ├── Finds available MCP_LISTENER_PORT (20000-30000) │
│         └── Sets as User environment variables              │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│  2. User Launches AutoCAD                                   │
│     └── AutoCAD inherits environment variables              │
│         └── Sidecar plugin reads:                           │
│             ├── SESSION_TOKEN → for auth validation         │
│             └── MCP_LISTENER_PORT → HTTP server port        │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│  3. User Runs AEC Agent                                     │
│     └── aec-agent launch                                    │
│         ├── Reads SESSION_TOKEN → sends in HTTP headers     │
│         ├── Reads MCP_LISTENER_PORT → knows where sidecar is│
│         ├── Starts MCP Server (port 54321)                  │
│         └── Starts Chainlit UI (port 8000)                  │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│  4. Communication Flow                                      │
│                                                             │
│     Browser ──► Chainlit ──► MCP Server ──► AutoCAD Sidecar │
│     :8000       (LLM)        :54321         :MCP_LISTENER   │
│                                              PORT            │
│                                                             │
│     Auth: SESSION_TOKEN in HTTP headers                     │
└─────────────────────────────────────────────────────────────┘
```

### Multi-User RDP Environment

Each user session gets unique credentials:

```
┌─────────────────────────────────────────────────────────────┐
│  RDP Server                                                 │
│                                                             │
│  Session 1 (UserA):                                         │
│    SESSION_TOKEN = "abc-123..."                             │
│    MCP_LISTENER_PORT = 20001                                │
│                                                             │
│  Session 2 (UserB):                                         │
│    SESSION_TOKEN = "def-456..."                             │
│    MCP_LISTENER_PORT = 20002                                │
│                                                             │
│  Session 3 (UserC):                                         │
│    SESSION_TOKEN = "ghi-789..."                             │
│    MCP_LISTENER_PORT = 20003                                │
└─────────────────────────────────────────────────────────────┘
```

This ensures:
- Each user's chatbot only controls their AutoCAD instance
- No cross-session interference
- Secure authentication per session

## Troubleshooting

### "Plugin disabled - missing environment variables"

The AutoCAD sidecar shows this if SESSION_TOKEN or MCP_LISTENER_PORT aren't set.

**Solution:**
1. Run the logon script manually: `.\AECAgent-Logon.ps1`
2. Restart AutoCAD (it reads env vars at startup)

### Port conflicts

If you see errors about ports being in use:

```powershell
# Check what's using a port
netstat -ano | findstr :20001

# The logon script will automatically find another available port
```

### View current environment

```powershell
# PowerShell
$env:SESSION_TOKEN
$env:MCP_LISTENER_PORT

# Command Prompt
echo %SESSION_TOKEN%
echo %MCP_LISTENER_PORT%
```

### Clear and reset

```powershell
# Remove existing values
[Environment]::SetEnvironmentVariable("SESSION_TOKEN", $null, "User")
[Environment]::SetEnvironmentVariable("MCP_LISTENER_PORT", $null, "User")

# Re-run logon script
.\AECAgent-Logon.ps1
```
