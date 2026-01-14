# Phase 5: Chainlit Frontend & Orchestration

**Objective:** Provide a transparent, event-driven user interface that orchestrates the MCP server and handles user interactions.


## 1. Project Structure

```
src/aec_agent/
├── ...
├── launcher.py               # Master bootstrapper script
└── frontend/
    ├── __init__.py
    ├── app.py                # Chainlit application
    ├── agent.py              # LLM Agent logic
    └── mcp_client.py         # Client to connect to our own MCP server
```

## 2. The Bootstrapper (`launcher.py`)

The launcher is responsible for setting up the environment and running the components.

### Responsibilities:
1.  **Port Scanning**: Find available ports for MCP (default 54321) and Chainlit (default 8000).
2.  **Token Generation**: Generate a secure session token for sidecar authentication.
3.  **Process Management**: Launch the MCP server and Chainlit UI as subprocesses.
4.  **Signal Handling**: Ensure all processes terminate when the launcher is stopped.

```python
"""
AEC Agent Launcher.
Orchestrates the MCP server and Chainlit frontend.
"""
import sys
import time
import uuid
import socket
import logging
import subprocess
import signal
from pathlib import Path
from contextlib import closing

# ... (imports)

def find_free_port(start_port: int = 8000) -> int:
    """Find a free port starting from start_port."""
    # ... implementation ...

def main():
    # 1. Setup Environment
    session_token = str(uuid.uuid4())
    mcp_port = find_free_port(54321)
    chainlit_port = find_free_port(8000)
    
    env = os.environ.copy()
    env["MCP_SERVER_PORT"] = str(mcp_port)
    env["CHAINLIT_PORT"] = str(chainlit_port)
    env["SESSION_TOKEN"] = session_token
    
    # 2. Start MCP Server
    mcp_process = subprocess.Popen(
        [sys.executable, "-m", "aec_agent.server"],
        env=env
    )
    
    # 3. Start Chainlit
    # ... implementation ...
```

## 3. MCP Client Connection (`mcp_client.py`)

Bridging the Chainlit frontend to our running FastMCP server using SSE.

```python
"""
MCP Client for the Frontend.
Connects to the local FastMCP server via SSE to discover and use tools.
"""
from mcp import ClientSession, StdioServerParameters
from mcp.client.sse import sse_client
# ...
```

## 4. The Agent Logic (`agent.py`)

The intelligence loop that drives the conversation.

*   **Model**: GPT-4o or Claude 3.5 Sonnet (via API).
*   **System Prompt**: Defined to enforce the role of an AEC Specialist.
*   **Tool Usage**: Maps LLM tool calls to MCP client calls.

## 5. Chainlit Interface (`app.py`)

The user-facing application file.

*   **Startup**: Connects to the MCP server.
*   **Message Handling**: Passes user input to the Agent.
*   **"Steps" UI**: uses `cl.Step` to show the agent's thought process and tool executions.

```python
@cl.on_message
async def on_message(message: cl.Message):
    # 1. Initialize Agent
    # 2. Get Response
    # 3. Stream back to user
```
