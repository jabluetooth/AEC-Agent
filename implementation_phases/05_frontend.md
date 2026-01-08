# Phase 5: Chainlit Frontend & Orchestration

**Objective:** Provide a transparent, event-driven user interface.

## 1. Interface Build
*   **Framework:** **Chainlit**.
*   **Key Feature:** Use the **"Steps" UI** (`cl.Step`).
    *   *Why?* In Engineering, trust is paramount. The user must see the "Chain of Thought" and the specific JSON arguments being sent to the CAD tool ("Deleting 50 elements...") *before* execution, or at least in real-time.

## 2. LLM Selection
*   **Model:** **GPT-4o** or **Claude 3.5 Sonnet**.
*   *Requirement:* High reasoning capability is needed to decompose abstract intents ("Make this room bigger") into specific atomic API calls ("Move Wall A by 5ft", "Move Wall B by 5ft").

## 3. The Bootstrapper Script
Create a `start_agent.py` master script acting as the launcher.

1.  **Scan for Ports:** Find a free port for MCP (e.g., 54321) and Chainlit (e.g., 8000).
2.  **Generate Token:** Create the UUID.
3.  **Launch MCP:** `subprocess.Popen(["python", "mcp_server.py"], env={...})`
4.  **Launch Chainlit:** `subprocess.Popen(["chainlit", "run", "app.py", "--port", "8000"], env={...})`
5.  **Launch CAD (Optional):** Or simply instruct the user to open CAD, where the GPO script has already set the environment variables.
