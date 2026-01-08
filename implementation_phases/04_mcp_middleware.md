# Phase 4: Python MCP Middleware (The Intelligence Layer)

**Objective:** Create the semantic bridge that translates natural language into geometric commands.

## 1. Server Development
*   **Framework:** Use **FastMCP** (or `mcp` SDK) in Python.
*   **Transport:** Configure **SSE (Server-Sent Events) over HTTP**.
    *   *Why?* The MCP Server must run as a standalone process (orchestrator) that persists even if the Chatbot UI (Chainlit) is refreshed or closed.

## 2. Concurrency Control (The Foreman)
*   **Problem:** The LLM might output 5 distinct tool calls: "Draw line 1", "Draw line 2", ... simultaneously (or close to it).
*   **Solution:**
    *   Implement an **`asyncio.Lock`** or semaphore within the Python server.
    *   Ensure the agent processes **one geometric operation at a time**.
    *   Wait for the Sidecar to return "200 OK" (Action Completed) before sending the next command. This prevents race conditions in the CAD database.

## 3. Tool Discovery & Dynamic Routing
*   The middleware must read `MCP_LISTENER_PORT` to know which port the local AutoCAD/Revit instances are using.
*   **Example Tool Implementation:**

    ```python
    import httpx
    from tenacity import retry, stop_after_attempt, wait_exponential

    # Configure timeouts: 120s for CAD operations, 5s for connection
    SIDECAR_TIMEOUT = httpx.Timeout(120.0, connect=5.0)

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        reraise=True
    )
    async def call_sidecar(url: str, payload: dict, token: str) -> dict:
        """Call sidecar with retry logic and proper timeout handling."""
        async with httpx.AsyncClient(timeout=SIDECAR_TIMEOUT) as client:
            resp = await client.post(
                url,
                json=payload,
                headers={"Authorization": f"Bearer {token}"}
            )
            resp.raise_for_status()
            return resp.json()

    @mcp.tool()
    async def draw_wall(start: list, end: list, level: str):
        """Creates a wall in Revit."""
        port = os.getenv("MCP_LISTENER_PORT")
        token = os.getenv("SESSION_TOKEN")

        if not port or not token:
            return {"success": False, "error": {"code": 4001, "message": "Missing environment configuration"}}

        # 1. Validate input coordinates
        if not all(isinstance(c, (int, float)) for c in start + end):
            return {"success": False, "error": {"code": 4010, "message": "Invalid coordinates"}}

        # 2. Lookup Level ID from SQLite Cache (Fast)
        level_id = sqlite_db.execute("SELECT id FROM levels WHERE name=?", (level,))
        if not level_id:
            return {"success": False, "error": {"code": 4011, "message": f"Level '{level}' not found"}}

        # 3. Call Revit Sidecar with timeout and retry
        url = f"http://127.0.0.1:{port}/mcp/create_wall"
        try:
            result = await call_sidecar(url, {
                "start": start,
                "end": end,
                "level_id": level_id
            }, token)
            return result
        except httpx.TimeoutException:
            return {"success": False, "error": {"code": 4002, "message": "Sidecar timeout"}}
        except httpx.HTTPStatusError as e:
            return {"success": False, "error": {"code": 4003, "message": f"Sidecar error: {e.response.status_code}"}}
    ```
