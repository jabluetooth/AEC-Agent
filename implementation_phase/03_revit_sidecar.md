# Phase 3: Revit Sidecar (Rapid RAD Deployment)

**Objective:** Expose the Revit API using pyRevit to bypass complex C# boilerplate.

## 1. Framework Deployment
*   Install **pyRevit** command-line utility (CLI) on the master image.
*   Configure a shared extension directory.

## 1.1 Port Configuration
*   **Problem:** pyRevit Routes defaults to port 48884, which conflicts in multi-session environments.
*   **Solution:** Configure dynamic port on extension startup.
*   Create a file `startup.py` in your extension:

    ```python
    import os
    from pyrevit import routes
    from pyrevit.coreutils import logger

    def configure_routes_port():
        """Configure Routes API to use session-specific port."""
        port = os.environ.get("MCP_LISTENER_PORT")
        if port:
            try:
                routes.set_port(int(port))
                logger.info(f"Routes API configured on port {port}")
            except Exception as e:
                logger.error(f"Failed to set port: {e}")
                # Fallback to default
                routes.set_port(48884)
        else:
            logger.warn("MCP_LISTENER_PORT not set, using default 48884")

    # Run on extension load
    configure_routes_port()
    ```

## 2. Implement Routes
*   Use pyRevit's **Routes API** to create a web server without writing socket code.
*   Create a file `routes.py` in your extension:

    ```python
    from pyrevit import routes, revit, DB
    
    @routes.route('/mcp/create_wall', methods=['POST'])
    def mcp_create_wall(request):
        # Security: Check Session Token
        # ...
        
        # Geometry Logic
        # pyRevit handles the ExternalEvent automatically here
        doc = revit.doc
        with revit.Transaction("AI Create Wall"):
            wall = DB.Wall.Create(doc, ...)
            
        return "Success"
    ```

## 3. The "Cached Resource" Pattern
*   **Problem:** Querying the Revit model (e.g., "Find all 500 rooms") via API is slow and blocks the UI.
*   **Solution:**
    *   Create a "Sync" hook (on Document Open/Save).
    *   Export lightweight metadata (Element IDs, Names, Parameters, Locations) to a local **SQLite** database file.
    *   The MCP Middleware reads *this* SQLite file for all read-only queries (`mcp://room_list`), offering <50ms response times.
    *   Only wake the Revit API for *write* operations (Create/Modify/Delete).

### SQLite Cache Location Strategy
*   **Path:** `%LOCALAPPDATA%\AECAgent\cache\{document_hash}.sqlite`
*   **Why not %TEMP%?** Temp folder may be cleared; cache should persist across sessions.
*   **Document Hash:** Use MD5 of document path to avoid filename collisions.

    ```python
    import os
    import hashlib
    from pathlib import Path

    def get_cache_path(doc_path: str) -> Path:
        """Generate unique cache path for each document."""
        cache_dir = Path(os.environ.get("LOCALAPPDATA")) / "AECAgent" / "cache"
        cache_dir.mkdir(parents=True, exist_ok=True)

        doc_hash = hashlib.md5(doc_path.encode()).hexdigest()[:12]
        return cache_dir / f"{doc_hash}.sqlite"
    ```
