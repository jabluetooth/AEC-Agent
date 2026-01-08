# Phase 2: AutoCAD .NET Sidecar (The IPC Bridge)

**Objective:** Build an in-process HTTP listener to drive the AutoCAD database safely.

## 1. Project Setup
*   **Framework:** .NET Framework 4.8 (check specific AutoCAD version requirements).
*   **References:** `acmgd.dll`, `acdbmgd.dll`, `accoremgd.dll`.
*   **Build Tool:** Use **ILMerge** to bundle any dependencies (like `Newtonsoft.Json`) into the final DLL. This prevents "DLL Hell" where your plugin's dependencies conflict with AutoCAD's internal libraries.

## 2. Initialize Listener (`IExtensionApplication`)
*   Implement the `IExtensionApplication` interface.
*   In `Initialize()`:
    1.  Read `Environment.GetEnvironmentVariable("MCP_LISTENER_PORT")`.
    2.  Start `System.Net.HttpListener` bound to `http://127.0.0.1:{PORT}/`.
    3.  **CRITICAL:** Spin off the listener loop to a `Task.Run()` background thread to avoid freezing the AutoCAD splash screen.

## 3. Thread Marshaling Strategy
AutoCAD is a Single-Threaded Apartment (STA) application. You **cannot** modify the database from the HttpListener thread.

*   **The Queue Pattern:**
    *   Create a `static ConcurrentQueue<Action> _jobQueue`.
    *   When an HTTP request comes in, parse it, wrap the API work in an `Action`, and `Enqueue` it.
*   **Execution:**
    *   **Option A (Modern):** Use `Application.DocumentManager.ExecuteInCommandContextAsync` (if supported for your specific context).
    *   **Option B (Robust):** Register a handler for the `Application.Idle` event.
        ```csharp
        private static readonly ILogger _logger = LogManager.GetLogger();

        private void OnIdle(object sender, EventArgs e) {
            if (_jobQueue.TryDequeue(out Action action)) {
                Document doc = Application.DocumentManager.MdiActiveDocument;
                if (doc == null) {
                    _logger.Warn("No active document, re-queuing action");
                    _jobQueue.Enqueue(action);
                    return;
                }

                using (DocumentLock docLock = doc.LockDocument()) {
                    using (Transaction tr = doc.TransactionManager.StartTransaction()) {
                        try {
                            action();
                            tr.Commit();
                            _logger.Info("Transaction committed successfully");
                        }
                        catch (Autodesk.AutoCAD.Runtime.Exception acEx) {
                            tr.Abort();
                            _logger.Error($"AutoCAD error: {acEx.ErrorStatus} - {acEx.Message}");
                            // Notify the waiting HTTP request of failure
                            throw;
                        }
                        catch (Exception ex) {
                            tr.Abort();
                            _logger.Error($"Unexpected error: {ex.Message}");
                            throw;
                        }
                    }
                }
            }
        }
        ```

## 4. Atomic Tool Exposure
Define these endpoints mapping to atomic operations:
*   `POST /cad-command/draw_polyline`: Accepts `[[x,y], [x,y]]`.
*   `POST /cad-command/audit_layers`: Returns JSON list of layers.
*   **Security Check:** Every request headers must contain `Authorization: {SESSION_TOKEN}` matching the environment variable. Reject if mismatch.
