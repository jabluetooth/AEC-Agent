# Phase 7: Error Handling & Recovery

**Objective:** Define robust error handling patterns to ensure system stability and data integrity across all layers.

## 1. Error Code Taxonomy

Standardize error codes across all components:

| Code Range | Category | Example |
|------------|----------|---------|
| 1000-1999 | Infrastructure | 1001: Port unavailable |
| 2000-2999 | AutoCAD Sidecar | 2001: Document lock failed |
| 3000-3999 | Revit Sidecar | 3001: Transaction aborted |
| 4000-4999 | MCP Middleware | 4001: Sidecar unreachable |
| 5000-5999 | Frontend | 5001: Session expired |

### Standard Error Response Format
```json
{
    "success": false,
    "error": {
        "code": 2001,
        "category": "AUTOCAD_SIDECAR",
        "message": "Failed to acquire document lock",
        "details": "Document is open in read-only mode",
        "timestamp": "2024-01-15T10:30:00Z",
        "request_id": "uuid-here"
    }
}
```

## 2. AutoCAD Error Handling

### Transaction Safety Pattern
```csharp
public async Task<ActionResult> ExecuteSafeTransaction(Action action)
{
    Document doc = Application.DocumentManager.MdiActiveDocument;
    if (doc == null)
        return ActionResult.Fail(2002, "No active document");

    using (DocumentLock docLock = doc.LockDocument())
    {
        using (Transaction tr = doc.TransactionManager.StartTransaction())
        {
            try
            {
                action();
                tr.Commit();
                return ActionResult.Success();
            }
            catch (Autodesk.AutoCAD.Runtime.Exception acEx)
            {
                tr.Abort();
                Logger.Error($"AutoCAD Exception: {acEx.ErrorStatus}");
                return ActionResult.Fail(2003, $"CAD Error: {acEx.Message}");
            }
            catch (Exception ex)
            {
                tr.Abort();
                Logger.Error($"Unexpected Exception: {ex}");
                return ActionResult.Fail(2099, "Unexpected error during transaction");
            }
        }
    }
}
```

### Document State Recovery
*   **Problem:** AutoCAD may crash mid-operation leaving corrupted state.
*   **Solution:**
    1. Enable AutoSave with 5-minute intervals via `SAVETIME` system variable.
    2. Implement pre-operation snapshots using `.bak` file rotation.
    3. On sidecar startup, check for recovery files (`.sv$`) and notify user.

## 3. Revit Error Handling

### pyRevit Transaction Wrapper
```python
from pyrevit import revit, DB
from pyrevit.coreutils import logger

def safe_transaction(doc, action_name, operation):
    """Execute operation within a safe transaction context."""
    try:
        with revit.Transaction(action_name):
            result = operation()
            return {"success": True, "data": result}
    except DB.Exceptions.InvalidOperationException as e:
        logger.error(f"Invalid operation: {e}")
        return {"success": False, "error": {"code": 3002, "message": str(e)}}
    except Exception as e:
        logger.error(f"Transaction failed: {e}")
        return {"success": False, "error": {"code": 3099, "message": str(e)}}
```

### Model Corruption Prevention
*   **Sync Before Heavy Operations:** Force a sync-to-central before batch modifications.
*   **Element Existence Check:** Always verify element IDs exist before modification:
    ```python
    element = doc.GetElement(element_id)
    if element is None or not element.IsValidObject:
        return error_response(3003, "Element no longer exists")
    ```

## 4. MCP Middleware Error Handling

### Circuit Breaker Pattern
```python
import asyncio
from dataclasses import dataclass
from enum import Enum
from datetime import datetime, timedelta

class CircuitState(Enum):
    CLOSED = "closed"      # Normal operation
    OPEN = "open"          # Failing, reject requests
    HALF_OPEN = "half_open"  # Testing recovery

@dataclass
class CircuitBreaker:
    failure_threshold: int = 5
    recovery_timeout: int = 30  # seconds
    _failure_count: int = 0
    _state: CircuitState = CircuitState.CLOSED
    _last_failure: datetime = None

    async def call(self, func, *args, **kwargs):
        if self._state == CircuitState.OPEN:
            if datetime.now() - self._last_failure > timedelta(seconds=self.recovery_timeout):
                self._state = CircuitState.HALF_OPEN
            else:
                raise CircuitBreakerOpen("Service unavailable, circuit is open")

        try:
            result = await func(*args, **kwargs)
            self._on_success()
            return result
        except Exception as e:
            self._on_failure()
            raise

    def _on_success(self):
        self._failure_count = 0
        self._state = CircuitState.CLOSED

    def _on_failure(self):
        self._failure_count += 1
        self._last_failure = datetime.now()
        if self._failure_count >= self.failure_threshold:
            self._state = CircuitState.OPEN
```

### Retry Logic with Exponential Backoff
```python
import httpx
from tenacity import retry, stop_after_attempt, wait_exponential

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=10)
)
async def call_sidecar(url: str, payload: dict, token: str):
    async with httpx.AsyncClient(timeout=httpx.Timeout(120.0, connect=5.0)) as client:
        response = await client.post(
            url,
            json=payload,
            headers={"Authorization": f"Bearer {token}"}
        )
        response.raise_for_status()
        return response.json()
```

## 5. User Notification Patterns

### Severity Levels
| Level | UI Treatment | Example |
|-------|--------------|---------|
| INFO | Toast notification | "Wall created successfully" |
| WARNING | Yellow banner, dismissable | "3 elements were skipped" |
| ERROR | Red banner, requires acknowledgment | "Transaction failed, changes rolled back" |
| CRITICAL | Modal dialog, blocks UI | "Connection to CAD lost, please restart" |

### Chainlit Error Display
```python
import chainlit as cl

async def notify_error(code: int, message: str, severity: str = "error"):
    """Send formatted error to user interface."""
    await cl.Message(
        content=f"**Error {code}:** {message}",
        author="System",
        type="error" if severity == "error" else "system_message"
    ).send()

    # Log for audit trail
    logger.error(f"[{code}] {message}")
```

## 6. Rollback Strategies

### Multi-Step Operation Rollback
For complex operations spanning multiple API calls:

```python
class TransactionContext:
    def __init__(self):
        self.completed_steps = []
        self.rollback_actions = []

    def add_step(self, step_name: str, rollback_action: callable):
        self.completed_steps.append(step_name)
        self.rollback_actions.append(rollback_action)

    async def rollback(self):
        """Execute rollback in reverse order."""
        for action in reversed(self.rollback_actions):
            try:
                await action()
            except Exception as e:
                logger.error(f"Rollback step failed: {e}")
                # Continue with remaining rollbacks
```

### Usage Example
```python
ctx = TransactionContext()
try:
    # Step 1: Create level
    level_id = await create_level(...)
    ctx.add_step("create_level", lambda: delete_level(level_id))

    # Step 2: Create walls on level
    wall_ids = await create_walls(level_id, ...)
    ctx.add_step("create_walls", lambda: delete_elements(wall_ids))

    # Step 3: Add doors to walls
    await add_doors(wall_ids, ...)

except Exception as e:
    await ctx.rollback()
    raise OperationFailed(f"Multi-step operation failed: {e}")
```

## 7. Health Check Endpoints

Each component must expose a health endpoint:

### Sidecar Health Check
```
GET /health
Response: {"status": "healthy", "cad_version": "2024", "document_open": true}
```

### MCP Middleware Health Check
```
GET /health
Response: {
    "status": "healthy",
    "sidecars": {
        "autocad": {"reachable": true, "latency_ms": 12},
        "revit": {"reachable": true, "latency_ms": 8}
    },
    "cache": {"sqlite_connected": true, "last_sync": "2024-01-15T10:00:00Z"}
}
```
