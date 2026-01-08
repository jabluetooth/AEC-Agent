# Phase 8: Testing & Validation

**Objective:** Establish comprehensive testing strategies to ensure reliability, performance, and security across all system layers.

## 1. Testing Pyramid

```
                    /\
                   /  \
                  / E2E \        <- 10% (Slow, Expensive)
                 /--------\
                /Integration\    <- 30% (Medium)
               /--------------\
              /   Unit Tests   \ <- 60% (Fast, Cheap)
             /------------------\
```

## 2. Unit Testing

### AutoCAD Sidecar (C#)
**Framework:** NUnit or xUnit with Moq

```csharp
[TestFixture]
public class GeometryToolsTests
{
    [Test]
    public void DrawPolyline_ValidPoints_CreatesEntity()
    {
        // Arrange
        var points = new[] { new Point2d(0, 0), new Point2d(10, 10) };
        var mockDb = new Mock<Database>();

        // Act
        var result = GeometryTools.CreatePolyline(mockDb.Object, points, "Layer1");

        // Assert
        Assert.IsNotNull(result);
        Assert.AreEqual(2, result.NumberOfVertices);
    }

    [Test]
    public void DrawPolyline_EmptyPoints_ReturnsError()
    {
        var points = new Point2d[0];

        var result = GeometryTools.CreatePolyline(null, points, "Layer1");

        Assert.IsFalse(result.Success);
        Assert.AreEqual(2004, result.ErrorCode);
    }
}
```

### MCP Middleware (Python)
**Framework:** pytest with pytest-asyncio

```python
import pytest
from unittest.mock import AsyncMock, patch

@pytest.mark.asyncio
async def test_draw_wall_calls_sidecar():
    """Verify draw_wall sends correct payload to Revit sidecar."""
    with patch("httpx.AsyncClient.post") as mock_post:
        mock_post.return_value = AsyncMock(
            status_code=200,
            json=lambda: {"success": True, "element_id": 12345}
        )

        result = await draw_wall(
            start=[0, 0],
            end=[10, 0],
            level="Level 1"
        )

        assert result["success"] is True
        mock_post.assert_called_once()
        call_args = mock_post.call_args
        assert call_args[1]["json"]["start"] == [0, 0]

@pytest.mark.asyncio
async def test_draw_wall_handles_timeout():
    """Verify graceful handling of sidecar timeout."""
    with patch("httpx.AsyncClient.post") as mock_post:
        mock_post.side_effect = httpx.TimeoutException("Connection timed out")

        result = await draw_wall(start=[0, 0], end=[10, 0], level="Level 1")

        assert result["success"] is False
        assert result["error"]["code"] == 4002  # Timeout error code
```

### Revit Sidecar (pyRevit/Python)
```python
import pytest
from unittest.mock import MagicMock

def test_create_wall_validates_level():
    """Verify level validation before wall creation."""
    mock_doc = MagicMock()
    mock_doc.GetElement.return_value = None  # Level doesn't exist

    result = create_wall(
        doc=mock_doc,
        start=[0, 0],
        end=[10, 0],
        level_id=99999
    )

    assert result["success"] is False
    assert "level" in result["error"]["message"].lower()
```

## 3. Integration Testing

### Sidecar-to-MCP Integration
**Objective:** Verify end-to-end communication without CAD applications.

```python
import pytest
import subprocess
import time
import httpx

@pytest.fixture(scope="module")
def mock_sidecar():
    """Start a mock sidecar server for integration tests."""
    proc = subprocess.Popen(
        ["python", "tests/mock_sidecar.py", "--port", "29999"]
    )
    time.sleep(2)  # Wait for server startup
    yield "http://127.0.0.1:29999"
    proc.terminate()

@pytest.fixture(scope="module")
def mcp_server(mock_sidecar):
    """Start MCP server pointing to mock sidecar."""
    import os
    env = os.environ.copy()
    env["MCP_LISTENER_PORT"] = "29999"
    env["SESSION_TOKEN"] = "test-token-12345"

    proc = subprocess.Popen(
        ["python", "mcp_server.py"],
        env=env
    )
    time.sleep(3)
    yield "http://127.0.0.1:54321"
    proc.terminate()

@pytest.mark.integration
async def test_full_tool_invocation(mcp_server):
    """Test complete flow from MCP tool to sidecar response."""
    async with httpx.AsyncClient() as client:
        response = await client.post(
            f"{mcp_server}/tools/draw_wall",
            json={"start": [0, 0], "end": [100, 0], "level": "Level 1"}
        )

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
```

### Database Cache Integration
```python
@pytest.mark.integration
def test_sqlite_cache_sync():
    """Verify SQLite cache correctly stores model data."""
    # Setup
    cache = SQLiteCache(":memory:")
    mock_elements = [
        {"id": 1, "name": "Room 101", "level": "Level 1"},
        {"id": 2, "name": "Room 102", "level": "Level 1"},
    ]

    # Act
    cache.sync_rooms(mock_elements)

    # Assert
    rooms = cache.query_rooms(level="Level 1")
    assert len(rooms) == 2
    assert rooms[0]["name"] == "Room 101"
```

## 4. End-to-End Testing

### Test Scenarios

| Scenario | Steps | Expected Outcome |
|----------|-------|------------------|
| Simple Line | User: "Draw a line from 0,0 to 10,10" | Line appears in AutoCAD |
| Wall Creation | User: "Create a 10ft wall on Level 1" | Wall appears in Revit |
| Query Rooms | User: "List all rooms" | Returns room list from cache |
| Error Recovery | Kill sidecar mid-operation | User sees error, system recovers |

### Automated E2E Test Script
```python
import asyncio
from playwright.async_api import async_playwright

async def test_chatbot_creates_wall():
    """E2E test using browser automation."""
    async with async_playwright() as p:
        browser = await p.chromium.launch()
        page = await browser.new_page()

        # Navigate to Chainlit UI
        await page.goto("http://localhost:8000")

        # Send command
        await page.fill("textarea", "Create a wall from 0,0 to 10,0 on Level 1")
        await page.click("button[type='submit']")

        # Wait for response
        await page.wait_for_selector(".step-complete", timeout=30000)

        # Verify success message
        response = await page.text_content(".message-content")
        assert "wall created" in response.lower()

        await browser.close()
```

## 5. Load Testing

### Concurrent User Simulation
**Tool:** Locust

```python
# locustfile.py
from locust import HttpUser, task, between

class MCPUser(HttpUser):
    wait_time = between(1, 3)

    def on_start(self):
        """Setup session token."""
        self.token = "load-test-token"

    @task(3)
    def query_rooms(self):
        """Simulate read-heavy workload."""
        self.client.get(
            "/resources/room_list",
            headers={"Authorization": f"Bearer {self.token}"}
        )

    @task(1)
    def create_element(self):
        """Simulate write operation."""
        self.client.post(
            "/tools/draw_line",
            json={"start": [0, 0], "end": [10, 10]},
            headers={"Authorization": f"Bearer {self.token}"}
        )
```

### Performance Benchmarks

| Metric | Target | Measurement Method |
|--------|--------|-------------------|
| SQLite cache query | < 50ms | pytest-benchmark |
| Sidecar HTTP response | < 200ms | httpx timing |
| Full tool execution | < 5s | End-to-end timer |
| Concurrent users | 50 | Locust load test |
| Memory per session | < 500MB | Windows Performance Monitor |

## 6. Security Testing

### Token Validation Tests
```python
@pytest.mark.security
async def test_request_without_token_rejected():
    """Verify requests without token are rejected."""
    async with httpx.AsyncClient() as client:
        response = await client.post(
            "http://127.0.0.1:20001/cad-command/draw_line",
            json={"start": [0, 0], "end": [10, 10]}
            # No Authorization header
        )

        assert response.status_code == 401

@pytest.mark.security
async def test_invalid_token_rejected():
    """Verify requests with invalid token are rejected."""
    async with httpx.AsyncClient() as client:
        response = await client.post(
            "http://127.0.0.1:20001/cad-command/draw_line",
            json={"start": [0, 0], "end": [10, 10]},
            headers={"Authorization": "Bearer invalid-token"}
        )

        assert response.status_code == 403

@pytest.mark.security
async def test_cross_session_token_rejected():
    """Verify token from another session is rejected."""
    # Token from session 1 should not work on session 2's port
    session1_token = "session-1-token"
    session2_port = 20002  # Different session

    async with httpx.AsyncClient() as client:
        response = await client.post(
            f"http://127.0.0.1:{session2_port}/cad-command/draw_line",
            json={"start": [0, 0], "end": [10, 10]},
            headers={"Authorization": f"Bearer {session1_token}"}
        )

        assert response.status_code == 403
```

### Input Validation Tests
```python
@pytest.mark.security
async def test_sql_injection_prevention():
    """Verify SQLite queries are parameterized."""
    malicious_input = "'; DROP TABLE rooms; --"

    result = await query_rooms(level=malicious_input)

    # Should return empty, not crash
    assert result["success"] is True
    assert len(result["data"]) == 0

@pytest.mark.security
async def test_coordinate_bounds_validation():
    """Verify extremely large coordinates are rejected."""
    response = await draw_line(
        start=[0, 0],
        end=[1e308, 1e308]  # Astronomical value
    )

    assert response["success"] is False
    assert response["error"]["code"] == 4010  # Invalid coordinates
```

## 7. Test Environment Setup

### Docker Compose for Test Infrastructure
```yaml
# docker-compose.test.yml
version: '3.8'
services:
  mock-autocad:
    build: ./tests/mock-autocad
    ports:
      - "20001:20001"
    environment:
      - SESSION_TOKEN=test-token

  mock-revit:
    build: ./tests/mock-revit
    ports:
      - "20002:20002"
    environment:
      - SESSION_TOKEN=test-token

  mcp-server:
    build: ./mcp-middleware
    ports:
      - "54321:54321"
    environment:
      - AUTOCAD_PORT=20001
      - REVIT_PORT=20002
      - SESSION_TOKEN=test-token
    depends_on:
      - mock-autocad
      - mock-revit
```

### CI/CD Pipeline (GitHub Actions)
```yaml
# .github/workflows/test.yml
name: Test Suite

on: [push, pull_request]

jobs:
  unit-tests:
    runs-on: windows-latest
    steps:
      - uses: actions/checkout@v4
      - name: Setup Python
        uses: actions/setup-python@v5
        with:
          python-version: '3.11'
      - name: Install dependencies
        run: pip install -r requirements-test.txt
      - name: Run unit tests
        run: pytest tests/unit -v --cov=src

  integration-tests:
    runs-on: windows-latest
    needs: unit-tests
    steps:
      - uses: actions/checkout@v4
      - name: Start mock services
        run: docker-compose -f docker-compose.test.yml up -d
      - name: Run integration tests
        run: pytest tests/integration -v -m integration
      - name: Cleanup
        run: docker-compose -f docker-compose.test.yml down
```

## 8. Validation Checklist

Before deployment, verify:

- [ ] All unit tests pass (>90% coverage)
- [ ] Integration tests pass with mock sidecars
- [ ] Load test achieves 50 concurrent users
- [ ] Security tests pass (token validation, input sanitization)
- [ ] E2E test completes successfully with real CAD applications
- [ ] Memory usage within limits after 8-hour session
- [ ] Error recovery works (kill sidecar, verify reconnection)
- [ ] Rollback procedures tested and documented
