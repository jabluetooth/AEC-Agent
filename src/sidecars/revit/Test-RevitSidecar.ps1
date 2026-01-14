<#
.SYNOPSIS
    Test script for AEC Agent Revit Sidecar

.DESCRIPTION
    Tests all HTTP endpoints exposed by the Revit sidecar.
    Requires Revit to be running with a document open and the AEC Agent extension loaded.

.PARAMETER Port
    The port the sidecar is listening on. Default: value of MCP_LISTENER_PORT env var or 20001

.PARAMETER Token
    The session token for authentication. Default: value of SESSION_TOKEN env var or "test-token"

.EXAMPLE
    .\Test-RevitSidecar.ps1

.EXAMPLE
    .\Test-RevitSidecar.ps1 -Port 25000 -Token "my-secret-token"
#>

param(
    [int]$Port = $($env:MCP_LISTENER_PORT ?? 20001),
    [string]$Token = $($env:SESSION_TOKEN ?? "test-token")
)

$baseUrl = "http://localhost:$Port"

$headers = @{
    "X-Session-Token" = $Token
    "Content-Type" = "application/json"
}

function Test-Endpoint {
    param(
        [string]$Name,
        [string]$Method,
        [string]$Url,
        [string]$Body = $null,
        [bool]$RequireAuth = $true
    )

    Write-Host "`n[$Name]" -ForegroundColor Yellow
    Write-Host "  $Method $Url" -ForegroundColor Gray

    try {
        $params = @{
            Uri = $Url
            Method = $Method
            ErrorAction = "Stop"
        }

        if ($RequireAuth) {
            $params.Headers = $headers
        }

        if ($Body) {
            $params.Body = $Body
            Write-Host "  Body: $Body" -ForegroundColor Gray
        }

        $response = Invoke-RestMethod @params
        $json = $response | ConvertTo-Json -Depth 10

        if ($response.success) {
            Write-Host "  Status: SUCCESS" -ForegroundColor Green
        } else {
            Write-Host "  Status: FAILED" -ForegroundColor Red
        }

        Write-Host $json -ForegroundColor Cyan
        return $response
    }
    catch {
        Write-Host "  Status: ERROR" -ForegroundColor Red
        Write-Host "  $_" -ForegroundColor Red
        return $null
    }
}

# Header
Write-Host "================================================" -ForegroundColor Cyan
Write-Host "AEC Agent Revit Sidecar Test Suite" -ForegroundColor Cyan
Write-Host "================================================" -ForegroundColor Cyan
Write-Host "Base URL: $baseUrl"
Write-Host "Token: $(if ($Token) { '***configured***' } else { 'NOT SET' })"
Write-Host "================================================" -ForegroundColor Cyan

# Test 1: Health Check (no auth)
Test-Endpoint -Name "Health Check" -Method "GET" -Url "$baseUrl/health" -RequireAuth $false

# Test 2: Document Status
Test-Endpoint -Name "Document Status" -Method "GET" -Url "$baseUrl/mcp/status"

# Test 3: List Levels
Test-Endpoint -Name "List Levels" -Method "GET" -Url "$baseUrl/mcp/levels"

# Test 4: List Walls
Test-Endpoint -Name "List Walls" -Method "GET" -Url "$baseUrl/mcp/walls"

# Test 5: List Rooms (live)
Test-Endpoint -Name "List Rooms (Live)" -Method "GET" -Url "$baseUrl/mcp/rooms"

# Test 6: List Rooms (cached)
Test-Endpoint -Name "List Rooms (Cached)" -Method "GET" -Url "$baseUrl/mcp/rooms/cached"

# Test 7: Cache Status
Test-Endpoint -Name "Cache Status" -Method "GET" -Url "$baseUrl/mcp/cache/status"

# Test 8: Sync Cache
Test-Endpoint -Name "Sync Cache" -Method "POST" -Url "$baseUrl/mcp/cache/sync" -Body '{"categories": ["rooms", "levels"]}'

# Test 9: Create Level (optional - commented out to avoid modifying model)
# Test-Endpoint -Name "Create Level" -Method "POST" -Url "$baseUrl/mcp/levels/create" -Body '{"name": "Test Level", "elevation": 15.0}'

# Test 10: Unauthorized request (should fail)
Write-Host "`n[Unauthorized Request Test]" -ForegroundColor Yellow
Write-Host "  GET $baseUrl/mcp/status (no token)" -ForegroundColor Gray
try {
    $response = Invoke-RestMethod -Uri "$baseUrl/mcp/status" -Method GET -ErrorAction Stop
    Write-Host "  Status: UNEXPECTED SUCCESS (security issue!)" -ForegroundColor Red
}
catch {
    if ($_.Exception.Response.StatusCode -eq 401 -or $_.ToString() -match "Unauthorized") {
        Write-Host "  Status: CORRECTLY REJECTED" -ForegroundColor Green
    } else {
        Write-Host "  Status: ERROR - $_" -ForegroundColor Red
    }
}

# Summary
Write-Host "`n================================================" -ForegroundColor Cyan
Write-Host "Test Suite Complete" -ForegroundColor Cyan
Write-Host "================================================" -ForegroundColor Cyan
