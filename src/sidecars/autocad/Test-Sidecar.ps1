<#
.SYNOPSIS
    Tests the AEC Agent AutoCAD Sidecar.
.EXAMPLE
    .\Test-Sidecar.ps1 -Port 20001 -Token "test-token"
#>
param([int]$Port = 20001, [string]$Token = "test-token-12345")

$baseUrl = "http://localhost:$Port"
$headers = @{ "Authorization" = $Token; "Content-Type" = "application/json" }

Write-Host "`n=== AEC Agent Sidecar Test ===" -ForegroundColor Cyan
Write-Host "Target: $baseUrl`n"

# Health Check
Write-Host "1. Health Check" -ForegroundColor Yellow
try {
    $r = Invoke-RestMethod -Uri "$baseUrl/health" -Method GET -TimeoutSec 5
    Write-Host "[PASS] Status: $($r.status)" -ForegroundColor Green
} catch {
    Write-Host "[FAIL] $($_.Exception.Message)" -ForegroundColor Red
    Write-Host "`nSidecar not responding. Check:" -ForegroundColor Red
    Write-Host "  - AutoCAD is running with plugin loaded"
    Write-Host "  - Environment variables are set"
    exit 1
}

# Audit Layers
Write-Host "`n2. Audit Layers" -ForegroundColor Yellow
try {
    $body = '{"command":"audit_layers","parameters":{}}'
    $r = Invoke-RestMethod -Uri "$baseUrl/" -Method POST -Headers $headers -Body $body -TimeoutSec 10
    if ($r.success) { Write-Host "[PASS] Found $($r.data.count) layers" -ForegroundColor Green }
    else { Write-Host "[FAIL] $($r.error.message)" -ForegroundColor Red }
} catch { Write-Host "[FAIL] $($_.Exception.Message)" -ForegroundColor Red }

# Draw Line
Write-Host "`n3. Draw Line" -ForegroundColor Yellow
try {
    $body = '{"command":"draw_line","parameters":{"start":[0,0,0],"end":[100,100,0]}}'
    $r = Invoke-RestMethod -Uri "$baseUrl/" -Method POST -Headers $headers -Body $body -TimeoutSec 10
    if ($r.success) { Write-Host "[PASS] Created: $($r.data.handle)" -ForegroundColor Green }
    else { Write-Host "[FAIL] $($r.error.message)" -ForegroundColor Red }
} catch { Write-Host "[FAIL] $($_.Exception.Message)" -ForegroundColor Red }

# Draw Circle
Write-Host "`n4. Draw Circle" -ForegroundColor Yellow
try {
    $body = '{"command":"draw_circle","parameters":{"center":[50,50,0],"radius":25}}'
    $r = Invoke-RestMethod -Uri "$baseUrl/" -Method POST -Headers $headers -Body $body -TimeoutSec 10
    if ($r.success) { Write-Host "[PASS] Created: $($r.data.handle)" -ForegroundColor Green }
    else { Write-Host "[FAIL] $($r.error.message)" -ForegroundColor Red }
} catch { Write-Host "[FAIL] $($_.Exception.Message)" -ForegroundColor Red }

Write-Host "`n=== Test Complete ===" -ForegroundColor Cyan
Write-Host "Use ZOOM EXTENTS in AutoCAD to see geometry.`n"
