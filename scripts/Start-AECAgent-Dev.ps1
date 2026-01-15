<#
.SYNOPSIS
    AEC Agent Development Launcher

.DESCRIPTION
    Sets up the environment and optionally launches AutoCAD and AEC Agent
    for local development. This is a convenience script that replaces
    the need for GPO in development environments.

.PARAMETER LaunchAutoCAD
    If specified, launches AutoCAD after setting environment variables.

.PARAMETER AutoCADPath
    Path to AutoCAD executable. Auto-detected if not specified.

.PARAMETER LaunchAgent
    If specified, launches the AEC Agent (MCP Server + Chainlit).

.PARAMETER SkipEnvSetup
    Skip environment variable setup (use existing values).

.EXAMPLE
    # Just set up environment
    .\Start-AECAgent-Dev.ps1

    # Set up environment and launch AutoCAD
    .\Start-AECAgent-Dev.ps1 -LaunchAutoCAD

    # Set up environment and launch everything
    .\Start-AECAgent-Dev.ps1 -LaunchAutoCAD -LaunchAgent

    # Launch agent only (env already set)
    .\Start-AECAgent-Dev.ps1 -LaunchAgent -SkipEnvSetup
#>

[CmdletBinding()]
param(
    [switch]$LaunchAutoCAD,
    [string]$AutoCADPath,
    [switch]$LaunchAgent,
    [switch]$SkipEnvSetup
)

$ErrorActionPreference = "Stop"

# =============================================================================
# Helper Functions
# =============================================================================

function Find-AutoCAD {
    # Common AutoCAD installation paths
    $searchPaths = @(
        "C:\Program Files\Autodesk\AutoCAD 2026\acad.exe",
        "C:\Program Files\Autodesk\AutoCAD 2025\acad.exe",
        "C:\Program Files\Autodesk\AutoCAD 2024\acad.exe",
        "C:\Program Files\Autodesk\AutoCAD 2023\acad.exe",
        "C:\Program Files\Autodesk\AutoCAD 2022\acad.exe",
        "C:\Program Files\Autodesk\AutoCAD 2021\acad.exe"
    )

    foreach ($path in $searchPaths) {
        if (Test-Path $path) {
            return $path
        }
    }

    return $null
}

function Show-Banner {
    Write-Host ""
    Write-Host "  ================================================" -ForegroundColor Cyan
    Write-Host "       AEC Agent Development Launcher" -ForegroundColor Cyan
    Write-Host "  ================================================" -ForegroundColor Cyan
    Write-Host ""
}

function Show-Status {
    $token = $env:SESSION_TOKEN
    $port = $env:MCP_LISTENER_PORT

    Write-Host "  Environment Status:" -ForegroundColor Yellow
    Write-Host "  -------------------"

    if ($token) {
        Write-Host "  SESSION_TOKEN:      " -NoNewline
        Write-Host $token -ForegroundColor Green
    }
    else {
        Write-Host "  SESSION_TOKEN:      " -NoNewline
        Write-Host "NOT SET" -ForegroundColor Red
    }

    if ($port) {
        Write-Host "  MCP_LISTENER_PORT:  " -NoNewline
        Write-Host $port -ForegroundColor Green
    }
    else {
        Write-Host "  MCP_LISTENER_PORT:  " -NoNewline
        Write-Host "NOT SET" -ForegroundColor Red
    }

    Write-Host ""
}

# =============================================================================
# Main Script
# =============================================================================

Show-Banner

# Get script directory and project root
$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$projectRoot = Split-Path -Parent $scriptDir

Write-Host "  Project Root: $projectRoot" -ForegroundColor Gray
Write-Host ""

# --- Environment Setup ---
if (-not $SkipEnvSetup) {
    Write-Host "  [1/3] Setting up environment..." -ForegroundColor Yellow

    # Run the logon script
    $logonScript = Join-Path $scriptDir "AECAgent-Logon.ps1"
    if (Test-Path $logonScript) {
        & $logonScript
        if ($LASTEXITCODE -ne 0) {
            Write-Host "  ERROR: Logon script failed!" -ForegroundColor Red
            exit 1
        }

        # Refresh environment variables in current session
        $env:SESSION_TOKEN = [Environment]::GetEnvironmentVariable("SESSION_TOKEN", "User")
        $env:MCP_LISTENER_PORT = [Environment]::GetEnvironmentVariable("MCP_LISTENER_PORT", "User")
    }
    else {
        Write-Host "  ERROR: Logon script not found: $logonScript" -ForegroundColor Red
        exit 1
    }
}
else {
    Write-Host "  [1/3] Skipping environment setup (using existing values)" -ForegroundColor Gray
}

Write-Host ""
Show-Status

# --- Launch AutoCAD ---
if ($LaunchAutoCAD) {
    Write-Host "  [2/3] Launching AutoCAD..." -ForegroundColor Yellow

    if (-not $AutoCADPath) {
        $AutoCADPath = Find-AutoCAD
    }

    if ($AutoCADPath -and (Test-Path $AutoCADPath)) {
        Write-Host "  Starting: $AutoCADPath" -ForegroundColor Gray

        # Start AutoCAD (it will inherit current environment variables)
        Start-Process -FilePath $AutoCADPath

        Write-Host "  AutoCAD launched. Wait for it to fully load before proceeding." -ForegroundColor Green
        Write-Host "  Check AutoCAD command line for: [AEC Agent] Sidecar started on port $env:MCP_LISTENER_PORT" -ForegroundColor Cyan
        Write-Host ""

        # Give AutoCAD time to start
        Write-Host "  Waiting 10 seconds for AutoCAD to initialize..." -ForegroundColor Gray
        Start-Sleep -Seconds 10
    }
    else {
        Write-Host "  WARNING: AutoCAD not found. Please launch it manually." -ForegroundColor Yellow
        Write-Host "  Searched paths:" -ForegroundColor Gray
        Write-Host "    - C:\Program Files\Autodesk\AutoCAD 2024\acad.exe" -ForegroundColor Gray
        Write-Host "    - C:\Program Files\Autodesk\AutoCAD 2025\acad.exe" -ForegroundColor Gray
        Write-Host ""
    }
}
else {
    Write-Host "  [2/3] Skipping AutoCAD launch (use -LaunchAutoCAD to enable)" -ForegroundColor Gray
}

Write-Host ""

# --- Launch AEC Agent ---
if ($LaunchAgent) {
    Write-Host "  [3/3] Launching AEC Agent..." -ForegroundColor Yellow

    # Change to project directory
    Push-Location $projectRoot

    try {
        # Check for virtual environment
        $venvActivate = Join-Path $projectRoot "venv\Scripts\Activate.ps1"
        if (Test-Path $venvActivate) {
            Write-Host "  Activating virtual environment..." -ForegroundColor Gray
            & $venvActivate
        }

        Write-Host "  Starting AEC Agent (MCP Server + Chainlit)..." -ForegroundColor Gray
        Write-Host ""
        Write-Host "  ================================================" -ForegroundColor Cyan
        Write-Host "  Press Ctrl+C to stop the agent" -ForegroundColor Cyan
        Write-Host "  ================================================" -ForegroundColor Cyan
        Write-Host ""

        # Run the agent
        python -m aec_agent.cli launch
    }
    finally {
        Pop-Location
    }
}
else {
    Write-Host "  [3/3] Skipping AEC Agent launch (use -LaunchAgent to enable)" -ForegroundColor Gray
    Write-Host ""
    Write-Host "  To launch manually, run:" -ForegroundColor Yellow
    Write-Host "    cd $projectRoot" -ForegroundColor White
    Write-Host "    .\venv\Scripts\Activate.ps1" -ForegroundColor White
    Write-Host "    aec-agent launch" -ForegroundColor White
}

Write-Host ""
Write-Host "  ================================================" -ForegroundColor Cyan
Write-Host "  Done!" -ForegroundColor Cyan
Write-Host "  ================================================" -ForegroundColor Cyan
Write-Host ""
