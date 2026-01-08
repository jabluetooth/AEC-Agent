<#
.SYNOPSIS
    AEC Agent Compatibility Test Script

.DESCRIPTION
    Tests the current system for AEC Agent compatibility.
    Checks Python version, .NET Framework, CAD applications, and environment.

.EXAMPLE
    .\Test-Compatibility.ps1
#>

[CmdletBinding()]
param()

$ErrorActionPreference = "SilentlyContinue"

# Color helpers
function Write-Pass { param([string]$Text) Write-Host "[PASS] " -ForegroundColor Green -NoNewline; Write-Host $Text }
function Write-Warn { param([string]$Text) Write-Host "[WARN] " -ForegroundColor Yellow -NoNewline; Write-Host $Text }
function Write-Fail { param([string]$Text) Write-Host "[FAIL] " -ForegroundColor Red -NoNewline; Write-Host $Text }
function Write-Skip { param([string]$Text) Write-Host "[SKIP] " -ForegroundColor Gray -NoNewline; Write-Host $Text }

$Results = @{
    Pass = 0
    Warn = 0
    Fail = 0
    Skip = 0
}

Write-Host "`n" + ("=" * 60) -ForegroundColor Cyan
Write-Host "AEC Agent - System Compatibility Check" -ForegroundColor Cyan
Write-Host ("=" * 60) + "`n" -ForegroundColor Cyan

# =============================================================================
# Operating System
# =============================================================================
Write-Host "Operating System" -ForegroundColor White
Write-Host ("-" * 40)

$OS = Get-CimInstance Win32_OperatingSystem
$OSVersion = $OS.Caption

if ($OSVersion -like "*Server 2022*") {
    Write-Pass "Windows Server 2022 (Recommended)"
    $Results.Pass++
} elseif ($OSVersion -like "*Server 2019*") {
    Write-Pass "Windows Server 2019 (Supported)"
    $Results.Pass++
} elseif ($OSVersion -like "*Server*") {
    Write-Warn "$OSVersion (Check compatibility)"
    $Results.Warn++
} elseif ($OSVersion -like "*Windows 10*" -or $OSVersion -like "*Windows 11*") {
    Write-Warn "$OSVersion (Development only - use Server for production)"
    $Results.Warn++
} else {
    Write-Fail "$OSVersion (Windows Server required)"
    $Results.Fail++
}

Write-Host ""

# =============================================================================
# .NET Framework
# =============================================================================
Write-Host ".NET Framework" -ForegroundColor White
Write-Host ("-" * 40)

try {
    $NetKey = Get-ItemProperty "HKLM:\SOFTWARE\Microsoft\NET Framework Setup\NDP\v4\Full" -ErrorAction Stop
    $Release = $NetKey.Release

    if ($Release -ge 528040) {
        Write-Pass ".NET Framework 4.8 or later (Release: $Release)"
        $Results.Pass++
    } elseif ($Release -ge 461808) {
        Write-Warn ".NET Framework 4.7.2 (Release: $Release) - 4.8 recommended"
        $Results.Warn++
    } else {
        Write-Fail ".NET Framework version too old (Release: $Release)"
        $Results.Fail++
    }
} catch {
    Write-Fail ".NET Framework 4.x not detected"
    $Results.Fail++
}

# Check for .NET 8 (for Revit 2025)
$DotNet8 = & dotnet --list-runtimes 2>&1 | Select-String "Microsoft.NETCore.App 8"
if ($DotNet8) {
    Write-Pass ".NET 8 Runtime installed (Required for Revit 2025)"
    $Results.Pass++
} else {
    Write-Warn ".NET 8 Runtime not found (Only needed for Revit 2025)"
    $Results.Warn++
}

Write-Host ""

# =============================================================================
# Python
# =============================================================================
Write-Host "Python Environment" -ForegroundColor White
Write-Host ("-" * 40)

$Python = Get-Command python -ErrorAction SilentlyContinue
if ($Python) {
    $PythonVersion = & python --version 2>&1
    $VersionMatch = $PythonVersion -match "Python (\d+)\.(\d+)"

    if ($Matches) {
        $Major = [int]$Matches[1]
        $Minor = [int]$Matches[2]

        if ($Major -eq 3 -and $Minor -ge 9 -and $Minor -le 12) {
            if ($Minor -eq 11) {
                Write-Pass "$PythonVersion (Recommended)"
            } else {
                Write-Pass "$PythonVersion (Supported)"
            }
            $Results.Pass++
        } elseif ($Major -eq 3 -and $Minor -lt 9) {
            Write-Fail "$PythonVersion (3.9+ required)"
            $Results.Fail++
        } else {
            Write-Warn "$PythonVersion (Untested version)"
            $Results.Warn++
        }
    }
} else {
    Write-Fail "Python not found in PATH"
    $Results.Fail++
}

# Check pip packages
$RequiredPackages = @("httpx", "pydantic", "tenacity", "aiosqlite")
foreach ($Package in $RequiredPackages) {
    $Installed = & pip show $Package 2>&1
    if ($LASTEXITCODE -eq 0) {
        $Version = ($Installed | Select-String "Version:").ToString().Split(":")[1].Trim()
        Write-Pass "  $Package $Version"
        $Results.Pass++
    } else {
        Write-Warn "  $Package not installed"
        $Results.Warn++
    }
}

Write-Host ""

# =============================================================================
# AutoCAD
# =============================================================================
Write-Host "AutoCAD" -ForegroundColor White
Write-Host ("-" * 40)

$AutoCADVersions = @("2025", "2024", "2023", "2022", "2021")
$AutoCADFound = $false

foreach ($Version in $AutoCADVersions) {
    $Path = "C:\Program Files\Autodesk\AutoCAD $Version"
    if (Test-Path $Path) {
        if ($Version -eq "2024") {
            Write-Pass "AutoCAD $Version detected (Recommended)"
        } elseif ($Version -eq "2025") {
            Write-Warn "AutoCAD $Version detected (Testing - verify API compatibility)"
        } else {
            Write-Pass "AutoCAD $Version detected (Supported)"
        }
        $Results.Pass++
        $AutoCADFound = $true
        break
    }
}

if (-not $AutoCADFound) {
    Write-Warn "AutoCAD not detected in standard paths"
    $Results.Warn++
}

Write-Host ""

# =============================================================================
# Revit
# =============================================================================
Write-Host "Revit" -ForegroundColor White
Write-Host ("-" * 40)

$RevitVersions = @("2025", "2024", "2023", "2022", "2021")
$RevitFound = $false

foreach ($Version in $RevitVersions) {
    $Path = "C:\Program Files\Autodesk\Revit $Version"
    if (Test-Path $Path) {
        if ($Version -eq "2024") {
            Write-Pass "Revit $Version detected (Recommended)"
        } elseif ($Version -eq "2025") {
            Write-Warn "Revit $Version detected (Requires .NET 8 and pyRevit 5.0+)"
        } elseif ($Version -eq "2021") {
            Write-Warn "Revit $Version detected (End of support - upgrade recommended)"
        } else {
            Write-Pass "Revit $Version detected (Supported)"
        }
        $Results.Pass++
        $RevitFound = $true
        break
    }
}

if (-not $RevitFound) {
    Write-Warn "Revit not detected in standard paths"
    $Results.Warn++
}

Write-Host ""

# =============================================================================
# pyRevit
# =============================================================================
Write-Host "pyRevit" -ForegroundColor White
Write-Host ("-" * 40)

$PyRevitPaths = @(
    "$env:APPDATA\pyRevit-Master",
    "$env:PROGRAMDATA\pyRevit",
    "C:\pyRevit"
)

$PyRevitFound = $false
foreach ($Path in $PyRevitPaths) {
    if (Test-Path $Path) {
        Write-Pass "pyRevit detected at $Path"
        $Results.Pass++
        $PyRevitFound = $true
        break
    }
}

if (-not $PyRevitFound) {
    Write-Warn "pyRevit not detected - install from https://pyrevitlabs.io"
    $Results.Warn++
}

Write-Host ""

# =============================================================================
# Environment Variables
# =============================================================================
Write-Host "Environment Variables" -ForegroundColor White
Write-Host ("-" * 40)

$MCPPort = $env:MCP_LISTENER_PORT
if ($MCPPort) {
    $PortInt = [int]$MCPPort
    if ($PortInt -ge 20000 -and $PortInt -le 30000) {
        Write-Pass "MCP_LISTENER_PORT = $MCPPort (Valid range)"
        $Results.Pass++
    } else {
        Write-Warn "MCP_LISTENER_PORT = $MCPPort (Outside recommended 20000-30000)"
        $Results.Warn++
    }
} else {
    Write-Skip "MCP_LISTENER_PORT not set (Will be set by GPO logon script)"
    $Results.Skip++
}

$SessionToken = $env:SESSION_TOKEN
if ($SessionToken) {
    Write-Pass "SESSION_TOKEN = $($SessionToken.Substring(0,8))... (Set)"
    $Results.Pass++
} else {
    Write-Skip "SESSION_TOKEN not set (Will be set by GPO logon script)"
    $Results.Skip++
}

Write-Host ""

# =============================================================================
# Network Ports
# =============================================================================
Write-Host "Network Ports" -ForegroundColor White
Write-Host ("-" * 40)

$PortsToCheck = @(
    @{ Port = 8000; Name = "Chainlit UI" },
    @{ Port = 54321; Name = "MCP Server" }
)

foreach ($PortInfo in $PortsToCheck) {
    $InUse = Get-NetTCPConnection -LocalPort $PortInfo.Port -ErrorAction SilentlyContinue
    if ($InUse) {
        Write-Warn "Port $($PortInfo.Port) ($($PortInfo.Name)) is in use"
        $Results.Warn++
    } else {
        Write-Pass "Port $($PortInfo.Port) ($($PortInfo.Name)) is available"
        $Results.Pass++
    }
}

Write-Host ""

# =============================================================================
# GPU (NVIDIA)
# =============================================================================
Write-Host "GPU" -ForegroundColor White
Write-Host ("-" * 40)

$GPU = Get-CimInstance Win32_VideoController | Where-Object { $_.Name -like "*NVIDIA*" }
if ($GPU) {
    Write-Pass "$($GPU.Name)"
    $Results.Pass++

    # Check driver version
    $NvidiaSmi = & nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>&1
    if ($LASTEXITCODE -eq 0) {
        Write-Pass "  Driver version: $NvidiaSmi"
    }
} else {
    Write-Warn "NVIDIA GPU not detected (Required for CAD acceleration)"
    $Results.Warn++
}

Write-Host ""

# =============================================================================
# Summary
# =============================================================================
Write-Host ("=" * 60) -ForegroundColor Cyan
Write-Host "Summary" -ForegroundColor Cyan
Write-Host ("=" * 60) -ForegroundColor Cyan

Write-Host "  Passed:   $($Results.Pass)" -ForegroundColor Green
Write-Host "  Warnings: $($Results.Warn)" -ForegroundColor Yellow
Write-Host "  Failed:   $($Results.Fail)" -ForegroundColor Red
Write-Host "  Skipped:  $($Results.Skip)" -ForegroundColor Gray

Write-Host ""

if ($Results.Fail -gt 0) {
    Write-Host "Some checks failed. Please address the issues above." -ForegroundColor Red
    exit 1
} elseif ($Results.Warn -gt 0) {
    Write-Host "Some checks have warnings. Review before production use." -ForegroundColor Yellow
    exit 0
} else {
    Write-Host "All checks passed!" -ForegroundColor Green
    exit 0
}
