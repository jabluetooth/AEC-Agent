<#
.SYNOPSIS
    AEC Agent GPO Logon Script

.DESCRIPTION
    This script is designed to run at user logon via Group Policy.
    It assigns a unique port and session token for the AEC Agent sidecar,
    ensuring each user session has isolated credentials.

.NOTES
    Deployment: Computer Configuration > Policies > Windows Settings > Scripts > Logon
    Or: User Configuration > Policies > Windows Settings > Scripts > Logon

.EXAMPLE
    # Test locally (run as administrator for system-wide env vars)
    .\AECAgent-Logon.ps1

    # Test with verbose output
    .\AECAgent-Logon.ps1 -Verbose
#>

[CmdletBinding()]
param(
    # Minimum port in the allocation range
    [int]$MinPort = 20000,

    # Maximum port in the allocation range
    [int]$MaxPort = 30000,

    # Log file path (optional)
    [string]$LogFile = "$env:LOCALAPPDATA\AECAgent\logs\logon.log"
)

# =============================================================================
# Logging Functions
# =============================================================================

function Write-Log {
    param([string]$Message, [string]$Level = "INFO")

    $timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    $logEntry = "[$timestamp] [$Level] $Message"

    # Console output
    switch ($Level) {
        "ERROR" { Write-Host $logEntry -ForegroundColor Red }
        "WARN"  { Write-Host $logEntry -ForegroundColor Yellow }
        "INFO"  { Write-Host $logEntry -ForegroundColor Green }
        default { Write-Host $logEntry }
    }

    # File output (if log directory exists or can be created)
    try {
        $logDir = Split-Path $LogFile -Parent
        if (-not (Test-Path $logDir)) {
            New-Item -ItemType Directory -Path $logDir -Force | Out-Null
        }
        Add-Content -Path $LogFile -Value $logEntry -ErrorAction SilentlyContinue
    }
    catch {
        # Silently ignore logging failures
    }
}

# =============================================================================
# Port Allocation Functions
# =============================================================================

function Test-PortAvailable {
    param([int]$Port)

    try {
        $listener = New-Object System.Net.Sockets.TcpListener([System.Net.IPAddress]::Loopback, $Port)
        $listener.Start()
        $listener.Stop()
        return $true
    }
    catch {
        return $false
    }
}

function Get-AvailablePort {
    param(
        [int]$MinPort,
        [int]$MaxPort
    )

    # Try random ports first (better distribution in multi-user environments)
    $random = New-Object System.Random
    $attempts = 0
    $maxAttempts = 50

    while ($attempts -lt $maxAttempts) {
        $port = $random.Next($MinPort, $MaxPort + 1)
        if (Test-PortAvailable -Port $port) {
            return $port
        }
        $attempts++
    }

    # Fallback: sequential scan
    for ($port = $MinPort; $port -le $MaxPort; $port++) {
        if (Test-PortAvailable -Port $port) {
            return $port
        }
    }

    throw "No available port found in range $MinPort-$MaxPort"
}

# =============================================================================
# Environment Variable Functions
# =============================================================================

function Set-UserEnvironmentVariable {
    param(
        [string]$Name,
        [string]$Value
    )

    # Set for current process
    [Environment]::SetEnvironmentVariable($Name, $Value, [EnvironmentVariableTarget]::Process)

    # Set for current user (persists across processes in this session)
    [Environment]::SetEnvironmentVariable($Name, $Value, [EnvironmentVariableTarget]::User)

    Write-Log "Set $Name = $Value"
}

function Get-ExistingSessionToken {
    # Check if there's already a valid session token set
    $existing = [Environment]::GetEnvironmentVariable("SESSION_TOKEN", [EnvironmentVariableTarget]::User)

    if ($existing -and $existing -match '^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$') {
        return $existing
    }

    return $null
}

function Get-ExistingPort {
    # Check if there's already a valid port set
    $existing = [Environment]::GetEnvironmentVariable("MCP_LISTENER_PORT", [EnvironmentVariableTarget]::User)

    if ($existing -and $existing -match '^\d+$') {
        $port = [int]$existing
        if ($port -ge $MinPort -and $port -le $MaxPort) {
            # Verify the port is still available or already in use by our sidecar
            if (Test-PortAvailable -Port $port) {
                return $port
            }
            else {
                Write-Log "Existing port $port is in use, will allocate new one" -Level "WARN"
            }
        }
    }

    return $null
}

# =============================================================================
# Main Script
# =============================================================================

function Main {
    Write-Log "=========================================="
    Write-Log "AEC Agent Logon Script Starting"
    Write-Log "User: $env:USERNAME"
    Write-Log "Computer: $env:COMPUTERNAME"
    Write-Log "Session: $env:SESSIONNAME"
    Write-Log "=========================================="

    try {
        # --- Session Token ---
        $sessionToken = Get-ExistingSessionToken
        if ($sessionToken) {
            Write-Log "Using existing SESSION_TOKEN"
        }
        else {
            $sessionToken = [guid]::NewGuid().ToString()
            Write-Log "Generated new SESSION_TOKEN"
        }
        Set-UserEnvironmentVariable -Name "SESSION_TOKEN" -Value $sessionToken

        # --- Listener Port ---
        $listenerPort = Get-ExistingPort
        if ($listenerPort) {
            Write-Log "Using existing MCP_LISTENER_PORT: $listenerPort"
        }
        else {
            $listenerPort = Get-AvailablePort -MinPort $MinPort -MaxPort $MaxPort
            Write-Log "Allocated new MCP_LISTENER_PORT: $listenerPort"
        }
        Set-UserEnvironmentVariable -Name "MCP_LISTENER_PORT" -Value $listenerPort

        # --- Create AECAgent directories ---
        $cacheDir = "$env:LOCALAPPDATA\AECAgent\cache"
        $logsDir = "$env:LOCALAPPDATA\AECAgent\logs"

        if (-not (Test-Path $cacheDir)) {
            New-Item -ItemType Directory -Path $cacheDir -Force | Out-Null
            Write-Log "Created cache directory: $cacheDir"
        }

        if (-not (Test-Path $logsDir)) {
            New-Item -ItemType Directory -Path $logsDir -Force | Out-Null
            Write-Log "Created logs directory: $logsDir"
        }

        # --- Summary ---
        Write-Log "=========================================="
        Write-Log "AEC Agent Environment Configured"
        Write-Log "  SESSION_TOKEN: $sessionToken"
        Write-Log "  MCP_LISTENER_PORT: $listenerPort"
        Write-Log "=========================================="
        Write-Log "Logon script completed successfully"

        return 0
    }
    catch {
        Write-Log "ERROR: $($_.Exception.Message)" -Level "ERROR"
        Write-Log $_.ScriptStackTrace -Level "ERROR"
        return 1
    }
}

# Run main function
exit (Main)
