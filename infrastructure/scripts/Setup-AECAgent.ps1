<#
.SYNOPSIS
    AEC Agent Setup Script for Windows Server

.DESCRIPTION
    This script configures a Windows Server for running the AEC Agent.
    It should be run as Administrator.

.PARAMETER InstallPython
    Install Python 3.11 if not present

.PARAMETER InstallAutodesk
    Configure Autodesk application settings (assumes apps are pre-installed)

.PARAMETER ConfigureGPO
    Create GPO logon script configuration

.EXAMPLE
    .\Setup-AECAgent.ps1 -InstallPython -ConfigureGPO
#>

[CmdletBinding()]
param(
    [switch]$InstallPython,
    [switch]$InstallAutodesk,
    [switch]$ConfigureGPO,
    [switch]$All
)

$ErrorActionPreference = "Stop"

# Require Administrator
$CurrentUser = [Security.Principal.WindowsIdentity]::GetCurrent()
$Principal = New-Object Security.Principal.WindowsPrincipal($CurrentUser)
if (-not $Principal.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)) {
    Write-Error "This script must be run as Administrator"
    exit 1
}

$AECAgentRoot = "C:\AECAgent"
$ScriptsDir = "$AECAgentRoot\scripts"
$LogsDir = "$AECAgentRoot\logs"
$CacheDir = "$AECAgentRoot\cache"

function Write-Log {
    param([string]$Message, [string]$Level = "INFO")
    $Timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    $LogMessage = "[$Timestamp] [$Level] $Message"
    Write-Host $LogMessage -ForegroundColor $(
        switch ($Level) {
            "ERROR" { "Red" }
            "WARN"  { "Yellow" }
            "INFO"  { "Cyan" }
            default { "White" }
        }
    )
    Add-Content -Path "$LogsDir\setup.log" -Value $LogMessage
}

# =============================================================================
# Create Directory Structure
# =============================================================================
Write-Host "`n=== AEC Agent Setup ===" -ForegroundColor Green
Write-Host "Creating directory structure..."

New-Item -ItemType Directory -Force -Path $AECAgentRoot | Out-Null
New-Item -ItemType Directory -Force -Path $ScriptsDir | Out-Null
New-Item -ItemType Directory -Force -Path $LogsDir | Out-Null
New-Item -ItemType Directory -Force -Path $CacheDir | Out-Null

Write-Log "Directory structure created"

# =============================================================================
# Install Python
# =============================================================================
if ($InstallPython -or $All) {
    Write-Host "`nInstalling Python 3.11..." -ForegroundColor Yellow

    $PythonVersion = "3.11.7"
    $PythonUrl = "https://www.python.org/ftp/python/$PythonVersion/python-$PythonVersion-amd64.exe"
    $InstallerPath = "$AECAgentRoot\python-installer.exe"

    # Check if Python is already installed
    $ExistingPython = Get-Command python -ErrorAction SilentlyContinue
    if ($ExistingPython) {
        $Version = & python --version 2>&1
        Write-Log "Python already installed: $Version"
    } else {
        try {
            Write-Log "Downloading Python $PythonVersion..."
            Invoke-WebRequest -Uri $PythonUrl -OutFile $InstallerPath -UseBasicParsing

            Write-Log "Installing Python..."
            Start-Process -FilePath $InstallerPath -ArgumentList "/quiet InstallAllUsers=1 PrependPath=1 Include_pip=1" -Wait -NoNewWindow

            # Refresh PATH
            $env:Path = [System.Environment]::GetEnvironmentVariable("Path","Machine") + ";" +
                        [System.Environment]::GetEnvironmentVariable("Path","User")

            Write-Log "Python $PythonVersion installed successfully"
        } catch {
            Write-Log "Failed to install Python: $_" -Level "ERROR"
        }
    }

    # Install AEC Agent Python dependencies
    Write-Log "Installing Python dependencies..."
    try {
        & python -m pip install --upgrade pip
        & pip install httpx pydantic tenacity aiosqlite structlog python-dotenv
        Write-Log "Python dependencies installed"
    } catch {
        Write-Log "Failed to install dependencies: $_" -Level "WARN"
    }
}

# =============================================================================
# Configure Autodesk Settings
# =============================================================================
if ($InstallAutodesk -or $All) {
    Write-Host "`nConfiguring Autodesk settings..." -ForegroundColor Yellow

    # AutoCAD settings
    $AutoCADPath = "C:\Program Files\Autodesk\AutoCAD 2024"
    if (Test-Path $AutoCADPath) {
        Write-Log "AutoCAD 2024 detected at $AutoCADPath"

        # Configure AutoCAD to disable command echo (reduces COM timeouts)
        # This would typically be done via profile or registry
    } else {
        Write-Log "AutoCAD 2024 not found at expected path" -Level "WARN"
    }

    # Revit settings
    $RevitPath = "C:\Program Files\Autodesk\Revit 2024"
    if (Test-Path $RevitPath) {
        Write-Log "Revit 2024 detected at $RevitPath"

        # Configure Revit journal file location to NVMe if available
        $NVMeDrive = Get-Volume | Where-Object { $_.FileSystemLabel -eq "TempStorage" }
        if ($NVMeDrive) {
            $JournalPath = "$($NVMeDrive.DriveLetter):\RevitJournals"
            New-Item -ItemType Directory -Force -Path $JournalPath | Out-Null
            Write-Log "Revit journal directory created at $JournalPath"
        }
    } else {
        Write-Log "Revit 2024 not found at expected path" -Level "WARN"
    }

    # Check for pyRevit
    $PyRevitPath = "$env:APPDATA\pyRevit-Master"
    if (Test-Path $PyRevitPath) {
        Write-Log "pyRevit detected at $PyRevitPath"
    } else {
        Write-Log "pyRevit not found - install from https://pyrevitlabs.io" -Level "WARN"
    }
}

# =============================================================================
# Configure GPO Logon Script
# =============================================================================
if ($ConfigureGPO -or $All) {
    Write-Host "`nConfiguring GPO logon script..." -ForegroundColor Yellow

    # Create the logon script
    $LogonScriptContent = @'
<#
.SYNOPSIS
    AEC Agent User Logon Script

.DESCRIPTION
    This script runs when a user logs into an RDP session.
    It configures session-specific environment variables for the MCP sidecar.

    Deploy this script via GPO: User Configuration > Windows Settings > Scripts > Logon
#>

$ErrorActionPreference = "SilentlyContinue"

# Get current session ID
$SessionId = (Get-Process -Id $PID).SessionId

# Calculate unique port base (with bounds checking)
# Port range: 20000-30000 to avoid conflicts
$Port = 20000 + ($SessionId % 10000)
if ($Port -gt 30000 -or $Port -lt 20000) {
    $Port = 20000 + (Get-Random -Minimum 1 -Maximum 9999)
}

# Verify port is available
$MaxAttempts = 100
$Attempt = 0
while ($Attempt -lt $MaxAttempts) {
    $PortInUse = Get-NetTCPConnection -LocalPort $Port -ErrorAction SilentlyContinue
    if (-not $PortInUse) {
        break
    }
    $Port++
    if ($Port -gt 30000) { $Port = 20000 }
    $Attempt++
}

if ($Attempt -eq $MaxAttempts) {
    Write-EventLog -LogName Application -Source "AEC Agent" -EventId 1001 -EntryType Error -Message "Failed to find available port"
    exit 1
}

# Generate a secure Session Token for this session
$Token = [guid]::NewGuid().ToString()

# Set Environment Variable for the Session (Process scope for immediate availability)
[Environment]::SetEnvironmentVariable("MCP_LISTENER_PORT", $Port, "Process")
# Also persist to User scope for child processes
[Environment]::SetEnvironmentVariable("MCP_LISTENER_PORT", $Port, "User")

# Generate and set session token
[Environment]::SetEnvironmentVariable("SESSION_TOKEN", $Token, "Process")
[Environment]::SetEnvironmentVariable("SESSION_TOKEN", $Token, "User")

# Configure Chainlit port (unique per session)
$ChainlitPort = 8000 + ($SessionId % 100)
[Environment]::SetEnvironmentVariable("CHAINLIT_PORT", $ChainlitPort, "Process")
[Environment]::SetEnvironmentVariable("CHAINLIT_PORT", $ChainlitPort, "User")

# Log session information
$LogFile = "C:\AECAgent\logs\sessions.log"
$Timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
$Username = $env:USERNAME
$Computer = $env:COMPUTERNAME

$LogEntry = "[$Timestamp] Computer: $Computer, User: $Username, Session: $SessionId, Port: $Port, Chainlit: $ChainlitPort, Token: $($Token.Substring(0,8))..."
Add-Content -Path $LogFile -Value $LogEntry

# Create Windows Event Log entry
try {
    New-EventLog -LogName Application -Source "AEC Agent" -ErrorAction SilentlyContinue
    Write-EventLog -LogName Application -Source "AEC Agent" -EventId 1000 -EntryType Information -Message "Session configured: Port=$Port, User=$Username"
} catch {
    # Event log registration may require elevated privileges
}

# Output for debugging (visible in logon script logs)
Write-Host "AEC Agent Session Configured"
Write-Host "  Port: $Port"
Write-Host "  Chainlit: $ChainlitPort"
Write-Host "  Token: $($Token.Substring(0,8))..."
'@

    $LogonScriptPath = "$ScriptsDir\AECAgent-Logon.ps1"
    $LogonScriptContent | Out-File -FilePath $LogonScriptPath -Encoding UTF8

    Write-Log "GPO logon script created at $LogonScriptPath"

    # Create batch wrapper for GPO (GPO may require .bat)
    $BatchWrapper = @"
@echo off
powershell.exe -ExecutionPolicy Bypass -NoProfile -File "$LogonScriptPath"
"@
    $BatchWrapper | Out-File -FilePath "$ScriptsDir\AECAgent-Logon.bat" -Encoding ASCII

    Write-Host @"

=== GPO Configuration Instructions ===
1. Open Group Policy Management (gpmc.msc)
2. Create or edit a GPO linked to your RDP users OU
3. Navigate to: User Configuration > Windows Settings > Scripts > Logon
4. Add script: $ScriptsDir\AECAgent-Logon.bat
5. Ensure script runs synchronously (Properties > Run Logon Scripts Synchronously = Enabled)

"@ -ForegroundColor Cyan
}

# =============================================================================
# Configure Windows Firewall
# =============================================================================
Write-Host "`nConfiguring Windows Firewall..." -ForegroundColor Yellow

try {
    # Remove existing rules if any
    Remove-NetFirewallRule -DisplayName "AEC Agent*" -ErrorAction SilentlyContinue

    # Allow Chainlit UI (8000-8100)
    New-NetFirewallRule -DisplayName "AEC Agent - Chainlit UI" `
        -Direction Inbound -Protocol TCP -LocalPort 8000-8100 -Action Allow `
        -Description "Allow inbound connections to Chainlit web UI"

    # Allow MCP Server SSE (54000-55000)
    New-NetFirewallRule -DisplayName "AEC Agent - MCP Server" `
        -Direction Inbound -Protocol TCP -LocalPort 54000-55000 -Action Allow `
        -Description "Allow inbound connections to MCP SSE server"

    # Localhost sidecar communication (always allowed for 127.0.0.1)
    New-NetFirewallRule -DisplayName "AEC Agent - Localhost Sidecar" `
        -Direction Inbound -Protocol TCP -LocalPort 20000-30000 `
        -RemoteAddress 127.0.0.1 -Action Allow `
        -Description "Allow localhost sidecar communication"

    Write-Log "Firewall rules configured successfully"
} catch {
    Write-Log "Failed to configure firewall: $_" -Level "ERROR"
}

# =============================================================================
# Register HTTP URL ACLs
# =============================================================================
Write-Host "`nRegistering HTTP URL ACLs..." -ForegroundColor Yellow

try {
    # Register URL ACLs for HttpListener
    # This allows non-admin users to bind to these ports
    $UrlAcls = @(
        "http://127.0.0.1:20000/",
        "http://127.0.0.1:25000/",
        "http://127.0.0.1:30000/",
        "http://+:8000/",
        "http://+:54321/"
    )

    foreach ($Url in $UrlAcls) {
        $Result = netsh http add urlacl url=$Url user="Everyone" 2>&1
        if ($LASTEXITCODE -eq 0) {
            Write-Log "Registered URL ACL: $Url"
        } else {
            # May already exist
            Write-Log "URL ACL may already exist: $Url" -Level "WARN"
        }
    }
} catch {
    Write-Log "Failed to register URL ACLs: $_" -Level "WARN"
}

# =============================================================================
# Summary
# =============================================================================
Write-Host "`n=== Setup Complete ===" -ForegroundColor Green
Write-Host @"

AEC Agent has been configured on this system.

Directory Structure:
  $AECAgentRoot
  ├── cache/     - SQLite cache files
  ├── logs/      - Application logs
  └── scripts/   - Logon and utility scripts

Next Steps:
1. Install Autodesk applications (AutoCAD 2024, Revit 2024)
2. Install pyRevit from https://pyrevitlabs.io
3. Deploy GPO logon script (see instructions above)
4. Clone the AEC Agent repository and install dependencies
5. Configure .env file with API keys

For more information, see the implementation guides.

"@ -ForegroundColor Cyan

Write-Log "Setup completed successfully"
