<powershell>
# AEC Agent - EC2 Instance Bootstrap Script
# This script runs on first boot to configure the Windows Server instance

param(
    [string]$Environment = "${environment}",
    [string]$CloudWatchGroup = "${cloudwatch_group}"
)

$ErrorActionPreference = "Stop"
$LogFile = "C:\AECAgent\logs\bootstrap.log"

function Write-Log {
    param([string]$Message)
    $Timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    $LogMessage = "[$Timestamp] $Message"
    Add-Content -Path $LogFile -Value $LogMessage
    Write-Host $LogMessage
}

# Create directories
New-Item -ItemType Directory -Force -Path "C:\AECAgent\logs" | Out-Null
New-Item -ItemType Directory -Force -Path "C:\AECAgent\cache" | Out-Null
New-Item -ItemType Directory -Force -Path "C:\AECAgent\scripts" | Out-Null

Write-Log "Starting AEC Agent bootstrap..."
Write-Log "Environment: $Environment"

# =============================================================================
# Install NVIDIA GRID Drivers
# =============================================================================
Write-Log "Installing NVIDIA GRID drivers..."

try {
    $NvidiaUrl = "https://ec2-windows-nvidia-drivers.s3.amazonaws.com/latest/nvidia-driver-installer.exe"
    $InstallerPath = "C:\AECAgent\nvidia-installer.exe"

    Invoke-WebRequest -Uri $NvidiaUrl -OutFile $InstallerPath -UseBasicParsing
    Start-Process -FilePath $InstallerPath -ArgumentList "-s -n" -Wait -NoNewWindow

    Write-Log "NVIDIA drivers installed successfully"
} catch {
    Write-Log "Warning: NVIDIA driver installation failed: $_"
}

# =============================================================================
# Install CloudWatch Agent
# =============================================================================
Write-Log "Installing CloudWatch Agent..."

try {
    $CWAgentUrl = "https://s3.amazonaws.com/amazoncloudwatch-agent/windows/amd64/latest/amazon-cloudwatch-agent.msi"
    $CWInstallerPath = "C:\AECAgent\amazon-cloudwatch-agent.msi"

    Invoke-WebRequest -Uri $CWAgentUrl -OutFile $CWInstallerPath -UseBasicParsing
    Start-Process msiexec.exe -ArgumentList "/i $CWInstallerPath /quiet" -Wait -NoNewWindow

    # Configure CloudWatch Agent
    $CWConfig = @"
{
    "logs": {
        "logs_collected": {
            "files": {
                "collect_list": [
                    {
                        "file_path": "C:\\AECAgent\\logs\\*.log",
                        "log_group_name": "$CloudWatchGroup",
                        "log_stream_name": "{instance_id}/aec-agent"
                    }
                ]
            },
            "windows_events": {
                "collect_list": [
                    {
                        "event_name": "Application",
                        "event_levels": ["ERROR", "WARNING"],
                        "log_group_name": "$CloudWatchGroup",
                        "log_stream_name": "{instance_id}/windows-events"
                    }
                ]
            }
        }
    },
    "metrics": {
        "namespace": "AECAgent",
        "metrics_collected": {
            "Memory": {
                "measurement": ["% Committed Bytes In Use"],
                "metrics_collection_interval": 60
            },
            "LogicalDisk": {
                "measurement": ["% Free Space"],
                "metrics_collection_interval": 60,
                "resources": ["*"]
            }
        }
    }
}
"@

    $CWConfig | Out-File -FilePath "C:\ProgramData\Amazon\AmazonCloudWatchAgent\amazon-cloudwatch-agent.json" -Encoding UTF8
    & "C:\Program Files\Amazon\AmazonCloudWatchAgent\amazon-cloudwatch-agent-ctl.ps1" -a fetch-config -m ec2 -s -c file:"C:\ProgramData\Amazon\AmazonCloudWatchAgent\amazon-cloudwatch-agent.json"

    Write-Log "CloudWatch Agent installed and configured"
} catch {
    Write-Log "Warning: CloudWatch Agent installation failed: $_"
}

# =============================================================================
# Configure NVMe Instance Store for Temp Files
# =============================================================================
Write-Log "Configuring NVMe instance store..."

try {
    # Get the NVMe instance store disk
    $NVMeDisk = Get-Disk | Where-Object { $_.FriendlyName -like "*NVMe*" -and $_.PartitionStyle -eq 'RAW' }

    if ($NVMeDisk) {
        # Initialize and format the disk
        $NVMeDisk | Initialize-Disk -PartitionStyle GPT
        $Partition = $NVMeDisk | New-Partition -AssignDriveLetter -UseMaximumSize
        $Partition | Format-Volume -FileSystem NTFS -NewFileSystemLabel "TempStorage" -Confirm:$false

        $DriveLetter = $Partition.DriveLetter
        Write-Log "NVMe disk initialized at $DriveLetter`:"

        # Create temp directories on NVMe
        New-Item -ItemType Directory -Force -Path "$DriveLetter`:\Temp" | Out-Null
        New-Item -ItemType Directory -Force -Path "$DriveLetter`:\AutodeskCache" | Out-Null

        # Set system TEMP to NVMe drive
        [Environment]::SetEnvironmentVariable("TEMP", "$DriveLetter`:\Temp", "Machine")
        [Environment]::SetEnvironmentVariable("TMP", "$DriveLetter`:\Temp", "Machine")

        Write-Log "System TEMP configured to NVMe storage"
    } else {
        Write-Log "No NVMe instance store found - using default temp location"
    }
} catch {
    Write-Log "Warning: NVMe configuration failed: $_"
}

# =============================================================================
# Install Python
# =============================================================================
Write-Log "Installing Python 3.11..."

try {
    $PythonUrl = "https://www.python.org/ftp/python/3.11.7/python-3.11.7-amd64.exe"
    $PythonInstaller = "C:\AECAgent\python-installer.exe"

    Invoke-WebRequest -Uri $PythonUrl -OutFile $PythonInstaller -UseBasicParsing
    Start-Process -FilePath $PythonInstaller -ArgumentList "/quiet InstallAllUsers=1 PrependPath=1" -Wait -NoNewWindow

    # Refresh PATH
    $env:Path = [System.Environment]::GetEnvironmentVariable("Path","Machine") + ";" + [System.Environment]::GetEnvironmentVariable("Path","User")

    Write-Log "Python 3.11 installed successfully"
} catch {
    Write-Log "Warning: Python installation failed: $_"
}

# =============================================================================
# Create Logon Script
# =============================================================================
Write-Log "Creating user logon script..."

$LogonScript = @'
# AEC Agent - User Logon Script
# This script runs when a user logs into an RDP session

$ErrorActionPreference = "SilentlyContinue"

# Get current session ID
$SessionId = (Get-Process -Id $PID).SessionId

# Calculate unique port (20000-30000 range)
$Port = 20000 + ($SessionId % 10000)
if ($Port -gt 30000 -or $Port -lt 20000) {
    $Port = 20000 + (Get-Random -Minimum 1 -Maximum 9999)
}

# Verify port is available
$PortInUse = Get-NetTCPConnection -LocalPort $Port -ErrorAction SilentlyContinue
if ($PortInUse) {
    for ($i = $Port; $i -lt 30000; $i++) {
        $Check = Get-NetTCPConnection -LocalPort $i -ErrorAction SilentlyContinue
        if (-not $Check) {
            $Port = $i
            break
        }
    }
}

# Generate session token
$Token = [guid]::NewGuid().ToString()

# Set environment variables (Process and User scope)
[Environment]::SetEnvironmentVariable("MCP_LISTENER_PORT", $Port, "Process")
[Environment]::SetEnvironmentVariable("MCP_LISTENER_PORT", $Port, "User")
[Environment]::SetEnvironmentVariable("SESSION_TOKEN", $Token, "Process")
[Environment]::SetEnvironmentVariable("SESSION_TOKEN", $Token, "User")

# Log session info
$LogFile = "C:\AECAgent\logs\sessions.log"
$Timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
$Username = $env:USERNAME
Add-Content -Path $LogFile -Value "[$Timestamp] User: $Username, Session: $SessionId, Port: $Port, Token: $($Token.Substring(0,8))..."

Write-Host "AEC Agent configured - Port: $Port"
'@

$LogonScript | Out-File -FilePath "C:\AECAgent\scripts\UserLogon.ps1" -Encoding UTF8

Write-Log "Logon script created"

# =============================================================================
# Configure Windows Firewall
# =============================================================================
Write-Log "Configuring Windows Firewall..."

try {
    # Allow Chainlit UI
    New-NetFirewallRule -DisplayName "AEC Agent - Chainlit UI" `
        -Direction Inbound -Protocol TCP -LocalPort 8000-8100 -Action Allow

    # Block external access to sidecar ports (extra security layer)
    New-NetFirewallRule -DisplayName "AEC Agent - Block External Sidecar" `
        -Direction Inbound -Protocol TCP -LocalPort 20000-30000 `
        -RemoteAddress "0.0.0.0/0" -Action Block

    # Allow localhost sidecar communication
    New-NetFirewallRule -DisplayName "AEC Agent - Localhost Sidecar" `
        -Direction Inbound -Protocol TCP -LocalPort 20000-30000 `
        -RemoteAddress "127.0.0.1" -Action Allow

    Write-Log "Firewall rules configured"
} catch {
    Write-Log "Warning: Firewall configuration failed: $_"
}

# =============================================================================
# Register HTTP URL ACL for HttpListener
# =============================================================================
Write-Log "Registering HTTP URL ACLs..."

try {
    # Register URL ACL for all users in sidecar port range
    netsh http add urlacl url="http://127.0.0.1:20000/" user="Everyone" | Out-Null
    netsh http add urlacl url="http://127.0.0.1:25000/" user="Everyone" | Out-Null
    netsh http add urlacl url="http://127.0.0.1:30000/" user="Everyone" | Out-Null

    Write-Log "HTTP URL ACLs registered"
} catch {
    Write-Log "Warning: URL ACL registration failed: $_"
}

# =============================================================================
# Schedule restart to apply changes
# =============================================================================
Write-Log "Bootstrap complete. Scheduling restart in 1 minute..."

shutdown /r /t 60 /c "AEC Agent bootstrap complete - restarting to apply changes"

Write-Log "Bootstrap script finished successfully"
</powershell>
