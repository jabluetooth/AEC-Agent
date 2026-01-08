# Phase 1: Infrastructure & High-Density Environment Setup

**Objective:** Prepare the AWS environment for 50 concurrent users and resolve port/I/O conflicts.

## 1. Provision Compute
*   **Instance Type:** Deploy **13 Amazon EC2 G4dn.4xlarge instances**.
    *   *Configuration:* 16 vCPU, 64 GB RAM, NVIDIA T4 GPU.
    *   *Density:* Host 4 users per instance.
    *   *Rationale:* This ratio balances the single-core CPU clock speed requirements of Revit with the vGPU capabilities of the T4, ensuring smooth 3D rotation.

## 2. I/O Optimization
*   **Profile Management:** Use **FSLogix** for roaming profiles.
*   **Latency Mitigation:**
    *   Edit `Redirections.xml`.
    *   **Exclude:** `%TEMP%` and `C:\Users\%USERNAME%\AppData\Local\Autodesk\Revit\PacCache`.
    *   **Redirect:** Point these folders to the local **NVMe instance store** (ephemeral disk).
    *   *Why?* Revit automation generates thousands of small temporary files. Writing these to a network share (FSx) causes API timeouts.

## 3. Dynamic Port Orchestration
*   **Problem:** In a multi-session environment, you cannot hardcode port `8080`.
*   **Solution:** Assign unique ports via a PowerShell Logon Script.
*   **Script Logic (Deploy via GPO):**

    ```powershell
    # Get current session ID
    $session = (Get-Process -Id $pid).SessionId

    # Calculate unique port base (with bounds checking)
    # Port range: 20000-30000 to avoid conflicts
    $port = 20000 + ($session % 10000)
    if ($port -gt 30000 -or $port -lt 20000) {
        $port = 20000 + (Get-Random -Minimum 1 -Maximum 9999)
    }

    # Verify port is available
    $portInUse = Get-NetTCPConnection -LocalPort $port -ErrorAction SilentlyContinue
    if ($portInUse) {
        # Find next available port
        for ($i = $port; $i -lt 30000; $i++) {
            $check = Get-NetTCPConnection -LocalPort $i -ErrorAction SilentlyContinue
            if (-not $check) { $port = $i; break }
        }
    }

    # Set Environment Variable for the Session (Process scope for immediate availability)
    [Environment]::SetEnvironmentVariable("MCP_LISTENER_PORT", $port, "Process")
    # Also persist to User scope for child processes
    [Environment]::SetEnvironmentVariable("MCP_LISTENER_PORT", $port, "User")

    # Generate a secure Session Token for this session
    $token = [guid]::NewGuid().ToString()
    [Environment]::SetEnvironmentVariable("SESSION_TOKEN", $token, "Process")
    [Environment]::SetEnvironmentVariable("SESSION_TOKEN", $token, "User")

    # Log for debugging
    Write-Host "MCP Port: $port, Token: $($token.Substring(0,8))..."
    ```

## 4. Environment Variables
Ensure the following variables are accessible to both Python and .NET processes:
*   `MCP_LISTENER_PORT`: The port the Sidecar plugins will listen on.
*   `SESSION_TOKEN`: The UUID used to authenticate API calls (Security Isolation).
