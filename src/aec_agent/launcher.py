"""
AEC Agent Launcher.

Orchestrates the MCP server and Chainlit frontend as a unified system.
Handles process management, port allocation, and graceful shutdown.
"""

import os
import sys
import time
import uuid
import socket
import signal
import subprocess
import atexit
from pathlib import Path
from contextlib import closing
from typing import Optional

import structlog

from aec_agent.config.settings import get_settings
from aec_agent.utils.logging import setup_logging

logger = structlog.get_logger(__name__)


def find_free_port(start_port: int, max_attempts: int = 100) -> int:
    """
    Find a free port starting from start_port.

    Args:
        start_port: The port to start searching from.
        max_attempts: Maximum number of ports to try.

    Returns:
        An available port number.

    Raises:
        RuntimeError: If no free port is found within max_attempts.
    """
    for offset in range(max_attempts):
        port = start_port + offset
        with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as sock:
            try:
                sock.bind(("127.0.0.1", port))
                return port
            except OSError:
                continue
    raise RuntimeError(f"No free port found in range {start_port}-{start_port + max_attempts}")


def wait_for_port(port: int, timeout: float = 30.0, interval: float = 0.5) -> bool:
    """
    Wait for a port to become available (server started).

    Args:
        port: The port to check.
        timeout: Maximum time to wait in seconds.
        interval: Time between checks in seconds.

    Returns:
        True if the port is accepting connections, False if timeout.
    """
    start_time = time.time()
    while time.time() - start_time < timeout:
        with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as sock:
            sock.settimeout(1.0)
            try:
                sock.connect(("127.0.0.1", port))
                return True
            except (ConnectionRefusedError, OSError):
                time.sleep(interval)
    return False


class ProcessManager:
    """
    Manages subprocess lifecycle for MCP server and Chainlit.

    Ensures graceful shutdown of all processes on exit.
    """

    def __init__(self):
        self._processes: list[subprocess.Popen] = []
        self._shutting_down = False

        # Register cleanup handlers
        atexit.register(self.shutdown)
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        # SIGBREAK is Windows-specific for Ctrl+Break
        if sys.platform == "win32":
            signal.signal(signal.SIGBREAK, self._signal_handler)

    def _signal_handler(self, signum: int, frame) -> None:
        """Handle termination signals."""
        logger.info("Received shutdown signal", signal=signum)
        self.shutdown()
        sys.exit(0)

    def start_process(
        self,
        args: list[str],
        env: dict[str, str],
        name: str,
        capture_output: bool = False
    ) -> subprocess.Popen:
        """
        Start a subprocess with the given arguments and environment.

        Args:
            args: Command line arguments.
            env: Environment variables.
            name: Name for logging.
            capture_output: If False, output goes to terminal (useful for debugging).

        Returns:
            The started subprocess.
        """
        logger.info(f"Starting {name}", command=args[0])

        # Use CREATE_NEW_PROCESS_GROUP on Windows for proper signal handling
        creationflags = 0
        if sys.platform == "win32":
            creationflags = subprocess.CREATE_NEW_PROCESS_GROUP

        # Don't capture output by default - let it go to terminal for debugging
        stdout = subprocess.PIPE if capture_output else None
        stderr = subprocess.STDOUT if capture_output else None

        process = subprocess.Popen(
            args,
            env=env,
            stdout=stdout,
            stderr=stderr,
            creationflags=creationflags,
        )
        self._processes.append(process)
        return process

    def shutdown(self) -> None:
        """Gracefully shutdown all managed processes."""
        if self._shutting_down:
            return
        self._shutting_down = True

        logger.info("Shutting down all processes", count=len(self._processes))

        for process in reversed(self._processes):
            if process.poll() is None:  # Still running
                try:
                    if sys.platform == "win32":
                        # On Windows, send CTRL_BREAK_EVENT
                        process.send_signal(signal.CTRL_BREAK_EVENT)
                    else:
                        process.terminate()

                    # Wait for graceful shutdown
                    process.wait(timeout=5.0)
                except subprocess.TimeoutExpired:
                    logger.warning("Process did not terminate gracefully, killing")
                    process.kill()
                except Exception as e:
                    logger.error("Error shutting down process", error=str(e))

        self._processes.clear()


def launch() -> int:
    """
    Launch the AEC Agent system.

    Starts the MCP server and Chainlit frontend with proper orchestration.

    Returns:
        Exit code (0 for success, non-zero for error).
    """
    settings = get_settings()
    setup_logging(settings.log_level, settings.log_format)

    logger.info(
        "AEC Agent Launcher starting",
        environment=settings.environment.value,
        llm_provider=settings.llm_provider.value,
    )

    # Use existing SESSION_TOKEN if set (from GPO/logon script), otherwise generate
    session_token = os.environ.get("SESSION_TOKEN")
    if session_token:
        logger.info("Using existing SESSION_TOKEN from environment")
    else:
        session_token = str(uuid.uuid4())
        logger.info("Generated new SESSION_TOKEN")

    # Check for MCP_LISTENER_PORT (sidecar port from GPO/logon script)
    sidecar_port = os.environ.get("MCP_LISTENER_PORT")
    if sidecar_port:
        logger.info("Sidecar port from environment", sidecar_port=sidecar_port)
    else:
        logger.warning(
            "MCP_LISTENER_PORT not set - sidecar communication will fail. "
            "Run the logon script first: scripts/AECAgent-Logon.ps1"
        )

    # Find available ports for MCP server and Chainlit
    try:
        mcp_port = find_free_port(settings.mcp_server_port)
        chainlit_port = find_free_port(settings.chainlit_port)
    except RuntimeError as e:
        logger.error("Failed to find free ports", error=str(e))
        return 1

    logger.info(
        "Ports allocated",
        mcp_server_port=mcp_port,
        chainlit_port=chainlit_port,
        sidecar_port=sidecar_port or "NOT SET",
    )

    # Build environment for subprocesses
    # IMPORTANT: Copy existing env to preserve MCP_LISTENER_PORT and SESSION_TOKEN
    env = os.environ.copy()
    env["MCP_SERVER_PORT"] = str(mcp_port)
    env["CHAINLIT_PORT"] = str(chainlit_port)
    env["SESSION_TOKEN"] = session_token
    # MCP_LISTENER_PORT is already in os.environ if set by logon script

    # Ensure cache directory exists
    settings.ensure_cache_dir()

    # Initialize process manager
    manager = ProcessManager()

    # Start MCP Server
    logger.info("Starting MCP Server...")
    mcp_process = manager.start_process(
        args=[sys.executable, "-m", "aec_agent.server"],
        env=env,
        name="MCP Server",
    )

    # Wait for MCP server to be ready
    if not wait_for_port(mcp_port, timeout=30.0):
        logger.error("MCP Server failed to start within timeout")
        manager.shutdown()
        return 1

    logger.info("MCP Server ready", port=mcp_port)

    # Start Chainlit Frontend
    logger.info("Starting Chainlit Frontend...")

    # Get the path to app.py
    frontend_app = Path(__file__).parent / "frontend" / "app.py"

    chainlit_process = manager.start_process(
        args=[
            sys.executable, "-m", "chainlit", "run",
            str(frontend_app),
            "--port", str(chainlit_port),
            "--host", "127.0.0.1",
        ],
        env=env,
        name="Chainlit Frontend",
    )

    # Wait for Chainlit to be ready
    if not wait_for_port(chainlit_port, timeout=30.0):
        logger.error("Chainlit failed to start within timeout")
        manager.shutdown()
        return 1

    logger.info("Chainlit Frontend ready", port=chainlit_port)

    # Print startup message
    print("\n" + "=" * 60)
    print("  AEC Agent Started Successfully!")
    print("=" * 60)
    print(f"\n  UI:           http://127.0.0.1:{chainlit_port}")
    print(f"  MCP Server:   http://127.0.0.1:{mcp_port}")
    if sidecar_port:
        print(f"  Sidecar Port: {sidecar_port}")
    else:
        print("  Sidecar Port: NOT SET (run logon script first!)")
    print(f"\n  SESSION_TOKEN: {session_token[:8]}...")
    print(f"\n  Press Ctrl+C to stop")
    print("=" * 60 + "\n")

    # Monitor processes
    try:
        while True:
            # Check if MCP server is still running
            if mcp_process.poll() is not None:
                logger.error(
                    "MCP Server exited unexpectedly",
                    exit_code=mcp_process.returncode
                )
                break

            # Check if Chainlit is still running
            if chainlit_process.poll() is not None:
                logger.error(
                    "Chainlit exited unexpectedly",
                    exit_code=chainlit_process.returncode
                )
                break

            time.sleep(1.0)
    except KeyboardInterrupt:
        logger.info("Received keyboard interrupt")
    finally:
        manager.shutdown()

    return 0


def main() -> None:
    """Entry point for the launcher."""
    sys.exit(launch())


if __name__ == "__main__":
    main()
