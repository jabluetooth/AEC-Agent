"""
AEC Agent Command Line Interface.

Provides commands for checking compatibility, running the agent,
and managing the system.
"""

import argparse
import sys
from typing import Optional

from aec_agent import __version__
from aec_agent.config.settings import get_settings
from aec_agent.utils.version_checker import VersionChecker


def cmd_launch(args: argparse.Namespace) -> int:
    """Launch the full AEC Agent system (MCP server + Chainlit UI)."""
    from aec_agent.launcher import launch
    return launch()


def cmd_server(args: argparse.Namespace) -> int:
    """Run only the MCP server."""
    from aec_agent.server import main as server_main
    try:
        server_main()
    except KeyboardInterrupt:
        pass
    return 0


def cmd_ui(args: argparse.Namespace) -> int:
    """Run only the Chainlit UI (assumes MCP server is running)."""
    import subprocess
    from pathlib import Path

    settings = get_settings()
    frontend_app = Path(__file__).parent / "frontend" / "app.py"

    print(f"Starting Chainlit UI on port {settings.chainlit_port}...")

    result = subprocess.run(
        [
            sys.executable, "-m", "chainlit", "run",
            str(frontend_app),
            "--port", str(settings.chainlit_port),
            "--host", "127.0.0.1",
        ],
    )
    return result.returncode


def cmd_check(args: argparse.Namespace) -> int:
    """Run compatibility checks."""
    checker = VersionChecker()
    results = checker.run_all_checks()

    if args.json:
        import json
        output = [
            {
                "component": r.component,
                "status": r.status.value,
                "current_version": r.current_version,
                "required_version": r.required_version,
                "message": r.message,
            }
            for r in results
        ]
        print(json.dumps(output, indent=2))
    else:
        checker.print_report(results)

    # Return exit code based on results
    from aec_agent.utils.version_checker import CheckStatus
    if any(r.status == CheckStatus.FAIL for r in results):
        return 1
    if args.strict and any(r.status == CheckStatus.WARN for r in results):
        return 1
    return 0


def cmd_config(args: argparse.Namespace) -> int:
    """Display current configuration."""
    settings = get_settings()

    if args.json:
        import json
        # Convert settings to dict, excluding sensitive values
        config_dict = {}
        for key, value in settings.model_dump().items():
            if 'key' in key.lower() or 'secret' in key.lower() or 'token' in key.lower():
                config_dict[key] = "***REDACTED***" if value else None
            else:
                config_dict[key] = str(value) if value is not None else None
        print(json.dumps(config_dict, indent=2))
    else:
        print(f"\nAEC Agent Configuration")
        print("=" * 50)
        print(f"Environment:      {settings.environment.value}")
        print(f"LLM Provider:     {settings.llm_provider.value}")
        print(f"Chainlit Port:    {settings.chainlit_port}")
        print(f"MCP Server Port:  {settings.mcp_server_port}")
        print(f"Sidecar Port:     {settings.mcp_listener_port or 'Not set (GPO)'}")
        print(f"Cache Directory:  {settings.cache_dir}")
        print(f"Log Level:        {settings.log_level}")
        print(f"Log Format:       {settings.log_format.value}")
        print("=" * 50)

    return 0


def cmd_version(args: argparse.Namespace) -> int:
    """Display version information."""
    from aec_agent.config.versions import VERSION_MATRIX

    if args.matrix:
        print("\nAutoCAD Versions:")
        print("-" * 60)
        for v in VERSION_MATRIX.autocad_versions:
            status_icon = {
                "recommended": "[*]",
                "supported": "[+]",
                "testing": "[?]",
                "deprecated": "[-]",
            }.get(v.status.value, "[ ]")
            print(f"  {status_icon} {v.version} (.NET {v.dotnet_framework}) - {v.notes or v.status.value}")

        print("\nRevit Versions:")
        print("-" * 60)
        for v in VERSION_MATRIX.revit_versions:
            status_icon = {
                "recommended": "[*]",
                "supported": "[+]",
                "testing": "[?]",
                "deprecated": "[-]",
            }.get(v.status.value, "[ ]")
            print(f"  {status_icon} {v.version} (.NET {v.dotnet_framework}, pyRevit {v.pyrevit_version}) - {v.notes or v.status.value}")

        print("\nLegend: [*] Recommended  [+] Supported  [?] Testing  [-] Deprecated")
    else:
        print(f"AEC Agent v{__version__}")

    return 0


def cmd_init(args: argparse.Namespace) -> int:
    """Initialize the AEC Agent environment."""
    import os
    from pathlib import Path

    settings = get_settings()

    print("Initializing AEC Agent environment...")

    # Create cache directory
    cache_dir = settings.ensure_cache_dir()
    print(f"  Created cache directory: {cache_dir}")

    # Create logs directory
    log_dir = cache_dir.parent / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    print(f"  Created logs directory: {log_dir}")

    # Create .env file if it doesn't exist
    env_file = Path.cwd() / ".env"
    if not env_file.exists():
        env_example = Path.cwd() / ".env.example"
        if env_example.exists():
            import shutil
            shutil.copy(env_example, env_file)
            print(f"  Created .env from .env.example")
        else:
            print(f"  Skipped .env (no .env.example found)")
    else:
        print(f"  .env already exists")

    print("\nInitialization complete!")
    print("\nNext steps:")
    print("  1. Edit .env and add your API keys")
    print("  2. Run 'aec-check-compat' to verify system compatibility")
    print("  3. Start the agent with 'aec-agent run'")

    return 0


def main(argv: Optional[list[str]] = None) -> int:
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        prog="aec-agent",
        description="AEC Agent - AI-powered AutoCAD and Revit automation",
    )
    parser.add_argument(
        "-V", "--version",
        action="version",
        version=f"%(prog)s {__version__}"
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Check command
    check_parser = subparsers.add_parser(
        "check",
        help="Check system compatibility"
    )
    check_parser.add_argument(
        "--json",
        action="store_true",
        help="Output results as JSON"
    )
    check_parser.add_argument(
        "--strict",
        action="store_true",
        help="Treat warnings as failures"
    )
    check_parser.set_defaults(func=cmd_check)

    # Config command
    config_parser = subparsers.add_parser(
        "config",
        help="Display current configuration"
    )
    config_parser.add_argument(
        "--json",
        action="store_true",
        help="Output as JSON"
    )
    config_parser.set_defaults(func=cmd_config)

    # Version command
    version_parser = subparsers.add_parser(
        "version",
        help="Display version information"
    )
    version_parser.add_argument(
        "--matrix",
        action="store_true",
        help="Show full version compatibility matrix"
    )
    version_parser.set_defaults(func=cmd_version)

    # Init command
    init_parser = subparsers.add_parser(
        "init",
        help="Initialize AEC Agent environment"
    )
    init_parser.set_defaults(func=cmd_init)

    # Launch command (full system)
    launch_parser = subparsers.add_parser(
        "launch",
        help="Launch full AEC Agent system (MCP server + Chainlit UI)"
    )
    launch_parser.set_defaults(func=cmd_launch)

    # Server command (MCP server only)
    server_parser = subparsers.add_parser(
        "server",
        help="Run only the MCP server"
    )
    server_parser.set_defaults(func=cmd_server)

    # UI command (Chainlit only)
    ui_parser = subparsers.add_parser(
        "ui",
        help="Run only the Chainlit UI (requires MCP server to be running)"
    )
    ui_parser.set_defaults(func=cmd_ui)

    # Parse arguments
    args = parser.parse_args(argv)

    if args.command is None:
        parser.print_help()
        return 0

    # Execute command
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
