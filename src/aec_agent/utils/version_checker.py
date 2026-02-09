"""
Version compatibility checker for AEC Agent.

This utility validates that the current environment meets the
requirements defined in the version matrix.
"""

import os
import platform
import re
import sys
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

from aec_agent.config.versions import VERSION_MATRIX, SupportStatus


class CheckStatus(Enum):
    """Status of a compatibility check."""
    PASS = "pass"
    WARN = "warn"
    FAIL = "fail"
    SKIP = "skip"


@dataclass
class CheckResult:
    """Result of a single compatibility check."""
    component: str
    status: CheckStatus
    current_version: str | None
    required_version: str
    message: str


class VersionChecker:
    """
    Validates environment compatibility with AEC Agent requirements.

    Example:
        checker = VersionChecker()
        results = checker.run_all_checks()
        checker.print_report(results)
    """

    def __init__(self):
        self.matrix = VERSION_MATRIX

    def check_python_version(self) -> CheckResult:
        """Check Python version compatibility."""
        current = f"{sys.version_info.major}.{sys.version_info.minor}"
        full_version = f"{current}.{sys.version_info.micro}"

        major, minor = sys.version_info.major, sys.version_info.minor

        if major != 3:
            return CheckResult(
                component="Python",
                status=CheckStatus.FAIL,
                current_version=full_version,
                required_version="3.9 - 3.12",
                message="Python 3.x required"
            )

        if minor < 9:
            return CheckResult(
                component="Python",
                status=CheckStatus.FAIL,
                current_version=full_version,
                required_version="3.9 - 3.12",
                message="Python 3.9+ required"
            )

        if minor > 12:
            return CheckResult(
                component="Python",
                status=CheckStatus.WARN,
                current_version=full_version,
                required_version="3.9 - 3.12",
                message="Python version newer than tested"
            )

        if minor == 11:
            status = CheckStatus.PASS
            message = "Recommended version"
        else:
            status = CheckStatus.PASS
            message = "Supported version"

        return CheckResult(
            component="Python",
            status=status,
            current_version=full_version,
            required_version="3.9 - 3.12 (3.11 recommended)",
            message=message
        )

    def check_package_version(self, package: str, min_version: str) -> CheckResult:
        """Check if a Python package meets minimum version."""
        try:
            from importlib.metadata import version as get_version
            current = get_version(package)

            if self._compare_versions(current, min_version) >= 0:
                return CheckResult(
                    component=package,
                    status=CheckStatus.PASS,
                    current_version=current,
                    required_version=f">= {min_version}",
                    message="Version OK"
                )
            else:
                return CheckResult(
                    component=package,
                    status=CheckStatus.FAIL,
                    current_version=current,
                    required_version=f">= {min_version}",
                    message=f"Upgrade required: pip install {package}>={min_version}"
                )
        except Exception:
            return CheckResult(
                component=package,
                status=CheckStatus.FAIL,
                current_version=None,
                required_version=f">= {min_version}",
                message=f"Not installed: pip install {package}>={min_version}"
            )

    def check_environment_variables(self) -> list[CheckResult]:
        """Check required environment variables."""
        results = []

        # Check MCP_LISTENER_PORT
        port = os.environ.get("MCP_LISTENER_PORT")
        if port:
            try:
                port_int = int(port)
                if 20000 <= port_int <= 30000:
                    results.append(CheckResult(
                        component="MCP_LISTENER_PORT",
                        status=CheckStatus.PASS,
                        current_version=port,
                        required_version="20000-30000",
                        message="Port configured"
                    ))
                else:
                    results.append(CheckResult(
                        component="MCP_LISTENER_PORT",
                        status=CheckStatus.WARN,
                        current_version=port,
                        required_version="20000-30000",
                        message="Port outside recommended range"
                    ))
            except ValueError:
                results.append(CheckResult(
                    component="MCP_LISTENER_PORT",
                    status=CheckStatus.FAIL,
                    current_version=port,
                    required_version="20000-30000",
                    message="Invalid port number"
                ))
        else:
            results.append(CheckResult(
                component="MCP_LISTENER_PORT",
                status=CheckStatus.WARN,
                current_version=None,
                required_version="20000-30000",
                message="Not set (will be assigned by GPO script)"
            ))

        # Check SESSION_TOKEN
        token = os.environ.get("SESSION_TOKEN")
        if token:
            if len(token) >= 32:
                results.append(CheckResult(
                    component="SESSION_TOKEN",
                    status=CheckStatus.PASS,
                    current_version=f"{token[:8]}...",
                    required_version="UUID format",
                    message="Token configured"
                ))
            else:
                results.append(CheckResult(
                    component="SESSION_TOKEN",
                    status=CheckStatus.WARN,
                    current_version=f"{token[:8]}...",
                    required_version="UUID format",
                    message="Token appears short"
                ))
        else:
            results.append(CheckResult(
                component="SESSION_TOKEN",
                status=CheckStatus.WARN,
                current_version=None,
                required_version="UUID format",
                message="Not set (will be assigned by GPO script)"
            ))

        return results

    def check_os(self) -> CheckResult:
        """Check operating system compatibility."""
        system = platform.system()

        if system == "Windows":
            version = platform.version()
            # Check for Windows Server
            if "Server" in platform.platform():
                return CheckResult(
                    component="Operating System",
                    status=CheckStatus.PASS,
                    current_version=f"Windows Server ({version})",
                    required_version="Windows Server 2019/2022",
                    message="Server OS detected"
                )
            else:
                return CheckResult(
                    component="Operating System",
                    status=CheckStatus.WARN,
                    current_version=f"Windows ({version})",
                    required_version="Windows Server 2019/2022",
                    message="Development: OK, Production: Use Windows Server"
                )
        else:
            return CheckResult(
                component="Operating System",
                status=CheckStatus.WARN,
                current_version=f"{system} {platform.release()}",
                required_version="Windows Server 2019/2022",
                message="Windows required for CAD applications"
            )

    def check_dotnet(self) -> CheckResult:
        """Check .NET Framework version (Windows only)."""
        if platform.system() != "Windows":
            return CheckResult(
                component=".NET Framework",
                status=CheckStatus.SKIP,
                current_version=None,
                required_version="4.8",
                message="Windows only check"
            )

        try:
            # Check .NET Framework via registry
            import winreg
            key = winreg.OpenKey(
                winreg.HKEY_LOCAL_MACHINE,
                r"SOFTWARE\Microsoft\NET Framework Setup\NDP\v4\Full"
            )
            release, _ = winreg.QueryValueEx(key, "Release")
            winreg.CloseKey(key)

            # Release number for .NET 4.8 is 528040 or higher
            if release >= 528040:
                return CheckResult(
                    component=".NET Framework",
                    status=CheckStatus.PASS,
                    current_version="4.8+",
                    required_version="4.8",
                    message=".NET 4.8 installed"
                )
            else:
                return CheckResult(
                    component=".NET Framework",
                    status=CheckStatus.FAIL,
                    current_version=f"< 4.8 (release: {release})",
                    required_version="4.8",
                    message="Upgrade to .NET Framework 4.8"
                )
        except Exception as e:
            return CheckResult(
                component=".NET Framework",
                status=CheckStatus.WARN,
                current_version=None,
                required_version="4.8",
                message=f"Could not detect: {e}"
            )

    def check_autocad(self) -> CheckResult:
        """Check for AutoCAD installation."""
        if platform.system() != "Windows":
            return CheckResult(
                component="AutoCAD",
                status=CheckStatus.SKIP,
                current_version=None,
                required_version="2021-2025",
                message="Windows only check"
            )

        # Common AutoCAD installation paths
        acad_paths = [
            Path(r"C:\Program Files\Autodesk\AutoCAD 2024"),
            Path(r"C:\Program Files\Autodesk\AutoCAD 2023"),
            Path(r"C:\Program Files\Autodesk\AutoCAD 2022"),
            Path(r"C:\Program Files\Autodesk\AutoCAD 2021"),
            Path(r"C:\Program Files\Autodesk\AutoCAD 2025"),
        ]

        for path in acad_paths:
            if path.exists():
                version = path.name.split()[-1]
                version_info = self.matrix.get_autocad_version(version)

                if version_info:
                    if version_info.status == SupportStatus.RECOMMENDED:
                        status = CheckStatus.PASS
                        message = "Recommended version"
                    elif version_info.status == SupportStatus.SUPPORTED:
                        status = CheckStatus.PASS
                        message = "Supported version"
                    elif version_info.status == SupportStatus.TESTING:
                        status = CheckStatus.WARN
                        message = "Testing version"
                    else:
                        status = CheckStatus.WARN
                        message = version_info.notes or "Check compatibility"
                else:
                    status = CheckStatus.WARN
                    message = "Version not in compatibility matrix"

                return CheckResult(
                    component="AutoCAD",
                    status=status,
                    current_version=version,
                    required_version="2021-2025 (2024 recommended)",
                    message=message
                )

        return CheckResult(
            component="AutoCAD",
            status=CheckStatus.WARN,
            current_version=None,
            required_version="2021-2025 (2024 recommended)",
            message="Not detected in standard paths"
        )

    def check_revit(self) -> CheckResult:
        """Check for Revit installation."""
        if platform.system() != "Windows":
            return CheckResult(
                component="Revit",
                status=CheckStatus.SKIP,
                current_version=None,
                required_version="2021-2025",
                message="Windows only check"
            )

        # Common Revit installation paths
        revit_paths = [
            Path(r"C:\Program Files\Autodesk\Revit 2024"),
            Path(r"C:\Program Files\Autodesk\Revit 2023"),
            Path(r"C:\Program Files\Autodesk\Revit 2022"),
            Path(r"C:\Program Files\Autodesk\Revit 2021"),
            Path(r"C:\Program Files\Autodesk\Revit 2025"),
        ]

        for path in revit_paths:
            if path.exists():
                version = path.name.split()[-1]
                version_info = self.matrix.get_revit_version(version)

                if version_info:
                    if version_info.status == SupportStatus.RECOMMENDED:
                        status = CheckStatus.PASS
                        message = "Recommended version"
                    elif version_info.status == SupportStatus.SUPPORTED:
                        status = CheckStatus.PASS
                        message = "Supported version"
                    elif version_info.status == SupportStatus.TESTING:
                        status = CheckStatus.WARN
                        message = "Testing version - .NET 8 required"
                    else:
                        status = CheckStatus.WARN
                        message = version_info.notes or "Check compatibility"
                else:
                    status = CheckStatus.WARN
                    message = "Version not in compatibility matrix"

                return CheckResult(
                    component="Revit",
                    status=status,
                    current_version=version,
                    required_version="2021-2025 (2024 recommended)",
                    message=message
                )

        return CheckResult(
            component="Revit",
            status=CheckStatus.WARN,
            current_version=None,
            required_version="2021-2025 (2024 recommended)",
            message="Not detected in standard paths"
        )

    def check_pyrevit(self) -> CheckResult:
        """Check for pyRevit installation."""
        if platform.system() != "Windows":
            return CheckResult(
                component="pyRevit",
                status=CheckStatus.SKIP,
                current_version=None,
                required_version="4.8.12+",
                message="Windows only check"
            )

        # Check common pyRevit paths
        pyrevit_paths = [
            Path(os.environ.get("APPDATA", "")) / "pyRevit-Master",
            Path(os.environ.get("PROGRAMDATA", "")) / "pyRevit",
            Path(r"C:\pyRevit"),
        ]

        for path in pyrevit_paths:
            if path.exists():
                # Try to get version from version file
                version_file = path / "pyrevitlib" / "pyrevit" / "version"
                if version_file.exists():
                    version = version_file.read_text().strip()
                else:
                    version = "detected"

                return CheckResult(
                    component="pyRevit",
                    status=CheckStatus.PASS,
                    current_version=version,
                    required_version="4.8.12+ (5.0+ for Revit 2025)",
                    message="pyRevit installed"
                )

        return CheckResult(
            component="pyRevit",
            status=CheckStatus.WARN,
            current_version=None,
            required_version="4.8.12+ (5.0+ for Revit 2025)",
            message="Not detected - install from pyrevitlabs.io"
        )

    def run_all_checks(self) -> list[CheckResult]:
        """Run all compatibility checks."""
        results = []

        # System checks
        results.append(self.check_python_version())
        results.append(self.check_os())
        results.append(self.check_dotnet())

        # CAD application checks
        results.append(self.check_autocad())
        results.append(self.check_revit())
        results.append(self.check_pyrevit())

        # Environment variable checks
        results.extend(self.check_environment_variables())

        # Python package checks
        packages = [
            ("httpx", "0.27.0"),
            ("pydantic", "2.0.0"),
            ("tenacity", "8.2.0"),
            ("aiosqlite", "0.19.0"),
        ]
        for package, min_version in packages:
            results.append(self.check_package_version(package, min_version))

        return results

    def print_report(self, results: list[CheckResult]) -> None:
        """Print a formatted compatibility report."""
        print("\n" + "=" * 60)
        print("AEC Agent - Compatibility Check Report")
        print("=" * 60 + "\n")

        # Status symbols
        symbols = {
            CheckStatus.PASS: "[PASS]",
            CheckStatus.WARN: "[WARN]",
            CheckStatus.FAIL: "[FAIL]",
            CheckStatus.SKIP: "[SKIP]",
        }

        # Group by status
        passed = [r for r in results if r.status == CheckStatus.PASS]
        warned = [r for r in results if r.status == CheckStatus.WARN]
        failed = [r for r in results if r.status == CheckStatus.FAIL]
        skipped = [r for r in results if r.status == CheckStatus.SKIP]

        # Print results by status
        for result in results:
            symbol = symbols[result.status]
            version_str = result.current_version or "N/A"
            print(f"{symbol} {result.component}")
            print(f"       Current:  {version_str}")
            print(f"       Required: {result.required_version}")
            print(f"       Status:   {result.message}")
            print()

        # Summary
        print("-" * 60)
        print("Summary:")
        print(f"  Passed:  {len(passed)}")
        print(f"  Warnings: {len(warned)}")
        print(f"  Failed:  {len(failed)}")
        print(f"  Skipped: {len(skipped)}")
        print("-" * 60)

        if failed:
            print("\n[!] Some checks failed. Please address the issues above.")
            sys.exit(1)
        elif warned:
            print("\n[!] Some checks have warnings. Review before production use.")
        else:
            print("\n[OK] All checks passed!")

    @staticmethod
    def _compare_versions(v1: str, v2: str) -> int:
        """
        Compare two version strings.
        Returns: -1 if v1 < v2, 0 if equal, 1 if v1 > v2
        """
        def normalize(v):
            return [int(x) for x in re.sub(r'[^0-9.]', '', v).split('.')]

        n1, n2 = normalize(v1), normalize(v2)

        # Pad to same length
        while len(n1) < len(n2):
            n1.append(0)
        while len(n2) < len(n1):
            n2.append(0)

        for a, b in zip(n1, n2):
            if a < b:
                return -1
            elif a > b:
                return 1
        return 0


def main(argv=None):
    """CLI entry point for version checker."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Check AEC Agent environment compatibility"
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output results as JSON"
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Treat warnings as failures"
    )

    args = parser.parse_args(argv)

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

        # Exit code based on results
        if any(r.status == CheckStatus.FAIL for r in results):
            sys.exit(1)
        elif args.strict and any(r.status == CheckStatus.WARN for r in results):
            sys.exit(1)
    else:
        checker.print_report(results)


if __name__ == "__main__":
    main()
