@echo off
REM AEC Agent GPO Logon Script Wrapper
REM This batch file calls the PowerShell script for GPO compatibility

REM Get the directory where this script is located
set SCRIPT_DIR=%~dp0

REM Run PowerShell script with execution policy bypass (required for GPO)
powershell.exe -ExecutionPolicy Bypass -NoProfile -File "%SCRIPT_DIR%AECAgent-Logon.ps1"

exit /b %ERRORLEVEL%
