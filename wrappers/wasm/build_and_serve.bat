@echo off
REM Build and Serve ZXing WASM Demo
REM This batch file wraps the PowerShell script for easy execution

echo === ZXing-cpp WASM Build and Serve ===
echo.

REM Check if PowerShell is available
where powershell >nul 2>&1
if %ERRORLEVEL% neq 0 (
    echo ERROR: PowerShell not found
    pause
    exit /b 1
)

REM Run the PowerShell script
powershell -ExecutionPolicy Bypass -File "%~dp0build_and_serve.ps1" %*

pause
