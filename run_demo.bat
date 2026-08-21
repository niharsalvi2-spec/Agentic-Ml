@echo off
setlocal EnableDelayedExpansion
title Agentic-ML Demo Launcher

echo.
echo ======================================================================
echo   Agentic-ML Autonomous Platform  ^|  Demo Launcher
echo ======================================================================
echo.

REM ── Pick Python interpreter ────────────────────────────────────────────
if exist ".\venv\Scripts\python.exe" (
    set "PYTHON=.\venv\Scripts\python.exe"
    echo [*] Using virtualenv Python: .\venv\Scripts\python.exe
) else (
    set "PYTHON=python"
    echo [*] Virtualenv not found – using system Python
)

REM ── Check uvicorn is installed ─────────────────────────────────────────
%PYTHON% -c "import uvicorn" 2>nul
if errorlevel 1 (
    echo [!] uvicorn not found. Installing ...
    %PYTHON% -m pip install uvicorn --quiet
)

REM ── Run the orchestrator ───────────────────────────────────────────────
%PYTHON% scripts\run_demo.py

pause
