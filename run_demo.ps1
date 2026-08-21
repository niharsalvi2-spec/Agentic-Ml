# Agentic-ML Automated Demo Launcher for PowerShell
Write-Host "======================================================================" -ForegroundColor Cyan
Write-Host "  Agentic-ML Autonomous Platform Launcher (PowerShell)" -ForegroundColor Cyan
Write-Host "======================================================================" -ForegroundColor Cyan

$PythonPath = ".\venv\Scripts\python.exe"
if (-Not (Test-Path $PythonPath)) {
    $PythonPath = "python"
}

& $PythonPath scripts\run_demo.py
