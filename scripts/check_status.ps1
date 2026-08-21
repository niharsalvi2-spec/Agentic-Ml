$fe = "Not ready"
$be = "Not ready"

try {
    $r = Invoke-WebRequest -Uri "http://localhost:3000" -UseBasicParsing -TimeoutSec 3
    $fe = "ONLINE (HTTP $($r.StatusCode))"
} catch {}

try {
    $r = Invoke-WebRequest -Uri "http://localhost:8000/health" -UseBasicParsing -TimeoutSec 3
    $be = "ONLINE - $($r.Content)"
} catch {}

Write-Host "Backend  : $be"
Write-Host "Frontend : $fe"

Get-Process -Name "node","python" -ErrorAction SilentlyContinue |
    Select-Object Name, Id, CPU |
    Format-Table -AutoSize
