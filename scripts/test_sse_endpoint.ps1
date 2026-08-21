$proc = Start-Process -PassThru -NoNewWindow `
    -FilePath ".\venv\Scripts\python.exe" `
    -ArgumentList "-m","uvicorn","src.agentic_ml.api.main:app","--host","0.0.0.0","--port","8001","--log-level","warning"

Start-Sleep -Seconds 6

try {
    $body = '{"prompt":"Predict Customer Churn"}'
    $resp = Invoke-WebRequest -Uri "http://localhost:8001/api/pipeline/stream" `
        -Method POST -Body $body -ContentType "application/json" `
        -UseBasicParsing -TimeoutSec 10

    Write-Host "Status: $($resp.StatusCode)"
    $snippet = $resp.Content.Substring(0, [Math]::Min(400, $resp.Content.Length))
    Write-Host "Body snippet:"
    Write-Host $snippet
} catch {
    Write-Host "Request failed: $_"
} finally {
    Stop-Process -Id $proc.Id -ErrorAction SilentlyContinue
}
