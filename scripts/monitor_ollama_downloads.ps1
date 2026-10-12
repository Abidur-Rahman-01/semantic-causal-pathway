# Monitor for Ollama 7B-20B Model Downloads
$host.UI.RawUI.WindowTitle = "Ollama 7B-20B Downloads Monitor"

$logFile = "F:\Sadik\semantic-causal-pathway\outputs\ollama_downloads.log"

Clear-Host
Write-Host "==================================================================" -ForegroundColor Cyan
Write-Host "  Ollama 7B to 20B Multi-Model Download Monitor                  " -ForegroundColor Yellow
Write-Host "==================================================================" -ForegroundColor Cyan
Write-Host "Active Log File: $logFile`n" -ForegroundColor Gray

if (-not (Test-Path $logFile)) {
    Write-Warning "Waiting for download log to initialize..."
    while (-not (Test-Path $logFile)) { Start-Sleep -Milliseconds 500 }
}

Write-Host "Currently Installed Models in Ollama:" -ForegroundColor Green
ollama list
Write-Host "`nStreaming download progress below (Press Ctrl+C to close)...`n" -ForegroundColor DarkGray

Get-Content -Path $logFile -Wait -Tail 30
