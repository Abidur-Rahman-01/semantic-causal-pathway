# Live Monitor for 1,000-Image Study (400 Train / 200 Val / 400 Locked Test)
$host.UI.RawUI.WindowTitle = "1,000-Image Confirmatory Study - Live Monitor"

$logPath = "F:\Sadik\semantic-causal-pathway\.venv\outputs\study_1000\models\qwen2_5_vl_7b\study_1000_experiment.log"
$resultsPath = "F:\Sadik\semantic-causal-pathway\.venv\outputs\study_1000\models\qwen2_5_vl_7b\semantic_pathway_results.jsonl"

Clear-Host
Write-Host "==================================================================" -ForegroundColor Cyan
Write-Host " Fresh 1,000-Image Confirmatory Study Live Monitor (400/200/400)   " -ForegroundColor Yellow
Write-Host "==================================================================" -ForegroundColor Cyan
Write-Host "Active Log File : $logPath" -ForegroundColor Gray
Write-Host "Results Output  : $resultsPath" -ForegroundColor Gray

if (-not (Test-Path $logPath)) {
    Write-Warning "Waiting for experiment to initialize log file..."
    while (-not (Test-Path $logPath)) { Start-Sleep -Milliseconds 500 }
}

if (Test-Path $resultsPath) {
    $doneCount = (Get-Content $resultsPath | Measure-Object -Line).Lines
    Write-Host "Current Progress: $doneCount / 1,000 samples processed`n" -ForegroundColor Green
}

Write-Host "Streaming live logs below (Press Ctrl+C to close)...`n" -ForegroundColor DarkGray
Get-Content -Path $logPath -Wait -Tail 50
