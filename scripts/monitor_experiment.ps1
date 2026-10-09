# Live Monitor for 500-Image Confirmatory Semantic Causal Pathway Study
$host.UI.RawUI.WindowTitle = "Semantic Causal Pathway - Live Experiment Monitor"

$logPath = "F:\Sadik\semantic-causal-pathway\.venv\outputs\pathway_confirmatory\models\qwen2_5_vl_7b\confirmatory_experiment.log"
$resultsPath = "F:\Sadik\semantic-causal-pathway\.venv\outputs\pathway_confirmatory\models\qwen2_5_vl_7b\semantic_pathway_results.jsonl"

Clear-Host
Write-Host "==================================================================" -ForegroundColor Cyan
Write-Host "  500-Image Confirmatory Semantic Causal Pathway Study Monitor    " -ForegroundColor Yellow
Write-Host "==================================================================" -ForegroundColor Cyan
Write-Host "Active Log File : $logPath" -ForegroundColor Gray
Write-Host "Results Output  : $resultsPath" -ForegroundColor Gray

if (Test-Path $resultsPath) {
    $doneCount = (Get-Content $resultsPath | Measure-Object -Line).Lines
    Write-Host "Current Status  : $doneCount / 500 samples processed" -ForegroundColor Green
}
Write-Host "Streaming live logs below (Press Ctrl+C to stop viewing)...`n" -ForegroundColor DarkGray

if (-not (Test-Path $logPath)) {
    Write-Warning "Waiting for experiment to initialize log file..."
    while (-not (Test-Path $logPath)) { Start-Sleep -Milliseconds 500 }
}

Get-Content -Path $logPath -Wait -Tail 50
