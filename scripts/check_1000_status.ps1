# Status and Progress Checker for 1,000-Image Confirmatory Study
$ErrorActionPreference = "SilentlyContinue"

$root = "F:\Sadik\semantic-causal-pathway"
$resultsFile = "$root\.venv\outputs\study_1000\models\qwen2_5_vl_7b\semantic_pathway_results.jsonl"
$logFile = "$root\.venv\outputs\study_1000\models\qwen2_5_vl_7b\study_1000_experiment.log"

Clear-Host
Write-Host "================================================================================" -ForegroundColor Cyan
Write-Host "  1,000-IMAGE CONFIRMATORY STUDY (400 TRAIN / 200 VAL / 400 LOCKED TEST)        " -ForegroundColor Yellow
Write-Host "================================================================================" -ForegroundColor Cyan

# 1. Check Process
$proc = Get-Process python -ErrorAction SilentlyContinue | Where-Object {
    $cmd = (Get-CimInstance Win32_Process -Filter "ProcessId = $($_.Id)").CommandLine
    $cmd -match "run_fresh_1000_study.py"
}

if ($proc) {
    $pids = ($proc | ForEach-Object { $_.Id }) -join ", "
    $mem = [math]::Round((($proc | Measure-Object -Property WorkingSet64 -Sum).Sum / 1MB), 1)
    Write-Host "[ACTIVE] Experiment Orchestrator is RUNNING (PID: $pids, RAM: $mem MB)" -ForegroundColor Green
} else {
    Write-Host "[IDLE] Experiment Orchestrator is NOT currently running." -ForegroundColor Yellow
}

# 2. Check GPU
$gpuInfo = nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu,temperature.gpu --format=csv,noheader,nounits 2>$null
if ($gpuInfo) {
    $parts = $gpuInfo.Split(',')
    Write-Host "GPU Status: $($parts[0].Trim()) | VRAM: $($parts[1].Trim()) / $($parts[2].Trim()) MB | Util: $($parts[3].Trim())% | Temp: $($parts[4].Trim()) C" -ForegroundColor Gray
}

# 3. Progress Breakdown
if (Test-Path $resultsFile) {
    $lines = Get-Content $resultsFile
    $totalCount = $lines.Count
    $trainCount = 0
    $valCount = 0
    $testCount = 0
    $vqaCount = 0
    $gqaCount = 0

    foreach ($l in $lines) {
        if ($l -match '"analysis_split":\s*"train"') { $trainCount++ }
        elseif ($l -match '"analysis_split":\s*"validation"') { $valCount++ }
        elseif ($l -match '"analysis_split":\s*"test"') { $testCount++ }
        if ($l -match '"dataset":\s*"VQA v2"') { $vqaCount++ }
        elseif ($l -match '"dataset":\s*"GQA"') { $gqaCount++ }
    }

    $pct = [math]::Round(($totalCount / 1000) * 100, 1)

    Write-Host "`n--------------------------------------------------------------------------------" -ForegroundColor Cyan
    Write-Host " OVERALL PROGRESS: $totalCount / 1,000 Images ($pct%)" -ForegroundColor White
    Write-Host "--------------------------------------------------------------------------------" -ForegroundColor Cyan
    Write-Host "  * Train Split       : $trainCount / 400 ($([math]::Round(($trainCount/400)*100, 1))%)" -ForegroundColor $(if ($trainCount -eq 400) { "Green" } else { "Yellow" })
    Write-Host "  * Validation Split  : $valCount / 200 ($([math]::Round(($valCount/200)*100, 1))%)" -ForegroundColor $(if ($valCount -eq 200) { "Green" } else { "Yellow" })
    Write-Host "  * Locked Test Split : $testCount / 400 ($([math]::Round(($testCount/400)*100, 1))%)" -ForegroundColor $(if ($testCount -eq 400) { "Green" } else { "DarkGray" })
    Write-Host "  * Dataset Coverage  : VQAv2 = $vqaCount / 500 | GQA = $gqaCount / 500" -ForegroundColor Gray
} else {
    Write-Host "`nResults file not found yet ($resultsFile)." -ForegroundColor Red
}

# 4. Recent Log Lines
Write-Host "`n--------------------------------------------------------------------------------" -ForegroundColor Cyan
Write-Host " RECENT EVALUATIONS (Tail of Log)" -ForegroundColor White
Write-Host "--------------------------------------------------------------------------------" -ForegroundColor Cyan
if (Test-Path $logFile) {
    Get-Content $logFile -Tail 5 | ForEach-Object { Write-Host "  $_" -ForegroundColor Gray }
} else {
    Write-Host "  Log file not found." -ForegroundColor DarkGray
}

Write-Host "`n================================================================================" -ForegroundColor Cyan
Write-Host " HELPFUL COMMANDS:" -ForegroundColor Yellow
Write-Host "  1. Watch live streaming log : Get-Content $logFile -Wait -Tail 30" -ForegroundColor White
Write-Host "  2. Compute statistical table: .\.venv\Scripts\python.exe scripts\report_1000_metrics.py" -ForegroundColor White
Write-Host "  3. Launch / Resume study    : .\.venv\Scripts\python.exe scripts\run_fresh_1000_study.py" -ForegroundColor White
Write-Host "================================================================================`n" -ForegroundColor Cyan
