# Pipeline Orchestrator: Chains Primary 1,000 Study -> Ollama 13B Benchmark
$ErrorActionPreference = "Continue"
$root = "F:\Sadik\semantic-causal-pathway"
Set-Location $root

$host.UI.RawUI.WindowTitle = "Pipeline Orchestrator: 1000 Study -> Ollama 13B"

Clear-Host
Write-Host "================================================================================" -ForegroundColor Cyan
Write-Host " PIPELINE: MONITORING PRIMARY 1,000-IMAGE STUDY ON GPU...                       " -ForegroundColor Yellow
Write-Host "================================================================================" -ForegroundColor Cyan

$resultsFile = "$root\.venv\outputs\study_1000\models\qwen2_5_vl_7b\semantic_pathway_results.jsonl"

while ($true) {
    $proc = Get-Process python -ErrorAction SilentlyContinue | Where-Object {
        $cmd = (Get-CimInstance Win32_Process -Filter "ProcessId = $($_.Id)").CommandLine
        $cmd -match "run_fresh_1000_study.py"
    }

    if (-not $proc) {
        Write-Host "`n[SUCCESS] Primary 1,000-image study has finished and released GPU VRAM!" -ForegroundColor Green
        break
    }

    $doneCount = 0
    if (Test-Path $resultsFile) {
        $doneCount = (Get-Content $resultsFile | Measure-Object -Line).Lines
    }

    Write-Host "[ACTIVE] Primary study in progress: $doneCount / 1,000 images completed. Checking again in 60s..." -ForegroundColor Gray
    Start-Sleep -Seconds 60
}

Write-Host "`n================================================================================" -ForegroundColor Cyan
Write-Host " LAUNCHING VERIFIED OLLAMA 13B BENCHMARK ACROSS 1,000 IMAGES                    " -ForegroundColor Yellow
Write-Host "================================================================================" -ForegroundColor Cyan

& "$root\.venv\Scripts\python.exe" "$root\scripts\run_ollama_vision_benchmark.py" --models "llava:13b"

Write-Host "`n================================================================================" -ForegroundColor Green
Write-Host " ALL EXPERIMENTS COMPLETE! Final reports available in outputs/ollama_benchmark/ " -ForegroundColor Green
Write-Host "================================================================================" -ForegroundColor Green
