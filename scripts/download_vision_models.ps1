# Automated Multimodal Vision Model Downloader for Ollama (Under 14GB VRAM)
$ErrorActionPreference = "Continue"

$visionModels = @(
    "llama3.2-vision:11b",
    "llava:13b",
    "minicpm-v:latest"
)

$logFile = "F:\Sadik\semantic-causal-pathway\outputs\ollama_vision_downloads.log"
"Starting download of vision models at $(Get-Date)" | Out-File -FilePath $logFile -Encoding utf8

$total = $visionModels.Count
$current = 0

foreach ($m in $visionModels) {
    $current++
    $header = "[$current/$total] Pulling vision model $m at $(Get-Date)..."
    Write-Host "`n$header" -ForegroundColor Cyan
    $header | Out-File -FilePath $logFile -Append -Encoding utf8

    $success = $false
    for ($attempt = 1; $attempt -le 3; $attempt++) {
        Write-Host "  Attempt $attempt/3 for $m..." -ForegroundColor Yellow
        ollama pull $m
        if ($LASTEXITCODE -eq 0) {
            $success = $true
            Write-Host "  Successfully pulled $m" -ForegroundColor Green
            "Successfully pulled $m at $(Get-Date)" | Out-File -FilePath $logFile -Append -Encoding utf8
            break
        } else {
            Write-Warning "  Attempt $attempt failed for $m. Retrying in 5 seconds..."
            Start-Sleep -Seconds 5
        }
    }

    if (-not $success) {
        Write-Error "Failed to pull $m after 3 attempts."
        "FAILED to pull $m at $(Get-Date)" | Out-File -FilePath $logFile -Append -Encoding utf8
    }
}

Write-Host "`n========================================================" -ForegroundColor Green
Write-Host "Vision model downloads completed at $(Get-Date)!" -ForegroundColor Green
Write-Host "Currently available models in Ollama:" -ForegroundColor Green
ollama list
ollama list | Out-File -FilePath $logFile -Append -Encoding utf8
