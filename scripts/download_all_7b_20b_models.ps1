# Automated Multi-Model Downloader for Ollama (7B to 20B Models)
$ErrorActionPreference = "Continue"

$models = @(
    "llama3.1:8b",
    "mistral:7b",
    "gemma2:9b",
    "qwen2.5:7b",
    "qwen2.5:14b",
    "qwen2.5-coder:7b",
    "qwen2.5-coder:14b",
    "deepseek-r1:8b",
    "deepseek-r1:14b",
    "phi-4:14b"
)

$logFile = "F:\Sadik\semantic-causal-pathway\outputs\ollama_downloads.log"
"Starting download of $( $models.Count ) models at $(Get-Date)" | Out-File -FilePath $logFile -Encoding utf8

$total = $models.Count
$current = 0

foreach ($m in $models) {
    $current++
    $header = "[$current/$total] Pulling $m at $(Get-Date)..."
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
Write-Host "All downloads completed at $(Get-Date)!" -ForegroundColor Green
Write-Host "Installed models in Ollama:" -ForegroundColor Green
ollama list
ollama list | Out-File -FilePath $logFile -Append -Encoding utf8
