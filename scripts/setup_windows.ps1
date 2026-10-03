$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root

if (-not (Test-Path ".venv\Scripts\python.exe")) {
    & py -3.11 -m venv .venv
}
$VenvPython = Join-Path $Root ".venv\Scripts\python.exe"
& $VenvPython -m pip install --upgrade pip
& $VenvPython -m pip install -r requirements.txt
& $VenvPython -m ipykernel install --sys-prefix --name semantic-circuits --display-name "Python (Semantic Circuits)"
New-Item -ItemType Directory -Force -Path ".venv\hf_home", ".venv\data" | Out-Null
Write-Host "Environment ready at $Root\.venv"
Write-Host "If you need CUDA, install the appropriate PyTorch build from https://pytorch.org/get-started/locally/ before running model commands."
