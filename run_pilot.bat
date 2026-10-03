@echo off
setlocal
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  echo Project environment is missing. Follow the README setup steps first.
  pause
  exit /b 1
)
".venv\Scripts\python.exe" scripts\run_pilot.py %*
if errorlevel 1 echo Pilot stopped. Read the error above and the README troubleshooting section.
pause
