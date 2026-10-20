@echo off
rem Starts VN Voice Actors. Use "run.bat console" to keep a console window open for troubleshooting.
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  echo Run setup.bat first.
  pause
  exit /b 1
)
if /i "%~1"=="console" (
  ".venv\Scripts\python.exe" -m vn_voice
  pause
) else (
  start "" ".venv\Scripts\pythonw.exe" -m vn_voice
)
