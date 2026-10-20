@echo off
rem One-time setup. Run "setup.bat --no-gpu" to skip the local GPU voices (about 3 GB).
setlocal
cd /d "%~dp0"

where uv >nul 2>nul
if errorlevel 1 (
  echo This setup uses "uv" to install Python packages, and it isn't installed.
  echo Install it with:  winget install astral-sh.uv
  echo then run setup.bat again.
  pause
  exit /b 1
)

if not exist ".venv\Scripts\python.exe" (
  echo Creating the Python environment...
  uv venv --python 3.12 .venv || goto :error
)

echo Installing the app...
uv pip install --python .venv\Scripts\python.exe -r requirements.txt || goto :error

if /i "%~1"=="--no-gpu" goto :done
echo.
echo Installing local GPU voices: PyTorch with CUDA and Qwen3-TTS (about 3 GB)...
uv pip install --python .venv\Scripts\python.exe "torch==2.11.0+cu128" "torchaudio==2.11.0+cu128" --index-url https://download.pytorch.org/whl/cu128 || goto :error
uv pip install --python .venv\Scripts\python.exe -r requirements-local.txt || goto :error
echo The voice model itself (about 4.5 GB) downloads the first time you pick "Local GPU" in the app.

:done
echo.
echo All set. Start the app with run.bat
pause
exit /b 0

:error
echo.
echo Setup failed. Scroll up for the error message.
pause
exit /b 1
