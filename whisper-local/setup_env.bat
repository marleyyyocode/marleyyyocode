@echo off
echo ============================================================
echo  Whisper Local - One-Time Environment Setup
echo ============================================================
echo.

call conda create -n whisper-env python=3.10 -y
if errorlevel 1 (
    echo ERROR: Failed to create conda environment.
    pause
    exit /b 1
)

call conda activate whisper-env
if errorlevel 1 (
    echo ERROR: Failed to activate conda environment.
    pause
    exit /b 1
)

pip install -r requirements.txt
if errorlevel 1 (
    echo ERROR: Failed to install requirements.
    pause
    exit /b 1
)

echo.
echo ============================================================
echo  Setup complete! Use run.bat to launch the app next time.
echo ============================================================
pause
