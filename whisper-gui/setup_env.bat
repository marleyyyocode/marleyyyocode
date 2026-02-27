@echo off
echo ============================================
echo  Whisper GUI - One-Time Environment Setup
echo ============================================
echo.

call conda create -n whisper-gui-env python=3.10 -y
if errorlevel 1 (
    echo ERROR: Failed to create conda environment.
    pause
    exit /b 1
)

call conda activate whisper-gui-env
if errorlevel 1 (
    echo ERROR: Failed to activate conda environment.
    pause
    exit /b 1
)

pip install -r requirements.txt
if errorlevel 1 (
    echo ERROR: Failed to install dependencies.
    pause
    exit /b 1
)

echo.
echo ============================================
echo  Setup complete! Double-click run.bat to launch the app.
echo ============================================
pause
