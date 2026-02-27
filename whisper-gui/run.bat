@echo off
cd /d "%~dp0"
call conda activate whisper-gui-env
if errorlevel 1 (
    echo ERROR: Could not activate whisper-gui-env.
    echo Please run setup_env.bat first.
    pause
    exit /b 1
)
python app.py
pause
