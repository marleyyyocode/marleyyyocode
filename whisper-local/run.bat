@echo off
call conda activate whisper-env
cd /d "%~dp0"
python whisper_app.py
pause
