@echo off
call C:\Users\twder\anaconda3\Scripts\activate.bat
call conda activate whisper-gui-env
cd /d "%~dp0"
python app.py
pause
