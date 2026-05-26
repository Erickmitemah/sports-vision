@echo off
echo ============================================================
echo   FOOTBALL INTELLIGENCE SYSTEM - Starting...
echo ============================================================
call venv\Scripts\activate.bat
echo.
echo   Dashboard → http://localhost:8000
echo   API Docs  → http://localhost:8000/docs
echo.
python app.py
pause
