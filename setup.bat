@echo off
echo ============================================================
echo   FOOTBALL INTELLIGENCE SYSTEM - Setup
echo ============================================================

echo.
echo [1/4] Creating virtual environment...
python -m venv venv
if errorlevel 1 (
    echo ERROR: Python not found. Install Python 3.10+ from https://python.org
    pause
    exit /b 1
)

echo.
echo [2/4] Activating virtual environment...
call venv\Scripts\activate.bat

echo.
echo [3/4] Installing dependencies...
pip install --upgrade pip
pip install fastapi uvicorn[standard] numpy pandas scikit-learn joblib statsbombpy pydantic

echo.
echo [4/4] Setup complete!
echo ============================================================
echo   Run the app:  run.bat
echo   Or manually:  venv\Scripts\activate ^&^& python app.py
echo ============================================================
pause
