#!/bin/bash
echo "============================================================"
echo "  FOOTBALL INTELLIGENCE SYSTEM - Setup"
echo "============================================================"

echo ""
echo "[1/4] Creating virtual environment..."
python3 -m venv venv
if [ $? -ne 0 ]; then
    echo "ERROR: Python 3 not found. Install Python 3.10+ first."
    exit 1
fi

echo ""
echo "[2/4] Activating virtual environment..."
source venv/bin/activate

echo ""
echo "[3/4] Installing dependencies..."
pip install --upgrade pip
pip install fastapi "uvicorn[standard]" numpy pandas scikit-learn joblib statsbombpy pydantic

echo ""
echo "[4/4] Setup complete!"
echo "============================================================"
echo "  Run the app:  ./run.sh"
echo "  Or manually:  source venv/bin/activate && python app.py"
echo "============================================================"
