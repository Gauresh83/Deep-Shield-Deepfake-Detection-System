#!/bin/bash
# M3-ID Setup Script (Mac/Linux)
set -e
echo "=========================================="
echo "  M3-ID — Multi-Modal Identity Defender"
echo "=========================================="

# Check Python version
python3 --version 2>/dev/null || { echo "ERROR: Python 3 not found. Install Python 3.10+"; exit 1; }

echo ""
echo "[1/4] Creating virtual environment..."
cd backend
python3 -m venv venv
source venv/bin/activate

echo "[2/4] Installing dependencies..."
pip install --upgrade pip -q
pip install -r requirements.txt

echo "[3/4] Initialising database..."
cd ..
python3 -c "from backend.core.database import init_db; init_db(); print('DB ready.')"

echo "[4/4] Starting backend server..."
echo ""
echo "  Swagger UI : http://localhost:8000/docs"
echo "  Health     : http://localhost:8000/health"
echo "  Frontend   : Open frontend/index.html with Live Server in VS Code"
echo "  Demo login : demo@m3id.ai / Demo@1234"
echo ""
uvicorn backend.main:app --reload --port 8000
