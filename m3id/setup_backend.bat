@echo off
echo ==========================================
echo   M3-ID -- Multi-Modal Identity Defender
echo ==========================================

python --version >nul 2>&1 || (echo ERROR: Python not found. Install Python 3.10+ && pause && exit /b 1)

echo.
echo [1/4] Creating virtual environment...
cd backend
python -m venv venv
call venv\Scripts\activate

echo [2/4] Installing dependencies...
pip install --upgrade pip -q
pip install -r requirements.txt

echo [3/4] Initialising database...
cd ..
python -c "from backend.core.database import init_db; init_db(); print('DB ready.')"

echo [4/4] Starting backend server...
echo.
echo   Swagger UI : http://localhost:8000/docs
echo   Health     : http://localhost:8000/health
echo   Frontend   : Open frontend/index.html with Live Server in VS Code
echo   Demo login : demo@m3id.ai / Demo@1234
echo.
uvicorn backend.main:app --reload --port 8000
pause
