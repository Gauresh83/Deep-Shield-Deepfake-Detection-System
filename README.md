# Deep-Shield-Deepfake-Detection-System
# M3-ID — Multi-Modal Identity Defender

> Detects AI-generated voice clones, deepfake faces, and synthetic text using a multi-modal fusion engine.

---

## What's Inside

| Part | Module | Status |
|------|--------|--------|
| 1 | Frontend — Landing, Login, Signup, Dashboard | ✅ |
| 2 | Backend — FastAPI + JWT + SQLite | ✅ |
| 3 | Voice Clone Detection — MFCC + CNN-LSTM | ✅ |
| 4 | Face Deepfake Detection — EfficientNet-B4 + OpenCV | ✅ |
| 5 | Linguistic NLP — BERT + Statistical Analysis | ✅ |
| 6 | Risk Fusion Engine — XAI + Confidence Intervals | ✅ |
| 7 | Final Integration — WebSocket + Alerts + Export + Analytics | ✅ |

---

## Project Structure

```
m3id/
├── backend/                   ← FastAPI backend (Python)
│   ├── main.py                ← App entry point
│   ├── .env                   ← Environment config (edit this)
│   ├── requirements.txt       ← Python dependencies
│   ├── core/                  ← Config, DB, security
│   ├── models/                ← SQLAlchemy DB models
│   ├── schemas/               ← Pydantic schemas
│   ├── routes/                ← All API route handlers
│   │   ├── auth.py            ← Login / Register / Refresh
│   │   ├── users.py           ← User profile and stats
│   │   ├── scans.py           ← Scan CRUD
│   │   ├── voice.py           ← Voice clone API
│   │   ├── face.py            ← Face deepfake API
│   │   ├── nlp.py             ← NLP text API
│   │   ├── fusion.py          ← Multi-modal fusion API
│   │   ├── alerts.py          ← Alert notifications
│   │   ├── export.py          ← CSV/JSON export
│   │   ├── analytics.py       ← Trends and metrics
│   │   └── ws.py              ← WebSocket real-time stream
│   ├── modules/               ← AI detection engines
│   │   ├── voice/detector.py  ← Voice analysis (tiered)
│   │   ├── face/detector.py   ← Face analysis (tiered)
│   │   ├── nlp/detector.py    ← NLP analysis (tiered)
│   │   └── fusion/engine.py   ← Risk score fusion
│   ├── services/              ← Business logic
│   └── utils/                 ← File handlers
│
├── frontend/                  ← Static HTML/CSS/JS frontend
│   ├── index.html             ← Landing page
│   └── pages/
│       ├── login.html         ← Auth
│       ├── signup.html        ← Auth
│       ├── dashboard.html     ← Main dashboard
│       ├── voice.html         ← Voice scan UI
│       ├── face.html          ← Face scan UI
│       ├── nlp.html           ← NLP scan UI
│       ├── fusion.html        ← Fusion scan UI
│       ├── history.html       ← Scan history
│       └── alerts.html        ← Alert centre
│
├── .vscode/                   ← Pre-configured VS Code settings
│   ├── settings.json          ← Live Server + Python config
│   ├── launch.json            ← Debug config for backend
│   └── extensions.json        ← Recommended extensions
│
├── setup_backend.sh           ← One-command setup (Mac/Linux)
├── setup_backend.bat          ← One-command setup (Windows)
├── Dockerfile                 ← Docker support
└── docker-compose.yml         ← Docker Compose (optional)
```

---

## HOW TO RUN IN VS CODE (Step-by-Step)

### Prerequisites — Install These First

| Tool | Version | Download |
|------|---------|----------|
| Python | 3.10 or 3.11 | https://python.org/downloads |
| VS Code | Latest | https://code.visualstudio.com |

---

### Step 1 — Open the Project in VS Code

1. Unzip `M3-ID_Combined.zip` to a folder (e.g. `C:\Projects\m3id`)
2. Open **VS Code**
3. Click **File → Open Folder**
4. Select the `m3id` folder (the one containing `backend/` and `frontend/`)

VS Code will detect the `.vscode/` settings automatically.

---

### Step 2 — Install Recommended Extensions

VS Code will show a popup: **"Install recommended extensions?"** — click **Install**.

Or install manually via the Extensions panel (`Ctrl+Shift+X`):

| Extension | Publisher | Purpose |
|-----------|-----------|---------|
| **Python** | Microsoft | Python language support |
| **Pylance** | Microsoft | Type hints and IntelliSense |
| **Live Server** | Ritwick Dey | Serve the frontend locally |

---

### Step 3 — Set Up the Backend

Open the **VS Code integrated terminal** (`Ctrl+` `` ` ``) and run:

#### Windows:
```cmd
cd backend
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
cd ..
```

#### Mac / Linux:
```bash
cd backend
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
cd ..
```

> Alternatively, just double-click `setup_backend.bat` (Windows) or run `./setup_backend.sh` (Mac/Linux) to do all of this automatically.

---

### Step 4 — Start the Backend Server

In the VS Code terminal (with venv still active):

```bash
uvicorn backend.main:app --reload --port 8000
```

You should see:
```
M3-ID v1.0.0 — FINAL BUILD (Part 7)
Voice    → Librosa
Face     → OpenCV
NLP      → Statistical
Swagger  → http://localhost:8000/docs
INFO:     Uvicorn running on http://127.0.0.1:8000
```

Keep this terminal open — the backend must stay running.

---

### Step 5 — Start the Frontend (Live Server)

1. In the VS Code **Explorer** panel, find `frontend/index.html`
2. **Right-click** on `index.html`
3. Select **"Open with Live Server"**
4. Your browser opens at `http://127.0.0.1:5500`

**Important:** Use Live Server (port 5500), NOT just opening the file directly in a browser. The backend's CORS is pre-configured for port 5500.

---

### Step 6 — Log In and Test

Use the demo account:

| Field | Value |
|-------|-------|
| Email | `demo@m3id.ai` |
| Password | `Demo@1234` |

Or click **Sign Up** to create a new account.

---

## All URLs at a Glance

| Service | URL |
|---------|-----|
| Frontend (Landing) | http://127.0.0.1:5500 |
| Login | http://127.0.0.1:5500/pages/login.html |
| Dashboard | http://127.0.0.1:5500/pages/dashboard.html |
| Voice Scan | http://127.0.0.1:5500/pages/voice.html |
| Face Scan | http://127.0.0.1:5500/pages/face.html |
| NLP Scan | http://127.0.0.1:5500/pages/nlp.html |
| Fusion Scan | http://127.0.0.1:5500/pages/fusion.html |
| History | http://127.0.0.1:5500/pages/history.html |
| Alerts | http://127.0.0.1:5500/pages/alerts.html |
| Swagger API Docs | http://localhost:8000/docs |
| Health Check | http://localhost:8000/health |

---

## VS Code Debug Mode (Optional)

Instead of running uvicorn manually, use VS Code's debugger:

1. Press `F5` or click **Run → Start Debugging**
2. Select **"M3-ID Backend (uvicorn)"**
3. Set breakpoints anywhere in the Python code

---

## Configuration

Edit `backend/.env` to change settings:

```env
APP_VERSION=1.0.0
DEBUG=True
FRONTEND_URL=http://127.0.0.1:5500
SECRET_KEY=change-this-in-production
DATABASE_URL=sqlite:///./m3id.db
UPLOAD_DIR=uploads
MAX_UPLOAD_SIZE_MB=50
```

---

## Detection Module Tiers

The system works out-of-the-box without GPU or ML libraries. Install extras for higher accuracy:

### Voice Detection (Part 3)
| Tier | Install | Mode |
|------|---------|------|
| Best | `pip install torch torchaudio` + model weights | CNN-LSTM (ASVspoof 2019) |
| Good | `pip install librosa` (already in requirements) | MFCC heuristic |
| Basic | Nothing extra needed | Raw WAV heuristic |

### Face Detection (Part 4)
| Tier | Install | Mode |
|------|---------|------|
| Best | `pip install torch torchvision timm` + weights | EfficientNet-B4 |
| Good | `pip install opencv-python` (already in requirements) | OpenCV heuristic |
| Basic | Nothing extra needed | JPEG DCT heuristic |

### NLP Detection (Part 5)
| Tier | Install | Mode |
|------|---------|------|
| Best | `pip install transformers torch` + BERT weights | Fine-tuned BERT |
| Basic | Nothing extra needed | 10-feature statistical (always active) |

### Fusion Engine (Part 6)
Always active — pure Python, no dependencies.

---

## API Reference

| Group | Endpoints |
|-------|-----------|
| Auth | `POST /api/auth/register` · `login` · `refresh` · `GET /api/auth/me` |
| Users | `GET/PUT /api/users/profile` · `/stats` |
| Scans | `POST/GET/DELETE /api/scans/` |
| Voice | `GET /api/voice/status` · `POST /api/voice/analyze` |
| Face | `GET /api/face/status` · `POST /api/face/analyze` |
| NLP | `GET /api/nlp/status` · `POST /api/nlp/analyze` · `analyze-batch` |
| Fusion | `GET /api/fusion/status` · `POST /api/fusion/analyze` · `fuse-scores` |
| Alerts | `GET/PATCH/DELETE /api/alerts/` |
| Export | `GET /api/export/scans.csv` · `scans.json` · `report/{id}.txt` |
| Analytics | `GET /api/analytics/overview` · `weekly-activity` · `verdict-distribution` |
| WebSocket | `ws://localhost:8000/ws/scan/<token>` |
| WebSocket | `ws://localhost:8000/ws/alerts/<token>` |

Full interactive docs at http://localhost:8000/docs

---

## Docker (Alternative — One Command)

```bash
docker-compose up --build
# Backend:  http://localhost:8000/docs
# Frontend: http://localhost:80
```

---

## Troubleshooting

| Problem | Fix |
|---------|-----|
| `ModuleNotFoundError` | Run `pip install -r requirements.txt` inside the venv |
| `uvicorn not found` | Activate venv first: `venv\Scripts\activate` (Win) or `source venv/bin/activate` (Mac) |
| CORS error in browser | Make sure Live Server is on port **5500**, not another port |
| `Address already in use` | Run `uvicorn backend.main:app --port 8001` and update `FRONTEND_URL` in `.env` |
| Database errors | Delete `backend/m3id.db` and restart — it recreates automatically |
| Face scan error | Upload an image file (jpg, png, webp) |
| Voice scan error | Upload an audio file (wav, mp3, flac) |

---

## Dataset and Research Links

- FaceForensics++ — https://github.com/ondyari/FaceForensics
- ASVspoof 2019 — https://datashare.ed.ac.uk/handle/10283/3336
- HC3 (NLP) — https://huggingface.co/datasets/Hello-SimpleAI/HC3
- FaceForensics++ paper (ICCV 2019)
- ASVspoof 2019 challenge paper
- BERT: Pre-training of Deep Bidirectional Transformers (2018)
