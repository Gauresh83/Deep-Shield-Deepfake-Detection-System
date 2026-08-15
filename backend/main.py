"""
M3-ID Backend — Part 7: Final Integration
==========================================
All modules active + WebSocket + Alerts + Export + Analytics

Run:  uvicorn backend.main:app --reload --port 8000
Docs: http://localhost:8000/docs
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager
import os

from backend.core.config import settings
from backend.core.database import init_db
from backend.routes import (
    auth_router, users_router, scans_router,
    voice_router, face_router, nlp_router,
    fusion_router, alerts_router, export_router,
    analytics_router, ws_router,
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    print(f"\n🚀  M3-ID v{settings.APP_VERSION} — FINAL BUILD (Part 7)")
    init_db()
    os.makedirs(settings.UPLOAD_DIR, exist_ok=True)

    from backend.modules.voice.detector  import voice_detector
    from backend.modules.face.detector   import face_detector
    from backend.modules.nlp.detector    import nlp_detector

    print(f"🎙️  Voice    → {'CNN-LSTM' if voice_detector.model_loaded else 'Librosa' if voice_detector.librosa_available else 'Raw-WAV'}")
    print(f"👤  Face     → {'EfficientNet-B4' if face_detector.model_loaded else 'OpenCV' if face_detector.cv2_available else 'Python'}")
    print(f"📝  NLP      → {'BERT' if nlp_detector.model_loaded else 'Statistical'}")
    print(f"⚡  Fusion   → Weighted XAI + Alerts")
    print(f"🔌  WebSocket→ /ws/scan/<token>  |  /ws/alerts/<token>")
    print(f"📤  Export   → /api/export/scans.csv | scans.json")
    print(f"📊  Analytics→ /api/analytics/overview | weekly | verdict")
    print(f"✅  Database → {settings.DATABASE_URL}")
    print(f"📖  Swagger  → http://localhost:8000/docs\n")
    yield
    print("\n👋  M3-ID shutting down.")


app = FastAPI(
    title="M3-ID — Multi-Modal Identity Defender",
    description=(
        "**FINAL BUILD — Part 7 — All Systems Active**\n\n"
        "**Detection Modules:**\n"
        "- 🎙️ Voice Clone Detection  — MFCC + CNN-LSTM\n"
        "- 👤 Face Deepfake Detection — EfficientNet-B4\n"
        "- 📝 Linguistic NLP          — BERT + Statistical\n"
        "- ⚡ Risk Fusion Engine       — XAI + Confidence Intervals\n\n"
        "**Part 7 Additions:**\n"
        "- 🔌 WebSocket real-time scan stream\n"
        "- 🚨 Alert system (CRITICAL/HIGH/MEDIUM/LOW/CLEAR)\n"
        "- 📤 CSV + JSON export\n"
        "- 📊 Analytics API (weekly trends, verdict distribution)\n"
    ),
    version=settings.APP_VERSION,
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[settings.FRONTEND_URL, "http://localhost:5500",
                   "http://127.0.0.1:5500", "http://localhost:3000", "null"],
    allow_credentials=True, allow_methods=["*"], allow_headers=["*"],
)

if os.path.exists(settings.UPLOAD_DIR):
    app.mount("/uploads", StaticFiles(directory=settings.UPLOAD_DIR), name="uploads")

# API routers
app.include_router(auth_router)
app.include_router(users_router)
app.include_router(scans_router)
app.include_router(voice_router)
app.include_router(face_router)
app.include_router(nlp_router)
app.include_router(fusion_router)
app.include_router(alerts_router)
app.include_router(export_router)
app.include_router(analytics_router)
app.include_router(ws_router)


@app.get("/", tags=["Health"])
def root():
    from backend.modules.voice.detector import voice_detector
    from backend.modules.face.detector  import face_detector
    from backend.modules.nlp.detector   import nlp_detector
    return {
        "name": "M3-ID", "version": settings.APP_VERSION,
        "status": "online", "build": "Part 7 — Final",
        "docs": "http://localhost:8000/docs",
        "api_groups": [
            "/api/auth   — register, login, refresh, me",
            "/api/users  — profile, stats, update",
            "/api/scans  — run, list, get, delete",
            "/api/voice  — analyze audio",
            "/api/face   — analyze image/video",
            "/api/nlp    — analyze text",
            "/api/fusion — full multi-modal scan",
            "/api/alerts — notifications",
            "/api/export — CSV/JSON download",
            "/api/analytics — trends & metrics",
            "/ws/scan/<token>   — real-time scan progress",
            "/ws/alerts/<token> — live alert stream",
        ],
        "modules": {
            "voice":  "CNN-LSTM" if voice_detector.model_loaded else "Librosa" if voice_detector.librosa_available else "raw-wav",
            "face":   "EfficientNet-B4" if face_detector.model_loaded else "OpenCV" if face_detector.cv2_available else "python",
            "nlp":    "BERT" if nlp_detector.model_loaded else "statistical",
            "fusion": "weighted-xai",
        },
    }


@app.get("/health", tags=["Health"])
def health():
    return JSONResponse({"status": "healthy", "version": settings.APP_VERSION,
                         "modules": 4, "build": "Part7-Final"})
