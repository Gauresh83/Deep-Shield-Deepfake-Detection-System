"""
Voice Detection API — Part 3
Dedicated endpoints for voice clone analysis.
"""
from typing import Optional
from fastapi import APIRouter, Depends, UploadFile, File, HTTPException
from sqlalchemy.orm import Session
import os, tempfile

from ..core.database import get_db
from ..core.security import get_current_active_user
from ..modules.voice.detector import voice_detector

router = APIRouter(prefix="/api/voice", tags=["Voice Detection"])


@router.get("/status", response_model=dict)
def voice_module_status(current_user=Depends(get_current_active_user)):
    """Check the voice detection module status and capabilities."""
    return {
        "module": "Voice Clone Detection",
        "version": "1.0.0 (Part 3)",
        "model_loaded": voice_detector.model_loaded,
        "librosa_available": voice_detector.librosa_available,
        "torch_available": voice_detector.torch_available,
        "mode": (
            "CNN-LSTM (trained weights)" if voice_detector.model_loaded
            else "Librosa heuristic" if voice_detector.librosa_available
            else "Raw WAV heuristic"
        ),
        "supported_formats": ["wav", "mp3", "flac", "m4a", "ogg"],
        "model_architecture": "CNN-LSTM on MFCC features (ASVspoof 2019)",
        "weights_path": voice_detector.WEIGHTS_PATH,
        "weights_exist": os.path.exists(voice_detector.WEIGHTS_PATH),
        "instructions": {
            "install_librosa": "pip install librosa soundfile",
            "install_torch": "pip install torch torchaudio",
            "model_weights": "Place voice_model.pt in backend/modules/voice/weights/",
        }
    }


@router.post("/analyze", response_model=dict)
async def analyze_voice(
    file: UploadFile = File(...),
    current_user=Depends(get_current_active_user),
):
    """
    Analyze an audio file for voice cloning / synthesis artifacts.

    Returns detailed scores including:
    - Overall authenticity score (0-100)
    - MFCC anomaly score
    - Spectral consistency
    - Pitch naturalness
    - Temporal coherence
    - Clone probability
    """
    allowed_exts = {"wav", "mp3", "flac", "m4a", "ogg", "aac", "wma"}
    filename = file.filename or "audio"
    ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""

    if ext not in allowed_exts:
        raise HTTPException(400, f"Unsupported audio format: .{ext}. Allowed: {', '.join(allowed_exts)}")

    # Save to temp file
    content = await file.read()
    if len(content) > 50 * 1024 * 1024:
        raise HTTPException(413, "File too large. Max 50 MB.")

    with tempfile.NamedTemporaryFile(delete=False, suffix=f".{ext}") as tmp:
        tmp.write(content)
        tmp_path = tmp.name

    try:
        result = voice_detector.analyze_file(tmp_path)
    finally:
        os.unlink(tmp_path)

    return {
        "file_name": filename,
        "file_size_bytes": len(content),
        "analysis": result.to_dict(),
        "summary": {
            "score": result.score,
            "verdict": result.verdict,
            "confidence": result.confidence,
            "clone_probability": result.clone_probability,
            "processing_time_ms": result.processing_time_ms,
        }
    }


@router.post("/analyze-url", response_model=dict)
async def analyze_voice_url(
    url: str,
    current_user=Depends(get_current_active_user),
):
    """Analyze audio from a URL (fetches and runs detection)."""
    import httpx, tempfile
    try:
        async with httpx.AsyncClient(timeout=30) as client:
            response = await client.get(url)
            response.raise_for_status()
    except Exception as e:
        raise HTTPException(400, f"Could not fetch audio from URL: {e}")

    content = response.content
    ext = url.rsplit(".", 1)[-1].lower().split("?")[0] or "wav"

    with tempfile.NamedTemporaryFile(delete=False, suffix=f".{ext}") as tmp:
        tmp.write(content)
        tmp_path = tmp.name

    try:
        result = voice_detector.analyze_file(tmp_path)
    finally:
        os.unlink(tmp_path)

    return {
        "source_url": url,
        "file_size_bytes": len(content),
        "analysis": result.to_dict(),
        "summary": {
            "score": result.score,
            "verdict": result.verdict,
            "confidence": result.confidence,
            "clone_probability": result.clone_probability,
        }
    }
