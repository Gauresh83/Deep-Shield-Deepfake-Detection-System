"""
Risk Fusion API — Part 6
Runs all detection modules in parallel and returns a unified verdict.
"""
import json
import asyncio
from typing import Optional
from fastapi import APIRouter, Depends, UploadFile, File, Form, Body, HTTPException
from sqlalchemy.orm import Session
import os, tempfile

from ..core.database import get_db
from ..core.security import get_current_active_user
from ..modules.voice.detector  import voice_detector
from ..modules.face.detector   import face_detector
from ..modules.nlp.detector    import nlp_detector
from ..modules.fusion.engine   import fusion_engine

router = APIRouter(prefix="/api/fusion", tags=["Risk Fusion"])


@router.get("/status", response_model=dict)
def fusion_status(current_user=Depends(get_current_active_user)):
    return {
        "module": "Risk Fusion Engine",
        "version": "1.0.0 (Part 6)",
        "algorithm": "Weighted multi-modal aggregation with anomaly boosting",
        "weights": {"face": 0.35, "voice": 0.35, "nlp": 0.30},
        "severity_levels": ["CRITICAL (0-30)", "HIGH (30-45)", "MEDIUM (45-65)", "LOW (65-80)", "CLEAR (80-100)"],
        "xai_enabled": True,
        "confidence_intervals": True,
        "sub_modules": {
            "voice": "active" if voice_detector.librosa_available or voice_detector.model_loaded else "heuristic",
            "face":  "active" if face_detector.cv2_available or face_detector.model_loaded else "heuristic",
            "nlp":   "active" if nlp_detector.model_loaded else "statistical",
        }
    }


@router.post("/analyze", response_model=dict)
async def full_fusion_analyze(
    # Optional media file
    file: Optional[UploadFile] = File(None),
    # Text for NLP
    text: Optional[str] = Form(None),
    # Module toggles
    use_face:  bool = Form(True),
    use_voice: bool = Form(True),
    use_nlp:   bool = Form(True),
    db: Session = Depends(get_db),
    current_user=Depends(get_current_active_user),
):
    """
    Run ALL enabled detection modules and return fused verdict.

    - Upload a video/audio file for face + voice detection
    - Provide text for NLP analysis
    - All modules run, results fused into single M3-ID verdict
    """
    tmp_path = None
    file_content = None
    original_ext = None

    if file and file.filename:
        file_content = await file.read()
        if len(file_content) > 100 * 1024 * 1024:
            raise HTTPException(413, "File too large. Max 100 MB.")
        original_ext = file.filename.rsplit(".", 1)[-1].lower() if "." in file.filename else "mp4"
        with tempfile.NamedTemporaryFile(delete=False, suffix=f".{original_ext}") as tmp:
            tmp.write(file_content)
            tmp_path = tmp.name

    face_result  = None
    voice_result = None
    nlp_result   = None
    face_dict    = {}
    voice_dict   = {}
    nlp_dict     = {}

    try:
        # Run modules (synchronously for simplicity; asyncio.gather in prod)
        if use_face and tmp_path and original_ext in {"mp4","avi","mov","mkv","webm","jpg","jpeg","png","webp"}:
            try:
                r = face_detector.analyze_file(tmp_path)
                face_result = r.score
                face_dict   = r.to_dict()
            except Exception as e:
                face_dict = {"error": str(e)}

        if use_voice and tmp_path and original_ext in {"wav","mp3","flac","m4a","ogg","aac"}:
            try:
                r = voice_detector.analyze_file(tmp_path)
                voice_result = r.score
                voice_dict   = r.to_dict()
            except Exception as e:
                voice_dict = {"error": str(e)}

        if use_nlp and text and len(text.strip()) >= 10:
            try:
                r = nlp_detector.analyze_text(text)
                nlp_result = r.score
                nlp_dict   = r.to_dict()
            except Exception as e:
                nlp_dict = {"error": str(e)}

        # Fuse
        fusion = fusion_engine.fuse(
            face_score=face_result,
            voice_score=voice_result,
            nlp_score=nlp_result,
            face_details=face_dict,
            voice_details=voice_dict,
            nlp_details=nlp_dict,
        )

    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)

    return {
        "fusion": fusion.to_dict(),
        "module_results": {
            "face":  face_dict  if face_dict  else None,
            "voice": voice_dict if voice_dict else None,
            "nlp":   nlp_dict   if nlp_dict   else None,
        },
        "summary": {
            "fusion_score":   fusion.fusion_score,
            "verdict":        fusion.verdict,
            "alert_severity": fusion.alert_severity,
            "confidence":     fusion.confidence,
            "ci_low":         fusion.confidence_interval_low,
            "ci_high":        fusion.confidence_interval_high,
            "recommendation": fusion.recommendation,
            "modules_active": fusion.modules_active,
        }
    }


@router.post("/fuse-scores", response_model=dict)
def fuse_existing_scores(
    face_score:  Optional[float] = Body(None),
    voice_score: Optional[float] = Body(None),
    nlp_score:   Optional[float] = Body(None),
    current_user=Depends(get_current_active_user),
):
    """
    Fuse already-computed module scores (e.g. from previous /api/face/analyze calls).
    Useful for integrating pre-computed results without re-running modules.
    """
    if all(s is None for s in [face_score, voice_score, nlp_score]):
        raise HTTPException(400, "Provide at least one module score.")

    fusion = fusion_engine.fuse(
        face_score=face_score,
        voice_score=voice_score,
        nlp_score=nlp_score,
    )
    return fusion.to_dict()
