"""
M3-ID Scan Service — Part 6
All 4 modules active: Voice + Face + NLP + Risk Fusion Engine
"""
import time, json, math
from datetime import datetime, timezone
from typing import Optional
from sqlalchemy.orm import Session
from fastapi import HTTPException, UploadFile
from ..models.scan import Scan
from ..models.user import User
from ..core.config import settings
from ..utils.file_handler import save_upload_file
from ..modules.voice.detector  import voice_detector
from ..modules.face.detector   import face_detector
from ..modules.nlp.detector    import nlp_detector
from ..modules.fusion.engine   import fusion_engine

def _maybe_create_alert(db, scan, user):
    """Create an alert record when a scan finds a fake or suspicious result."""
    try:
        from ..models.alert import Alert
        severity_map = {"fake": "CRITICAL", "suspicious": "HIGH"}
        severity = severity_map.get(scan.verdict)
        if not severity:
            return
        score_pct = round(scan.fusion_score or 0)
        title = ("🚨 Deepfake Detected" if scan.verdict == "fake"
                 else "⚠️ Suspicious Content Flagged")
        message = (f"M3-ID flagged '{scan.file_name or scan.scan_type + ' input'}' "
                   f"as {scan.verdict.upper()} with a fusion score of {score_pct}%. "
                   f"Confidence: {round(scan.confidence or 0)}%.")
        alert = Alert(user_id=user.id, scan_id=scan.id,
                      severity=severity, title=title, message=message)
        db.add(alert)
        db.commit()
    except Exception:
        pass  # Never let alert creation crash the scan




def _detect_voice(file_path, scan_type):
    is_audio = scan_type in ("audio","voice") or (
        file_path and file_path.rsplit(".",1)[-1].lower() in settings.allowed_audio_list)
    if not file_path or not is_audio:
        return {"score": None, "skipped": True}
    try:
        r = voice_detector.analyze_file(file_path)
        d = r.to_dict(); d["skipped"] = False; return d
    except Exception as e:
        return {"score": 50.0, "skipped": False, "error": str(e)}


def _detect_face(file_path, scan_type):
    is_visual = scan_type in ("video","image","live") or (
        file_path and file_path.rsplit(".",1)[-1].lower()
        in settings.allowed_video_list + settings.allowed_image_list)
    if not file_path or not is_visual:
        return {"score": None, "skipped": True}
    try:
        r = face_detector.analyze_file(file_path)
        d = r.to_dict(); d["skipped"] = False; return d
    except Exception as e:
        return {"score": 50.0, "skipped": False, "error": str(e)}


def _detect_nlp(text):
    if not text or len(text.strip()) < 10:
        return {"score": None, "skipped": True}
    try:
        r = nlp_detector.analyze_text(text)
        d = r.to_dict(); d["skipped"] = False; return d
    except Exception as e:
        return {"score": 50.0, "skipped": False, "error": str(e)}


async def create_scan(db, user, scan_type, face_enabled=True, voice_enabled=True,
                      nlp_enabled=True, fusion_enabled=True, input_text=None, file=None):
    t0 = time.time()
    file_path = file_name = file_size = None
    if file and file.filename:
        file_path, file_name, file_size = await save_upload_file(file, user.id)

    scan = Scan(user_id=user.id, file_name=file_name, file_path=file_path,
                file_size_bytes=file_size, scan_type=scan_type, input_text=input_text,
                face_enabled=face_enabled, voice_enabled=voice_enabled,
                nlp_enabled=nlp_enabled, fusion_enabled=fusion_enabled,
                status="processing", verdict="pending")
    db.add(scan); db.commit(); db.refresh(scan)

    try:
        face_r  = _detect_face(file_path, scan_type)  if face_enabled  else {"score": None, "skipped": True}
        voice_r = _detect_voice(file_path, scan_type) if voice_enabled else {"score": None, "skipped": True}
        nlp_r   = _detect_nlp(input_text)             if nlp_enabled   else {"score": None, "skipped": True}

        # Use the real Fusion Engine
        fusion_obj = fusion_engine.fuse(
            face_score=face_r.get("score"),
            voice_score=voice_r.get("score"),
            nlp_score=nlp_r.get("score"),
            face_details=face_r,
            voice_details=voice_r,
            nlp_details=nlp_r,
        )
        fusion_r = fusion_obj.to_dict()

        scan.face_score   = face_r.get("score")
        scan.voice_score  = voice_r.get("score")
        scan.nlp_score    = nlp_r.get("score")
        scan.fusion_score = fusion_obj.fusion_score
        scan.verdict      = fusion_obj.verdict
        scan.confidence   = fusion_obj.confidence
        scan.processing_time_ms = int((time.time()-t0)*1000)
        scan.face_details   = json.dumps(face_r)
        scan.voice_details  = json.dumps(voice_r)
        scan.nlp_details    = json.dumps(nlp_r)
        scan.fusion_details = json.dumps(fusion_r)
        scan.status         = "done"
        scan.completed_at   = datetime.now(timezone.utc)
        user.total_scans    = (user.total_scans or 0) + 1
        if scan.verdict == "fake":
            user.fakes_caught = (user.fakes_caught or 0) + 1
        db.commit(); db.refresh(scan)
        # Auto-create alert for fake / suspicious results
        _maybe_create_alert(db, scan, user)
    except Exception as e:
        scan.status = "error"; scan.error_message = str(e)
        scan.verdict = "pending"; db.commit(); db.refresh(scan)
    return scan


def get_scan_by_id(db, scan_id, user_id):
    scan = db.query(Scan).filter(Scan.id == scan_id, Scan.user_id == user_id).first()
    if not scan: raise HTTPException(404, "Scan not found.")
    return scan

def get_user_scans(db, user_id, page=1, page_size=20, scan_type=None, verdict=None):
    q = db.query(Scan).filter(Scan.user_id == user_id)
    if scan_type: q = q.filter(Scan.scan_type == scan_type)
    if verdict:   q = q.filter(Scan.verdict == verdict)
    total = q.count()
    scans = q.order_by(Scan.created_at.desc()).offset((page-1)*page_size).limit(page_size).all()
    return {"scans": scans, "total": total, "page": page, "page_size": page_size,
            "total_pages": math.ceil(total/page_size) if total else 1}

def delete_scan(db, scan_id, user_id):
    scan = get_scan_by_id(db, scan_id, user_id)
    db.delete(scan); db.commit()
    return {"message": f"Scan {scan_id} deleted."}

def get_user_stats(db, user_id):
    total = db.query(Scan).filter(Scan.user_id == user_id).count()
    fakes = db.query(Scan).filter(Scan.user_id == user_id, Scan.verdict == "fake").count()
    susp  = db.query(Scan).filter(Scan.user_id == user_id, Scan.verdict == "suspicious").count()
    auth  = db.query(Scan).filter(Scan.user_id == user_id, Scan.verdict == "authentic").count()
    return {"total_scans": total, "fakes_caught": fakes, "suspicious_count": susp,
            "authentic_count": auth, "accuracy_rate": round(auth/total*100 if total else 0, 1)}
