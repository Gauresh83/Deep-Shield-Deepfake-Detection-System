"""
Export API — scan history CSV/JSON download.
"""
import csv
import json
import io
from fastapi import APIRouter, Depends, Query
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session
from ..core.database import get_db
from ..core.security import get_current_active_user
from ..models.scan import Scan

router = APIRouter(prefix="/api/export", tags=["Export"])


@router.get("/scans.csv")
def export_scans_csv(
    db: Session = Depends(get_db),
    current_user=Depends(get_current_active_user),
):
    """Download all scans as CSV."""
    scans = db.query(Scan).filter(Scan.user_id==current_user.id).order_by(Scan.created_at.desc()).all()
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(["id","file_name","scan_type","face_score","voice_score",
                     "nlp_score","fusion_score","verdict","confidence",
                     "processing_time_ms","status","created_at"])
    for s in scans:
        writer.writerow([s.id, s.file_name or "", s.scan_type,
                         s.face_score or "", s.voice_score or "",
                         s.nlp_score or "", s.fusion_score or "",
                         s.verdict, s.confidence or "", s.processing_time_ms or "",
                         s.status, str(s.created_at)])
    output.seek(0)
    return StreamingResponse(
        iter([output.getvalue()]),
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=m3id_scans.csv"},
    )


@router.get("/scans.json")
def export_scans_json(
    db: Session = Depends(get_db),
    current_user=Depends(get_current_active_user),
):
    """Download all scans as JSON."""
    scans = db.query(Scan).filter(Scan.user_id==current_user.id).order_by(Scan.created_at.desc()).all()
    data = [{"id":s.id,"file_name":s.file_name,"scan_type":s.scan_type,
             "face_score":s.face_score,"voice_score":s.voice_score,
             "nlp_score":s.nlp_score,"fusion_score":s.fusion_score,
             "verdict":s.verdict,"confidence":s.confidence,
             "status":s.status,"created_at":str(s.created_at)} for s in scans]
    output = io.StringIO(json.dumps({"scans": data, "total": len(data)}, indent=2))
    return StreamingResponse(
        iter([output.getvalue()]),
        media_type="application/json",
        headers={"Content-Disposition": "attachment; filename=m3id_scans.json"},
    )


@router.get("/report/{scan_id}.txt")
def export_single_scan_report(
    scan_id: int,
    db: Session = Depends(get_db),
    current_user=Depends(get_current_active_user),
):
    """Download full text report for a single scan."""
    s = db.query(Scan).filter(Scan.id==scan_id, Scan.user_id==current_user.id).first()
    if not s:
        from fastapi import HTTPException
        raise HTTPException(404, "Scan not found.")
    report = f"""M3-ID SCAN REPORT
==================
Scan ID      : {s.id}
File         : {s.file_name or "N/A"}
Type         : {s.scan_type}
Date         : {s.created_at}
Status       : {s.status}

SCORES
------
Face Score   : {s.face_score or "N/A"}%
Voice Score  : {s.voice_score or "N/A"}%
NLP Score    : {s.nlp_score or "N/A"}%
Fusion Score : {s.fusion_score or "N/A"}%

VERDICT      : {s.verdict.upper() if s.verdict else "N/A"}
Confidence   : {s.confidence or "N/A"}%
Process Time : {s.processing_time_ms or "N/A"}ms

M3-ID Multi-Modal Identity Defender
"""
    return StreamingResponse(
        iter([report]),
        media_type="text/plain",
        headers={"Content-Disposition": f"attachment; filename=m3id_scan_{scan_id}.txt"},
    )
