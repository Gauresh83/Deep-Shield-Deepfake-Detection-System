from typing import Optional
from fastapi import APIRouter, Depends, UploadFile, File, Form, Query
from sqlalchemy.orm import Session
from ..core.database import get_db
from ..core.security import get_current_active_user
from ..schemas.scan import ScanResponse, ScanListResponse
from ..services.scan_service import create_scan, get_scan_by_id, get_user_scans, delete_scan

router = APIRouter(prefix="/api/scans", tags=["Scans"])

@router.post("/", response_model=ScanResponse, status_code=201)
async def run_scan(
    scan_type: str = Form("video"),
    face_enabled: bool = Form(True),
    voice_enabled: bool = Form(True),
    nlp_enabled: bool = Form(True),
    fusion_enabled: bool = Form(True),
    input_text: Optional[str] = Form(None),
    file: Optional[UploadFile] = File(None),
    db: Session = Depends(get_db),
    current_user=Depends(get_current_active_user),
):
    return await create_scan(db=db, user=current_user, scan_type=scan_type,
        face_enabled=face_enabled, voice_enabled=voice_enabled,
        nlp_enabled=nlp_enabled, fusion_enabled=fusion_enabled,
        input_text=input_text, file=file)

@router.get("/", response_model=ScanListResponse)
def list_scans(
    page: int = Query(1, ge=1), page_size: int = Query(20, ge=1, le=100),
    scan_type: Optional[str] = None, verdict: Optional[str] = None,
    db: Session = Depends(get_db), current_user=Depends(get_current_active_user),
):
    return ScanListResponse(**get_user_scans(db, current_user.id, page, page_size, scan_type, verdict))

@router.get("/{scan_id}", response_model=ScanResponse)
def get_scan(scan_id: int, db: Session = Depends(get_db), current_user=Depends(get_current_active_user)):
    return get_scan_by_id(db, scan_id, current_user.id)

@router.delete("/{scan_id}", response_model=dict)
def remove_scan(scan_id: int, db: Session = Depends(get_db), current_user=Depends(get_current_active_user)):
    return delete_scan(db, scan_id, current_user.id)
