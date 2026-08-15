"""
Alerts API — real-time notification management.
"""
import math
from typing import Optional
from fastapi import APIRouter, Depends, Query, HTTPException
from sqlalchemy.orm import Session
from ..core.database import get_db
from ..core.security import get_current_active_user
from ..models.alert import Alert

router = APIRouter(prefix="/api/alerts", tags=["Alerts"])


@router.get("/", response_model=dict)
def list_alerts(
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
    unread_only: bool = Query(False),
    severity: Optional[str] = Query(None),
    db: Session = Depends(get_db),
    current_user=Depends(get_current_active_user),
):
    q = db.query(Alert).filter(Alert.user_id == current_user.id)
    if unread_only: q = q.filter(Alert.is_read == False)
    if severity:    q = q.filter(Alert.severity == severity.upper())
    total = q.count()
    alerts = q.order_by(Alert.created_at.desc()).offset((page-1)*page_size).limit(page_size).all()
    return {
        "alerts": [{"id":a.id,"severity":a.severity,"title":a.title,
                    "message":a.message,"is_read":a.is_read,
                    "scan_id":a.scan_id,"created_at":str(a.created_at)} for a in alerts],
        "total": total, "unread": q.filter(Alert.is_read==False).count(),
        "page": page, "total_pages": math.ceil(total/page_size) if total else 1,
    }


@router.get("/unread-count")
def unread_count(db: Session = Depends(get_db), current_user=Depends(get_current_active_user)):
    count = db.query(Alert).filter(Alert.user_id==current_user.id, Alert.is_read==False).count()
    return {"unread_count": count}


@router.patch("/{alert_id}/read")
def mark_read(alert_id: int, db: Session = Depends(get_db), current_user=Depends(get_current_active_user)):
    a = db.query(Alert).filter(Alert.id==alert_id, Alert.user_id==current_user.id).first()
    if not a: raise HTTPException(404, "Alert not found.")
    a.is_read = True; db.commit()
    return {"message": "Alert marked read."}


@router.patch("/mark-all-read")
def mark_all_read(db: Session = Depends(get_db), current_user=Depends(get_current_active_user)):
    db.query(Alert).filter(Alert.user_id==current_user.id, Alert.is_read==False).update({"is_read": True})
    db.commit()
    return {"message": "All alerts marked read."}


@router.delete("/{alert_id}")
def delete_alert(alert_id: int, db: Session = Depends(get_db), current_user=Depends(get_current_active_user)):
    a = db.query(Alert).filter(Alert.id==alert_id, Alert.user_id==current_user.id).first()
    if not a: raise HTTPException(404, "Alert not found.")
    db.delete(a); db.commit()
    return {"message": "Alert deleted."}
