"""
Analytics API — performance metrics and trend data for the dashboard.
"""
from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from sqlalchemy import func, extract
from ..core.database import get_db
from ..core.security import get_current_active_user
from ..models.scan import Scan
from datetime import datetime, timezone, timedelta

router = APIRouter(prefix="/api/analytics", tags=["Analytics"])


@router.get("/overview")
def get_overview(
    db: Session = Depends(get_db),
    current_user=Depends(get_current_active_user),
):
    """Summary stats for the dashboard overview cards."""
    uid = current_user.id
    total  = db.query(Scan).filter(Scan.user_id==uid).count()
    fakes  = db.query(Scan).filter(Scan.user_id==uid, Scan.verdict=="fake").count()
    susp   = db.query(Scan).filter(Scan.user_id==uid, Scan.verdict=="suspicious").count()
    auth   = db.query(Scan).filter(Scan.user_id==uid, Scan.verdict=="authentic").count()

    # This week vs last week
    now   = datetime.now(timezone.utc)
    week  = now - timedelta(days=7)
    week2 = now - timedelta(days=14)
    this_week = db.query(Scan).filter(Scan.user_id==uid, Scan.created_at>=week).count()
    last_week = db.query(Scan).filter(Scan.user_id==uid, Scan.created_at>=week2, Scan.created_at<week).count()
    pct_change = round((this_week-last_week)/max(1,last_week)*100, 1) if last_week else 0

    return {
        "total_scans": total,
        "fakes_caught": fakes,
        "suspicious": susp,
        "authentic": auth,
        "accuracy_rate": round(auth/total*100, 1) if total else 0,
        "this_week": this_week,
        "last_week": last_week,
        "pct_change": pct_change,
    }


@router.get("/weekly-activity")
def weekly_activity(
    db: Session = Depends(get_db),
    current_user=Depends(get_current_active_user),
):
    """Last 7 days scan + fakes counts — for Chart.js bar/line chart."""
    uid = current_user.id
    now = datetime.now(timezone.utc)
    days = []
    for i in range(6, -1, -1):
        day_start = (now - timedelta(days=i)).replace(hour=0, minute=0, second=0, microsecond=0)
        day_end   = day_start + timedelta(days=1)
        scans = db.query(Scan).filter(Scan.user_id==uid,
            Scan.created_at>=day_start, Scan.created_at<day_end).count()
        fakes = db.query(Scan).filter(Scan.user_id==uid,
            Scan.created_at>=day_start, Scan.created_at<day_end,
            Scan.verdict=="fake").count()
        days.append({
            "date":  day_start.strftime("%a"),
            "scans": scans,
            "fakes": fakes,
        })
    return {"days": days}


@router.get("/module-accuracy")
def module_accuracy(
    db: Session = Depends(get_db),
    current_user=Depends(get_current_active_user),
):
    """Average score per module across all completed scans."""
    uid = current_user.id
    q = db.query(
        func.avg(Scan.face_score),
        func.avg(Scan.voice_score),
        func.avg(Scan.nlp_score),
        func.avg(Scan.fusion_score),
    ).filter(Scan.user_id==uid, Scan.status=="done").first()
    return {
        "face_avg":   round(q[0] or 0, 1),
        "voice_avg":  round(q[1] or 0, 1),
        "nlp_avg":    round(q[2] or 0, 1),
        "fusion_avg": round(q[3] or 0, 1),
    }


@router.get("/verdict-distribution")
def verdict_distribution(
    db: Session = Depends(get_db),
    current_user=Depends(get_current_active_user),
):
    """Count by verdict — for pie/doughnut chart."""
    uid = current_user.id
    rows = db.query(Scan.verdict, func.count(Scan.id)).filter(
        Scan.user_id==uid, Scan.status=="done"
    ).group_by(Scan.verdict).all()
    return {r[0]: r[1] for r in rows}
