from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from ..core.database import get_db
from ..core.security import get_current_active_user
from ..schemas.user import UserPublic, UserUpdateRequest, UserStatsResponse
from ..services.scan_service import get_user_stats

router = APIRouter(prefix="/api/users", tags=["Users"])

@router.get("/profile", response_model=UserPublic)
def get_profile(current_user=Depends(get_current_active_user)):
    return current_user

@router.put("/profile", response_model=UserPublic)
def update_profile(data: UserUpdateRequest, db: Session = Depends(get_db), current_user=Depends(get_current_active_user)):
    if data.first_name: current_user.first_name = data.first_name.strip()
    if data.last_name:  current_user.last_name  = data.last_name.strip()
    if data.role:       current_user.role = data.role
    current_user.avatar_initials = (current_user.first_name[0] + current_user.last_name[0]).upper()
    db.commit(); db.refresh(current_user)
    return current_user

@router.get("/stats", response_model=UserStatsResponse)
def get_stats(db: Session = Depends(get_db), current_user=Depends(get_current_active_user)):
    return UserStatsResponse(**get_user_stats(db, current_user.id))

@router.delete("/account", response_model=dict)
def delete_account(db: Session = Depends(get_db), current_user=Depends(get_current_active_user)):
    current_user.is_active = False; db.commit()
    return {"message": "Account deactivated."}
