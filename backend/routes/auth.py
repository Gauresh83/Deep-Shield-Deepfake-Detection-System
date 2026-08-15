from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from ..core.database import get_db
from ..core.security import get_current_active_user
from ..schemas.auth import RegisterRequest, LoginRequest, TokenResponse, RefreshRequest, PasswordChangeRequest
from ..schemas.user import UserPublic
from ..services.auth_service import register_user, login_user, refresh_access_token, change_password

router = APIRouter(prefix="/api/auth", tags=["Authentication"])

@router.post("/register", response_model=dict, status_code=201)
def register(data: RegisterRequest, db: Session = Depends(get_db)):
    user = register_user(db, data)
    return {"message": f"Account created. Welcome, {user.first_name}!", "user_id": user.id, "email": user.email}

@router.post("/login", response_model=TokenResponse)
def login(data: LoginRequest, db: Session = Depends(get_db)):
    return login_user(db, data)

@router.post("/refresh", response_model=TokenResponse)
def refresh(data: RefreshRequest, db: Session = Depends(get_db)):
    return refresh_access_token(data.refresh_token, db)

@router.get("/me", response_model=UserPublic)
def get_me(current_user=Depends(get_current_active_user)):
    return current_user

@router.post("/change-password", response_model=dict)
def change_pwd(data: PasswordChangeRequest, db: Session = Depends(get_db), current_user=Depends(get_current_active_user)):
    return change_password(current_user, data.current_password, data.new_password, db)

@router.post("/logout", response_model=dict)
def logout(current_user=Depends(get_current_active_user)):
    return {"message": "Logged out. Please discard your token."}
