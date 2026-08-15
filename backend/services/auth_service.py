from datetime import datetime, timezone
from sqlalchemy.orm import Session
from fastapi import HTTPException
from ..models.user import User
from ..schemas.auth import RegisterRequest, LoginRequest, TokenResponse
from ..core.security import hash_password, verify_password, create_access_token, create_refresh_token
from ..core.config import settings

def register_user(db, data):
    if db.query(User).filter(User.email == data.email.lower()).first():
        raise HTTPException(400, "Email already registered.")
    user = User(
        email=data.email.lower(), hashed_password=hash_password(data.password),
        first_name=data.first_name.strip(), last_name=data.last_name.strip(),
        role=data.role or "User",
        avatar_initials=(data.first_name[0]+data.last_name[0]).upper(),
    )
    db.add(user); db.commit(); db.refresh(user)
    return user

def login_user(db, data):
    user = db.query(User).filter(User.email == data.email.lower()).first()
    if not user or not verify_password(data.password, user.hashed_password):
        raise HTTPException(401, "Invalid email or password.")
    if not user.is_active:
        raise HTTPException(400, "Account deactivated.")
    user.last_login = datetime.now(timezone.utc); db.commit()
    td = {"sub": str(user.id), "email": user.email}
    return TokenResponse(access_token=create_access_token(td),
                         refresh_token=create_refresh_token(td),
                         token_type="bearer",
                         expires_in=settings.ACCESS_TOKEN_EXPIRE_MINUTES*60)

def refresh_access_token(refresh_token, db):
    from ..core.security import decode_token
    payload = decode_token(refresh_token)
    if payload.get("type") != "refresh":
        raise HTTPException(401, "Invalid token type.")
    user = db.query(User).filter(User.id == int(payload["sub"])).first()
    if not user or not user.is_active:
        raise HTTPException(401, "User not found.")
    td = {"sub": str(user.id), "email": user.email}
    return TokenResponse(access_token=create_access_token(td),
                         refresh_token=create_refresh_token(td),
                         token_type="bearer",
                         expires_in=settings.ACCESS_TOKEN_EXPIRE_MINUTES*60)

def change_password(user, current_password, new_password, db):
    if not verify_password(current_password, user.hashed_password):
        raise HTTPException(400, "Current password incorrect.")
    user.hashed_password = hash_password(new_password); db.commit()
    return {"message": "Password updated."}
