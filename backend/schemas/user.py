from pydantic import BaseModel, EmailStr
from typing import Optional
from datetime import datetime

class UserPublic(BaseModel):
    id: int
    email: EmailStr
    first_name: str
    last_name: str
    role: Optional[str] = "User"
    avatar_initials: str
    is_active: bool
    is_verified: bool
    total_scans: int
    fakes_caught: int
    created_at: datetime
    last_login: Optional[datetime] = None
    model_config = {"from_attributes": True}

class UserUpdateRequest(BaseModel):
    first_name: Optional[str] = None
    last_name: Optional[str] = None
    role: Optional[str] = None

class UserStatsResponse(BaseModel):
    total_scans: int
    fakes_caught: int
    suspicious_count: int
    authentic_count: int
    accuracy_rate: float
