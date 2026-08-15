from sqlalchemy import Column, Integer, String, Boolean, DateTime, ForeignKey, Text
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from ..core.database import Base

class Alert(Base):
    __tablename__ = "alerts"
    id         = Column(Integer, primary_key=True, index=True)
    user_id    = Column(Integer, ForeignKey("users.id"), nullable=False)
    scan_id    = Column(Integer, ForeignKey("scans.id"), nullable=True)
    severity   = Column(String(20), default="MEDIUM")    # CRITICAL HIGH MEDIUM LOW CLEAR
    title      = Column(String(255), nullable=False)
    message    = Column(Text, nullable=False)
    is_read    = Column(Boolean, default=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    user       = relationship("User", backref="alerts")
