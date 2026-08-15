from sqlalchemy import Column, Integer, String, Float, Boolean, DateTime, ForeignKey, Text
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from ..core.database import Base

class Scan(Base):
    __tablename__ = "scans"
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    file_name = Column(String(255), nullable=True)
    file_path = Column(String(500), nullable=True)
    file_size_bytes = Column(Integer, nullable=True)
    scan_type = Column(String(20), default="video")
    input_text = Column(Text, nullable=True)
    face_score = Column(Float, nullable=True)
    voice_score = Column(Float, nullable=True)
    nlp_score = Column(Float, nullable=True)
    fusion_score = Column(Float, nullable=True)
    face_enabled = Column(Boolean, default=True)
    voice_enabled = Column(Boolean, default=True)
    nlp_enabled = Column(Boolean, default=True)
    fusion_enabled = Column(Boolean, default=True)
    verdict = Column(String(20), default="pending")
    confidence = Column(Float, nullable=True)
    processing_time_ms = Column(Integer, nullable=True)
    face_details = Column(Text, nullable=True)
    voice_details = Column(Text, nullable=True)
    nlp_details = Column(Text, nullable=True)
    fusion_details = Column(Text, nullable=True)
    status = Column(String(20), default="pending")
    error_message = Column(Text, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    completed_at = Column(DateTime(timezone=True), nullable=True)
    owner = relationship("User", back_populates="scans")
