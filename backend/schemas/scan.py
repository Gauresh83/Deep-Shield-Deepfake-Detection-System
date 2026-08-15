from pydantic import BaseModel
from typing import Optional, Dict, Any, List
from datetime import datetime

class ScanResponse(BaseModel):
    id: int
    user_id: int
    file_name: Optional[str] = None
    file_size_bytes: Optional[int] = None
    scan_type: str
    input_text: Optional[str] = None
    face_score: Optional[float] = None
    voice_score: Optional[float] = None
    nlp_score: Optional[float] = None
    fusion_score: Optional[float] = None
    face_enabled: bool
    voice_enabled: bool
    nlp_enabled: bool
    fusion_enabled: bool
    verdict: str
    confidence: Optional[float] = None
    processing_time_ms: Optional[int] = None
    status: str
    face_details: Optional[str] = None
    voice_details: Optional[str] = None
    nlp_details: Optional[str] = None
    fusion_details: Optional[str] = None
    error_message: Optional[str] = None
    created_at: datetime
    completed_at: Optional[datetime] = None
    model_config = {"from_attributes": True}

class ScanListResponse(BaseModel):
    scans: List[ScanResponse]
    total: int
    page: int
    page_size: int
    total_pages: int
