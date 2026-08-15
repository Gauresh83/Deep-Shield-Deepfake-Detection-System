"""
Face Detection API — Part 4
Dedicated endpoints for face deepfake analysis.
"""
import os, tempfile
from fastapi import APIRouter, Depends, UploadFile, File, HTTPException
from ..core.security import get_current_active_user
from ..modules.face.detector import face_detector

router = APIRouter(prefix="/api/face", tags=["Face Detection"])


@router.get("/status", response_model=dict)
def face_module_status(current_user=Depends(get_current_active_user)):
    """Check face detection module capabilities."""
    return {
        "module": "Face Deepfake Detection",
        "version": "1.0.0 (Part 4)",
        "model_loaded": face_detector.model_loaded,
        "cv2_available": face_detector.cv2_available,
        "torch_available": face_detector.torch_available,
        "timm_available": face_detector.timm_available,
        "mode": (
            "EfficientNet-B4 (FaceForensics++ weights)" if face_detector.model_loaded
            else "OpenCV + Heuristic Analysis" if face_detector.cv2_available
            else "Pure-Python Heuristic"
        ),
        "supported_formats": {
            "images": ["jpg", "jpeg", "png", "webp", "bmp"],
            "videos": ["mp4", "avi", "mov", "mkv", "webm"],
        },
        "model_architecture": "EfficientNet-B4 binary classifier (real vs deepfake)",
        "dataset": "FaceForensics++ / DFDC",
        "weights_path": face_detector.WEIGHTS_PATH,
        "weights_exist": os.path.exists(face_detector.WEIGHTS_PATH),
        "instructions": {
            "install_opencv": "pip install opencv-python",
            "install_torch":  "pip install torch torchvision",
            "install_timm":   "pip install timm",
            "model_weights":  "Place face_model.pt in backend/modules/face/weights/",
            "dataset":        "https://github.com/ondyari/FaceForensics",
        }
    }


@router.post("/analyze", response_model=dict)
async def analyze_face(
    file: UploadFile = File(...),
    current_user=Depends(get_current_active_user),
):
    """
    Analyze an image or video for face deepfake artifacts.

    Returns:
    - Overall authenticity score (0-100)
    - Per-frame breakdown
    - Blending artifact score
    - Texture consistency
    - Temporal stability (videos)
    - Frequency anomaly (DCT-based)
    """
    allowed = {"jpg","jpeg","png","webp","bmp","mp4","avi","mov","mkv","webm"}
    filename = file.filename or "media"
    ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""

    if ext not in allowed:
        raise HTTPException(400, f"Unsupported format: .{ext}")

    content = await file.read()
    if len(content) > 100 * 1024 * 1024:
        raise HTTPException(413, "File too large. Max 100 MB.")

    with tempfile.NamedTemporaryFile(delete=False, suffix=f".{ext}") as tmp:
        tmp.write(content)
        tmp_path = tmp.name

    try:
        result = face_detector.analyze_file(tmp_path)
    finally:
        os.unlink(tmp_path)

    return {
        "file_name": filename,
        "file_size_bytes": len(content),
        "analysis": result.to_dict(),
        "summary": {
            "score": result.score,
            "verdict": result.verdict,
            "confidence": result.confidence,
            "frames_analyzed": result.frames_analyzed,
            "faces_detected": result.faces_detected,
            "worst_frame_score": result.worst_frame_score,
            "blending_artifact_score": result.blending_artifact_score,
            "processing_time_ms": result.processing_time_ms,
        }
    }


@router.post("/analyze-frames", response_model=dict)
async def analyze_frames_detail(
    file: UploadFile = File(...),
    current_user=Depends(get_current_active_user),
):
    """Detailed per-frame analysis for video files."""
    filename = file.filename or "video"
    ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    if ext not in {"mp4","avi","mov","mkv","webm"}:
        raise HTTPException(400, "Only video files supported for frame analysis.")

    content = await file.read()
    with tempfile.NamedTemporaryFile(delete=False, suffix=f".{ext}") as tmp:
        tmp.write(content); tmp_path = tmp.name

    try:
        result = face_detector.analyze_file(tmp_path)
    finally:
        os.unlink(tmp_path)

    return {
        "file_name": filename,
        "total_frames_analyzed": result.frames_analyzed,
        "frame_scores": result.frame_scores,
        "verdict": result.verdict,
        "temporal_stability": result.temporal_stability,
        "blending_artifact_score": result.blending_artifact_score,
        "summary": f"{result.frames_analyzed} frames analysed, {result.faces_detected} with detected faces",
    }
