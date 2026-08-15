import os, uuid, aiofiles
from fastapi import UploadFile, HTTPException
from ..core.config import settings

ALLOWED = settings.allowed_video_list + settings.allowed_audio_list + settings.allowed_image_list

async def save_upload_file(file: UploadFile, user_id: int):
    original = file.filename or "unknown"
    ext = original.rsplit(".", 1)[-1].lower() if "." in original else ""
    if ext not in ALLOWED:
        raise HTTPException(400, f"File type .{ext} not allowed.")
    user_dir = os.path.join(settings.UPLOAD_DIR, str(user_id))
    os.makedirs(user_dir, exist_ok=True)
    unique_name = f"{user_id}_{uuid.uuid4().hex[:12]}.{ext}"
    path = os.path.join(user_dir, unique_name)
    content = await file.read()
    if len(content) > settings.max_upload_bytes:
        raise HTTPException(413, f"File exceeds {settings.MAX_UPLOAD_SIZE_MB} MB limit.")
    async with aiofiles.open(path, "wb") as out:
        await out.write(content)
    return path, original, len(content)

def delete_file(path: str):
    try:
        if os.path.exists(path):
            os.remove(path)
            return True
    except OSError:
        pass
    return False
