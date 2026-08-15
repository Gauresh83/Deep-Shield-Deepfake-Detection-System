from pydantic_settings import BaseSettings
from typing import List
import os

class Settings(BaseSettings):
    APP_NAME: str = "M3-ID Backend"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = True
    FRONTEND_URL: str = "http://127.0.0.1:5500"
    SECRET_KEY: str = "m3id-super-secret-key-change-in-production-2025"
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60
    REFRESH_TOKEN_EXPIRE_DAYS: int = 7
    DATABASE_URL: str = "sqlite:///./m3id.db"
    UPLOAD_DIR: str = "uploads"
    MAX_UPLOAD_SIZE_MB: int = 50
    ALLOWED_VIDEO_TYPES: str = "mp4,avi,mov,mkv,webm"
    ALLOWED_AUDIO_TYPES: str = "wav,mp3,flac,m4a,ogg"
    ALLOWED_IMAGE_TYPES: str = "jpg,jpeg,png,webp"

    @property
    def allowed_video_list(self) -> List[str]:
        return [x.strip() for x in self.ALLOWED_VIDEO_TYPES.split(",")]

    @property
    def allowed_audio_list(self) -> List[str]:
        return [x.strip() for x in self.ALLOWED_AUDIO_TYPES.split(",")]

    @property
    def allowed_image_list(self) -> List[str]:
        return [x.strip() for x in self.ALLOWED_IMAGE_TYPES.split(",")]

    @property
    def max_upload_bytes(self) -> int:
        return self.MAX_UPLOAD_SIZE_MB * 1024 * 1024

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}

settings = Settings()
os.makedirs(settings.UPLOAD_DIR, exist_ok=True)
