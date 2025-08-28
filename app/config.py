from pydantic import BaseModel
import os

class Settings(BaseModel):
    ELEVENLABS_API_KEY: str = os.getenv("ELEVENLABS_API_KEY", "")
    DEFAULT_VOICE_ID: str = os.getenv("DEFAULT_VOICE_ID", "REPLACE_WITH_VOICE_ID")
    DEFAULT_MODEL_ID: str = os.getenv("DEFAULT_MODEL_ID", "eleven_turbo_v2_5")
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "info")
    MAX_TEXT_CHARS: int = int(os.getenv("MAX_TEXT_CHARS", "1000"))
    # Regex som matchar Lovable-subdomäner och localhost under dev
    ALLOWED_ORIGIN_REGEX: str = os.getenv(
        "ALLOWED_ORIGIN_REGEX",
        r"^https?://(localhost(:\d+)?|.*\.lovable\.app)$",
    )

settings = Settings()
