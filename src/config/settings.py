"""Application settings loaded from environment variables."""
import os
from dotenv import load_dotenv

load_dotenv()


class Settings:
    # Server
    HOST: str = os.getenv("HOST", "0.0.0.0")
    PORT: int = int(os.getenv("PORT", "8000"))

    # Model
    CLIP_MODEL_NAME: str = os.getenv("CLIP_MODEL_NAME", "openai/clip-vit-base-patch32")
    DEVICE: str = os.getenv("DEVICE", "cpu")  # "cpu" or "cuda"

    # Scoring thresholds
    BLUR_THRESHOLD: float = float(os.getenv("BLUR_THRESHOLD", "80.0"))
    EAR_THRESHOLD: float = float(os.getenv("EAR_THRESHOLD", "0.20"))   # Eye Aspect Ratio
    DUPLICATE_HASH_THRESHOLD: int = int(os.getenv("DUPLICATE_HASH_THRESHOLD", "8"))

    # Batch processing
    CLIP_BATCH_SIZE: int = int(os.getenv("CLIP_BATCH_SIZE", "8"))
    SYNC_BATCH_LIMIT: int = int(os.getenv("SYNC_BATCH_LIMIT", "200"))  # images above this → async

    # Supported image extensions
    SUPPORTED_EXTENSIONS: set = {
        ".jpg", ".jpeg", ".png", ".tiff", ".tif",
        ".bmp", ".webp", ".heic", ".heif"
    }

    # Job TTL in seconds (in-memory job cleanup)
    JOB_TTL_SECONDS: int = int(os.getenv("JOB_TTL_SECONDS", "3600"))


settings = Settings()
