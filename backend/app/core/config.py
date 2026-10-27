from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    DATABASE_URL: str
    SECRET_KEY: str
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30

    # Storage (backend/app/core/config.py -> parents[3] = ρίζα project)
    STORAGE_DIR: Path = Path(__file__).resolve().parents[3] / "storage"
    MAX_MODEL_SIZE_MB: int = 500
    MAX_DATASET_ZIP_MB: int = 1024
    MAX_DATASET_UNCOMPRESSED_MB: int = 2048
    MAX_DATASET_FILES: int = 20000


settings = Settings()