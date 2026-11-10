from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    DATABASE_URL: str
    SECRET_KEY: str
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30

    # Redis
    REDIS_URL: str = "redis://localhost:6379/0"


    # Storage (backend/app/core/config.py -> parents[3] = project root)
    STORAGE_DIR: Path = Path(__file__).resolve().parents[3] / "storage"
    MAX_MODEL_SIZE_MB: int = 500
    MAX_DATASET_ZIP_MB: int = 1024
    MAX_DATASET_UNCOMPRESSED_MB: int = 2048
    MAX_DATASET_FILES: int = 20000
    XAI_DEVICE: str = "auto"   

        # --- Auth ---
    REFRESH_TOKEN_EXPIRE_DAYS: int = 14
    REQUIRE_EMAIL_VERIFICATION: bool = False
    EMAIL_VERIFICATION_EXPIRE_HOURS: int = 24

    # --- Email ---
    SMTP_HOST: str | None = None
    SMTP_PORT: int = 587
    SMTP_USER: str | None = None
    SMTP_PASSWORD: str | None = None
    SMTP_STARTTLS: bool = True
    SMTP_SSL: bool = False
    EMAIL_FROM: str = "XAI Model Auditor <no-reply@localhost>"
    EMAIL_DEV_MODE: bool = False                     # True: without SMTP_HOST, print emails to the console (offline dev only)
    SMTP_TIMEOUT_S: float = 15.0
    PUBLIC_API_URL: str = "http://localhost:8000"    # where the user's browser can reach this API
    VERIFICATION_URL_TEMPLATE: str | None = None     # e.g. "http://localhost:5173/verify-email?token={token}" once a frontend exists

    # --- Rate limiting ---
    RATE_LIMIT_ENABLED: bool = True
    TRUST_PROXY_HEADERS: bool = False                # only behind a proxy that sets X-Forwarded-For
    RATE_LIMIT_LOGIN_PER_MINUTE: int = 10            # per IP
    RATE_LIMIT_LOGIN_PER_ACCOUNT: int = 20           # per email, per 15 minutes
    RATE_LIMIT_REGISTER_PER_HOUR: int = 10           # per IP
    RATE_LIMIT_REFRESH_PER_MINUTE: int = 30          # per IP
    RATE_LIMIT_EMAIL_PER_HOUR: int = 5               # resend verification, per IP and per email
    RATE_LIMIT_AUDITS_PER_HOUR: int = 20             # per user
    RATE_LIMIT_UPLOADS_PER_HOUR: int = 30            # per user (models and datasets)
    RATE_LIMIT_JOBS_PER_HOUR: int = 20               # per user (exports and LLM summaries)

    # --- LLM (Groq) ---
    GROQ_API_KEY: str | None = None
    GROQ_BASE_URL: str = "https://api.groq.com/openai/v1"
    GROQ_MODEL: str = "llama-3.3-70b-versatile"      # check console.groq.com/docs/models
    GROQ_TIMEOUT_S: float = 60.0
    GROQ_MAX_RETRIES: int = 3
    LLM_SUMMARY_MAX_TOKENS: int = 1500
    LLM_DRY_RUN: bool = False                        # build the prompt and run the whole flow, but never call the LLM
    
    # --- Cleanup ---
    CLEANUP_ORPHAN_GRACE_HOURS: int = 24

    # --- API docs ---
    DOCS_ENABLED: bool = True

        # --- PDF ---
    PDF_FONT_DIR: str | None = None                  # folder with DejaVuSans*.ttf; default: the fonts bundled with matplotlib
settings = Settings()