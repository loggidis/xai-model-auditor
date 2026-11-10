from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.router import api_router
from app.core.config import settings
from app.core.errors import register_error_handlers

DESCRIPTION = """
Audit object detection models with Explainable AI (Grad-CAM++, Integrated Gradients, LIME, SHAP).
"""

TAGS = [
    {"name": "auth", "description": "Registration, login, token refresh, email verification."},
    {"name": "models", "description": "Model registry (upload `.pt` files, inspect metadata and layers)."},
    {"name": "datasets", "description": "Image datasets (zip upload with optional YOLO labels)."},
    {"name": "xai", "description": "Available XAI methods, their parameters and masks."},
    {"name": "audits", "description": "Create and control audits; read per-image results and heatmaps."},
    {"name": "evaluations", "description": "User trust ratings and categories per image."},
    {"name": "reports", "description": "Report, flagged images, export for retraining, LLM summary."},
    {"name": "notifications", "description": "Notifications about finished background jobs."},
    {"name": "settings", "description": "Per-user settings."},
    {"name": "health", "description": "Liveness and queue status."},
]

app = FastAPI(
    title="XAI Model Auditor",
    description=DESCRIPTION,
    openapi_tags=TAGS,
    docs_url="/docs" if settings.DOCS_ENABLED else None,
    redoc_url="/redoc" if settings.DOCS_ENABLED else None,
    openapi_url="/openapi.json" if settings.DOCS_ENABLED else None,
)

register_error_handlers(app)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000"],   # frontend dev servers
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["Retry-After"],
)

app.include_router(api_router)