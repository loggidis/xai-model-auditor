from fastapi import APIRouter

from app.api.routes import (audit_stream, audits, auth, datasets, evaluations, health, models,
                            notifications, reports, user_settings, xai)
from app.schemas.error import ErrorResponse

_ERRORS = {
    401: {"model": ErrorResponse, "description": "Missing, invalid or expired credentials"},
    422: {"model": ErrorResponse, "description": "Request validation failed"},
    429: {"model": ErrorResponse, "description": "Rate limit exceeded (see the Retry-After header)"},
}

api_router = APIRouter(prefix="/api", responses=_ERRORS)
api_router.include_router(health.router)
api_router.include_router(auth.router)
api_router.include_router(models.router)
api_router.include_router(datasets.router)
api_router.include_router(xai.router)
api_router.include_router(user_settings.router)
api_router.include_router(audits.router)
api_router.include_router(audit_stream.router)
api_router.include_router(evaluations.router)
api_router.include_router(reports.router)
api_router.include_router(notifications.router)