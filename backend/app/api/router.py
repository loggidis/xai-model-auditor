from fastapi import APIRouter
from app.api.routes import audits, auth, datasets, health, models, user_settings, xai

api_router = APIRouter(prefix="/api")
api_router.include_router(health.router)
api_router.include_router(auth.router)
api_router.include_router(models.router)
api_router.include_router(datasets.router)
api_router.include_router(xai.router)
api_router.include_router(user_settings.router)
api_router.include_router(audits.router)