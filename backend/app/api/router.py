from fastapi import APIRouter
from app.api.routes import auth, models, datasets, xai

api_router = APIRouter(prefix="/api")
api_router.include_router(auth.router)
api_router.include_router(models.router)
api_router.include_router(datasets.router)
api_router.include_router(xai.router)