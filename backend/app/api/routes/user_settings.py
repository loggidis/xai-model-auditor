from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from app.api.deps import get_current_user
from app.db.session import get_db
from app.models.user import User
from app.schemas.settings import SettingsRead, SettingsUpdate
from app.services import settings_service

router = APIRouter(prefix="/settings", tags=["settings"])

@router.get("", response_model=SettingsRead)
def get_settings(db: Session = Depends(get_db), user: User = Depends(get_current_user)):
    return settings_service.get_settings(db, user)

@router.patch("", response_model=SettingsRead)
def update_settings(data: SettingsUpdate, db: Session = Depends(get_db),
                    user: User = Depends(get_current_user)):
    return settings_service.update_settings(db, user, data)