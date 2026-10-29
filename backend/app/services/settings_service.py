from sqlalchemy.orm import Session
from app.models.user import User
from app.models.user_settings import UserSettings
from app.schemas.settings import SettingsUpdate


def get_settings(db: Session, user: User) -> UserSettings:
    obj = db.get(UserSettings, user.id)
    if obj is None:                       # δημιουργείται την πρώτη φορά με τα defaults
        obj = UserSettings(user_id=user.id, batch_size=5, notifications_enabled=True, theme="dark")
        db.add(obj)
        db.commit()
        db.refresh(obj)
    return obj


def update_settings(db: Session, user: User, data: SettingsUpdate) -> UserSettings:
    obj = get_settings(db, user)
    for key, value in data.model_dump(exclude_none=True).items():
        setattr(obj, key, value)
    db.commit()
    db.refresh(obj)
    return obj