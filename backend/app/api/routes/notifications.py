import uuid
from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session
from app.api.deps import get_current_user
from app.db.session import get_db
from app.models.user import User
from app.schemas.notification import NotificationRead, UnreadCount
from app.services import notification_service

router = APIRouter(prefix="/notifications", tags=["notifications"])


@router.get("", response_model=list[NotificationRead])
def list_notifications(unread_only: bool = False, limit: int = Query(50, ge=1, le=200),
                       offset: int = Query(0, ge=0), db: Session = Depends(get_db),
                       user: User = Depends(get_current_user)):
    return notification_service.list_notifications(db, user, unread_only, limit, offset)


@router.get("/unread-count", response_model=UnreadCount)
def unread_count(db: Session = Depends(get_db), user: User = Depends(get_current_user)):
    return UnreadCount(count=notification_service.unread_count(db, user))


@router.patch("/{notification_id}/read", response_model=NotificationRead)
def mark_read(notification_id: uuid.UUID, db: Session = Depends(get_db),
              user: User = Depends(get_current_user)):
    return notification_service.mark_read(db, user, notification_id)