import hashlib
import secrets
import uuid
from datetime import datetime, timedelta, timezone

from fastapi import HTTPException
from sqlalchemy import select, update
from sqlalchemy.orm import Session

from app.core.config import settings
from app.models.auth_tokens import RefreshToken
from app.models.user import User


def _now() -> datetime:
    return datetime.now(timezone.utc)


def hash_token(raw: str) -> str:
    return hashlib.sha256(raw.encode()).hexdigest()


def _unauthorized(message: str = "Invalid or expired refresh token") -> HTTPException:
    return HTTPException(401, message, headers={"WWW-Authenticate": "Bearer"})


def issue_refresh_token(db: Session, user_id: uuid.UUID, family_id: uuid.UUID | None = None) -> str:
    raw = secrets.token_urlsafe(48)
    db.add(RefreshToken(
        user_id=user_id, family_id=family_id or uuid.uuid4(), token_hash=hash_token(raw),
        expires_at=_now() + timedelta(days=settings.REFRESH_TOKEN_EXPIRE_DAYS),
    ))
    db.commit()
    return raw


def revoke_family(db: Session, family_id: uuid.UUID) -> None:
    db.execute(update(RefreshToken)
               .where(RefreshToken.family_id == family_id, RefreshToken.revoked_at.is_(None))
               .values(revoked_at=_now()))
    db.commit()


def rotate_refresh_token(db: Session, raw: str) -> tuple[User, str]:
    """Validates a refresh token, revokes it and issues a new one in the same family."""
    row = db.scalar(select(RefreshToken).where(RefreshToken.token_hash == hash_token(raw)).with_for_update())
    if row is None:
        raise _unauthorized()
    if row.revoked_at is not None:
        revoke_family(db, row.family_id)       # a rotated token was presented again: assume it was stolen
        raise _unauthorized("Refresh token reuse detected, please log in again")
    if row.expires_at <= _now():
        raise _unauthorized()
    user = db.get(User, row.user_id)
    if user is None or not user.is_active:
        raise _unauthorized()
    row.revoked_at = _now()
    new_raw = issue_refresh_token(db, user.id, row.family_id)     # commits both changes
    return user, new_raw


def logout(db: Session, raw: str) -> None:
    """Revokes the whole login (family) of the given token. Idempotent: unknown tokens are ignored."""
    row = db.scalar(select(RefreshToken).where(RefreshToken.token_hash == hash_token(raw)))
    if row is not None:
        revoke_family(db, row.family_id)


def logout_all(db: Session, user_id: uuid.UUID) -> None:
    db.execute(update(RefreshToken)
               .where(RefreshToken.user_id == user_id, RefreshToken.revoked_at.is_(None))
               .values(revoked_at=_now()))
    db.commit()