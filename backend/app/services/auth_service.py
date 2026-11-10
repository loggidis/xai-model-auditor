import secrets
from datetime import datetime, timedelta, timezone

from fastapi import HTTPException
from sqlalchemy import select, update
from sqlalchemy.orm import Session

from app.core.config import settings
from app.core.security import hash_password, verify_password
from app.models.auth_tokens import EmailVerificationToken
from app.models.user import User
from app.schemas.user import UserCreate
from app.services.token_service import hash_token


def _now() -> datetime:
    return datetime.now(timezone.utc)


def get_user_by_email(db: Session, email: str) -> User | None:
    return db.scalar(select(User).where(User.email == email.lower()))


def create_user(db: Session, data: UserCreate) -> User:
    user = User(
        email=data.email.lower(),
        hashed_password=hash_password(data.password),
        full_name=data.full_name,
    )
    db.add(user)
    db.commit()
    db.refresh(user)
    return user


def authenticate(db: Session, email: str, password: str) -> User | None:
    user = get_user_by_email(db, email)
    if not user or not verify_password(password, user.hashed_password):
        return None
    return user


def create_verification_token(db: Session, user: User) -> str:
    """Creates a new verification token and invalidates the older unused ones."""
    db.execute(update(EmailVerificationToken)
               .where(EmailVerificationToken.user_id == user.id, EmailVerificationToken.used_at.is_(None))
               .values(used_at=_now()))
    raw = secrets.token_urlsafe(32)
    db.add(EmailVerificationToken(
        user_id=user.id, token_hash=hash_token(raw),
        expires_at=_now() + timedelta(hours=settings.EMAIL_VERIFICATION_EXPIRE_HOURS),
    ))
    db.commit()
    return raw


def verify_email(db: Session, raw: str) -> User:
    row = db.scalar(select(EmailVerificationToken).where(EmailVerificationToken.token_hash == hash_token(raw)))
    if row is None or row.used_at is not None or row.expires_at <= _now():
        raise HTTPException(400, "Invalid or expired verification token")
    user = db.get(User, row.user_id)
    if user is None:
        raise HTTPException(400, "Invalid or expired verification token")
    row.used_at = _now()
    user.email_verified = True
    db.commit()
    db.refresh(user)
    return user