from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordRequestForm
from fastapi.responses import HTMLResponse
from sqlalchemy.orm import Session

from app.api.deps import get_current_user
from app.api.rate_limits import limit_by_ip
from app.core.config import settings
from app.core.rate_limit import enforce
from app.core.security import create_access_token
from app.db.session import get_db
from app.models.user import User
from app.schemas.auth import (MessageResponse, RefreshRequest, ResendVerificationRequest, Token,
                              VerifyEmailRequest)
from app.schemas.user import UserCreate, UserRead
from app.services import auth_service, email_service, token_service

router = APIRouter(prefix="/auth", tags=["auth"])

_VERIFY_PAGE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Verify your email</title>
<style>
body{font-family:system-ui,sans-serif;max-width:28rem;margin:4rem auto;padding:0 1rem;text-align:center}
button{font-size:1rem;padding:.7rem 1.4rem;border:0;border-radius:.4rem;background:#2563eb;color:#fff;cursor:pointer}
button:disabled{opacity:.6;cursor:default}
#msg{margin-top:1.5rem}
</style></head>
<body>
<h1>Confirm your email</h1>
<p>Click the button to finish verifying your email address.</p>
<button id="go">Confirm email</button>
<p id="msg"></p>
<script>
const token = new URLSearchParams(location.search).get("token");
const btn = document.getElementById("go"), msg = document.getElementById("msg");
if (!token) { btn.disabled = true; msg.textContent = "This link is missing its token."; }
btn.addEventListener("click", async () => {
  btn.disabled = true; msg.textContent = "Verifying...";
  try {
    const r = await fetch("/api/auth/verify-email", {
      method: "POST", headers: {"Content-Type": "application/json"}, body: JSON.stringify({token})});
    if (r.ok) { msg.textContent = "Your email address is verified. You can close this page and log in."; btn.style.display = "none"; }
    else { const b = await r.json().catch(() => null);
           msg.textContent = (b && b.error && b.error.message) || "Verification failed."; btn.disabled = false; }
  } catch (e) { msg.textContent = "Network error, please try again."; btn.disabled = false; }
});
</script>
</body></html>"""


@router.get("/verify-email", response_class=HTMLResponse, include_in_schema=False)
def verify_email_page() -> str:
    """Page opened from the email link. It only shows a button: the token is consumed by the POST below,
    so mail scanners that pre-open links cannot use it up."""
    return _VERIFY_PAGE

def _token_response(user: User, refresh_token: str) -> Token:
    return Token(access_token=create_access_token(str(user.id)), refresh_token=refresh_token,
                 expires_in=settings.ACCESS_TOKEN_EXPIRE_MINUTES * 60)


@router.post("/register", response_model=UserRead, status_code=status.HTTP_201_CREATED,
             dependencies=[Depends(limit_by_ip("register", settings.RATE_LIMIT_REGISTER_PER_HOUR, 3600))])
def register(data: UserCreate, background: BackgroundTasks, db: Session = Depends(get_db)):
    if auth_service.get_user_by_email(db, data.email):
        raise HTTPException(status.HTTP_409_CONFLICT, "Email already registered")
    user = auth_service.create_user(db, data)
    raw = auth_service.create_verification_token(db, user)
    background.add_task(email_service.send_verification_email, user.email, raw)
    return user


@router.post("/login", response_model=Token,
             dependencies=[Depends(limit_by_ip("login", settings.RATE_LIMIT_LOGIN_PER_MINUTE, 60))])
def login(form: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(get_db)):
    # The OAuth2 form sends the email in the "username" field. Every attempt for the same
    # email counts, so one account cannot be brute-forced from many IP addresses.
    enforce(f"login:email:{form.username.lower()}", settings.RATE_LIMIT_LOGIN_PER_ACCOUNT, 900)
    user = auth_service.authenticate(db, form.username, form.password)
    if not user:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Incorrect email or password")
    if settings.REQUIRE_EMAIL_VERIFICATION and not user.email_verified:
        raise HTTPException(status.HTTP_403_FORBIDDEN, "Email address is not verified")
    return _token_response(user, token_service.issue_refresh_token(db, user.id))


@router.post("/refresh", response_model=Token,
             dependencies=[Depends(limit_by_ip("refresh", settings.RATE_LIMIT_REFRESH_PER_MINUTE, 60))])
def refresh(data: RefreshRequest, db: Session = Depends(get_db)):
    """Exchanges a refresh token for a new access token and a new refresh token (rotation)."""
    user, new_refresh = token_service.rotate_refresh_token(db, data.refresh_token)
    return _token_response(user, new_refresh)


@router.post("/logout", status_code=status.HTTP_204_NO_CONTENT)
def logout(data: RefreshRequest, db: Session = Depends(get_db)) -> None:
    """Revokes the refresh tokens of this login. The access token stays valid until it expires."""
    token_service.logout(db, data.refresh_token)


@router.post("/logout-all", status_code=status.HTTP_204_NO_CONTENT)
def logout_all(db: Session = Depends(get_db), user: User = Depends(get_current_user)) -> None:
    token_service.logout_all(db, user.id)


@router.get("/me", response_model=UserRead)
def me(current_user: User = Depends(get_current_user)):
    return current_user


@router.post("/verify-email", response_model=MessageResponse,
             dependencies=[Depends(limit_by_ip("verify", settings.RATE_LIMIT_EMAIL_PER_HOUR * 4, 3600))])
def verify_email(data: VerifyEmailRequest, db: Session = Depends(get_db)):
    auth_service.verify_email(db, data.token)
    return MessageResponse(message="Email verified")


@router.post("/resend-verification", response_model=MessageResponse, status_code=status.HTTP_202_ACCEPTED,
             dependencies=[Depends(limit_by_ip("resend", settings.RATE_LIMIT_EMAIL_PER_HOUR, 3600))])
def resend_verification(data: ResendVerificationRequest, background: BackgroundTasks,
                        db: Session = Depends(get_db)):
    enforce(f"resend:email:{data.email.lower()}", settings.RATE_LIMIT_EMAIL_PER_HOUR, 3600)
    user = auth_service.get_user_by_email(db, data.email)
    if user and user.is_active and not user.email_verified:
        raw = auth_service.create_verification_token(db, user)
        background.add_task(email_service.send_verification_email, user.email, raw)
    # Same answer in every case, so the endpoint does not reveal which emails are registered.
    return MessageResponse(message="If the account exists and is not verified, a new email has been sent")