import logging
import smtplib
import ssl
from email.message import EmailMessage
from html import escape

from app.core.config import settings

log = logging.getLogger(__name__)


class EmailError(Exception):
    """The email could not be handed to the SMTP server."""


def _build(to: str, subject: str, text: str, html: str | None) -> EmailMessage:
    msg = EmailMessage()
    msg["From"], msg["To"], msg["Subject"] = settings.EMAIL_FROM, to, subject
    msg.set_content(text)
    if html:
        msg.add_alternative(html, subtype="html")
    return msg


def _deliver(msg: EmailMessage) -> None:
    """The only function that talks to the SMTP server (tests replace it)."""
    context = ssl.create_default_context()                 # verifies the server certificate
    try:
        if settings.SMTP_SSL:
            server = smtplib.SMTP_SSL(settings.SMTP_HOST, settings.SMTP_PORT,
                                      timeout=settings.SMTP_TIMEOUT_S, context=context)
        else:
            server = smtplib.SMTP(settings.SMTP_HOST, settings.SMTP_PORT, timeout=settings.SMTP_TIMEOUT_S)
        with server:
            if not settings.SMTP_SSL and settings.SMTP_STARTTLS:
                server.starttls(context=context)
            if settings.SMTP_USER:
                server.login(settings.SMTP_USER, settings.SMTP_PASSWORD or "")
            server.send_message(msg)
    except smtplib.SMTPAuthenticationError as e:
        raise EmailError(f"SMTP authentication failed ({e.smtp_code}). Check SMTP_USER and SMTP_PASSWORD "
                         "(Gmail requires an app password, not the account password)") from e
    except (smtplib.SMTPException, OSError) as e:
        raise EmailError(f"SMTP error: {type(e).__name__}: {e}") from e


def send_email(to: str, subject: str, text: str, html: str | None = None, *, raise_errors: bool = False) -> bool:
    """Sends an email. Returns True if the SMTP server accepted it.

    Request handlers call this from background tasks, so by default failures are logged, not raised.
    """
    if not settings.SMTP_HOST:
        if settings.EMAIL_DEV_MODE:
            log.warning("SMTP is not configured; printing the email instead.\nTo: %s\nSubject: %s\n\n%s",
                        to, subject, text)
            return False
        error = EmailError("SMTP is not configured (SMTP_HOST is empty)")
        if raise_errors:
            raise error
        log.error("Email to %s was not sent: %s", to, error)
        return False
    try:
        _deliver(_build(to, subject, text, html))
    except EmailError as e:
        if raise_errors:
            raise
        log.error("Email to %s failed: %s", to, e)
        return False
    log.info("Email sent to %s", to)
    return True


def verification_link(raw_token: str) -> str:
    template = settings.VERIFICATION_URL_TEMPLATE or (
        settings.PUBLIC_API_URL.rstrip("/") + "/api/auth/verify-email?token={token}")
    return template.replace("{token}", raw_token)


def send_verification_email(to: str, raw_token: str) -> bool:
    link = verification_link(raw_token)
    hours = settings.EMAIL_VERIFICATION_EXPIRE_HOURS
    text = (
        "Welcome to XAI Model Auditor.\n\n"
        f"Confirm your email address by opening this link:\n{link}\n\n"
        f"The link expires in {hours} hours. If you did not create an account, ignore this message.\n"
    )
    html = (
        '<div style="font-family:Arial,sans-serif;max-width:480px;margin:auto;color:#111">'
        "<h2>Welcome to XAI Model Auditor</h2>"
        "<p>Confirm your email address to finish creating your account.</p>"
        f'<p><a href="{escape(link)}" style="display:inline-block;padding:12px 20px;background:#2563eb;'
        'color:#ffffff;text-decoration:none;border-radius:6px">Verify email</a></p>'
        f'<p style="font-size:13px;color:#555">Or paste this link in your browser:<br>{escape(link)}</p>'
        f'<p style="font-size:13px;color:#555">The link expires in {hours} hours. '
        "If you did not create an account, ignore this message.</p></div>"
    )
    return send_email(to, "Verify your email address", text, html)