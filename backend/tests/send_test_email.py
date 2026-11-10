"""Sends a real test email, to check the SMTP settings.

python -m scripts.send_test_email --to you@example.com
"""
import argparse

from app.core.config import settings
from app.services import email_service


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--to", required=True)
    a = ap.parse_args()
    if not settings.SMTP_HOST:
        raise SystemExit("SMTP_HOST is not set in backend/.env")
    print(f"Sending through {settings.SMTP_HOST}:{settings.SMTP_PORT} as {settings.SMTP_USER or '(no login)'} ...")
    try:
        email_service.send_email(a.to, "XAI Model Auditor: test email",
                                 "If you can read this, the SMTP settings are correct.", raise_errors=True)
    except email_service.EmailError as e:
        raise SystemExit(f"FAILED: {e}")
    print("OK: the SMTP server accepted the message. Check the inbox (and the spam folder).")


if __name__ == "__main__":
    main()