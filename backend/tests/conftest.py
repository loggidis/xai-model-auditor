import pytest

from app.core.config import settings
from app.services import email_service, llm_service


@pytest.fixture(autouse=True)
def _no_rate_limits(monkeypatch):
    monkeypatch.setattr(settings, "RATE_LIMIT_ENABLED", False)


@pytest.fixture(autouse=True)
def _block_real_llm(request, monkeypatch):
    """Safety net: a test can never spend LLM tokens by accident.

    Tests that need a summary patch llm_service.chat_completion themselves (the patch overrides this one).
    Only tests marked with @pytest.mark.llm_live may reach the real provider.
    """
    if request.node.get_closest_marker("llm_live"):
        return

    def blocked(*args, **kwargs):
        raise AssertionError(
            "A real LLM call was attempted in a test. Patch llm_service.chat_completion, "
            "or mark the test with @pytest.mark.llm_live."
        )

    monkeypatch.setattr(llm_service, "chat_completion", blocked)


@pytest.fixture(autouse=True)
def outbox(monkeypatch):
    """Safety net: tests never send real emails, even when SMTP is configured in .env.

    The messages that would have been sent are collected in this list.
    """
    sent: list = []
    monkeypatch.setattr(email_service, "_deliver", lambda msg: sent.append(msg))
    return sent