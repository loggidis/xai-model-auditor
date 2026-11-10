import os

import pytest

from app.core.config import settings
from app.services import llm_service

pytestmark = pytest.mark.llm_live


@pytest.mark.skipif(os.environ.get("RUN_LLM_TESTS") != "1",
                    reason="Set RUN_LLM_TESTS=1 to run the live Groq smoke test (it spends a few tokens)")
def test_groq_smoke():
    if not settings.GROQ_API_KEY:
        pytest.skip("GROQ_API_KEY is not set")
    text, model = llm_service.chat_completion(
        [{"role": "user", "content": "Reply with the single word: OK"}], max_tokens=512, temperature=0)
    assert text.strip() and model