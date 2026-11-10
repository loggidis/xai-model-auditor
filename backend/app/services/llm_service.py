import logging
import time

import httpx

from app.core.config import settings

log = logging.getLogger(__name__)
_RETRY_STATUSES = {429, 500, 502, 503, 504}


class LLMError(Exception):
    """The LLM provider failed or is not configured."""


def _retry_delay(resp: httpx.Response, attempt: int) -> float:
    try:
        return min(float(resp.headers.get("retry-after")), 30.0)
    except (TypeError, ValueError):
        return float(2 ** attempt)


def _error_message(resp: httpx.Response) -> str:
    try:
        return str(resp.json()["error"]["message"])[:300]
    except Exception:
        return resp.text[:300]


def chat_completion(messages: list[dict], *, max_tokens: int | None = None,
                    temperature: float = 0.2) -> tuple[str, str]:
    """Calls the Groq chat completions API. Returns (text, model). Retries on 429 and 5xx errors."""
    if not settings.GROQ_API_KEY:
        raise LLMError("GROQ_API_KEY is not configured")
    url = f"{settings.GROQ_BASE_URL.rstrip('/')}/chat/completions"
    payload = {
        "model": settings.GROQ_MODEL,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens or settings.LLM_SUMMARY_MAX_TOKENS,
    }
    headers = {"Authorization": f"Bearer {settings.GROQ_API_KEY}"}    # the key is never logged

    last = "unknown error"
    for attempt in range(settings.GROQ_MAX_RETRIES + 1):
        delay = float(2 ** attempt)
        try:
            resp = httpx.post(url, json=payload, headers=headers, timeout=settings.GROQ_TIMEOUT_S)
        except httpx.HTTPError as e:
            last = f"network error ({type(e).__name__})"
        else:
            if resp.status_code == 200:
                try:
                    data = resp.json()
                    text = data["choices"][0]["message"]["content"]
                except (KeyError, IndexError, ValueError, TypeError):
                    raise LLMError("Unexpected response format from the LLM provider")
                if not text or not text.strip():
                    raise LLMError("The LLM returned an empty response")
                return text.strip(), str(data.get("model") or settings.GROQ_MODEL)
            if resp.status_code not in _RETRY_STATUSES:
                raise LLMError(f"LLM provider error {resp.status_code}: {_error_message(resp)}")
            last = f"HTTP {resp.status_code}"
            delay = _retry_delay(resp, attempt)
        if attempt < settings.GROQ_MAX_RETRIES:
            log.warning("LLM call failed (%s), retrying in %.1fs", last, delay)
            time.sleep(delay)
    raise LLMError(f"LLM provider unavailable after retries ({last})")