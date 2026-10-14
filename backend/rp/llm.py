"""LLM provider adapter. NVIDIA NIM (OpenAI-compatible) when NVIDIA_API_KEY is set; otherwise None,
and callers fall back to templates. Retry/transient logic follows CodeRush 2.0 llm.py."""

from __future__ import annotations

import time

from . import config

_client = None
_last_error: str | None = None
usage = {"calls": 0, "tokens": 0, "failures": 0}


def provider() -> dict:
    live = bool(config.NVIDIA_API_KEY) and not config.OFFLINE
    return {"provider": "nvidia-nim" if live else "template", "model": config.NVIDIA_MODEL if live else "template",
            "live": live, "last_error": _last_error}


def client():
    global _client
    if config.OFFLINE or not config.NVIDIA_API_KEY:
        return None
    if _client is None:
        from openai import OpenAI
        _client = OpenAI(api_key=config.NVIDIA_API_KEY, base_url=config.NVIDIA_BASE_URL, timeout=30, max_retries=0)
    return _client


def _transient(exc: Exception) -> bool:
    status = getattr(exc, "status_code", None)
    t = str(exc).lower()
    return status in (429, 500, 502, 503, 504) or any(k in t for k in ("timeout", "timed out", "connection", "overloaded", "rate limit"))


def chat(messages: list[dict], *, tools: list[dict] | None = None, max_tokens: int = 700, attempts: int = 2):
    """Returns the OpenAI-style message, or None when no live model is configured or it keeps failing."""
    global _last_error
    c = client()
    if c is None:
        return None
    for i in range(attempts):
        try:
            kw = dict(model=config.NVIDIA_MODEL, messages=messages, max_tokens=max_tokens, temperature=0.2)
            if tools:
                kw["tools"] = tools
            r = c.chat.completions.create(**kw)
            usage["calls"] += 1
            usage["tokens"] += getattr(r.usage, "total_tokens", 0) or 0
            _last_error = None
            return r.choices[0].message
        except Exception as exc:  # provider SDK errors vary
            _last_error = f"{type(exc).__name__}: {str(exc)[:200]}"
            usage["failures"] += 1
            if not _transient(exc) or i == attempts - 1:
                return None
            time.sleep(1.5 * 2 ** i)
    return None
