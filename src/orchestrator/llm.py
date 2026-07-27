"""Thin ChatOllama wrapper plus a JSON-mode extract-validate-repair loop.

One repair attempt after an initial schema-validation failure - the same
pattern Project 01 established for getting small local models to reliable
structured output: the retry prompt includes the previous bad output and
the exact validation error, not a blind resend.

Transport failures (connection errors, raw socket timeouts, and the
library's own timeout wrapper) are caught here and normalized into
``LLMError`` so callers don't need to know Ollama's exception taxonomy.
"""

from __future__ import annotations

import json
from typing import TypeVar

from langchain_ollama import ChatOllama
from pydantic import BaseModel, ValidationError

from orchestrator.config import CONFIG

T = TypeVar("T", bound=BaseModel)

# Everything we've observed a slow/loaded shared Ollama server raise for a
# stalled request: httpx's own timeout type, the stdlib socket timeout it
# wraps, and asyncio's TimeoutError alias (same class as builtins on 3.11+
# but kept explicit for clarity).
_TRANSPORT_EXCEPTIONS = (TimeoutError, ConnectionError, OSError)


class LLMError(Exception):
    """Raised for any transport failure or unrecoverable schema failure."""


def make_chat(*, json_mode: bool = False, temperature: float | None = None) -> ChatOllama:
    return ChatOllama(
        model=CONFIG.ollama_model,
        base_url=CONFIG.ollama_host,
        temperature=CONFIG.temperature if temperature is None else temperature,
        format="json" if json_mode else None,
        client_kwargs={"timeout": CONFIG.request_timeout_s},
    )


def _extract_text(response) -> str:
    content = response.content
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(part.get("text", "") if isinstance(part, dict) else str(part) for part in content)
    return str(content)


def complete_text(system_prompt: str, user_prompt: str, *, temperature: float | None = None) -> str:
    chat = make_chat(json_mode=False, temperature=temperature)
    try:
        response = chat.invoke([("system", system_prompt), ("human", user_prompt)])
    except _TRANSPORT_EXCEPTIONS as exc:
        raise LLMError(f"transport error talking to Ollama at {CONFIG.ollama_host}: {exc}") from exc
    return _extract_text(response).strip()


def complete_json(
    schema: type[T],
    system_prompt: str,
    user_prompt: str,
    *,
    max_attempts: int | None = None,
    temperature: float | None = None,
) -> T:
    """JSON-mode call -> validate against ``schema`` -> up to one repair retry."""
    attempts_budget = max_attempts or CONFIG.max_json_attempts
    chat = make_chat(json_mode=True, temperature=temperature)

    messages: list[tuple[str, str]] = [("system", system_prompt), ("human", user_prompt)]
    last_error = ""
    last_raw = ""

    for attempt in range(1, attempts_budget + 1):
        try:
            response = chat.invoke(messages)
        except _TRANSPORT_EXCEPTIONS as exc:
            raise LLMError(f"transport error talking to Ollama at {CONFIG.ollama_host}: {exc}") from exc

        raw = _extract_text(response)
        last_raw = raw
        try:
            payload = json.loads(raw)
            return schema.model_validate(payload)
        except (json.JSONDecodeError, ValidationError) as exc:
            last_error = str(exc)
            if attempt < attempts_budget:
                messages = [
                    ("system", system_prompt),
                    (
                        "human",
                        f"{user_prompt}\n\nYour previous response was not valid JSON for the "
                        f"required schema.\nPrevious response:\n{raw}\n\nValidation error:\n"
                        f"{last_error}\n\nRespond again with ONLY the corrected JSON object.",
                    ),
                ]

    raise LLMError(
        f"failed to produce a valid {schema.__name__} after {attempts_budget} attempt(s): "
        f"{last_error}\nlast raw output: {last_raw}"
    )
