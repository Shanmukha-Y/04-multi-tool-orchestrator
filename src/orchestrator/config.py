"""Central configuration. The model name lives here and nowhere else.

Every field can be overridden by an environment variable of the same name
(upper-cased, ``ORCHESTRATOR_`` prefixed) so the demo can be tuned without
editing source, but the *defaults* are the single source of truth for
"what model does this project use" and "where does Ollama live".
"""

from __future__ import annotations

import os
from dataclasses import dataclass


def _env(name: str, default: str) -> str:
    return os.environ.get(f"ORCHESTRATOR_{name.upper()}", default)


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(f"ORCHESTRATOR_{name.upper()}")
    return float(raw) if raw is not None else default


@dataclass(frozen=True)
class Config:
    ollama_host: str = "http://localhost:11434"
    ollama_model: str = "qwen3.5:9b"

    # Generous by default: this is a shared local Ollama server that other
    # projects also hit, and a 9B model doing JSON-mode planning under load
    # has been observed to take up to ~240s for a single call.
    request_timeout_s: float = 240.0
    temperature: float = 0.1

    # Planner / resolver JSON-mode retry budget: one repair attempt after
    # an initial failure, matching the pattern established in Project 01.
    max_json_attempts: int = 2

    notes_dir: str = "notes_output"
    data_dir: str = "data"


def load_config() -> Config:
    return Config(
        ollama_host=_env("host", Config.ollama_host),
        ollama_model=_env("model", Config.ollama_model),
        request_timeout_s=_env_float("request_timeout_s", Config.request_timeout_s),
        temperature=_env_float("temperature", Config.temperature),
    )


CONFIG = load_config()
