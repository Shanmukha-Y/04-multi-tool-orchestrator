"""Import side effects register the seven always-on demo tools into the
global registry (each uses the ``@tool_def`` decorator, which fires on
first import).

``weather_c`` is deliberately excluded from this default set. It lives in
``weather_c.py`` and is registered explicitly via ``register_weather_c()``
- that's the tool the Milestone 5 demo registers *mid-session* to show
dynamic routing, so it must not already be in the registry at startup.
"""

from __future__ import annotations

from orchestrator.registry import Registry

from orchestrator.tools import (  # noqa: F401
    calc,
    email,
    files,
    fx,
    notes,
    search,
    weather_a,
    weather_b,
)


def register_weather_c(registry: Registry | None = None) -> None:
    """Registers the third weather provider. Idempotent: safe to call more
    than once (``Registry.register`` is invoked with ``replace=True``)."""
    from orchestrator.tools import weather_c

    weather_c.register(registry)
