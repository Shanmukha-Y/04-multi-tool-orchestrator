"""The third weather provider - held back from auto-registration on purpose.

This module is imported and registered *explicitly*, mid-session, by
``orchestrator.tools.register_weather_c()``. That's the whole point: it is
the tool the M5 demo brings online after the process has already started,
to prove the router picks it up with zero code changes elsewhere.

Highest priority of the three, so under the ``priority`` resolution policy
it wins outright once it joins the pool - a visible, easy-to-narrate effect.
"""

from __future__ import annotations

import asyncio
import time

from orchestrator.manifests import ScopeClass, ToolManifest
from orchestrator.registry import REGISTRY, Registry

_DATA = {
    "tokyo": {"temp_c": 30.0, "condition": "clear"},
    "paris": {"temp_c": 20.0, "condition": "partly cloudy"},
    "new york": {"temp_c": 23.5, "condition": "clear"},
    "london": {"temp_c": 16.5, "condition": "light rain"},
    "san francisco": {"temp_c": 16.0, "condition": "fog"},
}

MANIFEST = ToolManifest(
    name="weather_c",
    description="Weather provider C (canned/mock, registered at runtime, highest priority).",
    capabilities=["weather.current"],
    scope=ScopeClass.READ,
    priority=3,
    cost_hint=1.2,
    timeout_s=5.0,
    param_schema={"location": "city name, e.g. 'Tokyo'"},
)


async def weather_c(location: str) -> dict:
    await asyncio.sleep(0.25)
    key = location.strip().lower()
    entry = _DATA.get(key)
    if entry is None:
        return {"location": location, "error": f"no data for '{location}'"}
    return {
        "location": location,
        "temp_c": entry["temp_c"],
        "condition": entry["condition"],
        "source": "weather_c",
        "observed_at": time.time() - 20,  # freshest of the three
    }


def register(registry: Registry | None = None) -> None:
    (registry or REGISTRY).register(MANIFEST, weather_c, replace=True)
