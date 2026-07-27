"""Mock weather provider B. Same capability as weather_a, deliberately
divergent readings, lower priority but fresher data - a genuine tradeoff
so ``priority`` and ``freshest`` resolution policies can disagree."""

from __future__ import annotations

import asyncio
import time

from orchestrator.manifests import ScopeClass
from orchestrator.registry import tool_def

_DATA = {
    "tokyo": {"temp_c": 31.5, "condition": "humid"},
    "paris": {"temp_c": 21.0, "condition": "sunny"},
    "new york": {"temp_c": 23.0, "condition": "clear"},
    "london": {"temp_c": 15.5, "condition": "overcast"},
    "san francisco": {"temp_c": 17.0, "condition": "clear"},
}


@tool_def(
    name="weather_b",
    description="Weather provider B (canned/mock, lower priority, fresher data).",
    capabilities=["weather.current"],
    scope=ScopeClass.READ,
    priority=1,
    cost_hint=0.8,
    timeout_s=5.0,
    param_schema={"location": "city name, e.g. 'Tokyo'"},
)
async def weather_b(location: str) -> dict:
    await asyncio.sleep(0.3)  # simulated network latency
    key = location.strip().lower()
    entry = _DATA.get(key)
    if entry is None:
        return {"location": location, "error": f"no data for '{location}'"}
    return {
        "location": location,
        "temp_c": entry["temp_c"],
        "condition": entry["condition"],
        "source": "weather_b",
        "observed_at": time.time() - 90,  # ~90s stale, much fresher than A
    }
