"""Mock weather provider A. Deliberately overlaps with weather_b on the
same capability so the two can disagree - that disagreement is what feeds
the conflict-resolution demo. Provider A is slower and its readings run
noticeably stale (simulated via ``observed_at``)."""

from __future__ import annotations

import asyncio
import time

from orchestrator.manifests import ScopeClass
from orchestrator.registry import tool_def

_DATA = {
    "tokyo": {"temp_c": 29.0, "condition": "clear"},
    "paris": {"temp_c": 19.0, "condition": "cloudy"},
    "new york": {"temp_c": 24.0, "condition": "partly cloudy"},
    "london": {"temp_c": 16.0, "condition": "rain"},
    "san francisco": {"temp_c": 15.0, "condition": "fog"},
}


@tool_def(
    name="weather_a",
    description="Weather provider A (canned/mock, higher priority, runs stale).",
    capabilities=["weather.current"],
    scope=ScopeClass.READ,
    priority=2,
    cost_hint=1.0,
    timeout_s=5.0,
    param_schema={"location": "city name, e.g. 'Tokyo'"},
)
async def weather_a(location: str) -> dict:
    await asyncio.sleep(0.4)  # simulated network latency
    key = location.strip().lower()
    entry = _DATA.get(key)
    if entry is None:
        return {"location": location, "error": f"no data for '{location}'"}
    return {
        "location": location,
        "temp_c": entry["temp_c"],
        "condition": entry["condition"],
        "source": "weather_a",
        "observed_at": time.time() - 900,  # ~15 min stale
    }
