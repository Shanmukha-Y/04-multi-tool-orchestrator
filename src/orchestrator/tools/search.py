"""Mock web search tool. Returns canned snippets - no real network call, no
API key, so the demo is fully offline-reproducible."""

from __future__ import annotations

import asyncio

from orchestrator.manifests import ScopeClass
from orchestrator.registry import tool_def

_CANNED = {
    "tokyo travel tips": [
        "Tokyo's subway is the fastest way to cross the city; get a Suica card.",
        "Visit Shinjuku Gyoen for a quiet break from the crowds.",
    ],
    "paris travel tips": [
        "The Paris Metro closes around 1am on weekdays, later on weekends.",
        "Book Louvre tickets online to skip the queue.",
    ],
}


@tool_def(
    name="search",
    description="Web search (canned/mock results, no live network call).",
    capabilities=["web.search"],
    scope=ScopeClass.NETWORK,
    priority=1,
    timeout_s=5.0,
    param_schema={"query": "search query text"},
)
async def search(query: str) -> dict:
    await asyncio.sleep(0.2)
    key = query.strip().lower()
    results = _CANNED.get(key) or [f"No canned results for '{query}' - this is a mock search tool."]
    return {"query": query, "results": results}
