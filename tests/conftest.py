from __future__ import annotations

import asyncio
import time

import pytest

from orchestrator.manifests import Plan, ScopeClass, Subtask, ToolManifest
from orchestrator.registry import Registry


@pytest.fixture
def registry() -> Registry:
    """A fresh, isolated registry per test - never touches the global
    REGISTRY that the demo tools populate on import."""
    return Registry()


def make_manifest(
    name: str,
    *,
    capabilities: list[str],
    scope: ScopeClass = ScopeClass.READ,
    priority: int = 1,
    timeout_s: float = 5.0,
    cost_hint: float = 1.0,
) -> ToolManifest:
    return ToolManifest(
        name=name,
        description=f"fake tool {name}",
        capabilities=capabilities,
        scope=scope,
        priority=priority,
        timeout_s=timeout_s,
        cost_hint=cost_hint,
    )


def make_fake_tool(delay_s: float = 0.0, payload: dict | None = None, raises: Exception | None = None):
    """Returns an async fn(**kwargs) that records call/finish timestamps on
    itself (so tests can assert real concurrency), sleeps ``delay_s``, then
    either raises or returns ``payload``."""
    calls: list[dict] = []

    async def fn(**kwargs) -> dict:
        started = time.monotonic()
        await asyncio.sleep(delay_s)
        finished = time.monotonic()
        calls.append({"started": started, "finished": finished, "kwargs": kwargs})
        if raises is not None:
            raise raises
        return dict(payload or {})

    fn.calls = calls  # type: ignore[attr-defined]
    return fn


def make_plan(*subtasks: Subtask, notes: str = "") -> Plan:
    return Plan(subtasks=list(subtasks), notes=notes)
