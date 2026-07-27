"""The tool registry: runtime register/unregister, queried fresh per-request.

This is the piece that makes "capability-based routing" real rather than
aspirational. Nothing downstream (router, planner prompt, CLI `orc tools`)
caches a snapshot of what tools exist - every lookup goes through
``Registry.find_by_capability`` / ``Registry.list_tools``, which read the
live dict. A tool registered mid-process via ``register()`` is therefore
immediately routable on the very next request, with zero changes anywhere
else in the pipeline. That's the property Milestone 5's demo exercises.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field

from orchestrator.manifests import ScopeClass, ToolManifest

ToolFn = Callable[..., Awaitable[dict]]


@dataclass
class RegisteredTool:
    manifest: ToolManifest
    fn: ToolFn


class DuplicateToolError(ValueError):
    pass


class Registry:
    """A plain, mutable, in-process registry. No hidden global state beyond
    the module-level singleton below - this class is fully instantiable so
    tests can build isolated registries instead of sharing process state."""

    def __init__(self) -> None:
        self._tools: dict[str, RegisteredTool] = {}

    def register(self, manifest: ToolManifest, fn: ToolFn, *, replace: bool = False) -> None:
        if manifest.name in self._tools and not replace:
            raise DuplicateToolError(
                f"tool '{manifest.name}' is already registered (pass replace=True to override)"
            )
        self._tools[manifest.name] = RegisteredTool(manifest=manifest, fn=fn)

    def unregister(self, name: str) -> bool:
        return self._tools.pop(name, None) is not None

    def get(self, name: str) -> RegisteredTool | None:
        return self._tools.get(name)

    def list_tools(self) -> list[RegisteredTool]:
        return sorted(self._tools.values(), key=lambda t: t.manifest.name)

    def find_by_capability(self, capability: str) -> list[RegisteredTool]:
        """All registered tools offering ``capability``, highest priority first."""
        matches = [t for t in self._tools.values() if capability in t.manifest.capabilities]
        return sorted(matches, key=lambda t: t.manifest.priority, reverse=True)

    def capability_catalog(self) -> dict[str, dict]:
        """capability -> {description, param_schema, scopes, tool_count} for
        prompting the planner. Merges across every tool offering that
        capability so the planner sees one coherent arg contract."""
        catalog: dict[str, dict] = {}
        for tool in self._tools.values():
            for cap in tool.manifest.capabilities:
                entry = catalog.setdefault(
                    cap,
                    {"description": tool.manifest.description, "param_schema": {}, "scopes": set(), "tool_count": 0},
                )
                entry["param_schema"].update(tool.manifest.param_schema)
                entry["scopes"].add(tool.manifest.scope.value)
                entry["tool_count"] += 1
        return catalog

    def __len__(self) -> int:
        return len(self._tools)

    def __contains__(self, name: str) -> bool:
        return name in self._tools


REGISTRY = Registry()


def tool_def(
    *,
    name: str,
    description: str,
    capabilities: list[str],
    scope: ScopeClass,
    priority: int = 1,
    cost_hint: float = 1.0,
    timeout_s: float = 10.0,
    param_schema: dict[str, str] | None = None,
    registry: Registry | None = None,
):
    """Decorator: wraps an async ``fn(**kwargs) -> dict`` and registers it.

    Usage:
        @tool_def(name="weather_a", capabilities=["weather.current"],
                   scope=ScopeClass.READ, param_schema={"location": "city name"})
        async def weather_a(location: str) -> dict:
            ...

    The decorated function is returned unmodified (so it stays directly
    unit-testable / callable) - registration is a side effect on import.
    """

    target = registry if registry is not None else REGISTRY

    def decorator(fn: ToolFn) -> ToolFn:
        manifest = ToolManifest(
            name=name,
            description=description,
            capabilities=capabilities,
            scope=scope,
            priority=priority,
            cost_hint=cost_hint,
            timeout_s=timeout_s,
            param_schema=param_schema or {},
        )
        target.register(manifest, fn, replace=True)
        return fn

    return decorator
