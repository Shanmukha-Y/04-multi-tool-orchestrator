from __future__ import annotations

import pytest

from orchestrator.manifests import ScopeClass
from orchestrator.registry import DuplicateToolError, Registry, tool_def
from tests.conftest import make_fake_tool, make_manifest


def test_register_and_get(registry: Registry) -> None:
    manifest = make_manifest("alpha", capabilities=["thing.do"])
    fn = make_fake_tool()
    registry.register(manifest, fn)

    got = registry.get("alpha")
    assert got is not None
    assert got.manifest.name == "alpha"
    assert got.fn is fn
    assert len(registry) == 1
    assert "alpha" in registry


def test_duplicate_registration_raises_without_replace(registry: Registry) -> None:
    manifest = make_manifest("alpha", capabilities=["thing.do"])
    registry.register(manifest, make_fake_tool())
    with pytest.raises(DuplicateToolError):
        registry.register(manifest, make_fake_tool())


def test_duplicate_registration_allowed_with_replace(registry: Registry) -> None:
    manifest = make_manifest("alpha", capabilities=["thing.do"])
    fn1, fn2 = make_fake_tool(), make_fake_tool()
    registry.register(manifest, fn1)
    registry.register(manifest, fn2, replace=True)
    assert registry.get("alpha").fn is fn2


def test_unregister(registry: Registry) -> None:
    registry.register(make_manifest("alpha", capabilities=["thing.do"]), make_fake_tool())
    assert registry.unregister("alpha") is True
    assert registry.get("alpha") is None
    assert len(registry) == 0
    # unregistering something not present is a no-op, not an error
    assert registry.unregister("alpha") is False


def test_find_by_capability_sorted_by_priority_desc(registry: Registry) -> None:
    registry.register(make_manifest("low", capabilities=["weather.current"], priority=1), make_fake_tool())
    registry.register(make_manifest("high", capabilities=["weather.current"], priority=5), make_fake_tool())
    registry.register(make_manifest("mid", capabilities=["weather.current"], priority=3), make_fake_tool())
    registry.register(make_manifest("unrelated", capabilities=["math.calculate"], priority=9), make_fake_tool())

    matches = registry.find_by_capability("weather.current")
    assert [t.manifest.name for t in matches] == ["high", "mid", "low"]


def test_find_by_capability_no_match_returns_empty(registry: Registry) -> None:
    assert registry.find_by_capability("nonexistent.capability") == []


def test_capability_catalog_merges_across_tools(registry: Registry) -> None:
    registry.register(
        make_manifest("a", capabilities=["weather.current"], scope=ScopeClass.READ),
        make_fake_tool(),
    )
    registry.register(
        make_manifest("b", capabilities=["weather.current"], scope=ScopeClass.NETWORK),
        make_fake_tool(),
    )
    catalog = registry.capability_catalog()
    assert catalog["weather.current"]["tool_count"] == 2
    assert catalog["weather.current"]["scopes"] == {"read", "network"}


def test_tool_def_decorator_registers_into_target_registry(registry: Registry) -> None:
    @tool_def(
        name="decorated",
        description="a decorated tool",
        capabilities=["thing.do"],
        scope=ScopeClass.READ,
        registry=registry,
    )
    async def decorated(**kwargs):
        return {"ok": True}

    assert "decorated" in registry
    assert registry.get("decorated").manifest.capabilities == ["thing.do"]


def test_tool_def_decorator_defaults_to_global_registry() -> None:
    from orchestrator.registry import REGISTRY

    @tool_def(
        name="global_decorated_test_tool",
        description="registers into the global registry by default",
        capabilities=["thing.global"],
        scope=ScopeClass.READ,
    )
    async def fn(**kwargs):
        return {}

    try:
        assert "global_decorated_test_tool" in REGISTRY
    finally:
        REGISTRY.unregister("global_decorated_test_tool")


def test_runtime_register_then_unregister_lifecycle(registry: Registry) -> None:
    """The exact lifecycle the M5 dynamic-registration demo exercises."""
    assert registry.find_by_capability("weather.current") == []

    manifest = make_manifest("weather_c", capabilities=["weather.current"], priority=3)
    registry.register(manifest, make_fake_tool(payload={"temp_c": 30}))
    assert [t.manifest.name for t in registry.find_by_capability("weather.current")] == ["weather_c"]

    registry.unregister("weather_c")
    assert registry.find_by_capability("weather.current") == []
