from __future__ import annotations

from orchestrator.manifests import Subtask
from orchestrator.registry import Registry
from orchestrator.router import RouterMode, route
from tests.conftest import make_fake_tool, make_manifest, make_plan


def test_route_single_candidate(registry: Registry) -> None:
    registry.register(make_manifest("fx", capabilities=["currency.convert"]), make_fake_tool())
    plan = make_plan(Subtask(id="t1", capability="currency.convert", args={"amount": 1}))

    result = route(plan, registry, RouterMode.ALL)

    assert result.unroutable == []
    assert [c.tool for c, _ in result.candidates["t1"]] == ["fx"]


def test_route_all_mode_returns_every_candidate(registry: Registry) -> None:
    registry.register(make_manifest("weather_a", capabilities=["weather.current"], priority=2), make_fake_tool())
    registry.register(make_manifest("weather_b", capabilities=["weather.current"], priority=1), make_fake_tool())
    plan = make_plan(Subtask(id="t1", capability="weather.current", args={"location": "Tokyo"}))

    result = route(plan, registry, RouterMode.ALL)

    tool_names = sorted(c.tool for c, _ in result.candidates["t1"])
    assert tool_names == ["weather_a", "weather_b"]


def test_route_priority_mode_returns_only_top_candidate(registry: Registry) -> None:
    registry.register(make_manifest("weather_a", capabilities=["weather.current"], priority=2), make_fake_tool())
    registry.register(make_manifest("weather_b", capabilities=["weather.current"], priority=1), make_fake_tool())
    plan = make_plan(Subtask(id="t1", capability="weather.current", args={"location": "Tokyo"}))

    result = route(plan, registry, RouterMode.PRIORITY)

    assert [c.tool for c, _ in result.candidates["t1"]] == ["weather_a"]


def test_route_no_candidate_is_unroutable(registry: Registry) -> None:
    plan = make_plan(Subtask(id="t1", capability="nonexistent.capability"))

    result = route(plan, registry, RouterMode.ALL)

    assert result.unroutable == ["t1"]
    assert "t1" not in result.candidates


def test_route_carries_subtask_args_into_tool_call(registry: Registry) -> None:
    registry.register(make_manifest("fx", capabilities=["currency.convert"]), make_fake_tool())
    args = {"amount": 100, "from_currency": "USD", "to_currency": "JPY"}
    plan = make_plan(Subtask(id="t1", capability="currency.convert", args=args))

    result = route(plan, registry, RouterMode.ALL)

    call, _ = result.candidates["t1"][0]
    assert call.args == args
    assert call.subtask_id == "t1"
    assert call.capability == "currency.convert"


def test_dynamically_registered_tool_is_immediately_routable(registry: Registry) -> None:
    """The core M5 property: a tool registered *after* the registry has
    already been queried once must be a valid candidate on the very next
    route() call - no restart, no cache to invalidate."""
    plan = make_plan(Subtask(id="t1", capability="weather.current", args={"location": "Tokyo"}))

    first = route(plan, registry, RouterMode.ALL)
    assert first.unroutable == ["t1"]

    registry.register(
        make_manifest("weather_c", capabilities=["weather.current"], priority=3), make_fake_tool()
    )

    second = route(plan, registry, RouterMode.ALL)
    assert second.unroutable == []
    assert [c.tool for c, _ in second.candidates["t1"]] == ["weather_c"]
