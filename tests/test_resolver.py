"""Conflict detection and all three resolution policies on canned fixtures.

The LLM adjudicator is monkeypatched, so this file remains offline and fast.
The live path is exercised separately by integration tests.
"""

from __future__ import annotations

import pytest

from orchestrator import resolver as resolver_mod
from orchestrator.llm import LLMError
from orchestrator.manifests import Conflict, ToolResult
from orchestrator.registry import Registry
from orchestrator.resolver import ResolvePolicy, detect_conflicts, resolve
from tests.conftest import make_fake_tool, make_manifest


def _result(
    tool: str,
    subtask_id: str,
    capability: str,
    data: dict,
    observed_at: float,
    ok: bool = True,
) -> ToolResult:
    return ToolResult(
        subtask_id=subtask_id,
        capability=capability,
        tool=tool,
        ok=ok,
        data=data,
        started_at=0.0,
        finished_at=0.1,
        observed_at=observed_at,
    )


def test_detect_conflicts_identical_data_is_not_a_conflict() -> None:
    results = [
        _result("weather_a", "t1", "weather.current", {"temp_c": 20}, observed_at=100.0),
        _result("weather_b", "t1", "weather.current", {"temp_c": 20}, observed_at=200.0),
    ]
    assert detect_conflicts(results) == []


def test_detect_conflicts_divergent_data_is_a_conflict() -> None:
    results = [
        _result("weather_a", "t1", "weather.current", {"temp_c": 29}, observed_at=100.0),
        _result("weather_b", "t1", "weather.current", {"temp_c": 31}, observed_at=200.0),
    ]
    conflicts = detect_conflicts(results)
    assert len(conflicts) == 1
    assert conflicts[0].subtask_id == "t1"
    assert len(conflicts[0].results) == 2


def test_detect_conflicts_ignores_failed_results() -> None:
    results = [
        _result("weather_a", "t1", "weather.current", {"temp_c": 29}, observed_at=100.0),
        _result("weather_b", "t1", "weather.current", {}, observed_at=200.0, ok=False),
    ]
    assert detect_conflicts(results) == []


def test_detect_conflicts_ignores_single_result_subtasks() -> None:
    results = [_result("fx", "t1", "currency.convert", {"converted": 100}, observed_at=100.0)]
    assert detect_conflicts(results) == []


@pytest.fixture
def weather_registry() -> Registry:
    registry = Registry()
    registry.register(
        make_manifest("weather_a", capabilities=["weather.current"], priority=2),
        make_fake_tool(),
    )
    registry.register(
        make_manifest("weather_b", capabilities=["weather.current"], priority=1),
        make_fake_tool(),
    )
    return registry


@pytest.fixture
def weather_conflict() -> Conflict:
    return Conflict(
        subtask_id="t1",
        capability="weather.current",
        results=[
            _result("weather_a", "t1", "weather.current", {"temp_c": 29}, observed_at=100.0),
            _result("weather_b", "t1", "weather.current", {"temp_c": 31}, observed_at=500.0),
        ],
    )


def _lower_priority_first(conflict: Conflict) -> Conflict:
    return conflict.model_copy(update={"results": list(reversed(conflict.results))})


async def test_priority_policy_picks_higher_priority_tool(
    weather_registry: Registry,
    weather_conflict: Conflict,
) -> None:
    resolutions = await resolve([weather_conflict], ResolvePolicy.PRIORITY, weather_registry)
    assert len(resolutions) == 1
    assert resolutions[0].chosen_tool == "weather_a"
    assert resolutions[0].policy == "priority"
    assert "priority" in resolutions[0].rationale.lower()


async def test_freshest_policy_picks_latest_observed_at(
    weather_registry: Registry,
    weather_conflict: Conflict,
) -> None:
    resolutions = await resolve([weather_conflict], ResolvePolicy.FRESHEST, weather_registry)
    assert resolutions[0].chosen_tool == "weather_b"
    assert resolutions[0].policy == "freshest"


async def test_llm_adjudicate_uses_model_choice_and_rationale(
    monkeypatch,
    weather_registry: Registry,
    weather_conflict: Conflict,
) -> None:
    def fake_complete_json(schema, system_prompt, user_prompt):
        return resolver_mod._Adjudication(
            choice="weather_b",
            rationale="Provider B's data is fresher.",
        )

    monkeypatch.setattr(resolver_mod, "complete_json", fake_complete_json)

    resolutions = await resolve(
        [weather_conflict],
        ResolvePolicy.LLM_ADJUDICATE,
        weather_registry,
    )

    assert resolutions[0].chosen_tool == "weather_b"
    assert resolutions[0].rationale == "Provider B's data is fresher."
    assert resolutions[0].policy == "llm_adjudicate"


async def test_llm_adjudicate_rejects_hallucinated_tool_name_and_uses_priority(
    monkeypatch,
    weather_registry: Registry,
    weather_conflict: Conflict,
) -> None:
    def fake_complete_json(schema, system_prompt, user_prompt):
        return resolver_mod._Adjudication(
            choice="weather_z_made_up",
            rationale="not a real tool",
        )

    monkeypatch.setattr(resolver_mod, "complete_json", fake_complete_json)
    conflict = _lower_priority_first(weather_conflict)

    resolutions = await resolve(
        [conflict],
        ResolvePolicy.LLM_ADJUDICATE,
        weather_registry,
    )

    assert conflict.results[0].tool == "weather_b"  # prove this is not first-item fallback
    assert resolutions[0].chosen_tool == "weather_a"
    assert "non-candidate" in resolutions[0].rationale.lower()
    assert "priority" in resolutions[0].rationale.lower()


async def test_llm_adjudicate_transport_error_uses_priority_fallback(
    monkeypatch,
    weather_registry: Registry,
    weather_conflict: Conflict,
) -> None:
    def failing_complete_json(schema, system_prompt, user_prompt):
        raise LLMError("connection refused")

    monkeypatch.setattr(resolver_mod, "complete_json", failing_complete_json)
    conflict = _lower_priority_first(weather_conflict)

    resolutions = await resolve(
        [conflict],
        ResolvePolicy.LLM_ADJUDICATE,
        weather_registry,
    )

    assert conflict.results[0].tool == "weather_b"
    assert resolutions[0].chosen_tool == "weather_a"
    assert "failed" in resolutions[0].rationale.lower()
    assert "priority" in resolutions[0].rationale.lower()
