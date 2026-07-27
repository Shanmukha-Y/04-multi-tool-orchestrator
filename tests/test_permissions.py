from __future__ import annotations

import itertools

import pytest

from orchestrator.manifests import ScopeClass, ToolCall
from orchestrator.permissions import fully_denied_subtasks, gate
from orchestrator.registry import Registry
from tests.conftest import make_fake_tool, make_manifest


def _one_candidate(registry: Registry, subtask_id: str, tool_name: str, capability: str, scope: ScopeClass):
    manifest = make_manifest(tool_name, capabilities=[capability], scope=scope)
    registry.register(manifest, make_fake_tool())
    call = ToolCall(subtask_id=subtask_id, capability=capability, tool=tool_name, args={})
    return {subtask_id: [(call, registry.get(tool_name))]}


@pytest.mark.parametrize("scope_class", list(ScopeClass))
@pytest.mark.parametrize("granted", [True, False])
def test_gate_matrix_per_scope_class(registry: Registry, scope_class: ScopeClass, granted: bool) -> None:
    """Every ScopeClass x {granted, denied} outcome, individually."""
    candidates = _one_candidate(registry, "t1", "tool1", "cap.one", scope_class)
    session_scope = {scope_class} if granted else (set(ScopeClass) - {scope_class})

    result = gate(candidates, session_scope)

    if granted:
        assert len(result.allowed) == 1
        assert result.denials == []
    else:
        assert result.allowed == []
        assert len(result.denials) == 1
        assert result.denials[0].scope_required == scope_class
        assert result.denials[0].tool == "tool1"


def test_gate_allows_when_any_candidate_is_in_scope(registry: Registry) -> None:
    """A subtask with two candidate tools of different scopes survives the
    gate as long as at least one is allowed - only the denied candidate is
    recorded, the subtask itself is not dropped."""
    registry.register(make_manifest("weather_a", capabilities=["weather.current"], scope=ScopeClass.READ), make_fake_tool())
    registry.register(make_manifest("weather_net", capabilities=["weather.current"], scope=ScopeClass.NETWORK), make_fake_tool())
    candidates = {
        "t1": [
            (ToolCall(subtask_id="t1", capability="weather.current", tool="weather_a", args={}), registry.get("weather_a")),
            (ToolCall(subtask_id="t1", capability="weather.current", tool="weather_net", args={}), registry.get("weather_net")),
        ]
    }

    result = gate(candidates, {ScopeClass.READ})

    assert [c.tool for c in result.allowed] == ["weather_a"]
    assert len(result.denials) == 1
    assert result.denials[0].tool == "weather_net"
    assert fully_denied_subtasks(candidates, result) == set()


def test_fully_denied_subtasks_when_every_candidate_denied(registry: Registry) -> None:
    candidates = _one_candidate(registry, "t1", "email_mock", "email.send", ScopeClass.WRITE)
    result = gate(candidates, {ScopeClass.READ})
    assert fully_denied_subtasks(candidates, result) == {"t1"}


def test_gate_empty_scope_denies_everything(registry: Registry) -> None:
    candidates = _one_candidate(registry, "t1", "calc", "math.calculate", ScopeClass.READ)
    result = gate(candidates, set())
    assert result.allowed == []
    assert len(result.denials) == 1
