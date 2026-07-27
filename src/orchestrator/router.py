"""Capability -> candidate tools. This is where "capability-based routing"
actually happens: subtasks never name a tool, only a capability tag, and
the router looks up whoever is *currently* registered for it. A tool
registered mid-session is picked up here with no code change - see
tests/test_router.py::test_dynamically_registered_tool_is_routable.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from orchestrator.manifests import Plan, ToolCall
from orchestrator.registry import RegisteredTool, Registry


class RouterMode(str, Enum):
    """How many candidates to call when a capability has more than one
    tool registered for it.

    ALL      - call every candidate (redundant on purpose - this is what
               surfaces conflicts for the resolver to adjudicate).
    PRIORITY - call only the single highest-priority candidate.
    """

    ALL = "all"
    PRIORITY = "priority"


@dataclass
class RouteResult:
    candidates: dict[str, list[tuple[ToolCall, RegisteredTool]]]
    """subtask_id -> [(ToolCall, RegisteredTool), ...]"""
    unroutable: list[str]
    """subtask ids whose capability has zero registered tools at all."""


def route(plan: Plan, registry: Registry, mode: RouterMode = RouterMode.ALL) -> RouteResult:
    candidates: dict[str, list[tuple[ToolCall, RegisteredTool]]] = {}
    unroutable: list[str] = []

    for subtask in plan.subtasks:
        matches = registry.find_by_capability(subtask.capability)
        if not matches:
            unroutable.append(subtask.id)
            continue

        chosen = matches if mode == RouterMode.ALL else matches[:1]
        candidates[subtask.id] = [
            (
                ToolCall(
                    subtask_id=subtask.id,
                    capability=subtask.capability,
                    tool=tool.manifest.name,
                    args=subtask.args,
                ),
                tool,
            )
            for tool in chosen
        ]

    return RouteResult(candidates=candidates, unroutable=unroutable)
