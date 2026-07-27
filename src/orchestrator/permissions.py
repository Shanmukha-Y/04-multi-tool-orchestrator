"""The permission gate: filters router-selected candidates down to what the
session scope actually allows, producing structured ``Denial`` records for
anything it drops.

Denials are data, not exceptions - the whole point (per spec) is that a
denial is something the planner adapts to on a replan, not a crash."""

from __future__ import annotations

from dataclasses import dataclass

from orchestrator.manifests import Denial, ScopeClass, ToolCall
from orchestrator.registry import RegisteredTool


@dataclass
class GateResult:
    allowed: list[ToolCall]
    denials: list[Denial]


def gate(
    candidates: dict[str, list[tuple[ToolCall, RegisteredTool]]],
    session_scope: set[ScopeClass],
) -> GateResult:
    """``candidates``: subtask_id -> [(ToolCall, RegisteredTool), ...] as
    produced by the router. Returns the allowed calls plus a Denial for
    every candidate whose tool.scope is outside ``session_scope``.

    A subtask with multiple candidate tools survives the gate as long as
    at least one candidate is allowed; only the individually-denied
    candidates are recorded as denials.
    """
    allowed: list[ToolCall] = []
    denials: list[Denial] = []

    for subtask_id, pairs in candidates.items():
        for call, tool in pairs:
            if tool.manifest.scope in session_scope:
                allowed.append(call)
            else:
                denials.append(
                    Denial(
                        subtask_id=subtask_id,
                        capability=call.capability,
                        tool=tool.manifest.name,
                        scope_required=tool.manifest.scope,
                        reason=(
                            f"tool '{tool.manifest.name}' requires "
                            f"'{tool.manifest.scope.value}' scope, which this "
                            f"session does not grant (granted: "
                            f"{', '.join(s.value for s in sorted(session_scope, key=lambda s: s.value)) or 'none'})"
                        ),
                    )
                )

    return GateResult(allowed=allowed, denials=denials)


def fully_denied_subtasks(candidates: dict[str, list], gate_result: GateResult) -> set[str]:
    """Subtask ids where *every* candidate was denied (nothing runnable)."""
    allowed_subtasks = {c.subtask_id for c in gate_result.allowed}
    return {sid for sid in candidates if sid not in allowed_subtasks}
