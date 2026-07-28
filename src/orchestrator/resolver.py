"""Conflict detection and resolution.

In ``ALL`` routing mode, more than one tool may answer the same capability.
Divergent successful payloads form a ``Conflict`` resolved by one of three
policies:

- ``priority``: choose the highest manifest priority.
- ``freshest``: choose the latest ``observed_at`` timestamp.
- ``llm_adjudicate``: ask the model to select a named candidate, then fall
  back to the deterministic priority policy on transport, validation, or
  hallucinated-candidate failure.
"""

from __future__ import annotations

import asyncio
import time
from enum import Enum

from pydantic import BaseModel

from orchestrator.llm import LLMError, complete_json
from orchestrator.manifests import Conflict, Resolution, ToolResult
from orchestrator.registry import Registry


class ResolvePolicy(str, Enum):
    PRIORITY = "priority"
    FRESHEST = "freshest"
    LLM_ADJUDICATE = "llm_adjudicate"


def detect_conflicts(results: list[ToolResult]) -> list[Conflict]:
    """Return successful same-subtask result groups whose payloads differ."""
    by_subtask: dict[str, list[ToolResult]] = {}
    for result in results:
        if result.ok:
            by_subtask.setdefault(result.subtask_id, []).append(result)

    conflicts: list[Conflict] = []
    for subtask_id, group in by_subtask.items():
        if len(group) < 2:
            continue
        first = group[0].data
        if any(result.data != first for result in group[1:]):
            conflicts.append(
                Conflict(subtask_id=subtask_id, capability=group[0].capability, results=group)
            )
    return conflicts


def _resolve_priority(conflict: Conflict, registry: Registry) -> Resolution:
    def priority(result: ToolResult) -> int:
        tool = registry.get(result.tool)
        return tool.manifest.priority if tool else 0

    chosen = max(conflict.results, key=priority)
    others = ", ".join(result.tool for result in conflict.results)
    return Resolution(
        subtask_id=conflict.subtask_id,
        capability=conflict.capability,
        policy=ResolvePolicy.PRIORITY.value,
        chosen_tool=chosen.tool,
        rationale=(
            f"'{chosen.tool}' has the highest configured priority ({priority(chosen)}) "
            f"among {len(conflict.results)} candidates ({others})."
        ),
    )


def _resolve_freshest(conflict: Conflict) -> Resolution:
    chosen = max(conflict.results, key=lambda result: result.observed_at)
    age_s = max(0.0, time.time() - chosen.observed_at)
    return Resolution(
        subtask_id=conflict.subtask_id,
        capability=conflict.capability,
        policy=ResolvePolicy.FRESHEST.value,
        chosen_tool=chosen.tool,
        rationale=(
            f"'{chosen.tool}' produced the most recent data (~{age_s:.0f}s old) "
            f"among {len(conflict.results)} candidates."
        ),
    )


class _Adjudication(BaseModel):
    choice: str
    rationale: str


_ADJUDICATE_SYSTEM = (
    "You are adjudicating a conflict between two or more tools that reported "
    "different results for the same request. Pick the single most likely "
    "correct/trustworthy result and explain your reasoning in 1-2 sentences.\n"
    'Respond with ONLY JSON: {"choice": "<tool name, exactly as given>", "rationale": "..."}'
)


def _priority_fallback(conflict: Conflict, registry: Registry, reason: str) -> Resolution:
    """Preserve the requested policy while making fallback deterministic."""
    fallback = _resolve_priority(conflict, registry)
    return Resolution(
        subtask_id=conflict.subtask_id,
        capability=conflict.capability,
        policy=ResolvePolicy.LLM_ADJUDICATE.value,
        chosen_tool=fallback.chosen_tool,
        rationale=f"{reason} Fell back to deterministic priority resolution. {fallback.rationale}",
    )


async def _resolve_llm(conflict: Conflict, registry: Registry) -> Resolution:
    options = "\n".join(f"- {result.tool}: {result.data}" for result in conflict.results)
    user_prompt = f"Capability: {conflict.capability}\nCandidate results:\n{options}"

    try:
        adjudication = await asyncio.to_thread(
            complete_json,
            _Adjudication,
            _ADJUDICATE_SYSTEM,
            user_prompt,
        )
    except LLMError as exc:
        return _priority_fallback(conflict, registry, f"LLM adjudication failed ({exc}).")

    valid_tools = {result.tool for result in conflict.results}
    if adjudication.choice not in valid_tools:
        return _priority_fallback(
            conflict,
            registry,
            f"LLM adjudication named non-candidate tool '{adjudication.choice}'.",
        )

    return Resolution(
        subtask_id=conflict.subtask_id,
        capability=conflict.capability,
        policy=ResolvePolicy.LLM_ADJUDICATE.value,
        chosen_tool=adjudication.choice,
        rationale=adjudication.rationale,
    )


async def resolve(
    conflicts: list[Conflict],
    policy: ResolvePolicy,
    registry: Registry,
) -> list[Resolution]:
    resolutions: list[Resolution] = []
    for conflict in conflicts:
        if policy == ResolvePolicy.PRIORITY:
            resolutions.append(_resolve_priority(conflict, registry))
        elif policy == ResolvePolicy.FRESHEST:
            resolutions.append(_resolve_freshest(conflict))
        else:
            resolutions.append(await _resolve_llm(conflict, registry))
    return resolutions
