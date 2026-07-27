"""Conflict detection + resolution.

The router (in ALL mode) may call more than one tool for the same
capability/subtask. If their results disagree, that's a ``Conflict`` this
module resolves under one of three policies, each producing a
human-readable rationale that lands in the execution report:

- priority:       pick the candidate with the highest manifest priority.
- freshest:       pick the candidate whose data has the latest observed_at.
- llm_adjudicate: ask qwen3.5:9b to pick and explain, one JSON retry.
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
    """Groups successful results by subtask_id; any group of 2+ whose
    ``data`` payloads aren't all identical is a conflict."""
    by_subtask: dict[str, list[ToolResult]] = {}
    for r in results:
        if r.ok:
            by_subtask.setdefault(r.subtask_id, []).append(r)

    conflicts: list[Conflict] = []
    for subtask_id, group in by_subtask.items():
        if len(group) < 2:
            continue
        first = group[0].data
        if any(r.data != first for r in group[1:]):
            conflicts.append(
                Conflict(subtask_id=subtask_id, capability=group[0].capability, results=group)
            )
    return conflicts


def _resolve_priority(conflict: Conflict, registry: Registry) -> Resolution:
    def prio(r: ToolResult) -> int:
        tool = registry.get(r.tool)
        return tool.manifest.priority if tool else 0

    chosen = max(conflict.results, key=prio)
    others = ", ".join(r.tool for r in conflict.results)
    return Resolution(
        subtask_id=conflict.subtask_id,
        capability=conflict.capability,
        policy=ResolvePolicy.PRIORITY.value,
        chosen_tool=chosen.tool,
        rationale=(
            f"'{chosen.tool}' has the highest configured priority ({prio(chosen)}) "
            f"among {len(conflict.results)} candidates ({others})."
        ),
    )


def _resolve_freshest(conflict: Conflict) -> Resolution:
    chosen = max(conflict.results, key=lambda r: r.observed_at)
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


async def _resolve_llm(conflict: Conflict) -> Resolution:
    options = "\n".join(f"- {r.tool}: {r.data}" for r in conflict.results)
    user_prompt = f"Capability: {conflict.capability}\nCandidate results:\n{options}"

    try:
        adjudication = await asyncio.to_thread(complete_json, _Adjudication, _ADJUDICATE_SYSTEM, user_prompt)
    except LLMError as exc:
        fallback_tool = conflict.results[0].tool
        return Resolution(
            subtask_id=conflict.subtask_id,
            capability=conflict.capability,
            policy=ResolvePolicy.LLM_ADJUDICATE.value,
            chosen_tool=fallback_tool,
            rationale=f"LLM adjudication failed ({exc}); defaulted to '{fallback_tool}'.",
        )

    valid_tools = {r.tool for r in conflict.results}
    chosen_tool = adjudication.choice if adjudication.choice in valid_tools else conflict.results[0].tool
    return Resolution(
        subtask_id=conflict.subtask_id,
        capability=conflict.capability,
        policy=ResolvePolicy.LLM_ADJUDICATE.value,
        chosen_tool=chosen_tool,
        rationale=adjudication.rationale,
    )


async def resolve(conflicts: list[Conflict], policy: ResolvePolicy, registry: Registry) -> list[Resolution]:
    resolutions: list[Resolution] = []
    for conflict in conflicts:
        if policy == ResolvePolicy.PRIORITY:
            resolutions.append(_resolve_priority(conflict, registry))
        elif policy == ResolvePolicy.FRESHEST:
            resolutions.append(_resolve_freshest(conflict))
        else:
            resolutions.append(await _resolve_llm(conflict))
    return resolutions
