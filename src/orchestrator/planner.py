"""LangGraph planner node: decompose a request into capability-tagged
subtasks. The planner never emits a tool name - only a capability tag
drawn from the *live* registry catalog, re-read on every call (including
replans), so a tool registered mid-session is something the planner can
route to on its very next planning call."""

from __future__ import annotations

from orchestrator.llm import complete_json
from orchestrator.manifests import Denial, Plan
from orchestrator.registry import Registry

_SYSTEM_TEMPLATE = """You are the planning component of a tool-orchestrating agent.
Decompose the user's request into an ordered list of subtasks. Each subtask
must be tagged with exactly one CAPABILITY from the catalog below - never a
tool name, only a capability tag. Only use capabilities from this catalog.

CAPABILITY CATALOG:
{catalog}

Rules:
- Every subtask needs a unique short id (e.g. "t1", "t2", ...).
- "args" must satisfy the parameter schema shown for that capability.
- Use "depends_on" (list of subtask ids) only when a subtask genuinely
  needs the output of an earlier subtask (e.g. a note that summarizes
  results depends on the subtasks that produced those results).
  Independent subtasks must have an empty depends_on list so they can run
  in parallel.
- If part of the request cannot be served by any capability in the
  catalog, do not invent a capability - omit that part and say what you
  dropped and why in "notes".
{denial_note}
Respond with ONLY a JSON object matching this schema, no prose:
{{"subtasks": [{{"id": "t1", "capability": "...", "args": {{}}, "depends_on": [], "rationale": "..."}}], "notes": "..."}}
"""

_DENIAL_NOTE = """
IMPORTANT - this is a replan. The following capabilities were denied by the
session's permission scope and MUST NOT be used again; drop the subtasks
that needed them and explain what you couldn't do in "notes":
{denials}
"""


def _format_catalog(registry: Registry) -> str:
    catalog = registry.capability_catalog()
    if not catalog:
        return "(no tools currently registered)"
    lines = []
    for cap, info in sorted(catalog.items()):
        params = ", ".join(f"{k} ({v})" for k, v in info["param_schema"].items()) or "no args"
        lines.append(
            f"- {cap}: {info['description']} | args: {params} | "
            f"scope(s): {', '.join(sorted(info['scopes']))}"
        )
    return "\n".join(lines)


def build_system_prompt(registry: Registry, denials: list[Denial] | None = None) -> str:
    denial_note = ""
    if denials:
        lines = [
            f"- {d.capability} (tool '{d.tool}' needs '{d.scope_required.value}' scope)" for d in denials
        ]
        denial_note = _DENIAL_NOTE.format(denials="\n".join(lines))
    return _SYSTEM_TEMPLATE.format(catalog=_format_catalog(registry), denial_note=denial_note)


def plan(request: str, registry: Registry, denials: list[Denial] | None = None) -> Plan:
    system_prompt = build_system_prompt(registry, denials)
    return complete_json(Plan, system_prompt, request)
