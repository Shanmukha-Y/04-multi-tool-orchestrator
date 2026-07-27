"""Wires the orchestrator's nodes into a LangGraph ``StateGraph``.

    planner -> router -> permission_gate --[denied]--> planner (bounded to
                                          \\--[ok]-----> executor -> check_conflicts --[conflict]--> resolver -> aggregator
                                                                                     \\--[none]-------------------> aggregator

Two conditional edges do real work here:

1. permission_gate -> planner: a scope denial that leaves a subtask with
   zero runnable candidates triggers exactly one replan, feeding the
   planner structured Denial records so it can drop or route around the
   gap instead of erroring. Bounded to one replan so a stubborn model
   can't loop forever.
2. check_conflicts -> resolver: only entered when two or more candidate
   tools actually disagree; otherwise we skip straight to the aggregator.

A capability with zero *candidates at all* (the router found nothing, most
likely a small model hallucinating a capability outside the catalog) is
treated as an honest failure rather than a replan trigger - see
router.py's module docstring.
"""

from __future__ import annotations

import asyncio
import time
from typing import Any, TypedDict

from langgraph.graph import END, START, StateGraph

from orchestrator import executor, permissions, planner
from orchestrator import resolver as resolver_mod
from orchestrator.llm import LLMError, complete_text
from orchestrator.manifests import (
    Conflict,
    Denial,
    ExecutionReport,
    Plan,
    Resolution,
    ScopeClass,
    ToolCall,
    ToolResult,
)
from orchestrator.registry import REGISTRY, RegisteredTool, Registry
from orchestrator.resolver import ResolvePolicy
from orchestrator.router import RouteResult, RouterMode
from orchestrator.router import route as route_capabilities

MAX_REPLANS = 1


class OrchestratorState(TypedDict, total=False):
    request: str
    scope: set[ScopeClass]
    registry: Registry
    router_mode: RouterMode
    resolve_policy: ResolvePolicy
    sequential: bool

    plan: Plan
    pending_denials: list[Denial] | None
    all_denials: list[Denial]
    replans: int
    should_replan: bool

    candidates: dict[str, list[tuple[ToolCall, RegisteredTool]]]
    unroutable: list[str]

    allowed_calls: list[ToolCall]
    results: list[ToolResult]
    batches: list[Any]

    conflicts: list[Conflict]
    resolutions: list[Resolution]

    answer: str


async def planner_node(state: OrchestratorState) -> dict:
    registry = state["registry"]
    denials = state.get("pending_denials")
    plan = await asyncio.to_thread(planner.plan, state["request"], registry, denials)
    return {"plan": plan, "pending_denials": None}


async def router_node(state: OrchestratorState) -> dict:
    route_result: RouteResult = route_capabilities(state["plan"], state["registry"], state["router_mode"])
    return {"candidates": route_result.candidates, "unroutable": route_result.unroutable}


async def permission_gate_node(state: OrchestratorState) -> dict:
    gate_result = permissions.gate(state["candidates"], state["scope"])
    fully_denied = permissions.fully_denied_subtasks(state["candidates"], gate_result)
    replans = state.get("replans", 0)
    should_replan = bool(fully_denied) and replans < MAX_REPLANS

    return {
        "allowed_calls": gate_result.allowed,
        "all_denials": state.get("all_denials", []) + gate_result.denials,
        "pending_denials": gate_result.denials if should_replan else state.get("pending_denials"),
        "should_replan": should_replan,
        "replans": replans + (1 if should_replan else 0),
    }


def route_after_gate(state: OrchestratorState) -> str:
    return "replan" if state.get("should_replan") else "execute"


async def executor_node(state: OrchestratorState) -> dict:
    results, batches = await executor.run(
        state["plan"], state["allowed_calls"], state["registry"], sequential=state.get("sequential", False)
    )
    return {"results": results, "batches": batches}


def check_conflicts_node(state: OrchestratorState) -> dict:
    return {"conflicts": resolver_mod.detect_conflicts(state["results"])}


def route_after_conflicts(state: OrchestratorState) -> str:
    return "resolve" if state.get("conflicts") else "skip"


async def resolver_node(state: OrchestratorState) -> dict:
    resolutions = await resolver_mod.resolve(state["conflicts"], state["resolve_policy"], state["registry"])
    return {"resolutions": resolutions}


def _summarize_for_answer(state: OrchestratorState) -> str:
    """Deterministic, LLM-free digest of what happened - the aggregator
    reads this to write prose. Keeping the *facts* out of the LLM's hands
    means the execution report (wall-time, which tools ran) is always
    accurate even if the model's writing is not."""
    lines: list[str] = [f"User request: {state['request']}"]

    resolutions_by_subtask = {r.subtask_id: r for r in state.get("resolutions", [])}
    conflicted_subtasks = {c.subtask_id for c in state.get("conflicts", [])}

    for subtask in state["plan"].subtasks:
        sid = subtask.id
        if sid in resolutions_by_subtask:
            res = resolutions_by_subtask[sid]
            winner = next(r for r in state["results"] if r.subtask_id == sid and r.tool == res.chosen_tool)
            lines.append(
                f"[{sid}] {subtask.capability}: CONFLICT resolved via '{res.policy}' -> "
                f"{res.chosen_tool} said {winner.data} ({res.rationale})"
            )
        elif sid in conflicted_subtasks:
            continue
        else:
            matches = [r for r in state["results"] if r.subtask_id == sid]
            if not matches:
                continue
            for r in matches:
                if r.ok:
                    lines.append(f"[{sid}] {subtask.capability} via {r.tool}: {r.data}")
                else:
                    lines.append(f"[{sid}] {subtask.capability} via {r.tool}: FAILED ({r.error})")

    if state.get("all_denials"):
        for d in state["all_denials"]:
            lines.append(f"DENIED: {d.tool} for {d.capability} ({d.reason})")

    if state.get("unroutable"):
        by_id = {st.id: st.capability for st in state["plan"].subtasks}
        for sid in state["unroutable"]:
            lines.append(f"UNROUTABLE: no tool registered for capability '{by_id.get(sid, sid)}'")

    if state["plan"].notes:
        lines.append(f"Planner notes: {state['plan'].notes}")

    return "\n".join(lines)


_AGGREGATOR_SYSTEM = (
    "You are the final-answer component of a tool-orchestrating agent. You are given "
    "a factual digest of what was executed on the user's behalf: results, conflicts "
    "already resolved, and anything that was denied or unavailable. Write a clear, "
    "direct answer to the user's original request using only these facts. If "
    "something couldn't be done, say so plainly and briefly explain why - do not "
    "apologize excessively. Do not invent data that isn't in the digest."
)


async def aggregator_node(state: OrchestratorState) -> dict:
    digest = _summarize_for_answer(state)
    try:
        answer = await asyncio.to_thread(complete_text, _AGGREGATOR_SYSTEM, digest)
    except LLMError as exc:
        answer = f"(LLM summary unavailable: {exc})\n\nRaw execution digest:\n{digest}"
    return {"answer": answer}


def build_graph():
    graph = StateGraph(OrchestratorState)
    graph.add_node("planner", planner_node)
    graph.add_node("router", router_node)
    graph.add_node("permission_gate", permission_gate_node)
    graph.add_node("executor", executor_node)
    graph.add_node("check_conflicts", check_conflicts_node)
    graph.add_node("resolver", resolver_node)
    graph.add_node("aggregator", aggregator_node)

    graph.add_edge(START, "planner")
    graph.add_edge("planner", "router")
    graph.add_edge("router", "permission_gate")
    graph.add_conditional_edges(
        "permission_gate", route_after_gate, {"replan": "planner", "execute": "executor"}
    )
    graph.add_edge("executor", "check_conflicts")
    graph.add_conditional_edges(
        "check_conflicts", route_after_conflicts, {"resolve": "resolver", "skip": "aggregator"}
    )
    graph.add_edge("resolver", "aggregator")
    graph.add_edge("aggregator", END)

    return graph.compile()


async def run_orchestrator(
    request: str,
    *,
    scope: set[ScopeClass],
    registry: Registry | None = None,
    router_mode: RouterMode = RouterMode.ALL,
    resolve_policy: ResolvePolicy = ResolvePolicy.PRIORITY,
    sequential: bool = False,
) -> ExecutionReport:
    """Runs one full request through the compiled graph and assembles the
    final ``ExecutionReport``. This is the single entry point the CLI and
    the integration tests both call."""
    app = build_graph()
    initial_state: OrchestratorState = {
        "request": request,
        "scope": scope,
        "registry": registry or REGISTRY,
        "router_mode": router_mode,
        "resolve_policy": resolve_policy,
        "sequential": sequential,
        "all_denials": [],
        "replans": 0,
    }

    final_state = await app.ainvoke(initial_state, config={"recursion_limit": 50})

    return ExecutionReport(
        plan=final_state["plan"],
        denials=final_state.get("all_denials", []),
        unroutable=final_state.get("unroutable", []),
        replans=final_state.get("replans", 0),
        batches=final_state.get("batches", []),
        results=final_state.get("results", []),
        conflicts=final_state.get("conflicts", []),
        resolutions=final_state.get("resolutions", []),
        answer=final_state.get("answer", ""),
    )
