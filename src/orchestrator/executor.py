"""Dependency-aware, batched, parallel executor.

Subtasks form a dependency DAG via ``Subtask.depends_on``. We topologically
sort them into batches (Kahn's algorithm); within a batch every tool call
runs concurrently via ``asyncio.gather``. Each call is individually wrapped
so a single tool's timeout or exception produces a failed ``ToolResult``
rather than aborting its siblings - see tests/test_executor.py for the
concurrency and fault-isolation proof via fake async tools + timestamps.
"""

from __future__ import annotations

import asyncio
import time
from collections import defaultdict

from orchestrator.manifests import BatchReport, Plan, ToolCall, ToolResult
from orchestrator.registry import Registry


def _topo_batches(deps: dict[str, list[str]]) -> list[list[str]]:
    """Kahn's algorithm, batched: each batch is every node whose
    dependencies are already satisfied by prior batches. Falls back to
    dumping any remaining nodes as a last batch if a cycle sneaks in
    (defensive only - a well-formed plan never produces one)."""
    remaining = dict(deps)
    done: set[str] = set()
    batches: list[list[str]] = []

    while remaining:
        ready = sorted(sid for sid, d in remaining.items() if all(x in done for x in d))
        if not ready:
            batches.append(sorted(remaining.keys()))
            break
        batches.append(ready)
        done.update(ready)
        for sid in ready:
            remaining.pop(sid)

    return batches


async def _invoke(call: ToolCall, registry: Registry) -> ToolResult:
    tool = registry.get(call.tool)
    started = time.monotonic()
    if tool is None:
        finished = time.monotonic()
        return ToolResult(
            subtask_id=call.subtask_id,
            capability=call.capability,
            tool=call.tool,
            ok=False,
            error=f"tool '{call.tool}' is not registered",
            started_at=started,
            finished_at=finished,
        )

    try:
        data = await asyncio.wait_for(tool.fn(**call.args), timeout=tool.manifest.timeout_s)
        finished = time.monotonic()
        observed_at = data.pop("observed_at", None) if isinstance(data, dict) else None
        if data and "error" in data:
            return ToolResult(
                subtask_id=call.subtask_id,
                capability=call.capability,
                tool=call.tool,
                ok=False,
                error=str(data["error"]),
                data=data,
                started_at=started,
                finished_at=finished,
                observed_at=observed_at if observed_at is not None else time.time(),
            )
        return ToolResult(
            subtask_id=call.subtask_id,
            capability=call.capability,
            tool=call.tool,
            ok=True,
            data=data or {},
            started_at=started,
            finished_at=finished,
            observed_at=observed_at if observed_at is not None else time.time(),
        )
    except (asyncio.TimeoutError, TimeoutError) as exc:
        finished = time.monotonic()
        return ToolResult(
            subtask_id=call.subtask_id,
            capability=call.capability,
            tool=call.tool,
            ok=False,
            error=f"timed out after {tool.manifest.timeout_s}s: {exc}",
            started_at=started,
            finished_at=finished,
        )
    except Exception as exc:  # noqa: BLE001 - a tool's own bug must not kill the batch
        finished = time.monotonic()
        return ToolResult(
            subtask_id=call.subtask_id,
            capability=call.capability,
            tool=call.tool,
            ok=False,
            error=f"{type(exc).__name__}: {exc}",
            started_at=started,
            finished_at=finished,
        )


async def _run_batch(
    index: int, calls: list[ToolCall], registry: Registry, sequential: bool
) -> tuple[list[ToolResult], BatchReport]:
    batch_start = time.monotonic()
    if sequential:
        results = [await _invoke(c, registry) for c in calls]
    else:
        results = list(await asyncio.gather(*(_invoke(c, registry) for c in calls)))
    wall_time = time.monotonic() - batch_start

    sequential_estimate = sum(r.duration_s for r in results)
    report = BatchReport(
        batch_index=index,
        call_count=len(calls),
        wall_time_s=round(wall_time, 4),
        sequential_estimate_s=round(sequential_estimate, 4),
    )
    return results, report


async def run(
    plan: Plan,
    allowed_calls: list[ToolCall],
    registry: Registry,
    sequential: bool = False,
) -> tuple[list[ToolResult], list[BatchReport]]:
    """Groups ``allowed_calls`` into dependency batches and runs them.

    ``sequential=True`` runs each batch's calls one-at-a-time (used by the
    CLI's ``--sequential`` flag for an honest, actually-measured baseline
    rather than an estimate) instead of via ``asyncio.gather``.
    """
    calls_by_subtask: dict[str, list[ToolCall]] = defaultdict(list)
    for call in allowed_calls:
        calls_by_subtask[call.subtask_id].append(call)

    runnable_ids = set(calls_by_subtask)
    deps = {
        st.id: [d for d in st.depends_on if d in runnable_ids]
        for st in plan.subtasks
        if st.id in runnable_ids
    }

    batch_id_groups = _topo_batches(deps)

    all_results: list[ToolResult] = []
    all_reports: list[BatchReport] = []
    for i, batch_ids in enumerate(batch_id_groups):
        batch_calls = [c for sid in batch_ids for c in calls_by_subtask[sid]]
        results, report = await _run_batch(i, batch_calls, registry, sequential)
        all_results.extend(results)
        all_reports.append(report)

    return all_results, all_reports
