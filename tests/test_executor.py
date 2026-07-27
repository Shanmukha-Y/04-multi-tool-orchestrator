"""Proves the executor's parallelism is real (not just asyncio.gather
syntax with no actual concurrency benefit) using fake async tools whose
own start/finish timestamps we can inspect after the run, plus proves a
single tool's timeout or exception never takes down its batch."""

from __future__ import annotations

import asyncio

import pytest

from orchestrator import executor
from orchestrator.manifests import Subtask, ToolCall
from orchestrator.registry import Registry
from tests.conftest import make_fake_tool, make_manifest, make_plan

DELAY = 0.2


def _register(registry: Registry, name: str, delay: float = 0.0, payload: dict | None = None, raises=None):
    fn = make_fake_tool(delay_s=delay, payload=payload or {"ok": True}, raises=raises)
    registry.register(make_manifest(name, capabilities=[f"cap.{name}"], timeout_s=1.0), fn)
    return fn


@pytest.mark.timeout(10)
async def test_independent_calls_run_concurrently_not_sequentially(registry: Registry) -> None:
    fn_a = _register(registry, "a", delay=DELAY)
    fn_b = _register(registry, "b", delay=DELAY)
    plan = make_plan(
        Subtask(id="t1", capability="cap.a"),
        Subtask(id="t2", capability="cap.b"),
    )
    calls = [
        ToolCall(subtask_id="t1", capability="cap.a", tool="a", args={}),
        ToolCall(subtask_id="t2", capability="cap.b", tool="b", args={}),
    ]

    results, batches = await executor.run(plan, calls, registry, sequential=False)

    assert len(batches) == 1
    assert batches[0].call_count == 2
    # If these ran sequentially, wall time would be >= 2*DELAY. True
    # concurrency keeps it close to a single DELAY.
    assert batches[0].wall_time_s < DELAY * 1.6
    assert all(r.ok for r in results)

    # Direct proof from the fakes' own timestamps: both must have been
    # in-flight at the same instant (b started before a finished).
    a_call, b_call = fn_a.calls[0], fn_b.calls[0]
    assert a_call["started"] < b_call["finished"]
    assert b_call["started"] < a_call["finished"]


@pytest.mark.timeout(10)
async def test_sequential_mode_wall_time_is_roughly_additive(registry: Registry) -> None:
    _register(registry, "a", delay=DELAY)
    _register(registry, "b", delay=DELAY)
    plan = make_plan(Subtask(id="t1", capability="cap.a"), Subtask(id="t2", capability="cap.b"))
    calls = [
        ToolCall(subtask_id="t1", capability="cap.a", tool="a", args={}),
        ToolCall(subtask_id="t2", capability="cap.b", tool="b", args={}),
    ]

    results, batches = await executor.run(plan, calls, registry, sequential=True)

    assert batches[0].wall_time_s >= DELAY * 1.8
    assert all(r.ok for r in results)


@pytest.mark.timeout(10)
async def test_dependent_subtask_runs_in_a_later_batch(registry: Registry) -> None:
    _register(registry, "a", delay=0.05)
    _register(registry, "b", delay=0.05)
    plan = make_plan(
        Subtask(id="t1", capability="cap.a"),
        Subtask(id="t2", capability="cap.b", depends_on=["t1"]),
    )
    calls = [
        ToolCall(subtask_id="t1", capability="cap.a", tool="a", args={}),
        ToolCall(subtask_id="t2", capability="cap.b", tool="b", args={}),
    ]

    results, batches = await executor.run(plan, calls, registry, sequential=False)

    assert len(batches) == 2
    assert batches[0].call_count == 1
    assert batches[1].call_count == 1
    t1_result = next(r for r in results if r.subtask_id == "t1")
    t2_result = next(r for r in results if r.subtask_id == "t2")
    assert t1_result.finished_at <= t2_result.started_at


@pytest.mark.timeout(10)
async def test_one_tool_exception_does_not_kill_the_batch(registry: Registry) -> None:
    _register(registry, "good", delay=0.05, payload={"value": 42})
    _register(registry, "bad", delay=0.0, raises=RuntimeError("boom"))
    plan = make_plan(Subtask(id="t1", capability="cap.good"), Subtask(id="t2", capability="cap.bad"))
    calls = [
        ToolCall(subtask_id="t1", capability="cap.good", tool="good", args={}),
        ToolCall(subtask_id="t2", capability="cap.bad", tool="bad", args={}),
    ]

    results, batches = await executor.run(plan, calls, registry, sequential=False)

    assert len(batches) == 1
    assert batches[0].call_count == 2
    good = next(r for r in results if r.subtask_id == "t1")
    bad = next(r for r in results if r.subtask_id == "t2")
    assert good.ok is True
    assert good.data == {"value": 42}
    assert bad.ok is False
    assert "boom" in bad.error


@pytest.mark.timeout(10)
async def test_tool_timeout_is_captured_as_failed_result_not_raised(registry: Registry) -> None:
    slow_fn = make_fake_tool(delay_s=0.5)
    registry.register(make_manifest("slow", capabilities=["cap.slow"], timeout_s=0.05), slow_fn)
    _register(registry, "fast", delay=0.01, payload={"value": 1})
    plan = make_plan(Subtask(id="t1", capability="cap.slow"), Subtask(id="t2", capability="cap.fast"))
    calls = [
        ToolCall(subtask_id="t1", capability="cap.slow", tool="slow", args={}),
        ToolCall(subtask_id="t2", capability="cap.fast", tool="fast", args={}),
    ]

    results, batches = await executor.run(plan, calls, registry, sequential=False)

    slow = next(r for r in results if r.subtask_id == "t1")
    fast = next(r for r in results if r.subtask_id == "t2")
    assert slow.ok is False
    assert "timed out" in slow.error
    assert fast.ok is True


@pytest.mark.timeout(10)
async def test_unknown_tool_produces_failed_result(registry: Registry) -> None:
    plan = make_plan(Subtask(id="t1", capability="cap.ghost"))
    calls = [ToolCall(subtask_id="t1", capability="cap.ghost", tool="ghost", args={})]

    results, batches = await executor.run(plan, calls, registry, sequential=False)

    assert results[0].ok is False
    assert "not registered" in results[0].error
